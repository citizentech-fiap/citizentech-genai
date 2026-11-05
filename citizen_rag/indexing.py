"""Construção/persistência dos índices BM25 + denso, um diretório por configuração.

Diretório: indexes/<corpus_fp>_<strategy>_s<size>_o<overlap>_<p|n>_<embedding>/
  chunks.jsonl     chunks com metadados (fonte do BM25, reconstruído em memória ao carregar — ms)
  embeddings.npy   matriz [n_chunks, dim] normalizada (vector store "numpy")
  chroma/          coleção persistida (somente se vector_store=chroma)
  meta.json        configuração usada, fingerprint do corpus, contagens

Uso: python -m citizen_rag.indexing [--corpus X] [--strategy hybrid] [--size 160] [--overlap 40]
                                    [--grid] [--reuse] [--backend hashing]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk, chunk_corpus
from .config import get_logger, load_config, resolve_path
from .embeddings import Embedder, embedder_slug, make_embedder
from .ingest import corpus_fingerprint, load_corpus
from .textproc import tokenize

log = get_logger("indexing")


def default_corpus_path(cfg: dict) -> Path:
    """Corpus real do robô; se ainda não existir, o fixture de testes (com aviso no log)."""
    real = resolve_path(cfg["paths"]["corpus"])
    if real.exists():
        return real
    fx = resolve_path(cfg["paths"]["corpus_fixture"])
    log.warning("Corpus real não encontrado em %s — usando fixture %s", real, fx)
    return fx


INDEX_SCHEMA = 3   # incrementar quando a lógica de chunking/tokenização mudar (invalida índices antigos)


def index_key(cfg: dict, corpus_fp: str) -> str:
    c = cfg["chunking"]
    p = "p" if c.get("prefix_title", True) else "n"
    return (f"{corpus_fp}_{c['strategy']}_s{c['size']}_o{c['overlap']}_m{c.get('min_size', 8)}_{p}"
            f"_{embedder_slug(cfg)}_v{INDEX_SCHEMA}")


class Index:
    """Índice híbrido em memória: BM25 (rank-bm25) + matriz densa (numpy ou Chroma)."""

    def __init__(self, chunks: list[Chunk], cfg: dict, embeddings: np.ndarray | None,
                 embedder: Embedder | None, path: Path | None = None, chroma_collection=None):
        self.chunks = chunks
        self.cfg = cfg
        self.path = path
        b = cfg.get("bm25", {})
        self._tok_kw = {"stemming": b.get("stemming", "light"), "remove_stopwords": b.get("remove_stopwords", True)}
        corpus_tokens = [tokenize(c.text, **self._tok_kw) or ["_vazio_"] for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokens, k1=float(b.get("k1", 1.5)), b=float(b.get("b", 0.75)))
        self.embeddings = embeddings
        self.embedder = embedder
        self.chroma = chroma_collection
        self.id_to_pos = {c.id: i for i, c in enumerate(chunks)}

    @property
    def has_dense(self) -> bool:
        return self.embedder is not None and (self.embeddings is not None or self.chroma is not None)

    def query_tokens(self, query: str) -> list[str]:
        return tokenize(query, **self._tok_kw)

    def bm25_scores(self, query: str) -> np.ndarray:
        toks = self.query_tokens(query)
        if not toks:
            return np.zeros(len(self.chunks), dtype=np.float32)
        return np.asarray(self.bm25.get_scores(toks), dtype=np.float32)

    def dense_scores(self, query: str) -> np.ndarray | None:
        if not self.has_dense:
            return None
        q = self.embedder.embed_query(query)
        if self.embeddings is not None:
            return (self.embeddings @ q).astype(np.float32)
        # Chroma (espaço cosseno): distance = 1 - cos
        res = self.chroma.query(query_embeddings=[q.tolist()], n_results=len(self.chunks))
        out = np.full(len(self.chunks), -1.0, dtype=np.float32)
        for cid, dist in zip(res["ids"][0], res["distances"][0]):
            out[self.id_to_pos[cid]] = 1.0 - float(dist)
        return out


def _save_chunks(path: Path, chunks: list[Chunk]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def _load_chunks(path: Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk.from_dict(json.loads(l)) for l in f if l.strip()]


def _chroma_collection(dir_: Path, chunks: list[Chunk], emb: np.ndarray | None, create: bool):
    import chromadb
    client = chromadb.PersistentClient(path=str(dir_ / "chroma"))
    col = client.get_or_create_collection("chunks", metadata={"hnsw:space": "cosine"})
    if create and emb is not None:
        meta = [{k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))} for c in chunks]
        col.upsert(ids=[c.id for c in chunks], embeddings=emb.tolist(),
                   documents=[c.text for c in chunks], metadatas=meta)
    return col


def build_index(cfg: dict | None = None, corpus_path: str | Path | None = None, force: bool = False,
                with_dense: bool = True) -> Index:
    """Carrega o índice persistido da configuração, ou constrói se não existir/`force`."""
    cfg = cfg or load_config()
    corpus_path = resolve_path(corpus_path) if corpus_path else default_corpus_path(cfg)
    fp = corpus_fingerprint(corpus_path)
    dir_ = resolve_path(cfg["paths"]["index_dir"]) / index_key(cfg, fp)
    store = cfg.get("vector_store", "numpy")

    embedder: Embedder | None = None
    if with_dense:
        try:
            embedder = make_embedder(cfg)
        except Exception as e:  # sem rede/modelo -> segue só com BM25
            log.error("Embedder indisponível (%s: %s) — índice denso desativado", type(e).__name__, e)

    meta_p, chunks_p, emb_p = dir_ / "meta.json", dir_ / "chunks.jsonl", dir_ / "embeddings.npy"
    if not force and meta_p.exists() and chunks_p.exists():
        chunks = _load_chunks(chunks_p)
        emb = None
        if embedder is not None:
            if store == "numpy" and emb_p.exists():
                emb = np.load(emb_p)
            elif store == "numpy":
                emb = embedder.embed_passages([c.text for c in chunks])
                np.save(emb_p, emb)
        col = _chroma_collection(dir_, chunks, None, create=False) if (store == "chroma" and embedder) else None
        log.info("Índice carregado de %s (%d chunks)", dir_, len(chunks))
        return Index(chunks, cfg, emb, embedder, dir_, col)

    t0 = time.time()
    docs = load_corpus(corpus_path)
    chunks = chunk_corpus(docs, cfg)
    dir_.mkdir(parents=True, exist_ok=True)
    _save_chunks(chunks_p, chunks)
    emb = None
    col = None
    if embedder is not None:
        emb = embedder.embed_passages([c.text for c in chunks])
        if store == "chroma":
            col = _chroma_collection(dir_, chunks, emb, create=True)
            emb_mem = None
        else:
            np.save(emb_p, emb)
            emb_mem = emb
    else:
        emb_mem = None
    n_words = [c.metadata.get("n_words", 0) for c in chunks]
    meta = {"corpus_path": str(corpus_path), "corpus_fingerprint": fp, "n_services": len(docs),
            "n_chunks": len(chunks), "avg_words": round(float(np.mean(n_words)) if n_words else 0, 1),
            "max_words": int(max(n_words) if n_words else 0), "chunking": cfg["chunking"],
            "embedding": cfg.get("embedding"), "vector_store": store, "dense": emb is not None,
            "build_seconds": round(time.time() - t0, 2)}
    meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Índice construído em %s: %s", dir_, meta)
    return Index(chunks, cfg, emb_mem if store != "chroma" else None, embedder, dir_, col)


# Grade de comparação para a avaliação de chunking.
GRID = [
    {"strategy": "structural", "size": 160, "overlap": 0},
    {"strategy": "fixed", "size": 120, "overlap": 30},
    {"strategy": "fixed", "size": 250, "overlap": 50},
    {"strategy": "hybrid", "size": 160, "overlap": 40},
    {"strategy": "hybrid", "size": 80, "overlap": 20},
]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="(Re)constrói os índices do citizen_rag")
    ap.add_argument("--corpus", help="caminho do corpus JSONL (padrão: config.paths.corpus ou fixture)")
    ap.add_argument("--strategy", choices=["structural", "fixed", "hybrid"])
    ap.add_argument("--size", type=int)
    ap.add_argument("--overlap", type=int)
    ap.add_argument("--backend", choices=["sentence_transformers", "hashing"])
    ap.add_argument("--grid", action="store_true", help="constrói todas as configurações de GRID")
    ap.add_argument("--reuse", action="store_true", help="reaproveita índice existente (padrão: reconstrói)")
    a = ap.parse_args(argv)
    base = load_config()
    if a.backend:
        base["embedding"]["backend"] = a.backend
    configs = GRID if a.grid else [{k: v for k, v in {"strategy": a.strategy, "size": a.size,
                                                       "overlap": a.overlap}.items() if v is not None}]
    for over in configs:
        cfg = load_config(overrides={"chunking": over, "embedding": base["embedding"]})
        idx = build_index(cfg, a.corpus, force=not a.reuse)
        meta = json.loads((idx.path / "meta.json").read_text(encoding="utf-8"))
        print(f"{idx.path.name}: {meta['n_services']} serviços, {meta['n_chunks']} chunks, "
              f"média {meta['avg_words']} palavras (máx {meta['max_words']}), denso={meta['dense']}, "
              f"{meta['build_seconds']}s")


if __name__ == "__main__":
    main()
