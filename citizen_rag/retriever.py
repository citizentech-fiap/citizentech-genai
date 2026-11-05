"""Retriever: BM25 (lexical), denso (embeddings) e híbrido (Reciprocal Rank Fusion), com filtro por
metadado `service_id` e reranking opcional por cross-encoder.

Todo resultado traz os sinais brutos (`bm25`, `dense`, `rrf`, `rerank`) — o gate de evidência e a
avaliação de componentes usam esses números independentemente do modo escolhido.

Demo: python -m citizen_rag.retriever "como peço o seguro-desemprego?" [--mode bm25|dense|hybrid|all]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np

from .config import get_logger, load_config
from .indexing import Index, build_index

log = get_logger("retriever")
_RERANKER_CACHE: dict[str, object] = {}


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float
    metadata: dict
    rank: int = 0
    signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "score": round(float(self.score), 6),
                "metadata": dict(self.metadata, **{"signals": {k: (round(float(v), 4) if v is not None else None)
                                                               for k, v in self.signals.items()}})}


def rrf_fuse(rankings: dict[str, list[int]], k: int = 60, weights: dict[str, float] | None = None) -> dict[int, float]:
    """RRF: score(d) = Σ_r w_r / (k + rank_r(d)), rank começando em 1."""
    weights = weights or {}
    fused: dict[int, float] = {}
    for name, order in rankings.items():
        w = float(weights.get(name, 1.0))
        for r, pos in enumerate(order, 1):
            fused[pos] = fused.get(pos, 0.0) + w / (k + r)
    return fused


class Retriever:
    def __init__(self, index: Index, cfg: dict | None = None):
        self.index = index
        self.cfg = cfg or index.cfg
        self._reranker = None

    # ------------------------------------------------------------------ util
    def _mask(self, service_id: str | list[str] | None) -> np.ndarray:
        n = len(self.index.chunks)
        if not service_id:
            return np.ones(n, dtype=bool)
        ids = {service_id} if isinstance(service_id, str) else set(service_id)
        return np.array([c.metadata.get("service_id") in ids for c in self.index.chunks], dtype=bool)

    @staticmethod
    def _top(scores: np.ndarray, mask: np.ndarray, k: int, require_positive: bool = False) -> list[int]:
        idx = np.where(mask & (scores > 0) if require_positive else mask)[0]
        if idx.size == 0:
            return []
        order = idx[np.argsort(-scores[idx], kind="stable")]
        return order[:k].tolist()

    def _get_reranker(self, model: str):
        if model not in _RERANKER_CACHE:
            from sentence_transformers import CrossEncoder
            log.info("Carregando reranker %s", model)
            _RERANKER_CACHE[model] = CrossEncoder(model, device=self.cfg.get("embedding", {}).get("device", "cpu"))
        return _RERANKER_CACHE[model]

    # ------------------------------------------------------------------ API
    def retrieve(self, query: str, top_k: int | None = None, mode: str | None = None,
                 service_id: str | list[str] | None = None, rerank: bool | None = None) -> list[RetrievedChunk]:
        rc = self.cfg.get("retriever", {})
        mode = (mode or rc.get("mode", "hybrid")).lower()
        top_k = int(top_k or rc.get("top_k", 5))
        cand_k = max(top_k, int(rc.get("candidate_k", 20)))
        rr_cfg = rc.get("reranker", {}) or {}
        do_rerank = rr_cfg.get("enabled", False) if rerank is None else rerank

        if not query or not query.strip() or not self.index.chunks:
            return []
        mask = self._mask(service_id)
        if service_id and not mask.any():
            if rc.get("fallback_unfiltered", True):
                log.info("Filtro service_id=%s sem chunks — buscando sem filtro", service_id)
                mask = self._mask(None)
            else:
                return []

        bm25 = self.index.bm25_scores(query)
        dense = self.index.dense_scores(query)
        if mode in ("dense", "hybrid") and dense is None:
            log.warning("Índice denso indisponível — modo %s degradado para bm25", mode)
            mode = "bm25"

        if mode == "bm25":
            order = self._top(bm25, mask, cand_k, require_positive=True)
            base = {p: float(bm25[p]) for p in order}
        elif mode == "dense":
            order = self._top(dense, mask, cand_k)
            base = {p: float(dense[p]) for p in order}
        elif mode == "hybrid":
            rankings = {"bm25": self._top(bm25, mask, cand_k, require_positive=True),
                        "dense": self._top(dense, mask, cand_k)}
            fused = rrf_fuse(rankings, k=int(rc.get("rrf_k", 60)), weights=rc.get("weights"))
            order = sorted(fused, key=lambda p: -fused[p])[:cand_k]
            base = fused
        else:
            raise ValueError(f"Modo de retriever desconhecido: {mode}")

        results = [RetrievedChunk(id=self.index.chunks[p].id, text=self.index.chunks[p].text, score=base[p],
                                  metadata=dict(self.index.chunks[p].metadata),
                                  signals={"bm25": float(bm25[p]),
                                           "dense": float(dense[p]) if dense is not None else None,
                                           "rrf": float(base[p]) if mode == "hybrid" else None,
                                           "rerank": None})
                   for p in order]

        if do_rerank and results:
            try:
                ce = self._get_reranker(rr_cfg.get("model", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"))
                pool = results[: int(rr_cfg.get("top_n", 20))]
                scores = ce.predict([(query, r.text) for r in pool], show_progress_bar=False)
                for r, s in zip(pool, scores):
                    r.signals["rerank"] = float(s)
                    r.score = float(s)
                results = sorted(pool, key=lambda r: -r.score)
            except Exception as e:
                log.error("Reranker falhou (%s) — mantendo ordem original", e)

        results = results[:top_k]
        for i, r in enumerate(results, 1):
            r.rank = i
        log.info("retrieve mode=%s k=%d filtro=%s -> %s", mode, top_k, service_id,
                 [(r.id, round(r.score, 3)) for r in results])
        return results


def make_retriever(cfg: dict | None = None, corpus_path=None) -> Retriever:
    cfg = cfg or load_config()
    return Retriever(build_index(cfg, corpus_path), cfg)


SAMPLE_QUESTIONS = [
    "Fui demitido sem justa causa, como peço o seguro-desemprego?",
    "Quais documentos preciso levar para tirar o passaporte?",
    "Quanto custa emitir o passaporte?",
    "Qual é a renda máxima para entrar no CadÚnico?",
    "Qual o prazo para pedir o seguro depois da demissão?",
    "Onde faço a entrevista do Cadastro Único?",
    "Como me formalizo como MEI?",
    "Quem tem direito ao BPC?",
    "Como renovo minha CNH?",                       # fora de escopo
    "Qual a previsão do tempo para amanhã?",        # fora de escopo
]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Demonstração do retriever")
    ap.add_argument("questions", nargs="*")
    ap.add_argument("--mode", default="all", choices=["bm25", "dense", "hybrid", "all"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--corpus")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--service-id")
    a = ap.parse_args(argv)
    r = make_retriever(corpus_path=a.corpus)
    modes = ["bm25", "dense", "hybrid"] if a.mode == "all" else [a.mode]
    for q in a.questions or SAMPLE_QUESTIONS:
        print(f"\n=== {q}")
        for m in modes:
            res = r.retrieve(q, top_k=a.k, mode=m, service_id=a.service_id, rerank=a.rerank or None)
            top = " | ".join(f"{x.id} ({x.score:.3f}; bm25={x.signals['bm25']:.2f}, "
                             f"cos={x.signals['dense'] if x.signals['dense'] is None else round(x.signals['dense'], 3)})"
                             for x in res)
            print(f"  [{m:6}] {top}")


if __name__ == "__main__":
    main()
