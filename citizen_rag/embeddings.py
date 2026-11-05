"""Backends de embedding.

- SentenceTransformerEmbedder: modelo multilíngue (padrão intfloat/multilingual-e5-small), com os
  prefixos assimétricos exigidos pela família e5 ("query: " / "passage: ") e vetores normalizados
  (produto interno = cosseno).
- HashingEmbedder: fallback offline determinístico (n-gramas de caracteres -> hashing trick).
  Não precisa de download; útil em testes e como baseline "denso" fraco na comparação.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

from .config import get_logger
from .textproc import strip_accents

log = get_logger("embeddings")
_MODEL_CACHE: dict[tuple[str, str], object] = {}


class Embedder:
    name = "base"
    dim = 0

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: str, query_prefix: str | None = None, passage_prefix: str | None = None,
                 batch_size: int = 32, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer  # import tardio: pesado
        self.name = model
        is_e5 = "e5" in model.lower()
        self.query_prefix = query_prefix if query_prefix is not None else ("query: " if is_e5 else "")
        self.passage_prefix = passage_prefix if passage_prefix is not None else ("passage: " if is_e5 else "")
        self.batch_size = batch_size
        key = (model, device)
        if key not in _MODEL_CACHE:
            log.info("Carregando modelo de embedding %s (%s)", model, device)
            _MODEL_CACHE[key] = SentenceTransformer(model, device=device)
        self.model = _MODEL_CACHE[key]
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode([self.passage_prefix + t for t in texts], batch_size=self.batch_size,
                              normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(v, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        v = self.model.encode([self.query_prefix + text], normalize_embeddings=True,
                              show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(v[0], dtype=np.float32)


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 512, ngram: tuple[int, int] = (3, 5)):
        self.name = f"hashing-{dim}"
        self.dim = dim
        self.ngram = ngram

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = " " + re.sub(r"\s+", " ", strip_accents(text.lower())) + " "
        for n in range(self.ngram[0], self.ngram[1] + 1):
            for i in range(len(t) - n + 1):
                h = int.from_bytes(hashlib.md5(t[i:i + n].encode()).digest()[:4], "little")
                v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vec(text)[None, :])[0]


def make_embedder(cfg: dict) -> Embedder:
    e = cfg.get("embedding", {})
    backend = e.get("backend", "sentence_transformers")
    if backend == "hashing":
        return HashingEmbedder(dim=int(e.get("dim", 512)))
    return SentenceTransformerEmbedder(e.get("model", "intfloat/multilingual-e5-small"),
                                       e.get("query_prefix"), e.get("passage_prefix"),
                                       int(e.get("batch_size", 32)), e.get("device", "cpu"))


def embedder_slug(cfg: dict) -> str:
    e = cfg.get("embedding", {})
    if e.get("backend") == "hashing":
        return f"hashing{e.get('dim', 512)}"
    return re.sub(r"[^a-zA-Z0-9]+", "-", e.get("model", "e5")).strip("-").lower()[-40:]
