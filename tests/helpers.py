"""Utilidades compartilhadas pelos testes (offline: LLM stub, corpus fixture)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

GENAI = Path(__file__).resolve().parent.parent
if str(GENAI) not in sys.path:
    sys.path.insert(0, str(GENAI))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

FIXTURE = GENAI / "tests" / "fixtures" / "corpus_fixture.jsonl"
# CITIZEN_RAG_TEST_BACKEND=hashing roda sem baixar o modelo de embedding.
BACKEND = os.environ.get("CITIZEN_RAG_TEST_BACKEND", "sentence_transformers")
OVERRIDES = {"embedding": {"backend": BACKEND}, "llm": {"provider": "none"},
             "paths": {"index_dir": "tests/.indexes", "log_file": "tests/.logs/rag_test.log"}}
if BACKEND == "hashing":   # limiar do gate depende do embedding (hashing: cosseno ~0.1 fora de escopo, ~0.25-0.35 dentro)
    OVERRIDES["gate"] = {"min_dense_score": 0.2}

_cache: dict = {}


def fixture_index():
    from citizen_rag.config import load_config
    from citizen_rag.indexing import build_index
    if "index" not in _cache:
        _cache["index"] = build_index(load_config(overrides=OVERRIDES), FIXTURE)
    return _cache["index"]


def make_assistant(llm=None, overrides=None, corpus=FIXTURE, index=None):
    from citizen_rag.config import deep_merge
    from citizen_rag.rag import RAGAssistant
    ov = deep_merge(OVERRIDES, overrides or {})
    idx = index if index is not None else (fixture_index() if corpus == FIXTURE else None)
    return RAGAssistant(corpus_path=corpus, overrides=ov, llm=llm, index=idx)
