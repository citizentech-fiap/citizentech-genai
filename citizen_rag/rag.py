"""RAGAssistant — orquestra guardrails de entrada, recuperação, gate de evidência, geração e
validação da saída. Interface (RPA/CONTRACTS.md):

    RAGAssistant(corpus_path=None).answer(question, history=None, service_id=None) -> dict
      answer: str, citations: list[{n, service_id, titulo, url, secao}], abstained: bool,
      reason: str | None, chunks: list[{id, text, score, metadata}], blocked: bool

Nunca lança exceção: qualquer falha vira abstained=True + reason, e o robô cai no template.
Valores de `reason`: pergunta_vazia | pergunta_longa | prompt_injection | indice_indisponivel |
evidencia_insuficiente | llm_indisponivel | llm_erro:<tipo> | sem_evidencia_llm | citacao_invalida |
sem_citacao | vazamento_prompt | baixo_suporte | nao_fundamentado | erro_interno | None (respondido).
Diagnóstico completo da última chamada em `assistant.last_trace` (usado pela avaliação).
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

from . import guardrails as G
from .config import get_logger, load_config, resolve_path
from .generator import CANARY, Generator, GenerationResult
from .indexing import build_index
from .llm import BaseLLM, make_llm
from .retriever import RetrievedChunk, Retriever
from .textproc import tokenize

_ANAPHORA = {"isso", "esse", "essa", "este", "esta", "ele", "ela", "dele", "dela", "nele", "nela",
             "mesmo", "tambem", "disso", "desse", "dessa", "ai", "entao"}


def normalize_history(history) -> list[dict]:
    """Aceita [{role, content}], [(role, content)] ou [str] (tratadas como falas do cidadão)."""
    out = []
    for h in history or []:
        if isinstance(h, dict):
            role, content = h.get("role", "user"), h.get("content") or h.get("text") or ""
        elif isinstance(h, (list, tuple)) and len(h) == 2:
            role, content = h
        else:
            role, content = "user", str(h)
        role = "user" if str(role).lower() in ("user", "usuario", "cidadao", "human") else "assistant"
        out.append({"role": role, "content": str(content)})
    return out


def condense_query(question: str, history: list[dict]) -> str:
    """Multi-turno simples: se a pergunta é curta ou anafórica ("e o prazo disso?"), concatena a
    última fala do cidadão para a busca recuperar o mesmo serviço."""
    prev_user = [h["content"] for h in history if h["role"] == "user"]
    if not prev_user:
        return question
    raw = set(tokenize(question, stemming="none", remove_stopwords=False))
    content = tokenize(question)
    if len(content) <= 4 or raw & _ANAPHORA:
        return f"{prev_user[-1]} {question}"
    return question


class RAGAssistant:
    def __init__(self, corpus_path: str | Path | None = None, config_path: str | Path | None = None,
                 overrides: dict | None = None, llm: BaseLLM | None = None, index=None):
        self.cfg = load_config(config_path, overrides)
        self.log = get_logger("rag", self.cfg["paths"].get("log_file"))
        self.corpus_path = resolve_path(corpus_path) if corpus_path else None
        self.last_trace: dict = {}
        self._init_error: str | None = None
        self.llm = llm if llm is not None else make_llm(self.cfg)
        self.generator = Generator(self.cfg, self.llm)
        try:
            self.index = index or build_index(self.cfg, self.corpus_path)
            self.retriever = Retriever(self.index, self.cfg)
        except Exception as e:
            self._init_error = f"{type(e).__name__}: {e}"
            self.index = self.retriever = None
            self.log.error("Falha ao preparar índice: %s", self._init_error)
        self.log.info("RAGAssistant pronto: chunking=%s retriever=%s llm=%s prompt=%s",
                      self.cfg["chunking"], self.cfg["retriever"]["mode"], self.llm.model, self.generator.version)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _result(answer: str, abstained: bool, reason: str | None, chunks: list | None = None,
                citations: list | None = None, blocked: bool = False) -> dict:
        return {"answer": answer, "citations": citations or [], "abstained": abstained, "reason": reason,
                "chunks": chunks or [], "blocked": blocked}

    def _finish(self, res: dict, trace: dict, t0: float) -> dict:
        trace["reason"] = res["reason"]
        trace["abstained"] = res["abstained"]
        trace["blocked"] = res["blocked"]
        trace["total_s"] = round(time.time() - t0, 3)
        self.last_trace = trace
        self.log.info("answer: abstained=%s reason=%s blocked=%s citations=%s t=%.2fs",
                      res["abstained"], res["reason"], res["blocked"], [c["n"] for c in res["citations"]],
                      trace["total_s"])
        return res

    def sanitize(self, chunks: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], list[str]]:
        flagged = []
        if not self.cfg["guardrails"].get("sanitize_chunks", True):
            return chunks, flagged
        for c in chunks:
            clean, hits = G.sanitize_chunk_text(c.text)
            if hits:
                flagged.append(c.id)
                self.log.warning("Injeção indireta neutralizada no chunk %s: %s", c.id, hits)
            c.text = clean
        return chunks, flagged

    def retrieve(self, query: str, service_id=None, **kw) -> list[RetrievedChunk]:
        """Recuperação + sanitização (exposto para a avaliação de componentes)."""
        chunks = self.retriever.retrieve(query, service_id=service_id, **kw)
        return self.sanitize(chunks)[0]

    # ------------------------------------------------------------------ API
    def answer(self, question: str, history=None, service_id: str | None = None) -> dict:
        t0 = time.time()
        trace: dict = {"service_id_filter": service_id}
        try:
            return self._answer(question, history, service_id, trace, t0)
        except Exception as e:  # nunca propaga
            self.log.error("Erro interno no RAG: %s\n%s", e, traceback.format_exc())
            trace["erro"] = f"{type(e).__name__}: {e}"
            return self._finish(self._result(G.ABSTENTION_TEXT, True, "erro_interno"), trace, t0)

    def _answer(self, question, history, service_id, trace, t0) -> dict:
        gcfg = self.cfg["guardrails"]
        question = (question or "").strip()
        if not question:
            return self._finish(self._result(G.ABSTENTION_TEXT, True, "pergunta_vazia"), trace, t0)
        if len(question) > int(gcfg.get("max_question_chars", 1500)):
            return self._finish(self._result(G.BLOCKED_TEXT, True, "pergunta_longa", blocked=True), trace, t0)

        # 1. PII: tudo a jusante (busca, LLM, logs) usa a versão mascarada
        mask = gcfg.get("mask_pii", True)
        q = G.mask_pii(question) if mask else question
        trace["question_masked"] = q

        # 2. injeção direta na pergunta
        inj = G.detect_injection(q)
        trace["injection"] = inj.patterns
        if inj.flagged and gcfg.get("block_injection", True):
            self.log.warning("Pergunta bloqueada por injeção: %s", inj.patterns)
            return self._finish(self._result(G.BLOCKED_TEXT, True, "prompt_injection", blocked=True), trace, t0)

        # 3. histórico: mascarado; turnos com injeção são neutralizados
        hist = normalize_history(history)
        for h in hist:
            h["content"] = G.mask_pii(h["content"]) if mask else h["content"]
            if G.detect_injection(h["content"]).flagged:
                h["content"] = G.REMOVED_MARK
        query = condense_query(q, hist)
        trace["retrieval_query"] = query

        if self.retriever is None:
            trace["erro"] = self._init_error
            return self._finish(self._result(G.ABSTENTION_TEXT, True, "indice_indisponivel"), trace, t0)

        # 4. recuperação + sanitização (injeção indireta)
        t1 = time.time()
        chunks = self.retriever.retrieve(query, service_id=service_id)
        chunks, flagged = self.sanitize(chunks)
        trace["retrieval"] = [{"id": c.id, "score": round(c.score, 4), **{k: (round(v, 4) if v is not None else None)
                                                                          for k, v in c.signals.items()}} for c in chunks]
        trace["sanitized_chunks"] = flagged
        trace["retrieval_s"] = round(time.time() - t1, 3)
        out_chunks = [c.to_dict() for c in chunks]

        # 5. gate de evidência (abstém sem gastar LLM)
        gate = G.evidence_gate(chunks, self.cfg)
        trace["gate"] = gate.__dict__
        if not gate.passed:
            return self._finish(self._result(G.ABSTENTION_TEXT, True, "evidencia_insuficiente", out_chunks), trace, t0)

        # 6. geração
        if not getattr(self.llm, "available", True):
            return self._finish(self._result(G.ABSTENTION_TEXT, True, "llm_indisponivel", out_chunks), trace, t0)
        gen: GenerationResult = self.generator.generate(q, chunks, hist)
        trace["llm"] = {"model": gen.model, "ok": gen.ok, "error": gen.error, "cached": gen.cached,
                        "latency_s": gen.latency_s, "prompt_version": gen.prompt_version, "usage": gen.usage}
        trace["raw_answer"] = gen.text
        if not gen.ok:
            reason = "llm_indisponivel" if gen.error == "llm_indisponivel" else f"llm_erro:{gen.error}"
            return self._finish(self._result(G.ABSTENTION_TEXT, True, reason, out_chunks), trace, t0)
        text = gen.text.strip()

        # 7. validação da saída
        if G.prompt_leak(text, gen.system_prompt, CANARY):
            self.log.warning("Saída descartada: possível vazamento do prompt")
            return self._finish(self._result(G.ABSTENTION_TEXT, True, "vazamento_prompt", out_chunks), trace, t0)
        cit = G.verify_citations(text, len(chunks), gcfg.get("citation_policy", "reject"),
                                 gcfg.get("require_citation", True))
        trace["citations_check"] = {"ok": cit.ok, "valid": cit.valid, "invalid": cit.invalid, "reason": cit.reason}
        if G.is_abstention(text) and not cit.valid:
            return self._finish(self._result(G.ABSTENTION_TEXT, True, "sem_evidencia_llm", out_chunks), trace, t0)
        if not cit.ok:
            return self._finish(self._result(G.ABSTENTION_TEXT, True, cit.reason, out_chunks), trace, t0)
        text = cit.cleaned_answer
        support = G.lexical_support(text, [c.text for c in chunks])
        trace["lexical_support"] = support
        if support < float(gcfg.get("min_lexical_support", 0.0)):
            return self._finish(self._result(G.ABSTENTION_TEXT, True, "baixo_suporte", out_chunks), trace, t0)
        if gcfg.get("groundedness_check", False):
            grounded, info = G.judge_groundedness(self.llm, text, G.judge_evidence(chunks))
            trace["groundedness"] = {"grounded": grounded, **info}
            if grounded is False:
                return self._finish(self._result(G.ABSTENTION_TEXT, True, "nao_fundamentado", out_chunks), trace, t0)

        if mask:
            text = G.mask_pii(text)
        citations = []
        for n in cit.valid:
            m = chunks[n - 1].metadata
            citations.append({"n": n, "service_id": m.get("service_id"), "titulo": m.get("titulo"),
                              "url": m.get("url"), "secao": m.get("secao")})
        return self._finish(self._result(text, False, None, out_chunks, citations), trace, t0)


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Pergunta ao assistente RAG")
    ap.add_argument("question")
    ap.add_argument("--service-id")
    ap.add_argument("--corpus")
    a = ap.parse_args()
    rag = RAGAssistant(a.corpus)
    r = rag.answer(a.question, service_id=a.service_id)
    r["chunks"] = [{"id": c["id"], "score": c["score"]} for c in r["chunks"]]
    print(json.dumps(r, ensure_ascii=False, indent=2))
