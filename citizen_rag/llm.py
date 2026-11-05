"""Clientes de LLM com interface única `complete(system, user, **params) -> LLMResponse`.

- GeminiLLM: Google Gemini (free tier) via SDK `google-genai`; timeout, retry com backoff exponencial
  para 429/5xx/timeout (respeita o retryDelay sugerido pela API), modelo de fallback e tratamento de
  resposta vazia/bloqueada. NUNCA lança exceção: erros viram LLMResponse(ok=False, error=...).
- CachedLLM: cache em disco por sha256(modelo + prompt + parâmetros) — avaliações repetidas não gastam cota.
- StubLLM: LLM falso e determinístico, injetável nos testes e na avaliação offline.
- UnavailableLLM: usado quando não há chave/SDK; o pipeline abstém com reason="llm_indisponivel".
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import gemini_api_key, get_logger, resolve_path

log = get_logger("llm")


@dataclass
class LLMResponse:
    text: str
    ok: bool
    model: str = ""
    error: str | None = None          # llm_indisponivel | rate_limit | timeout | resposta_vazia | bloqueado | erro_api
    cached: bool = False
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)


class BaseLLM:
    model = "base"
    available = True

    def complete(self, system: str, user: str, **params) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


class UnavailableLLM(BaseLLM):
    available = False

    def __init__(self, reason: str = "sem GEMINI_API_KEY"):
        self.model = "indisponivel"
        self.reason = reason

    def complete(self, system: str, user: str, **params) -> LLMResponse:
        return LLMResponse("", False, self.model, "llm_indisponivel")


class StubLLM(BaseLLM):
    """LLM falso. `responder(system, user) -> str` define a resposta; o padrão cita o trecho [1].
    Registra as chamadas em `self.calls` para os testes conferirem o que foi enviado."""

    def __init__(self, responder: Callable[[str, str], str] | str | None = None, model: str = "stub"):
        self.model = model
        self.calls: list[dict] = []
        if isinstance(responder, str):
            self._responder = lambda s, u, _r=responder: _r
        else:
            self._responder = responder or self._default

    @staticmethod
    def _default(system: str, user: str) -> str:
        m = re.search(r'<documento n="1"[^>]*>\s*(.*?)\s*</documento>', user, re.S)
        if not m:
            return "Não encontrei essa informação nas Cartas de Serviço consultadas."
        trecho = " ".join(m.group(1).split()[:40])
        return f"**O que fazer:** {trecho} [1]"

    def complete(self, system: str, user: str, **params) -> LLMResponse:
        self.calls.append({"system": system, "user": user, "params": params})
        try:
            return LLMResponse(self._responder(system, user), True, self.model)
        except Exception as e:
            return LLMResponse("", False, self.model, f"erro_api: {e}")


class GeminiLLM(BaseLLM):
    def __init__(self, cfg: dict, api_key: str):
        from google import genai
        from google.genai import types
        self._types = types
        c = cfg.get("llm", {})
        self.model = c.get("model", "gemini-3.6-flash")
        self.fallback_model = c.get("fallback_model") or None
        self.params = c
        self.max_retries = int(c.get("max_retries", 3))
        self.backoff = float(c.get("backoff_s", 2))
        self.client = genai.Client(api_key=api_key,
                                   http_options=types.HttpOptions(timeout=int(float(c.get("timeout_s", 30)) * 1000)))

    def _config(self, system: str, params: dict):
        t = self._types
        p = {**self.params, **params}
        kw = {"system_instruction": system, "max_output_tokens": int(p.get("max_output_tokens", 700)),
              "candidate_count": 1}
        if p.get("temperature") is not None:
            kw["temperature"] = float(p["temperature"])
        if p.get("top_p") is not None:
            kw["top_p"] = float(p["top_p"])
        if p.get("thinking_level"):
            kw["thinking_config"] = t.ThinkingConfig(thinking_level=str(p["thinking_level"]).upper())
        if p.get("json_mode"):
            kw["response_mime_type"] = "application/json"
        return t.GenerateContentConfig(**kw)

    @staticmethod
    def _retry_delay(err) -> float | None:
        try:
            for d in (err.details or {}).get("error", {}).get("details", []):
                if "retryDelay" in d:
                    return float(str(d["retryDelay"]).rstrip("s"))
        except Exception:
            pass
        return None

    def _call(self, model: str, system: str, user: str, params: dict) -> LLMResponse:
        from google.genai import errors
        last = "erro_api"
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            try:
                resp = self.client.models.generate_content(model=model, contents=user,
                                                           config=self._config(system, params))
                text = (getattr(resp, "text", None) or "").strip()
                if not text:
                    fr = ""
                    try:
                        fr = str(resp.candidates[0].finish_reason)
                    except Exception:
                        fr = str(getattr(getattr(resp, "prompt_feedback", None), "block_reason", ""))
                    err = "bloqueado" if ("SAFETY" in fr or "BLOCK" in fr or "PROHIBITED" in fr) else "resposta_vazia"
                    log.warning("Gemini %s retornou texto vazio (%s)", model, fr)
                    return LLMResponse("", False, model, err, latency_s=time.time() - t0)
                usage = {}
                um = getattr(resp, "usage_metadata", None)
                if um is not None:
                    usage = {"prompt": getattr(um, "prompt_token_count", None),
                             "output": getattr(um, "candidates_token_count", None)}
                return LLMResponse(text, True, model, latency_s=time.time() - t0, usage=usage)
            except errors.APIError as e:
                code = getattr(e, "code", 0) or 0
                if code == 429:
                    last = "rate_limit"
                elif code >= 500:
                    last = "erro_servidor"
                else:  # 400/401/403/404: não adianta repetir
                    log.error("Gemini erro %s: %s", code, str(e)[:200])
                    return LLMResponse("", False, model, f"erro_api_{code}")
                delay = self._retry_delay(e) if code == 429 else None
            except Exception as e:  # timeout (httpx), rede, etc.
                name = type(e).__name__.lower()
                last = "timeout" if "timeout" in name or "timed out" in str(e).lower() else "erro_rede"
                delay = None
            if attempt < self.max_retries:
                wait = min(delay if delay else self.backoff * (2 ** attempt), 60) + random.uniform(0, 0.5)
                log.warning("Gemini %s falhou (%s), tentativa %d/%d — aguardando %.1fs",
                            model, last, attempt + 1, self.max_retries, wait)
                time.sleep(wait)
        return LLMResponse("", False, model, last)

    def complete(self, system: str, user: str, **params) -> LLMResponse:
        r = self._call(self.model, system, user, params)
        if not r.ok and self.fallback_model and r.error in ("rate_limit", "erro_servidor", "timeout", "erro_api_404"):
            log.warning("Usando modelo de fallback %s após %s", self.fallback_model, r.error)
            r = self._call(self.fallback_model, system, user, params)
        return r


class CachedLLM(BaseLLM):
    """Cache em disco: chave = sha256(modelo, system, user, parâmetros). Só respostas ok são gravadas."""

    def __init__(self, inner: BaseLLM, cache_dir: str | Path, params: dict | None = None):
        self.inner = inner
        self.model = inner.model
        self.available = inner.available
        self.dir = resolve_path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.params = {k: params.get(k) for k in ("temperature", "top_p", "max_output_tokens", "thinking_level")} if params else {}

    def key(self, system: str, user: str, params: dict) -> str:
        blob = json.dumps({"model": self.inner.model, "system": system, "user": user,
                           "params": {**self.params, **params}}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def complete(self, system: str, user: str, **params) -> LLMResponse:
        k = self.key(system, user, params)
        p = self.dir / f"{k}.json"
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                return LLMResponse(d["text"], True, d.get("model", self.model), cached=True)
            except (OSError, json.JSONDecodeError, KeyError):
                pass
        r = self.inner.complete(system, user, **params)
        if r.ok and r.text:
            try:
                p.write_text(json.dumps({"model": r.model, "text": r.text, "usage": r.usage,
                                         "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False),
                             encoding="utf-8")
            except OSError:
                pass
        return r


def make_llm(cfg: dict) -> BaseLLM:
    """LLM conforme config; sem chave ou sem SDK -> UnavailableLLM (o pipeline degrada com segurança)."""
    c = cfg.get("llm", {})
    if c.get("provider", "gemini") == "none":
        return UnavailableLLM("provider=none")
    key = gemini_api_key()
    if not key:
        log.warning("GEMINI_API_KEY não configurada — gerador indisponível (respostas por abstenção/template)")
        return UnavailableLLM("sem GEMINI_API_KEY")
    try:
        llm: BaseLLM = GeminiLLM(cfg, key)
    except Exception as e:
        log.error("Falha ao inicializar Gemini: %s", e)
        return UnavailableLLM(f"erro init: {e}")
    if c.get("cache", True):
        llm = CachedLLM(llm, cfg["paths"]["llm_cache_dir"], c)
    return llm
