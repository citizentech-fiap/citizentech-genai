"""Carregamento de configuração (config.yaml), variáveis de ambiente e logging do citizen_rag."""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

GENAI_ROOT = Path(__file__).resolve().parent.parent          # .../GS/genAI
DEFAULT_CONFIG_PATH = GENAI_ROOT / "config.yaml"
_LOGGER_NAME = "citizen_rag"
_env_loaded = False


def load_env() -> None:
    """Carrega .env de genAI/ e, se existir, de ../RPA/.env (sem sobrescrever variáveis já definidas)."""
    global _env_loaded
    if _env_loaded:
        return
    try:
        from dotenv import load_dotenv
        for p in (GENAI_ROOT / ".env", GENAI_ROOT.parent / "RPA" / ".env"):
            if p.exists():
                load_dotenv(p, override=False)
    except Exception:  # python-dotenv ausente não pode derrubar o assistente
        pass
    _env_loaded = True


def deep_merge(base: dict, override: dict | None) -> dict:
    """Mescla recursivamente `override` sobre `base` (sem alterar os originais)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(config_path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return deep_merge(cfg, overrides)


def resolve_path(p: str | Path) -> Path:
    """Caminhos relativos são relativos a genAI/."""
    p = Path(p)
    return p if p.is_absolute() else (GENAI_ROOT / p).resolve()


def get_logger(name: str = _LOGGER_NAME, log_file: str | Path | None = None) -> logging.Logger:
    """Logger do pacote. Se o robô já configurou logging, os registros propagam para ele também."""
    root = logging.getLogger(_LOGGER_NAME)
    if not getattr(root, "_citizen_rag_configured", False):
        root.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        if log_file:
            try:
                lf = resolve_path(log_file)
                lf.parent.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(lf, encoding="utf-8")
                fh.setFormatter(fmt)
                root.addHandler(fh)
            except OSError:
                pass
        if not logging.getLogger().handlers:   # sem logging do robô -> stderr só para WARNING+
            sh = logging.StreamHandler()
            sh.setLevel(logging.WARNING)
            sh.setFormatter(fmt)
            root.addHandler(sh)
        root._citizen_rag_configured = True  # type: ignore[attr-defined]
    return logging.getLogger(name if name.startswith(_LOGGER_NAME) else f"{_LOGGER_NAME}.{name}")


def gemini_api_key() -> str | None:
    load_env()
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return key.strip() if key and key.strip() else None
