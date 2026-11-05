"""Ingestão do corpus JSONL (um serviço por linha, schema em RPA/CONTRACTS.md) -> documentos com metadados."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import get_logger, resolve_path

log = get_logger("ingest")

# Ordem canônica das seções da Carta de Serviço e rótulo legível usado no texto dos chunks.
SECTION_LABELS: dict[str, str] = {
    "descricao": "Descrição",
    "requisitos": "Requisitos",
    "documentos": "Documentos necessários",
    "etapas": "Etapas",
    "canais": "Canais de atendimento",
    "prazos": "Prazos",
    "custos": "Custos",
}


@dataclass
class ServiceDoc:
    service_id: str
    titulo: str
    orgao: str
    url: str
    data_coleta: str
    secoes: dict[str, str]
    texto_completo: str
    content_hash: str = ""
    fonte: str = ""
    extra: dict = field(default_factory=dict)

    def base_metadata(self) -> dict:
        return {"service_id": self.service_id, "titulo": self.titulo, "orgao": self.orgao,
                "url": self.url, "data_coleta": self.data_coleta}


def load_corpus(path: str | Path) -> list[ServiceDoc]:
    """Lê o JSONL; linhas inválidas são registradas e ignoradas (o corpus vem de scraping)."""
    path = resolve_path(path)
    docs: list[ServiceDoc] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning("Linha %d do corpus inválida (%s) — ignorada", ln, e)
                continue
            sid = str(rec.get("service_id") or rec.get("id") or "").strip()
            if not sid:
                log.warning("Linha %d sem service_id — ignorada", ln)
                continue
            if sid in seen:
                log.warning("service_id duplicado '%s' na linha %d — mantida a primeira ocorrência", sid, ln)
                continue
            secoes_raw = rec.get("secoes") or {}
            secoes = {k: (str(v).strip() if v else "") for k, v in secoes_raw.items()}
            texto = str(rec.get("texto_completo") or "").strip()
            if not texto and not any(secoes.values()):
                log.warning("Serviço '%s' sem texto — ignorado", sid)
                continue
            docs.append(ServiceDoc(
                service_id=sid,
                titulo=str(rec.get("titulo") or sid),
                orgao=str(rec.get("orgao") or ""),
                url=str(rec.get("url") or ""),
                data_coleta=str(rec.get("data_coleta") or ""),
                secoes=secoes,
                texto_completo=texto,
                content_hash=str(rec.get("content_hash") or ""),
                fonte=str(rec.get("fonte") or ""),
            ))
            seen.add(sid)
    log.info("Corpus carregado: %d serviços de %s", len(docs), path)
    return docs


def corpus_fingerprint(path: str | Path) -> str:
    """Hash curto do arquivo do corpus — entra na chave do diretório de índice."""
    h = hashlib.sha256(resolve_path(path).read_bytes()).hexdigest()
    return h[:10]
