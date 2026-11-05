"""Estratégias de chunking das Cartas de Serviço.

- structural: um chunk por seção da carta (requisitos, documentos, etapas...). Seção longa é quebrada
  em fronteiras de frase, sem sobreposição. Preserva a unidade semântica que o cidadão pergunta
  ("quais documentos?") e permite citar a seção exata.
- fixed: janelas de N palavras com sobreposição sobre o texto completo, ignorando a estrutura.
  Baseline clássico; serve de comparação (mistura seções, perde o metadado `secao`).
- hybrid (padrão): structural + teto de tamanho + sobreposição alinhada a frases dentro da seção
  longa — mantém a fronteira de seção e evita perder contexto no corte.

Todo chunk recebe metadados (service_id, titulo, orgao, url, data_coleta, secao, chunk_index,
strategy) e um id estável `<service_id>:<secao>:<nn>` (determinístico para a mesma configuração).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .ingest import SECTION_LABELS, ServiceDoc

_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
_ENUM_ONLY = re.compile(r"^\(?\d{1,2}[.)]?$|^[-•*]$")


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "metadata": dict(self.metadata)}

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        return Chunk(id=d["id"], text=d["text"], metadata=dict(d.get("metadata") or {}))


def split_sentences(text: str) -> list[str]:
    """Divide em frases; junta marcadores de enumeração soltos ("2.") à frase seguinte."""
    parts = [p.strip() for p in _SENT_SPLIT.split(text or "") if p and p.strip()]
    out: list[str] = []
    carry = ""
    for p in parts:
        if _ENUM_ONLY.match(p):
            carry = (carry + " " + p).strip()
            continue
        out.append((carry + " " + p).strip() if carry else p)
        carry = ""
    if carry:
        out.append(carry)
    return out


def _word_windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words] if words else []
    stride = max(1, size - max(0, min(overlap, size - 1)))
    wins = []
    for start in range(0, len(words), stride):
        wins.append(words[start:start + size])
        if start + size >= len(words):
            break
    return wins


def pack_sentences(text: str, size: int, overlap: int = 0) -> list[str]:
    """Agrupa frases em blocos de até `size` palavras; a sobreposição repete as últimas frases
    do bloco anterior até somar >= `overlap` palavras. Frase maior que `size` é cortada por palavras."""
    sents: list[list[str]] = []
    for s in split_sentences(text):
        w = s.split()
        if len(w) > size:
            sents.extend(_word_windows(w, size, overlap))
        elif w:
            sents.append(w)
    blocks: list[list[list[str]]] = []
    cur: list[list[str]] = []
    cur_len = 0
    for s in sents:
        if cur and cur_len + len(s) > size:
            blocks.append(cur)
            tail: list[list[str]] = []
            if overlap > 0:
                acc = 0
                for prev in reversed(cur):
                    if acc >= overlap or acc + len(prev) + len(s) > size:
                        break
                    tail.insert(0, prev)
                    acc += len(prev)
            cur, cur_len = list(tail), sum(len(t) for t in tail)
        cur.append(s)
        cur_len += len(s)
    if cur:
        blocks.append(cur)
    return [" ".join(" ".join(s) for s in b) for b in blocks]


def _prefix(doc: ServiceDoc, secao: str | None, enabled: bool) -> str:
    if not enabled:
        return ""
    if secao and secao in SECTION_LABELS:
        return f"{doc.titulo} — {SECTION_LABELS[secao]}: "
    if secao and secao not in ("geral",):
        return f"{doc.titulo} — {secao.capitalize()}: "
    return f"{doc.titulo}: "


def _ordered_sections(doc: ServiceDoc) -> list[tuple[str, str]]:
    known = [(k, doc.secoes.get(k, "")) for k in SECTION_LABELS]
    extra = [(k, v) for k, v in doc.secoes.items() if k not in SECTION_LABELS]
    return [(k, v) for k, v in known + extra if v and v.strip()]


def chunk_document(doc: ServiceDoc, strategy: str = "hybrid", size: int = 160, overlap: int = 40,
                   min_size: int = 8, prefix_title: bool = True) -> list[Chunk]:
    strategy = strategy.lower()
    if strategy not in ("structural", "fixed", "hybrid"):
        raise ValueError(f"Estratégia de chunking desconhecida: {strategy}")
    pieces: list[tuple[str, str]] = []          # (secao, texto)
    sections = _ordered_sections(doc)
    if strategy == "fixed" or not sections:
        full = doc.texto_completo or "\n".join(v for _, v in sections)
        for w in _word_windows(full.split(), size, overlap):
            pieces.append(("geral", " ".join(w)))
    else:
        ov = overlap if strategy == "hybrid" else 0
        for secao, txt in sections:
            if len(txt.split()) <= size:
                pieces.append((secao, txt.strip()))
            else:
                pieces.extend((secao, p) for p in pack_sentences(txt, size, ov))

    chunks: list[Chunk] = []
    counters: dict[str, int] = {}
    base = doc.base_metadata()
    for secao, body in pieces:
        i = counters.get(secao, 0)
        # min_size só descarta fragmentos de corte (2º pedaço em diante); seção curta é informação
        # válida (ex.: custos = "O serviço é gratuito.") e é sempre mantida.
        if i > 0 and len(body.split()) < min_size:
            continue
        counters[secao] = i + 1
        meta = dict(base, secao=secao, chunk_index=i, strategy=strategy, n_words=len(body.split()))
        chunks.append(Chunk(id=f"{doc.service_id}:{secao}:{i:02d}",
                            text=_prefix(doc, secao, prefix_title) + body, metadata=meta))
    return chunks


def chunk_corpus(docs: list[ServiceDoc], cfg: dict) -> list[Chunk]:
    c = cfg.get("chunking", cfg)
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, strategy=c.get("strategy", "hybrid"), size=int(c.get("size", 160)),
                                  overlap=int(c.get("overlap", 40)), min_size=int(c.get("min_size", 8)),
                                  prefix_title=bool(c.get("prefix_title", True))))
    return out
