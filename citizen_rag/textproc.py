"""Normalização e tokenização para o BM25 (português)."""
from __future__ import annotations

import re
import unicodedata

# Stopwords PT-BR enxutas (sem dependência do NLTK). Mantemos palavras de negação/condição
# ("nao", "sem") porque mudam o sentido de requisitos.
STOPWORDS = set("""
a o as os um uma uns umas de do da dos das no na nos nas em ao aos à às pelo pela pelos pelas por para
pra com e ou que se sua seu suas seus meu minha meus minhas eu voce voces ele ela eles elas isso isto
esse essa esses essas este esta estes estas aquele aquela qual quais como quando onde porque pois ja
mais muito muita quanto quanta tambem so ser estar ter foi sao era sera seria tem tenho ha la aqui entao me mim te
the of to and is are what how do i my
""".split())

_TOKEN = re.compile(r"[a-z0-9]+")


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def light_stem(tok: str) -> str:
    """Stemming leve: remove plurais simples (documentos->documento, certidoes->certidao,
    digitais->digital) e a vogal temática final (custa/custo/custos->cust, prazo->praz).
    Suficiente para BM25 em textos curtos, sem depender do NLTK/RSLP."""
    s = _plural(tok)
    if len(s) > 4 and s[-1] in "aoe":
        s = s[:-1]
    return s


def _plural(tok: str) -> str:
    if len(tok) <= 4:
        return tok
    if tok.endswith("oes") or tok.endswith("aes"):
        return tok[:-3] + "ao"
    if tok.endswith("ais") and len(tok) > 5:
        return tok[:-2] + "l"
    if tok.endswith("eis") and len(tok) > 5:
        return tok[:-2] + "l"
    if tok.endswith("ns"):
        return tok[:-2] + "m"
    if tok.endswith("res") or tok.endswith("zes"):
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str, stemming: str = "light", remove_stopwords: bool = True) -> list[str]:
    toks = _TOKEN.findall(strip_accents((text or "").lower()))
    if remove_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    if stemming == "light":
        toks = [light_stem(t) for t in toks]
    return toks
