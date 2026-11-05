"""Carga, validação, métricas auxiliares e persistência das avaliações (PT-BR)."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import csv
import os
from pathlib import Path
import re
import sys
import time
import unicodedata

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
# Os modelos necessários estão no cache local; não tentar baixar silenciosamente.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from citizen_rag import config as configuracao
from citizen_rag.config import deep_merge, load_config
from citizen_rag import guardrails as G
from citizen_rag.llm import BaseLLM, CachedLLM, StubLLM, make_llm

# A avaliação nunca abre .env, nem indiretamente pelo cliente do pacote.
# Gemini usa apenas GEMINI_API_KEY/GOOGLE_API_KEY já exportada no processo.
configuracao._env_loaded = True
CORPUS = RAIZ / "data/corpus/servicos.jsonl"
PERGUNTAS = RAIZ / "eval/perguntas.jsonl"
RESULTADOS = RAIZ / "eval/resultados"
CATEGORIAS = {"respondivel", "sem_resposta", "ambigua", "fora_escopo", "multi_turno", "adversarial"}
COMPORTAMENTOS = {"responder", "abster", "bloquear", "esclarecer_ou_responder"}


def normalizar(texto):
    """Ignora acentos/caixa e compacta espaços; preserva números e pontuação."""
    texto = unicodedata.normalize("NFKD", texto or "").casefold()
    return " ".join("".join(c for c in texto if not unicodedata.combining(c)).split())


def ler_jsonl(caminho):
    with open(caminho, encoding="utf-8-sig") as arquivo:
        return [json.loads(linha) for linha in arquivo if linha.strip()]


def carregar_perguntas(caminho=PERGUNTAS, limite=None):
    """Valida todas as anotações contra as seções do corpus antes de recortar."""
    perguntas = ler_jsonl(caminho)
    if not perguntas:
        raise ValueError("O conjunto de perguntas está vazio.")
    corpus = {d["service_id"]: d for d in ler_jsonl(CORPUS)}
    ids = set()
    campos = {"id", "categoria", "pergunta", "historico", "servicos_relevantes",
              "secoes_relevantes", "fatos_esperados", "comportamento_esperado", "observacao"}
    for p in perguntas:
        if not campos <= p.keys() or p["id"] in ids:
            raise ValueError("Pergunta incompleta ou identificador duplicado.")
        ids.add(p["id"])
        if p["categoria"] not in CATEGORIAS or p["comportamento_esperado"] not in COMPORTAMENTOS:
            raise ValueError(f"Categoria/comportamento inválido: {p['id']}")
        if not isinstance(p["pergunta"], str) or not p["pergunta"].strip() or not p["observacao"].strip():
            raise ValueError(f"Pergunta/observação vazia: {p['id']}")
        for campo in ("servicos_relevantes", "secoes_relevantes", "fatos_esperados"):
            if not isinstance(p[campo], list) or any(not isinstance(v, str) for v in p[campo]):
                raise ValueError(f"Lista inválida: {p['id']} / {campo}")
            if len(p[campo]) != len(set(p[campo])):
                raise ValueError(f"Anotação duplicada: {p['id']} / {campo}")
        if not isinstance(p["historico"], list) or any(
            h.get("role") not in {"user", "assistant"} or not isinstance(h.get("content"), str)
            for h in p["historico"]
        ):
            raise ValueError(f"Histórico inválido: {p['id']}")
        if p["categoria"] == "multi_turno" and not p["historico"]:
            raise ValueError(f"Multi-turno sem histórico: {p['id']}")
        if p["categoria"] == "ambigua" and len(p["servicos_relevantes"]) < 2:
            raise ValueError(f"Ambiguidade sem alternativas: {p['id']}")
        if len(p["fatos_esperados"]) > 3 or (p["comportamento_esperado"] == "responder" and not p["fatos_esperados"]):
            raise ValueError(f"Quantidade de fatos inválida: {p['id']}")
        textos = []
        for sid in p["servicos_relevantes"]:
            if sid not in corpus:
                raise ValueError(f"Serviço inexistente: {p['id']} / {sid}")
            for secao in p["secoes_relevantes"]:
                if secao not in corpus[sid]["secoes"]:
                    raise ValueError(f"Seção inexistente: {p['id']} / {secao}")
                textos.append(corpus[sid]["secoes"][secao])
        for fato in p["fatos_esperados"]:
            if not fato or not any(fato in texto for texto in textos):
                raise ValueError(f"Fato não literal nas seções anotadas: {p['id']} / {fato}")
        if p["categoria"] == "sem_resposta" and p["fatos_esperados"]:
            raise ValueError(f"Sem resposta não pode ter fato positivo: {p['id']}")
        if p["categoria"] in {"respondivel", "sem_resposta", "multi_turno", "ambigua"} and (
            not p["servicos_relevantes"] or not p["secoes_relevantes"]
        ):
            raise ValueError(f"Categoria exige serviços e seções anotados: {p['id']}")
    return perguntas[:limite] if limite is not None else perguntas


def inteiro_positivo(valor):
    n = int(valor)
    if n <= 0:
        raise argparse.ArgumentTypeError("Informe um inteiro maior que zero.")
    return n


def nao_negativo(valor):
    n = float(valor)
    if not 0 <= n < float("inf"):
        raise argparse.ArgumentTypeError("Informe um número finito não negativo.")
    return n


def argumentos(descricao, com_llm=True):
    ap = argparse.ArgumentParser(description=descricao)
    ap.add_argument("--perguntas", type=Path, default=PERGUNTAS)
    ap.add_argument("--saida", type=Path, default=RESULTADOS)
    ap.add_argument("--limite", type=inteiro_positivo, help="primeiras N perguntas; resumo marca execução parcial")
    ap.add_argument("--backend", choices=["sentence_transformers", "hashing"], default="sentence_transformers",
                    help="hashing é baseline distinto; não substitui e5 silenciosamente")
    if com_llm:
        ap.add_argument("--llm", choices=["gemini", "stub"], default="gemini")
        ap.add_argument("--pausa", type=nao_negativo, default=4.0,
                        help="segundos entre chamadas reais, inclusive ao juiz (padrão: 4)")
    return ap


def overrides_base(backend="sentence_transformers"):
    return {"embedding": {"backend": backend},
            "paths": {"index_dir": "eval/.cache/indexes", "llm_cache_dir": "eval/.cache/llm",
                      "log_file": "eval/.cache/avaliacao.log"}, "llm": {"cache": True}}


def configuracao_avaliacao(backend="sentence_transformers", adicionais=None):
    return load_config(overrides=deep_merge(overrides_base(backend), adicionais or {}))


class LLMRegistrado(BaseLLM):
    """Captura somente prompts/respostas de avaliação; nunca lê credenciais."""
    def __init__(self, interno):
        self.interno = interno
        self.available = interno.available
        self.model = interno.model
        self.chamadas = []

    def complete(self, system, user, **params):
        resposta = self.interno.complete(system, user, **params)
        self.chamadas.append({"system": system, "user": user, "params": params,
                              "resposta": asdict(resposta)})
        return resposta


class LLMComPausa(BaseLLM):
    def __init__(self, interno, pausa):
        self.interno, self.pausa = interno, pausa
        self.available, self.model = interno.available, interno.model
        self.ultima = None

    def complete(self, system, user, **params):
        if self.ultima is not None:
            time.sleep(max(0, self.pausa - (time.monotonic() - self.ultima)))
        resposta = self.interno.complete(system, user, **params)
        self.ultima = time.monotonic()
        return resposta


def criar_llm(nome, cfg, pausa=4):
    if nome == "stub":
        return LLMRegistrado(StubLLM())
    llm = make_llm(cfg)
    if not llm.available:
        raise RuntimeError("Gemini indisponível. Exporte GEMINI_API_KEY no processo ou use --llm stub; arquivos .env não são lidos.")
    if isinstance(llm, CachedLLM):
        llm.inner = LLMComPausa(llm.inner, pausa)
    else:
        llm = LLMComPausa(llm, pausa)
    return LLMRegistrado(llm)


def conferir_chamadas(chamadas):
    falhas = [c["resposta"].get("error") for c in chamadas if not c["resposta"]["ok"]]
    if falhas:
        raise RuntimeError(f"LLM falhou ({falhas[0]}). Dados parciais foram preservados; confira chave/rede/cota ou use --llm stub.")


def cobertura(resposta, fatos):
    presentes = [f for f in fatos if normalizar(f) in normalizar(resposta)]
    return (len(presentes) / len(fatos) if fatos else None), presentes


def metricas_resposta(p, resposta, chunks):
    cit = G.verify_citations(resposta, len(chunks))
    servicos = set(p["servicos_relevantes"])
    metas = [c["metadata"] if isinstance(c, dict) else c.metadata for c in chunks]
    precisao = (sum(metas[n - 1].get("service_id") in servicos for n in cit.valid) / len(cit.valid)
                if cit.valid else None)
    cob, fatos = cobertura(resposta, p["fatos_esperados"])
    textos = [c["text"] if isinstance(c, dict) else c.text for c in chunks]
    return {"cobertura_fatos": cob, "fatos_presentes": fatos, "citacoes_check": asdict(cit),
            "tem_citacao_valida": bool(cit.valid), "precisao_citacoes": precisao,
            "abstencao_textual": G.is_abstention(resposta),
            "abstencao_correta": G.is_abstention(resposta) if p["categoria"] == "sem_resposta" else None,
            "suporte_lexical": G.lexical_support(resposta, textos)}


def media(linhas, chave):
    valores = [r[chave] for r in linhas if r.get(chave) is not None]
    return sum(valores) / len(valores) if valores else None


def formato(valor):
    return "n/a" if valor is None else f"{valor:.4f}"


def tabela(linhas, colunas):
    saida = ["| " + " | ".join(colunas) + " |", "| " + " | ".join("---" for _ in colunas) + " |"]
    for linha in linhas:
        saida.append("| " + " | ".join(str(linha.get(c, "")).replace("|", "\\|").replace("\n", " ")
                                       for c in colunas) + " |")
    return "\n".join(saida)


def serializar(valor):
    if is_dataclass(valor):
        return asdict(valor)
    if isinstance(valor, Path):
        return str(valor)
    if hasattr(valor, "item"):
        return valor.item()
    raise TypeError(type(valor).__name__)


def gravar_resultados(pasta, nome, linhas, resumo):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    with open(pasta / f"{nome}.jsonl", "w", encoding="utf-8") as arq:
        for linha in linhas:
            arq.write(json.dumps(linha, ensure_ascii=False, default=serializar, allow_nan=False) + "\n")
    campos = list(dict.fromkeys(k for r in linhas for k in r))
    with open(pasta / f"{nome}.csv", "w", encoding="utf-8-sig", newline="") as arq:
        csvw = csv.DictWriter(arq, fieldnames=campos)
        csvw.writeheader()
        for r in linhas:
            csvw.writerow({k: json.dumps(v, ensure_ascii=False, default=serializar) if isinstance(v, (dict, list)) else v
                           for k, v in r.items()})
    (pasta / f"{nome}.md").write_text(resumo.rstrip() + "\n", encoding="utf-8")


def metadados(args, cfg, perguntas):
    versoes = {}
    for pacote in ("numpy", "sentence-transformers", "torch", "rank-bm25", "google-genai"):
        try:
            versoes[pacote] = importlib.metadata.version(pacote)
        except importlib.metadata.PackageNotFoundError:
            versoes[pacote] = "ausente"
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "versoes": versoes, "argumentos": vars(args), "configuracao": cfg,
            "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
            "perguntas_sha256": hashlib.sha256(Path(args.perguntas).read_bytes()).hexdigest(),
            "n_perguntas": len(perguntas), "categorias": dict(Counter(p["categoria"] for p in perguntas)),
            "parcial": args.limite is not None, "modelos_offline": os.environ.get("HF_HUB_OFFLINE")}


def gravar_meta(pasta, nome, meta):
    Path(pasta).mkdir(parents=True, exist_ok=True)
    (Path(pasta) / f"{nome}_metadados.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=serializar, allow_nan=False), encoding="utf-8")


def executar(main):
    """CLI sem stack trace; erros não viram amostras aprovadas."""
    try:
        return main() or 0
    except KeyboardInterrupt:
        print("Avaliação interrompida pelo usuário.", file=sys.stderr)
        return 130
    except Exception as erro:
        print(f"Não foi possível concluir a avaliação: {type(erro).__name__}: {erro}", file=sys.stderr)
        return 1
