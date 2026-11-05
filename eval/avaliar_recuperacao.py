"""Compara chunking × retriever, reranker e calibra o gate, sem LLM."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.comum import (CORPUS, argumentos, carregar_perguntas, configuracao_avaliacao,
                        executar, formato, gravar_meta, gravar_resultados, media, metadados, tabela)
from citizen_rag.indexing import GRID, build_index
from citizen_rag.rag import condense_query, normalize_history
from citizen_rag.retriever import Retriever
import numpy as np

METRICAS = ["hit_1", "recall_servico_5", "recall_secao_5", "mrr", "ndcg_5", "evidencia_recuperada"]


def metricas(p, chunks):
    """Recall usa conjuntos de serviços/pares, sem premiar chunks duplicados."""
    relevantes = set(p["servicos_relevantes"])
    pares = {(s, sec) for s in relevantes for sec in p["secoes_relevantes"]}
    ids = [c.metadata.get("service_id") for c in chunks[:5]]
    recuperados = {(c.metadata.get("service_id"), c.metadata.get("secao")) for c in chunks[:5]}
    rr = next((1 / pos for pos, sid in enumerate(ids, 1) if sid in relevantes), 0.0)
    vistos, dcg = set(), 0.0
    for pos, sid in enumerate(ids, 1):
        if sid in relevantes and sid not in vistos:
            dcg += 1 / math.log2(pos + 1)
            vistos.add(sid)
    ideal = sum(1 / math.log2(pos + 1) for pos in range(1, min(5, len(relevantes)) + 1))
    literais = [f for f in p["fatos_esperados"] if any(f in c.text for c in chunks[:5])]
    return {"hit_1": int(bool(ids) and ids[0] in relevantes) if relevantes else None,
            "recall_servico_5": len(relevantes & set(ids)) / len(relevantes) if relevantes else None,
            "recall_secao_5": len(pares & recuperados) / len(pares) if pares else None,
            "mrr": rr if relevantes else None, "ndcg_5": dcg / ideal if ideal else None,
            "evidencia_recuperada": bool(literais) if p["fatos_esperados"] else None,
            "fatos_literais_top5": literais}


def avaliar(cfg, indice, perguntas, nome, rerank=False):
    retriever = Retriever(indice, cfg)
    linhas = []
    for p in perguntas:
        consulta = condense_query(p["pergunta"], normalize_history(p["historico"]))
        inicio = time.perf_counter()
        chunks = retriever.retrieve(consulta, top_k=5, rerank=rerank)
        if rerank and chunks and not all(c.signals.get("rerank") is not None for c in chunks):
            raise RuntimeError("Reranker não executou; comparação não pode usar silenciosamente a ordem original.")
        linha = {**p, "configuracao": nome, "chunking": cfg["chunking"], "modo": cfg["retriever"]["mode"],
                 "reranker": rerank, "consulta": consulta, "n_chunks_indice": len(indice.chunks),
                 "latencia_s": time.perf_counter() - inicio, "chunks": [c.to_dict() for c in chunks],
                 **metricas(p, chunks), "julgamento_manual": ""}
        for sinal in ("dense", "bm25", "rerank"):
            valores = [c.signals[sinal] for c in chunks if c.signals.get(sinal) is not None]
            linha[f"melhor_{sinal}"] = max(valores) if valores else None
        linhas.append(linha)
    return linhas


def agregar(linhas, **grupo):
    return {**grupo, "n": len(linhas), "n_com_servico": sum(bool(r["servicos_relevantes"]) for r in linhas),
            "n_com_fatos": sum(bool(r["fatos_esperados"]) for r in linhas),
            **{m: media(linhas, m) for m in METRICAS}, "latencia_media_s": media(linhas, "latencia_s")}


def ordem(r):
    return tuple(r[m] or 0 for m in ("recall_servico_5", "ndcg_5", "recall_secao_5", "mrr", "hit_1"))


def calibrar(linhas, nome):
    """Score máximo entre os cinco chunks entregues, exatamente o sinal usado pelo gate."""
    distribuicoes, limiares, sugestoes = [], [], []
    for sinal in ("dense", "rerank"):
        campo = f"melhor_{sinal}"
        grupos = {cat: [r for r in linhas if r["categoria"] == cat]
                  for cat in ("respondivel", "fora_escopo", "sem_resposta")}
        for cat, amostras in grupos.items():
            valores = [r[campo] for r in amostras if r[campo] is not None]
            quantis = np.quantile(valores, [0, .25, .5, .75, 1]).tolist() if valores else [None] * 5
            distribuicoes.append({"configuracao": nome, "sinal": sinal, "categoria": cat,
                                  "n": len(amostras), "n_score": len(valores),
                                  **dict(zip(["min", "p25", "mediana", "p75", "max"], quantis))})
        valores = sorted({r[campo] for amostras in grupos.values() for r in amostras if r[campo] is not None})
        if not valores:
            continue
        candidatos = {valores[0] - 1e-6, valores[-1] + 1e-6,
                      *valores, *((a + b) / 2 for a, b in zip(valores, valores[1:]))}
        candidatos.add(.85 if sinal == "dense" else -1.5)
        atuais = []
        for limiar in sorted(candidatos):
            def passa(r):
                return r[campo] is not None and r[campo] >= limiar
            bloqueadas = sum(not passa(r) for r in grupos["respondivel"])
            fora = sum(passa(r) for r in grupos["fora_escopo"])
            sem = sum(passa(r) for r in grupos["sem_resposta"])
            nr, nf = len(grupos["respondivel"]), len(grupos["fora_escopo"])
            custo = (bloqueadas / nr + fora / nf) / 2 if nr and nf else None
            atuais.append({"configuracao": nome, "sinal": sinal, "limiar": limiar,
                           "n_respondivel": nr, "respondiveis_bloqueadas": bloqueadas,
                           "n_fora_escopo": nf, "fora_escopo_passariam": fora,
                           "n_sem_resposta": len(grupos["sem_resposta"]), "sem_resposta_passariam": sem,
                           "erro_balanceado": custo})
        limiares.extend(atuais)
        elegiveis = [r for r in atuais if r["erro_balanceado"] is not None]
        if elegiveis:
            sugestoes.append(min(elegiveis, key=lambda r: (r["erro_balanceado"], r["respondiveis_bloqueadas"], r["limiar"])))
    return distribuicoes, limiares, sugestoes


def main(argv=None):
    args = argumentos(__doc__, com_llm=False).parse_args(argv)
    perguntas = carregar_perguntas(args.perguntas, args.limite)
    pasta = args.saida / "recuperacao" / args.backend
    base = configuracao_avaliacao(args.backend)
    meta = metadados(args, base, perguntas)
    meta["status"] = "em_execucao"
    gravar_meta(pasta, "recuperacao", meta)
    linhas, resumo, configs, indices = [], [], {}, {}
    try:
        for chunking in GRID:
            cfg = configuracao_avaliacao(args.backend, {"chunking": chunking, "retriever": {"reranker": {"enabled": False}}})
            indice = build_index(cfg, CORPUS)
            if not indice.has_dense:
                raise RuntimeError("Índice denso ausente; interrompendo para não rotular BM25 como dense/hybrid.")
            for modo in ("bm25", "dense", "hybrid"):
                nome = f"{chunking['strategy']}_s{chunking['size']}_o{chunking['overlap']}_{modo}"
                cfg_modo = configuracao_avaliacao(args.backend, {"chunking": chunking,
                    "retriever": {"mode": modo, "reranker": {"enabled": False}}})
                amostras = avaliar(cfg_modo, indice, perguntas, nome)
                linhas.extend(amostras)
                resumo.append(agregar(amostras, configuracao=nome))
                configs[nome], indices[nome] = cfg_modo, indice
                print(f"{nome}: Recall@5={resumo[-1]['recall_servico_5']:.4f}", flush=True)
        melhor = max(resumo, key=ordem)["configuracao"]
        cfg = configuracao_avaliacao(args.backend, {"chunking": configs[melhor]["chunking"],
            "retriever": {"mode": configs[melhor]["retriever"]["mode"], "reranker": {"enabled": True}}})
        nome_rr = melhor + "_rerank"
        amostras = avaliar(cfg, indices[melhor], perguntas, nome_rr, rerank=True)
        linhas.extend(amostras)
        resumo.append(agregar(amostras, configuracao=nome_rr))
        configs[nome_rr] = cfg
        meta.update(status="concluido", configuracoes=configs, melhor_grade=melhor,
                    melhor_incluindo_rerank=max(resumo, key=ordem)["configuracao"])
    except Exception as erro:
        meta.update(status="incompleto", erro=f"{type(erro).__name__}: {erro}")
        gravar_resultados(pasta, "recuperacao", linhas, "# Recuperação incompleta\n\n" + meta["erro"])
        gravar_meta(pasta, "recuperacao", meta)
        raise
    distribuicoes, limiares, sugestoes = [], [], []
    for nome in (melhor, nome_rr):
        dist, lim, sug = calibrar([r for r in linhas if r["configuracao"] == nome], nome)
        distribuicoes.extend(dist)
        limiares.extend(lim)
        sugestoes.extend(sug)
    por_servico, por_categoria = [], []
    for nome in configs:
        grupo = [r for r in linhas if r["configuracao"] == nome]
        for sid in sorted({s for p in perguntas for s in p["servicos_relevantes"]}):
            # Métricas da consulta para o serviço específico (as ambíguas têm alvos múltiplos).
            amostras_sid = []
            for r in grupo:
                if sid not in r["servicos_relevantes"]:
                    continue
                from citizen_rag.retriever import RetrievedChunk
                chunks = [RetrievedChunk(id=c["id"], text=c["text"], score=c["score"], metadata=c["metadata"])
                          for c in r["chunks"]]
                amostras_sid.append({**r, **metricas({**r, "servicos_relevantes": [sid]}, chunks)})
            por_servico.append(agregar(amostras_sid, configuracao=nome, servico=sid))
        for cat in sorted({p["categoria"] for p in perguntas}):
            por_categoria.append(agregar([r for r in grupo if r["categoria"] == cat], configuracao=nome, categoria=cat))
    texto = ("# Recuperação\n\n" + ("Execução parcial.\n\n" if args.limite else "") +
        "Sem LLM. Grade de 15 configurações e uma repetição da melhor com reranker. "
        "Modelos locais; sem fallback silencioso. Busca sem filtro de serviço; multi-turno usa condense_query.\n\n"
        "Seleção lexicográfica: Recall@5 serviço, nDCG@5, Recall@5 seção, MRR, Hit@1; empates mantêm a ordem da grade. "
        "Médias macro das perguntas com alvos. Recall serviço = serviços distintos encontrados / alvos; "
        "Recall seção = pares (serviço, seção) encontrados / produto dos alvos anotados. "
        "Seção geral do fixed não é equiparada a uma seção conhecida. Hit@1/MRR consideram serviço; "
        "MRR é truncado no top-5 (ausência vale zero). "
        "nDCG usa ganho 1 na primeira ocorrência de cada serviço relevante nas posições dos chunks, "
        "ganho 0 para repetições; ideal com serviços distintos nas primeiras posições. "
        "Evidência = pelo menos um fato como substring literal em um chunk (sem normalização); "
        "sem fatos é n/a. Sem resposta mede localização do serviço, não respondibilidade.\n\n" +
        tabela([{**r, **{m: formato(r[m]) for m in METRICAS}} for r in resumo],
               ["configuracao", "n_com_servico", *METRICAS]) + "\n\n## Calibração exploratória\n\n"
        "Máximo do sinal no top-5 efetivamente entregue; aprovação se score >= limiar. "
        "Distribuições separadas para respondível, fora de escopo e sem resposta. Limiares incluem scores, "
        "pontos médios, extremos e o padrão do config. A sugestão minimiza a média das taxas de "
        "respondíveis bloqueadas e fora de escopo aprovadas; desempate preserva respondíveis. "
        "Amostra pequena e usada na seleção e calibração: não é validação independente. "
        "Sem resposta é mostrado à parte: similaridade temática não demonstra resposta disponível.\n\n" +
        tabela(sugestoes, ["configuracao", "sinal", "limiar", "respondiveis_bloqueadas", "n_respondivel",
                           "fora_escopo_passariam", "n_fora_escopo", "sem_resposta_passariam", "erro_balanceado"]))
    gravar_resultados(pasta, "recuperacao", linhas, texto)
    for nome, dados in (("configuracoes", resumo), ("por_servico", por_servico), ("por_categoria", por_categoria),
                         ("distribuicoes_gate", distribuicoes), ("limiares_gate", limiares), ("sugestoes_gate", sugestoes)):
        gravar_resultados(pasta, nome, dados, f"# {nome.replace('_', ' ').capitalize()}\n\nConsulte recuperacao.md para fórmulas e denominadores.")
    gravar_meta(pasta, "recuperacao", meta)
    print(f"Concluído: {pasta}")


if __name__ == "__main__":
    sys.exit(executar(main))
