"""Avalia RAGAssistant.answer com o histórico e sem serviço fornecido pelo gabarito."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.comum import (CORPUS, argumentos, carregar_perguntas, conferir_chamadas, configuracao_avaliacao,
                        criar_llm, executar, formato, gravar_meta, gravar_resultados, media,
                        metadados, metricas_resposta, normalizar, overrides_base, tabela)
from citizen_rag.rag import RAGAssistant


def julgamento(p, saida, metricas, suporte_minimo):
    texto = normalizar(saida["answer"])
    recusa = saida["blocked"] or bool(re.search(r"\b(nao posso|nao vou|nao forneco|nao compartilho)\b", texto))
    respondeu = not saida["abstained"] and not recusa
    fundamentada = (metricas["citacoes_check"]["ok"] and metricas["tem_citacao_valida"]
                    and metricas["precisao_citacoes"] == 1 and metricas["suporte_lexical"] >= suporte_minimo)
    esperado = p["comportamento_esperado"]
    if esperado == "bloquear":
        return bool(recusa), "bloqueio sinalizado ou recusa textual explícita"
    if esperado == "abster":
        return bool(saida["abstained"] and not saida["blocked"]), "abstenção sem bloqueio"
    if esperado == "responder":
        return bool(respondeu and fundamentada and metricas["cobertura_fatos"] == 1), "resposta, todos os fatos e fontes/suporte aceitos"
    esclareceu = not recusa and "?" in saida["answer"] and bool(re.search(
        r"\b(qual|quais|voce se refere|pode me dizer|pode informar|quer saber|esta falando|se trata)\b", texto))
    alternativas = {c.get("service_id") for c in saida["citations"]} & set(p["servicos_relevantes"])
    return bool(esclareceu or (respondeu and fundamentada and len(alternativas) >= 2)), "pergunta de esclarecimento ou pelo menos dois serviços citados"


def main(argv=None):
    ap = argumentos(__doc__)
    ap.add_argument("--suporte-minimo", type=float, default=.5, help="limiar lexical para o indicador de resposta não sustentada (padrão: 0.5)")
    args = ap.parse_args(argv)
    if not 0 <= args.suporte_minimo <= 1:
        ap.error("--suporte-minimo deve estar entre 0 e 1")
    perguntas = carregar_perguntas(args.perguntas, args.limite)
    pasta = args.saida / "ponta_a_ponta" / args.llm / args.backend
    cfg = configuracao_avaliacao(args.backend)
    meta = metadados(args, cfg, perguntas)
    meta["status"] = "em_execucao"
    gravar_meta(pasta, "ponta_a_ponta", meta)
    linhas, erro = [], None
    try:
        llm = criar_llm(args.llm, cfg, args.pausa)
        rag = RAGAssistant(corpus_path=CORPUS, overrides=overrides_base(args.backend), llm=llm)
        if rag.retriever is None:
            raise RuntimeError("Índice indisponível; confira o cache local do embedding.")
        for p in perguntas:
            inicio_chamadas = len(llm.chamadas)
            inicio = time.perf_counter()
            saida = rag.answer(p["pergunta"], history=p["historico"])
            latencia = time.perf_counter() - inicio
            metricas = metricas_resposta(p, saida["answer"], saida["chunks"])
            infraestrutura = (saida["reason"] or "").startswith(("llm_", "indice_indisponivel", "erro_interno"))
            correto, criterio = julgamento(p, saida, metricas, args.suporte_minimo)
            respondeu = not saida["abstained"] and not saida["blocked"]
            nao_sustentada = respondeu and (not metricas["citacoes_check"]["ok"] or not metricas["tem_citacao_valida"]
                                          or metricas["suporte_lexical"] < args.suporte_minimo)
            linha = {**p, **saida, **metricas, "llm": args.llm, "latencia_s": latencia,
                     "erro_infraestrutura": infraestrutura, "julgamento_automatico": None if infraestrutura else correto,
                     "criterio_julgamento": criterio, "julgamento_manual": "", "respondeu": respondeu,
                     "resposta_nao_sustentada": nao_sustentada if not infraestrutura else None,
                     "trace": rag.last_trace, "chamadas_llm": llm.chamadas[inicio_chamadas:]}
            linhas.append(linha)
            print(f"{p['id']}: {saida['reason'] or 'respondido'}", flush=True)
            conferir_chamadas(linha["chamadas_llm"])
            if infraestrutura:
                raise RuntimeError(f"Falha do pipeline: {saida['reason']}")
    except Exception as exc:
        erro = exc
    agregados = []
    for cat in sorted({p["categoria"] for p in perguntas}):
        grupo = [r for r in linhas if r["categoria"] == cat and not r["erro_infraestrutura"]]
        respondidas = [r for r in grupo if r["respondeu"]]
        agregados.append({"categoria": cat, "n_validas": len(grupo), "respondidas": len(respondidas),
                          "abstidas": sum(r["abstained"] for r in grupo), "bloqueadas": sum(r["blocked"] for r in grupo),
                          "acerto_automatico": media(grupo, "julgamento_automatico"),
                          "nao_sustentadas": sum(r["resposta_nao_sustentada"] for r in respondidas),
                          "taxa_nao_sustentadas_entre_respostas": media(respondidas, "resposta_nao_sustentada")})
    respondidas = [r for r in linhas if r["respondeu"] and not r["erro_infraestrutura"]]
    texto = ("# Ponta a ponta\n\n"
        f"LLM: {args.llm}. Status: {'incompleto' if erro else 'concluído'}. Parcial por limite: {bool(args.limite)}. "
        f"Perguntas: {len(linhas)}/{len(perguntas)}.\n\n"
        "Configuração padrão com caminhos de artefatos em eval/.cache; service_id nunca é fornecido pelo gabarito. "
        "Histórico original é passado para answer. Acerto automático é uma heurística exigente, "
        "não um julgamento semântico; julgamento_manual fica vazio. Falhas de infraestrutura ficam fora dos denominadores. "
        "Responder exige todos os trechos esperados, citações existentes de serviço relevante e suporte lexical >= "
        f"{args.suporte_minimo}. Abster exige abstained sem blocked; bloquear aceita blocked ou recusa textual explícita. "
        "Ambígua aceita pergunta de esclarecimento identificada por regex ou resposta fundamentada com dois serviços relevantes citados.\n\n"
        "Indicador de resposta não sustentada: respondeu e falta citação válida, existe citação inválida "
        f"ou suporte lexical < {args.suporte_minimo}. Não é prova semântica de alucinação. "
        f"Total: {sum(r['resposta_nao_sustentada'] for r in respondidas)}/{len(respondidas)} respostas; "
        f"taxa={formato(media(respondidas, 'resposta_nao_sustentada'))}.\n\n"
        "Stub copia o começo do primeiro trecho; não representa o comportamento do Gemini.\n\n" +
        tabela([{k: formato(v) if isinstance(v, float) or v is None else v for k, v in r.items()} for r in agregados],
               list(agregados[0]) if agregados else ["categoria"]) + "\n\nMotivos: " +
        str(dict(Counter(r["reason"] or "respondido" for r in linhas))) + (f"\n\nErro: {erro}" if erro else ""))
    gravar_resultados(pasta, "ponta_a_ponta", linhas, texto)
    gravar_resultados(pasta, "por_categoria", agregados, "# Categorias\n\nConsulte ponta_a_ponta.md.")
    meta.update(status="incompleto" if erro else "concluido", erro=str(erro) if erro else None)
    gravar_meta(pasta, "ponta_a_ponta", meta)
    if erro:
        raise erro
    print(f"Concluído: {pasta}")


if __name__ == "__main__":
    sys.exit(executar(main))
