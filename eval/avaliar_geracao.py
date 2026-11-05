"""Compara prompts v1/v2 com seções controladas, sem usar o retriever."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.comum import (CORPUS, argumentos, carregar_perguntas, conferir_chamadas,
                        configuracao_avaliacao, criar_llm, executar, formato, gravar_meta,
                        gravar_resultados, media, metadados, metricas_resposta, tabela)
from citizen_rag.chunking import chunk_corpus
from citizen_rag.generator import Generator
from citizen_rag.ingest import load_corpus
from citizen_rag import guardrails as G


def contexto_controlado(p, chunks):
    selecionados = [c for c in chunks if c.metadata["service_id"] in p["servicos_relevantes"]
                    and c.metadata["secao"] in p["secoes_relevantes"]]
    if not selecionados:
        raise ValueError(f"Contexto controlado vazio: {p['id']}")
    # O controle de contexto não pode omitir o gabarito por corte de janela.
    for fato in p["fatos_esperados"]:
        if not any(" ".join(fato.split()) in " ".join(c.text.split()) for c in selecionados):
            raise ValueError(f"Fato não incluído no contexto controlado: {p['id']} / {fato}")
    return selecionados


def main(argv=None):
    ap = argumentos(__doc__)
    ap.add_argument("--juiz", action="store_true", help="chamada adicional ao juiz de fundamentação; também respeita pausa/cache")
    args = ap.parse_args(argv)
    todas = carregar_perguntas(args.perguntas)
    perguntas = [p for p in todas if p["categoria"] in {"respondivel", "sem_resposta"}]
    if args.limite:
        perguntas = perguntas[:args.limite]
    pasta = args.saida / "geracao" / args.llm
    cfg = configuracao_avaliacao(args.backend, {"chunking": {"strategy": "structural", "size": 160, "overlap": 0}})
    meta = metadados(args, cfg, perguntas)
    meta["status"] = "em_execucao"
    gravar_meta(pasta, "geracao", meta)
    linhas, erro = [], None
    try:
        llm = criar_llm(args.llm, cfg, args.pausa)
        gerador = Generator(cfg, llm)
        chunks = chunk_corpus(load_corpus(CORPUS), cfg)
        for p in perguntas:
            contexto = contexto_controlado(p, chunks)
            for versao in ("v1", "v2"):
                inicio = len(llm.chamadas)
                gen = gerador.generate(p["pergunta"], contexto, version=versao)
                linha = {**p, "llm": args.llm, "prompt_version": versao, "contexto": [c.to_dict() for c in contexto],
                         "criterio_contexto": "todas as janelas das seções anotadas; serviço conhecido pelo avaliador",
                         "geracao": asdict(gen), "resposta": gen.text, "ok": gen.ok,
                         "erro": gen.error, "latencia_s": gen.latency_s, "cached": gen.cached,
                         "julgamento_manual": ""}
                if gen.ok:
                    linha.update(metricas_resposta(p, gen.text, contexto))
                    if args.juiz:
                        juiz, info = G.judge_groundedness(llm, gen.text, G.judge_evidence(contexto))
                        linha.update(juiz_fundamentada=juiz, juiz_info=info)
                    linha["julgamento_automatico"] = (
                        linha["abstencao_textual"] and not linha["tem_citacao_valida"] if p["categoria"] == "sem_resposta"
                        else (not linha["abstencao_textual"] and linha["cobertura_fatos"] == 1
                              and linha["citacoes_check"]["ok"] and linha["tem_citacao_valida"]
                              and linha["precisao_citacoes"] == 1))
                else:
                    linha["julgamento_automatico"] = None
                linha["chamadas_llm"] = llm.chamadas[inicio:]
                linhas.append(linha)
                print(f"{p['id']} {versao}: ok={gen.ok}", flush=True)
                conferir_chamadas(linha["chamadas_llm"])
    except Exception as exc:
        erro = exc
    agregados = []
    for versao in ("v1", "v2"):
        for categoria in ("respondivel", "sem_resposta"):
            grupo = [r for r in linhas if r["prompt_version"] == versao and r["categoria"] == categoria and r["ok"]]
            agregados.append({"prompt": versao, "categoria": categoria, "n_validas": len(grupo),
                              **{k: media(grupo, k) for k in ("cobertura_fatos", "tem_citacao_valida", "precisao_citacoes",
                                    "abstencao_correta", "suporte_lexical", "julgamento_automatico", "juiz_fundamentada")}})
    texto = ("# Geração com contexto controlado\n\n"
        f"LLM: {args.llm}. Status: {'incompleto' if erro else 'concluído'}. Perguntas selecionadas: {len(perguntas)}. "
        f"Registros: {len(linhas)}. Parcial por limite: {bool(args.limite)}.\n\n"
        "Todos os chunks estruturais das seções anotadas, sem recuperação nem gate. "
        "Sem resposta recebe seções tematicamente relacionadas que não fornecem a informação solicitada; "
        "a ausência foi conferida no corpus e justificada em observacao. Não se injeta um fato fictício no gabarito.\n\n"
        "Cobertura é a fração de trechos esperados encontrados com acentos/caixa normalizados; sem fatos = n/a. "
        "Citação válida aponta para posição existente; precisão é a fração das citações válidas únicas para serviço relevante "
        "(neste contexto controlado, todos os chunks são relevantes por serviço: métrica limitada). "
        "Validade e coincidência lexical não provam sustentação semântica. Abstenção correta usa is_abstention; "
        "o julgamento também exige ausência de citação para sem resposta. Para respondível exige todos os fatos "
        "e citações válidas do serviço esperado. Juiz ausente/falho = n/a, nunca aprovação.\n\n"
        "Stub apenas reproduz até 40 palavras do primeiro documento; resultados demonstram execução e métricas, "
        "não qualidade do Gemini. Mesmos contextos e cliente/cache para v1 e v2.\n\n" +
        tabela([{k: formato(v) if isinstance(v, float) or v is None else v for k, v in r.items()} for r in agregados],
               list(agregados[0])) + (f"\n\nErro: {erro}" if erro else ""))
    gravar_resultados(pasta, "geracao", linhas, texto)
    gravar_resultados(pasta, "agregados", agregados, "# Agregados\n\nConsulte geracao.md.")
    meta.update(status="incompleto" if erro else "concluido", erro=str(erro) if erro else None)
    gravar_meta(pasta, "geracao", meta)
    if erro:
        raise erro
    print(f"Concluído: {pasta}")


if __name__ == "__main__":
    sys.exit(executar(main))
