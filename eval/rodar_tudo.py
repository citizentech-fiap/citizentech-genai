"""Executa os quatro avaliadores em sequência e reúne links e códigos de saída."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.comum import RAIZ, argumentos, executar, tabela


def main(argv=None):
    ap = argumentos(__doc__)
    ap.add_argument("--juiz", action="store_true", help="ativar juiz somente na geração")
    args = ap.parse_args(argv)
    args.saida = args.saida.resolve()
    args.perguntas = args.perguntas.resolve()
    args.saida.mkdir(parents=True, exist_ok=True)
    linhas, links = [], []
    scripts = ["avaliar_recuperacao", "avaliar_geracao", "avaliar_ponta_a_ponta", "avaliar_seguranca"]
    for script in scripts:
        comando = [sys.executable, str(RAIZ / "eval" / f"{script}.py"), "--backend", args.backend,
                   "--saida", str(args.saida), "--perguntas", str(args.perguntas)]
        if script != "avaliar_recuperacao":
            comando += ["--llm", args.llm, "--pausa", str(args.pausa)]
        if args.limite:
            comando += ["--limite", str(args.limite)]
        if script == "avaliar_geracao" and args.juiz:
            comando.append("--juiz")
        print(f"Executando {script}...", flush=True)
        inicio = time.perf_counter()
        try:
            resultado = subprocess.run(comando, cwd=RAIZ, check=False)
            codigo, erro = resultado.returncode, ""
        except OSError as exc:
            codigo, erro = 1, str(exc)
        linhas.append({"script": script, "codigo_saida": codigo,
                       "duracao_s": round(time.perf_counter() - inicio, 3), "erro": erro,
                       "comando": subprocess.list2cmdline(comando)})
        nome = script.removeprefix("avaliar_")
        relativo = ({"recuperacao": Path("recuperacao") / args.backend,
                     "geracao": Path("geracao") / args.llm,
                     "ponta_a_ponta": Path("ponta_a_ponta") / args.llm / args.backend,
                     "seguranca": Path("seguranca") / args.llm}[nome] / f"{nome}.md")
        if codigo == 0 and (args.saida / relativo).exists():
            links.append(f"- [{nome.replace('_', ' ')}]({relativo.as_posix()})")
        else:
            links.append(f"- {nome.replace('_', ' ')}: execução falhou (código {codigo}); conferir dados parciais e metadados.")
        texto = ("# Execuções das avaliações\n\n"
                 f"LLM solicitado: {args.llm}; embedding: {args.backend}; limite: {args.limite or 'sem limite'}. "
                 "Este é um índice técnico dos dados, não o relatório final de avaliação. "
                 "Cada resumo documenta métricas, denominadores e limitações.\n\n" + tabela(linhas,
                 ["script", "codigo_saida", "duracao_s", "erro", "comando"]) + "\n\n" + "\n".join(links) + "\n")
        (args.saida / "RESUMO.md").write_text(texto, encoding="utf-8")
    return int(any(r["codigo_saida"] != 0 for r in linhas))


if __name__ == "__main__":
    sys.exit(executar(main))
