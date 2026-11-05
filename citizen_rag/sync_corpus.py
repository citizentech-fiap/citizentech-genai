"""Atualiza o snapshot do corpus do assistente a partir da saída do robô de RPA.

O repositório genAI roda sozinho com o snapshot em data/corpus/. Quando o robô coleta uma versão
nova das Cartas de Serviço, este script copia o JSONL e regenera data/corpus/FONTES.md
(fonte, URL e data de consulta de cada carta — exigido no entregável "Corpus e configuração").

    python -m citizen_rag.sync_corpus                       # usa ../RPA/output/corpus/servicos.jsonl
    python -m citizen_rag.sync_corpus --origem caminho.jsonl
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .config import GENAI_ROOT, load_config, resolve_path

ORIGEM_PADRAO = GENAI_ROOT.parent / "RPA" / "output" / "corpus" / "servicos.jsonl"


def escrever_fontes(corpus: Path) -> Path:
    linhas = ["# Fontes do corpus", "",
              "Cartas de Serviço oficiais coletadas pelo robô CitizenTech (repositório de RPA) "
              "na API pública do Portal de Serviços do gov.br, com fallback para a página HTML.", "",
              "| Serviço | Órgão | URL | Consultada em | Origem |", "|---|---|---|---|---|"]
    for raw in corpus.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        r = json.loads(raw)
        linhas.append(f"| {r['titulo']} | {r.get('orgao', '')} | {r['url']} | "
                      f"{str(r.get('data_coleta', ''))[:10]} | {r.get('fonte', '')} |")
    out = corpus.parent / "FONTES.md"
    out.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--origem", type=Path, default=ORIGEM_PADRAO)
    args = ap.parse_args(argv)
    destino = resolve_path(load_config()["paths"]["corpus"])
    if not args.origem.exists():
        raise SystemExit(f"Corpus de origem não encontrado: {args.origem}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.origem, destino)
    fontes = escrever_fontes(destino)
    print(f"Snapshot atualizado: {destino}\nFontes: {fontes}")
    print("Reconstrua os índices: python -m citizen_rag.indexing")


if __name__ == "__main__":
    main()
