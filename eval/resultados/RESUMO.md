# Resultados da avaliação (05/11/2026)

A interpretação completa está em [`report/relatorio_avaliacao.md`](../../report/relatorio_avaliacao.md).
Esta pasta guarda os dados brutos de cada etapa, pergunta a pergunta.

| Etapa | Script | Resultados |
|---|---|---|
| Chunking e recuperação (16 configurações, sem LLM) e calibração do gate | `eval/avaliar_recuperacao.py` | [`recuperacao/sentence_transformers/`](recuperacao/sentence_transformers/recuperacao.md) |
| Geração com contexto controlado (prompts v1 × v2, juiz de fundamentação) | `eval/avaliar_geracao.py --juiz` | [`geracao/gemini/`](geracao/gemini/geracao.md) |
| Ponta a ponta (40 perguntas) | `eval/avaliar_ponta_a_ponta.py` | [`ponta_a_ponta/gemini/sentence_transformers/`](ponta_a_ponta/gemini/sentence_transformers/ponta_a_ponta.md) |
| Revisão humana das 40 respostas | revisão independente | [`ponta_a_ponta/revisao_manual.md`](ponta_a_ponta/revisao_manual.md) |
| Segurança (4 ataques × proteções ligadas/desligadas × 3 LLMs) | `eval/avaliar_seguranca.py` | [`seguranca/gemini/`](seguranca/gemini/seguranca.md) |

LLM: `gemini-3.5-flash-lite` (free tier). `verificacao_dados.json` confirma que os 40 fatos esperados aparecem
literalmente nas seções anotadas do corpus.
