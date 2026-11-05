# Como reproduzir os dados

Execute na raiz `genAI`, usando a venv do projeto. Nenhuma dependência foi acrescentada.
Os scripts também aceitam `python -m eval.avaliar_...` e exibem opções com `--help`.

```powershell
..\.venv\Scripts\python eval\avaliar_recuperacao.py
..\.venv\Scripts\python eval\avaliar_geracao.py --llm stub --pausa 0
..\.venv\Scripts\python eval\avaliar_ponta_a_ponta.py --llm stub --pausa 0
..\.venv\Scripts\python eval\avaliar_seguranca.py --llm stub --pausa 0
..\.venv\Scripts\python -m unittest discover -s tests -q
..\.venv\Scripts\python -m unittest discover -s eval -p test_avaliadores.py -q
```

Para reunir tudo, incluindo os resumos técnicos de cada script:

```powershell
..\.venv\Scripts\python eval\rodar_tudo.py --llm stub --pausa 0
# Depois, com GEMINI_API_KEY/GOOGLE_API_KEY já exportada no processo:
..\.venv\Scripts\python eval\rodar_tudo.py --llm gemini --pausa 5
..\.venv\Scripts\python eval\avaliar_geracao.py --llm gemini --juiz --pausa 5
```

Por regra do projeto, nenhum script abre `.env`, inclusive via `make_llm`. O carregamento implícito
de dotenv do pacote é desativado apenas no processo de avaliação. Gemini usa as variáveis já exportadas,
cache ligado e pausa entre chamadas reais, inclusive chamadas ao juiz; acertos de cache não aguardam.
Falhas de chave/rede/cota têm mensagem curta, código de saída diferente de zero e dados parciais
identificados. Não se trocam modelos ou backend silenciosamente.

Os modelos Hugging Face são carregados em modo offline, do cache local. Para permitir download de
modelos em outra máquina, exporte `HF_HUB_OFFLINE=0` e `TRANSFORMERS_OFFLINE=0` antes de executar.
`--backend hashing` é um baseline explicitamente diferente; seus scores não calibram o gate do e5.
O reranker continua sendo o cross-encoder configurado, mesmo com hashing.

As saídas ficam em `eval/resultados/`:

- `recuperacao/<backend>/`: 15 configurações + melhor da grade com reranker, resultados por pergunta,
  tabelas agregadas, métricas por serviço/categoria, distribuições e limiares do gate.
- `geracao/<llm>/`: duas versões de prompt por pergunta respondível/sem resposta, com contexto controlado.
- `ponta_a_ponta/<llm>/<backend>/`: todas as categorias, histórico, chunks, resposta, trace, citações,
  julgamento automático e coluna vazia para revisão manual.
- `seguranca/<llm>/`: quatro ataques × dois braços × stub padrão/ingênuo; `--llm gemini` acrescenta
  o Gemini à mesma matriz (16 registros em stub, 24 com Gemini).
- `RESUMO.md`: índice técnico gerado por `rodar_tudo.py`, com comandos e códigos de saída.

Cada avaliação gera CSV (UTF-8 com BOM), JSONL (UTF-8), Markdown e metadados JSON com hashes completos
do corpus/perguntas, configuração efetiva, versões das dependências, argumentos e status. Campos
estruturados são preservados como JSON nas células do CSV. Saídas stub e Gemini ficam separadas;
reexecutar o mesmo braço sobrescreve seus dados. Índices, logs e cache ficam em `eval/.cache/`.
Os prompts e respostas guardados pertencem somente ao ensaio, com dados pessoais sintéticos no teste
de segurança. O gabarito contém 40 fatos literais, conferidos também automaticamente ao carregar.

`--limite N` marca execução parcial: primeiras N perguntas elegíveis na geração; primeiras N perguntas
nos avaliadores de busca/pipeline; primeiros N ataques na segurança. O programa principal propaga
a opção, mas não limita a grade de configurações. `--saida` permite guardar outra rodada separadamente.

As definições e limitações das métricas estão nos resumos produzidos. Em especial: MRR é truncado no
top-5; nDCG só dá ganho à primeira ocorrência de um serviço; chunking fixo não recebe seção por
inferência. Evidência literal, cobertura normalizada e suporte lexical são indicadores distintos.
Similaridade temática não distingue necessariamente uma pergunta sem resposta de uma respondível.
A sugestão de gate é exploratória, calibrada na mesma amostra usada na comparação.

Na segurança, o gate é desligado e todos os chunks do serviço são elegíveis nos dois braços, para
isolar as proteções. O prompt v2 é mantido. A verificação de vazamento de prompt do pacote é
incondicional e continua ativa mesmo com guardrails configuráveis desligados. O ensaio guarda a
saída bruta para tornar isso visível. O LLM ingênuo é um controle positivo hostil, não uma simulação
do Gemini. Julgamentos automáticos, inclusive os sinais de vazamento, exigem revisão manual:
menções a um payload em uma refutação podem produzir falsos positivos.
