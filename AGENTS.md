# Regras do projeto para agentes de código

Repositório `citizentech-genai` — assistente RAG (pacote `citizen_rag`) que responde a agricultores
familiares do semiárido sobre serviços públicos (Garantia-Safra, CAF, Pronaf, benefícios rurais…), com base
nas Cartas de Serviço do gov.br em `data/corpus/servicos.jsonl`. FIAP Global Solution 2026.2, disciplina
Generative AI. Faz parte da solução CitizenTech com `citizentech-rpa` (`../RPA`) e `citizentech-dashboard`.

## Convenções
- Código, comentários, logs, perguntas de teste e relatórios em **português do Brasil**.
- Python: `..\.venv\Scripts\python.exe` (Windows). O pacote está instalado em modo editável.
  Não instale pacotes novos sem necessidade; se instalar, adicione em `requirements.txt` e `pyproject.toml`.
- Reutilize o que existe: `citizen_rag.config` (`load_config`, `resolve_path`, `get_logger`,
  `gemini_api_key`), `RAGAssistant(overrides=..., llm=...)`, `Retriever.retrieve(...)`,
  `Generator.generate(question, chunks, version=...)`, `guardrails.*`, `llm.StubLLM`, `llm.make_llm`.
  Configurações alternativas para comparação vão por `overrides`, nunca editando `config.yaml`.
- O LLM é o Gemini (free tier). Todo script que dependa dele precisa funcionar também em modo `--llm stub`
  e sair com mensagem clara (sem stack trace) quando não houver chave ou rede.
  Respeite a cota: cache ligado e pausa configurável entre chamadas.
- O agente nunca lê, grava ou exibe o conteúdo de `.env`. O **código da aplicação** carrega a chave do ambiente/`.env` normalmente (python-dotenv) — não desative isso; chaves ficam fora do código e do git.
- Perguntas e "fatos esperados" das avaliações precisam ser conferidos no texto do corpus — nada inventado.

- Grave **todos** os arquivos em UTF-8. No PowerShell nunca use `Set-Content`/`Out-File` sem `-Encoding utf8` (o padrão é ANSI e troca acentos por "?"); prefira o editor/patch.

## Antes de terminar
- Rode os testes: `..\.venv\Scripts\python -m unittest discover -s tests -q` (todos devem passar).
- Mexa só nos arquivos que a tarefa pede. Não faça commit nem push.
- Termine com um relatório curto: arquivos alterados, comandos rodados e resultados, decisões, pendências.
