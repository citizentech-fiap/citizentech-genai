# CitizenTech — Assistente RAG (genAI)

Assistente que responde a cidadãos sobre serviços públicos federais (requisitos, documentos, etapas,
canais, prazos) com base nas Cartas de Serviço do gov.br coletadas pelo robô de RPA.
O relatório completo (avaliação e justificativas) vem depois; este arquivo cobre só o uso.

Global Solution 2026.2 · FIAP · Generative AI & Advanced Nets · Turma 2TIAPF

## Integrantes
| Nome completo | RM |
|---|---|
| Luís Fernando de Oliveira Salgado | RM561401 |
| Ricardo de Paiva Melo | RM565522 |
| Jonas Alaf | RM566479 |

## Parte da solução CitizenTech
| Repositório | Disciplina | Papel |
|---|---|---|
| `citizentech-rpa` | AI for RPA | robô que coleta as Cartas de Serviço e faz a pré-triagem dos pedidos |
| **`citizentech-genai`** (este) | Generative AI | assistente RAG, importado pelo robô e pelo dashboard |
| `citizentech-dashboard` | Front-end | painel de decisão do gestor |

Este repositório roda sozinho: o corpus está em `data/corpus/servicos.jsonl` (fontes e datas de consulta em
[`data/corpus/FONTES.md`](data/corpus/FONTES.md)). Para trazer a coleta mais recente do robô:
`python -m citizen_rag.sync_corpus`.

## Setup
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .          # instala o pacote citizen_rag e as dependências
# opcional: chave do Gemini (free tier) em .env
#   GEMINI_API_KEY=...
```
Sem chave, o assistente funciona em modo degradado: recupera os trechos e devolve
`abstained=True, reason="llm_indisponivel"` (o robô usa o template).

## Índices
```powershell
cd genAI
..\.venv\Scripts\python -m citizen_rag.indexing                # configuração do config.yaml
..\.venv\Scripts\python -m citizen_rag.indexing --grid         # todas as configurações de comparação
..\.venv\Scripts\python -m citizen_rag.indexing --corpus tests\fixtures\corpus_fixture.jsonl
```
Corpus padrão: `data/corpus/servicos.jsonl` (se não existir, usa o fixture de testes).
Cada configuração (corpus + chunking + embedding) ganha seu diretório em `indexes/`.
O índice também é construído sob demanda na primeira chamada.

## Uso
```python
from citizen_rag.rag import RAGAssistant    # após pip install -e .

rag = RAGAssistant()                       # ou RAGAssistant(corpus_path=..., overrides={...}, llm=...)
out = rag.answer("Fui demitido, como peço o seguro-desemprego?",
                 history=[{"role": "user", "content": "..."}], service_id=None)
# out: answer, citations[{n, service_id, titulo, url, secao}], abstained, reason, chunks, blocked
rag.last_trace                             # diagnóstico da última chamada (sinais, gate, LLM, citações)
```
Demonstrações:
```powershell
..\.venv\Scripts\python -m citizen_rag.retriever                     # compara bm25/dense/hybrid
..\.venv\Scripts\python -m citizen_rag.retriever --mode hybrid --rerank "quanto custa o passaporte?"
..\.venv\Scripts\python -m citizen_rag.rag "Quais documentos para o passaporte?"
```

## Componentes (`citizen_rag/`)
| módulo | papel |
|---|---|
| `config.py` | config.yaml, `.env`, logging |
| `ingest.py` | JSONL do corpus → `ServiceDoc` com metadados |
| `chunking.py` | estratégias `structural`, `fixed`, `hybrid` |
| `embeddings.py` | e5 multilíngue (prefixos query/passage) ou hashing offline |
| `indexing.py` | BM25 + vetores (numpy `np.save`; Chroma opcional), um diretório por configuração |
| `retriever.py` | bm25 / dense / hybrid (RRF), filtro `service_id`, reranker cross-encoder opcional |
| `guardrails.py` | injeção direta/indireta, PII, gate de evidência, citações, vazamento de prompt, juiz |
| `llm.py` | Gemini (retry/backoff/fallback/timeout), cache em disco, `StubLLM` para testes |
| `generator.py` | prompts versionados (`v1`, `v2`) e montagem do contexto |
| `rag.py` | `RAGAssistant` (orquestração; nunca lança exceção) |

Para avaliação: `Retriever.retrieve(query, top_k, mode, service_id, rerank)` e
`Generator.generate(question, chunks, version=...)` podem ser chamados separadamente, com contexto controlado.

## Testes (offline, LLM stub)
```powershell
cd genAI\tests
..\..\.venv\Scripts\python -m unittest -v
$env:CITIZEN_RAG_TEST_BACKEND="hashing"; ..\..\.venv\Scripts\python -m unittest   # sem baixar modelo
```
