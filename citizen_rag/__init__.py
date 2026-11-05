"""citizen_rag — assistente RAG CitizenTech sobre as Cartas de Serviço do gov.br.

Uso principal (contrato com o robô):
    from citizen_rag.rag import RAGAssistant
    rag = RAGAssistant()               # corpus padrão: ../RPA/output/corpus/servicos.jsonl
    out = rag.answer("Como peço o Garantia-Safra?")
"""

__version__ = "0.1.0"
