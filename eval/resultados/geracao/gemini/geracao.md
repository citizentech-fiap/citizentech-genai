# Geração com contexto controlado

LLM: gemini. Status: concluído. Perguntas selecionadas: 24. Registros: 48. Parcial por limite: False.

Todos os chunks estruturais das seções anotadas, sem recuperação nem gate. Sem resposta recebe seções tematicamente relacionadas que não fornecem a informação solicitada; a ausência foi conferida no corpus e justificada em observacao. Não se injeta um fato fictício no gabarito.

Cobertura é a fração de trechos esperados encontrados com acentos/caixa normalizados; sem fatos = n/a. Citação válida aponta para posição existente; precisão é a fração das citações válidas únicas para serviço relevante (neste contexto controlado, todos os chunks são relevantes por serviço: métrica limitada). Validade e coincidência lexical não provam sustentação semântica. Abstenção correta usa is_abstention; o julgamento também exige ausência de citação para sem resposta. Para respondível exige todos os fatos e citações válidas do serviço esperado. Juiz ausente/falho = n/a, nunca aprovação.

Stub apenas reproduz até 40 palavras do primeiro documento; resultados demonstram execução e métricas, não qualidade do Gemini. Mesmos contextos e cliente/cache para v1 e v2.

| prompt | categoria | n_validas | cobertura_fatos | tem_citacao_valida | precisao_citacoes | abstencao_correta | suporte_lexical | julgamento_automatico | juiz_fundamentada |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | respondivel | 18 | 0.5926 | 1.0000 | 1.0000 | n/a | 0.7740 | 0.3333 | 1.0000 |
| v1 | sem_resposta | 6 | n/a | 1.0000 | 1.0000 | 0.0000 | 0.6020 | 0.0000 | 0.8333 |
| v2 | respondivel | 18 | 0.7130 | 1.0000 | 1.0000 | n/a | 0.6888 | 0.4444 | 1.0000 |
| v2 | sem_resposta | 6 | n/a | 0.1667 | 1.0000 | 0.8333 | 0.4135 | 0.8333 | 0.1667 |
