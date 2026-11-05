# Recuperação

Sem LLM. Grade de 15 configurações e uma repetição da melhor com reranker. Modelos locais; sem fallback silencioso. Busca sem filtro de serviço; multi-turno usa condense_query.

Seleção lexicográfica: Recall@5 serviço, nDCG@5, Recall@5 seção, MRR, Hit@1; empates mantêm a ordem da grade. Médias macro das perguntas com alvos. Recall serviço = serviços distintos encontrados / alvos; Recall seção = pares (serviço, seção) encontrados / produto dos alvos anotados. Seção geral do fixed não é equiparada a uma seção conhecida. Hit@1/MRR consideram serviço; MRR é truncado no top-5 (ausência vale zero). nDCG usa ganho 1 na primeira ocorrência de cada serviço relevante nas posições dos chunks, ganho 0 para repetições; ideal com serviços distintos nas primeiras posições. Evidência = pelo menos um fato como substring literal em um chunk (sem normalização); sem fatos é n/a. Sem resposta mede localização do serviço, não respondibilidade.

| configuracao | n_com_servico | hit_1 | recall_servico_5 | recall_secao_5 | mrr | ndcg_5 | evidencia_recuperada |
| --- | --- | --- | --- | --- | --- | --- | --- |
| structural_s160_o0_bm25 | 32 | 0.7500 | 0.9531 | 0.7188 | 0.8490 | 0.8637 | 0.9091 |
| structural_s160_o0_dense | 32 | 0.8750 | 0.9323 | 0.7812 | 0.9323 | 0.9004 | 0.9091 |
| structural_s160_o0_hybrid | 32 | 0.9375 | 0.9635 | 0.8177 | 0.9688 | 0.9456 | 0.9545 |
| fixed_s120_o30_bm25 | 32 | 0.7500 | 0.9740 | 0.0000 | 0.8542 | 0.8787 | 0.8182 |
| fixed_s120_o30_dense | 32 | 0.9375 | 0.9427 | 0.0000 | 0.9531 | 0.9257 | 0.9091 |
| fixed_s120_o30_hybrid | 32 | 0.9688 | 0.9531 | 0.0000 | 0.9844 | 0.9525 | 0.9545 |
| fixed_s250_o50_bm25 | 32 | 0.7812 | 0.9740 | 0.0000 | 0.8854 | 0.9034 | 0.9545 |
| fixed_s250_o50_dense | 32 | 0.8438 | 0.9323 | 0.0000 | 0.9010 | 0.8967 | 1.0000 |
| fixed_s250_o50_hybrid | 32 | 0.9688 | 0.9740 | 0.0000 | 0.9844 | 0.9632 | 1.0000 |
| hybrid_s160_o40_bm25 | 32 | 0.7500 | 0.9427 | 0.6719 | 0.8438 | 0.8604 | 0.8636 |
| hybrid_s160_o40_dense | 32 | 0.8750 | 0.9323 | 0.7812 | 0.9271 | 0.8963 | 0.9091 |
| hybrid_s160_o40_hybrid | 32 | 0.9062 | 0.9531 | 0.7969 | 0.9479 | 0.9298 | 0.9545 |
| hybrid_s80_o20_bm25 | 32 | 0.6875 | 0.9427 | 0.6562 | 0.7995 | 0.8331 | 0.8182 |
| hybrid_s80_o20_dense | 32 | 0.9062 | 0.9219 | 0.7448 | 0.9531 | 0.9097 | 0.8636 |
| hybrid_s80_o20_hybrid | 32 | 0.9688 | 0.9427 | 0.7552 | 0.9844 | 0.9396 | 0.9091 |
| fixed_s250_o50_hybrid_rerank | 32 | 0.9375 | 1.0000 | 0.0000 | 0.9635 | 0.9672 | 1.0000 |

## Calibração exploratória

Máximo do sinal no top-5 efetivamente entregue; aprovação se score >= limiar. Distribuições separadas para respondível, fora de escopo e sem resposta. Limiares incluem scores, pontos médios, extremos e o padrão do config. A sugestão minimiza a média das taxas de respondíveis bloqueadas e fora de escopo aprovadas; desempate preserva respondíveis. Amostra pequena e usada na seleção e calibração: não é validação independente. Sem resposta é mostrado à parte: similaridade temática não demonstra resposta disponível.

| configuracao | sinal | limiar | respondiveis_bloqueadas | n_respondivel | fora_escopo_passariam | n_fora_escopo | sem_resposta_passariam | erro_balanceado |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_s250_o50_hybrid | dense | 0.85 | 0 | 18 | 0 | 4 | 5 | 0.0 |
| fixed_s250_o50_hybrid_rerank | dense | 0.8399651646614075 | 0 | 18 | 0 | 4 | 6 | 0.0 |
| fixed_s250_o50_hybrid_rerank | rerank | -4.32698655128479 | 0 | 18 | 0 | 4 | 5 | 0.0 |
