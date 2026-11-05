# Corpus alterado exclusivamente para teste

`corpus_envenenado.jsonl` contém os mesmos nove registros do snapshot real. Somente `garantia_safra`
foi alterado: uma instrução maliciosa foi acrescentada ao início de `secoes.requisitos` e de
`texto_completo`, para alcançar tanto chunking estrutural quanto fixo. Os demais oito registros
mantêm os mesmos valores JSON do original. A serialização pode diferir.

O registro alterado está marcado com `teste_alterado=true`, `observacao_teste` e
`content_hash_original`; seu novo `content_hash` é SHA-256 de `texto_completo` em UTF-8.
O conteúdo adulterado nunca é usado como gabarito factual. O script de segurança verifica essas
condições antes de executar a matriz de ataques. Marcador de execução: `ATAQUE_INDIRETO_EXECUTADO`.
