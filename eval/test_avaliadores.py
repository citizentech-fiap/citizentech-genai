"""Verificações locais de gabarito, métricas e controles positivos de segurança."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.comum import CORPUS, carregar_perguntas, configuracao_avaliacao, cobertura, normalizar
from eval.avaliar_recuperacao import calibrar, metricas
from eval.avaliar_geracao import contexto_controlado
from eval.avaliar_seguranca import (ATAQUES, CANARY, ENVENENADO, LLMIngenuo,
                                    sinais_vazamento, validar_fixture)
from citizen_rag.chunking import chunk_corpus
from citizen_rag.config import deep_merge
from citizen_rag.indexing import Index
from citizen_rag.ingest import load_corpus
from citizen_rag.rag import RAGAssistant
from citizen_rag.retriever import RetrievedChunk


def trecho(sid, secao="custos", texto="O custo é R$ 24,00."):
    return RetrievedChunk(id=f"{sid}:{secao}", text=texto, score=1,
                          metadata={"service_id": sid, "secao": secao})


class TesteGabarito(unittest.TestCase):
    def test_quarenta_perguntas_e_fatos_literais(self):
        perguntas = carregar_perguntas()
        self.assertEqual(len(perguntas), 40)
        self.assertEqual(sum(len(p["fatos_esperados"]) for p in perguntas), 40)
        self.assertEqual({s for p in perguntas for s in p["servicos_relevantes"]},
                         {d.service_id for d in load_corpus(CORPUS)})

    def test_contexto_controlado_nao_perde_fatos(self):
        cfg = configuracao_avaliacao(adicionais={"chunking": {"strategy": "structural", "size": 160, "overlap": 0}})
        chunks = chunk_corpus(load_corpus(CORPUS), cfg)
        for p in carregar_perguntas():
            if p["categoria"] in {"respondivel", "sem_resposta"}:
                with self.subTest(p=p["id"]):
                    self.assertTrue(contexto_controlado(p, chunks))

    def test_fixture_altera_somente_um_servico(self):
        validar_fixture()

    def test_normalizacao_e_nulos(self):
        self.assertEqual(normalizar("  INSCRIÇÃO\nCAF "), "inscricao caf")
        self.assertEqual(cobertura("custa r$ 24,00", ["R$ 24,00"])[0], 1)
        self.assertIsNone(cobertura("qualquer texto", [])[0])


class TesteMetricas(unittest.TestCase):
    def setUp(self):
        self.p = {"servicos_relevantes": ["a", "b"], "secoes_relevantes": ["custos"],
                  "fatos_esperados": ["R$ 24,00"]}

    def test_sem_repetir_ganho_por_chunks_do_mesmo_servico(self):
        m = metricas(self.p, [trecho("a"), trecho("a"), trecho("b")])
        self.assertEqual(m["hit_1"], 1)
        self.assertEqual(m["recall_servico_5"], 1)
        self.assertEqual(m["recall_secao_5"], 1)
        self.assertAlmostEqual(m["ndcg_5"], (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))

    def test_primeiro_relevante_na_terceira_posicao(self):
        m = metricas(self.p, [trecho("x"), trecho("y"), trecho("b")])
        self.assertEqual(m["hit_1"], 0)
        self.assertEqual(m["recall_servico_5"], .5)
        self.assertAlmostEqual(m["mrr"], 1 / 3)

    def test_fixed_nao_recebe_secao_por_inferencia(self):
        m = metricas(self.p, [trecho("a", "geral"), trecho("b", "geral")])
        self.assertEqual(m["recall_servico_5"], 1)
        self.assertEqual(m["recall_secao_5"], 0)

    def test_fato_fora_do_top5_nao_conta(self):
        chunks = [trecho("x", texto="Sem esse valor.") for _ in range(5)] + [trecho("a")]
        m = metricas(self.p, chunks)
        self.assertFalse(m["evidencia_recuperada"])
        self.assertEqual(m["mrr"], 0)

    def test_sem_alvo_nao_e_zero_artificial(self):
        m = metricas({"servicos_relevantes": [], "secoes_relevantes": [], "fatos_esperados": []}, [])
        for chave in ("hit_1", "recall_servico_5", "recall_secao_5", "mrr", "ndcg_5", "evidencia_recuperada"):
            self.assertIsNone(m[chave])

    def test_calibracao_conta_fronteira_inclusiva_e_score_ausente(self):
        amostras = [{"categoria": c, "melhor_dense": v, "melhor_rerank": None} for c, v in
                    [("respondivel", .9), ("respondivel", None), ("fora_escopo", .2), ("sem_resposta", .95)]]
        dist, candidatos, sugestoes = calibrar(amostras, "ensaio")
        fronteira = next(r for r in candidatos if r["limiar"] == .9)
        self.assertEqual(fronteira["respondiveis_bloqueadas"], 1)
        self.assertEqual(fronteira["fora_escopo_passariam"], 0)
        self.assertEqual(fronteira["sem_resposta_passariam"], 1)
        self.assertEqual(len(sugestoes), 1)
        self.assertEqual(next(r for r in dist if r["categoria"] == "respondivel" and r["sinal"] == "dense")["n_score"], 1)


class TesteControlesSeguranca(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = configuracao_avaliacao("hashing", {
            "chunking": {"strategy": "structural", "size": 160, "overlap": 0},
            "gate": {"enabled": False}, "retriever": {"mode": "bm25", "top_k": 100, "candidate_k": 100}})
        cls.indices = {str(p): Index(chunk_corpus(load_corpus(p), cls.cfg), cls.cfg, None, None)
                       for p in (CORPUS, ENVENENADO)}

    def ensaio(self, ataque, ligado):
        filtros = {"block_injection": ligado, "sanitize_chunks": ligado, "mask_pii": ligado,
                   "require_citation": ligado, "citation_policy": "reject" if ligado else "strip"}
        cfg = deep_merge(self.cfg, {"guardrails": filtros})
        llm = LLMIngenuo()
        rag = RAGAssistant(corpus_path=ataque["corpus"], overrides=cfg, llm=llm,
                           index=self.indices[str(ataque["corpus"])])
        saida = rag.answer(ataque["pergunta"], service_id=ataque["servico"])
        self.assertNotEqual(saida["reason"], "erro_interno")
        return saida, rag.last_trace, llm

    def test_injecao_direta_tem_controle_positivo(self):
        saida, _, _ = self.ensaio(ATAQUES[0], False)
        self.assertTrue(sinais_vazamento("injecao_direta", saida["answer"]))
        saida, _, llm = self.ensaio(ATAQUES[0], True)
        self.assertTrue(saida["blocked"])
        self.assertFalse(llm.calls)

    def test_canario_interceptado_mesmo_com_overrides_desligados(self):
        saida, trace, llm = self.ensaio(ATAQUES[1], False)
        self.assertIn(CANARY, llm.calls[0]["system"])
        self.assertIn(CANARY, trace["raw_answer"])
        self.assertEqual(saida["reason"], "vazamento_prompt")
        self.assertNotIn(CANARY, saida["answer"])

    def test_pii_mascarada_no_prompt_e_na_saida(self):
        saida, trace, llm = self.ensaio(ATAQUES[2], True)
        self.assertIn("[CPF]", llm.calls[0]["user"])
        self.assertIn("[TELEFONE]", llm.calls[0]["user"])
        self.assertFalse(sinais_vazamento("exposicao_pii", llm.calls[0]["user"]))
        self.assertTrue(sinais_vazamento("exposicao_pii", trace["raw_answer"]))
        self.assertFalse(sinais_vazamento("exposicao_pii", saida["answer"]))
        saida, _, llm = self.ensaio(ATAQUES[2], False)
        self.assertTrue(sinais_vazamento("exposicao_pii", llm.calls[0]["user"]))
        self.assertTrue(sinais_vazamento("exposicao_pii", saida["answer"]))

    def test_indireta_remove_payload_sem_contaminar_indice(self):
        saida, trace, _ = self.ensaio(ATAQUES[3], True)
        self.assertTrue(trace["sanitized_chunks"])
        self.assertFalse(sinais_vazamento("injecao_indireta", saida["answer"]))
        saida, _, _ = self.ensaio(ATAQUES[3], False)
        self.assertTrue(sinais_vazamento("injecao_indireta", saida["answer"]))


if __name__ == "__main__":
    unittest.main()
