import unittest

import helpers
from citizen_rag.chunking import chunk_document, pack_sentences
from citizen_rag.ingest import load_corpus
from citizen_rag.retriever import Retriever, rrf_fuse

PERGUNTAS = [
    ("Fui demitido sem justa causa, como peço o seguro-desemprego?", "seguro_desemprego"),
    ("Quais documentos preciso levar para tirar o passaporte?", "passaporte"),
    ("Quanto custa emitir o passaporte?", "passaporte"),
    ("Qual é a renda máxima para entrar no CadÚnico?", "cadunico"),
    ("Qual o prazo para pedir o seguro depois da demissão?", "seguro_desemprego"),
    ("Onde faço a entrevista do Cadastro Único?", "cadunico"),
]


class TestChunking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = load_corpus(helpers.FIXTURE)

    def test_structural_um_chunk_por_secao(self):
        d = self.docs[0]
        chunks = chunk_document(d, "structural", size=500, overlap=0)
        secoes = [k for k, v in d.secoes.items() if v]
        self.assertEqual(sorted(c.metadata["secao"] for c in chunks), sorted(secoes))
        self.assertTrue(all(c.text.startswith(d.titulo) for c in chunks))
        for k in ("service_id", "titulo", "orgao", "url", "data_coleta", "secao"):
            self.assertIn(k, chunks[0].metadata)

    def test_ids_estaveis(self):
        a = [c.id for c in chunk_document(self.docs[1], "hybrid", 30, 10)]
        b = [c.id for c in chunk_document(self.docs[1], "hybrid", 30, 10)]
        self.assertEqual(a, b)
        self.assertEqual(len(a), len(set(a)))

    def test_fixed_respeita_tamanho_e_sobreposicao(self):
        chunks = chunk_document(self.docs[0], "fixed", size=50, overlap=10, prefix_title=False)
        self.assertTrue(all(c.metadata["n_words"] <= 50 for c in chunks))
        w0, w1 = chunks[0].text.split(), chunks[1].text.split()
        self.assertEqual(w0[-10:], w1[:10])

    def test_hybrid_teto_de_tamanho(self):
        chunks = chunk_document(self.docs[0], "hybrid", size=30, overlap=10)
        self.assertGreater(len(chunks), len([v for v in self.docs[0].secoes.values() if v]))
        self.assertTrue(all(c.metadata["n_words"] <= 30 for c in chunks))

    def test_pack_sentences_overlap(self):
        txt = "Um dois três quatro cinco. Seis sete oito nove dez. Onze doze treze catorze quinze."
        blocks = pack_sentences(txt, size=10, overlap=5)
        self.assertEqual(len(blocks), 2)
        self.assertTrue(blocks[1].startswith("Seis"))


class TestRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        idx = helpers.fixture_index()
        cls.r = Retriever(idx, idx.cfg)

    def test_rrf(self):
        f = rrf_fuse({"a": [1, 2, 3], "b": [3, 1]}, k=60)
        self.assertEqual(max(f, key=f.get), 1)

    def _check(self, mode, min_acc):
        hits = 0
        for q, sid in PERGUNTAS:
            res = self.r.retrieve(q, top_k=3, mode=mode)
            hits += bool(res) and res[0].metadata["service_id"] == sid
        self.assertGreaterEqual(hits / len(PERGUNTAS), min_acc, mode)

    def test_hybrid_acerta_servico(self):
        self._check("hybrid", 1.0)

    def test_bm25_acerta_servico(self):
        self._check("bm25", 0.8)

    def test_dense_acerta_servico(self):
        self._check("dense", 0.8)

    def test_hybrid_secao_documentos(self):
        res = self.r.retrieve("Quais documentos preciso levar para tirar o passaporte?", top_k=3, mode="hybrid")
        self.assertEqual(res[0].metadata["secao"], "documentos")

    def test_filtro_service_id(self):
        res = self.r.retrieve("quais documentos?", top_k=5, mode="hybrid", service_id="cadunico")
        self.assertTrue(res)
        self.assertTrue(all(c.metadata["service_id"] == "cadunico" for c in res))

    def test_filtro_inexistente_cai_para_sem_filtro(self):
        res = self.r.retrieve("passaporte documentos", top_k=3, mode="hybrid", service_id="nao_existe")
        self.assertTrue(res)

    def test_sinais_presentes(self):
        res = self.r.retrieve("passaporte", top_k=2, mode="hybrid")
        self.assertIn("bm25", res[0].signals)
        self.assertIn("dense", res[0].signals)


if __name__ == "__main__":
    unittest.main()
