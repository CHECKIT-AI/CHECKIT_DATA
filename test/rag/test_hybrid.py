"""python -m unittest discover -s test/rag -p 'test_*.py'"""

import json
from pathlib import Path
import tempfile
import unittest

from RAG.indexing.bm25 import BM25Retriever, build_bm25_index
from RAG.preprocessing.tokenizer import KoreanTokenizer
from RAG.retrieval.hybrid import reciprocal_rank_fusion


def result(key):
    return {"chunk_id": key, "content": key, "metadata": {"article_id": key}}


class HybridTests(unittest.TestCase):
    def test_agreement_is_promoted_and_provenance_retained(self):
        dense = [dict(result("a"), cosine_distance=0.1), dict(result("b"), cosine_distance=0.2)]
        bm25 = [dict(result("b"), bm25_score=4.0), dict(result("c"), bm25_score=3.0)]
        rows = reciprocal_rank_fusion(dense, bm25)
        self.assertEqual(rows[0]["chunk_id"], "b")
        self.assertEqual(rows[0]["dense_rank"], 2)
        self.assertEqual(rows[0]["bm25_rank"], 1)

    def test_one_source_duplicates_do_not_double_score(self):
        rows = reciprocal_rank_fusion([result("a"), result("a")], [])
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["rrf_score"], 1 / 61)

    def test_missing_lexical_results_preserve_dense_order(self):
        self.assertEqual([r["chunk_id"] for r in reciprocal_rank_fusion([result("b"), result("a")], [])], ["b", "a"])

    def test_tokenizer_and_index_integrity(self):
        tokenizer = KoreanTokenizer()
        self.assertIn("article:20:2", tokenizer.tokenize("제 20 조의 2"))
        self.assertNotIn("article:20:0", tokenizer.tokenize("제20조의2"))
        self.assertEqual(tokenizer.tokenize("!!!"), [])
        self.assertTrue(set(tokenizer.tokenize("지급하지 않는다")) & {"않", "안", "못"})
        self.assertEqual(tokenizer.tokenize("ＡＢＣ"), tokenizer.tokenize("abc"))
        rows = [
            {"chunk_id": "a", "article_id": "a", "embedding_text": "퇴사 위약금", "content": "퇴사 위약금"},
            {"chunk_id": "b", "article_id": "b", "embedding_text": "연차 유급휴가", "content": "연차 유급휴가"},
            {"chunk_id": "c", "article_id": "c", "embedding_text": "해고 서면 통지", "content": "해고 서면 통지"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "chunks.jsonl"
            source.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
            folder = Path(temp) / "bm25"
            build_bm25_index(rows, source, folder, "toy")
            index = BM25Retriever(folder, rows, source, "toy")
            self.assertEqual(index.search("퇴사 위약금")[0]["chunk_id"], "a")
            self.assertEqual(index.search("!!!"), [])
            self.assertEqual(index.search("qzxvplkjhgfdsa"), [])
            with self.assertRaises(FileExistsError):
                build_bm25_index(rows, source, folder, "toy")
            source.write_text("changed", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                BM25Retriever(folder, rows, source, "toy")


if __name__ == "__main__":
    unittest.main()
