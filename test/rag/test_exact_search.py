"""정확 검색이 벡터 크기가 아닌 cosine과 안정적인 동점으로 정렬하는지 검증."""

import unittest

from RAG.pipelines.compare_exact import ExactCosineSearch


class ExactSearchTests(unittest.TestCase):
    def setUp(self):
        self.searcher = ExactCosineSearch({
            "ids": ["z", "b", "a"],
            "embeddings": [[0, 10], [2, 0], [100, 0]],
            "documents": ["Z", "B", "A"],
            "metadatas": [{"article_id": x} for x in ["z", "b", "a"]],
        })

    def test_cosine_and_stable_ties_preserve_document_alignment(self):
        rows = self.searcher.search([3, 0])
        self.assertEqual([r["chunk_id"] for r in rows], ["a", "b", "z"])
        self.assertEqual([r["content"] for r in rows], ["A", "B", "Z"])
        self.assertEqual([r["cosine_distance"] for r in rows], [0, 0, 1])
        self.assertEqual([r["metadata"]["article_id"] for r in rows], ["a", "b", "z"])

    def test_invalid_query_rejected(self):
        for query in ([0, 0], [1], [float("nan"), 0]):
            with self.assertRaises(ValueError):
                self.searcher.search(query)

    def test_zero_document_rejected(self):
        with self.assertRaises(ValueError):
            ExactCosineSearch({"ids": ["x"], "embeddings": [[0, 0]], "documents": ["x"], "metadatas": [{}]})


if __name__ == "__main__":
    unittest.main()
