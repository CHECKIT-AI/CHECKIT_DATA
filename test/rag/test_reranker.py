"""가중치 RRF와 전체 후보 재정렬의 동작을 모델 다운로드 없이 검사한다."""

import unittest

from RAG.retrieval.hybrid import reciprocal_rank_fusion
from RAG.retrieval.reranker import rerank_candidates


def candidate(key, rank):
    return {"chunk_id": key, "rank": rank, "content": key, "metadata": {"law_name": "테스트", "article_id": key}, "rrf_score": 1.0 / (60 + rank)}


class RerankerTests(unittest.TestCase):
    def test_dense_weight_changes_equal_rank_conflict(self):
        rows = reciprocal_rank_fusion([candidate("dense", 1)], [candidate("bm25", 1)], dense_weight=0.8, bm25_weight=0.2)
        self.assertEqual(rows[0]["chunk_id"], "dense")
        self.assertAlmostEqual(rows[0]["rrf_score"], 0.8 / 61)

    def test_invalid_weights_rejected(self):
        for value in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                reciprocal_rank_fusion([], [], dense_weight=value)

    def test_all_candidates_scored_before_topk(self):
        rows = [candidate(str(i), i + 1) for i in range(40)]
        received = []
        def scorer(query, passages):
            received.extend(passages)
            return list(range(40))
        output = rerank_candidates("query", rows, scorer, top_k=5)
        self.assertEqual(len(received), 40)
        self.assertEqual(output[0]["chunk_id"], "39")
        self.assertEqual(output[0]["pre_rerank_rank"], 40)
        self.assertEqual(output[0]["rank"], 1)

    def test_bad_scores_and_duplicate_candidates_rejected(self):
        with self.assertRaises(RuntimeError):
            rerank_candidates("query", [candidate("a", 1)], lambda q, p: [float("nan")], 1)
        with self.assertRaises(RuntimeError):
            rerank_candidates("query", [candidate("a", 1)], lambda q, p: [], 1)
        with self.assertRaises(ValueError):
            rerank_candidates("query", [candidate("a", 1), candidate("a", 2)], lambda q, p: [1, 2], 1)


if __name__ == "__main__":
    unittest.main()
