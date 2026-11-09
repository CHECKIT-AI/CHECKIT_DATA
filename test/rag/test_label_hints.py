import unittest
from types import SimpleNamespace

from RAG.retrieval.label_hints import LABELS, prioritize, validate_labels
from RAG.service import LawSearchService


def hit(identifier, rank, article, distance):
    return {"chunk_id": identifier, "rank": rank, "cosine_distance": distance,
            "content": "원문 금액 2,500,000원", "metadata": {"law_name": "근로기준법", "article_no": article}}


class LabelHintTests(unittest.TestCase):
    def test_only_ten_unique_known_labels(self):
        self.assertEqual(len(LABELS), 10)
        self.assertEqual(validate_labels([]), [])
        for bad in ["임금", ["임대차"], ["임금", "임금"], [1]]:
            with self.assertRaises(ValueError):
                validate_labels(bad)

    def test_soft_priority_keeps_global_candidates_and_official_content(self):
        rows = [hit("hours", 1, "제54조", .10), hit("wage", 2, "제43조", .11), hit("other", 3, "제55조", .30)]
        ranked = prioritize(rows, ["임금"])
        self.assertEqual([r["chunk_id"] for r in ranked], ["wage", "hours", "other"])
        self.assertEqual(ranked[0]["base_rank"], 2)
        self.assertEqual(ranked[0]["matched_labels"], ["임금"])
        self.assertEqual(ranked[0]["content"], rows[1]["content"])
        self.assertEqual(rows[1]["rank"], 2)

    def test_wrong_label_cannot_override_large_semantic_gap(self):
        rows = [hit("correct", 1, "제54조", .10), hit("label-only", 2, "제43조", .50)]
        self.assertEqual(prioritize(rows, ["임금"])[0]["chunk_id"], "correct")
        self.assertEqual(prioritize(rows, ["그 외"])[0]["chunk_id"], "correct")

    def test_multilabel_does_not_accumulate_bonus(self):
        row = hit("shared", 1, "제17조", .2)
        one = prioritize([row], ["임금"])[0]
        many = prioritize([row], ["임금", "근로시간", "휴일"])[0]
        self.assertEqual(one["label_priority_score"], many["label_priority_score"])

    def test_clause_service_searches_original_text_and_echoes_link(self):
        captured = []
        service = LawSearchService.__new__(LawSearchService)
        service.corpus, service.method = "test", "dense"
        def search(text, top_k):
            captured.append((text, top_k))
            return [hit("correct", 1, "제43조", .1)]
        service.retriever = SimpleNamespace(chunks=[], search=search)
        text = "시급 10,320원, 월급 2,500,000원. 공제하지 않는다."
        result = service.search_clause("c-7", text, ["임금"], top_k=1)
        self.assertEqual(captured, [(text, 20)])
        self.assertEqual(result["clause_id"], "c-7")
        self.assertEqual(result["labels"], ["임금"])
        self.assertEqual(result["retrieval_policy"], "labels_soft_priority_v1")
        with self.assertRaises(ValueError):
            service.search_clause("c-7", text, ["임금"], top_k=True)

    def test_empty_labels_keep_original_search_order_without_priority_policy(self):
        rows = [hit("first", 1, "제54조", .10), hit("second", 2, "제43조", .11)]
        service = LawSearchService.__new__(LawSearchService)
        service.corpus, service.method = "test", "dense"
        service.retriever = SimpleNamespace(chunks=[], search=lambda *_args, **_kwargs: rows)

        result = service.search_clause("c-empty", "분류되지 않은 조항", [], top_k=2)

        self.assertEqual(result["labels"], [])
        self.assertIsNone(result["retrieval_policy"])
        self.assertEqual(result["results"], rows)


if __name__ == "__main__":
    unittest.main()
