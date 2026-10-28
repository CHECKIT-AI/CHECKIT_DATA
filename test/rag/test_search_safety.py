"""검색용 정리·고시 연도 필터의 의미 보존과 누락 방지를 검증."""
import unittest
from types import SimpleNamespace

from RAG.preprocessing.search_text import clean_search_text
from RAG.retrieval.temporal import explicit_years, filter_notice_years
from RAG.retrieval.retriever import LawRetriever
from RAG.service import LawSearchService


class SearchSafetyTests(unittest.TestCase):
    def test_markup_removed_without_losing_numbers_or_negation(self):
        text = '① 월급 2,000,000원은 &lt;b&gt;공제하지 않는다&lt;/b&gt;.\n┃ 시급 │ 10,320원 ┃\n<img src="secret">'
        cleaned = clean_search_text(text)
        self.assertNotIn('<', cleaned)
        self.assertIn('공제하지 않는다', cleaned)
        self.assertIn('2,000,000원', cleaned)
        self.assertIn('시급 | 10,320원', cleaned)

    def test_notice_dates_do_not_filter_general_law(self):
        law = {'metadata': {'law_name': '근로기준법'}}
        notice = {'metadata': {'source_type': 'administrative_notice', 'valid_from': '20260101', 'valid_to': '20261231'}}
        unknown = {'metadata': {'source_type': 'administrative_notice'}}
        self.assertEqual(explicit_years('2025년과 2026-01-01, 시급 2030원'), {2025, 2026})
        self.assertEqual(filter_notice_years([law, notice, unknown], {2025}), [law])
        self.assertEqual(filter_notice_years([law, notice], {2026}), [law, notice])
        self.assertEqual(filter_notice_years([law, notice], set()), [law, notice])
        ranked_notice = {**notice, 'rank': 1}
        ranked_law = {**law, 'rank': 2}
        self.assertEqual(filter_notice_years([ranked_notice, ranked_law], {2025})[0]['rank'], 1)
        self.assertEqual(ranked_law['rank'], 2)

    def test_filter_happens_before_dense_topk_cutoff(self):
        notice = {'metadata': {'source_type': 'administrative_notice', 'valid_from': '20260101', 'valid_to': '20261231'}}
        law = {'metadata': {'law_name': '근로기준법'}}
        requested = []
        retriever = LawRetriever.__new__(LawRetriever)
        retriever.method, retriever.candidate_k = 'dense', 20
        retriever.chunks = [1, 2]
        retriever.embedder = SimpleNamespace(encode=lambda *args, **kwargs: [[1]])
        def search(vector, top_k):
            requested.append(top_k)
            return [notice, law][:top_k]
        retriever.store = SimpleNamespace(search=search)
        self.assertEqual(retriever.search('2025년 최저임금', top_k=1), [law])
        self.assertEqual(requested, [2])

    def test_missing_annual_notice_is_explicit_in_service(self):
        service = LawSearchService.__new__(LawSearchService)
        service.corpus, service.method = 'test', 'dense'
        service.retriever = SimpleNamespace(chunks=[{'source_type': 'administrative_notice',
            'valid_from': '20260101', 'valid_to': '20261231'}], search=lambda *args, **kwargs: [])
        self.assertIn('2025', service.search('2025년 최저임금')['warnings'][0])
        self.assertEqual(service.search('2026년 최저임금')['warnings'], [])


if __name__ == '__main__':
    unittest.main()
