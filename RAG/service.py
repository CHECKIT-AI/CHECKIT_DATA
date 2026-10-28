"""백엔드에서 호출하는 검색 인터페이스. 별도 HTTP 서버는 실행하지 않는다."""

from RAG.settings import DEFAULT_CORPUS
from RAG.retrieval.temporal import explicit_years
from RAG.retrieval.label_hints import POLICY_VERSION, prioritize, validate_labels


class LawSearchService:
    def __init__(self, corpus=DEFAULT_CORPUS, device="cpu", method="dense"):
        from RAG.retrieval.retriever import LawRetriever

        self.corpus = corpus
        self.method = method
        self.retriever = LawRetriever(corpus=corpus, method=method, device=device)

    def search(self, query, top_k=5):
        results = self.retriever.search(query, top_k=top_k)
        years = explicit_years(query)
        available = {year for row in self.retriever.chunks if row.get("source_type") == "administrative_notice"
                     and row.get("valid_from") and row.get("valid_to")
                     for year in range(int(row["valid_from"][:4]), int(row["valid_to"][:4]) + 1)}
        warnings = []
        if years - available:
            warnings.append("요청한 연도 중 " + ", ".join(str(y) for y in sorted(years - available))
                            + "년의 연간 고시는 수집되어 있지 않습니다. 반환된 일반 법령의 과거 시점 적용 여부는 별도 확인이 필요합니다.")
        return {
            "corpus": self.corpus,
            "method": self.method,
            "results": results,
            "warnings": warnings,
        }

    def search_clause(self, clause_id, text, labels, top_k=5):
        validate_labels(labels)
        if not isinstance(clause_id, str) or not clause_id.strip():
            raise ValueError("조항 ID가 필요합니다.")
        if not isinstance(text, str) or not text.strip() or type(top_k) is not int or not 1 <= top_k <= 10:
            raise ValueError("조항 원문과 1~10의 반환 개수가 필요합니다.")
        # 라벨을 원문 문자열에 섞지 않는다. 숫자·날짜·부정 표현을 그대로 검색한다.
        response = self.search(text, top_k=max(20, top_k))
        response["results"] = prioritize(response["results"], labels)[:top_k]
        response.update(clause_id=clause_id, labels=labels, retrieval_policy=POLICY_VERSION)
        return response
