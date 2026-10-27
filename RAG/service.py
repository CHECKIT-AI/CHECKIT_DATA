"""백엔드에서 호출하는 검색 인터페이스. 별도 HTTP 서버는 실행하지 않는다."""


class LawSearchService:
    def __init__(self, corpus="labor_standards_v001", device="cpu"):
        from RAG.retrieval.retriever import LawRetriever

        self.corpus = corpus
        self.retriever = LawRetriever(corpus=corpus, method="dense", device=device)

    def search(self, query, top_k=5):
        return {
            "corpus": self.corpus,
            "method": "dense",
            "results": self.retriever.search(query, top_k=top_k),
        }
