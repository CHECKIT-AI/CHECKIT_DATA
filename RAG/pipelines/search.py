"""같은 corpus에 대해 dense / bm25 / hybrid 검색을 실행한다."""

import argparse
import json

from RAG.pipelines.build_index import corpus_name
from RAG.retrieval.retriever import LawRetriever
from RAG.settings import DEFAULT_CORPUS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=corpus_name, default=DEFAULT_CORPUS, help=f"검색 인덱스 (기본값: BGE-M3 {DEFAULT_CORPUS})")
    parser.add_argument("--query", required=True)
    parser.add_argument("--method", choices=["dense", "bm25", "hybrid", "weighted_rrf", "hybrid_rerank"], default="dense")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-weight", type=float, default=0.8)
    parser.add_argument("--bm25-weight", type=float, default=0.2)
    parser.add_argument("--reranker-batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.query.strip() or min(args.top_k, args.candidate_k, args.rrf_k) < 1:
        parser.error("검색 문장과 양수 검색 설정을 입력하세요.")
    retriever = LawRetriever(args.corpus, method=args.method, device=args.device, candidate_k=args.candidate_k, rrf_k=args.rrf_k, dense_weight=args.dense_weight, bm25_weight=args.bm25_weight, reranker_batch_size=args.reranker_batch_size)
    results = retriever.search(args.query.strip(), top_k=args.top_k)
    print(json.dumps({
        "query": args.query, "corpus": args.corpus, "method": args.method,
        "retrieval_config": retriever.describe(), "results": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
