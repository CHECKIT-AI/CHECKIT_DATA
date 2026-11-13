"""Evaluate replacing only rank five with a strong candidate from the retrieval pool."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from RAG.evaluation.retrieval_metrics import score_query
from RAG.retrieval.reranker import BGEReranker, rerank_candidates
from RAG.retrieval.retriever import LawRetriever
from RAG.settings import STORAGE_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margins", type=float, nargs="+", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    queries = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    prepared = json.loads((STORAGE_ROOT / "models/bge-reranker-v2-m3.json").read_text(encoding="utf-8"))
    retriever = LawRetriever(method="dense_bm25_top5", device=args.device)
    reranker = BGEReranker(revision=prepared["model_revision"], device=args.device,
                           batch_size=2, max_length=2048)
    rows = []
    retriever.search("법령 검색 준비", top_k=10)
    for query in queries:
        started = perf_counter()
        baseline = retriever.search(query["query"], top_k=10)
        pool = retriever.last_candidates
        scored = rerank_candidates(query["query"], pool, reranker.score, len(pool))
        scores = {row["chunk_id"]: row["reranker_score"] for row in scored}
        top_ids = {row["metadata"]["article_id"] for row in baseline[:5]}
        outsider = next((row for row in scored if row["metadata"]["article_id"] not in top_ids), None)
        margin = (scores[outsider["chunk_id"]] - scores[baseline[4]["chunk_id"]]
                  if outsider and len(baseline) >= 5 else None)
        metrics = {}
        for threshold in args.margins:
            chosen = baseline[:4] + [outsider] if margin is not None and margin >= threshold else baseline[:5]
            metrics[str(threshold)] = score_query(chosen, query["relevant_article_ids"], ks=(1, 5))
        rows.append({
            "query_id": query["query_id"], "split": query.get("split"),
            "baseline": score_query(baseline[:5], query["relevant_article_ids"], ks=(1, 5)),
            "pool_has_gold": bool(set(query["relevant_article_ids"]) &
                                  {row["metadata"]["article_id"] for row in pool}),
            "promoted_article_id": outsider["metadata"]["article_id"] if outsider else None,
            "score_margin": margin, "metrics": metrics,
            "elapsed_ms": (perf_counter() - started) * 1000,
        })
        print(query["query_id"], flush=True)
    summary = {"query_count": len(rows),
               "baseline_hit1": sum(row["baseline"]["hit@1"] for row in rows),
               "baseline_hit5": sum(row["baseline"]["hit@5"] for row in rows),
               "pool_gold": sum(row["pool_has_gold"] for row in rows),
               "margins": {str(m): {
                   "hit1": sum(row["metrics"][str(m)]["hit@1"] for row in rows),
                   "hit5": sum(row["metrics"][str(m)]["hit@5"] for row in rows),
                   "promotions": sum(row["score_margin"] is not None and row["score_margin"] >= m for row in rows),
               } for m in args.margins}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"dataset": str(args.dataset),
                                       "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                                       "index_manifest_sha256": hashlib.sha256(retriever.manifest_path.read_bytes()).hexdigest(),
                                       "chunks_sha256": retriever.manifest["source_sha256"],
                                       "corpus": retriever.manifest["corpus_id"],
                                       "model": reranker.describe(),
                                       "summary": summary, "results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
