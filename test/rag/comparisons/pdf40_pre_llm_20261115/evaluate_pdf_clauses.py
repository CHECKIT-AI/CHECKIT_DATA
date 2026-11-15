"""Evaluate the real PDF clause strings against provisional article relevance."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from RAG.service import LawSearchService


ROOT = Path(__file__).resolve().parents[4]
SEGMENTS = ROOT / "outputs/rag/pdf40_pre_llm_20261115/segments_clause_only.jsonl"
GOLD = Path(__file__).with_name("provisional_clause_relevance.jsonl")
OUTPUT = ROOT / "outputs/rag/pdf40_pre_llm_20261115/retrieval_clause_only.jsonl"
REPORT = ROOT / "outputs/rag/pdf40_pre_llm_20261115/retrieval_clause_only_summary.json"


def read(path):
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ignore-labels", action="store_true")
    args = parser.parse_args()
    name = "retrieval_clause_only_no_labels" if args.ignore_labels else "retrieval_clause_only"
    output = OUTPUT.with_name(name + ".jsonl")
    report_path = REPORT.with_name(name + "_summary.json")
    segments = read(SEGMENTS)
    gold = {row["clause_id"]: row for row in read(GOLD)}
    assert len(segments) == len(gold) == 478
    assert {row["gold_clause_id"] for row in segments} == set(gold)
    service = LawSearchService(device="cpu")
    cache = {}
    rows = []
    for index, clause in enumerate(segments, 1):
        query = clause["text"]
        if query not in cache:
            response = service.search_clause(clause["clause_id"], query,
                                             [] if args.ignore_labels else clause["predicted_labels"], top_k=5)
            cache[query] = (
                [{"article_id": item["metadata"]["article_id"], "article_no": item["metadata"]["article_no"],
                  "law_name": item["metadata"]["law_name"]} for item in response["results"]],
                list(dict.fromkeys(item["metadata"]["article_id"] for item in service.retriever.last_candidates)),
            )
        results, candidates = cache[query]
        target = gold[clause["gold_clause_id"]]
        ids = [item["article_id"] for item in results]
        direct = target["direct_article_ids"]
        support = target["support_article_ids"]
        direct_rank = next((rank for rank, item in enumerate(ids, 1) if item in direct), None)
        support_rank = next((rank for rank, item in enumerate(ids, 1) if item in support), None)
        rows.append({"contract_id": clause["contract_id"], "clause_id": clause["gold_clause_id"],
                     "query": query, "predicted_labels": clause["predicted_labels"],
                     "direct_article_ids": direct, "support_article_ids": support,
                     "direct_rank": direct_rank, "support_rank": support_rank,
                     "direct_in_candidate_pool": bool(set(direct) & set(candidates)),
                     "results": results})
        if index % 25 == 0:
            print(f"searched {index}/{len(segments)} clauses; {len(cache)} unique queries", flush=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    eligible = [row for row in rows if row["direct_article_ids"]]
    supported = [row for row in rows if row["support_article_ids"]]
    by_contract = defaultdict(list)
    for row in eligible:
        by_contract[row["contract_id"]].append(row)
    report = {
        "status": "PROVISIONAL_ENGINEERING_REVIEW", "corpus": service.corpus, "method": service.method,
        "query_policy": "original_clause_text",
        "labels_used": not args.ignore_labels,
        "input_sha256": {"segments": sha256(SEGMENTS), "gold": sha256(GOLD)},
        "clauses": len(rows), "unique_queries": len(cache), "direct_eligible": len(eligible),
        "support_eligible": len(supported), "no_direct_target": len(rows) - len(eligible),
        "direct_hit_at_1": sum(row["direct_rank"] == 1 for row in eligible),
        "direct_hit_at_5": sum(row["direct_rank"] is not None for row in eligible),
        "direct_candidate_recall": sum(row["direct_in_candidate_pool"] for row in eligible),
        "direct_mrr_at_5": sum(1 / row["direct_rank"] for row in eligible if row["direct_rank"]) / len(eligible),
        "support_hit_at_5": sum(row["support_rank"] is not None for row in supported),
        "contract_direct_hit_at_5": {key: [sum(row["direct_rank"] is not None for row in value), len(value)]
                                      for key, value in sorted(by_contract.items())},
        "direct_miss_top1_by_first_article": dict(Counter(row["results"][0]["article_no"]
                                                       for row in eligible if row["direct_rank"] != 1)),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"contract_direct_hit_at_5", "direct_miss_top1_by_first_article"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
