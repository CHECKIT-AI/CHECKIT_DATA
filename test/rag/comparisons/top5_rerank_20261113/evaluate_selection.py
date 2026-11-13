"""Offline citation selection audit using the backend's existing Gemini reviewer."""

import argparse
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

from app.analysis_llm import GeminiReviewer
from app.config import Settings


DATA = Path(__file__).resolve().parents[4]
BENCH = DATA.parent / "CheckIt_Eval_Benchmark" / "ground_truth"
RUN = sorted((DATA / "outputs/rag").glob("benchmark40_baseline_20261113_*/results.jsonl"))[-1]
OUTPUT = DATA / "outputs/rag/benchmark40_selection_20261113.jsonl"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--contract")
    parser.add_argument("--effective-date", type=date.fromisoformat,
                        help="테스트 입력에만 적용할 날짜 (YYYY-MM-DD)")
    args = parser.parse_args()
    output = (DATA / f"outputs/rag/benchmark40_selection_effective_{args.effective_date:%Y%m%d}.jsonl"
              if args.effective_date else OUTPUT)
    queries = read_jsonl(RUN)
    if args.effective_date:
        latest = max(date.fromisoformat(raw if "-" in raw else f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}")
                     for row in queries for hit in row["results"][:5]
                     if (raw := hit["metadata"].get("article_effective_date") or
                         hit["metadata"].get("snapshot_effective_date")))
        if args.effective_date <= latest:
            parser.error(f"테스트 날짜는 후보 법령의 최신 기록 날짜 {latest}보다 늦어야 합니다.")
    issues = {row["issue_id"]: row for row in read_jsonl(BENCH / "issues.jsonl")}
    contracts = {row["contract_id"]: row for row in read_jsonl(BENCH / "contracts.jsonl")}
    clauses = defaultdict(list)
    for row in read_jsonl(BENCH / "clauses.jsonl"):
        clauses[row["contract_id"]].append({"clause_id": row["clause_id"], "text": row["clause_text"]})
    groups = defaultdict(list)
    for row in queries:
        groups[row["contract_id"]].append(row)
    completed = {row["contract_id"] for row in read_jsonl(output) if "results" in row} if output.exists() and not args.contract else set()
    reviewer = GeminiReviewer(Settings())
    count = 0
    for contract_id, rows in groups.items():
        if args.contract and contract_id != args.contract:
            continue
        if contract_id in completed:
            continue
        reviews = []
        for row in rows:
            issue = issues[row["query_id"]]
            category = {"LEGAL_VIOLATION": "LEGAL_RISK", "MISSING_REQUIRED": "MISSING", "UNFAIR_TERM": "DISADVANTAGE"}[issue["issue_type"]]
            reviews.append(({
                "candidate_id": row["query_id"], "check_code": issue["issue_code"],
                "category": category, "clause_id": row["clause_id"],
                "explanation": row["query"], "required_information": [],
                "retrieval_warnings": [],
            }, row["results"][:5]))
        contract = contracts[contract_id]
        context = {"effective_date": str(args.effective_date or contract["contract_date"]),
                   "employee_count": contract["workplace_size"],
                   "work_type": contract["employment_type"]}
        try:
            answers = reviewer.judge(clauses[contract_id], reviews, context, contract_id)
            records = []
            for row in rows:
                answer = answers[row["query_id"]]
                by_chunk = {hit["chunk_id"]: hit["metadata"]["article_id"] for hit in row["results"][:5]}
                records.append({
                    "query_id": row["query_id"], "gold_article_ids": row["relevant_article_ids"],
                    "candidate_article_ids": list(by_chunk.values()),
                    "citation_chunk_ids": answer.citation_chunk_ids,
                    "selected_article_ids": [by_chunk[chunk] for chunk in answer.citation_chunk_ids],
                    "status": answer.status,
                })
            result = {"contract_id": contract_id, "split": contract["split"],
                      "effective_date": context["effective_date"], "results": records}
        except Exception as exc:
            result = {"contract_id": contract_id, "split": contract["split"],
                      "effective_date": context["effective_date"], "error_type": type(exc).__name__}
        with output.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
        count += 1
        print(contract_id, "ok" if "results" in result else result["error_type"], flush=True)
        if args.limit and count >= args.limit:
            break


if __name__ == "__main__":
    main()
