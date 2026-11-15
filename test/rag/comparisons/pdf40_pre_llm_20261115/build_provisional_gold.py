"""Create an explicit, provisional relevance table for the synthetic PDF clauses."""

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BENCH = ROOT.parent / "CheckIt_Eval_Benchmark/ground_truth"
CHUNKS = ROOT / "data/rag/processed/employment_basic_v004/chunks.jsonl"
REVIEW = ROOT / "test/rag/comparisons/benchmark_ground_truth_review_20261113.jsonl"
OUTPUT = Path(__file__).with_name("provisional_clause_relevance.jsonl")


def read(path):
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def main():
    clauses = read(BENCH / "clauses.jsonl")
    contracts = {row["contract_id"]: row for row in read(BENCH / "contracts.jsonl")}
    issues = {row["clause_id"]: row for row in read(BENCH / "issues.jsonl") if row["clause_id"]}
    queries = {row["query_id"]: row for row in read(BENCH / "retrieval_queries.jsonl")}
    reviews = {row["query_id"]: row for row in read(REVIEW)}
    articles = {(row["law_name"], row["article_no"]): row["article_id"] for row in read(CHUNKS)}

    def aid(law, number):
        return articles[(law, number)]

    labor = lambda number: aid("근로기준법", number)
    fixed = aid("기간제 및 단시간근로자 보호 등에 관한 법률", "제17조")
    workplace = aid("근로기준법 시행령", "제8조")
    rows = []
    for clause in clauses:
        contract = contracts[clause["contract_id"]]
        size = contract["workplace_size"]
        kind = contract["employment_type"]
        direct, support = [], []
        issue = issues.get(clause["clause_id"])
        if issue:
            code = issue["issue_code"]
            review = reviews[issue["issue_id"]]
            if review["verdict"] in {"DIRECT", "CONDITIONAL"} and code not in {"UT01", "UT04"}:
                direct = list(queries[issue["issue_id"]]["relevant_article_ids"])
            if code == "LV06" and size < 5:
                support, direct = direct, []
            if code == "LV10" and size < 5:
                direct.remove(labor("제27조"))
            if code == "UT08":
                direct = [labor("제23조")] if size >= 5 else []
                support = [labor("제17조")]
            elif code == "UT01":
                support = [labor("제4조"), labor("제17조")]
            elif code == "UT04":
                support = [labor("제17조")]
            elif code == "UT03":
                support = [labor("제20조")]
            elif code == "UT09":
                support = [labor("제23조")]
            elif code == "UT10":
                support = [labor("제7조")]
            basis = f"issue {code}; 2026-11-13 first-pass review"
        else:
            category = clause["clause_categories"][0]
            if category in {"WORKPLACE", "JOB_DUTIES"}:
                direct = [workplace] + ([fixed] if kind != "정규직" else [])
                support = [labor("제17조")]
            elif category == "CONTRACT_PERIOD":
                direct = [fixed] if kind != "정규직" else []
            elif category == "WORK_HOURS":
                direct = [labor("제17조")] + ([labor("제50조")] if size >= 5 else [])
                if kind != "정규직":
                    direct.append(fixed)
            elif category == "BREAK":
                direct = [labor("제54조")]
                support = [fixed] if kind != "정규직" else []
            elif category == "HOLIDAY":
                direct = [labor("제55조")]
                support = [labor("제17조")] + ([fixed] if kind != "정규직" else [])
            elif category == "WAGE":
                direct = [labor("제17조"), labor("제43조")]
                if kind != "정규직":
                    direct.append(fixed)
                support = [labor("제48조")]
            elif category == "LEAVE":
                direct = [labor("제60조")] if size >= 5 else []
                support = [labor("제17조")] + ([fixed] if kind != "정규직" else [])
            elif category == "TERMINATION":
                direct = [labor("제26조")] + ([labor("제27조")] if size >= 5 else [])
            elif category == "OTHER":
                support = [labor("제17조")]
            basis = f"normal clause category {category}; first-pass source-content review"
        direct = list(dict.fromkeys(direct))
        support = [item for item in dict.fromkeys(support) if item not in direct]
        rows.append({
            "contract_id": clause["contract_id"], "clause_id": clause["clause_id"],
            "direct_article_ids": direct, "support_article_ids": support,
            "annotation_status": "PROVISIONAL_ENGINEERING_REVIEW",
            "basis": basis, "corpus_gap_or_no_direct_target": not direct,
        })
    assert len(rows) == 478 and len({row["clause_id"] for row in rows}) == 478
    assert all(set(row["direct_article_ids"] + row["support_article_ids"]) <= set(articles.values()) for row in rows)
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print({"clauses": len(rows), "direct_target": sum(bool(row["direct_article_ids"]) for row in rows),
           "support_only": sum(not row["direct_article_ids"] and bool(row["support_article_ids"]) for row in rows),
           "no_target": sum(not row["direct_article_ids"] and not row["support_article_ids"] for row in rows)})


if __name__ == "__main__":
    main()
