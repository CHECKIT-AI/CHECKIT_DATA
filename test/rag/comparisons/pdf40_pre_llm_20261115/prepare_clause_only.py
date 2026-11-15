"""Remove the synthetic PDF footer and rerun changed clauses through the BE classifier."""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT.parent / "CheckIt_BE"))

from app.analysis_rules import candidates  # noqa: E402
from app.classifier import HuggingFaceClassifier  # noqa: E402
from app.config import Settings  # noqa: E402


SOURCE = ROOT / "outputs/rag/pdf40_pre_llm_20261115/segments_prospective.jsonl"
OUTPUT = ROOT / "outputs/rag/pdf40_pre_llm_20261115/segments_clause_only.jsonl"
CANDIDATES = ROOT / "outputs/rag/pdf40_pre_llm_20261115/candidates_clause_only.jsonl"
CONTRACTS = ROOT.parent / "CheckIt_Eval_Benchmark/ground_truth/contracts.jsonl"
FOOTER = re.compile(r"\n본 문서는 가상 정보로 생성된 AI 평가용 합성 계약서입니다\. \d+ / \d+\s*$")


def read(path):
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def write(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main():
    rows = read(SOURCE)
    assert len(rows) == 478
    changed = []
    for row in rows:
        text, count = FOOTER.subn("", row["text"])
        if count:
            row["text"] = text.rstrip()
            changed.append(row)
    assert len(changed) == len({row["contract_id"] for row in rows}) == 40
    assert not any("본 문서는 가상 정보로 생성된 AI 평가용 합성 계약서입니다." in row["text"] for row in rows)
    classifier = HuggingFaceClassifier(Settings(classifier_device="cpu"))
    for row in changed:
        row["predicted_labels"] = classifier.predict(row["text"])["labels"]
    write(OUTPUT, rows)

    contracts = {row["contract_id"]: row for row in read(CONTRACTS)}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["contract_id"]].append(row)
    found = []
    for contract_id, clauses in grouped.items():
        contract = contracts[contract_id]
        inputs = [{"clause_id": row["clause_id"], "page": row["page"],
                   "ordinal_on_page": int(row["clause_id"].split(":c")[-1]) - 1,
                   "text": row["text"]} for row in clauses]
        classifications = [{"labels": row["predicted_labels"]} for row in clauses]
        # Test-only effective date is later than the current law snapshot.
        context = {"effective_date": "2026-11-15", "employee_count": contract["workplace_size"]}
        found.extend({"contract_id": contract_id, **item} for item in candidates(inputs, classifications, context))
    write(CANDIDATES, found)
    print({"clauses": len(rows), "footer_removed": len(changed), "candidates": len(found)})


if __name__ == "__main__":
    main()
