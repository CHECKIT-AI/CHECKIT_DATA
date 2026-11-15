"""Evaluate frozen, newly authored clauses with the deployed BE classifier."""

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT.parent / "CheckIt_BE"))

from app.ai_contracts import CLAUSE_LABELS  # noqa: E402
from app.classifier import HuggingFaceClassifier  # noqa: E402
from app.config import Settings  # noqa: E402


DATA = Path(__file__).with_name("gold.jsonl")
PREDICTIONS = DATA.with_name("predictions.jsonl")
SUMMARY = DATA.with_name("summary.json")
GOLD_SHA256 = "f0a6e12eb35e0ac964db04094d1e3958284a93e7ce8fcd975b9420fcb94d8f41"


def score(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def main():
    assert hashlib.sha256(DATA.read_bytes()).hexdigest() == GOLD_SHA256, "gold changed after freezing"
    rows = [json.loads(line) for line in DATA.open(encoding="utf-8")]
    assert len(rows) == 60 and len({row["id"] for row in rows}) == 60
    assert len({row["text"] for row in rows}) == 60
    assert all(row["labels"] and set(row["labels"]) <= set(CLAUSE_LABELS) for row in rows)

    classifier = HuggingFaceClassifier(Settings(classifier_device="cpu"))
    results = []
    per_label = {label: {"tp": 0, "fp": 0, "fn": 0} for label in CLAUSE_LABELS}
    for row in rows:
        predicted = classifier.predict(row["text"])
        gold, actual = set(row["labels"]), set(predicted["labels"])
        results.append({
            "id": row["id"], "text": row["text"], "gold_labels": row["labels"],
            "predicted_labels": predicted["labels"], "scores": predicted["scores"],
            "exact_match": gold == actual,
        })
        for label in CLAUSE_LABELS:
            counts = per_label[label]
            counts["tp"] += label in gold and label in actual
            counts["fp"] += label not in gold and label in actual
            counts["fn"] += label in gold and label not in actual
    PREDICTIONS.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    tp, fp, fn = (sum(row[key] for row in per_label.values()) for key in ("tp", "fp", "fn"))
    summary = {
        "status": "AUTHORED_PROVISIONAL_GOLD", "gold_sha256": GOLD_SHA256,
        "model_id": classifier.model_id, "model_revision": classifier.revision,
        "count": len(rows), "exact_match": sum(row["exact_match"] for row in results),
        "micro_precision": tp / (tp + fp), "micro_recall": tp / (tp + fn),
        "micro_f1": score(tp, fp, fn),
        "macro_f1": sum(score(**counts) for counts in per_label.values()) / len(per_label),
        "single_label": {
            "count": sum(len(row["gold_labels"]) == 1 for row in results),
            "exact_match": sum(row["exact_match"] for row in results if len(row["gold_labels"]) == 1),
        },
        "multi_label": {
            "count": sum(len(row["gold_labels"]) > 1 for row in results),
            "exact_match": sum(row["exact_match"] for row in results if len(row["gold_labels"]) > 1),
        },
        "per_label": {
            label: {**counts, "f1": score(**counts)} for label, counts in per_label.items()
        },
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_label"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
