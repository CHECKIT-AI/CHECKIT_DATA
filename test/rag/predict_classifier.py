"""Run the pinned CheckIt_BE production classifier on evaluation clauses."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from app.ai_contracts import CLAUSE_LABELS
from app.classifier import HuggingFaceClassifier
from app.config import Settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({row["query_id"] for row in rows}) != len(rows):
        raise ValueError("평가 데이터가 비었거나 query_id가 중복되었습니다.")

    classifier = HuggingFaceClassifier(Settings(_env_file=None, classifier_device=args.device))
    predictions = []
    elapsed_ms = 0.0
    for index, row in enumerate(rows, 1):
        started = perf_counter()
        result = classifier.predict(row["query"])
        elapsed_ms += (perf_counter() - started) * 1000
        predictions.append({
            "query_id": row["query_id"], "predicted_labels": result["labels"],
            "classifier_scores": result["scores"],
        })
        if index % 25 == 0:
            print(f"분류 {index}/{len(rows)}", flush=True)

    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    args.metadata.write_text(json.dumps({
        "model": {"model_id": classifier.model_id, "revision": classifier.revision,
                  "max_length": classifier.max_length},
        "labels": list(CLAUSE_LABELS),
        "thresholds": dict(zip(CLAUSE_LABELS, classifier.thresholds)),
        "total_query_ms": elapsed_ms,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
