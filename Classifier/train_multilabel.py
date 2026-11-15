"""Train a new classifier with generated multi-topic clauses and held-out selection."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from Classifier.train import ROOT, read_jsonl, sha256
from Classifier.train_core import BASE_MODEL, BASE_REVISION, evaluate_rows, train_one


BASE_SPLIT = ROOT / "outputs/classifier/combined_split_v004"
MULTI_SPLIT = ROOT / "data/classifier/multilabel_aug_20261115"
FROZEN = ROOT / "test/classifier/novel_clause_60_20261115/gold.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs는 1 이상이어야 합니다.")
    labels = json.loads((BASE_SPLIT / "manifest.json").read_text(encoding="utf-8"))["labels"]
    multi_manifest = json.loads((MULTI_SPLIT / "manifest.json").read_text(encoding="utf-8"))
    base_manifest = json.loads((BASE_SPLIT / "manifest.json").read_text(encoding="utf-8"))
    base = {name: read_jsonl(BASE_SPLIT / f"{name}.jsonl") for name in ("train", "validation", "test")}
    multi = {name: read_jsonl(MULTI_SPLIT / f"{name}.jsonl") for name in ("train", "validation", "test")}
    frozen = read_jsonl(FROZEN)
    assert len(labels) == 10 and all(
        row["label_vec"] == [int(label in row["labels"]) for label in labels]
        for rows in multi.values() for row in rows
    )
    assert all(sha256(BASE_SPLIT / f"{name}.jsonl") == base_manifest["output_sha256"][name]
               for name in base)
    assert all(sha256(MULTI_SPLIT / f"{name}.jsonl") == multi_manifest["sha256"][name]
               for name in multi)
    assert len(base["train"]) == 3710 and len(multi["train"]) == 552
    texts = [set(row["text"] for row in rows) for rows in
             (base["train"], base["validation"], base["test"],
              multi["train"], multi["validation"], multi["test"], frozen)]
    assert all(not texts[i] & texts[j] for i in range(len(texts))
               for j in range(i + 1, len(texts)))
    device = torch.device(args.device)
    output = ROOT / "outputs/classifier" / datetime.now(timezone.utc).strftime(
        "multilabel_run_%Y%m%dT%H%M%S%fZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION, trust_remote_code=False)
    result = train_one(
        "core_plus_multilabel", base["train"] + multi["train"], multi["validation"],
        multi["test"], labels, tokenizer, output, device, args.epochs, 2e-5,
    )
    model_dir = output / "core_plus_multilabel_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(device)
    thresholds = {label: 0.5 for label in labels}
    checks = {
        "core_validation": evaluate_rows(model, tokenizer, base["validation"], labels, thresholds, device),
        "core_test": evaluate_rows(model, tokenizer, base["test"], labels, thresholds, device),
        "frozen_60": evaluate_rows(model, tokenizer, frozen, labels, thresholds, device),
        "frozen_60_multi": evaluate_rows(
            model, tokenizer, [row for row in frozen if len(row["labels"]) > 1],
            labels, thresholds, device,
        ),
    }
    report = {
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "base_split_manifest_sha256": sha256(BASE_SPLIT / "manifest.json"),
        "multi_manifest_sha256": sha256(MULTI_SPLIT / "manifest.json"),
        "frozen_60_sha256": sha256(FROZEN),
        "settings": {"epochs": args.epochs, "learning_rate": 2e-5, "batch_size": 16,
                     "threshold": 0.5, "seed": 42, "selection": "multi validation macro F1"},
        "train_count": len(base["train"]) + len(multi["train"]),
        "multi_validation_count": len(multi["validation"]),
        "multi_test_count": len(multi["test"]),
        "training": result, "checks": checks,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "selected_epoch": result["selected_epoch"],
        "multi_test_exact": result["test"]["exact_match"],
        "core_test_exact": checks["core_test"]["exact_match"],
        "frozen_60_exact": checks["frozen_60"]["exact_match"],
        "frozen_60_multi_exact": checks["frozen_60_multi"]["exact_match"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
