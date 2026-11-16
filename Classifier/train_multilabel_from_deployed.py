"""Conservatively fine-tune the pinned deployed classifier on mixed clauses."""

import argparse
import json
from datetime import datetime, timezone

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from Classifier.train import ROOT, read_jsonl, sha256
from Classifier.train_core import evaluate_rows, train_one


DEPLOYED_MODEL = "hur03/checkit-contract-classifier-core-synthetic-3epoch-v2"
DEPLOYED_REVISION = "1accf527c060396c5801b044e2cebf8bedb5ac12"
BASE = ROOT / "outputs/classifier/combined_split_v004"
PAIRED = ROOT / "data/classifier/multilabel_aug_20261115"
INTEGRATED = ROOT / "data/classifier/integrated_multilabel_20261115_v001"
HARD = ROOT / "data/classifier/multilabel_hard_20261116_v001"
FROZEN = ROOT / "test/classifier/novel_clause_60_20261115/gold.jsonl"
PDF = ROOT / "outputs/rag/pdf40_pre_llm_20261115/segments_clause_only.jsonl"
PDF_GOLD = ROOT.parent / "CheckIt_Eval_Benchmark_v1.0.3/ground_truth/clauses.jsonl"


def verified(directory, key):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows = {split: read_jsonl(directory / f"{split}.jsonl")
            for split in ("train", "validation", "test")}
    assert all(sha256(directory / f"{split}.jsonl") == manifest[key][split] for split in rows)
    return manifest, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard", action="store_true", help="include the hard multi-topic split")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    args = parser.parse_args()
    if not 0 < args.learning_rate < 1:
        parser.error("learning-rate는 0과 1 사이여야 합니다.")
    base_manifest, base = verified(BASE, "output_sha256")
    paired_manifest, paired = verified(PAIRED, "sha256")
    integrated_manifest, integrated = verified(INTEGRATED, "sha256")
    _, hard = verified(HARD, "sha256") if args.hard else (None, None)
    labels = base_manifest["labels"]
    assert len(labels) == 10
    frozen = read_jsonl(FROZEN)
    pdf_gold = {row["clause_id"]: row["model_labels"] for row in read_jsonl(PDF_GOLD)}
    pdf = [{"text": row["text"], "labels": pdf_gold[row["gold_clause_id"]]}
           for row in read_jsonl(PDF)]
    all_sets = [set(row["text"] for row in rows) for rows in (
        *base.values(), *paired.values(), *integrated.values(),
        *(hard.values() if hard else ()), frozen, pdf
    )]
    assert all(not all_sets[i] & all_sets[j] for i in range(len(all_sets))
               for j in range(i + 1, len(all_sets)))
    train_rows = base["train"] + paired["train"] + integrated["train"] + (hard["train"] if hard else [])
    assert len(train_rows) == (4796 if hard else 4412)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    output = ROOT / "outputs/classifier" / datetime.now(timezone.utc).strftime(
        ("multilabel_hard_run_" if hard else "multilabel_integrated_run_") + "%Y%m%dT%H%M%S%fZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(
        DEPLOYED_MODEL, revision=DEPLOYED_REVISION, trust_remote_code=False,
    )
    result = train_one(
        "from_deployed", train_rows,
        hard["validation"] if hard else integrated["validation"],
        hard["test"] if hard else integrated["test"],
        labels, tokenizer, output, device, epochs=1, learning_rate=args.learning_rate,
        model_name=DEPLOYED_MODEL, model_revision=DEPLOYED_REVISION,
    )
    model_dir = output / "from_deployed_model"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
    ).to(device)
    thresholds = {label: 0.5 for label in labels}
    checks = {
        "core_validation": evaluate_rows(model, tokenizer, base["validation"], labels, thresholds, device),
        "core_test": evaluate_rows(model, tokenizer, base["test"], labels, thresholds, device),
        "paired_test": evaluate_rows(model, tokenizer, paired["test"], labels, thresholds, device),
        "integrated_test": evaluate_rows(model, tokenizer, integrated["test"], labels, thresholds, device),
        "frozen_60": evaluate_rows(model, tokenizer, frozen, labels, thresholds, device),
        "frozen_60_multi": evaluate_rows(
            model, tokenizer, [row for row in frozen if len(row["labels"]) > 1],
            labels, thresholds, device,
        ),
        "pdf_40": evaluate_rows(model, tokenizer, pdf, labels, thresholds, device),
        "pdf_40_multi": evaluate_rows(
            model, tokenizer, [row for row in pdf if len(row["labels"]) > 1],
            labels, thresholds, device,
        ),
    }
    report = {
        "parent_model": DEPLOYED_MODEL, "parent_revision": DEPLOYED_REVISION,
        "base_manifest_sha256": sha256(BASE / "manifest.json"),
        "paired_manifest_sha256": sha256(PAIRED / "manifest.json"),
        "integrated_manifest_sha256": sha256(INTEGRATED / "manifest.json"),
        "hard_manifest_sha256": sha256(HARD / "manifest.json") if hard else None,
        "frozen_60_sha256": sha256(FROZEN),
        "pdf_gold_sha256": sha256(PDF_GOLD),
        "settings": {"epochs": 1, "learning_rate": args.learning_rate, "batch_size": 16,
                     "threshold": 0.5, "seed": 42,
                     "selection": "hard validation macro F1" if hard else "integrated validation macro F1"},
        "train_count": len(train_rows), "training": result, "checks": checks,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "selected_test_exact": result["test"]["exact_match"],
        "core_test_exact": checks["core_test"]["exact_match"],
        "frozen_60_exact": checks["frozen_60"]["exact_match"],
        "frozen_60_multi_exact": checks["frozen_60_multi"]["exact_match"],
        "pdf_40_exact": checks["pdf_40"]["exact_match"],
        "pdf_40_multi_exact": checks["pdf_40_multi"]["exact_match"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
