"""Compare a classifier on core, generated multi-topic, and frozen clause sets."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from Classifier.train import ROOT, read_jsonl
from Classifier.train_core import evaluate_rows


BASE_SPLIT = ROOT / "outputs/classifier/combined_split_v004"
MULTI_SPLIT = ROOT / "data/classifier/multilabel_aug_20261115"
FROZEN = ROOT / "test/classifier/novel_clause_60_20261115/gold.jsonl"
PDF_CLAUSES = ROOT / "outputs/rag/pdf40_pre_llm_20261115/segments_clause_only.jsonl"
PDF_GOLD = ROOT.parent / "CheckIt_Eval_Benchmark/ground_truth/clauses.jsonl"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("기존 평가 결과를 덮어쓰지 않습니다.")
    model_args = {"revision": args.revision} if args.revision else {"local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False, **model_args)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, trust_remote_code=False, **model_args,
    ).to(args.device)
    labels = json.loads((BASE_SPLIT / "manifest.json").read_text(encoding="utf-8"))["labels"]
    assert model.config.num_labels == len(labels) == 10
    thresholds = {label: 0.5 for label in labels}
    datasets = {
        "core_test": read_jsonl(BASE_SPLIT / "test.jsonl"),
        "multi_validation": read_jsonl(MULTI_SPLIT / "validation.jsonl"),
        "multi_test": read_jsonl(MULTI_SPLIT / "test.jsonl"),
        "frozen_60": read_jsonl(FROZEN),
    }
    pdf_gold = {row["clause_id"]: row["model_labels"] for row in read_jsonl(PDF_GOLD)}
    datasets["pdf_40"] = [{"text": row["text"], "labels": pdf_gold[row["gold_clause_id"]]}
                          for row in read_jsonl(PDF_CLAUSES)]
    results = {}
    for name, rows in datasets.items():
        results[name] = {
            "count": len(rows),
            "all": evaluate_rows(model, tokenizer, rows, labels, thresholds, torch.device(args.device)),
            "single": evaluate_rows(model, tokenizer, [row for row in rows if len(row["labels"]) == 1],
                                    labels, thresholds, torch.device(args.device))
            if any(len(row["labels"]) == 1 for row in rows) else None,
            "multi": evaluate_rows(model, tokenizer, [row for row in rows if len(row["labels"]) > 1],
                                   labels, thresholds, torch.device(args.device))
            if any(len(row["labels"]) > 1 for row in rows) else None,
        }
    report = {"model": args.model, "revision": args.revision, "threshold": 0.5, "datasets": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {
        "count": score["count"], "exact": score["all"]["exact_match"],
        "micro_f1": score["all"]["micro_f1"],
        "multi_exact": score["multi"]["exact_match"] if score["multi"] else None,
    } for name, score in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
