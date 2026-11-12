"""Compare core-only and core-plus-synthetic training on one held-out split."""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from Classifier.train import ROOT, batches, encode, evaluate, read_jsonl, sha256


BASE_MODEL = "klue/roberta-base"
BASE_REVISION = "02f94ba5e3fcb7e2a58a390b8639b0fac974a8da"
SPLIT = ROOT / "outputs/classifier/combined_split_v004"


def evaluate_rows(model, tokenizer, rows, labels, thresholds, device):
    return evaluate(model, tokenizer,
                    [{"query": row["text"], "model_labels": row["labels"]} for row in rows],
                    labels, thresholds, device)


def train_one(name, train_rows, validation, test, labels, tokenizer, output, device,
              epochs, learning_rate):
    torch.manual_seed(42)
    random.seed(42)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, num_labels=len(labels),
        problem_type="multi_label_classification", trust_remote_code=False,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    thresholds = {label: 0.5 for label in labels}
    best = None
    history = []
    model_dir = output / f"{name}_model"
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = train_rows[:]
        random.shuffle(shuffled)
        losses = []
        for index, batch in enumerate(batches(shuffled, 16), 1):
            targets = torch.tensor([row["label_vec"] for row in batch],
                                   dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(**encode(
                tokenizer, [row["text"] for row in batch], device,
            )).logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if index % 50 == 0:
                print(f"{name}: epoch {epoch}, batch {index}", flush=True)
        scores = evaluate_rows(model, tokenizer, validation, labels, thresholds, device)
        record = {"epoch": epoch, "mean_train_loss": sum(losses) / len(losses),
                  "validation": scores}
        history.append(record)
        print(f"{name}: epoch {epoch}, validation macro F1={scores['macro_f1']:.3f}", flush=True)
        if best is None or scores["macro_f1"] > best["validation"]["macro_f1"]:
            best = record
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
    del model, optimizer
    if device.type == "mps":
        torch.mps.empty_cache()
    selected = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False,
    ).to(device)
    result = {"train_count": len(train_rows), "history": history,
              "selected_epoch": best["epoch"], "validation": best["validation"],
              "test": evaluate_rows(selected, tokenizer, test, labels, thresholds, device)}
    del selected
    (model_dir / "labels.json").write_text(
        json.dumps({"labels": labels, "thresholds": thresholds},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-set", choices=("both", "core_plus_synthetic"), default="both")
    args = parser.parse_args()
    if args.epochs < 1 or not 0 < args.learning_rate < 1:
        parser.error("epochs는 1 이상, learning-rate는 0과 1 사이여야 합니다.")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS를 사용할 수 없습니다.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA를 사용할 수 없습니다.")
    manifest = json.loads((args.split / "manifest.json").read_text(encoding="utf-8"))
    rows = {name: read_jsonl(args.split / f"{name}.jsonl")
            for name in ("train", "validation", "test")}
    for name, items in rows.items():
        if sha256(args.split / f"{name}.jsonl") != manifest["output_sha256"][name] or not items:
            raise ValueError(f"{name} 분할 파일의 해시 또는 행 수가 잘못되었습니다.")
    labels = manifest["labels"]
    if len(labels) != 10:
        raise ValueError("10개 라벨 순서가 없습니다.")
    texts = [{row["text"] for row in rows[name]} for name in rows]
    groups = [{row["group"] for row in rows[name] if row["source"] == "core"} for name in rows]
    if any(texts[i] & texts[j] or groups[i] & groups[j]
           for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("분할 간 문장 또는 문서 묶음이 겹칩니다.")
    if any(row["source"] != "core" for name in ("validation", "test") for row in rows[name]):
        raise ValueError("합성 문장이 평가셋에 들어갔습니다.")
    device = torch.device(args.device)
    output = ROOT / "outputs/classifier" / datetime.now(timezone.utc).strftime(
        "core_comparison_%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, trust_remote_code=False)
    core = [row for row in rows["train"] if row["source"] == "core"]
    results = {}
    if args.train_set == "both":
        results["core_only"] = train_one(
            "core_only", core, rows["validation"], rows["test"], labels,
            tokenizer, output, device, args.epochs, args.learning_rate)
    results["core_plus_synthetic"] = train_one(
        "core_plus_synthetic", rows["train"], rows["validation"], rows["test"],
        labels, tokenizer, output, device, args.epochs, args.learning_rate)
    report = {
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "split_manifest_sha256": sha256(args.split / "manifest.json"),
        "split_counts": manifest["counts"], "device": args.device,
        "settings": {"epochs": args.epochs, "batch_size": 16, "learning_rate": args.learning_rate,
                     "loss": "BCEWithLogitsLoss", "threshold": 0.5, "seed": 42,
                     "selection": "highest validation macro F1"},
        "results": results,
        "interpretation": "Core test is held out by document group and exact text. "
                          "Synthetic labels are unreviewed; this is not real-world accuracy.",
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "test": {
        name: {key: value["test"][key] for key in ("micro_f1", "macro_f1", "exact_match")}
        for name, value in results.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
