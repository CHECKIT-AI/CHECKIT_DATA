"""Fine-tune the pinned CheckIt clause classifier on synthetic train rows."""

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/classifier/train_synthetic_contract_2026_v001.jsonl"
MANIFEST = TRAIN.with_suffix(".manifest.json")
DEV = ROOT / "test/rag/datasets/retrieval_contract_2026_dev_v001.jsonl"
MODEL_ID = "rladmsgki/checkit-contract-classifier"
REVISION = "aedfc58fa73fe5bc58a4e66a0cadaeef091fe59e"
MAX_LENGTH = 256


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_data(labels):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if sha256(TRAIN) != manifest["sha256"] or manifest["labels"] != labels:
        raise ValueError("학습 파일 해시 또는 라벨 순서가 manifest와 다릅니다.")
    rows = read_jsonl(TRAIN)
    dev = read_jsonl(DEV)
    if len(rows) != manifest["row_count"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("학습 행 수 또는 ID가 잘못되었습니다.")
    if not dev or len({row["query_id"] for row in dev}) != len(dev):
        raise ValueError("개발 평가셋이 비었거나 ID가 중복되었습니다.")
    for row in rows:
        expected = [int(label in row["labels"]) for label in labels]
        if (not isinstance(row.get("text"), str) or not row["text"].strip()
                or not row["labels"] or len(row["labels"]) != len(set(row["labels"]))
                or row["label_vec"] != expected):
            raise ValueError(f"잘못된 학습 라벨: {row['id']}")
    for row in dev:
        if (row.get("split") != "dev" or not isinstance(row.get("query"), str)
                or not row["query"].strip() or not row.get("model_labels")
                or not set(row["model_labels"]).issubset(labels)):
            raise ValueError(f"잘못된 개발 평가 행: {row.get('query_id')}")
    if {row["text"] for row in rows} & {row["query"] for row in dev}:
        raise ValueError("학습·개발 평가 문장이 겹칩니다.")
    return rows, dev


def batches(rows, batch_size):
    for start in range(0, len(rows), batch_size):
        yield rows[start:start + batch_size]


def encode(tokenizer, texts, device):
    return {key: value.to(device) for key, value in tokenizer(
        texts, padding=True, truncation=True, max_length=MAX_LENGTH,
        return_tensors="pt",
    ).items()}


def evaluate(model, tokenizer, rows, labels, thresholds, device):
    model.eval()
    tp = fp = fn = exact = 0
    per_label = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    with torch.no_grad():
        for batch in batches(rows, 8):
            probabilities = torch.sigmoid(model(**encode(
                tokenizer, [row["query"] for row in batch], device,
            )).logits).cpu().tolist()
            for row, scores in zip(batch, probabilities):
                predicted = {label for label, score in zip(labels, scores)
                             if score >= thresholds[label]}
                gold = set(row["model_labels"])
                exact += predicted == gold
                for label in labels:
                    counts = per_label[label]
                    counts["tp"] += label in predicted and label in gold
                    counts["fp"] += label in predicted and label not in gold
                    counts["fn"] += label not in predicted and label in gold
    for counts in per_label.values():
        tp += counts["tp"]
        fp += counts["fp"]
        fn += counts["fn"]

    def f1(a, b, c):
        return 2 * a / (2 * a + b + c) if 2 * a + b + c else 0.0

    return {"count": len(rows), "micro_f1": f1(tp, fp, fn),
            "macro_f1": sum(f1(counts["tp"], counts["fp"], counts["fn"])
                            for counts in per_label.values()) / len(labels),
            "exact_match": exact / len(rows),
            "per_label": {label: {**counts, "f1": f1(counts["tp"], counts["fp"], counts["fn"])}
                          for label, counts in per_label.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS를 사용할 수 없습니다.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA를 사용할 수 없습니다.")
    device = torch.device(args.device)
    torch.manual_seed(42)
    random.seed(42)
    label_path = hf_hub_download(MODEL_ID, "labels.json", revision=REVISION)
    label_data = json.loads(Path(label_path).read_text(encoding="utf-8"))
    labels = label_data["labels"]
    thresholds = label_data["thresholds"]
    if len(labels) != 10 or set(thresholds) != set(labels):
        raise ValueError("기존 모델의 라벨 계약이 잘못되었습니다.")
    train, dev = load_data(labels)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=REVISION)
    if model.config.num_labels != len(labels):
        raise ValueError("모델 출력 크기와 라벨 수가 다릅니다.")
    model.to(device)
    before = evaluate(model, tokenizer, dev, labels, thresholds, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    random.shuffle(train)
    model.train()
    losses = []
    for index, batch in enumerate(batches(train, 8), 1):
        targets = torch.tensor([row["label_vec"] for row in batch], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(**encode(tokenizer, [row["text"] for row in batch], device)).logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if index % 10 == 0:
            print(f"학습 {index}/30 batches", flush=True)
    after = evaluate(model, tokenizer, dev, labels, thresholds, device)
    output = ROOT / "outputs/classifier" / datetime.now(timezone.utc).strftime("train_%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output / "model")
    tokenizer.save_pretrained(output / "model")
    (output / "model/labels.json").write_text(json.dumps(label_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "base_model": MODEL_ID, "base_revision": REVISION,
        "training": {"rows": len(train), "epochs": 1, "batch_size": 8,
                     "learning_rate": 2e-5, "seed": 42, "mean_loss": sum(losses) / len(losses),
                     "dataset_sha256": sha256(TRAIN), "annotation_status": "DRAFT_UNREVIEWED"},
        "development_evaluation": {"dataset_sha256": sha256(DEV), "before": before, "after": after,
                                   "note": "합성·미검수 개발셋. 잠금 테스트셋 미사용."},
        "device": args.device, "torch": torch.__version__,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "before": before["micro_f1"],
                      "after": after["micro_f1"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
