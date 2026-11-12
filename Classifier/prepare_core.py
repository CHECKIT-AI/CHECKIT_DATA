"""Make a document-group split from clean core clauses and synthetic train rows."""

import argparse
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

from Classifier.train import ROOT, TRAIN, MANIFEST, sha256, read_jsonl


CORE = ROOT / "data/classifier/processed_core_clean"
DEFAULT_OUTPUT = ROOT / "outputs/classifier/combined_split_v004"
SEED = 705  # Chosen before training for group diversity and rare-label coverage.


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def prepare(core, output):
    labels = json.loads((core / "labels.json").read_text(encoding="utf-8"))
    synthetic_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if labels != synthetic_manifest["labels"] or sha256(TRAIN) != synthetic_manifest["sha256"]:
        raise ValueError("core와 합성 자료의 라벨 순서 또는 학습 파일 해시가 다릅니다.")
    original = read_jsonl(core / "train.jsonl") + read_jsonl(core / "test.jsonl")
    synthetic = read_jsonl(TRAIN)
    if not original or len(synthetic) != synthetic_manifest["row_count"]:
        raise ValueError("입력 데이터가 비었거나 합성 행 수가 manifest와 다릅니다.")
    clean = []
    for row in original:
        if row.get("conflict"):
            continue
        if (not row.get("group") or not isinstance(row.get("text"), str)
                or not normalized(row["text"]) or not isinstance(row.get("labels"), list)
                or not row["labels"] or len(row["labels"]) != len(set(row["labels"]))
                or row["label_vec"] != [int(label in row["labels"]) for label in labels]):
            raise ValueError(f"잘못된 core 행: {row.get('id')}")
        clean.append(row)
    for row in synthetic:
        if (not isinstance(row.get("text"), str) or not normalized(row["text"])
                or not row.get("labels")
                or row["label_vec"] != [int(label in row["labels"]) for label in labels]):
            raise ValueError(f"잘못된 합성 행: {row.get('id')}")

    group_sizes = Counter(row["group"] for row in clean)
    groups = sorted(group_sizes)
    random.Random(SEED).shuffle(groups)
    targets = {"test": round(len(clean) * .10), "validation": round(len(clean) * .10)}
    assigned = {}
    counts = Counter()
    for split in ("test", "validation"):
        for group in groups:
            if group not in assigned and counts[split] + group_sizes[group] <= targets[split]:
                assigned[group] = split
                counts[split] += group_sizes[group]
    for group in groups:
        assigned.setdefault(group, "train")

    # Prefer held-out rows when the same boilerplate text occurs in several groups.
    by_split = {name: [] for name in ("train", "validation", "test")}
    seen = set()
    dropped_duplicates = Counter()
    for split in ("test", "validation", "train"):
        for row in clean:
            if assigned[row["group"]] != split:
                continue
            key = normalized(row["text"])
            if key in seen:
                dropped_duplicates[split] += 1
                continue
            seen.add(key)
            by_split[split].append({
                "id": row["id"], "text": row["text"], "labels": row["labels"],
                "label_vec": row["label_vec"], "group": row["group"], "source": "core",
            })
    for row in synthetic:
        key = normalized(row["text"])
        if key in seen:
            dropped_duplicates["synthetic"] += 1
            continue
        seen.add(key)
        by_split["train"].append({
            "id": row["id"], "text": row["text"], "labels": row["labels"],
            "label_vec": row["label_vec"], "group": "synthetic", "source": "synthetic",
        })
    if not all(by_split.values()):
        raise ValueError("분할 중 비어 있는 집합이 있습니다.")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for split, rows in by_split.items():
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "seed": SEED, "target": "core 80/10/10 by group; synthetic train only",
        "labels": labels,
        "source_sha256": {name: sha256(core / name) for name in
                          ("train.jsonl", "test.jsonl", "labels.json")},
        "synthetic_sha256": sha256(TRAIN),
        "excluded_conflict_rows": len(original) - len(clean),
        "dropped_duplicate_texts": dict(dropped_duplicates),
        "counts": {split: {
            "total": len(rows),
            "core": sum(row["source"] == "core" for row in rows),
            "synthetic": sum(row["source"] == "synthetic" for row in rows),
            "groups": len({row["group"] for row in rows if row["source"] == "core"}),
        } for split, rows in by_split.items()},
        "output_sha256": {split: sha256(output / f"{split}.jsonl") for split in by_split},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, default=CORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.core, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
