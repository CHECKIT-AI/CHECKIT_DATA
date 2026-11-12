"""Evaluate the BE clause classifier and RAG methods on one labeled dataset."""

import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

from RAG.evaluation.retrieval_metrics import score_query
from RAG.retrieval.retriever import LawRetriever
from RAG.settings import PROJECT_ROOT, STORAGE_ROOT


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path, split):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({row.get("query_id") for row in rows}) != len(rows):
        raise ValueError("평가 데이터가 비었거나 query_id가 중복되었습니다.")
    for row in rows:
        if (not isinstance(row.get("query"), str) or not row["query"].strip()
                or row.get("split") != split or not row.get("relevant_article_ids")
                or row.get("primary_article_ids") != row.get("relevant_article_ids")):
            raise ValueError(f"{row.get('query_id')}의 검색 입력, split 또는 정답 조문을 확인하세요.")
        labels = row.get("model_labels")
        if not isinstance(labels, list) or not labels or len(labels) != len(set(labels)):
            raise ValueError(f"{row['query_id']}의 분류 정답 라벨을 확인하세요.")
    if split == "test" and any(
        row.get("annotation_status") != "REVIEWED"
        or row.get("label_annotation_status") != "REVIEWED"
        for row in rows
    ):
        raise ValueError("test 평가는 법령 정답과 분류 라벨의 독립 검수 완료 후 실행하세요.")
    return rows


def classification_metrics(rows, labels):
    counts = {label: [0, 0, 0] for label in labels}
    exact = 0
    for row in rows:
        truth, predicted = set(row["model_labels"]), set(row["predicted_labels"])
        exact += truth == predicted
        for label in labels:
            counts[label][0] += label in truth and label in predicted
            counts[label][1] += label not in truth and label in predicted
            counts[label][2] += label in truth and label not in predicted
    tp, fp, fn = (sum(x[i] for x in counts.values()) for i in range(3))

    def f1(a, b, c):
        precision = a / (a + b) if a + b else 0.0
        recall = a / (a + c) if a + c else 0.0
        return {"precision": precision, "recall": recall,
                "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                "support": a + c}

    per_label = {label: f1(*values) for label, values in counts.items()}
    return {
        "query_count": len(rows), "micro": f1(tp, fp, fn),
        "macro_f1": sum(x["f1"] for x in per_label.values()) / len(labels),
        "exact_match": exact / len(rows), "per_label": per_label,
    }


def retrieval_metrics(rows, method):
    metrics = [row["retrieval"][method]["metrics"] for row in rows]
    summary = {"query_count": len(metrics)}
    for k in (1, 3, 5):
        for key in (f"hit@{k}", f"recall@{k}"):
            summary[key] = sum(item[key] for item in metrics) / len(metrics)
    summary["mrr@5"] = sum(item["rr@5"] for item in metrics) / len(metrics)
    summary["mean_query_ms"] = sum(row["retrieval"][method]["elapsed_ms"] for row in rows) / len(rows)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--corpus", default="employment_basic_v004")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    rows = load_rows(dataset, args.split)

    with tempfile.TemporaryDirectory(prefix="checkit-classifier-rag-") as temp:
        prediction_path = Path(temp) / "classifier_predictions.jsonl"
        classifier_metadata_path = Path(temp) / "classifier_metadata.json"
        be_root = PROJECT_ROOT.parent / "CheckIt_BE"
        be_python = be_root / ".venv" / "bin" / "python"
        if not be_python.exists():
            raise FileNotFoundError(f"CheckIt_BE 가상환경을 찾을 수 없습니다: {be_python}")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(be_root) + os.pathsep + environment.get("PYTHONPATH", "")
        subprocess.run([
            str(be_python), str(PROJECT_ROOT / "test/rag/predict_classifier.py"),
            "--dataset", str(dataset), "--output", str(prediction_path),
            "--metadata", str(classifier_metadata_path), "--device", args.device,
        ], cwd=be_root, env=environment, check=True)
        predictions = {
            row["query_id"]: row
            for row in map(json.loads, prediction_path.read_text(encoding="utf-8").splitlines())
        }
        classifier_metadata = json.loads(classifier_metadata_path.read_text(encoding="utf-8"))
        labels = classifier_metadata["labels"]
        if len(labels) != 10 or any(not set(row["model_labels"]).issubset(labels) for row in rows):
            raise ValueError("평가 데이터의 gold labels가 CheckIt_BE 10개 라벨 계약과 다릅니다.")
        if set(predictions) != {row["query_id"] for row in rows}:
            raise ValueError("분류 예측과 평가 데이터의 query_id가 일치하지 않습니다.")
        for row in rows:
            row.update(predictions[row["query_id"]])
        classification_elapsed = classifier_metadata["total_query_ms"]

        snapshot = Path(temp) / "chroma"
        shutil.copytree(STORAGE_ROOT / args.corpus / "chroma", snapshot)
        for method in ("dense", "bm25", "hybrid"):
            retriever = LawRetriever(args.corpus, method=method, device=args.device,
                                     chroma_directory=snapshot)
            for row in rows:
                started = perf_counter()
                hits = retriever.search(row["query"], top_k=5)
                elapsed_ms = (perf_counter() - started) * 1000
                scores = score_query(hits, row["primary_article_ids"], ks=(1, 3, 5))
                row.setdefault("retrieval", {})[method] = {
                    "article_ids_top5": list(dict.fromkeys(
                        hit["metadata"]["article_id"] for hit in hits
                    )),
                    "metrics": scores,
                    "elapsed_ms": elapsed_ms,
                }
            del retriever
            gc.collect()

    by_method = {method: retrieval_metrics(rows, method)
                 for method in ("dense", "bm25", "hybrid")}
    index_manifest = STORAGE_ROOT / args.corpus / "index_manifest.json"
    manifest = json.loads(index_manifest.read_text(encoding="utf-8"))
    code_files = [
        "test/rag/evaluate_classifier_rag.py", "test/rag/predict_classifier.py",
        "RAG/indexing/embeddings.py", "RAG/indexing/bm25.py",
        "RAG/retrieval/retriever.py", "RAG/retrieval/label_hints.py",
        "RAG/retrieval/hybrid.py", "RAG/evaluation/retrieval_metrics.py",
    ]
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split, "dataset": str(dataset.relative_to(PROJECT_ROOT)),
        "dataset_sha256": sha256(dataset), "corpus": args.corpus,
        "index_manifest_sha256": sha256(index_manifest),
        "chunks_sha256": manifest["source_sha256"],
        "embedding_model": {key: manifest.get(key) for key in ("model_name", "model_revision")},
        "retrieval_config": {
            "methods": ["dense", "bm25", "hybrid"], "final_top_k": 5,
            "label_candidate_k": 20, "hybrid_candidate_k_per_retriever": 20,
            "rrf_k": 60, "label_policy": "labels_soft_priority_v1",
        },
        "code_sha256": {name: sha256(PROJECT_ROOT / name) for name in code_files},
        "classifier": classifier_metadata["model"],
        "classifier_thresholds": classifier_metadata["thresholds"],
        "classifier_device": args.device,
        "classifier_total_query_ms": classification_elapsed,
        "packages": {name: version(name) for name in
                     ("torch", "transformers", "chromadb", "sentence-transformers", "rank-bm25", "kiwipiepy")},
        "interpretation": "RAG scores use clause text only; classifier-predicted labels are not passed to retrieval. Classification is evaluated separately. Gold annotations remain provisional until independently reviewed.",
    }
    output = PROJECT_ROOT / "outputs" / "rag" / (
        "classifier_rag_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "classification": classification_metrics(rows, labels),
        "retrieval": by_method,
        "retrieval_input": "contract clause text only; classifier labels not applied",
        "classification_total_query_ms": classification_elapsed,
        "details": rows,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = [
        f"# 분류·RAG 개발 평가 ({args.split})", "",
        "**RAG 평가는 예측 라벨을 적용하지 않았습니다.** 검색 입력은 계약 조항 원문뿐이며 분류 성능은 별도로 계산했습니다.", "",
        "정답은 전문가 독립 검수 전 잠정값입니다. 이 결과로 운영 성능을 확정할 수 없습니다.", "",
        f"- Corpus: `{args.corpus}` / 기준일 2026-10-28",
        f"- Classifier: `{metadata['classifier']['model_id']}@{metadata['classifier']['revision']}`",
        f"- Dataset SHA-256: `{metadata['dataset_sha256']}`", "",
        "## 분류", "",
        "| 문장 수 | Micro F1 | Macro F1 | Exact match |", "|---:|---:|---:|---:|",
        f"| {report['classification']['query_count']} | {report['classification']['micro']['f1']:.3f} | {report['classification']['macro_f1']:.3f} | {report['classification']['exact_match']:.3f} |", "",
        "## RAG 원문 검색 (분류 라벨 미적용)", "",
        "| 방식 | Hit@5 | Recall@5 | MRR@5 | 평균 검색 시간(ms) |", "|---|---:|---:|---:|---:|",
    ]
    for method, metrics in by_method.items():
        markdown.append(
            f"| {method} | {metrics['hit@5']:.3f} | {metrics['recall@5']:.3f} | "
            f"{metrics['mrr@5']:.3f} | {metrics['mean_query_ms']:.2f} |"
        )
    (output / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["scope", "method", "input", "hit@5", "recall@5", "mrr@5", "mean_query_ms"])
        for method, values in by_method.items():
            writer.writerow([args.split, method, "text_only", values["hit@5"], values["recall@5"],
                             values["mrr@5"], values["mean_query_ms"]])
    print(json.dumps({"classification": report["classification"], "retrieval": by_method,
                      "output": str(output.relative_to(PROJECT_ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
