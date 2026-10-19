"""사전 지정한 가중치 2개와 후보 합집합 재정렬을 동일 질의셋으로 비교한다."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from RAG.pipelines.build_index import corpus_name
from RAG.settings import PROJECT_ROOT, OUTPUT_ROOT, relative_path, relative_log

SCENARIOS = [
    ("dense", "dense", 1.0, 1.0),
    ("hybrid_equal", "hybrid", 1.0, 1.0),
    ("weighted_80_20", "weighted_rrf", 0.8, 0.2),
    ("weighted_70_30", "weighted_rrf", 0.7, 0.3),
    ("union_rerank", "hybrid_rerank", 0.8, 0.2),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=corpus_name, required=True)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "test/rag/datasets/retrieval_dev_v002.jsonl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--reranker-batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.candidate_k < 10 or min(args.rrf_k, args.reranker_batch_size) < 1:
        parser.error("candidate-k는 10 이상, 나머지 정수 설정은 양수여야 합니다.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = OUTPUT_ROOT / f"improvements_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    reports, metadata, runs, per_query = {}, {}, {}, {}
    for label, method, dense_weight, bm25_weight in SCENARIOS:
        print(f"평가 실행: {label}", flush=True)
        command = [
            sys.executable, "-B", "-m", "RAG.pipelines.evaluate",
            "--corpus", args.corpus, "--dataset", str(args.dataset.resolve()),
            "--method", method, "--run-name", label + "_v002",
            "--device", args.device, "--candidate-k", str(args.candidate_k),
            "--rrf-k", str(args.rrf_k), "--dense-weight", str(dense_weight),
            "--bm25-weight", str(bm25_weight),
            "--reranker-batch-size", str(args.reranker_batch_size),
        ]
        log_path = output / f"{label}.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        log_path.write_text(relative_log(log_path.read_text(encoding="utf-8")), encoding="utf-8")
        if process.returncode:
            raise RuntimeError(f"{label} 평가 실패. 로그: {log_path}")
        marker = "평가 결과 저장: "
        folders = [line[len(marker):] for line in log_path.read_text(encoding="utf-8").splitlines() if line.startswith(marker)]
        if len(folders) != 1:
            raise RuntimeError(f"평가 결과 폴더를 확인하세요: {log_path}")
        folder = PROJECT_ROOT / Path(folders[0])
        runs[label] = str(folder.relative_to(PROJECT_ROOT))
        reports[label] = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        metadata[label] = json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))
        per_query[label] = {
            row["query_id"]: row
            for row in map(json.loads, (folder / "results.jsonl").read_text(encoding="utf-8").splitlines())
        }
    for key in ["dataset_sha256", "chunks_sha256", "retrieval_depth", "device", "model"]:
        values = [meta[key] for meta in metadata.values()]
        if any(value != values[0] for value in values):
            raise RuntimeError(f"비교 조건이 다릅니다: {key}")
    if any(set(rows) != set(per_query["dense"]) for rows in per_query.values()):
        raise RuntimeError("질의 ID 집합이 일치하지 않습니다.")
    fields = ["scenario", "query_count", "hit@1", "hit@5", "mrr@10", "mean_query_ms", "candidate_hit_rate"]
    table = [{
        "scenario": label,
        **{key: report["overall"][key] for key in fields if key not in {"scenario", "candidate_hit_rate"}},
        "candidate_hit_rate": report["candidate_pool"]["representative_target_hit_rate"],
    } for label, report in reports.items()]
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    changes = {}
    for label in reports:
        if label == "dense":
            continue
        changes[label] = []
        for qid, base in per_query["dense"].items():
            current = per_query[label][qid]
            old, new = base["metrics"]["hit@5"], current["metrics"]["hit@5"]
            if old != new:
                changes[label].append({
                    "query_id": qid, "query": base["query"],
                    "change": "improved" if new > old else "regressed",
                    "dense_first_rank": base["metrics"]["first_relevant_rank"],
                    "new_first_rank": current["metrics"]["first_relevant_rank"],
                })
    comparison = {
        "corpus": args.corpus, "dataset_sha256": metadata["dense"]["dataset_sha256"],
        "chunks_sha256": metadata["dense"]["chunks_sha256"],
        "runs": runs, "overall": table, "reports": reports,
        "settings": {label: meta["retrieval_config"] for label, meta in metadata.items()},
        "paired_hit5_changes": changes,
        "note": "90개 DRAFT 개발셋에 사전 지정한 설정을 1회 비교. 최종 테스트 성능 아님. reranker는 최대40개 후보 전체를 채점하며 RRF 가중치는 후보 사전 순서와 동점 처리에만 영향.",
    }
    (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(table, ensure_ascii=False, indent=2))
    print(f"개선 실험 결과 저장: {relative_path(output)}")


if __name__ == "__main__":
    main()
