"""동일 평가셋으로 세 방식을 실행하고 지표·질의별 개선/악화를 비교한다."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from RAG.pipelines.build_index import corpus_name
from RAG.settings import PROJECT_ROOT, OUTPUT_ROOT, relative_path, relative_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=corpus_name, required=True)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "test/rag/datasets/retrieval_dev_v002.jsonl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()
    if args.candidate_k < 10 or args.rrf_k < 1:
        parser.error("candidate-k는 10 이상, rrf-k는 1 이상이어야 합니다.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = OUTPUT_ROOT / f"comparison_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    reports, runs, metas, per_query = {}, {}, {}, {}
    for method in ["dense", "bm25", "hybrid"]:
        print(f"평가 실행: {method}", flush=True)
        command = [
            sys.executable, "-B", "-m", "RAG.pipelines.evaluate",
            "--corpus", args.corpus, "--dataset", str(args.dataset.resolve()),
            "--method", method, "--device", args.device,
            "--candidate-k", str(args.candidate_k), "--rrf-k", str(args.rrf_k),
            "--run-name", f"{method}_v002",
        ]
        process = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True)
        (output / f"{method}.log").write_text(relative_log(process.stdout + "\n" + process.stderr), encoding="utf-8")
        if process.returncode:
            raise RuntimeError(f"{method} 평가 실패. 로그: {output / (method + '.log')}")
        marker = "평가 결과 저장: "
        folders = [line[len(marker):] for line in process.stdout.splitlines() if line.startswith(marker)]
        if len(folders) != 1:
            raise RuntimeError("평가 결과 폴더를 확인할 수 없습니다.")
        folder = PROJECT_ROOT / Path(folders[0])
        runs[method] = str(folder.relative_to(PROJECT_ROOT))
        reports[method] = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        metas[method] = json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))
        per_query[method] = {
            row["query_id"]: row
            for row in map(json.loads, (folder / "results.jsonl").read_text(encoding="utf-8").splitlines())
        }
    for field in ["dataset_sha256", "chunks_sha256", "retrieval_depth", "device"]:
        values = [metas[method][field] for method in reports]
        if any(value != values[0] for value in values):
            raise RuntimeError(f"평가 조건이 일치하지 않습니다: {field}")
    if metas["dense"]["model"] != metas["hybrid"]["model"]:
        raise RuntimeError("dense와 hybrid의 임베딩 모델이 다릅니다.")
    if any(set(rows) != set(per_query["dense"]) for rows in per_query.values()):
        raise RuntimeError("방법별 질의 ID가 다릅니다.")

    fields = ["method", "query_count", "hit@1", "hit@5", "mrr@10", "mean_query_ms"]
    table = [{"method": method, **{key: report["overall"][key] for key in fields if key != "method"}} for method, report in reports.items()]
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    paired = []
    for query_id, dense in per_query["dense"].items():
        hybrid = per_query["hybrid"][query_id]
        old, new = dense["metrics"]["hit@5"], hybrid["metrics"]["hit@5"]
        if old != new:
            paired.append({
                "query_id": query_id, "query": dense["query"],
                "change": "improved" if new > old else "regressed",
                "dense_first_rank": dense["metrics"]["first_relevant_rank"],
                "hybrid_first_rank": hybrid["metrics"]["first_relevant_rank"],
            })
    comparison = {
        "corpus": args.corpus, "dataset_sha256": metas["dense"]["dataset_sha256"],
        "chunks_sha256": metas["dense"]["chunks_sha256"],
        "runs": runs, "overall": table, "reports": reports,
        "paired_hit5_changes": paired,
        "hybrid_minus_dense": {
            key: reports["hybrid"]["overall"][key] - reports["dense"]["overall"][key]
            for key in ["hit@1", "hit@5", "mrr@10"]
        },
        "note": "DRAFT 개발셋 단일 실행. 후보 수와 워밍업 조건은 메타데이터 참조. 미사용 테스트셋 성능 또는 시간 측정의 통계적 우열을 뜻하지 않음.",
    }
    (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(table, ensure_ascii=False, indent=2))
    print(f"비교 결과 저장: {relative_path(output)}")


if __name__ == "__main__":
    main()
