"""dense / bm25 / hybrid 검색을 동일한 고정 질의로 평가한다."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

from RAG.evaluation.retrieval_metrics import score_query, summarize
from RAG.indexing.vector_store import COLLECTION_NAME
from RAG.retrieval.retriever import LawRetriever
from RAG.pipelines.build_index import corpus_name
from RAG.settings import DATA_ROOT, STORAGE_ROOT, OUTPUT_ROOT, PROJECT_ROOT, DEFAULT_CORPUS, relative_path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_queries(path):
    queries, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        query = json.loads(line)
        for field in ("query_id", "query", "query_kind", "topic", "annotation_status"):
            if not isinstance(query.get(field), str) or not query[field].strip():
                raise ValueError(f"평가 데이터 {number}번째 줄의 {field}를 확인하세요.")
        gold = query.get("relevant_article_ids")
        if not isinstance(gold, list) or not gold or any(not isinstance(g, str) or not g for g in gold):
            raise ValueError(f"{number}번째 질의에 정답 조문 ID가 필요합니다.")
        if query["query_id"] in seen:
            raise ValueError("중복 query_id입니다.")
        seen.add(query["query_id"])
        queries.append(query)
    if not queries:
        raise ValueError("평가 질의가 없습니다.")
    return queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=corpus_name, default=DEFAULT_CORPUS, help=f"평가 인덱스 (기본값: BGE-M3 {DEFAULT_CORPUS})")
    parser.add_argument("--dataset", type=Path, default=DATA_ROOT / "evaluation" / "retrieval_dev_v002.jsonl")
    parser.add_argument("--run-name", type=corpus_name)
    parser.add_argument("--method", choices=["dense", "bm25", "hybrid", "weighted_rrf", "hybrid_rerank"], default="dense")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-weight", type=float, default=0.8)
    parser.add_argument("--bm25-weight", type=float, default=0.2)
    parser.add_argument("--reranker-batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.candidate_k, args.rrf_k) < 1 or (args.method in {"hybrid", "weighted_rrf", "hybrid_rerank"} and args.candidate_k < 10):
        parser.error("정수 설정은 양수이고 hybrid candidate-k는 10 이상이어야 합니다.")
    queries = load_queries(args.dataset)
    index_dir = STORAGE_ROOT / args.corpus
    manifest_path = index_dir / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "READY"
        or manifest.get("corpus_id") != args.corpus
        or manifest.get("collection_name") != COLLECTION_NAME
        or manifest.get("distance_metric") != "cosine"
        or not manifest.get("normalize_embeddings")
    ):
        raise RuntimeError("완료된 cosine BGE 인덱스가 아닙니다.")
    chunk_path = DATA_ROOT / manifest["source_file"]
    if sha256(chunk_path) != manifest["source_sha256"]:
        raise RuntimeError("인덱스 생성 후 청크 파일이 변경됐습니다.")
    chunks = [json.loads(line) for line in chunk_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    available = {row["article_id"] for row in chunks}
    for query in queries:
        if not set(query["relevant_article_ids"]).issubset(available):
            raise ValueError(f"현재 corpus에 없는 정답 조문: {query['query_id']}")

    print(f"검색 방식: {args.method} / 평가 질의: {len(queries)}개", flush=True)
    ks = (1, 3, 5, 10)
    results = []
    with tempfile.TemporaryDirectory(prefix="checkit-retrieval-eval-") as temp:
        snapshot = None
        if args.method in {"dense", "hybrid", "weighted_rrf", "hybrid_rerank"}:
            snapshot = Path(temp) / "chroma"
            shutil.copytree(index_dir / "chroma", snapshot)
        retriever = LawRetriever(
            args.corpus, method=args.method, device=args.device,
            candidate_k=args.candidate_k, rrf_k=args.rrf_k,
            chroma_directory=snapshot,
            dense_weight=args.dense_weight, bm25_weight=args.bm25_weight,
            reranker_batch_size=args.reranker_batch_size,
        )
        retriever.search("법령 검색 준비", top_k=max(ks))
        for query in queries:
            started = perf_counter()
            retrieved = retriever.search(query["query"], top_k=max(ks))
            elapsed_ms = (perf_counter() - started) * 1000
            row = {
                **query,
                "metrics": score_query(retrieved, query["relevant_article_ids"], ks),
                "elapsed_ms": elapsed_ms,
                "results": retrieved,
                "candidate_article_ids": sorted({r["metadata"]["article_id"] for r in retriever.last_candidates}),
                "candidate_count": len(retriever.last_candidates),
            }
            results.append(row)
            rank = row["metrics"]["first_relevant_rank"]
            print(f"{query['query_id']} | {query['query_kind']} | 대표 정답 순위: {rank if rank else '10위 밖'}", flush=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = OUTPUT_ROOT / f"{args.run_name or args.method}_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "method": args.method,
        "corpus": args.corpus,
        "annotation_statuses": sorted({q["annotation_status"] for q in queries}),
        "overall": summarize(results, ks),
        "by_query_kind": {
            kind: summarize([r for r in results if r["query_kind"] == kind], ks)
            for kind in sorted({r["query_kind"] for r in results})
        },
    }
    summary["candidate_pool"] = {
        "mean_candidate_count": sum(r["candidate_count"] for r in results) / len(results),
        "representative_target_hit_rate": sum(bool(set(r["relevant_article_ids"]) & set(r["candidate_article_ids"])) for r in results) / len(results),
        "note": "dense 후보 깊이는 최종10개, hybrid 계열은 각 검색기 candidate-k의 합집합. 서로 다른 후보 예산임.",
    }
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": args.dataset.name,
        "dataset_sha256": sha256(args.dataset),
        "chunks_sha256": manifest["source_sha256"],
        "index_manifest_sha256": sha256(manifest_path),
        "model": retriever.embedder.describe() if retriever.embedder else None,
        "method": args.method,
        "retrieval_config": retriever.describe(),
        "retrieval_depth": max(ks),
        "metrics_at": list(ks),
        "device": args.device,
        "timing": "warm retriever: query preparation + search + optional fusion; excludes startup, warmup, DB copy and metrics",
        "code_sha256": {
            name: sha256(PROJECT_ROOT / name)
            for name in ["RAG/indexing/embeddings.py", "RAG/indexing/vector_store.py", "RAG/pipelines/evaluate.py", "RAG/evaluation/retrieval_metrics.py", "RAG/preprocessing/tokenizer.py", "RAG/indexing/bm25.py", "RAG/retrieval/hybrid.py", "RAG/retrieval/retriever.py", "RAG/retrieval/reranker.py", "RAG/retrieval/temporal.py", "RAG/preprocessing/search_text.py", "RAG/config/search_concepts.json"]
        },
        "packages": {name: version(name) for name in ["chromadb", "sentence-transformers", "transformers", "torch", "rank-bm25", "kiwipiepy", "kiwipiepy_model"]},
    }
    for name, content in [("summary.json", summary), ("run_metadata.json", metadata)]:
        (output / name).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "results.jsonl").open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        fields = ["query_id", "query_kind", "topic", "query", "expected", "first_relevant_rank", "hit@1", "hit@3", "hit@5", "hit@10", "rr@10", "elapsed_ms", "top5"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({
                **{key: row[key] for key in ["query_id", "query_kind", "topic", "query"]},
                "expected": ";".join(row.get("relevant_article_nos", row["relevant_article_ids"])),
                **{key: row["metrics"][key] for key in ["first_relevant_rank", "hit@1", "hit@3", "hit@5", "hit@10", "rr@10"]},
                "elapsed_ms": round(row["elapsed_ms"], 2),
                "top5": ";".join(r["metadata"]["article_no"] for r in row["results"][:5]),
            })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"평가 결과 저장: {relative_path(output)}")


if __name__ == "__main__":
    main()
