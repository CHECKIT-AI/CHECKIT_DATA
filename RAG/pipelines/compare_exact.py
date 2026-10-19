"""저장된 동일 벡터로 Chroma와 exhaustive cosine 검색을 비교한다."""

import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter

import numpy as np

from RAG.evaluation.retrieval_metrics import score_query, summarize
# from RAG.indexing.embeddings import BGEEmbedder
from RAG.indexing.embeddings import create_embedder
from RAG.indexing.vector_store import COLLECTION_NAME, LawVectorStore
from RAG.pipelines.build_index import corpus_name
from RAG.pipelines.evaluate import load_queries, sha256
from RAG.settings import DATA_ROOT, OUTPUT_ROOT, PROJECT_ROOT, STORAGE_ROOT, relative_path


class ExactCosineSearch:
    """모든 문서와 cosine을 계산. 동점은 chunk_id 오름차순."""

    def __init__(self, raw):
        order = sorted(range(len(raw["ids"])), key=lambda i: raw["ids"][i])
        self.ids = [raw["ids"][i] for i in order]
        self.documents = [raw["documents"][i] for i in order]
        self.metadatas = [raw["metadatas"][i] for i in order]
        vectors = np.asarray(raw["embeddings"], dtype=np.float64)
        if vectors.ndim != 2 or not len(vectors) or not np.isfinite(vectors).all():
            raise ValueError("문서 벡터 형식을 확인하세요.")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            raise ValueError("문서에 영벡터가 있습니다.")
        self.vectors = (vectors / norms[:, None])[order]

    def search(self, query_vector):
        query = np.asarray(query_vector, dtype=np.float64)
        if query.shape != (self.vectors.shape[1],) or not np.isfinite(query).all() or np.linalg.norm(query) == 0:
            raise ValueError("질의 벡터 차원과 값을 확인하세요.")
        distances = 1.0 - np.clip(self.vectors @ (query / np.linalg.norm(query)), -1.0, 1.0)
        order = np.argsort(distances, kind="stable")
        return [
            {"rank": rank, "chunk_id": self.ids[i],
             "cosine_distance": float(distances[i]),
             "content": self.documents[i], "metadata": self.metadatas[i]}
            for rank, i in enumerate(order, 1)
        ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=corpus_name, required=True)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "test/rag/datasets/retrieval_dev_v002.jsonl")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    queries = load_queries(args.dataset)
    index_dir = STORAGE_ROOT / args.corpus
    manifest_path = index_dir / "index_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "READY" or manifest.get("corpus_id") != args.corpus
            or manifest.get("collection_name") != COLLECTION_NAME
            or manifest.get("distance_metric") != "cosine"):
        raise ValueError("완료된 cosine 인덱스가 필요합니다.")
    chunks_path = DATA_ROOT / manifest["source_file"]
    if sha256(chunks_path) != manifest["source_sha256"]:
        raise ValueError("인덱스 생성 이후 청크 파일이 변경됐습니다.")
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ks = (1, 3, 5, 10, 20)
    tolerance = 1e-6
    comparisons, evaluated = [], {"chroma": [], "exact": []}
    with tempfile.TemporaryDirectory(prefix="checkit-exact-") as temp:
        snapshot = Path(temp) / "chroma"
        shutil.copytree(index_dir / "chroma", snapshot)
        store = LawVectorStore(snapshot)
        raw = store.collection.get(include=["embeddings", "documents", "metadatas"])
        if len(raw["ids"]) != manifest["chunk_count"] or set(raw["ids"]) != {c["chunk_id"] for c in chunks}:
            raise ValueError("DB와 청크 파일의 개수/ID가 다릅니다.")
        exact = ExactCosineSearch(raw)
        if exact.vectors.shape[1] != manifest["dimension"]:
            raise ValueError("저장된 벡터 차원이 manifest와 다릅니다.")
        available = {m["article_id"] for m in exact.metadatas}
        if any(not set(q["relevant_article_ids"]).issubset(available) for q in queries):
            raise ValueError("현재 DB에 없는 정답 조문이 있습니다.")
        print(f"문서 {len(exact.ids)}개 / 질의 {len(queries)}개 / 모델 로딩", flush=True)
        # embedder = BGEEmbedder(manifest["model_name"], revision=manifest["model_revision"], device=args.device)
        embedder = create_embedder(
            model_name=manifest["model_name"],
            revision=manifest["model_revision"],
            device=args.device,
            input_type="query",
            query_instruction=manifest.get("query_instruction"),
        )
        warmup = embedder.encode(["법령 검색 준비"])[0]
        store.search(warmup, top_k=max(ks))
        exact.search(warmup)
        for number, query in enumerate(queries):
            started = perf_counter()
            vector = embedder.encode([query["query"]])[0]
            embedding_ms = (perf_counter() - started) * 1000
            rankings, timings = {}, {}
            # 측정 순서를 번갈아 적용. 모델 로딩/DB 복사/워밍업 제외.
            methods = ("chroma", "exact") if number % 2 == 0 else ("exact", "chroma")
            for method in methods:
                started = perf_counter()
                rankings[method] = store.search(vector, top_k=max(ks)) if method == "chroma" else exact.search(vector)
                timings[method] = (perf_counter() - started) * 1000
                evaluated[method].append({
                    "metrics": score_query(rankings[method][:max(ks)], query["relevant_article_ids"], ks),
                    "elapsed_ms": timings[method],
                })
            oracle = {r["chunk_id"]: r for r in rankings["exact"]}
            overlap, material_misses = {}, {}
            for k in ks:
                actual = {r["chunk_id"] for r in rankings["chroma"][:k]}
                target = {r["chunk_id"] for r in rankings["exact"][:k]}
                overlap[str(k)] = len(actual & target) / len(target)
                # 경계의 부동소수점 동점을 유의한 ANN 누락으로 세지 않는다.
                worst = max(oracle[c]["cosine_distance"] for c in actual)
                material_misses[str(k)] = [
                    c for c in sorted(target - actual)
                    if oracle[c]["cosine_distance"] < worst - tolerance
                ]
            row = {
                **query, "embedding_ms": embedding_ms, "search_ms": timings,
                "metrics": {m: evaluated[m][-1]["metrics"] for m in evaluated},
                "exact_full_gold_rank": score_query(rankings["exact"], query["relevant_article_ids"], ks)["first_relevant_rank"],
                "overlap": overlap, "material_misses": material_misses,
                "max_distance_error": max(abs(r["cosine_distance"] - oracle[r["chunk_id"]]["cosine_distance"]) for r in rankings["chroma"]),
                "results": rankings,
            }
            comparisons.append(row)
            print(f"{number + 1}/{len(queries)} {query['query_id']} | overlap@20={overlap['20']:.2f} | 정답 전체 순위={row['exact_full_gold_rank']}", flush=True)

    collection_configuration = store.collection.configuration
    summary = {
        "corpus": args.corpus, "query_count": len(queries), "document_count": len(exact.ids),
        "methods": {m: summarize(rows, ks) for m, rows in evaluated.items()},
        "mean_embedding_ms": sum(r["embedding_ms"] for r in comparisons) / len(comparisons),
        "mean_overlap": {str(k): sum(r["overlap"][str(k)] for r in comparisons) / len(comparisons) for k in ks},
        "queries_with_material_misses": {str(k): sum(bool(r["material_misses"][str(k)]) for r in comparisons) for k in ks},
        "identical_order_queries_at20": sum([x["chunk_id"] for x in r["results"]["chroma"]] == [x["chunk_id"] for x in r["results"]["exact"][:20]] for r in comparisons),
        "max_distance_error": max(r["max_distance_error"] for r in comparisons),
        "note": "DRAFT 개발셋. mean_query_ms는 검색만(질의 임베딩 별도). Exact는 전체 순위 생성, Chroma는 최대20개 반환. 단일 실행으로 속도 우열을 단정하지 않음.",
    }
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = OUTPUT_ROOT / f"exact_comparison_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": relative_path(args.dataset), "dataset_sha256": sha256(args.dataset),
        "index_manifest_sha256": sha256(manifest_path), "chunks_sha256": sha256(chunks_path),
        "index_manifest": manifest, "model": embedder.describe(), "device": args.device,
        "collection_configuration": collection_configuration,
        "metrics_at": list(ks), "distance_tolerance": tolerance,
        "exact": "numpy float64 exhaustive cosine; chunk_id tie-break; same stored vectors and one shared query embedding",
        "packages": {p: version(p) for p in ["numpy", "chromadb", "sentence-transformers", "torch"]},
        "code_sha256": {p: sha256(PROJECT_ROOT / p) for p in ["RAG/pipelines/compare_exact.py", "RAG/indexing/vector_store.py", "RAG/indexing/embeddings.py", "RAG/evaluation/retrieval_metrics.py"]},
    }
    for name, content in [("summary.json", summary), ("run_metadata.json", metadata)]:
        (output / name).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "results.jsonl").open("w", encoding="utf-8") as file:
        for row in comparisons:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output / "results.csv").open("w", encoding="utf-8-sig", newline="") as file:
        fields = ["query_id", "query", "chroma_gold_rank_at20", "exact_full_gold_rank", "overlap_at5", "overlap_at20", "material_misses_at20"]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for r in comparisons:
            writer.writerow(dict(zip(fields, [r["query_id"], r["query"], r["metrics"]["chroma"]["first_relevant_rank"], r["exact_full_gold_rank"], r["overlap"]["5"], r["overlap"]["20"], ";".join(r["material_misses"]["20"])])))
    report = ["# Chroma와 전체 벡터 정확 검색 비교", "", summary["note"], "", "| 방식 | Hit@1 | Hit@5 | Hit@20 | MRR@20 | 검색 평균 ms |", "|---|---:|---:|---:|---:|---:|"]
    for method, scores in summary["methods"].items():
        report.append(f"| {method} | {scores['hit@1']:.1%} | {scores['hit@5']:.1%} | {scores['hit@20']:.1%} | {scores['mrr@20']:.4f} | {scores['mean_query_ms']:.3f} |")
    report.extend(["", f"상위20개 순서까지 동일: {summary['identical_order_queries_at20']}/{len(queries)}", f"상위20개 유의한 검색 누락 질의: {summary['queries_with_material_misses']['20']}", "", "기존 MRR@10과 이 보고서의 MRR@20은 평가 깊이가 다릅니다."])
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"비교 결과 저장: {relative_path(output)}", flush=True)


if __name__ == "__main__":
    main()
