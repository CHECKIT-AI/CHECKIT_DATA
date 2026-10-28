"""원문 단독·라벨+원문·오라벨의 근거 검색을 동일 조항 개발셋으로 비교."""
import hashlib
import json
from datetime import datetime, timezone
from time import perf_counter

from RAG.evaluation.retrieval_metrics import score_query, summarize
from RAG.retrieval.label_hints import POLICY_VERSION, validate_labels
from RAG.service import LawSearchService
from RAG.settings import DEFAULT_CORPUS, OUTPUT_ROOT, PROJECT_ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    dataset = PROJECT_ROOT / "test/rag/datasets/retrieval_labeled_clauses_v001.jsonl"
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    for row in rows:
        validate_labels(row["labels"])
    service = LawSearchService(corpus=DEFAULT_CORPUS, method="dense")
    available = {row["article_id"] for row in service.retriever.chunks}
    if any(not set(row["relevant_article_ids"]).issubset(available) for row in rows):
        raise ValueError("인덱스에 없는 정답")
    service.search("근로계약서 관련 근거", top_k=10)
    results = {"text_only": [], "labels_and_text": [], "wrong_label_stress": []}
    # 지연 측정 중 방식간 동시 실행하지 않는다.
    for mode in results:
        for row in rows:
            started = perf_counter()
            labels = row["labels"] if mode != "wrong_label_stress" else (["휴가"] if "임금" in row["labels"] else ["임금"])
            if mode == "text_only":
                response = service.search(row["text"], top_k=10)
            else:
                response = service.search_clause(row["clause_id"], row["text"], labels, top_k=10)
            elapsed = (perf_counter() - started) * 1000
            results[mode].append({**row, "input_labels": labels if mode != "text_only" else [],
                "metrics": score_query(response["results"], row["relevant_article_ids"]),
                "elapsed_ms": elapsed, "response": response})
    summary = {mode: summarize(values) for mode, values in results.items()}
    before = {row["clause_id"]: row for row in results["text_only"]}
    deltas = {}
    for mode in ["labels_and_text", "wrong_label_stress"]:
        deltas[mode] = {"gained_hit5": sum(r["metrics"]["hit@5"] > before[r["clause_id"]]["metrics"]["hit@5"] for r in results[mode]),
                       "lost_hit5": sum(r["metrics"]["hit@5"] < before[r["clause_id"]]["metrics"]["hit@5"] for r in results[mode])}
    output = PROJECT_ROOT / "test/rag/comparisons/labeled_clauses_v001_20261028"
    output.mkdir(exist_ok=False)
    for name, value in [("summary.json", summary), ("deltas.json", deltas), ("results.json", results)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "corpus": DEFAULT_CORPUS,
                "method": "dense", "policy": POLICY_VERSION, "dataset_sha256": sha(dataset),
                "chunks_sha256": service.retriever.manifest["source_sha256"],
                "label_source": "manual_topic_mapping_not_classifier_output", "status": "DRAFT",
                "code_sha256": {name: sha(PROJECT_ROOT / name) for name in
                    ["RAG/retrieval/label_hints.py", "RAG/service.py", "RAG/pipelines/evaluate_clauses.py"]}}
    (output / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "deltas": deltas}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
