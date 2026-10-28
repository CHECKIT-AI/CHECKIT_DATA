"""python test/rag/audit_search_improvements.py: 최종 보완 corpus의 로컬 무결성 검사."""
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from RAG.indexing.vector_store import LawVectorStore, METADATA_KEYS
from RAG.preprocessing.search_text import clean_search_text
from RAG.settings import PROJECT_ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def flatten(value):
    return [value] if isinstance(value, str) else [s for child in value for s in flatten(child)]


def main():
    root = PROJECT_ROOT
    p = root / "data/rag/processed/employment_basic_v004"
    index = root / "storage/rag/employment_basic_v004"
    rows = records(p / "chunks.jsonl")
    old = records(root / "data/rag/processed/employment_basic_v002/chunks.jsonl")
    by = {r["chunk_id"]: r for r in rows}
    manifest, im, bm = read(p / "manifest.json"), read(index / "index_manifest.json"), read(index / "bm25/manifest.json")
    checks = {}
    checks["503_unique_records"] = len(rows) == len(by) == 503
    checks["498_original_contents_preserved"] = all(by[r["chunk_id"]]["content"] == r["content"] for r in old)
    checks["search_cleanup_reproducible"] = all(r["search_text"] == clean_search_text(r["content"]) and "<img" not in r["search_text"] for r in rows)
    checks["input_hashes_match"] = sha(p / "chunks.jsonl") == manifest["chunks_sha256"] == im["source_sha256"] == bm["source_sha256"]
    checks["bm25_documents_hash_count"] = bm["document_count"] == 503 and bm["status"] == "READY" and sha(index / "bm25/tokenized_documents.json") == bm["documents_sha256"]
    checks["five_appendix_sources_preserved"] = len(manifest["additional_appendices"]) == 5 and all(sha(root / a["raw_file"]) == a["raw_sha256"] for a in manifest["additional_appendices"])
    appendix_texts = []
    for a in manifest["additional_appendices"]:
        row = next(r for r in rows if r["article_id"] == a["article_id"])
        entries = read(root / a["raw_file"])["법령"]["별표"]["별표단위"]
        entries = entries if isinstance(entries, list) else [entries]
        entry = next(e for e in entries if e["별표키"] == row["article_key"] and e["별표구분"] == "별표")
        appendix_texts.append(row["content"] == "\n".join(flatten(entry["별표내용"])))
    checks["five_appendix_contents_match_raw"] = all(appendix_texts)
    config = manifest["search_concepts"]
    checks["curated_terms_config_hash"] = sha(root / config["config"]) == config["config_sha256"]
    annotated = [r for r in rows if r.get("search_hints")]
    checks["seven_annotations_separate_from_original"] = len(annotated) == 7 and all(
        "검색용 보충어(공식 원문 아님): " + r["search_hints"] in r["embedding_text"] and
        r["search_hints_version"] == config["version"] for r in annotated)
    store = LawVectorStore(index / "chroma")
    raw = store.collection.get(include=["documents", "metadatas", "embeddings"])
    checks["chroma_count_and_ids"] = store.count() == 503 and set(raw["ids"]) == set(by)
    checks["chroma_official_text_and_metadata"] = all(d == by[i]["content"] and all(md.get(k) == by[i][k]
        for k in METADATA_KEYS if by[i].get(k) is not None)
        for i, d, md in zip(raw["ids"], raw["documents"], raw["metadatas"]))
    checks["vectors_finite_1024d_normalized"] = all(len(v) == 1024 and all(math.isfinite(float(x)) for x in v)
        and abs(sum(float(x)**2 for x in v) - 1) < 1e-4 for v in raw["embeddings"])
    for corpus in ["employment_basic_v002", "employment_laws_v001", "labor_standards_v001"]:
        oi = read(root / "storage/rag" / corpus / "index_manifest.json")
        checks[corpus + "_input_preserved"] = sha(root / "data/rag" / oi["source_file"]) == oi["source_sha256"]
    report = {"corpus": "employment_basic_v004", "checks": checks, "passed": sum(checks.values()),
              "total": len(checks), "expert_review_status": "PENDING"}
    output = root / "test/rag/comparisons/employment_basic_v004_improvements_20261028"
    output.mkdir(exist_ok=True)
    (output / "integrity_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
