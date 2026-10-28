"""Local corpus integrity audit; run with python test/rag/audit_employment_basic_v002.py."""
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from RAG.indexing.vector_store import LawVectorStore, METADATA_KEYS
from RAG.preprocessing.chunk import build_chunks
from RAG.preprocessing.normalize import normalize_law
from RAG.settings import PROJECT_ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    root = PROJECT_ROOT
    processed = root / "data/rag/processed/employment_basic_v002"
    index = root / "storage/rag/employment_basic_v002"
    manifest = read(processed / "manifest.json")
    articles = lines(processed / "articles.jsonl")
    chunks = lines(processed / "chunks.jsonl")
    indexed = read(index / "index_manifest.json")
    bm25 = read(index / "bm25/manifest.json")
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)

    check("articles_511_chunks_498_deleted_13", len(articles) == 511 and len(chunks) == 498
          and sum(row["is_deleted"] for row in articles) == 13)
    check("unique_article_and_chunk_ids", len({r["article_id"] for r in articles}) == len(articles)
          and len({r["chunk_id"] for r in chunks}) == len(chunks))
    check("processed_and_index_chunks_hash", digest(processed / "chunks.jsonl")
          == manifest["chunks_sha256"] == indexed["source_sha256"])
    check("index_ready_model_and_count", indexed["status"] == "READY" and indexed["chunk_count"] == 498
          and indexed["model_name"] == "BAAI/bge-m3" and indexed["dimension"] == 1024
          and indexed["model_revision"] == "5617a9f61b028005a4858fdac845db406aefb181")
    check("input_manifest_hashes", all(digest(root / name) == expected
          for name, expected in manifest["input_manifests"].items()))
    check("bm25_ready_count_and_input", bm25["status"] == "READY" and bm25["document_count"] == 498
          and bm25["source_sha256"] == indexed["source_sha256"])

    by_article = {row["article_id"]: row for row in articles}
    source_results = []
    for corpus in ["employment_laws_v001", "basic_labor_supplements_v001"]:
        original = read(root / "data/rag/processed" / corpus / "manifest.json")
        for source in original["sources"]:
            path = root / original["source_directory"] / source["body_file"]
            payload = read(path)
            basic = payload["법령"]["기본정보"]
            normalized = normalize_law(payload, source["mst"])
            passed = (digest(path) == source["source_sha256"]
                      and str(basic["법령ID"]) == source["law_id"]
                      and basic["법령명_한글"] == source["law_name"]
                      and str(basic["시행일자"]) == source["effective_date"]
                      and source["effective_date"] <= manifest["as_of"]
                      and all(by_article.get(row["article_id"]) == row for row in normalized))
            source_results.append({"law": source["law_name"], "passed": passed,
                                   "article_count": len(normalized)})
    check("all_12_raw_sources_versions_and_normalization", len(source_results) == 12
          and all(row["passed"] for row in source_results))

    rebuilt = build_chunks(articles)
    for row in rebuilt:
        if row.get("source_type"):
            row["chunk_type"] = row["source_type"]
    check("all_chunks_reproduce_from_articles", rebuilt == chunks)
    assets = root / "data/rag/raw/basic_labor_assets_20261028"
    asset_manifest = read(assets / "manifest.json")
    check("all_4_pdf_hashes", len(asset_manifest["assets"]) == 4 and all(
        digest(assets / asset["raw_file"]) == asset["sha256"]
        and (assets / asset["raw_file"]).read_bytes().startswith(b"%PDF-")
        for asset in asset_manifest["assets"]))
    notice = next(row for row in chunks if row.get("source_type") == "administrative_notice")
    check("notice_text_and_validity", notice["content"] == (assets / "minimum_wage_2026_notice_2025_47.txt").read_text()
          and notice["valid_from"] == "20260101" and notice["valid_to"] == "20261231")
    for corpus in ["labor_standards_v001", "employment_laws_v001"]:
        old_index = read(root / "storage/rag" / corpus / "index_manifest.json")
        check(corpus + "_input_preserved", digest(root / "data/rag" / old_index["source_file"])
              == old_index["source_sha256"])
    old_chunks = lines(root / "data/rag/processed/employment_laws_v001/chunks.jsonl")
    by_chunk = {row["chunk_id"]: row for row in chunks}
    check("all_existing_444_records_preserved", all(by_chunk[row["chunk_id"]] == row for row in old_chunks))

    store = LawVectorStore(index / "chroma")
    raw = store.collection.get(include=["documents", "metadatas", "embeddings"])
    check("chroma_count_and_ids", store.count() == 498 and set(raw["ids"]) == set(by_chunk))
    content_ok = metadata_ok = vector_ok = True
    norm_errors = []
    for identifier, document, metadata, vector in zip(raw["ids"], raw["documents"], raw["metadatas"], raw["embeddings"]):
        row = by_chunk[identifier]
        content_ok &= document == row["content"]
        metadata_ok &= all(metadata.get(key) == row[key] for key in METADATA_KEYS if row.get(key) is not None)
        norm_error = abs(sum(float(x) ** 2 for x in vector) - 1)
        norm_errors.append(norm_error)
        vector_ok &= len(vector) == 1024 and all(math.isfinite(float(x)) for x in vector) and norm_error < 1e-4
    check("chroma_official_content_matches", content_ok)
    check("chroma_metadata_matches_including_notice_dates", metadata_ok)
    check("all_498_vectors_finite_1024d_normalized", vector_ok)
    html_rows = [r for r in chunks if "<img" in r["content"] or "</img>" in r["content"]]
    report = {"corpus": "employment_basic_v002", "checks": checks,
              "passed_checks": sum(checks.values()), "total_checks": len(checks),
              "raw_sources": source_results, "max_vector_squared_norm_error": max(norm_errors),
              "quality_warnings": {
                  "html_image_markup_records": [{"article_id": r["article_id"], "law": r["law_name"],
                     "article_no": r["article_no"], "new_record": r not in old_chunks} for r in html_rows],
                  "date_filter_implemented": False, "expert_review_status": manifest["review_status"],
                  "scope_note": "Storage integrity is distinct from legal correctness, missing referenced appendices and retrieval accuracy."}}
    output = root / "test/rag/comparisons/employment_basic_v002_20261028"
    output.mkdir(parents=True, exist_ok=True)
    (output / "integrity_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
