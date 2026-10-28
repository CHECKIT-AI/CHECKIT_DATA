"""기존 498개 원문을 보존해 검색용 표현과 누락된 시행규칙 별표를 새 버전으로 준비."""
import hashlib
import json
from pathlib import Path

from RAG.preprocessing.chunk import build_chunks
from RAG.preprocessing.search_text import clean_search_text
from RAG.settings import PROJECT_ROOT, DATA_ROOT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for child in value for s in strings(child)]
    raise ValueError("별표 내용 형식 오류")


def main():
    old = DATA_ROOT / "processed/employment_basic_v002"
    output = DATA_ROOT / "processed/employment_basic_v003"
    if output.exists():
        raise FileExistsError("기존 산출물을 덮어쓰지 않습니다.")
    previous = read(old / "manifest.json")
    if sha(old / "chunks.jsonl") != previous["chunks_sha256"]:
        raise ValueError("기존 청크 해시 불일치")
    articles = [json.loads(line) for line in (old / "articles.jsonl").read_text().splitlines() if line.strip()]
    manifest_path = DATA_ROOT / "processed/basic_labor_supplements_v001/manifest.json"
    supplements = read(manifest_path)
    new_assets = []
    for source in supplements["sources"]:
        if not source["law_name"].endswith("시행규칙"):
            continue
        body_path = PROJECT_ROOT / supplements["source_directory"] / source["body_file"]
        if sha(body_path) != source["source_sha256"]:
            raise ValueError("공식 API 원본 해시 불일치")
        entries = read(body_path)["법령"].get("별표", {}).get("별표단위", [])
        if isinstance(entries, dict):
            entries = [entries]
        for entry in entries:
            if entry.get("별표구분") != "별표" or "삭제" in entry.get("별표제목", ""):
                continue
            content = "\n".join(strings(entry["별표내용"]))
            aid = f"{source['law_id']}_{source['mst']}_{source['effective_date']}_annex_{entry['별표키']}"
            record = {
                "article_id": aid, "law_id": source["law_id"], "version_id": source["mst"],
                "law_name": source["law_name"], "article_key": entry["별표키"],
                "article_no": "별표" if entry["별표번호"] == "0000" else f"별표 {int(entry['별표번호'])}",
                "article_title": entry["별표제목"], "content": content,
                "snapshot_effective_date": source["effective_date"],
                "article_effective_date": entry["별표시행일자"], "is_deleted": False,
                "source_url": "https://www.law.go.kr" + entry["별표서식PDF파일링크"],
                "source_type": "law_appendix",
            }
            articles.append(record)
            new_assets.append({"article_id": aid, "raw_file": str(body_path.relative_to(PROJECT_ROOT)),
                               "raw_sha256": sha(body_path), "title": entry["별표제목"]})
    chunks = build_chunks(articles)
    for chunk in chunks:
        content = clean_search_text(chunk["content"])
        if not content:
            raise ValueError("검색용 본문이 비었습니다.")
        chunk["search_text"] = content
        chunk["search_text_version"] = "html_and_table_cleanup_v1"
        chunk["embedding_text"] = "\n".join([
            f"법령명: {chunk['law_name']}", f"조문: {chunk['article_no']}",
            f"제목: {chunk.get('article_title', '')}", "본문:", content,
        ])
        if chunk.get("source_type"):
            chunk["chunk_type"] = chunk["source_type"]
    output.mkdir(exist_ok=False)
    for filename, rows in [("articles.jsonl", articles), ("chunks.jsonl", chunks)]:
        (output / filename).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    manifest = {**previous, "corpus_id": output.name, "previous_corpus": "employment_basic_v002",
                "previous_chunks_sha256": sha(old / "chunks.jsonl"), "article_count": len(articles),
                "chunk_count": len(chunks), "chunks_sha256": sha(output / "chunks.jsonl"),
                "scope": "12_laws_main_articles_plus_8_appendices_and_2026_notice",
                "additional_appendices": new_assets, "index_status": "NOT_BUILT",
                "search_text_code_sha256": sha(PROJECT_ROOT / "RAG/preprocessing/search_text.py")}
    manifest.pop("index_manifest", None)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "chunks_manifest.json").write_text(json.dumps({
        "source_file": "articles.jsonl", "source_sha256": sha(output / "articles.jsonl"),
        "chunking_strategy": "one_record_with_clean_search_text_v1", "input_article_count": len(articles),
        "excluded_deleted_count": len(articles) - len(chunks), "chunk_count": len(chunks),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{output.name}: {len(chunks)}청크, 추가 별표 {len(new_assets)}개. 공식 content 보존.")


if __name__ == "__main__":
    main()
