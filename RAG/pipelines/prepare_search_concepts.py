"""공식 원문과 구분한 검색용 용어 주석을 새 corpus에 적용한다."""
import hashlib
import json

from RAG.settings import DATA_ROOT, PROJECT_ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = DATA_ROOT / "processed/employment_basic_v003"
    output = DATA_ROOT / "processed/employment_basic_v004"
    if output.exists():
        raise FileExistsError("기존 corpus를 덮어쓰지 않습니다.")
    manifest = json.loads((source / "manifest.json").read_text())
    if digest(source / "chunks.jsonl") != manifest["chunks_sha256"]:
        raise ValueError("이전 청크 해시 불일치")
    config_path = PROJECT_ROOT / "RAG/config/search_concepts.json"
    config = json.loads(config_path.read_text())
    rows = [json.loads(line) for line in (source / "chunks.jsonl").read_text().splitlines() if line.strip()]
    matched = []
    for concept in config["concepts"]:
        targets = [r for r in rows if r["law_name"] == concept["law_name"] and r["article_title"] == concept["article_title"]]
        if len(targets) != 1:
            raise ValueError("보충 용어의 대상 조문을 하나로 확인할 수 없습니다.")
        row = targets[0]
        if concept["basis"] not in row["content"]:
            raise ValueError("보충 용어의 근거 원문 발췌가 일치하지 않습니다.")
        row["search_hints"] = concept["terms"]
        row["search_hints_version"] = config["version"]
        row["embedding_text"] += "\n검색용 보충어(공식 원문 아님): " + concept["terms"]
        matched.append(row["article_id"])
    output.mkdir(exist_ok=False)
    (output / "articles.jsonl").write_bytes((source / "articles.jsonl").read_bytes())
    (output / "chunks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    manifest.update(corpus_id=output.name, previous_corpus="employment_basic_v003",
                    previous_chunks_sha256=digest(source / "chunks.jsonl"), chunks_sha256=digest(output / "chunks.jsonl"),
                    index_status="NOT_BUILT", search_concepts={"config": str(config_path.relative_to(PROJECT_ROOT)),
                    "config_sha256": digest(config_path), "version": config["version"], "annotated_articles": matched,
                    "expert_review_status": "PENDING", "development_note": "기존 평가의 오류 분석 후 작성했으므로 같은 평가의 개선은 개발셋 개선이며 일반화 성능이 아님."})
    manifest.pop("index_manifest", None)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    chunks_manifest = json.loads((source / "chunks_manifest.json").read_text())
    chunks_manifest["chunking_strategy"] = "clean_search_text_with_curated_concepts_v1"
    (output / "chunks_manifest.json").write_text(json.dumps(chunks_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{output.name}: {len(rows)}개 청크, 원문 보존, 검색용 보충어 {len(matched)}개 조문 적용")


if __name__ == "__main__":
    main()
