"""설정한 여러 법령을 수집하고 하나의 새 corpus로 정규화·청킹한다."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from RAG.collection.law_api import fetch_law, search_laws
from RAG.pipelines.build_index import corpus_name
from RAG.preprocessing.chunk import build_chunks
from RAG.preprocessing.normalize import normalize_law
from RAG.settings import DATA_ROOT, PROJECT_ROOT, STORAGE_ROOT, relative_path


def write_json(path, value):
    # Checkpoint replacement keeps a completed previous checkpoint intact on interruption.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_targets(path):
    names = json.loads(Path(path).read_text(encoding="utf-8"))["laws"]
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name.strip() for name in names)):
        raise ValueError("법령 목록은 비어 있지 않은 이름의 배열이어야 합니다.")
    names = [name.strip() for name in names]
    if len(names) != len(set(names)):
        raise ValueError("중복 법령명이 있습니다.")
    return names


def validate_date(value):
    if len(value) != 8 or not value.isdigit():
        raise ValueError("시행일 형식은 YYYYMMDD여야 합니다.")
    datetime.strptime(value, "%Y%m%d")
    return value


def select_law(name, run_dir, prefix, as_of):
    matches = {}
    for page in range(1, 101):
        payload = search_laws(name, page=page)
        write_json(run_dir / f"{prefix}_search_{page}.json", payload)
        search = payload.get("LawSearch", {})
        if not isinstance(search, dict) or str(search.get("resultCode")) != "00":
            raise ValueError("정상적인 목록 조회 응답이 아닙니다.")
        rows = search.get("law", [])
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("목록 응답 형식이 다릅니다.")
        for row in rows:
            if row.get("법령명한글") == name:
                key = tuple(str(row[field]) for field in ("법령ID", "법령일련번호", "시행일자"))
                matches[key] = row
        if len(rows) < 100:
            break
    else:
        raise ValueError("목록 페이지 한도를 초과했습니다.")
    if len(matches) != 1:
        raise ValueError("정확히 일치하는 현행 법령 버전을 하나로 선택할 수 없습니다.")
    (law_id, mst, effective_date), _ = next(iter(matches.items()))
    if not law_id.isdigit() or not mst.isdigit():
        raise ValueError("법령 식별자가 숫자가 아닙니다.")
    validate_date(effective_date)
    if effective_date > as_of:
        raise ValueError("선택한 버전이 기준일 이후 시행 예정입니다. 버전을 확인하세요.")
    return law_id, mst, effective_date


def checked_records(payload, source):
    basic = payload.get("법령", {}).get("기본정보", {})
    if (str(basic.get("법령ID")) != source["law_id"]
            or basic.get("법령명_한글") != source["law_name"]
            or str(basic.get("시행일자")) != source["effective_date"]):
        raise ValueError("선택한 법령과 본문의 ID·법령명·시행일이 다릅니다.")
    return normalize_law(payload, mst=source["mst"])


def collect_sources(names, run_dir, as_of):
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "collection_manifest.json"
    manifest = {"status": "COLLECTING", "as_of": as_of, "target_laws": names, "sources": []}
    if checkpoint.exists():
        manifest = json.loads(checkpoint.read_text(encoding="utf-8"))
        if manifest["as_of"] != as_of or manifest["target_laws"] != names:
            raise ValueError("재개할 수집의 기준일·대상 목록이 다릅니다.")
    sources = manifest["sources"]
    if [source["law_name"] for source in sources] != names[:len(sources)]:
        raise ValueError("수집 이력의 법령 순서가 다릅니다.")
    articles = []
    for index, name in enumerate(names):
        if index < len(sources):
            source = sources[index]
            filename = source["body_file"]
            if Path(filename).name != filename:
                raise ValueError("본문 파일은 수집 폴더 내부 파일이어야 합니다.")
            raw = (run_dir / filename).read_bytes()
            if sha256(raw).hexdigest() != source["source_sha256"]:
                raise ValueError("보존한 본문 해시가 다릅니다.")
            records = checked_records(json.loads(raw), source)
        else:
            prefix = f"law_{index + 1:02d}"
            law_id, mst, effective_date = select_law(name, run_dir, prefix, as_of)
            source = {"law_name": name, "law_id": law_id, "mst": mst,
                      "effective_date": effective_date,
                      "body_file": f"{prefix}_{law_id}_{mst}_{effective_date}_body.json"}
            payload = fetch_law(mst=mst, effective_date=effective_date)
            # Save the untouched response even when validation fails, for diagnosis.
            write_json(run_dir / source["body_file"], payload)
            records = checked_records(payload, source)
            source.update(
                source_sha256=sha256((run_dir / source["body_file"]).read_bytes()).hexdigest(),
                article_count=len(records),
                chunk_count=sum(not row["is_deleted"] for row in records),
            )
            sources.append(source)
            write_json(checkpoint, manifest)
        articles.extend(records)
        print(f"수집 확인 ({index + 1}/{len(names)}): {name}, 조문 {len(records)}개", flush=True)
    manifest["status"] = "READY"
    write_json(checkpoint, manifest)
    return articles, manifest


def prepare_corpus(corpus, articles, collection, run_dir, processed_root):
    chunks = build_chunks(articles)  # Also rejects duplicate article/chunk IDs across laws.
    if not chunks:
        raise ValueError("검색 청크가 없습니다.")
    output = processed_root / corpus
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "BUILDING", "corpus_id": corpus,
                "as_of": collection["as_of"], "target_laws": collection["target_laws"],
                "source_directory": relative_path(run_dir), "sources": collection["sources"],
                "article_count": len(articles), "chunk_count": len(chunks),
                "deleted_count": len(articles) - len(chunks),
                "scope": "main_articles_only", "review_status": "PENDING"}
    write_json(output / "manifest.json", manifest)
    for filename, rows in (("articles.jsonl", articles), ("chunks.jsonl", chunks)):
        with (output / filename).open("x", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output / "chunks_manifest.json", {
        "source_file": "articles.jsonl",
        "source_sha256": sha256((output / "articles.jsonl").read_bytes()).hexdigest(),
        "chunking_strategy": "one_article_per_chunk_v1",
        "input_article_count": len(articles), "excluded_deleted_count": len(articles) - len(chunks),
        "chunk_count": len(chunks),
    })
    manifest["status"] = "READY"
    manifest["chunks_sha256"] = sha256((output / "chunks.jsonl").read_bytes()).hexdigest()
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=corpus_name)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "RAG/config/target_laws.json")
    parser.add_argument("--as-of", default=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d"))
    parser.add_argument("--resume", type=Path, help="부분 수집을 재개할 원본 폴더. 같은 기준일·설정 필요.")
    args = parser.parse_args()
    validate_date(args.as_of)
    if (DATA_ROOT / "processed" / args.corpus).exists() or (STORAGE_ROOT / args.corpus).exists():
        parser.error("이미 존재하는 corpus입니다. 기존 결과를 보존하고 새 이름을 사용하세요.")
    names = read_targets(args.config)
    run_dir = args.resume or DATA_ROOT / "raw" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    print(f"원본 수집 폴더: {relative_path(run_dir)}", flush=True)
    try:
        articles, collection = collect_sources(names, run_dir, args.as_of)
        result = prepare_corpus(args.corpus, articles, collection, run_dir, DATA_ROOT / "processed")
    except Exception as exc:
        # No raw response, request URL, OC or exception text is printed.
        print(f"수집·가공 실패: {type(exc).__name__}. 원본 폴더·설정·네트워크를 확인하세요.", flush=True)
        raise SystemExit(1) from None
    print(f"corpus 준비 완료: {args.corpus}, 법령 {len(names)}개, 청크 {result['chunk_count']}개", flush=True)
    print("벡터 인덱스는 build_index로 별도 생성하세요.", flush=True)


if __name__ == "__main__":
    main()
