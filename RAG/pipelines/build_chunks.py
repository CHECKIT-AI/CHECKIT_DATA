import argparse
import hashlib
import json
from pathlib import Path

from RAG.preprocessing.chunk import build_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("articles_file", type=Path)
    args = parser.parse_args()

    input_path = args.articles_file
    output_path = input_path.parent / "chunks.jsonl"
    manifest_path = input_path.parent / "chunks_manifest.json"

    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "청크 결과가 이미 있습니다. "
            "새 corpus 폴더를 사용하거나 기존 결과를 확인하세요."
        )

    # 1. 정규화한 조문 읽기
    raw_bytes = input_path.read_bytes()
    articles = []

    for line_no, line in enumerate(
        raw_bytes.decode("utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            article = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(
                f"articles.jsonl의 {line_no}번째 줄이 잘못된 JSON입니다."
            ) from None

        if not isinstance(article, dict):
            raise ValueError(
                f"{line_no}번째 줄이 JSON 객체가 아닙니다."
            )

        articles.append(article)

    if not articles:
        raise ValueError("입력 조문이 없습니다.")

    # 2. 청크 생성
    chunks = build_chunks(articles)

    if not chunks:
        raise ValueError("검색 대상으로 사용할 청크가 없습니다.")

    # 3. JSONL 저장
    with output_path.open("x", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    # 4. 생성 이력 저장
    manifest = {
        "source_file": input_path.name,
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "chunking_strategy": "one_article_per_chunk_v1",
        "input_article_count": len(articles),
        "excluded_deleted_count": sum(
            article["is_deleted"] for article in articles
        ),
        "chunk_count": len(chunks),
    }

    with manifest_path.open("x", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"입력 조문: {len(articles)}개")
    print(f"삭제 조문 제외: {manifest['excluded_deleted_count']}개")
    print(f"생성 청크: {len(chunks)}개")
    print(f"저장 위치: {output_path}")


if __name__ == "__main__":
    main()