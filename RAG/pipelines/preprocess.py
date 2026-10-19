import argparse
import hashlib
import json

from pathlib import Path

from RAG.preprocessing.normalize import normalize_law
from RAG.settings import DATA_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("body_file", type=Path)
    parser.add_argument("--mst", required=True)
    parser.add_argument("--corpus", required=True)
    args = parser.parse_args()

    if not args.mst.isdigit():
        raise ValueError("MST는 숫자여야 합니다.")

    # corpus를 단일 폴더 이름으로 제한
    if (
        args.corpus in {".", ".."}
        or "/" in args.corpus
        or "\\" in args.corpus
    ):
        raise ValueError("corpus는 단일 폴더 이름이어야 합니다.")

    # 기존 수집 코드의 파일명과 MST가 맞는지 확인
    if f"_{args.mst}_" not in args.body_file.name:
        raise ValueError("본문 파일명과 MST가 일치하지 않습니다.")

    raw_bytes = args.body_file.read_bytes()
    payload = json.loads(raw_bytes.decode("utf-8"))

    records = normalize_law(payload, mst=args.mst)

    output_dir = DATA_ROOT / "processed" / args.corpus

    # 기존 결과를 실수로 덮어쓰지 않도록 새 폴더만 생성
    output_dir.mkdir(parents=True, exist_ok=False)

    output_path = output_dir / "articles.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    manifest = {
        "corpus_id": args.corpus,
        "source_file": args.body_file.name,
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "mst": args.mst,
        "article_count": len(records),
        "deleted_count": sum(r["is_deleted"] for r in records),
        "review_status": "PENDING",
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"정규화 저장: {output_path}")
    print(f"전체 조문: {manifest['article_count']}")
    print(f"삭제 조문: {manifest['deleted_count']}")


if __name__ == "__main__":
    main()