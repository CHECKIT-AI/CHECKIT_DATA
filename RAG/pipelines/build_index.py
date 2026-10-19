"""chunks.jsonl -> BGE-M3 -> 로컬 Chroma. 기존 DB는 덮어쓰지 않는다."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

# from RAG.indexing.embeddings import BGEEmbedder
from RAG.indexing.embeddings import (
    MODEL_NAME,
    QWEN_MODEL_NAME,
    create_embedder,
)
from RAG.indexing.vector_store import COLLECTION_NAME, LawVectorStore
from RAG.settings import DATA_ROOT, STORAGE_ROOT


def corpus_name(value):
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise argparse.ArgumentTypeError("corpus는 단일 폴더 이름이어야 합니다.")
    return value


def load_chunks(path):
    rows, seen = [], set()
    required = (
        "chunk_id", "article_id", "law_name", "article_no", "content",
        "embedding_text", "law_id", "version_id", "source_url",
    )
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{line_no}번째 줄은 JSON 객체가 아닙니다.")
        for field in required:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{line_no}번째 줄의 {field}가 비어 있거나 형식이 다릅니다.")
        if row.get("is_deleted") is not False:
            raise ValueError(f"{line_no}번째 줄의 삭제 상태를 확인하세요.")
        if row["chunk_id"] in seen:
            raise ValueError(f"중복 chunk_id: {row['chunk_id']}")
        seen.add(row["chunk_id"])
        rows.append(row)
    if not rows:
        raise ValueError("입력 청크가 없습니다.")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=corpus_name, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--model-name",
        choices=[MODEL_NAME, QWEN_MODEL_NAME],
        default=MODEL_NAME,
    )   
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size는 1 이상이어야 합니다.")

    source = DATA_ROOT / "processed" / args.corpus / "chunks.jsonl"
    output = STORAGE_ROOT / args.corpus
    if output.exists():
        raise FileExistsError(
            f"인덱스 폴더가 이미 있습니다: {output}. "
            "기존 결과 또는 실패 기록을 확인하고 새 corpus 버전을 사용하세요."
        )
    chunks = load_chunks(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    print(f"입력 청크: {len(chunks)}개", flush=True)
    # print("BGE-M3 모델 로딩 중입니다. 첫 실행은 다운로드가 필요합니다.", flush=True)
    # embedder = BGEEmbedder(device=args.device)
    # 변경
    print(
        f"{args.model_name} 모델 로딩 중입니다. 첫 실행은 다운로드가 필요합니다.",
        flush=True,
    )

    embedder = create_embedder(
        model_name=args.model_name,
        device=args.device,
        input_type="document",
    )

    lengths = embedder.token_lengths([row["embedding_text"] for row in chunks])
    oversized = [
        (row["chunk_id"], length)
        for row, length in zip(chunks, lengths)
        if length > embedder.max_length
    ]
    if oversized:
        raise ValueError(f"입력 길이 초과 청크를 먼저 분할하세요: {oversized}")
    print(f"최대 입력 길이: {max(lengths)} / {embedder.max_length} 토큰", flush=True)
    vectors = embedder.encode(
        [row["embedding_text"] for row in chunks],
        batch_size=args.batch_size, show_progress=True,
    )

    # 길이 검사와 임베딩이 끝난 후에만 DB 폴더를 만든다.
    output.mkdir(parents=True, exist_ok=False)
    try:
        store = LawVectorStore(output / "chroma", create=True)
        store.add(chunks, vectors)
        expected_ids = {row["chunk_id"] for row in chunks}
        if store.count() != len(chunks) or store.ids() != expected_ids:
            raise RuntimeError("저장한 청크 개수 또는 ID 집합이 입력과 다릅니다.")

        manifest = {
            "status": "READY",
            "corpus_id": args.corpus,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_file": str(source.relative_to(DATA_ROOT)),
            "source_sha256": source_hash,
            "collection_name": COLLECTION_NAME,
            "chunk_count": len(chunks),
            "distance_metric": "cosine",
            "max_document_tokens": max(lengths),
            **embedder.describe(),
            "packages": {
                name: version(name)
                for name in ["chromadb", "sentence-transformers", "torch", "transformers"]
            },
        }
        (output / "index_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    except Exception as exc:
        (output / "FAILED.txt").write_text(
            f"인덱스 생성 실패: {type(exc).__name__}\n",
            encoding="utf-8",
        )
        raise
    print(f"검색 DB 저장 완료: {output / 'chroma'}")
    print(f"저장 문서: {len(chunks)}개")


if __name__ == "__main__":
    main()
