"""기존 청크로 BM25 자료를 추가 생성한다. 벡터 DB는 변경하지 않는다."""

import argparse
import json

from RAG.indexing.bm25 import build_bm25_index, digest
from RAG.pipelines.build_index import corpus_name, load_chunks
from RAG.settings import DATA_ROOT, STORAGE_ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=corpus_name, required=True)
    args = parser.parse_args()
    directory = STORAGE_ROOT / args.corpus
    manifest = json.loads((directory / "index_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "READY" or manifest.get("corpus_id") != args.corpus:
        raise RuntimeError("먼저 해당 corpus의 벡터 인덱스를 완성하세요.")
    source = DATA_ROOT / manifest["source_file"]
    if digest(source) != manifest["source_sha256"]:
        raise RuntimeError("벡터 인덱스와 청크 버전이 다릅니다.")
    chunks = load_chunks(source)
    if len(chunks) != manifest["chunk_count"]:
        raise RuntimeError("청크 개수가 인덱스 명세와 다릅니다.")
    result = build_bm25_index(chunks, source, directory / "bm25", args.corpus)
    print(f"BM25 인덱스 저장: {directory / 'bm25'}")
    print(f"저장 문서: {result['document_count']}개")


if __name__ == "__main__":
    main()
