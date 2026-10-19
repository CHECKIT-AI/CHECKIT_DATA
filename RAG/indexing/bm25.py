"""BM25 토큰 자료 저장·검증·검색. pickle 대신 JSON을 사용한다."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path

from RAG.indexing.vector_store import METADATA_KEYS
from RAG.preprocessing.tokenizer import KoreanTokenizer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_bm25_index(chunks, source_path, directory, corpus):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"기존 BM25 인덱스를 덮어쓰지 않습니다: {directory}")
    tokenizer = KoreanTokenizer()
    documents = [
        {"chunk_id": row["chunk_id"], "tokens": tokenizer.tokenize(row["embedding_text"])}
        for row in chunks
    ]
    if not documents or any(not row["tokens"] for row in documents):
        raise ValueError("비어 있는 BM25 문서가 있습니다.")
    if len({row["chunk_id"] for row in documents}) != len(documents):
        raise ValueError("중복된 BM25 문서 ID입니다.")
    data = json.dumps(documents, ensure_ascii=False, indent=2).encode("utf-8")
    manifest = {
        "status": "READY", "corpus_id": corpus,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": digest(source_path),
        "document_count": len(documents),
        "documents_sha256": hashlib.sha256(data).hexdigest(),
        "tokenizer": tokenizer.describe(),
        "tokenizer_code_sha256": digest(Path(__file__).parents[1] / "preprocessing/tokenizer.py"),
        "algorithm": "BM25Okapi", "rank_bm25_version": version("rank-bm25"),
        "parameters": {"k1": 1.5, "b": 0.75, "epsilon": 0.25},
        "min_score_exclusive": 0.0,
        "indexed_field": "embedding_text",
    }
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "tokenized_documents.json").write_bytes(data)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest


class BM25Retriever:
    def __init__(self, directory, chunks, source_path, corpus):
        from rank_bm25 import BM25Okapi
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        manifest = self.manifest
        if manifest.get("status") != "READY" or manifest.get("corpus_id") != corpus:
            raise RuntimeError("완료된 동일 corpus의 BM25 인덱스가 아닙니다.")
        if manifest["source_sha256"] != digest(source_path):
            raise RuntimeError("BM25와 현재 청크의 버전이 다릅니다.")
        path = directory / "tokenized_documents.json"
        if digest(path) != manifest["documents_sha256"]:
            raise RuntimeError("BM25 토큰 자료가 변경됐습니다.")
        if manifest["algorithm"] != "BM25Okapi" or manifest["rank_bm25_version"] != version("rank-bm25"):
            raise RuntimeError("BM25 알고리즘 또는 패키지 버전이 다릅니다.")
        self.tokenizer = KoreanTokenizer()
        if manifest["tokenizer"] != self.tokenizer.describe():
            raise RuntimeError("BM25 인덱스와 질의 토큰화 설정·버전이 다릅니다.")
        if manifest["tokenizer_code_sha256"] != digest(Path(__file__).parents[1] / "preprocessing/tokenizer.py"):
            raise RuntimeError("인덱스 생성 후 토큰화 코드가 변경됐습니다.")
        documents = json.loads(path.read_text(encoding="utf-8"))
        self.chunk_map = {row["chunk_id"]: row for row in chunks}
        self.chunk_ids = [row["chunk_id"] for row in documents]
        if (
            len(documents) != manifest["document_count"]
            or len(self.chunk_ids) != len(set(self.chunk_ids))
            or set(self.chunk_ids) != set(self.chunk_map)
        ):
            raise RuntimeError("BM25 인덱스의 문서 수 또는 ID가 다릅니다.")
        self.bm25 = BM25Okapi([row["tokens"] for row in documents], **manifest["parameters"])

    def search(self, text, top_k=5):
        if top_k < 1:
            raise ValueError("top_k는 1 이상이어야 합니다.")
        tokens = self.tokenizer.tokenize(text)
        if not tokens or not any(token in self.bm25.idf for token in tokens):
            return []
        scores = self.bm25.get_scores(tokens)
        candidates = [i for i, score in enumerate(scores) if math.isfinite(float(score)) and score > 0]
        candidates.sort(key=lambda i: (-float(scores[i]), self.chunk_ids[i]))
        results = []
        for rank, i in enumerate(candidates[:top_k], 1):
            row = self.chunk_map[self.chunk_ids[i]]
            results.append({
                "rank": rank, "chunk_id": row["chunk_id"],
                "bm25_score": float(scores[i]), "content": row["content"],
                "metadata": {key: row[key] for key in METADATA_KEYS if row.get(key) is not None},
            })
        return results
