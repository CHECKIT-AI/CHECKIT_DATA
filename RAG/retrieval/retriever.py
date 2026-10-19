"""dense / bm25 / hybrid를 같은 인터페이스로 호출한다."""

import json
import math
from pathlib import Path

from RAG.indexing.bm25 import BM25Retriever, digest
# from RAG.indexing.embeddings import BGEEmbedder
from RAG.indexing.embeddings import create_embedder
from RAG.indexing.vector_store import COLLECTION_NAME, LawVectorStore
from RAG.pipelines.build_index import corpus_name, load_chunks
from RAG.retrieval.hybrid import reciprocal_rank_fusion
from RAG.settings import DATA_ROOT, STORAGE_ROOT, DEFAULT_CORPUS
from RAG.retrieval.reranker import BGEReranker, rerank_candidates


class LawRetriever:
    def __init__(self, corpus=DEFAULT_CORPUS, method="dense", device="cpu", candidate_k=20, rrf_k=60, chroma_directory=None, dense_weight=0.8, bm25_weight=0.2, reranker_batch_size=2):
        corpus_name(corpus)
        if method not in {"dense", "bm25", "hybrid", "weighted_rrf", "hybrid_rerank"}:
            raise ValueError("지원하지 않는 검색 방식입니다.")
        if candidate_k < 1 or rrf_k < 1:
            raise ValueError("candidate_k와 rrf_k는 1 이상이어야 합니다.")
        if not all(math.isfinite(w) and w > 0 for w in [dense_weight, bm25_weight]) or reranker_batch_size < 1:
            raise ValueError("가중치는 유한한 양수, reranker batch size는 1 이상이어야 합니다.")
        self.method, self.candidate_k, self.rrf_k = method, candidate_k, rrf_k
        self.dense_weight, self.bm25_weight = dense_weight, bm25_weight
        self.last_candidates = []
        self.reranker = None
        self.index_directory = STORAGE_ROOT / corpus
        self.manifest_path = self.index_directory / "index_manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest = self.manifest
        if (
            manifest.get("status") != "READY" or manifest.get("corpus_id") != corpus
            or manifest.get("collection_name") != COLLECTION_NAME
            or manifest.get("distance_metric") != "cosine"
            or not manifest.get("normalize_embeddings")
        ):
            raise RuntimeError("완료된 동일 corpus의 cosine 인덱스가 아닙니다.")
        self.chunk_path = DATA_ROOT / manifest["source_file"]
        if digest(self.chunk_path) != manifest["source_sha256"]:
            raise RuntimeError("인덱스 생성 후 청크 파일이 변경됐습니다.")
        self.chunks = load_chunks(self.chunk_path)
        self.embedder, self.store, self.bm25 = None, None, None
        if method in {"dense", "hybrid", "weighted_rrf", "hybrid_rerank"}:
            self.store = LawVectorStore(chroma_directory or self.index_directory / "chroma")
            if self.store.count() != manifest["chunk_count"] or self.store.ids() != {r["chunk_id"] for r in self.chunks}:
                raise RuntimeError("Chroma 문서 수 또는 ID가 입력 청크와 다릅니다.")
            # self.embedder = BGEEmbedder(model_name=manifest["model_name"], revision=manifest["model_revision"], device=device)
            self.embedder = create_embedder(
                model_name=manifest["model_name"],
                revision=manifest["model_revision"],
                device=device,
                input_type="query",
                query_instruction=manifest.get("query_instruction"),
            )
            if self.embedder.dimension != manifest["dimension"] or self.embedder.max_length != manifest["max_seq_length"]:
                raise RuntimeError("인덱스와 질의 모델 설정이 다릅니다.")
        if method in {"bm25", "hybrid", "weighted_rrf", "hybrid_rerank"}:
            self.bm25 = BM25Retriever(self.index_directory / "bm25", self.chunks, self.chunk_path, corpus)
        if method == "hybrid_rerank":
            reranker_path = STORAGE_ROOT / "models" / "bge-reranker-v2-m3.json"
            if not reranker_path.exists():
                raise FileNotFoundError("먼저 python -m RAG.pipelines.prepare_reranker 를 실행하세요.")
            prepared = json.loads(reranker_path.read_text(encoding="utf-8"))
            if prepared.get("status") != "READY":
                raise RuntimeError("reranker 모델 준비가 완료되지 않았습니다.")
            self.reranker = BGEReranker(
                revision=prepared["model_revision"], device=device,
                batch_size=reranker_batch_size, max_length=prepared["max_length"],
            )

    def describe(self):
        return {
            "method": self.method,
            "model": self.embedder.describe() if self.embedder else None,
            "candidate_k_per_retriever": self.candidate_k if self.method in {"hybrid", "weighted_rrf", "hybrid_rerank"} else None,
            "rrf_k": self.rrf_k if self.method in {"hybrid", "weighted_rrf", "hybrid_rerank"} else None,
            "rrf_weights": ([1.0, 1.0] if self.method == "hybrid" else [self.dense_weight, self.bm25_weight]) if self.method in {"hybrid", "weighted_rrf", "hybrid_rerank"} else None,
            "reranker": self.reranker.describe() if self.reranker else None,
            "reranker_candidate_policy": "entire_union_before_final_topk" if self.reranker else None,
            "bm25_manifest": self.bm25.manifest if self.bm25 else None,
        }

    def search(self, text, top_k=5):
        if not isinstance(text, str) or not text.strip() or top_k < 1:
            raise ValueError("비어 있지 않은 검색 문장과 양수 top_k가 필요합니다.")
        if self.method == "dense":
            vector = self.embedder.encode([text], batch_size=1)[0]
            self.last_candidates = self.store.search(vector, top_k=top_k)
            return self.last_candidates
        if self.method == "bm25":
            self.last_candidates = self.bm25.search(text, top_k=top_k)
            return self.last_candidates
        if self.candidate_k < top_k:
            raise ValueError("hybrid의 candidate_k는 최종 top_k 이상이어야 합니다.")
        vector = self.embedder.encode([text], batch_size=1)[0]
        dense_results = self.store.search(vector, top_k=self.candidate_k)
        lexical_results = self.bm25.search(text, top_k=self.candidate_k)
        union_size = len({row["chunk_id"] for row in dense_results + lexical_results})
        dense_weight, bm25_weight = (1.0, 1.0) if self.method == "hybrid" else (self.dense_weight, self.bm25_weight)
        candidates = reciprocal_rank_fusion(
            dense_results, lexical_results, top_k=max(1, union_size), rrf_k=self.rrf_k,
            dense_weight=dense_weight, bm25_weight=bm25_weight,
        )
        self.last_candidates = candidates
        if self.reranker:
            return rerank_candidates(text, candidates, self.reranker.score, top_k)
        return candidates[:top_k]
