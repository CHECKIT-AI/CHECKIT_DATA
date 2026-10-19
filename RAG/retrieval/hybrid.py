"""벡터·BM25 점수를 섞지 않고 순위를 결합하는 가중치를 지정할 수 있는 RRF."""


import math


def reciprocal_rank_fusion(dense_results, bm25_results, top_k=10, rrf_k=60, dense_weight=1.0, bm25_weight=1.0):
    if top_k < 1 or rrf_k < 1:
        raise ValueError("top_k와 rrf_k는 1 이상이어야 합니다.")
    if not all(math.isfinite(w) and w > 0 for w in [dense_weight, bm25_weight]):
        raise ValueError("RRF 가중치는 유한한 양수여야 합니다.")
    weights = {"dense": dense_weight, "bm25": bm25_weight}
    combined = {}
    for name, results in [("dense", dense_results), ("bm25", bm25_results)]:
        seen = set()
        for rank, result in enumerate(results, 1):
            chunk_id = result["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            if chunk_id not in combined:
                combined[chunk_id] = {
                    "chunk_id": chunk_id, "content": result["content"],
                    "metadata": result["metadata"], "rrf_score": 0.0,
                    "dense_rank": None, "bm25_rank": None,
                    "cosine_distance": None, "bm25_score": None,
                }
            item = combined[chunk_id]
            item["rrf_score"] += weights[name] / (rrf_k + rank)
            item[f"{name}_rank"] = rank
            field = "cosine_distance" if name == "dense" else "bm25_score"
            item[field] = result.get(field)
    ordered = sorted(combined.values(), key=lambda row: (-row["rrf_score"], row["chunk_id"]))
    return [{**row, "rank": rank} for rank, row in enumerate(ordered[:top_k], 1)]
