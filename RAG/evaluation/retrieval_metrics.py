"""조문 단위 검색 지표. 여러 청크가 같은 조문이면 처음 나온 순위만 사용."""


def score_query(results, relevant_article_ids, ks=(1, 3, 5, 10)):
    gold = set(relevant_article_ids)
    if not gold:
        raise ValueError("정답 조문이 하나 이상 필요합니다.")
    ranking, seen = [], set()
    for result in results:
        article_id = result["metadata"]["article_id"]
        if article_id not in seen:
            seen.add(article_id)
            ranking.append(article_id)
    first = next((i for i, aid in enumerate(ranking, 1) if aid in gold), None)
    scores = {"first_relevant_rank": first}
    for k in ks:
        found = gold.intersection(ranking[:k])
        scores[f"hit@{k}"] = float(bool(found))
        scores[f"recall@{k}"] = len(found) / len(gold)
    cutoff = max(ks)
    scores[f"rr@{cutoff}"] = 1.0 / first if first is not None and first <= cutoff else 0.0
    return scores


def summarize(rows, ks=(1, 3, 5, 10)):
    if not rows:
        raise ValueError("평가 결과가 없습니다.")
    summary = {"query_count": len(rows)}
    for k in ks:
        for name in ("hit", "recall"):
            field = f"{name}@{k}"
            summary[field] = sum(row["metrics"][field] for row in rows) / len(rows)
    cutoff = max(ks)
    summary[f"mrr@{cutoff}"] = sum(row["metrics"][f"rr@{cutoff}"] for row in rows) / len(rows)
    summary["mean_query_ms"] = sum(row["elapsed_ms"] for row in rows) / len(rows)
    return summary
