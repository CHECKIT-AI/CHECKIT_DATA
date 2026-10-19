"""정규화한 조문을 검색용 청크로 변환한다."""


def build_chunks(articles: list[dict]) -> list[dict]:
    chunks = []
    seen_ids = set()

    for article in articles:
        # 삭제 여부가 누락된 데이터는 조용히 처리하지 않는다.
        if not isinstance(article.get("is_deleted"), bool):
            raise ValueError("is_deleted 값이 없거나 형식이 잘못됐습니다.")

        # 삭제 조문은 검색 대상에서 제외
        if article["is_deleted"]:
            continue

        required_fields = (
            "article_id",
            "law_name",
            "article_no",
            "content",
        )

        for field in required_fields:
            value = article.get(field)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"필수 필드가 없거나 비어 있습니다: {field}"
                )

        article_id = article["article_id"]
        chunk_id = f"{article_id}::full"

        if chunk_id in seen_ids:
            raise ValueError(f"중복 청크 ID: {chunk_id}")

        seen_ids.add(chunk_id)

        # 모델이 법령명과 조문 맥락을 함께 읽도록 구성
        embedding_text = "\n".join([
            f"법령명: {article['law_name']}",
            f"조문: {article['article_no']}",
            f"제목: {article.get('article_title', '')}",
            "본문:",
            article["content"],
        ])

        # 원래 조문의 메타데이터와 본문을 그대로 보존
        chunk = {
            **article,
            "chunk_id": chunk_id,
            "chunk_type": "article",
            "embedding_text": embedding_text,
        }

        chunks.append(chunk)

    return chunks