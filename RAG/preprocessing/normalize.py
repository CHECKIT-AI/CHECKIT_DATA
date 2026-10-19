import re
from typing import Any


TEXT_KEYS = ("조문내용", "항내용", "호내용", "목내용")
CHILD_KEYS = ("항", "호", "목")


def as_list(value):
    """객체 하나와 리스트를 동일하게 처리한다."""
    if value is None or value == "":
        return []
    return value if isinstance(value, list) else [value]


def text_parts(value):
    """문자열 또는 문자열 목록을 처리한다."""
    if value is None:
        return []

    if isinstance(value, str):
        return [value.strip()] if value.strip() else []

    if isinstance(value, list):
        return [
            text
            for item in value
            for text in text_parts(item)
        ]

    raise ValueError("예상하지 못한 본문 값의 형식입니다.")


def flatten_article(node, level=0):
    """조문 → 항 → 호 → 목 순서로 본문을 합친다."""
    parts = text_parts(node.get(TEXT_KEYS[level]))

    if level < len(CHILD_KEYS):
        children = as_list(node.get(CHILD_KEYS[level]))

        for child in children:
            if not isinstance(child, dict):
                raise ValueError("예상하지 못한 하위 조문 구조입니다.")

            parts.extend(flatten_article(child, level + 1))

    return parts


def normalize_law(
    payload: dict[str, Any],
    mst: str,
) -> list[dict[str, Any]]:
    """법령 본문 응답을 조문별 레코드로 변환한다."""
    law = payload["법령"]
    basic = law["기본정보"]

    law_id = str(basic["법령ID"])
    law_name = basic["법령명_한글"]
    snapshot_date = str(basic["시행일자"])

    rows = as_list(law["조문"]["조문단위"])

    records = []
    seen_ids = set()

    for row in rows:
        # 장·절 제목 등은 조문 레코드에서 제외
        if row.get("조문여부") != "조문":
            continue

        number = str(row["조문번호"])
        branch = row.get("조문가지번호")

        article_no = f"제{number}조"
        if branch:
            article_no += f"의{branch}"

        article_key = str(row["조문키"])
        article_id = (
            f"{law_id}_{mst}_{snapshot_date}_{article_key}"
        )

        if article_id in seen_ids:
            raise ValueError(f"중복 조문 ID: {article_id}")
        seen_ids.add(article_id)

        content = "\n".join(flatten_article(row))

        if not content:
            raise ValueError(f"빈 조문: {article_no}")

        is_deleted = bool(
            re.match(
                r"^제\d+조(?:의\d+)?\s+삭제",
                content,
            )
        )

        records.append({
            "article_id": article_id,
            "law_id": law_id,
            "version_id": str(mst),
            "law_name": law_name,
            "article_key": article_key,
            "article_no": article_no,
            "article_title": row.get("조문제목", ""),
            "content": content,
            "snapshot_effective_date": snapshot_date,
            "article_effective_date": row.get("조문시행일자"),
            "is_deleted": is_deleted,
            "source_url": (
                "https://www.law.go.kr/LSW/lsInfoP.do"
                f"?lsiSeq={mst}"
            ),
        })

    if not records:
        raise ValueError("변환할 실제 조문이 없습니다.")

    return records