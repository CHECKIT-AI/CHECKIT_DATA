"""분류 라벨은 근거 후보의 작은 우선순위 신호다. 법령 필터·위법 판정 규칙이 아니다."""
LABELS = (
    "임금", "근로시간", "휴일", "휴가", "근무 장소·업무", "계약기간", "계약 종료",
    "손해배상·위약금", "기타 근로조건", "그 외",
)
POLICY_VERSION = "labels_soft_priority_v1"

# 서비스 주제와 조문 사이의 개발용 연결. 적용 요건에 대한 전문가 검수는 별도다.
LABOR_ARTICLES = {
    "임금": {"제17조", "제36조", "제37조", "제43조", "제44조", "제44조의2", "제44조의3", "제45조", "제46조", "제47조", "제48조", "제49조"},
    "근로시간": {"제17조", "제50조", "제51조", "제51조의2", "제52조", "제53조", "제54조", "제56조", "제57조", "제58조", "제59조", "제63조", "제69조"},
    "휴일": {"제17조", "제55조", "제56조", "제63조", "제70조"},
    "휴가": {"제17조", "제60조", "제61조", "제62조", "제73조", "제74조"},
    "근무 장소·업무": {"제17조"},
    "계약기간": {"제17조", "제23조"},
    "계약 종료": {"제23조", "제24조", "제26조", "제27조", "제28조", "제30조", "제36조", "제37조"},
    "손해배상·위약금": {"제19조", "제20조", "제21조"},
}


def validate_labels(labels):
    if not isinstance(labels, list) or len(labels) > len(LABELS):
        raise ValueError("분류 라벨은 0~10개의 목록이어야 합니다.")
    if any(not isinstance(label, str) or label not in LABELS for label in labels):
        raise ValueError("지원하지 않는 분류 라벨입니다.")
    if len(set(labels)) != len(labels):
        raise ValueError("중복 분류 라벨입니다.")
    return labels


def matched_labels(metadata, labels):
    law, number = metadata.get("law_name", ""), metadata.get("article_no", "")
    title = metadata.get("article_title", "")
    matches = []
    for label in labels:
        match = law == "근로기준법" and number in LABOR_ARTICLES.get(label, set())
        if label == "임금":
            match |= law.startswith("최저임금법") or law.endswith("적용 최저임금 고시")
        if label == "근로시간":
            match |= any(term in title for term in ["근로시간", "휴게", "단시간근로자의 근로조건"])
        if label == "휴일":
            match |= law == "관공서의 공휴일에 관한 규정" or (law == "근로기준법 시행령" and number in {"제30조", "별표 2"})
        if label == "휴가":
            match |= any(term in title for term in ["연차", "유급휴가", "단시간근로자의 근로조건"])
        if label == "근무 장소·업무":
            match |= (law == "근로기준법 시행령" and number == "제8조") or (law == "기간제 및 단시간근로자 보호 등에 관한 법률" and number == "제17조")
        if label == "계약기간":
            match |= law.startswith("기간제 및 단시간근로자") or "수습" in title
        if label == "계약 종료":
            match |= any(term in title for term in ["해고", "퇴직금의 지급", "금품 청산"])
        if label == "손해배상·위약금":
            match |= any(term in title for term in ["위약", "손해배상", "전차금 상계"])
        if label == "기타 근로조건":
            match |= law.startswith("근로자퇴직급여 보장법")
        if match:
            matches.append(label)
    return matches


def prioritize(results, labels):
    """전체 검색 후보를 보존하고 점수에 제한된 보너스만 적용. 라벨 개수로 가중치 누적 안 함."""
    validate_labels(labels)
    output = []
    for row in results:
        matches = matched_labels(row["metadata"], labels)
        if row.get("reranker_score") is not None:
            score, bonus = row["reranker_score"], 0.05
        elif row.get("cosine_distance") is not None:
            score, bonus = -row["cosine_distance"], 0.015
        else:
            score, bonus = -float(row["rank"]), 0.0
        output.append({**row, "base_rank": row["rank"], "matched_labels": matches,
                       "label_priority_score": float(score) + (bonus if matches else 0.0)})
    output.sort(key=lambda row: (-row["label_priority_score"], row["base_rank"]))
    return [{**row, "rank": rank} for rank, row in enumerate(output, 1)]
