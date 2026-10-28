"""질문에 명시한 연도와 연간 고시의 적용 기간만 대조한다. 법령 이력 복원은 아님."""
import re


def explicit_years(text):
    return {int(match.group(1)) for match in re.finditer(
        r"(?<!\d)(20\d{2})\s*(?:년|[-./]\d{1,2}[-./]\d{1,2})", text)}


def filter_notice_years(results, years):
    if not years:
        return results
    output = []
    for row in results:
        metadata = row["metadata"]
        if metadata.get("source_type") == "administrative_notice":
            start, end = metadata.get("valid_from"), metadata.get("valid_to")
            # 기간 정보가 없는 고시를 요청 연도에 적용 가능한 것으로 추정하지 않는다.
            if not start or not end or not any(int(start[:4]) <= year <= int(end[:4]) for year in years):
                continue
        output.append({**row, "rank": len(output) + 1} if "rank" in row else row)
    return output
