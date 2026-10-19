import argparse
import json
from pathlib import Path

from RAG.collection.law_api import fetch_law


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("search_file", type=Path)
    parser.add_argument("--law-name", default="근로기준법")
    args = parser.parse_args()

    # 1. 저장한 목록 응답 읽기
    payload = json.loads(
        args.search_file.read_text(encoding="utf-8")
    )

    search = payload.get("LawSearch")

    if not isinstance(search, dict):
        raise RuntimeError("LawSearch가 없는 목록 응답입니다.")

    if str(search.get("resultCode")) != "00":
        raise RuntimeError("정상적인 목록 조회 응답이 아닙니다.")

    rows = search.get("law", [])

    # 결과가 하나면 객체로 내려오는 경우도 처리
    if isinstance(rows, dict):
        rows = [rows]

    # 2. 정확한 법령명으로 선택
    matches = [
        row
        for row in rows
        if row.get("법령명한글") == args.law_name
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"정확히 일치하는 법령이 {len(matches)}개입니다. "
            "목록과 버전을 직접 확인하세요."
        )

    selected = matches[0]

    mst = str(selected["법령일련번호"])
    effective_date = str(selected["시행일자"])
    law_id = str(selected["법령ID"])

    output_path = args.search_file.parent / (
        f"law_{law_id}_{mst}_{effective_date}_body.json"
    )

    if output_path.exists():
        raise FileExistsError(
            f"이미 본문 파일이 있습니다: {output_path}"
        )

    # 3. 선택한 버전의 본문 조회
    body = fetch_law(
        mst=mst,
        effective_date=effective_date,
    )

    # 정상 본문에 필요한 최상위 구조 확인
    law = body.get("법령")

    if not isinstance(law, dict) or not law.get("조문"):
        raise RuntimeError(
            "예상한 법령·조문 구조가 없습니다. "
            "오류 응답인지 또는 응답 구조가 다른지 확인하세요."
        )

    # 4. 본문 원본 저장
    output_path.write_text(
        json.dumps(body, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"선택 법령: {args.law_name}")
    print(f"MST: {mst}, 시행일자: {effective_date}")
    print(f"본문 응답 저장: {output_path}")


if __name__ == "__main__":
    main()