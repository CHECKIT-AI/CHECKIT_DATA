"""법제처 법령 목록·본문 조회 API 클라이언트."""

from typing import Any

import requests

from RAG.settings import LAW_API_OC


BASE_URL = "https://www.law.go.kr/DRF"
TIMEOUT = (10, 60)  # 연결 제한 시간, 응답 대기 시간


class LawAPIError(RuntimeError):
    """법제처 API 호출 또는 응답 처리 오류."""


def _request_json(
    endpoint: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """인증값을 추가해 API를 호출하고 JSON 객체를 반환한다."""

    if not LAW_API_OC or not LAW_API_OC.strip():
        raise LawAPIError(
            "루트 .env 파일에 LAW_API_OC를 설정하세요."
        )

    request_params = {
        **params,
        "OC": LAW_API_OC.strip(),
        "type": "JSON",
    }

    try:
        response = requests.get(
            f"{BASE_URL}/{endpoint}",
            params=request_params,
            timeout=TIMEOUT,
        )
    except requests.Timeout:
        raise LawAPIError(
            "법제처 API 응답 시간이 초과됐습니다."
        ) from None
    except requests.RequestException:
        # 원래 예외에는 OC가 포함된 요청 URL이 들어갈 수 있으므로
        # 예외 전문이나 요청 URL을 출력하지 않는다.
        raise LawAPIError(
            "법제처 API 연결에 실패했습니다. 네트워크를 확인하세요."
        ) from None

    if response.status_code != 200:
        raise LawAPIError(
            f"법제처 API HTTP 오류: {response.status_code}"
        )

    try:
        payload = response.json()
    except ValueError:
        raise LawAPIError(
            "JSON 응답이 아닙니다. OC·사용 승인·요청 설정을 확인하세요."
        ) from None

    if not isinstance(payload, dict) or not payload:
        raise LawAPIError(
            "비어 있거나 예상과 다른 JSON 응답입니다."
        )

    # API 원본 구조를 유지해 반환한다.
    # JSON 형태의 오류 응답인지, 정상 데이터인지는
    # 수집 단계에서 예상 필드와 함께 확인한다.
    return payload


def search_laws(
    law_name: str,
    *,
    page: int = 1,
    display: int = 100,
) -> dict[str, Any]:
    """법령명으로 현행 법령 목록의 한 페이지를 조회한다."""

    if not law_name.strip():
        raise ValueError("법령명을 입력하세요.")

    if page < 1:
        raise ValueError("page는 1 이상이어야 합니다.")

    if not 1 <= display <= 100:
        raise ValueError("display는 1~100이어야 합니다.")

    return _request_json(
        "lawSearch.do",
        {
            "target": "eflaw",
            "search": 1,  # 법령명 검색
            "query": law_name.strip(),
            "nw": 3,      # 현행 법령
            "page": page,
            "display": display,
        },
    )


def fetch_law(
    mst: str,
    effective_date: str,
) -> dict[str, Any]:
    """MST와 시행일자로 법령 본문 전체를 조회한다.

    Args:
        mst: 목록 응답의 '법령일련번호'.
        effective_date: 목록 응답의 '시행일자'. YYYYMMDD 형식.
    """

    mst = str(mst).strip()
    effective_date = str(effective_date).strip()

    if not mst.isdigit():
        raise ValueError("mst는 숫자로 된 법령일련번호여야 합니다.")

    if len(effective_date) != 8 or not effective_date.isdigit():
        raise ValueError("effective_date는 YYYYMMDD 형식이어야 합니다.")

    return _request_json(
        "lawService.do",
        {
            "target": "eflaw",
            "MST": mst,
            "efYd": effective_date,
            # JO를 지정하지 않아 전체 조문을 요청한다.
        },
    )