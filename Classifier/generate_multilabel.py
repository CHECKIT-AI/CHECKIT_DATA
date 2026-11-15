"""Generate split-disjoint synthetic two-topic contract clauses."""

import hashlib
import json
import random
from itertools import product
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/classifier/multilabel_aug_20261115"
LABELS = (
    "임금", "근로시간", "휴일", "휴가", "근무 장소·업무",
    "계약기간", "계약 종료", "손해배상·위약금", "기타 근로조건",
)

# Each topic has six training phrasings, two development phrasings, and two test phrasings.
PHRASES = {
    "임금": [
        "기본급은 매달 25일 지급한다", "성과 인센티브는 분기별로 계산한다",
        "급여를 근로자 명의 계좌로 이체한다", "직무수당을 월 12만원으로 정한다",
        "임금명세서에 지급 항목별 금액을 기재한다", "상여금은 연말 평가에 따라 지급한다",
        "월급은 280만원으로 약정한다", "매월 급여명세서를 전자문서로 교부한다",
        "해당 월 급여는 다음 달 5일까지 이체한다", "판매 실적에 따른 수당은 월말에 정산한다",
    ],
    "근로시간": [
        "하루 소정근로시간은 7시간으로 한다", "출근 시각은 오전 8시 30분이다",
        "오후 근무 중 30분의 휴게시간을 준다", "교대조의 근무시간표를 매주 확정한다",
        "업무 준비를 위한 대기시간도 근무시간에 포함한다", "주간 근무는 월요일부터 금요일까지 실시한다",
        "시업과 종업 시각을 전자 출퇴근기록에 남긴다", "하루 업무 사이에 1시간의 휴게를 보장한다",
        "주당 소정근로시간을 35시간으로 정한다", "오전조는 6시에 시작해 14시에 마친다",
    ],
    "휴일": [
        "매주 화요일을 유급 주휴일로 한다", "법정 공휴일에는 근무를 배정하지 않는다",
        "정기 휴무일은 매주 일요일이다", "공휴일이 겹치면 대체휴일을 부여한다",
        "주휴일을 변경할 때에는 사전에 알려 준다", "명절 당일은 유급휴일로 처리한다",
        "매주 금요일을 주휴일로 지정한다", "관공서 공휴일은 유급으로 쉰다",
        "주휴일은 한 주에 하루 이상 부여한다", "휴일 대체 일정은 사전 서면 합의에 따른다",
    ],
    "휴가": [
        "연차휴가는 근로자의 신청에 따라 사용한다", "경조사가 발생하면 특별휴가를 준다",
        "발생한 유급휴가의 잔여 일수를 매월 안내한다", "병가 신청에는 진단서 제출을 요청한다",
        "사용하지 않은 연차는 정해진 절차에 따라 이월한다", "휴가 신청은 인사시스템에 등록한다",
        "연차유급휴가 사용 계획을 분기 초에 협의한다", "가족 행사에 필요한 경조휴가를 허용한다",
        "입사 첫해 발생한 연차를 사용하도록 안내한다", "질병 치료를 위한 병가를 신청할 수 있다",
    ],
    "근무 장소·업무": [
        "근무지는 부산의 고객지원센터로 정한다", "재고 조사와 입출고 기록 업무를 맡는다",
        "현장 배치 장소는 인천 제2공장이다", "담당 직무는 장비 정비와 점검이다",
        "승인된 재택근무 장소에서 자료 분석을 수행한다", "고객사 사무실에서 시스템 운영을 담당한다",
        "주된 일터는 광주 물류창고다", "담당 업무는 상담 이력 검토와 보고서 작성이다",
        "대전 연구실에서 시료 관리 업무를 수행한다", "회계 자료의 입력과 검증을 담당한다",
    ],
    "계약기간": [
        "근로계약기간은 2027년 1월부터 2028년 1월까지다", "처음 두 달은 수습기간으로 한다",
        "계약은 2027년 4월 1일에 시작한다", "고용기간의 만료일은 2028년 3월 31일이다",
        "기간의 정함 없이 채용한다", "계약 연장은 만료 전에 별도로 협의한다",
        "수습기간은 채용일부터 3개월이다", "이 계약의 종료 예정일은 2029년 6월 30일이다",
        "채용일로부터 1년을 계약기간으로 정한다", "고용기간을 별도로 제한하지 않는다",
    ],
    "계약 종료": [
        "사직 의사는 퇴직 희망일 전에 서면으로 통보한다", "해고 사유를 서면으로 알려 준다",
        "계약 종료 시 업무 인계 절차를 진행한다", "합의해지는 양측이 종료일을 정하여 서명한다",
        "징계해고 전에 소명 기회를 제공한다", "고용관계 종료는 정해진 해고 절차에 따른다",
        "퇴직하려면 인사 담당자에게 사직서를 제출한다", "해고 통지서에 종료 사유와 날짜를 적는다",
        "정년 도달 시 고용관계를 종료한다", "근로관계 해지는 당사자 간 서면 합의에 따른다",
    ],
    "손해배상·위약금": [
        "고의로 설비를 파손하면 확인된 수리비를 배상한다", "의무를 위반하면 정액 위약금을 낸다",
        "재고 손실에 대한 손해배상 범위를 협의한다", "무단 반출한 물품의 가액을 배상한다",
        "작업 장비 고장에 따른 실제 손해액을 산정한다", "업무상 과실에 따른 배상 책임을 따로 정한다",
        "장비 분실 시 손해배상액을 조사한다", "위반 행위마다 20만원의 벌금을 부과한다",
        "파손된 비품의 배상액은 확인된 손실을 기준으로 한다", "계약 위반 시 예정된 위약금 50만원을 지급한다",
    ],
    "기타 근로조건": [
        "고객의 비공개 정보를 외부에 누설하지 않는다", "법정 사회보험 가입을 유지한다",
        "퇴직연금은 별도 규약에 따라 운영한다", "겸직으로 인한 이해충돌을 미리 신고한다",
        "업무용 계정의 비밀번호를 타인에게 공유하지 않는다", "경쟁업체와의 이해관계를 회사에 알린다",
        "업무 중 취득한 영업비밀을 보호한다", "산재보험 가입 내역을 근로자에게 안내한다",
        "퇴직급여제도의 가입 절차를 설명한다", "외부 업무를 병행하려면 겸직 사실을 신고한다",
    ],
}

PAIRS = (
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 8),
    (2, 3), (2, 6), (3, 6), (3, 8), (4, 8), (5, 6), (5, 8), (6, 7), (7, 8),
)


def write(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    assert len(PHRASES) == len(LABELS) == 9 and all(len(PHRASES[label]) == 10 for label in LABELS)
    rows = {"train": [], "validation": [], "test": []}
    for pair_no, (first, second) in enumerate(PAIRS, 1):
        labels = [LABELS[first], LABELS[second]]
        for split, positions, count in (
            ("train", range(6), 24), ("validation", range(6, 8), 4), ("test", range(8, 10), 4)
        ):
            combinations = list(product(positions, repeat=2))
            random.Random(20261115 + pair_no).shuffle(combinations)
            for item_no, (i, j) in enumerate(combinations[:count], 1):
                left, right = PHRASES[labels[0]][i], PHRASES[labels[1]][j]
                text = (left + ". " + right if item_no % 2 else right + "; " + left) + "."
                rows[split].append({
                    "id": f"ML26_{split}_{pair_no:02d}_{item_no:02d}",
                    "text": text, "labels": labels,
                    "label_vec": [int(label in labels) for label in (*LABELS, "그 외")],
                    "group": f"pair_{pair_no:02d}_{split}", "source": "authored_multi_composition_v1",
                    "annotation_status": "PROVISIONAL_UNREVIEWED",
                })
    assert {name: len(items) for name, items in rows.items()} == {
        "train": 552, "validation": 92, "test": 92,
    }
    texts = {name: {row["text"] for row in items} for name, items in rows.items()}
    assert all(len(texts[name]) == len(rows[name]) for name in rows)
    assert not any(texts[a] & texts[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
    frozen = [json.loads(line)["text"] for line in
              (ROOT / "test/classifier/novel_clause_60_20261115/gold.jsonl").open(encoding="utf-8")]
    assert not set(frozen) & set.union(*texts.values())
    OUTPUT.mkdir(parents=True)
    hashes = {name: write(OUTPUT / f"{name}.jsonl", items) for name, items in rows.items()}
    manifest = {
        "status": "PROVISIONAL_UNREVIEWED", "generator": "Classifier/generate_multilabel.py",
        "seed": 20261115, "pairs": [[LABELS[a], LABELS[b]] for a, b in PAIRS],
        "counts": {name: len(items) for name, items in rows.items()},
        "sha256": hashes, "frozen_60_overlap": 0,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "counts": manifest["counts"], "sha256": hashes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
