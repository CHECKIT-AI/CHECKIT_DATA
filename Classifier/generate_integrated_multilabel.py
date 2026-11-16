"""Create single-sentence multi-topic clauses with disjoint wording by split."""

import hashlib
import json
from pathlib import Path

from Classifier.train import ROOT, read_jsonl


OUTPUT = ROOT / "data/classifier/integrated_multilabel_20261115_v001"
LABELS = json.loads((ROOT / "outputs/classifier/combined_split_v004/manifest.json").read_text(encoding="utf-8"))["labels"]

# Two training wordings, one development wording, and one test wording per label pair.
TEMPLATES = [
    (("임금", "근로시간"), (
        "{hour}시까지 이어진 연장근무의 가산수당은 다음 급여일에 지급한다",
        "소정근로시간을 넘긴 {months}시간분 임금은 별도 항목으로 계산한다",
        "퇴근 시각 이후 일한 {months}시간은 시간외수당으로 정산한다",
        "근무표상 종료 시각을 지난 업무 {months}시간분은 추가 임금으로 지급한다",
    )),
    (("임금", "휴일"), (
        "주휴일에 출근한 경우 가산임금을 다음 달 {day}일에 지급한다",
        "유급휴일 근무에 대한 수당은 매월 {day}일 별도 정산한다",
        "공휴일에 제공한 근로의 대가는 급여와 함께 {day}일에 지급한다",
        "휴일근로로 발생한 추가 임금은 매달 {day}일 임금명세서에 표시한다",
    )),
    (("임금", "휴가"), (
        "남은 연차 {months}일에 대한 미사용수당을 다음 급여일에 정산한다",
        "사용하지 못한 유급휴가 {months}일분은 임금과 구분하여 지급한다",
        "연차휴가 미사용분 {months}일의 금액을 월급 지급일에 함께 이체한다",
        "미사용 연차 {months}일에 해당하는 수당을 임금명세서에 적는다",
    )),
    (("임금", "근무 장소·업무"), (
        "제{n}현장에 배치된 직원에게는 월 {amount}만원의 현장수당을 지급한다",
        "상담 업무를 맡는 제{n}센터 직원의 직무수당을 월 {amount}만원으로 정한다",
        "제{n}공장의 품질검사 담당자에게 별도 직무급을 지급한다",
        "재택 자료분석 업무에 배정된 근로자에게 월 {amount}만원의 업무수당을 준다",
    )),
    (("임금", "계약기간"), (
        "수습 {months}개월 동안 기본급을 월 {amount}만원으로 정한다",
        "계약 첫 {months}개월에 지급하는 급여는 월 {amount}만원이다",
        "고용 시작일부터 {months}개월간 적용되는 임금은 월 {amount}만원이다",
        "기간제 계약의 첫 {months}개월은 기본급 {amount}만원을 지급한다",
    )),
    (("임금", "계약 종료"), (
        "퇴직일까지 발생한 임금은 근로관계 종료 후 {day}일 이내 지급한다",
        "해고가 확정되면 마지막 근무일까지의 급여를 {day}일 안에 정산한다",
        "사직일을 기준으로 미지급 급여를 {day}일 이내 이체한다",
        "합의해지일 이후 {day}일 안에 남은 임금을 전액 지급한다",
    )),
    (("근로시간", "휴일"), (
        "매주 일요일을 쉬는 날로 하되 다음 주 근무표에는 {hour}시 시업을 적용한다",
        "유급 주휴일을 수요일로 지정하고 나머지 근무일은 {hour}시에 시작한다",
        "휴일 대체일을 금요일로 정하면 그 주의 시업 시각을 {hour}시로 다시 표시한다",
        "공휴일 근무를 배정할 때에는 해당 주의 소정근로시간을 {hours}시간으로 다시 계산한다",
    )),
    (("근로시간", "근무 장소·업무"), (
        "제{n}공장에서 {hour}시에 시작하는 교대 포장 업무를 맡는다",
        "부산 제{n}센터의 상담 업무는 {hour}시부터 수행한다",
        "재택 자료검수 업무는 매일 {hour}시에 시작한다",
        "제{n}물류창고의 검품 작업은 {hour}시에 마친다",
    )),
    (("근로시간", "계약기간"), (
        "수습 {months}개월 동안 주당 소정근로시간을 {hours}시간으로 정한다",
        "계약 시작 후 {months}개월은 {hour}시부터 일한다",
        "계약기간 첫 {months}개월에 적용할 교대근무시간은 주 {hours}시간이다",
        "고용 개시일부터 {months}개월간 하루 근무시간을 {day_hours}시간으로 정한다",
    )),
    (("휴일", "휴가"), (
        "유급휴일과 연차휴가가 겹치는 달에는 연차 {months}일을 별도로 보전한다",
        "공휴일과 승인된 연차 {months}일이 겹치면 휴가 잔여일을 차감하지 않는다",
        "주휴일인 {day}일에 연차를 신청한 경우 해당 연차 일수를 그대로 남겨 둔다",
        "법정 휴일과 휴가 사용일이 일치하면 연차 {months}일을 다시 배정한다",
    )),
    (("휴가", "계약 종료"), (
        "퇴직 예정일 {day}일 전까지 남은 연차의 사용 일정을 협의한다",
        "사직서를 제출한 근로자는 종료일 {day}일 전까지 미사용 휴가를 신청할 수 있다",
        "해고 통지 후 남아 있는 연차휴가 {months}일의 사용 시기를 근로자와 협의한다",
        "합의해지일 전에 잔여 유급휴가 {months}일을 사용할 수 있도록 한다",
    )),
    (("근무 장소·업무", "기타 근로조건"), (
        "제{n}사업장에서 고객정보를 취급하는 직원은 비밀유지 의무를 지킨다",
        "재택으로 제{n}고객사의 회계 자료를 검토하는 근로자는 업무용 계정의 보안규칙을 따른다",
        "제{n}센터에서 상담 기록을 처리하는 직원은 비공개 정보를 누설하지 않는다",
        "현장 유지보수 담당자는 제{n}고객사 사무실에서 알게 된 영업비밀을 보호한다",
    )),
    (("계약기간", "계약 종료"), (
        "계약 만료일 {day}일 전에 당사자가 합의하면 그날 근로관계를 종료한다",
        "수습 {months}개월이 끝나기 전에 해지하려면 종료 사유를 서면으로 알린다",
        "기간제 고용의 만료일 {day}일 전에 종료 절차를 근로자에게 통지한다",
        "계약 시작 후 {months}개월 안에 해고할 때에는 종료일을 서면에 적는다",
    )),
    (("계약기간", "기타 근로조건"), (
        "수습 {months}개월 동안에도 산재보험 가입을 유지한다",
        "기간제 계약을 시작한 날부터 {months}개월 동안 퇴직연금 가입 절차를 진행한다",
        "계약 첫 {months}개월에도 고객정보 비밀유지 의무를 적용한다",
        "고용기간이 {months}개월인 근로자에게 사회보험 가입을 안내한다",
    )),
    (("계약 종료", "손해배상·위약금"), (
        "중도 퇴사하면서 인수인계를 하지 않으면 {amount}만원의 정액 위약금을 지급한다",
        "해고 통지 후 업무 자료 {n}건을 고의로 훼손하면 확인된 손해액을 배상한다",
        "퇴직 절차를 무단으로 중단하면 예정 손해배상액 {amount}만원을 낸다",
        "사직일 이전에 장비 {n}대를 고의로 파손한 경우 퇴직 뒤에도 수리비를 배상한다",
    )),
]


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = {"train": [], "validation": [], "test": []}
    for pair_no, (labels, templates) in enumerate(TEMPLATES, 1):
        assert len(templates) == 4 and all(label in LABELS for label in labels)
        for split, template_indices, values in (
            ("train", (0, 1), range(1, 6)),
            ("validation", (2,), range(6, 8)),
            ("test", (3,), range(8, 10)),
        ):
            for template_index in template_indices:
                for n in values:
                    text = templates[template_index].format(
                        n=n, hour=n + 8, months=n + 2, day=n + 9,
                        amount=200 + n * 10, hours=n + 20, day_hours=5 + n % 4,
                    ) + "."
                    rows[split].append({
                        "id": f"INT26_{split}_{pair_no:02d}_{template_index}_{n}",
                        "text": text, "labels": list(labels),
                        "label_vec": [int(label in labels) for label in LABELS],
                        "group": f"integrated_{pair_no:02d}_{split}",
                        "source": "authored_integrated_multi_v1",
                        "annotation_status": "PROVISIONAL_UNREVIEWED",
                    })
    assert {split: len(items) for split, items in rows.items()} == {
        "train": 150, "validation": 30, "test": 30,
    }
    prior = set()
    for path in [
        ROOT / "test/classifier/novel_clause_60_20261115/gold.jsonl",
        ROOT / "outputs/rag/pdf40_pre_llm_20261115/segments_clause_only.jsonl",
        *(ROOT / "data/classifier/multilabel_aug_20261115").glob("*.jsonl"),
        *(ROOT / "outputs/classifier/combined_split_v004").glob("*.jsonl"),
    ]:
        prior.update(row["text"] for row in read_jsonl(path))
    texts = {split: {row["text"] for row in items} for split, items in rows.items()}
    assert all(len(texts[split]) == len(rows[split]) and not texts[split] & prior for split in rows)
    assert all(not texts[a] & texts[b] for a, b in
               (("train", "validation"), ("train", "test"), ("validation", "test")))
    OUTPUT.mkdir(parents=True)
    hashes = {}
    for split, items in rows.items():
        path = OUTPUT / f"{split}.jsonl"
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8")
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "status": "PROVISIONAL_UNREVIEWED", "generator": "Classifier/generate_integrated_multilabel.py",
        "pairs": [list(labels) for labels, _ in TEMPLATES],
        "counts": {split: len(items) for split, items in rows.items()}, "sha256": hashes,
        "exact_overlap_with_existing": 0,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "counts": manifest["counts"], "sha256": hashes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
