"""Generate clause-style multi-topic examples with template-disjoint splits."""

import hashlib
import json

from Classifier.train import ROOT, read_jsonl


OUTPUT = ROOT / "data/classifier/multilabel_hard_20261116_v001"
LABELS = json.loads((ROOT / "outputs/classifier/combined_split_v004/manifest.json").read_text(encoding="utf-8"))["labels"]
WORKERS = (
    "사무 담당 근로자", "고객 상담 근로자", "물류 담당 근로자", "생산 현장 근로자",
    "시설 관리 근로자", "영업 지원 근로자", "품질 검사 근로자", "매장 운영 근로자",
)

# First eight phrasings train; the next two develop; the last two test.
TEMPLATES = (
    (("계약 종료", "손해배상·위약금"), (
        "{worker}가 약정 종료일 전에 사직하면 교육비 {amount}만원을 배상한다",
        "{worker}가 인수인계 없이 퇴직할 때에는 정액 위약금 {amount}만원을 낸다",
        "회사가 {worker}를 해고한 뒤 업무용 장비의 고의 훼손이 확인되면 수리비를 청구한다",
        "{worker}가 고용관계를 중도 종료하고 훈련비를 반환하지 않으면 손해액을 배상한다",
        "퇴사일 전에 {worker}가 고의로 분실한 기기는 퇴직 후에도 확인된 손해액을 배상한다",
        "수습 중 {worker}가 계약을 해지하면 약정 위약금 {amount}만원을 지급한다",
        "계약을 조기 해지하는 {worker}는 대체 인력 채용비 {amount}만원을 부담한다",
        "해고된 {worker}가 인계 자료를 무단 삭제하면 복구에 든 실비를 배상한다",
        "{worker}가 퇴직 통지 없이 근무를 중단하면 회사가 정한 손해배상액 {amount}만원을 낸다",
        "고용 종료를 앞둔 {worker}가 회사 물품을 파손한 경우 퇴직 뒤 수리비를 정산한다",
        "{worker}의 중도 퇴사로 계약을 끝내는 경우 약정 위약금 {amount}만원을 청구한다",
        "회사는 {worker}의 해고 이후 발견된 고의적인 재고 훼손에 대해 실제 손해액을 청구한다",
    )),
    (("임금", "근로시간"), (
        "{worker}의 연장근로 {hours}시간분 임금은 다음 급여일에 가산하여 지급한다",
        "{worker}가 소정근로시간을 넘겨 일한 {hours}시간은 시간외수당으로 계산한다",
        "회사는 {worker}의 야간 근무 {hours}시간에 대한 가산임금을 별도 항목으로 지급한다",
        "{worker}의 업무 대기 {hours}시간을 근로시간에 넣고 해당 임금을 정산한다",
        "{worker}의 퇴근 후 추가 작업 {hours}시간분 급여를 다음 달 임금명세서에 표시한다",
        "{worker}에게 지시한 조기 출근 {hours}시간은 유급 근무시간으로 기록한다",
        "{worker}의 주당 근로시간이 약정을 초과한 경우 초과분의 가산수당을 지급한다",
        "{worker}가 야간 교대조에서 일한 {hours}시간의 임금을 별도로 산정한다",
        "{worker}의 정규 근무 종료 뒤 수행한 {hours}시간은 추가 급여로 이체한다",
        "회사는 {worker}의 연장 업무 시간을 기록하고 그 시간에 대한 수당을 다음 지급일에 준다",
        "{worker}가 하루 근로시간을 넘겨 수행한 {hours}시간의 대가를 가산임금에 반영한다",
        "{worker}의 교대 종료 후 인수인계 {hours}시간을 근무로 인정해 임금을 지급한다",
    )),
    (("계약기간", "계약 종료"), (
        "{worker}의 수습기간 {months}개월이 끝나기 전에 본채용을 거절하면 종료일을 서면으로 통지한다",
        "계약 시작 후 {months}개월 동안 {worker}를 수습 평가하고 본채용 거절 시 계약을 끝낸다",
        "{worker}의 기간제 계약 만료 {days}일 전에 갱신하지 않겠다는 뜻을 알린다",
        "수습 {months}개월 중 {worker}의 채용을 취소할 경우 종료 사유를 문서로 전달한다",
        "{worker}와 맺은 고용기간이 끝나면 별도 갱신 합의가 없는 한 근로관계를 종료한다",
        "계약기간 첫 {months}개월 안에 {worker}를 해고할 때에는 종료 예정일을 명시한다",
        "{worker}의 시용기간 {months}개월 종료 시 본채용을 하지 않기로 하면 그날 계약을 종료한다",
        "기간제 {worker}의 만료일을 {days}일 앞두고 계약 종료 여부를 서면 협의한다",
        "{worker}의 수습 {months}개월이 지나면 채용 유지 여부를 결정하고 거절 시 해지한다",
        "계약 개시 후 {months}개월 안에 {worker}의 고용을 끝내려면 해고 사유를 알린다",
        "{worker}의 약정 근무기간 만료 {days}일 전에 재계약 또는 종료를 결정한다",
        "수습기간을 {months}개월로 정한 {worker}에게 본채용 거부 통지와 종료일을 전달한다",
    )),
    (("임금", "계약 종료"), (
        "{worker}가 퇴직하면 마지막 근무일까지의 미지급 임금을 {days}일 안에 지급한다",
        "{worker}의 해고일을 기준으로 남은 급여를 정산하여 {days}일 안에 이체한다",
        "사직한 {worker}에게 계약 종료 후 {days}일 이내에 체불 임금을 전액 준다",
        "{worker}와 합의해지할 때에는 종료일까지 발생한 임금을 다음 지급일에 정산한다",
        "{worker}의 퇴사 이후 미지급 수당을 {days}일 기한으로 지급한다",
        "해고 통지를 받은 {worker}의 최종 근무기간 급여는 종료일에 계산한다",
        "계약 만료로 떠나는 {worker}에게 남은 기본급과 수당을 {days}일 내 청산한다",
        "{worker}가 중도 사직해도 이미 일한 기간의 임금은 퇴직 후 지급한다",
        "{worker}의 고용관계가 끝난 날까지의 임금 내역을 확정해 {days}일 안에 준다",
        "퇴직하는 {worker}에게 마지막 급여와 미지급 직무수당을 종료 후 {days}일 내 지급한다",
        "{worker}의 해지 통지가 효력을 갖는 날에 미지급 급여를 산정한다",
        "기간제 {worker}의 계약 종료 시 남은 임금을 {days}일 안에 지급한다",
    )),
    (("임금", "손해배상·위약금"), (
        "회사는 {worker}가 분실한 비품의 배상액을 월급에서 {amount}만원 공제한다",
        "{worker}의 업무 중 파손 비용을 손해액으로 정해 다음 급여에서 차감한다",
        "{worker}의 고객 민원 보상금 {amount}만원을 임금에서 일방적으로 공제한다",
        "업무 장비를 잃어버린 {worker}의 수리비를 기본급에서 상계한다",
        "{worker}에게 부과한 재고 손실 배상금을 매월 급여에서 나누어 뺀다",
        "회사는 {worker}의 위약금 {amount}만원을 지급할 임금과 상계한다",
        "{worker}가 훼손한 물품의 복구비를 계산해 수당에서 공제한다",
        "{worker}의 실수로 발생한 거래처 배상 비용을 임금명세서의 공제 항목에 넣는다",
        "{worker}에게 청구할 손해배상액 {amount}만원을 월급 지급액에서 차감한다",
        "회사는 {worker}의 시설 파손 책임액을 급여와 맞바꾸어 정산한다",
        "{worker}의 상품 분실 배상금을 해당 월급에서 일괄 공제한다",
        "고객 환불에 든 {amount}만원을 {worker}의 성과급에서 차감한다",
    )),
    (("휴가", "계약 종료"), (
        "{worker}는 퇴직 전 남은 연차휴가 {days}일의 사용 시기를 회사와 협의한다",
        "해고를 통지받은 {worker}에게 잔여 유급휴가 {days}일의 신청 기회를 준다",
        "{worker}가 계약 만료로 떠나기 전에 미사용 휴가를 신청할 수 있게 한다",
        "사직 예정인 {worker}의 남은 연차 {days}일은 퇴사일 전까지 사용하도록 협의한다",
        "고용 종료를 앞둔 {worker}가 잔여 유급휴가의 사용 날짜를 정한다",
        "{worker}에게 해지 통지 후 남은 휴가 {days}일의 사용 가능 기간을 알린다",
        "퇴직서를 낸 {worker}도 종료일 전까지 연차휴가를 신청할 수 있다",
        "기간제 {worker}의 계약 종료 전에 미사용 휴가 {days}일을 사용할 수 있도록 한다",
        "{worker}가 해고되기 전 남은 연차 {days}일의 사용 여부를 확인한다",
        "계약 만료일을 앞둔 {worker}에게 잔여 휴가 일수와 신청 기한을 알린다",
        "{worker}의 사직일 이전에 유급휴가 {days}일의 사용 일정을 확정한다",
        "고용관계 종료 전 {worker}에게 남은 연차휴가 사용 계획을 요청한다",
    )),
)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    rows = {"train": [], "validation": [], "test": []}
    for pair_no, (labels, templates) in enumerate(TEMPLATES, 1):
        assert len(templates) == 12 and all(label in LABELS for label in labels)
        for template_no, template in enumerate(templates):
            split = "train" if template_no < 8 else "validation" if template_no < 10 else "test"
            assert "{worker}" in template
            for n, worker in enumerate(WORKERS):
                text = template.format(worker=worker, amount=20 + n * 10,
                                       months=1 + n % 4, days=7 + n, hours=1 + n % 5) + "."
                rows[split].append({
                    "id": f"HARD26_{split}_{pair_no:02d}_{template_no:02d}_{n}",
                    "text": text, "labels": list(labels),
                    "label_vec": [int(label in labels) for label in LABELS],
                    "group": f"hard_{pair_no:02d}_{template_no:02d}",
                    "source": "authored_hard_multilabel_v1",
                    "annotation_status": "PROVISIONAL_UNREVIEWED",
                })
    assert {split: len(items) for split, items in rows.items()} == {
        "train": 384, "validation": 96, "test": 96,
    }
    prior = set()
    for path in [
        ROOT / "test/classifier/novel_clause_60_20261115/gold.jsonl",
        ROOT / "outputs/rag/pdf40_pre_llm_20261115/segments_clause_only.jsonl",
        *(ROOT / "outputs/classifier/combined_split_v004").glob("*.jsonl"),
        *(ROOT / "data/classifier/multilabel_aug_20261115").glob("*.jsonl"),
        *(ROOT / "data/classifier/integrated_multilabel_20261115_v001").glob("*.jsonl"),
    ]:
        prior.update(row["text"] for row in read_jsonl(path))
    texts = {split: {row["text"] for row in items} for split, items in rows.items()}
    groups = {split: {row["group"] for row in items} for split, items in rows.items()}
    assert all(len(texts[split]) == len(rows[split]) and not texts[split] & prior for split in rows)
    assert all(not texts[a] & texts[b] and not groups[a] & groups[b] for a, b in
               (("train", "validation"), ("train", "test"), ("validation", "test")))
    OUTPUT.mkdir(parents=True)
    hashes = {}
    for split, items in rows.items():
        path = OUTPUT / f"{split}.jsonl"
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8")
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"status": "PROVISIONAL_UNREVIEWED", "generator": "Classifier/generate_hard_multilabel.py",
                "counts": {split: len(items) for split, items in rows.items()}, "sha256": hashes,
                "template_count": 72, "worker_variants": len(WORKERS), "exact_overlap_with_existing": 0}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "counts": manifest["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
