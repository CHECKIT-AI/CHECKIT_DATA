"""Create fictional contract PDFs and freeze clause labels before inference."""

import hashlib
import json
import re
from pathlib import Path

import pdfplumber
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

from Classifier.train import ROOT, read_jsonl


HERE = Path(__file__).resolve().parent
GOLD = HERE / "gold.jsonl"
PDFS = HERE / "pdfs"
LABELS = json.loads((ROOT / "outputs/classifier/combined_split_v004/manifest.json").read_text(encoding="utf-8"))["labels"]

# Each list contributes one clause to each of twelve fictional contracts.
CLAUSES = (
    (("근무 장소·업무",), (
        "주된 근무지는 서울 중구의 고객운영실이며 주문 접수와 민원 기록을 담당한다.",
        "근로자는 부산 연제구 지점에서 재고 실사와 입출고 대장 관리를 맡는다.",
        "근무 장소를 대전 유성구 연구지원실로 하고 시료 운반 및 장비 점검을 수행한다.",
        "인천 서구 물류센터에서 송장 검수와 출고 확인 업무를 수행한다.",
        "광주 북구 상담센터를 취업 장소로 정하고 고객 문의 분류 업무를 맡는다.",
        "근로자는 울산 남구 사업장에서 품질 검사와 불량품 기록을 담당한다.",
        "수원 영통구 사무소에서 프로젝트 일정과 회의 자료를 관리한다.",
        "창원 성산구 제조시설에서 설비 상태 확인과 점검표 작성을 맡는다.",
        "제주 제주시 매장에서 상품 진열과 고객 응대 업무를 수행한다.",
        "근로자의 주된 업무는 전주 완산구 사무실의 회계 자료 입력과 확인이다.",
        "포항 남구 현장관리실에서 시설 보수 요청을 접수하고 처리 내역을 기록한다.",
        "청주 흥덕구 지원센터에서 온라인 주문의 반품 서류를 검토한다.",
    )),
    (("근로시간",), (
        "시업은 오전 9시, 종업은 오후 6시이며 정오부터 한 시간 휴게한다.",
        "월요일부터 금요일까지 하루 7시간 30분 근무하고 오후 1시에 휴게를 시작한다.",
        "교대조의 근무 구간은 오전 6시부터 오후 2시까지로 정한다.",
        "한 주의 소정근로시간은 36시간이며 세부 출퇴근 시각은 근무표에 적는다.",
        "오전 10시에 업무를 시작하여 오후 7시에 마치고 중간에 60분 쉰다.",
        "근로자는 화요일부터 토요일까지 매일 8시간씩 근무한다.",
        "근무표에는 시작 시각과 종료 시각을 매주 금요일에 확정해 기재한다.",
        "하루 근로시간은 6시간으로 하고 그 사이에 30분의 휴게를 둔다.",
        "오전조는 7시에 출근해 15시에 퇴근하며 인수인계 시간을 기록한다.",
        "주중 근무는 하루 8시간, 토요일 근무는 하루 4시간으로 배정한다.",
        "월별 교대 일정에 따라 8시부터 17시까지 또는 13시부터 22시까지 일한다.",
        "업무 준비를 위해 지시받은 대기시간은 근로시간에 포함한다.",
    )),
    (("임금",), (
        "기본급은 월 278만 원이며 매월 25일 근로자 명의 계좌로 송금한다.",
        "월 급여 263만 원과 직무수당 12만 원을 합산해 다음 달 5일 지급한다.",
        "시급은 1만 4천 원으로 하고 급여명세서에 공제 내역을 표시한다.",
        "성과수당은 확정된 실적에 따라 계산하여 매달 말일 입금한다.",
        "기본급 292만 원을 지급하고 식대 10만 원은 별도 항목으로 적는다.",
        "임금은 월 265만 원으로 하며 지급일은 매월 20일로 한다.",
        "월급의 구성은 기본급 250만 원과 직책수당 18만 원이다.",
        "급여는 지급일에 전액 이체하고 항목별 계산서를 전자문서로 제공한다.",
        "기본 시급 1만 3천 원을 적용하며 임금명세서를 매월 교부한다.",
        "월 기본급은 281만 원이며 지급 계좌 변경은 서면으로 접수한다.",
        "상여금은 지급 요건을 충족한 경우 분기 마지막 급여일에 지급한다.",
        "직무급 16만 원과 기본급 274만 원을 급여명세서에 구분해 적는다.",
    )),
    (("휴가",), (
        "발생한 연차휴가는 근로자가 신청한 날짜를 고려해 사용 일정을 정한다.",
        "연차 잔여일은 인사 시스템에 표시하고 사용 신청은 전자문서로 받는다.",
        "질병 치료를 위한 병가는 진단서를 제출하면 신청할 수 있다.",
        "가족 행사에 필요한 경조휴가의 일수는 사내 규정에 따른다.",
        "근로자가 요청한 유급휴가 일정은 업무 조정 후 확정한다.",
        "매년 발생하는 연차휴가의 사용 현황을 분기별로 안내한다.",
        "휴가 신청을 받은 관리자는 승인 또는 조정 사유를 서면으로 알린다.",
        "미사용 연차의 잔여 일수는 근로자에게 매월 확인시킨다.",
        "입사 첫해 발생한 연차도 다른 휴가와 같은 방법으로 신청한다.",
        "연차 사용 계획은 근로자와 부서장이 협의하여 기록한다.",
        "특별휴가를 신청할 때 필요한 증빙 종류는 취업규칙에 따른다.",
        "유급휴가의 사용 날짜를 바꾸려면 근로자에게 변경 사유를 알린다.",
    )),
    (("기타 근로조건",), (
        "업무 중 접한 고객의 비공개 자료는 승인 없이 외부에 전달하지 않는다.",
        "회사는 근로자의 국민연금과 건강보험 가입 절차를 진행한다.",
        "회사 계정의 접속 정보를 다른 직원에게 대여하지 않는다.",
        "근로자의 겸직 신청은 이해충돌 여부를 검토해 서면으로 처리한다.",
        "업무용 단말기의 고객정보는 회사가 정한 보안 절차에 따라 보관한다.",
        "퇴직연금 가입 내용은 별도 안내문으로 근로자에게 교부한다.",
        "영업비밀을 취급하는 직원은 접근 권한을 다른 사람과 공유하지 않는다.",
        "근로자의 사회보험 자격 취득 신고는 법정 기한 안에 처리한다.",
        "업무상 알게 된 거래처 가격 정보는 승인된 목적에만 사용한다.",
        "겸업으로 인한 이해충돌 가능성을 사전에 인사 담당자에게 알린다.",
        "개인정보가 포함된 업무 파일은 지정된 저장소에서만 열람한다.",
        "근로자에게 적용하는 퇴직연금 규약은 서면으로 열람할 수 있게 한다.",
    )),
    (("계약 종료", "손해배상·위약금"), (
        "근로자가 퇴사를 통지하지 않고 인계 자료를 고의로 삭제한 경우 복구에 든 손해액을 청구한다.",
        "계약을 중도 해지하면서 회사 장비를 반환하지 않으면 확인된 가액을 배상한다.",
        "해고일 이전에 고의로 파손한 기기는 근로관계 종료 후에도 실손해액을 배상한다.",
        "사직으로 업무가 끝난 뒤 발견된 고의적 재고 훼손은 확인된 손실만 청구한다.",
        "기간제 근로자가 만료 전에 일을 그만두면 약정된 위약금 40만 원을 지급한다.",
        "퇴직 절차를 완료하지 않고 회사 물품을 반출한 경우 그 가액을 배상한다.",
        "근로자가 조기 퇴사하며 인수인계를 거부하면 정액 손해배상금 25만 원을 낸다.",
        "해고된 직원이 회사 자료를 의도적으로 훼손했다면 복구 비용을 청구할 수 있다.",
        "고용관계가 끝나기 전 업무용 노트북을 고의로 분실하면 퇴사 뒤 손해액을 정산한다.",
        "계약 종료를 앞두고 시설을 파손한 근로자에게 확인된 수리비를 청구한다.",
        "근로자가 사직서를 제출한 후 재고를 무단 반출하면 실제 손실을 배상한다.",
        "중도 퇴사로 계약을 끝낸 근로자에게 훈련비 명목의 위약금 35만 원을 부과한다.",
    )),
    (("임금", "근로시간"), (
        "정규 근무가 끝난 뒤 수행한 추가 작업 3시간의 가산임금은 다음 급여일에 지급한다.",
        "야간 교대조에서 일한 2시간분 수당은 기본급과 구분해 계산한다.",
        "출근 전 지시받은 준비 업무 40분을 근로시간으로 기록하고 임금을 지급한다.",
        "소정근로시간을 넘긴 업무에는 별도 가산수당을 적용한다.",
        "회사 지시로 늦게 퇴근한 90분은 시간외근무 수당의 산정 대상이다.",
        "야간에 수행한 고객 응대 시간은 가산임금 내역에 표시한다.",
        "휴게 중 호출받아 처리한 업무 시간은 유급 근무로 정산한다.",
        "주당 약정 시간을 초과한 근무분은 다음 달 급여명세서에 따로 기재한다.",
        "교대 종료 뒤 진행한 인수인계 30분을 근로시간에 넣고 급여로 지급한다.",
        "회사는 추가로 지시한 근무 4시간분을 가산임금에 반영한다.",
        "업무 시작 전 의무 교육을 받은 시간은 임금 계산에 포함한다.",
        "종업 시각 이후 작성한 보고서 업무의 시간을 기록해 수당을 지급한다.",
    )),
    (("계약기간", "계약 종료"), (
        "수습 2개월을 마친 뒤 본채용을 하지 않기로 하면 근로관계 종료일을 서면으로 알린다.",
        "계약기간이 끝나는 날 갱신 합의가 없다면 그날 고용관계를 종료한다.",
        "시용기간 3개월 안에 본채용을 거절하려면 해지 사유를 문서로 전달한다.",
        "기간제 계약 만료 14일 전까지 재계약하지 않겠다는 뜻을 통지한다.",
        "수습평가가 종료되는 달에 채용을 거부하면 계약 종료일을 명시한다.",
        "약정 근무기간 도중 해고할 경우 남은 계약기간과 해지 사유를 적는다.",
        "수습 6주가 지나 정식 채용을 거절할 때에는 종료 통지를 교부한다.",
        "정해진 고용기간을 연장하지 않기로 한 경우 만료일에 근로관계를 끝낸다.",
        "계약 개시일부터 3개월 이내에 채용을 취소하면 종료 날짜를 알린다.",
        "기간제 근로자의 만료일을 앞두고 갱신 또는 계약 종료를 협의한다.",
        "수습기간 종료 전에 고용을 중단하기로 결정하면 해고 이유를 서면에 적는다.",
        "기간의 정함이 있는 계약을 조기에 해지할 때에는 종료 예정일을 통보한다.",
    )),
    (("임금", "계약 종료"), (
        "퇴직한 근로자에게 마지막 근무일까지 발생한 미지급 급여를 10일 안에 이체한다.",
        "사직일에 확정된 잔여 수당을 고용 종료 후 2주 안에 정산한다.",
        "해고된 직원의 미지급 기본급은 근로관계 종료일을 기준으로 계산한다.",
        "계약이 만료되면 그날까지 발생한 임금을 전액 청산한다.",
        "퇴사 절차가 끝난 근로자에게 남은 월급을 별도 계좌로 지급한다.",
        "합의해지 이후 남아 있는 직무수당은 종료일부터 12일 안에 이체한다.",
        "사직서가 수리된 날까지의 급여를 산정해 퇴직금품과 함께 지급한다.",
        "해고 통지 뒤 최종 출근일까지의 임금은 계약 종료 후 지급한다.",
        "기간제 근로자의 계약 종료일에 미지급 수당을 확인한다.",
        "고용관계가 끝난 후 14일 안에 남은 급여를 청산한다.",
        "퇴직일 이전에 제공한 근로의 대가를 종료 후 지정 계좌로 송금한다.",
        "계약 만료로 퇴사하는 직원의 기본급 잔액을 마지막 지급일에 준다.",
    )),
    (("임금", "손해배상·위약금"), (
        "회사는 비품 분실 배상금 18만 원을 근로자의 월급에서 공제한다.",
        "업무용 기기 파손에 따른 수리비를 다음 급여와 상계한다.",
        "고객에게 지급한 손해 보전액을 직원의 성과수당에서 차감한다.",
        "재고 부족분의 배상액을 기본급 지급액에서 뺀다.",
        "직원이 부담할 물품 훼손 비용을 월 임금명세서의 공제 항목에 적는다.",
        "고의 파손으로 확정된 손해액을 다음 달 급여에서 일괄 공제한다.",
        "회사에 낼 정액 위약금 20만 원을 직원의 미지급 임금과 상계한다.",
        "거래처 보상비를 근로자의 월급에서 나누어 공제한다.",
        "장비 수리비를 직원의 지급 예정 수당에서 차감할 수 있다.",
        "상품 반출로 발생한 손실액을 급여의 공제 항목으로 처리한다.",
        "업무상 파손의 배상금은 해당 월 기본급에서 우선 정산한다.",
        "고객 환불 비용을 직원의 성과급에서 일방적으로 뺀다.",
    )),
)

COMPANIES = (
    "가온운영", "서림유통", "모래빛연구", "다온물류", "파란상담", "새봄제조",
    "가람기획", "솔밭설비", "해오름상점", "봄날회계", "푸른시설", "한결지원",
)


def font_name():
    path = Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf")
    if not path.exists():
        raise RuntimeError("Korean PDF font unavailable")
    pdfmetrics.registerFont(TTFont("CheckItKorean", str(path)))
    return "CheckItKorean"


def wrap(text, font, size, width):
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}" if current else word
        if pdfmetrics.stringWidth(trial, font, size) > width and current:
            lines.append(current)
            current = word
        else:
            current = trial
    return lines + [current]


def render(path, company, clauses, font):
    canvas = Canvas(str(path), pagesize=A4)
    width, height = A4
    canvas.setTitle(f"{company} 근로계약서 (평가용 합성)")
    canvas.setFont(font, 16)
    canvas.drawCentredString(width / 2, height - 53, "근로계약서")
    canvas.setFont(font, 9)
    canvas.drawString(48, height - 79, f"사업장: {company} (가상)    체결일: 2026년 11월 16일")
    canvas.drawString(48, height - 95, "본 문서는 분류 모델 평가를 위해 작성한 합성 계약서입니다.")
    y = height - 128
    for number, clause in enumerate(clauses, 1):
        canvas.setFont(font, 9.5)
        lines = wrap(f"제{number}조 {clause}", font, 9.5, width - 96)
        for line in lines:
            canvas.drawString(48, y, line)
            y -= 15
        y -= 27
    assert y > 89, f"page overflow: {path.name} y={y}"
    canvas.setFont(font, 9)
    canvas.drawString(48, 61, "사용자: ____________________       근로자: ____________________")
    canvas.drawRightString(width - 48, 42, "평가용 합성 문서")
    canvas.save()


def main():
    if GOLD.exists() or PDFS.exists():
        raise FileExistsError("evaluation set already exists")
    assert len(CLAUSES) == 10 and all(len(texts) == 12 and all(label in LABELS for label in labels)
                                     for labels, texts in CLAUSES)
    rows = []
    for doc_no in range(12):
        for clause_no, (labels, texts) in enumerate(CLAUSES, 1):
            rows.append({"contract_id": f"NEW_DOC_{doc_no + 1:02d}",
                         "clause_id": f"NEW_DOC_{doc_no + 1:02d}_C{clause_no:02d}",
                         "text": texts[doc_no], "labels": list(labels),
                         "annotation_status": "PROVISIONAL_UNREVIEWED", "source": "authored_contract_style_trial_v1"})
    assert len(rows) == len({row["text"] for row in rows}) == 120
    prior = set()
    for path in [
        *(ROOT / "outputs/classifier/combined_split_v004").glob("*.jsonl"),
        *(ROOT / "data/classifier/multilabel_aug_20261115").glob("*.jsonl"),
        *(ROOT / "data/classifier/integrated_multilabel_20261115_v001").glob("*.jsonl"),
        *(ROOT / "data/classifier/multilabel_hard_20261116_v001").glob("*.jsonl"),
        ROOT / "test/classifier/novel_clause_60_20261115/gold.jsonl",
        ROOT / "outputs/rag/pdf40_pre_llm_20261115/segments_clause_only.jsonl",
    ]:
        prior.update(row["text"] for row in read_jsonl(path))
    assert not prior & {row["text"] for row in rows}
    font = font_name()
    PDFS.mkdir(parents=True)
    for doc_no, company in enumerate(COMPANIES):
        path = PDFS / f"new_contract_{doc_no + 1:02d}.pdf"
        texts = [row["text"] for row in rows if row["contract_id"] == f"NEW_DOC_{doc_no + 1:02d}"]
        render(path, company, texts, font)
        with pdfplumber.open(path) as pdf:
            assert len(pdf.pages) == 1
            extracted = re.sub(r"\s+", "", pdf.pages[0].extract_text() or "")
        assert all(re.sub(r"\s+", "", text) in extracted for text in texts), path.name
    GOLD.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {"source": "newly authored fictional contracts, not actual signed contracts",
                "contract_count": 12, "clause_count": len(rows),
                "single_count": sum(len(row["labels"]) == 1 for row in rows),
                "multi_count": sum(len(row["labels"]) > 1 for row in rows),
                "gold_sha256": hashlib.sha256(GOLD.read_bytes()).hexdigest(),
                "pdf_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(PDFS.glob("*.pdf"))},
                "generator": "test/classifier/contract_style_trial_20261116/generate.py",
                "label_status": "PROVISIONAL_UNREVIEWED", "exact_overlap_with_existing": 0}
    (HERE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("contract_count", "clause_count", "single_count", "multi_count", "gold_sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
