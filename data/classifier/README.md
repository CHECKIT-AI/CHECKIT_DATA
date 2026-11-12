# 합성 근로계약서 조항 분류 학습 보강셋

추가 학습 실행 방법은 [Classifier/README.md](../../Classifier/README.md)를 참고한다.

`train_synthetic_contract_2026_v001.jsonl`은 **2026-10-28 기준**으로 새로 작성한 근로계약서 조항 240개다. 고용노동부 [표준 근로계약서 작성방법](https://www.moel.go.kr/mainpop2.do)과 [2025년 개정 서식 안내](https://www.moel.go.kr/info/etc/dataroom/view.do?bbs_seq=20250300356)의 항목 구성을 참고했다. 실제 계약서를 수집하거나 사용 빈도를 측정한 자료는 아니다. 서식에 자주 등장하는 임금·시간·휴일·휴가·근무 장소·기간뿐 아니라 드문 위약금·비밀유지 문장도 분류 학습을 위해 포함했다. 파일 해시와 라벨별 건수는 `train_synthetic_contract_2026_v001.manifest.json`에 기록했다.

단일 라벨 문장 200개(라벨별 20개)와 두 라벨 문장 40개로 구성했다. 문장은 모두 서로 다르며 현재 RAG 개발·잠금 테스트셋과 40개 PDF 벤치마크의 조항 원문을 그대로 사용하지 않았다. 라벨은 작성자가 붙인 **잠정 정답**이며 `annotation_status=DRAFT_UNREVIEWED`로 표시했다. 사람의 라벨 검수와 실제 계약서 자료 검증 후 학습에 투입하는 것이 좋다. 기존 모델의 수천 건 학습 자료를 대체할 규모는 아니다.

## 형식

각 줄은 `id`, `text`, `labels`, `label_vec`, `doc_type`, `source`, `annotation_status`, `as_of`를 가진 JSON 객체다. `text`를 분류기에 입력하고 `labels` 또는 `label_vec`를 정답으로 사용한다. `label_vec`는 다음 고정 순서의 길이 10 정수 목록이다.

1. 임금
2. 근로시간
3. 휴일
4. 휴가
5. 근무 장소·업무
6. 계약기간
7. 계약 종료
8. 손해배상·위약금
9. 기타 근로조건
10. 그 외

이 순서는 [CheckIt_BE 라벨 계약](../../../CheckIt_BE/app/ai_contracts.py)의 `CLAUSE_LABELS`와 같다. `그 외`는 다른 내용 라벨과 함께 붙이지 않았다. 라벨 뜻과 원래 분류 기준은 [분류 데이터 계획](../../../CheckIt_BE/docs/clause-classification-data-plan.md)을 따른다. 합성 문장은 위 공식 양식의 *항목과 표현 방식*을 참고해 새로 작성했으며, 양식 원문을 복제한 자료가 아니다.

## 학습 시 사용

이 파일은 **train 보강 자료**로만 사용한다. 기존 모델 학습 데이터와 합칠 때 동일·유사 문장 및 라벨 충돌을 먼저 확인한다. 현재 [개발 평가셋](../../test/rag/datasets/retrieval_contract_2026_dev_v001.jsonl) 50개는 오류 분석과 설정 선택에 쓰고, [잠금 테스트셋](../../test/rag/datasets/retrieval_contract_2026_test_v001.jsonl) 100개는 학습·임계값 조정에 사용하지 않는다. 두 평가셋도 아직 정답 검수 전이다.

학습 결과를 비교할 때는 모델 revision, 10개 라벨 전체의 점수와 임계값, 라벨별 precision·recall·F1을 남긴다. 합성 문장의 높은 점수를 실제 계약서 분포의 성능으로 해석하지 않는다.
