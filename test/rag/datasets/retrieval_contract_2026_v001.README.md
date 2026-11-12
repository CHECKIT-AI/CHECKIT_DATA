# 2026년 계약 조항 분류·검색 평가 초안

`employment_basic_v004`의 **2026-10-28** 스냅샷에서 조항 분류와 법령 검색을 함께 평가하는 합성 자료다. `dev` 50개와 잠금 대상 `test` 100개는 모두 서로 다른 문장이다. 같은 15개 주제를 양쪽에 두고 통상적인 근로조건 문구와 검토가 필요한 문구를 포함했다. `scenario_intent`는 작성 의도일 뿐 위법 판정이 아니다.

- `retrieval_contract_2026_dev_v001.jsonl`: 검색 방식과 오류 분석용. 검수 전 결과는 잠정치다.
- `retrieval_contract_2026_test_v001.jsonl`: 분류·법령 정답을 독립 검수하고 설정을 동결할 때까지 실행하지 않는다.
- `retrieval_contract_2026_review_template_v001.csv`: 초안 정답을 숨긴 독립 검수 양식. 검수자마다 별도 사본을 사용한다.

## 문장과 정답 출처

문장은 [고용노동부 표준 근로계약서 작성방법](https://www.moel.go.kr/mainpop2.do)의 항목 구성과 [2025년 개정 표준 근로계약서 게시물](https://www.moel.go.kr/info/etc/dataroom/view.do?bbs_seq=20250300356), corpus의 공식 법령 원문을 참고해 새로 작성했다. 실제 계약서에서 수집한 문장이나 실제 사용 빈도를 확인한 표본이 아니다. 기존 RAG 평가 문장은 그대로 재사용하지 않았다. 각 행의 `source_urls`에 작성방법과 관련 공식 법령의 URL을 기록했다.

`query`가 분류기와 검색기에 동일하게 들어가는 조항 원문이다. `model_labels`는 CheckIt_BE `CLAUSE_LABELS` 10개에 맞춘 잠정 다중 라벨 정답이다. `primary_article_ids`는 직접 근거로 제안한 주정답이며 검색 평가기의 `relevant_article_ids`와 같다. `support_article_ids`는 보조 근거다. 두 종류 정답은 각각 `label_annotation_status`와 `annotation_status`가 `DRAFT_UNREVIEWED`다. 라벨 정의는 [벤치마크 라벨 스키마](../../../../CheckIt_Eval_Benchmark/label_schema.json) 및 `CheckIt_BE/app/ai_contracts.py`를 따른다. `topic`, `scenario_intent`, 정답 필드는 모델 입력에 넣지 않는다.

## 독립 검수와 테스트 잠금

1. 법률을 검토할 수 있는 두 사람이 서로의 결과와 모델 예측·검색 결과를 보지 않고 검수 양식을 각각 작성한다. 문장의 중심 주제로 `model_labels`를 지정하고, [현재 법령 청크](../../../data/rag/processed/employment_basic_v004/chunks.jsonl)의 `article_id`, 공식 원문 `content`, `source_url`을 확인해 직접 근거·보조 근거와 이유를 적는다. 통상 문구도 분류하고 관련 조문을 지정한다.
2. 의견이 다른 항목을 합의하고, 조문 시행일과 원문을 확인한다. 인덱스 밖 정답은 corpus 범위 누락으로 기록한다. 검색 방식 비교는 인덱스 안의 주정답으로 채점한다.
3. 합의한 정답은 새 버전으로 저장한다. 두 검수를 완료하고 검색 설정을 동결한 뒤 `test` 결과를 한 번 확인한다. 기존 파일과 인덱스를 덮어쓰지 않는다.

## 분류와 검색을 함께 평가

같은 문장으로 분류기 성능을 계산하고 RAG는 **분류 라벨을 적용하지 않은 원문 검색**만 비교한다. 저장소 루트에서 실행하며, 평가기는 각 저장소의 고정 의존성을 사용한다.

```bash
PYTHONPATH=. .venv/bin/python \
  test/rag/evaluate_classifier_rag.py \
  --corpus employment_basic_v004 \
  --dataset test/rag/datasets/retrieval_contract_2026_dev_v001.jsonl \
  --split dev --device cpu
```

분류 지표는 10개 라벨의 micro/macro F1, exact match, 라벨별 precision·recall·F1이다. RAG는 세 방식 모두 조항 원문만 검색하며 예측 라벨이나 정답 라벨을 우선순위 보정에 전달하지 않는다. 분류와 검색 점수는 독립적으로 기록한다. 검색 지표는 주정답의 Hit@5·Recall@5·MRR@5다. 실행 결과 `run_metadata.json`에 데이터·인덱스 SHA-256과 분류 모델 ID·revision이 저장된다.

하이브리드는 벡터·BM25 후보를 각각 20개 모으고 최종 5개를 결합한다. 단일 검색기는 5개를 요청하므로 후보 예산이 같은 실험은 아니다. 검색 점수는 법률 판단 확률이 아니다. 보조 근거의 유용성, 공식 원문이 실제 조항을 뒷받침하는지, 법률 적용 조건은 독립 검수에서 별도 확인한다.

현재 corpus에는 같은 조문의 과거 버전이 없다. 이 자료는 2026-10-28 기준 검색만 다루며, 과거 계약일 검색과 PDF OCR·조항 분리 성능은 측정하지 않는다.
