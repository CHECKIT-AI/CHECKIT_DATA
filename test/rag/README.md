# RAG 검색 테스트 자료

BM25 통합 전 BGE-M3 검색의 비교 기준 자료입니다. 원래 데이터와 outputs 아래의 실행 기록은 보존하고, 이 폴더에는 고정한 복사본을 둡니다.

## 구성

- [2026년 계약 조항 검색 평가 초안](datasets/retrieval_contract_2026_v001.README.md): 새로 작성한 조항 150개(개발 50·잠금 테스트 100), 독립 검수 양식. 전문가 검수 전 잠정 정답.
- datasets/retrieval_dev_v002.jsonl: 15개 주제, 키워드·사용자 질문·계약 문구 각 30개, 총 90개
- baseline/bge_m3_v002/results.csv: 질의별 정답 후보 순위와 검색 상위 결과
- baseline/bge_m3_v002/results.jsonl: 각 질의의 상위 10개 검색 결과 전문
- baseline/bge_m3_v002/summary.json: 전체·유형별 검색 지표
- baseline/bge_m3_v002/run_metadata.json: 데이터 해시·모델 버전·평가 조건

## 다시 실행

CheckIt_Data 루트에서 실행합니다.

```bash
source .venv/bin/activate
python -m RAG.pipelines.evaluate \
  --corpus labor_standards_v001 \
  --dataset test/rag/datasets/retrieval_dev_v002.jsonl \
  --run-name baseline_bge_m3_v002
```

새 실행 결과는 outputs/rag/ 아래 새 폴더에 저장됩니다. 이 폴더의 기준선은 자동으로 덮어쓰지 않습니다.

## 비교와 한계

BM25 통합 후에도 같은 질의셋·corpus·최종 검색 깊이로 평가하고 별도 결과로 저장하세요. run_metadata.json의 데이터 해시를 대조하면 동일한 입력인지 확인할 수 있습니다.
평가기는 --method dense|bm25|hybrid를 지원합니다. 비교 실행과 BM25 구축 방법은 HYBRID.md를 참조하세요.
이 자료는 직접 작성한 개발용 질의이며, 정답 조문은 검수 전 DRAFT 후보입니다. 실제 계약서 90건 또는 위법 판정 정답을 의미하지 않습니다. 근로기준법만 대상으로 합니다.

## 실험 기록

- [실험 이력과 개선 계획](EXPERIMENT_LOG.md)
- [전체 설정·결과 JSON](experiment_history.json)

## 가중치·재정렬 실험

사용법은 [RERANKER.md](RERANKER.md), 측정 이력은 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)를 확인하세요.
