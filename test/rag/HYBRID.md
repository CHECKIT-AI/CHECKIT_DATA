# BM25 통합 검색

## 구성

- RAG/preprocessing/tokenizer.py: Kiwi 토큰화, 조사·어미와 문장부호 제외, 부정 표현·숫자·조문 식별자 보존
- RAG/indexing/bm25.py: BM25Okapi, 토큰 JSON·설정·해시 검증, 양수 점수 문서만 반환
- RAG/retrieval/hybrid.py: 동일 가중치 RRF, 동점은 chunk_id 순서
- RAG/retrieval/retriever.py: dense / bm25 / hybrid 공통 검색 인터페이스
- RAG/pipelines/build_bm25.py: 기존 청크로 BM25 자료 생성
- RAG/pipelines/search.py: --method 선택. 기본은 기존과 같은 dense
- RAG/pipelines/evaluate.py: 같은 지표로 세 방식 개별 평가
- RAG/pipelines/compare_retrieval.py: 같은 평가셋으로 세 방식을 실행하고 비교표 저장

## 설치와 인덱스 구축

CheckIt_Data 루트에서 실행합니다.

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m RAG.pipelines.build_bm25 --corpus labor_standards_v001
```

기존 BGE-M3 임베딩과 Chroma는 재생성하지 않습니다. 추가 자료는 storage/rag/labor_standards_v001/bm25/에 저장됩니다.
기존 BM25 자료는 덮어쓰지 않습니다. 성공한 구축 명령은 반복 실행할 필요가 없습니다.
토큰화 코드·패키지·청크가 바뀌면 새 corpus 버전으로 인덱스를 구축해야 합니다.

## 검색

```bash
python -m RAG.pipelines.search --corpus labor_standards_v001 --method bm25 --query "중도 퇴사하면 위약금 50만 원을 지급한다."
python -m RAG.pipelines.search --corpus labor_standards_v001 --method hybrid --query "중도 퇴사하면 위약금 50만 원을 지급한다."
```

hybrid 기본 설정은 각 검색기 상위 20개, 동일 가중치, RRF k=60입니다. --candidate-k, --rrf-k로 명시할 수 있습니다.
가중치·검색어 확장·동의어 사전·정답 조문 강제 삽입은 적용하지 않습니다. Reranker도 추가하지 않습니다.
bm25_score, cosine_distance, rrf_score는 서로 다른 척도이며 위법 확률이 아닙니다.

## 같은 90개 질의로 비교

```bash
python -m RAG.pipelines.compare_retrieval \
  --corpus labor_standards_v001 \
  --dataset test/rag/datasets/retrieval_dev_v002.jsonl
```

개별 방식의 최종 검색 깊이는 10개입니다. hybrid는 내부적으로 각 검색기 20개를 사용하므로 후보 탐색 비용이 더 들 수 있습니다. 이 후보 수는 기록됩니다.
모든 방식에서 준비·워밍업 후 질의 처리 시간을 측정하며, 모델·인덱스 초기 로딩 시간은 제외합니다.
비교 실행마다 outputs/rag/comparison_<시각>/에 comparison.csv와 comparison.json을 저장하고 방법별 실행 폴더를 기록합니다.
comparison.json의 paired_hit5_changes에는 dense 대비 hybrid의 적중 개선·악화 질의가 있습니다.
비교기는 질의셋·청크 해시, 최종 검색 깊이, 모델 설정 등이 같은지 검증합니다.

개별 평가:

```bash
python -m RAG.pipelines.evaluate --corpus labor_standards_v001 --method hybrid --dataset test/rag/datasets/retrieval_dev_v002.jsonl --run-name hybrid_v001
```

이전 test/rag/baseline/bge_m3_v002/의 기준선은 그대로 보존됩니다.
새 비교 점수는 90개 DRAFT 개발 질의에 대한 단일 실행이며, 미사용 최종 테스트셋의 성능이 아닙니다.
같은 질의셋을 보며 매개변수를 조정했다면 이후 별도 테스트셋으로 검증해야 합니다.

## 기능 검사

```bash
python -m unittest discover -s test/rag -p 'test_*.py'
```

토큰화의 부정 표현·조문 식별자 보존, BM25 빈 검색 처리·버전 불일치 차단, RRF의 중복·순위·출처 처리를 확인합니다.
