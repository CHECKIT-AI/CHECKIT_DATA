# 가중치 RRF와 전체 후보 재정렬

기존 dense / bm25 / hybrid 방식은 유지합니다. 기본 검색 옵션은 dense입니다.

## 추가한 방식

- weighted_rrf: BGE-M3와 BM25의 RRF 기여도를 조정합니다. 기본 0.8:0.2입니다.
- hybrid_rerank: 각 검색기 상위20개의 합집합 전체(중복 제거 후 최대40개)를 CrossEncoder로 재정렬합니다.

hybrid_rerank는 weighted RRF로 후보의 사전 순서를 만들지만 최종10개로 자르지 않습니다. 모델은 모든 후보를 채점합니다.
최종 순위는 reranker 점수가 결정하고, 가중치 RRF는 사전 순서와 점수 동점 처리에 사용됩니다.
reranker 점수는 raw logit이며 위법 확률이 아닙니다. 사용 모델은 BAAI/bge-reranker-v2-m3입니다.
모델 커밋은 storage/rag/models/bge-reranker-v2-m3.json에 기록됩니다. 모델 파일 자체는 Hugging Face 로컬 캐시에 저장됩니다.

## 모델 준비: 처음 한 번

```bash
source .venv/bin/activate
python -m RAG.pipelines.prepare_reranker
```

CrossEncoder는 이미 설치된 sentence-transformers로 실행하므로 새 패키지는 필요하지 않습니다.
입력 쌍의 토큰 길이가 1024를 넘으면 조용히 자르지 않고 중단합니다.

## 검색

```bash
python -m RAG.pipelines.search --corpus labor_standards_v001 --method weighted_rrf --dense-weight 0.8 --bm25-weight 0.2 --query "중도 퇴사하면 위약금 50만 원을 지급한다."
python -m RAG.pipelines.search --corpus labor_standards_v001 --method hybrid_rerank --query "중도 퇴사하면 위약금 50만 원을 지급한다."
```

## 평가

```bash
python -m RAG.pipelines.compare_improvements --corpus labor_standards_v001
```

고정한 90개 질의로 dense, 동일 가중치 hybrid, 8:2 weighted RRF, 7:3 weighted RRF, 전체 후보 reranker를 비교합니다.
candidate-k=20, RRF k=60, 최종 깊이10, CPU, reranker batch-size=2가 기본값입니다.
결과는 outputs/rag/improvements_<시각>/에 저장되고 개별 실행 결과 폴더가 기록됩니다.
모델 준비 및 워밍업 시간은 질의 처리 시간에서 제외합니다. 재정렬은 각 후보를 읽으므로 처리 시간이 증가합니다.

기존 90개는 이미 결과를 관찰한 DRAFT 개발셋입니다. 이 결과에 맞춰 설정을 고른 뒤에는 별도의 미사용 평가셋에서 확인해야 합니다.
새 설정이 나아졌다고 단정하지 않고 모든 결과를 남깁니다. 이전 기준선은 덮어쓰지 않습니다.
