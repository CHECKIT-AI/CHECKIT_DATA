# 정확 검색으로 인덱스 누락 확인하기

기존 DB 벡터를 그대로 읽고 질의마다 한 번 만든 BGE-M3 벡터를 두 검색에 공유합니다.
Chroma는 상위20개를 요청하고, 정확 검색은 NumPy float64로 모든 문서의 cosine distance를 계산합니다.
문서·질의 벡터를 정규화하며 동점은 chunk_id 순서로 정렬합니다. 경계 동점에는 1e-6 허용오차를 사용합니다.
BM25·RRF·Reranker는 이 진단에 포함하지 않습니다.

## 실행

CheckIt_Data 루트에서 실행합니다. 추가 설치나 인덱스 재생성이 필요 없습니다.

```bash
source .venv/bin/activate
python -m RAG.pipelines.compare_exact --corpus labor_standards_v001
```

평가셋 기본값은 test/rag/datasets/retrieval_dev_v002.jsonl입니다.
--dataset 옵션으로 같은 형식의 다른 파일을 지정할 수 있습니다.
원본 Chroma는 임시 복사본을 만들어 사용하며 결과는 outputs/rag/exact_comparison_<시각>/에 저장합니다.
원본 응답·DB·실행 결과에 대한 기존 Git 제외 규칙은 유지됩니다.

## 결과 읽기

- summary.json: 검색별 Hit/Recall@1·3·5·10·20, MRR@20, 상위 목록 일치율과 유의한 누락 수.
- results.csv: 질문별 Chroma 정답 순위(상위20개 안), 정확 검색의 정답 전체 순위, 결과 겹침 비율.
- results.jsonl: 정확 검색의 전체 순위와 Chroma 상위20개 및 질의별 지표. outputs에 보존.
- run_metadata.json: 데이터/코드 해시, 모델 커밋, 인덱스 설정, 패키지 버전.

mean_query_ms는 임베딩을 제외한 검색 시간입니다. 임베딩은 mean_embedding_ms로 별도 기록합니다.
정확 검색은 메모리에 이미 올라간 행렬로 전체 순위를 만들며, Chroma는 DB API로 상위20개를 반환하므로
이 단일 실행의 시간을 서비스 성능의 일반적인 우열로 해석하지 않습니다.
이 실험의 MRR@20은 기존 MRR@10과 직접 비교하지 않습니다.

## 2026-10-19 실행 결과

근로기준법 131개 청크, DRAFT 개발 질의90개에서 상위20개 문서와 순서가 90/90개 모두 동일했습니다.
Hit@1=73.3%, Hit@5=91.1%, Hit@10=92.2%, Hit@20=95.6%로 양쪽이 같습니다.
이 범위에서는 Chroma 근사 검색에 따른 누락을 관찰하지 못했습니다.
데이터가 커지거나 인덱스를 바꾸면 같은 검사를 다시 해야 합니다.

예를 들어 “중도 퇴사하면 위약금 50만 원을 지급한다.”의 제20조는 정확 검색에서도46위입니다.
“그만두면 무조건 백만 원을 내라는데 어떤 규정을 찾아봐야 하나요?”는117위입니다.
이 사례는 인덱스 조정보다 임베딩이나 검색용 표현 개선 실험이 필요하다는 근거입니다.
현재 검색 기본값은 바꾸지 않습니다.

[보존한 비교 보고서](comparisons/exact_comparison_20261019T095622836381Z/REPORT.md)
