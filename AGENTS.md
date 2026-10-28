**CheckIt Data 기술 지침**

**저장소 역할 및 기술 스택**

법령 원문을 수집·정규화하고 검색 청크와 Chroma/BM25 인덱스를 구축·평가한다.

계약서 OCR, 분류 모델 추론, 위법 판정, 최종 설명 생성은 현재 범위가 아니다.

- Python 3.11+
- ChromaDB, sentence-transformers, Hugging Face, Kiwi, rank-bm25
- 배포 패키지: checkit-data
- Python 패키지: RAG
- 기본 corpus: employment_basic_v004 (RAG/settings.py 기준)

**주요 디렉터리 및 진입점**

| 경로 | 역할 |
| --- | --- |
| README.md | 실행법과 전체 파이프라인 |
| RAG/settings.py | 기본 corpus 및 저장 경로 |
| RAG/config/*.json | 법령 수집 대상 |
| RAG/collection/ | 법제처 API 연결 |
| RAG/preprocessing/ | 정규화, 청킹, 검색용 텍스트 |
| RAG/indexing/ | 임베딩, Chroma, BM25 |
| RAG/retrieval/ | 검색, 결합, 재정렬, 라벨 우선순위 |
| RAG/evaluation/ | Hit, Recall, MRR 계산 |
| RAG/pipelines/ | python -m CLI 진입점 |
| RAG/service.py | 백엔드 연동 LawSearchService |
| test/rag/ | 단위 테스트 및 평가 결과 |

data/, storage/, outputs/에는 대용량 산출물이 있으므로 관련 작업에서만 접근한다.

**실행 및 테스트**

모든 명령은 저장소 루트에서 실행한다.

**환경 준비**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

**기본 검색**

```bash
.venv/bin/python -m RAG.pipelines.search \
  --query "검색할 계약 조항" --top-k 5 --device cpu
```

**전체 단위 테스트**

```bash
.venv/bin/python -m unittest discover -s test/rag -p 'test_*.py'
```

검색 품질에 영향을 주는 변경은 관련 평가 데이터로 Hit, Recall, MRR 등을 확인한다. 단위 테스트 통과만으로 검색 품질이 검증되었다고 판단하지 않는다.

**법령 데이터 및 인덱스 계약**

- content는 공식 원문으로 유지한다. 검색 보충어는 search_text 또는 embedding_text에만 추가한다.
- 기존 corpus나 인덱스를 덮어쓰지 않는다. 재구축 시 새 corpus 버전을 사용한다.
- 수집·전처리·인덱싱 manifest, 입력 SHA-256, 문서 수, 모델 revision을 유지한다.
- 문서와 질의는 동일한 임베딩 모델과 revision을 사용한다.
- 삭제 조문, 시행일, 자료 유형, 적용 기간, 출처 URL 등 필수 메타데이터를 보존한다.
- 검색 점수와 cosine distance를 정확도 확률로 해석하지 않는다.
- 개발 평가셋의 결과를 실제 계약서 법률 판단이나 종단간 정확도로 확대 해석하지 않는다.

**대용량 작업 및 외부 의존성**

법제처 API 수집, 모델 다운로드, corpus 준비, 인덱스 구축·재구축은 실제 필요한 경우에만 실행한다.

법제처 API 인증값 LAW_API_OC는 .env에서 관리하며 외부에 노출하지 않는다.

**백엔드 연동 계약**

BE는 requirements-rag.txt를 통해 이 저장소를 editable 설치한다.

**공개 Python 인터페이스**

- RAG.service.LawSearchService
- search(query, top_k)
- search_clause(clause_id, text, labels, top_k)

**공유 산출물**

- data/rag/processed/<corpus>/chunks.jsonl
- storage/rag/<corpus>/index_manifest.json
- storage/rag/<corpus>/chroma/

서비스 입력·출력, 기본 corpus, 라벨, retrieval_policy, 결과 metadata 변경 시 CheckIt_BE/app/rag.py와 양쪽 관련 테스트의 호환성을 확인한다.

별도의 Data HTTP 서버가 있다고 가정하지 않는다.
