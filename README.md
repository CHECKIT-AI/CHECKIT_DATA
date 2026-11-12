# CheckIt Data

CheckIt에서 사용하는 법령 검색 자료와 근로계약서 조항 분류 학습 자료를 준비하고 평가하는 저장소입니다. 법령 검색은 `RAG/`, 분류 데이터 준비와 모델 학습 실험은 `Classifier/`에서 관리합니다. 계약서 OCR, 위법 판단, 최종 설명 생성은 백엔드의 영역입니다.

## 프로젝트 구조

| 경로 | 역할 |
| --- | --- |
| [`RAG/`](RAG/README.md) | 법령 수집·정규화·청킹, Chroma/BM25 인덱스, 검색·평가 및 `LawSearchService` |
| [`Classifier/`](Classifier/README.md) | 조항 분류 데이터 분할, KLUE-RoBERTa 학습·비교 실험과 결과 |
| [`data/rag/`](RAG/README.md#3-데이터와-디렉터리) | 법령 API 원본, 가공 청크, 평가 입력 데이터 |
| [`data/classifier/`](Classifier/README.md#데이터-구성) | 합성 조항 240개와 로컬 clean core 원자료 |
| `storage/rag/` | corpus별 Chroma/BM25 인덱스와 manifest (Git 제외) |
| `models/classifier/` | 보존한 분류 모델 산출물 및 보고서 |
| `outputs/rag/`, `outputs/classifier/` | 실행별 중간 산출물·평가 결과 (Git 제외) |
| `test/rag/` | RAG 단위 테스트, 고정 평가셋과 공유 결과 |
| `BASIC_LABOR_COVERAGE.md` | 기본 근로조건 관련 법령 원문 보완 기록 |

## 데이터 흐름

### 법령 검색

1. 법제처 API와 선택한 별표·고시에서 법령 원문을 수집하고 원본과 출처·시행일을 보존합니다.
2. 조문을 정규화하고 삭제 조문을 제외한 뒤 검색 청크를 만듭니다. 공식 원문 `content`는 유지하고 검색 보충어는 별도 필드에 둡니다.
3. 같은 corpus의 청크로 BGE-M3/Chroma 벡터 인덱스와 Kiwi/BM25 인덱스를 구축합니다. 문서와 질의에는 같은 임베딩 모델 revision을 사용합니다.
4. 조항 원문과 분류 라벨 또는 일반 질의를 받아 관련 법령 원문·출처를 반환합니다. 백엔드는 `RAG.service.LawSearchService`를 설치해 호출합니다.
5. 고정 개발 질의로 Hit, Recall, MRR 등을 평가하고 실행 조건·입력 해시·모델 revision을 기록합니다.

현재 기본 corpus는 `employment_basic_v004`이며 법령 본칙·선택 별표·2026년 최저임금 고시를 합쳐 **503개 청크**입니다. 수집 대상, 인덱스 구성, 검색 방식, 실행 명령과 시기별 평가 결과는 [RAG README](RAG/README.md)에 있습니다.

### 조항 분류

1. `processed_core_clean`의 조항과 별도로 작성한 합성 조항 240개의 라벨·형식을 확인합니다.
2. core를 문서 묶음 기준으로 train/validation/test에 나누고, 합성 240개는 train에만 추가합니다. 분할과 입력 해시는 manifest에 기록합니다.
3. 동일한 `klue/roberta-base` 시작점에서 core 단독 모델과 core+합성 모델을 학습하고 validation으로 epoch를 선택한 뒤 test를 평가합니다.
4. 비교 결과를 `Classifier/reports/`에 보존합니다. 공개 배포 모델은 백엔드가 Hugging Face에서 revision을 지정해 불러옵니다.

합성 자료의 라벨은 아직 독립 검수 전입니다. 분할 수치, 10개 라벨, 학습 명령, 테스트 성능과 배포 모델은 [Classifier README](Classifier/README.md)에 정리했습니다. RAG 검색 점수와 분류 test 결과는 실제 계약서에 대한 법률 판단 정확도를 뜻하지 않습니다.

## 시작하기

모든 명령은 저장소 루트에서 실행합니다. RAG 검색 환경은 `python3 -m venv .venv`와 `.venv/bin/python -m pip install -r requirements.txt`로 준비합니다. 기존 인덱스가 있는 환경에서는 다음처럼 검색합니다.

```bash
.venv/bin/python -m RAG.pipelines.search \
  --query "중도 퇴사하면 위약금 50만 원을 지급한다." --top-k 5 --device cpu
```

새 환경에는 Git에서 제외된 원본·가공 데이터와 인덱스가 없으므로 [RAG 재현 절차](RAG/README.md#5-실행-방법)를 먼저 확인하세요. 분류 학습에는 별도의 백엔드 가상환경 의존성과 원자료가 필요합니다. [Classifier 실행 절차](Classifier/README.md#학습과-실험)를 따르세요.
