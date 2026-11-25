# CheckIt Data

CheckIt의 법령 검색 자료·인덱스와 계약 조항 분류 모델을 준비하고 평가하는 저장소입니다.

## 실행 방법

모든 명령은 이 저장소 루트에서 실행합니다. Python 3.11 이상이 필요합니다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m RAG.pipelines.search --query "중도 퇴사하면 위약금 50만 원을 지급한다." --top-k 5 --device cpu
```

검색에는 Git에 포함되지 않은 `data/rag/processed/employment_basic_v004/chunks.jsonl`과 `storage/rag/employment_basic_v004/`의 인덱스·manifest가 필요합니다. 새 환경에서의 준비·재구축은 [RAG 실행 안내](RAG/README.md#5-실행-방법)를 따르세요. 분류 모델 학습은 [Classifier 실행 안내](Classifier/README.md#학습과-실험)에 있습니다. 단위 테스트: `.venv/bin/python -m unittest discover -s test/rag -p 'test_*.py'`.

## 파일 구조

| 경로 | 역할 |
| --- | --- |
| `RAG/collection/`, `RAG/preprocessing/` | 법령 수집·정규화·청킹 |
| `RAG/indexing/`, `RAG/retrieval/`, `RAG/service.py` | 인덱스 구축·검색과 백엔드 연결 |
| `RAG/pipelines/`, `RAG/evaluation/` | 실행 명령·검색 평가 |
| `Classifier/` | 분류 데이터 준비·학습·평가 |
| `data/`, `storage/`, `models/`, `outputs/` | 입력 자료·인덱스·모델·실행 산출물 |
| `test/` | 단위 테스트와 고정 평가셋·결과 |
| `docs/` | [기존 상세 안내](docs/README-details.md) |

## 평가

| 대상과 조건 | 기준 | 결과 | 해석 |
| --- | --- | --- | --- |
| 법령 검색: `employment_basic_v004` 503청크, 기존 개발 질의 90개, BGE-M3 dense 검색 | 보완 전 Hit@5 85.6% | 보완 후 Hit@5 95.6% | 같은 질의셋에서 검색용 표현·자료를 보완한 결과이며 실제 계약서 판단 정확도가 아님 |
| 조항 분류: 현재 백엔드 기본 `multilabel-hard-v1`, core test 446개 / 별도 고정 조항 60개 | 이전 배포 모델 완전 일치 374/446 (83.9%) / 47/60 (78.3%) | 현재 모델 376/446 (84.3%) / 53/60 (88.3%) | 잠정 정답이며 두 평가셋을 합산하지 않음 |

검색은 정답 근거가 상위 5개에 포함되는 비율(Hit@5), 분류는 예측 라벨 집합의 완전 일치를 측정했습니다. 현재 인덱스의 추가 합성 질의 42개는 Hit@5 100%였지만 기존 90개와 다른 평가셋이라 직접 비교하지 않습니다. 이 dense 평가는 백엔드의 현재 기본 검색 방식 전체에 대한 측정이 아닙니다. 입력·실험 조건과 오류 분석은 [검색 평가](test/rag/comparisons/employment_basic_v004_improvements_20261028/REPORT.md), [분류 모델 비교](Classifier/reports/multilabel_hard_20261116.md)에 있습니다.

## 현재 구현 범위

법령 수집·검색 인덱스와 분류 모델의 준비·평가, 백엔드가 호출하는 검색 인터페이스가 있습니다. 새 클론에는 대용량 원자료·인덱스가 없으며, 모델·검색 점수는 법률 판단 정확도가 아닙니다. 데이터 흐름과 재현 조건은 [상세 안내](docs/README-details.md)에 보존했습니다.
