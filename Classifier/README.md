# 조항 분류 데이터와 모델 실험

근로계약서 조항을 10개 범주로 다중 라벨 분류하기 위한 데이터 준비·분할·학습 기록입니다. 실험 모델의 기반은 `klue/roberta-base`(KLUE-RoBERTa)입니다. 법령 근거 검색과 평가는 [RAG README](../RAG/README.md)를 참고하세요.

모든 명령과 아래 상대 경로의 기준은 `CheckIt_Data/` 저장소 루트입니다.

## 데이터 구성

| 자료 | 경로 | 용도 |
| --- | --- | --- |
| clean core 원자료 | `data/classifier/processed_core_clean/train.jsonl`, `test.jsonl`, `labels.json` | 원래 train/test를 합쳐 새 문서 묶음 분할의 입력으로 사용. 원자료는 Git 제외 |
| 합성 조항 240개 | [`data/classifier/train_synthetic_contract_2026_v001.jsonl`](../data/classifier/train_synthetic_contract_2026_v001.jsonl) | train 보강에만 사용 |
| 합성 자료 manifest | [`data/classifier/train_synthetic_contract_2026_v001.manifest.json`](../data/classifier/train_synthetic_contract_2026_v001.manifest.json) | 파일 SHA-256과 라벨별 건수 |
| 새 분할 | `outputs/classifier/combined_split_v004/` | train/validation/test와 입력·출력 해시를 담은 `manifest.json`. Git 제외 |
| 실험 집계 | [`Classifier/reports/processed_core_clean_plus_synthetic_2026_v001.json`](reports/processed_core_clean_plus_synthetic_2026_v001.json) | 설정, 분할 건수, 라벨별 결과 |

합성 문장은 **2026-10-28 기준** 새로 작성했습니다. 고용노동부 [표준 근로계약서 작성방법](https://www.moel.go.kr/mainpop2.do)과 [2025년 개정 서식 안내](https://www.moel.go.kr/info/etc/dataroom/view.do?bbs_seq=20250300356)의 항목 구성을 참고했지만 양식 원문을 복제하거나 실제 계약서를 수집한 자료는 아닙니다. 임금·시간·휴일·휴가·근무 장소·기간과 드문 위약금·비밀유지 표현도 포함합니다. 단일 라벨 문장 200개(라벨별 20개), 두 라벨 문장 40개이며 문장은 서로 다릅니다. RAG 개발·잠금 테스트셋과 40개 PDF 벤치마크의 조항 원문을 그대로 사용하지 않았습니다.

각 JSONL 행에는 `id`, `text`, `labels`, `label_vec`, `doc_type`, `source`, `annotation_status`, `as_of`가 있습니다. `text`가 입력이고 `labels` 또는 `label_vec`가 정답입니다. `label_vec`는 아래 고정 순서의 길이 10 정수 목록입니다.

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

라벨 순서는 [백엔드 `CLAUSE_LABELS`](../../CheckIt_BE/app/ai_contracts.py)와 같습니다. `그 외`는 다른 라벨과 함께 붙이지 않았습니다. 라벨 뜻은 [분류 데이터 계획](../../CheckIt_BE/docs/clause-classification-data-plan.md)을 따릅니다. 합성 라벨은 작성자가 붙인 잠정 정답으로 `annotation_status=DRAFT_UNREVIEWED`이며 독립 검수 전입니다.

## 학습과 실험

### 기존 모델에 합성 240개 추가 학습

기존 `rladmsgki/checkit-contract-classifier`의 고정 revision에서 시작해 합성 240개만 한 epoch 추가 학습한 초기 비교 실험입니다. 개발셋 50개는 전후 평가에만 사용하고 별도의 잠금 테스트셋 100개는 학습·임계값 조정에 사용하지 않았습니다. 이 50/100개는 아래 새 core test 446개와 다른 자료입니다. 정답은 독립 검수 전입니다.

```bash
../CheckIt_BE/.venv/bin/python -m Classifier.train --device mps
```

MPS가 없으면 `--device cpu`를 사용합니다. 백엔드의 분류 모델 의존성이 설치된 가상환경에서 실행합니다. `outputs/classifier/train_*/model/`에는 가중치·토크나이저·기존 라벨별 임계값, `report.json`에는 원본 revision·입력 해시·설정·개발셋 전후 지표가 저장됩니다. 보존본은 `models/classifier/synthetic_2026_v001/`에 있으며 가중치는 Git LFS로 관리합니다.

### clean core와 합성 자료 비교

Drive `processed_core_clean`의 `train.jsonl`, `test.jsonl`, `labels.json`을 `data/classifier/processed_core_clean/`에 둡니다. 원자료는 Git에서 제외하며 [생성한 분할의 Drive 공유 폴더](https://drive.google.com/drive/folders/1ct1eGT-yiN-oIkU8rkhkABY4uiUgTTXR)를 별도로 사용합니다.

```bash
../CheckIt_BE/.venv/bin/python -m Classifier.prepare_core
../CheckIt_BE/.venv/bin/python -m Classifier.train_core --device mps
```

`prepare_core`는 core의 기존 train/test를 합치고 라벨 충돌 28행을 제외합니다. 문서 묶음(`group`) 기준으로 약 80/10/10으로 나누며 같은 문장은 분할 전체에서 한 번만 사용합니다. 합성 240개는 train에만 추가합니다. 생성 건수와 SHA-256은 `outputs/classifier/combined_split_v004/manifest.json`에 기록됩니다. 원본 Drive 파일은 수정하지 않습니다. 다시 분할할 때는 새 `--output` 경로를 지정하고 학습에는 같은 경로를 `--split`으로 전달합니다.

| 분할 | core | 합성 | 합계 |
| --- | ---: | ---: | ---: |
| train | 3,470 | 240 | 3,710 |
| validation | 434 | 0 | 434 |
| test | 446 | 0 | 446 |

`train_core`는 두 모델을 같은 `klue/roberta-base` revision에서 각각 시작해 동일 설정으로 2 epoch 학습합니다. validation macro F1으로 epoch를 선택하고 새 core test는 선택 후 평가합니다. 결과는 `outputs/classifier/core_comparison_*/`에 저장됩니다. 이 test는 Drive의 기존 `test.jsonl`이나 별도의 잠금 합성 테스트셋과 다릅니다.

#### core test 결과

| 학습 데이터 | Micro F1 | Macro F1 | Exact match |
| --- | ---: | ---: | ---: |
| core 3,470개 | 0.838 | 0.762 | 80.5% |
| core 3,470개 + 합성 240개 | 0.858 | 0.829 | 83.6% |

이 분할에서는 합성 자료 추가 후 점수가 높았습니다. 라벨별 점수와 실험 설정은 [집계 보고서](reports/processed_core_clean_plus_synthetic_2026_v001.json)에 있습니다. 합성 라벨은 미검수이고 일부 라벨은 test 표본이 적어 실제 계약서 성능이나 법률 판단 정확도로 확대 해석하지 않습니다.

## 배포와 백엔드 연결

현재 백엔드 기본 설정은 공개 [Hugging Face 모델 `hur03/checkit-contract-classifier-core-synthetic-v1`](https://huggingface.co/hur03/checkit-contract-classifier-core-synthetic-v1)의 revision `7fb6bc333b01919d7e59c1937e9457266d574cbe`를 참조합니다. 설정은 [백엔드 `app/config.py`](../../CheckIt_BE/app/config.py)와 `.env.example`의 `CHECKIT_CLASSIFIER_MODEL_ID`, `CHECKIT_CLASSIFIER_REVISION`에서 확인할 수 있습니다. 환경변수로 다른 값을 지정하면 기본값을 덮어씁니다. 이 모델 사용과 실제 계약서에서의 분류·판정 품질 검증은 별개입니다.
