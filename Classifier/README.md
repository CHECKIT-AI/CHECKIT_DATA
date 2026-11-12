# 조항 분류 모델 추가 학습

기존 `rladmsgki/checkit-contract-classifier`의 고정 revision에서 시작해
`data/classifier/train_synthetic_contract_2026_v001.jsonl` 240개를 한 epoch 추가 학습한다.
기본 모델은 `klue/roberta-base` 기반 10개 라벨 다중 분류 모델이다.

```bash
../CheckIt_BE/.venv/bin/python -m Classifier.train --device mps
```

Mac의 MPS가 없으면 `--device cpu`를 사용한다. 백엔드의 분류 모델 의존성이
설치된 가상환경에서 실행한다. 결과는 `outputs/classifier/train_*/`에 저장되며,
`model/`에는 가중치·토크나이저·기존 라벨별 임계값이, `report.json`에는
원본 모델 revision, 학습 데이터 해시, 학습 설정과 개발셋 전후 지표가 남는다.
이번 실행의 모델과 보고서는 `models/classifier/synthetic_2026_v001/`에도
보존했다. 가중치 파일은 Git LFS로 관리한다.

학습 입력은 합성 240개뿐이다. 개발셋 50개는 전후 평가에만 사용하고
잠금 테스트셋 100개는 읽지 않는다. 두 데이터의 정답은 아직 독립 검수 전이다.
기존 모델과 결과를 비교하기 위한 실험이며 백엔드 운영 모델 설정은 바꾸지 않는다.
실제 계약서 성능이나 법률 판단 품질을 이 결과로 확정하지 않는다.

## Drive clean core 자료와 합쳐 비교

Drive의 `processed_core_clean` 폴더에서 `train.jsonl`, `test.jsonl`,
`labels.json`을 `data/classifier/processed_core_clean/`에 둔다. 이 원자료는
Git에서 제외한다. 생성한 분할 파일은 [Drive 공유 폴더](https://drive.google.com/drive/folders/1ct1eGT-yiN-oIkU8rkhkABY4uiUgTTXR)에 있다.

```bash
../CheckIt_BE/.venv/bin/python -m Classifier.prepare_core
../CheckIt_BE/.venv/bin/python -m Classifier.train_core --device mps
```

분할은 core의 기존 train/test를 합쳐 라벨 충돌 28행을 제외한 뒤, 문서
묶음(`group`) 기준으로 약 80/10/10을 만든다. 같은 문장은 분할 전체에서
한 번만 사용한다. 합성 240개는 train에만 추가한다. 실제 생성 건수와
입력 SHA-256은 `outputs/classifier/combined_split_v004/manifest.json`에
기록된다. 이 분할은 현재 비교 실험용으로 고정했으며 기존 Drive 원본은
수정하지 않는다. 이미 생성된 분할을 다시 만들 때는 새 `--output`
경로를 지정하고 학습에도 같은 경로를 `--split`으로 전달한다.

`train_core`는 고정된 `klue/roberta-base`에서 각각 시작해 core 단독과
core+합성 모델을 같은 설정으로 학습한다. validation macro F1으로 epoch를
고르고 test는 선택 후 한 번만 평가한다. 산출물은
`outputs/classifier/core_comparison_*/`에 저장된다. 이 test는 Drive의
기존 `test.jsonl`이나 별도 잠금 합성 테스트셋과 다른 새 분할이다.
이번 실행의 집계 결과는 `Classifier/reports/processed_core_clean_plus_synthetic_2026_v001.json`에 보존했다.
