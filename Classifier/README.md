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
