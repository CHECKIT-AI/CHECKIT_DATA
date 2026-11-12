---
language: ko
library_name: transformers
pipeline_tag: text-classification
base_model: klue/roberta-base
tags:
- multi-label-classification
- contract-clauses
---

# CheckIt 근로계약서 조항 분류 모델 v2

`klue/roberta-base`에서 근로계약서 조항의 10개 라벨을 다중 분류하도록 학습한 모델입니다. `labels.json`의 순서대로 sigmoid 확률에 임계값 0.5를 적용합니다. 위법 여부를 판정하는 모델은 아닙니다.

- 학습 자료: clean core 3,470개와 별도 작성한 합성 조항 240개. 합성 자료는 train에만 사용했습니다.
- 분할: 문서 묶음 기준 train 3,710개 / validation 434개 / test 446개.
- 학습 설정: 3 epoch, batch size 16, AdamW learning rate 2e-5, BCEWithLogitsLoss. validation macro F1로 epoch를 선택했습니다.
- core test 446개: micro F1 0.860, macro F1 0.849, exact match 83.9%. 이전 2 epoch 실험은 각각 0.858, 0.829, 83.6%였습니다.

합성 라벨은 독립 검수 전입니다. test는 이전 실험에서도 확인한 자료이므로 수치는 탐색 결과이며, 실제 계약서나 법률 판단의 정확도를 보장하지 않습니다. 입력 데이터, 학습 설정과 라벨별 결과는 [실험 보고서](report.json)에 기록했습니다.
