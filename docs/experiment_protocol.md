# Experiment Protocol

## 상태

실험 pipeline은 아직 구현되지 않았다. 아래 항목은 구현 시 지켜야 할 최소 protocol이다.

## 원칙

- 실험 차이는 `configs/` 아래 configuration으로 기록한다.
- 하나의 canonical runner와 module을 재사용한다.
- random seed, code revision, dataset revision, config, metrics를 함께 기록한다.
- primary validation은 run-disjoint 또는 group-disjoint split을 사용한다.
- model family 비교에는 동일한 dataset split과 metric 정의를 적용한다.
- accuracy뿐 아니라 class별 metric, confusion matrix, causal latency, window 크기를 검토한다.
- 최종 export 전에 재현성과 artifact provenance를 확인한다.

Metric, split, acceptance threshold의 실제 값은 dataset 특성이 확인된 뒤 고정한다.
