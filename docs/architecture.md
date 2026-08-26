# Architecture

현재 repository는 Hazard Dataset Contract 단계이며 simulator, dataset, model은 아직 구현하지 않았다.

## Existing Frozen Terrain

```text
Foot FSR4 + Foot IMU6
  -> Frozen Terrain Classifier
  -> Concrete / Marble / Ice / Sand
```

Terrain classifier는 legacy repository에서 검증된 asset이다. 향후 provenance를 갖춘 frozen release로 명시적 검토 후 migration한다. 이번 milestone에서는 코드나 model을 옮기지 않는다.

## New Hazard Model

```text
Waist/Pelvis IMU6 @ 1 kHz
  -> Raw causal sequence
  -> Minimal normalization
  -> PyTorch temporal model
  -> NORMAL / SLIP / SINK
```

Candidate model families:

- MLP baseline
- CNN1D
- GRU
- LSTM

## 초기 연구 순서

1. Dataset과 provenance를 먼저 설계한다.
2. Raw signal과 label 가능성을 분석한다.
3. Causal window와 latency를 연구한다.
4. 동일한 validation protocol로 candidate model을 비교한다.
5. 검증된 Float model과 계약 artifact만 export한다.

첫 설계에서는 복잡한 handcrafted feature pipeline을 사용하지 않는다. Research 경계 뒤의 quantization, Vela, firmware, HIL은 E84 deployment repository가 담당한다.
