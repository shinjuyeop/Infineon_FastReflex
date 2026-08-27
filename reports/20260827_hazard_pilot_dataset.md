# Hazard Pilot Dataset

## 목적과 결론

`HAZARD_PILOT_DATASET`은 model training이 아니라 첫 authoritative raw Hazard artifact의 materialization과 structure/label/event/coverage 검증이다. Dataset ID는 `hazard_pilot_20260827`이며 local `data/raw/hazard_pilot_20260827/`에 생성했다.

- Source commit: `7e6fa1c168d8f9d419d49cf2fff96095e673761f`
- Policy SHA-256: `2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28`
- Experiment config: [`20260827_hazard_pilot_dataset.yaml`](../configs/experiment/20260827_hazard_pilot_dataset.yaml)
- Schema: `hazard_dataset_contract_v1`
- Created at: `2026-08-27T12:27:35+09:00`
- Manifest SHA-256: `2a539abe122f8d4e06429d39db2dc37ec6f7569a8360959e64681d9f0ac1cf3e`

40개 deterministic physical-condition run과 320,000개 raw pelvis IMU6 sample을 저장했다. Sensor invalid/drop은 0이며 observed BENIGN/SLIP/SINK가 모두 존재한다. 이는 raw signal separability나 training-window validity를 증명하지 않는다.

## Storage와 validation

```text
data/raw/hazard_pilot_20260827/
├── metadata.json
├── manifest.csv
└── runs/
    └── <run_id>.npz  # 40 files, one complete 8 s run per file
```

전체 크기는 약 25 MiB다. 각 NPZ는 `sequence [8000] int64`, `timestamp_us [8000] int64`, `pelvis_imu [8000,6] float32`, sample/channel validity와 simulator-only diagnostics를 pickle 없이 저장한다. Runtime-facing contract는 계속 sequence/timestamp/raw pelvis IMU6뿐이며 contact, terrain, patch, drift, penetration, root state와 oracle은 diagnostic/metadata 전용이다.

Collector는 temporary directory에서 모든 run의 shape/dtype/finite/timestamp/alignment, NPZ round trip, per-file SHA, manifest/metadata consistency와 orphan/missing file을 검증한 뒤 final path로 atomic rename했다. Final path는 `.gitignore`의 `data/raw/` 규칙으로 제외되고 기존 identity는 overwrite하지 않는다.

## Intended matrix와 observed outcome

모든 run은 8초, 2 kHz physics, 1 kHz sensor, fixed controller initial phase 0.0이다. 실제 random source가 없어 seed를 만들지 않았고 `simulator_deterministic=true`, `random_seed=null`로 기록했다.

| Intended group | Physical conditions | Runs | BENIGN | SLIP | SINK | DUAL | INVALID |
|---|---|---:|---:|---:|---:|---:|---:|
| NORMAL concrete | 0.10/0.15/0.20/0.25 m/s | 4 | 4 | 0 | 0 | 0 | 0 |
| NORMAL marble | 0.10/0.15/0.20/0.25 m/s | 4 | 4 | 0 | 0 | 0 | 0 |
| NORMAL uniform sand | 0.10/0.15/0.20/0.25 m/s | 4 | 4 | 0 | 0 | 0 | 0 |
| NORMAL benign Sink | left/right × mild/moderate, 0.15 m/s, p035 | 4 | 4 | 0 | 0 | 0 | 0 |
| SLIP finite Ice | 0.10/0.15/0.20/0.25 × p030/p035/p040 | 12 | 0 | 8 | 0 | 0 | 4 |
| SINK severe | left/right × 0.15/0.20/0.25 × p030/p035 | 12 | 0 | 0 | 9 | 0 | 3 |
| **Observed total** |  | **40** | **16** | **8** | **9** | **0** | **7** |

Scenario 이름으로 outcome을 강제하지 않았다. 모든 NORMAL-intended run은 BENIGN이었다. Uniform sand 4개와 mild/moderate finite Sink 4개, 총 8개 BENIGN run에는 `sink_physical` precursor가 있었지만 frozen degradation t2가 없어 SINK로 바꾸지 않았다.

## Slip coverage

8개 observed SLIP run의 coverage는 다음과 같다.

- t0 patch contact: 1,221–3,253 ms
- ANY-SLIP t1: 1,911–4,031 ms
- t0→t1 latency: 47–1,625 ms
- First-contact foot: left 4, right 4
- Slip ownership: unilateral right 2, bilateral 6
- Patch start: p030 3, p035 4, p040 1
- Speed: 0.10 3, 0.15 2, 0.20 1, 0.25 2
- Fall/non-fall: 7/1
- First-contact nominal policy phase: 0.000, 0.033, 0.400, 0.433, 0.500

| Run | t0 | ANY-SLIP t1 | Latency | First foot | Slip feet | t3 |
|---|---:|---:|---:|---|---|---:|
| `slip_ice_s010_p030` | 2,657 | 2,709 | 52 | right | right | 2,863 |
| `slip_ice_s010_p035` | 2,406 | 4,031 | 1,625 | left | bilateral | 5,062 |
| `slip_ice_s010_p040` | 3,253 | 3,300 | 47 | right | bilateral | 4,244 |
| `slip_ice_s015_p030` | 2,060 | 2,114 | 54 | right | right | 2,269 |
| `slip_ice_s015_p035` | 1,803 | 3,088 | 1,285 | left | bilateral | 5,133 |
| `slip_ice_s020_p035` | 1,505 | 2,634 | 1,129 | right | bilateral | 5,087 |
| `slip_ice_s025_p030` | 1,221 | 1,911 | 690 | left | bilateral | 4,531 |
| `slip_ice_s025_p035` | 1,221 | 1,915 | 694 | left | bilateral | N/A |

## Sink coverage

9개 observed SINK run은 severe transition의 left 5/right 4다.

- t0 patch contact: 1,221–2,422 ms
- Physical Sink t1: 1,262–2,454 ms
- Frozen degradation t2: 1,678–3,115 ms
- t0→t1 latency: 30–47 ms
- t1→t2 latency: 0–984 ms
- Patch start: p030 3, p035 6
- Speed: 0.15 4, 0.20 2, 0.25 3
- First-contact/affected side: left 5, right 4
- Fall/non-fall: 9/0

| Run | t0 | t1 | t2 | t1→t2 | t3 |
|---|---:|---:|---:|---:|---:|
| `sink_left_severe_s015_p030` | 2,422 | 2,454 | 2,454 | 0 | 2,631 |
| `sink_left_severe_s015_p035` | 1,803 | 1,845 | 2,338 | 493 | 3,852 |
| `sink_left_severe_s020_p035` | 1,809 | 1,856 | 2,332 | 476 | 3,256 |
| `sink_left_severe_s025_p030` | 1,221 | 1,262 | 1,678 | 416 | 2,660 |
| `sink_left_severe_s025_p035` | 1,221 | 1,262 | 1,678 | 416 | 2,658 |
| `sink_right_severe_s015_p030` | 2,060 | 2,090 | 2,090 | 0 | 2,291 |
| `sink_right_severe_s015_p035` | 2,096 | 2,131 | 3,115 | 984 | 4,746 |
| `sink_right_severe_s020_p035` | 1,505 | 1,539 | 2,500 | 961 | 4,961 |
| `sink_right_severe_s025_p035` | 1,508 | 1,540 | 2,495 | 955 | 5,774 |

두 p030 run의 t1→t2가 0 ms인 것은 degradation gate가 t1 시점에 이미 active여서 patch-linked hazard onset이 physical onset과 동시에 확정된 경우다. Raw arrays는 이를 그대로 보존하며 Time-to-Separation evidence로 과대해석하지 않는다. `sink_right_severe_s015_p030`, `sink_right_severe_s020_p035`에는 t2 뒤 established Slip도 있지만 frozen DUAL 정의인 “Slip이 t2보다 먼저 발생”에는 해당하지 않아 SINK outcome을 유지했다.

## Invalid, censor와 raw annotation

INVALID 7개는 모두 finite runtime input과 1 kHz timestamp가 정상이며, intended transition이 qualifying event를 만들기 전에 `nonfoot_surface_contact` censor가 발생한 physical-condition failure다. Outcome을 BENIGN이나 intended hazard로 강제하지 않고 run 전체를 `hazard_class_id=-1`, `training_eligible=false`로 보존했다.

| Run | Pre-censor t0 | t3 |
|---|---:|---:|
| `slip_ice_s015_p040` | 2,099 | 2,184 |
| `slip_ice_s020_p030` | N/A | 1,314 |
| `slip_ice_s020_p040` | N/A | 1,621 |
| `slip_ice_s025_p040` | N/A | 1,319 |
| `sink_left_severe_s020_p030` | N/A | 1,314 |
| `sink_right_severe_s020_p030` | N/A | 1,314 |
| `sink_right_severe_s025_p030` | 1,508 | 1,868 |

전체 320,000 sample의 conservative raw annotation은 NORMAL 159,754, SLIP 15,488, SINK 12,149, EXCLUDED/UNRESOLVED 132,609이며 `training_eligible=true`는 187,391개다. 이는 최종 training sample/window count가 아니다. Slip `[t0,t1)`, Sink `[t0,t2)`, censor 이후와 invalid run 전체를 unresolved로 둔 결과다.

## Acceptance, limitations와 다음 단계

- 40 independent physical conditions, observed BENIGN/SLIP/SINK, left/right Sink와 left/right Slip first contact를 확보했다.
- Transition의 pre-censor first-contact 분포는 left 11/right 12이고 nominal phase는 0.000/0.033/0.400/0.433/0.467/0.500으로 fixed p035 sanity보다 넓어졌다.
- 8개 benign physical-Sink control과 fall/non-fall Slip을 포함했다. DUAL은 0, INVALID은 7로 실제 비율을 보존했다.
- 단일 policy, fixed initial condition, straight constant-speed command와 deterministic simulation이므로 independent random realization이나 실제 센서/재료 variation은 아니다.
- SINK positive는 모두 fall로 이어져 non-fall hazardous Sink coverage가 없다. p030은 gait/censor sensitivity가 크다.
- Raw arrays의 event/class annotation은 authoritative physical state 보존용이다. Split, normalization, causal window와 `[t1,t2)` eligibility는 아직 freeze하지 않았다.

다음 단계는 별도 승인 뒤 Raw IMU sanity다. Pelvis IMU6의 event 전후 signal을 먼저 확인한 뒤 Time-to-Separation을 수행한다. Full Dataset, PyTorch, model training과 E84 작업은 시작하지 않았다.
