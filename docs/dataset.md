# Hazard Dataset Contract V1

## 상태와 범위

`HAZARD_DATASET_CONTRACT_V1`은 새 Hazard dataset의 schema와 label 원칙을 정의한다. 최소 simulator migration과 contract parity test는 완료했지만 dataset/window 생성, model 구현 또는 training은 수행하지 않았다.

Primary task는 하나의 3-class classification이다.

| ID | Class | 의미 |
|---:|---|---|
| 0 | `NORMAL` | 유효한 보행 중 frozen hazard criterion을 만족하지 않는 안정 상태 |
| 1 | `SLIP` | terrain 이름과 무관하게 established Slip physical oracle이 active |
| 2 | `SINK` | foot-ground sinking/compliance로 locomotion 또는 posture stability가 meaningfully degraded된 hazard state |

Terrain, scenario, contact, exact simulator state와 physical oracle은 label/diagnostic/metadata 전용이다. Runtime model input에는 넣지 않는다.

Transition study에서 `SINK_HAZARD_CRITERIA_FROZEN`을 확정했다. Contact penetration만 존재하고 자세와 보행이 안정적인 상태는 primary `SINK`가 아니다. 아래 `sink_physical_active`는 원인 측 precursor diagnostic이며 class label과 동치가 아니다. Frozen effect gate는 patch-linked physical sink 뒤 pelvis tilt가 benign-control envelope를 넘는지를 사용한다.

## Runtime sensor contract

한 sample의 model input은 pelvis에 부착된 IMU 6축뿐이다. 고정 channel order는 다음과 같다.

| Index | Channel | Unit | Axis |
|---:|---|---|---|
| 0 | `accel_x` | m/s² | pelvis +x, 전방 |
| 1 | `accel_y` | m/s² | pelvis +y, 좌측 |
| 2 | `accel_z` | m/s² | pelvis +z, 위쪽 |
| 3 | `gyro_x` | rad/s | pelvis +x축 기준 right-hand positive |
| 4 | `gyro_y` | rad/s | pelvis +y축 기준 right-hand positive |
| 5 | `gyro_z` | rad/s | pelvis +z축 기준 right-hand positive |

- Sensor type: Waist/Pelvis IMU 6-axis
- Sample rate: 정확히 1,000 Hz, nominal period 1,000 µs
- Coordinate frame: MuJoCo `imu` site-local frame. 이 site는 `pelvis` body 원점에 `pos="0 0 0"`, 별도 rotation 없이 부착되어 site frame과 pelvis body frame이 같다.
- Axis convention: legacy G1 model의 neutral pose에서 +x forward, +y left, +z up인 right-handed frame이다. 로봇이 회전하면 channel frame도 pelvis와 함께 회전한다. World-frame 값이 아니다.
- Accelerometer semantics: MuJoCo accelerometer의 raw local specific-force output을 저장한다. Dataset 경로에서 gravity compensation을 하지 않는다.
- Proposed value dtype: `float32`
- Timestamp: run 시작 기준 robot/simulator monotonic `timestamp_us`, `int64`, 단위 µs. Host arrival time으로 대체하지 않는다.
- Sequence: `sequence`, `int64`, 0부터 1씩 증가하며 timestamp와 같은 sample을 식별한다.

축 방향의 legacy source 근거는 forward command가 command vector의 x 성분이고 base forward displacement를 `qpos[0]`으로 측정하며, left/right body가 각각 +y/-y에 배치되고 height/fall이 z 성분으로 정의된 G1 model/controller다.

이 좌표계는 migrated MuJoCo G1 model에서 deterministic test로 검증했다. `imu` site의 pelvis binding, zero translation/identity rotation, neutral site frame, 좌우 body 위치와 accel-then-gyro channel order가 계약과 일치한다. 실제 G1 또는 target IMU mounting/orientation parity는 deployment 전 별도 검증 대상이다. 축을 바꿔야 한다면 기존 dataset과 같은 schema version으로 조용히 바꾸지 않는다.

### Missing, dropped, duplicate sample

- Collector는 sequence gap, non-monotonic timestamp, non-finite channel, duplicate 또는 out-of-order sample을 감추지 않는다.
- 누락 sample을 zero-fill, forward-fill 또는 interpolation하여 authoritative raw trace를 만들지 않는다.
- `sample_valid`와 channel validity 정보를 보존하고 manifest에 run별 missing/drop count를 기록한다.
- Training/evaluation window가 invalid sample 또는 timestamp discontinuity를 하나라도 가로지르면 그 window를 제외한다.
- Raw run 자체는 provenance와 진단을 위해 보존한다. 재수집 여부는 manifest 검토에서 결정한다.

## 최소 preprocessing

V1의 기본 입력은 위 순서의 raw IMU 6 channels다. 허용되는 유일한 추가 preprocessing은 train split sample에서만 계산한 per-channel mean/std z-score normalization이다.

- Validation/test 또는 전체 dataset 통계를 normalization에 사용하지 않는다.
- Mean/std와 이를 계산한 train run 목록 또는 split revision을 model artifact provenance에 기록한다.
- FFT, wavelet, residual, drift estimator, derivative, moving variance, complex filter, terrain-conditioned normalization을 사용하지 않는다.
- q/dq, FSR, Foot IMU, motor torque, terrain ID와 exact simulator state를 feature로 추가하지 않는다.
- Model family별 feature engineering을 만들지 않는다.

새 feature는 raw-signal/Time-to-Separation 연구 근거와 protocol revision이 있기 전에는 추가하지 않는다.

## Authoritative raw run

Dataset의 authoritative source는 미리 잘린 window가 아니라 simulation run 전체의 1 kHz time-series다. 각 run은 최소 다음을 보존한다.

### Runtime sensor arrays

| Field | Proposed shape/type | 의미 |
|---|---|---|
| `timestamp_us` | `[N] int64` | run-local monotonic timestamp |
| `sequence` | `[N] int64` | 연속 sample 번호 |
| `pelvis_imu` | `[N, 6] float32` | 고정 channel order의 raw IMU |
| `sample_valid` | `[N] bool` | 이 sample의 runtime input 완전성 |
| `channel_valid` | `[N, 6] bool` | channel별 validity |

### Simulator-only label and diagnostic arrays

현재 in-memory baseline은 좌/우 배열 순서를 `[left, right]`로 고정한다. 구체적인 NPZ key와 storage layout은 pilot dataset milestone에서 확정하되 다음 정보는 손실 없이 저장한다.

- `hazard_class_id`: `[N] int8`; eligible sample에서는 0/1/2
- `training_eligible`: `[N] bool`
- 좌/우 named sole-ground physical contact와 physical-contact rising-edge touchdown
- 좌/우 force-loaded state 및 raw contact episode ID
- `established_slip_active`, `established_slip_onset`
- Slip continuous diagnostics: foot-ground tangential relative velocity, touchdown anchor-relative x/y displacement와 그 norm
- `sink_physical_active`, `sink_physical_onset`, `sink_physical_episode_id`
- `soft_patch_contact`, 그 onset과 patch-linked `sink_physical_after_patch_onset`
- Sink continuous diagnostics: contact-foot world z/vertical velocity, touchdown-relative downward displacement, surface-relative sole depth, raw contact penetration, first-loaded-reference penetration과 그 변화량
- 좌/우 loaded penetration asymmetry와 loaded/contact state
- Effect diagnostics: pelvis/root world z, orientation/roll/pitch/tilt, angular/linear velocity, commanded-forward velocity tracking error
- t0 전 1,000 ms baseline mask와 pelvis z/tilt/forward velocity/angular-speed event-relative change
- `sink_degradation_active`/onset과 patch-linked `sink_hazard_active`/onset
- `dual_hazard_active`
- `pre_fall_valid`와 first-fall/censor marker
- 필요 시 exact root/foot pose와 velocity. 이는 항상 diagnostic-only로 표시한다.

`hazard_class_id=-1`은 primary class가 아니라 `EXCLUDED/UNRESOLVED` sentinel로 예약한다. Dual hazard, fall-censored sample, invalid sensor sample, 그리고 아직 검증되지 않은 early interval처럼 3-class 학습에 넣을 수 없는 sample에 사용한다. 이 sample을 `NORMAL`로 바꾸지 않는다.

### Run metadata

각 run에는 적어도 다음 metadata가 필요하다.

- `run_id`
- simulator/physics/sensor random `seed` 또는 명시적인 seed set
- `scenario_family`
- commanded speed 또는 commanded speed profile
- terrain configuration/realization
- `variation_group_id`
- scenario realization/group ID
- initial controller/gait phase
- dual-hazard 발생 여부와 run-level validity summary

이 metadata는 coverage와 split을 위한 것이며 runtime input이 아니다. Acceleration/deceleration처럼 command가 시간에 따라 달라질 때에는 command trace 또는 재현 가능한 profile reference도 diagnostic으로 보존한다.

## Physical label contract

### 공통 contact와 validity

Legacy reference에서 physical contact는 named sole collision sphere와 allowed ground geom 사이의 MuJoCo contact다. Touchdown은 raw physical-contact episode의 rising edge이며, force-loaded state와 구분한다. Legacy oracle은 force-derived hysteresis load state(load-on 5 N, load-off 2.5 N), touchdown 뒤 첫 10 ms 제외, first fall 이전이라는 validity를 사용했고 persistence는 contact episode 경계를 넘지 않았다.

V1 migration은 이 구분을 보존해야 한다.

1. `physical_contact`: exact MuJoCo contact, label-only
2. `touchdown`: 새 physical-contact episode의 첫 sample, label-only
3. `force_loaded`: contact load 조건, label/diagnostic-only
4. `pre_fall_valid`: first fall 이전만 true
5. `established_slip`: frozen Slip metric/persistence를 만족한 reference label
6. `sink_physical_active`: 침투 metric/persistence를 만족한 precursor diagnostic이며 primary class가 아님

### `NORMAL`

`NORMAL`은 Concrete 직진 보행이나 terrain identity를 뜻하지 않는다. Runtime sample이 유효하고 pre-fall이며 frozen hazard criterion을 만족하지 않는 정상 보행 상태다. `sink_physical_active`만 있고 posture/gait가 안정적인 uniform compliant ground도 `NORMAL` candidate가 될 수 있다. 다음 variation을 충분히 포함해야 한다.

- straight walking, gentle/wide turn
- acceleration/deceleration과 여러 command speed
- 여러 gait phase와 controller initial phase
- strong but valid touchdown, double-support transition
- left/right asymmetric normal movement
- 여러 random seed와 terrain realization

Concrete, Marble, Ice, Sand 어느 terrain에서도 실제 hazard criterion을 만족하지 않은 구간은 `NORMAL` candidate가 될 수 있다. 반대로 terrain 이름만으로 class를 정하지 않는다. 확정되지 않은 pre-established/early interval은 `NORMAL` training sample로 자동 포함하지 않는다.

### `SLIP`: legacy stable reference

V1은 새 incipient threshold를 만들지 않고 legacy의 `ESTABLISHED_SLIP`을 stable reference label로 보존한다.

- Physical contact: 해당 foot의 named sole-ground contact
- Touchdown anchor: raw contact episode 첫 sample의 foot world x/y
- Continuous motion: 현재 foot x/y와 anchor 차이 및 그 Euclidean norm; 고정 ground에서는 tangential foot-ground relative displacement다. Tangential velocity도 별도 저장한다.
- Eligibility: force loaded, touchdown transient 10 ms 이후, pre-fall, 같은 contact episode
- Established criterion: anchor displacement norm `>= 0.050 m`가 1 kHz에서 3 consecutive samples
- Reset: invalid/load loss/contact episode 변경 또는 criterion 미충족 시 persistence reset

`ESTABLISHED_SLIP` onset은 이미 50 mm 이동과 3 ms persistence 뒤의 reference이며 최초 물리 motion onset이 아니다. Legacy의 incipient-onset 후보들은 established-event coverage와 clean-normal false-onset validation을 통과하지 못했다. 따라서 V1은 incipient label을 invent하지 않는다. Raw continuous metrics와 stable established onset을 함께 보존하고 Phase 4 Time-to-Separation에서 early reference를 별도로 검증한다.

Pilot의 canonical Slip scenario 후보는 정상 concrete 보행 뒤 full-width finite low-friction patch에 진입하는 transition이다. `t0_patch_contact`는 named sole과 patch의 첫 physical contact, `t1_established_slip`은 기존 criterion의 첫 ANY-foot onset이다. Left/right onset은 diagnostic으로 보존하지만 한 발이라도 established Slip이면 primary 의미는 `SLIP`이다. Terrain identity나 affected-foot ownership으로 class를 만들지 않으며 first fall/non-foot censor 이후는 training-valid evidence에서 제외한다. 검증 결과는 [`20260826_slip_transition_sanity.md`](../reports/20260826_slip_transition_sanity.md)에 기록한다.

### `sink_physical`: physical precursor diagnostic

Legacy의 후반 walking physical oracle은 삭제하지 않고 `sink_physical_active`라는 simulator-only precursor diagnostic으로 보존한다.

- Physical quantity: named sole-ground contacts의 `max(0, -contact.dist)` contact penetration
- Touchdown/load reference: 같은 raw contact episode에서 첫 force-loaded sample의 penetration
- Continuous metric: 현재 penetration에서 first-loaded reference를 뺀 `loaded_penetration_change_m`
- Eligibility: force loaded, touchdown transient 10 ms 이후, finite reference, pre-fall, 같은 contact episode
- Physical criterion: `loaded_penetration_change_m >= 0.0055 m`가 1 kHz에서 20 consecutive samples
- Reset: invalid/load loss/contact episode 변경 또는 criterion 미충족 시 persistence reset

이 criterion은 "발-지면 contact에서 의미 있는 추가 침투가 발생했다"는 뜻일 뿐 `SINK` class를 확정하지 않는다. Legacy terrain과 현재 scenario는 deformable continuum/mesh depth state를 제공하지 않고 compliance를 MuJoCo contact response와 penetration으로 근사한다. 그러므로 이를 실제 재료 변형 깊이라고 과대해석하지 않는다.

### Primary `SINK` hazard

Primary `SINK`는 patch-linked `sink_physical_active`와 locomotion/posture degradation이 하나의 시간 사건으로 연결된 hazard state다. 단순 penetration, terrain 이름 또는 compliance profile만으로 class를 만들지 않는다.

Transition event의 simulator-only timeline은 다음과 같다.

1. `t0_patch_contact`: named sole geom과 지정 soft patch geom의 첫 physical contact
2. `t1_sink_physical`: 같은 raw contact episode에서 처음 발생한 `sink_physical_active` onset
3. `t2_degradation`: t1 뒤 pelvis tilt가 `0.04454633221030235 rad`보다 큰 상태가 1 kHz에서 20 consecutive samples 지속된 첫 active sample
4. `t3_censor`: first fall 또는 non-foot surface contact onset

Tilt threshold는 concrete, uniform sand, left/right mild transition의 8초 benign controls에서 관측한 최대 tilt `2.5523168°`의 upper envelope다. 이 값은 terrain이나 Slip 여부를 사용하지 않으며 fall 자체를 effect gate로 쓰지 않는다. Transition severe는 좌/우 모두 t2 뒤 약 1.5초 이상의 pre-censor interval을 보였고 mild/moderate와 uniform sand는 gate를 넘지 않았다. 근거와 한계는 [`20260826_sink_transition_criteria.md`](../reports/20260826_sink_transition_criteria.md)에 기록한다.

- `BENIGN_SINK_EPISODE`: t0와 t1은 있지만 관찰 가능한 pre-censor run에서 t2가 없음
- `HAZARDOUS_SINK_EPISODE`: t0 → t1 뒤 t2가 발생
- `DUAL_PHENOMENON`: established Slip onset이 t2보다 먼저 발생한 hazardous episode. Raw run은 보존하되 V1 train/evaluation에서는 제외
- Censor가 먼저 와 qualification할 수 없으면 `INCONCLUSIVE`이며 `training_eligible=false`

Hazardous episode의 future early-detection reference 후보는 결과가 이미 드러난 t2가 아니라 원인 onset t1이다. `[t1,t2)` sample/window eligibility와 latency gate는 Pilot/Time-to-Separation에서 검증하기 전까지 `training_eligible=false`로 유지한다. Frozen criterion은 episode qualification을 고정한 것이며 아직 dataset을 생성했다는 뜻은 아니다.

`SINK_HAZARD_CRITERIA_FROZEN`

### Terrain shortcut과 dual hazard

- Ice는 곧 `SLIP`이 아니며 Sand는 곧 `SINK`가 아니다.
- Terrain/scenario/exact state는 model input으로 사용하지 않는다.
- Established SLIP onset이 frozen SINK degradation onset보다 먼저인 raw trace는 보존하고 `dual_hazard_active=true`로 표시한다. `sink_physical_active`만으로 dual hazard를 선언하지 않는다.
- Dual-hazard sample/event/run은 V1 primary training과 evaluation에서 제외한다. 어느 class를 우선할지 규칙을 만들지 않는다.

### Sample과 window label의 경계

Raw run은 stable established states, onset pulses, validity와 continuous diagnostics를 sample level로 보존한다. Established onset 전 interval의 label이나 training window endpoint label 정책은 이번 milestone에서 확정하지 않는다.

Pilot과 Phase 4 이후 별도 config revision에서 다음을 결정한다.

- early interval의 검증된 시작과 제외 범위
- window endpoint/within-window label rule
- onset 전 prediction horizon을 둘지 여부
- causal latency와 Time-to-Separation 기준

그 전까지 unresolved early interval은 `training_eligible=false`이며 강제로 `NORMAL` 처리하지 않는다.

## Window contract

Raw run은 충분히 긴 sequence로 한 번 저장하고, training/evaluation 시점에 causal window를 자른다. Candidate는 다음과 같다.

| Window | 1 kHz sample count |
|---:|---:|
| 20 ms | 20 |
| 30 ms | 30 |
| 50 ms | 50 |
| 100 ms | 100 |

Window는 endpoint를 포함하고 미래 sample을 포함하지 않는다. 하나의 window size를 위해 dataset을 다시 생성하지 않는다. 모든 candidate는 같은 raw release와 같은 split assignment를 사용한다.

## Split과 leakage contract

Window random split은 금지한다. 같은 source run의 sample/window는 반드시 하나의 split에만 속한다.

- `TRAIN`: model parameter와 train-only normalization fitting
- `VALIDATION`: architecture, window와 hyperparameter 선택
- `TEST`: 선택이 끝난 frozen model의 최종 평가. Model 선택에 사용하지 않는다.

최소 기준은 run-disjoint다. 가능하면 seed group, variation group, scenario realization, initial gait phase도 validation/test 사이에서 분리한다. 일부 scenario combination 전체를 held out하는 generalization split을 별도로 둘 수 있다. 권장 시작 비율은 coverage가 허용할 때 run 기준 70/15/15지만, 실제 비율과 run count는 pilot 결과 뒤 config revision에서 고정한다.

Split manifest는 다음을 검증해야 한다.

- source run overlap 0
- declared group overlap 0 또는 불가피한 overlap의 명시적 사유
- class뿐 아니라 speed, gait phase, initiating side, scenario family, seed, terrain realization coverage
- TEST assignment의 model-selection 전 freeze

특히 `NORMAL`을 충분히 넓게 수집한다. Legacy의 좁은 normal-domain 실험에서 false-positive generalization 문제가 있었고, 이후 coverage-corrected 연구는 speed, contact age, impact, turn/trajectory와 sensor variation을 넓혀야 했다. Class count balance만으로 이 문제를 해결했다고 간주하지 않는다.

## Storage와 dataset identity 제안

복잡한 database나 framework 없이 다음 구조로 시작한다.

```text
data/
  raw/
    <dataset_id>/
      manifest.csv
      metadata.json
      shard_000.npz
      shard_001.npz
```

- `metadata.json`: dataset identity, 공통 schema, generator/simulator provenance
- `manifest.csv`: run ID, shard/offset, sample count, metadata group, class/event coverage, validity/drop summary와 split assignment
- `shard_XXX.npz`: 하나 이상의 complete raw run. 여러 run을 concatenate하면 explicit run offsets를 저장하고 run 경계를 넘는 window를 금지한다. Pickle/object array는 사용하지 않는다.

최종 shard shape와 field name은 Phase 2 simulator migration에서 정하되 authoritative run boundary와 위 필수 정보는 바꾸지 않는다. Generated dataset은 Git에 commit하지 않는다.

Dataset identity의 required fields는 다음과 같다.

- `dataset_id`
- `schema_version`
- `generator_version`
- `source_commit`
- `created_at`
- `sample_rate_hz`
- `channel_order`

추가로 simulator/model/policy version, config digest와 split revision을 기록하는 것을 권장한다. Dataset, model, report artifact provenance는 서로 분리한다.

## Legacy read-only reference note

이 contract는 legacy repository HEAD `4194af1e0d8db8d113609c11879713c29a583261`에서 다음 구현을 read-only로 확인해 작성했다.

- `g1_29dof.xml`: pelvis `imu` site와 MuJoCo accel/gyro sensors
- `bilateral_hil_sensor_v2.py`: pelvis `[accel XYZ, gyro XYZ]` read order
- `g1_upstream_locomotion.py`: `(forward, lateral, yaw)=(speed, 0, 0)` walking command 구성
- `walking_v2_sensor_timestamp_contract_v1.py`: 1 kHz monotonic µs timestamp 원칙
- `walking_hazard_ground_truth_v1.py`: physical contact/touchdown, anchor와 penetration diagnostics
- `walking_bilateral_slip_physical_oracle_audit_v1.py`: locked 50 mm/3 ms Slip oracle
- `walking_bounded_retraining_v1.py`: 5.5 mm/20 ms Sink physical oracle
- `run_walking_legacy_sensor_time_to_separation_audit_v1.py`: incipient onset validation failure

Legacy walking studies는 주로 0.10/0.15/0.20 m/s command, 좌/우와 여러 gait phase를 조합했고, low-friction target patch/control pair로 Slip을, contact-compliance profile과 hard controls로 Sink를 생성했다. 이후 연구는 arc/straight-dither, speed transition, contact/sensor variation을 추가했다. 이 값들은 migration 참고이며 새 dataset의 고정 coverage grid는 아니다. Scenario는 event를 유도할 뿐 class label은 항상 physical oracle이 결정한다.

이번 migration에서 G1 model/sensor binding, contact episode bookkeeping, locked physical-oracle parity와 walking command adapter만 최소 재구현했다. 과거 detector/state machine, training/model, experiment runner와 dataset은 가져오지 않았다. 상세 provenance와 smoke 근거는 [`milestones/20260826_mujoco_migration.md`](milestones/20260826_mujoco_migration.md)에 기록한다.
