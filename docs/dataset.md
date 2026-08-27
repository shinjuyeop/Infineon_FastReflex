# Hazard Dataset Contract

## 상태와 범위

`hazard_dataset_contract_v1`은 historical IMU6 Pilot schema와 frozen raw annotation을 보존한다. `hazard_dataset_contract_v2`는 이를 폐기하거나 label을 바꾸지 않고 candidate `foot_fsr` runtime field와 sensor validity만 확장한다. 기존 `hazard_pilot_20260827`은 read-only이며, 별도 `hazard_sensor_pilot_20260827`이 같은 40 conditions를 담는다.

Primary task는 하나의 3-class classification이다.

| ID | Class | 의미 |
|---:|---|---|
| 0 | `NORMAL` | 유효한 보행 중 frozen hazard criterion을 만족하지 않는 안정 상태 |
| 1 | `SLIP` | terrain 이름과 무관하게 established Slip physical oracle이 active |
| 2 | `SINK` | foot-ground sinking/compliance로 locomotion 또는 posture stability가 meaningfully degraded된 hazard state |

Terrain, scenario, contact, exact simulator state와 physical oracle은 label/diagnostic/metadata 전용이다. Runtime model input에는 넣지 않는다.

Transition study에서 `SINK_HAZARD_CRITERIA_FROZEN`을 확정했다. Contact penetration만 존재하고 자세와 보행이 안정적인 상태는 primary `SINK`가 아니다. 아래 `sink_physical_active`는 원인 측 precursor diagnostic이며 class label과 동치가 아니다. Frozen effect gate는 patch-linked physical sink 뒤 pelvis tilt가 benign-control envelope를 넘는지를 사용한다. 이 정의와 기존 Pilot raw annotation은 historical contract로 불변이다.

후속 bounded sanity는 passive support joint의 실제 vertical displacement spread를 outcome-independent `UNEVEN_SUPPORT_SINK` physical clock 후보로 검증했다. 이는 향후 observability study에만 적용하며 기존 Pilot을 소급 relabel하지 않는다. 새 clock을 materialize하려면 dataset schema/provenance를 명시적으로 revision해야 한다.

## Runtime sensor contract

Historical v1 한 sample의 model input은 pelvis에 부착된 IMU 6축이다. 고정 channel order는 다음과 같다.

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

### Candidate bilateral virtual FSR

V2 sensor-capable run은 `foot_fsr [N,8] float32`, unit Newton, 1 kHz를 추가한다. Channel order는 다음과 같이 고정한다.

| Index | Channel | Foot-local region |
|---:|---|---|
| 0 | `left_front_left` | left foot +x/+y |
| 1 | `left_front_right` | left foot +x/-y |
| 2 | `left_rear_left` | left foot -x/+y |
| 3 | `left_rear_right` | left foot -x/-y |
| 4 | `right_front_left` | right foot +x/+y |
| 5 | `right_front_right` | right foot +x/-y |
| 6 | `right_rear_left` | right foot -x/+y |
| 7 | `right_rear_right` | right foot -x/-y |

각 값은 named sole geom과 allowed terrain geom의 실제 MuJoCo contact에서 `mj_contactForce` contact-frame normal component를 읽고, contact world position을 해당 ankle-roll/sole body local frame으로 변환해 quadrant별 합산한 값이다. No contact는 정확히 0 N이다. Terrain identity, Slip/Sink oracle, penetration과 label은 값을 만들지 않는다. Contact position, geom ID와 full wrench는 runtime input으로 노출하지 않는다. Raw 값에는 noise, quantization, saturation, filter, calibration curve 또는 resistance conversion을 적용하지 않는다.

Candidate profile은 `imu6=pelvis_imu`, `fsr8=foot_fsr`, `fusion14=[pelvis_imu, foot_fsr]`다. 이는 Pilot comparison을 위한 선택지이며 final sensor architecture가 아니다.

### Missing, dropped, duplicate sample

- Collector는 sequence gap, non-monotonic timestamp, non-finite channel, duplicate 또는 out-of-order sample을 감추지 않는다.
- 누락 sample을 zero-fill, forward-fill 또는 interpolation하여 authoritative raw trace를 만들지 않는다.
- `sample_valid`와 channel validity 정보를 보존하고 manifest에 run별 missing/drop count를 기록한다.
- Training/evaluation window가 invalid sample 또는 timestamp discontinuity를 하나라도 가로지르면 그 window를 제외한다.
- Raw run 자체는 provenance와 진단을 위해 보존한다. 재수집 여부는 manifest 검토에서 결정한다.

## 최소 preprocessing

각 profile의 기본 입력은 고정 순서 raw channels다. 허용되는 유일한 추가 preprocessing은 해당 profile의 train split sample에서만 계산한 per-channel mean/std z-score normalization이다.

- Validation/test 또는 전체 dataset 통계를 normalization에 사용하지 않는다.
- Mean/std와 이를 계산한 train run 목록 또는 split revision을 model artifact provenance에 기록한다.
- FFT, wavelet, residual, drift estimator, derivative, moving variance, complex filter, terrain-conditioned normalization을 사용하지 않는다.
- q/dq, Foot IMU, motor torque, terrain ID와 exact simulator state를 feature로 추가하지 않는다. FSR은 V2의 명시된 raw `foot_fsr` candidate로만 허용한다.
- Model family별 feature engineering을 만들지 않는다.

새 feature는 raw-signal/Time-to-Separation 연구 근거와 protocol revision이 있기 전에는 추가하지 않는다.

## Authoritative raw run

Dataset의 authoritative source는 미리 잘린 window가 아니라 simulation run 전체의 1 kHz time-series다. 각 run은 최소 다음을 보존한다.

### Runtime sensor arrays

| Field | Shape/type | 의미 |
|---|---|---|
| `timestamp_us` | `[N] int64` | run-local monotonic timestamp |
| `sequence` | `[N] int64` | 연속 sample 번호 |
| `pelvis_imu` | `[N, 6] float32` | 고정 channel order의 raw IMU |
| `foot_fsr` | `[N, 8] float32` | V2 candidate raw virtual FSR, N |
| `sample_valid` | `[N] bool` | 이 sample의 runtime input 완전성 |
| `channel_valid` | `[N, 6] bool` | channel별 validity |
| `fsr_valid` | `[N, 8] bool` | V2 FSR channel별 validity |

### Simulator-only label and diagnostic arrays

좌/우 배열 순서는 `[left, right]`로 고정한다. Pilot의 NPZ는 다음 simulator-only field를 runtime input과 별도 key로 저장한다.

- `hazard_class_id`: `[N] int8`; eligible sample에서는 0/1/2
- `training_eligible`: `[N] bool`
- 좌/우 named sole-ground physical contact와 physical-contact rising-edge touchdown
- 좌/우 force-loaded state
- `established_slip_active`, per-foot onset, `any_slip_active`와 ANY onset
- Slip continuous diagnostics: touchdown anchor drift norm과 tangential velocity
- `sink_physical_active`, per-foot onset과 patch-linked onset
- `soft_patch_contact`, `low_friction_patch_contact`
- Sink continuous diagnostics: raw contact penetration과 first-loaded-reference 대비 변화량
- Effect diagnostics: pelvis/root world z와 tilt, angular/linear velocity, commanded-forward velocity와 tracking error
- `sink_degradation_active`/onset과 patch-linked `sink_hazard_active`/onset
- `dual_hazard_active`
- `pre_fall_valid`와 first-fall/censor marker
- 필요 시 exact root/foot pose와 velocity. 이는 항상 diagnostic-only로 표시한다.

Deformable-support future study는 기존 Pilot NPZ에 없는 simulator-only `support_surface_displacement_m [N,2,4]`, velocity, cell contact, spread와 s1 onset을 추가할 수 있다. 이 배열은 raw IMU/FSR runtime input이 아니며 기존 schema에 조용히 추가하지 않는다.

`hazard_class_id=-1`은 primary class가 아니라 `EXCLUDED/UNRESOLVED` sentinel로 예약한다. Pilot에서는 qualifying Slip run의 `[t0,t1)`, hazardous Sink run의 `[t0,t2)`, censor 이후, invalid/dual run 전체를 `-1`과 `training_eligible=false`로 보존한다. Slip은 t1부터 class 1, Sink는 frozen t2부터 class 2를 기록한다. 이는 raw-state annotation이며 최종 window policy가 아니다.

### Run metadata

Pilot manifest는 각 run에 다음 metadata를 기록한다.

- `run_id`
- `scenario_family`, `intended_role`, observed physical outcome
- terrain, commanded speed, patch start/width와 Sink side/severity
- fixed initial controller phase와 first patch-contact policy phase
- event timing, censor reason, left/right/dual coverage와 validity/drop summary
- policy와 run-file SHA-256

현재 simulator에는 random source가 없어 dataset-level `simulator_deterministic=true`, `random_seed=null`로 기록한다. 의미 없는 seed 반복을 independent data로 세지 않는다. 이 metadata는 coverage와 향후 split을 위한 것이며 runtime input이 아니다. Randomization이나 command profile이 실제로 추가될 때만 seed/group 또는 command trace를 도입한다.

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

Hazardous episode의 future early-detection reference 후보는 결과가 이미 드러난 t2가 아니라 원인 onset t1이다. `[t1,t2)` sample/window eligibility와 latency gate는 Raw IMU sanity/Time-to-Separation에서 검증하기 전까지 `training_eligible=false`로 유지한다. Pilot materialization은 이 원칙을 적용했으며 training label/window가 확정됐다는 뜻은 아니다.

`SINK_HAZARD_CRITERIA_FROZEN`

### Future deformable-support `SINK` proxy

`SINK_DEFORMABLE_SUPPORT_PROXY_SUPPORTED_FOR_OBSERVABILITY_STUDY`는 기존 outcome-based contract를 삭제하지 않는 future-study candidate다. Passive vertical slide support의 per-side cell 순서 `[entry_medial, entry_lateral, exit_medial, exit_lateral]`에서 positive-downward displacement `d`를 읽고 다음 metric을 사용한다.

```text
support_surface_spread_m = max(d) - min(d)
```

Candidate s1은 같은 foot의 deformable-patch physical contact episode에서 patch contact를 이미 보았고 foot이 loaded/pre-censor인 동안 spread `>= 0.010 m`가 1 kHz에서 20 consecutive samples 지속된 첫 active sample이다. 일부 cell이 unload되어도 그 cell joint displacement를 metric에서 제거하지 않는다. Foot episode 종료/변경 또는 censor에서 persistence를 reset한다.

이 clock은 future t2/fall/recovery, pelvis tilt, IMU, FSR, robot joint state와 terrain identity를 사용하지 않는다. Balanced support는 네 geom이 하나의 joint를 공유하므로 level displacement는 `BENIGN_SOFT`, 독립 cell이 공간적으로 비균일하게 내려가 criterion을 만족하면 `UNEVEN_SUPPORT_SINK`다. Fall, recovery와 posture degradation은 outcome diagnostic only다.

이 구현은 granular soil이나 measured soil mechanics가 아닌 deformable-support engineering proxy다. 32-run sanity에서 rigid/balanced benign firing 0/14, primary moderate uneven detection 11/12와 fall/non-fall diversity를 확인했지만 runtime sensor separability나 hardware validity를 뜻하지 않는다. 근거는 [`20260827_sink_deformable_support_proxy_sanity.md`](../reports/20260827_sink_deformable_support_proxy_sanity.md)에 둔다.

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

FSR Observability Pilot의 early-target은 raw field를 수정하지 않는 experiment-local derived label이다. Frozen Sink qualification을 나중에 만족한 run만 `[t1,t3)`를 SINK로 retrospectively 부여하고 `[t1,t2)`를 포함한다. Runtime input에는 future t2가 들어가지 않는다. SLIP은 기존 t1부터, hazard-positive stable prefix는 run start부터 t0 전까지 NORMAL이며 `[t0,t1)`은 ambiguous로 제외한다. Benign uniform sand와 mild/moderate Sink는 censor 전까지 NORMAL hard negative다.

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

## Storage와 dataset identity

Pilot은 복잡한 database나 shard framework 없이 한 NPZ를 한 complete run으로 저장한다.

```text
data/
  raw/
    <dataset_id>/
      manifest.csv
      metadata.json
      runs/
        <run_id>.npz
```

- `metadata.json`: dataset identity, 공통 schema, generator/simulator provenance
- `manifest.csv`: 한 row당 한 run의 condition, observed outcome, event timing, validity/drop과 NPZ SHA-256
- `runs/<run_id>.npz`: 한 complete 1 kHz raw run. Pickle/object array를 사용하지 않고 run 경계를 concatenate하지 않는다.

Collector는 temporary directory에서 전체 run/manifest/metadata/SHA validation을 통과한 뒤 final directory로 atomic rename하며 기존 dataset identity를 overwrite하지 않는다. Generated dataset은 `data/raw/`에 두고 Git에 commit하지 않는다.

Dataset identity의 required fields는 다음과 같다.

- `dataset_id`
- `schema_version`
- `generator_version`
- `source_commit`
- `created_at`
- `sample_rate_hz`
- `channel_order`

첫 IMU6 materialization과 별도 sensor materialization은 각각 40 runs/320,000 samples다. Sensor dataset collector는 같은 run ID의 모든 v1 common field를 baseline NPZ와 bit-identical 비교한 뒤에만 저장한다. 상세 결과는 [`20260827_hazard_pilot_dataset.md`](../reports/20260827_hazard_pilot_dataset.md)와 [`20260827_fsr_observability_pilot.md`](../reports/20260827_fsr_observability_pilot.md)에 기록한다. Metadata는 verified policy hash, simulator/config digest, deterministic/no-seed 상태와 manifest hash를 포함한다.

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
