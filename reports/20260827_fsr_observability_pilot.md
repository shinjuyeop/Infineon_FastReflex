# FSR Observability Pilot

## 1. 목적

같은 Unitree G1 physical conditions에서 idealized bilateral virtual FSR8 또는 IMU6+FSR8이 historical IMU6보다 physical Sink onset 직후의 관측성과 benign soft-ground 분리를 개선하는지만 검증했다. 이는 FSR 채택이나 final sensor architecture 결정이 아니다.

## 2. 기존 IMU-only limitation

Frozen established-state MLP 100 ms의 continuous replay는 SLIP Recall@100 ms `0.8333 ± 0.0589`였지만 SINK는 `0.1111`, positive-margin 100 ms pre-t2 detection은 `0/7`, BENIGN sustained FP는 seed별 `7~8/16`이었다. Uniform Sand `4/4`, benign moderate Sink `2/2` FP는 pelvis IMU가 foot-level cause보다 later gait/posture effect에 민감할 가능성을 보였다. 이 historical evidence는 수정하지 않았다.

## 3. Virtual FSR physical definition

매 1 kHz sample에서 기존 named sole collision geom과 allowed terrain geom의 실제 contact만 순회한다. `mj_contactForce`의 force:torque contact-frame 반환값 중 axis 0 normal scalar를 읽고, contact world position을 해당 `ankle_roll_link` body local frame으로 변환한 다음 +x/-x를 front/rear, +y/-y를 left/right로 나눈다. 같은 quadrant의 여러 contact normal force는 합산하고 contact가 없으면 0 N이다.

Installed MuJoCo 3.11.0 header는 `mj_contactForce`가 contact frame의 6D force:torque를 반환하고 `mjContact.frame[0:3]`이 geom0→geom1 normal임을 정의한다. Deterministic test는 no-contact 0, center tie(+x/+y), front/rear, left/right mapping과 FSR quadrant 합 대 existing exact sole normal load를 `rtol=1e-6`, `atol=1e-3 N`으로 검증했다.

Terrain ID는 contact filtering에만 사용하며 값 생성에는 쓰지 않는다. Penetration, Slip/Sink oracle, label, terrain profile은 FSR 값을 조작하지 않는다.

## 4. Channel order, unit, frame

`foot_fsr [N,8] float32`, unit `N`, sample rate 1 kHz다.

| Index | Channel |
|---:|---|
| 0 | `left_front_left` |
| 1 | `left_front_right` |
| 2 | `left_rear_left` |
| 3 | `left_rear_right` |
| 4 | `right_front_left` |
| 5 | `right_front_right` |
| 6 | `right_rear_left` |
| 7 | `right_rear_right` |

Quadrant frame은 각 foot의 ankle-roll/sole body local +x forward, +y left다. Raw에는 noise, ADC quantization, saturation, filter, smoothing, calibration curve 또는 resistance conversion을 적용하지 않았다.

## 5. Observer-only parity

FSR observer는 `MjData`를 쓰지 않고 기존 contact를 읽기만 한다. 별도 collision geom, mass/inertia, friction, `solref/solimp`, controller, policy와 terrain change는 없다. 200 ms observer OFF/ON direct test에서 timestamp, pelvis IMU, metadata와 PhysicalDiagnostics 46개 array가 모두 bit-identical했다.

Collector는 새 40개 run 각각을 저장하기 전에 기존 `hazard_pilot_20260827` 대응 NPZ의 모든 v1 common field를 dtype/shape 포함 bit-identical 비교했다. NaN diagnostic 위치도 동일해야 통과한다. 따라서 timestamp, IMU, raw annotation, event timing, censor, Slip/Sink diagnostic과 outcome이 모두 동일하다.

## 6. New dataset provenance

- Dataset: `hazard_sensor_pilot_20260827`
- Schema: `hazard_dataset_contract_v2`
- Local path: `data/raw/hazard_sensor_pilot_20260827/`
- Generator/source commit: `212d42249b8fccb908d64d8e1d2f23930996005f`
- Manifest SHA-256: `3b15ed5412682fe6a233334caf202b271b6a8d89155bd817568d28a73f54a3e3`
- Baseline manifest SHA-256: `2a539abe122f8d4e06429d39db2dc37ec6f7569a8360959e64681d9f0ac1cf3e`
- Policy SHA-256: `2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28`
- Config SHA-256: `afa0d112b0bbed45492bbe931c61832fb9f23cdf3fac7feea668f59f9ccf62d8`
- 40 complete runs, 320,000 aligned samples, 1 kHz, drop 0

Dataset/NPZ는 Git ignored이고 기존 raw dataset을 수정하지 않았다.

## 7. Outcome parity with old Pilot

| Outcome | Existing Pilot | Sensor Pilot |
|---|---:|---:|
| BENIGN | 16 | 16 |
| SLIP | 8 | 8 |
| SINK | 9 | 9 |
| DUAL | 0 | 0 |
| INVALID | 7 | 7 |

Run ID, scenario, speed, patch position, Sink side/severity 순서도 기존 40-run matrix와 같다.

## 8. Raw FSR sanity

320,000 samples 전체가 finite/non-negative였고 non-contact foot의 nonzero count는 0, loaded foot의 non-positive count도 0이었다. Pre-censor touchdown load median은 `84.16 N`; unload 다음 sample 최대는 load-off threshold 아래인 `2.491 N`이었다. Bilateral-loaded 40,585 samples의 left/right mean은 `244.49/248.34 N`이다. Pre-censor quadrant load fractions는 `[0.0659, 0.0816, 0.1570, 0.1700, 0.0988, 0.1154, 0.1715, 0.1397]`로 모든 channel이 실제 load를 받았다.

Loaded-foot mean은 concrete `309.15 N`, marble `312.08 N`, uniform sand `270.40 N`, benign mild `285.47 N`, benign moderate `273.50 N`, hazardous Sink `288.89 N`이다. Uniform Sand와 hazardous Sink scale은 크게 겹치므로 fixed force scale만으로 분리되지 않는다. Hazardous Sink affected foot은 t1 직전/이후 100 ms mean이 pooled `111.17→365.35 N`으로 바뀌었으나 이는 touchdown/load redistribution과 결합되어 있고 benign soft touchdown도 유사할 수 있다. 실제 FP 결과가 이 shortcut risk를 확인한다.

Representative concrete gait, Sink t1/t2 alignment, uniform-sand comparison plot은 local artifact에 있다.

## 9. Early-target label definition

Raw `hazard_class_id`와 `training_eligible`은 변경하지 않았다. Experiment-local derived target만 다음처럼 만들었다.

- Later frozen hazard qualification을 만족한 SINK run: `[t1 physical Sink, t3)`를 SINK. `[t1,t2)` 포함.
- SLIP run: frozen established ANY-SLIP `[t1,t3)`만 SLIP. `[t0,t1)` backfill 없음.
- Hazard-positive stable prefix: run start부터 t0 전까지 NORMAL. Ambiguous `[t0,t1)` 제외.
- BENIGN: uniform sand 및 mild/moderate Sink transition을 포함해 valid pre-censor 전체 NORMAL.
- INVALID 7 runs 제외.

SINK qualification은 offline ground truth의 retrospective episode 판정이다. Runtime model input/window에는 future t2나 diagnostic이 들어가지 않는다.

## 10. Fixed split

First classification PoC의 frozen split SHA-256 `3b1b29a5e009783da2db0d1bdd198df24695d44c4b0cc55228bf28dfefda2a75`와 정확히 같은 run IDs를 사용했다.

| Split | Runs | BENIGN | SLIP | SINK |
|---|---:|---:|---:|---:|
| Train | 20 | 10 | 5 | 5 |
| Validation | 6 | 3 | 1 | 2 |
| Pilot holdout | 7 | 3 | 2 | 2 |

Pairwise overlap은 0이다. Pilot holdout은 이미 개발 과정에서 공개되었으므로 sealed final test로 주장하지 않는다.

## 11. IMU6 / FSR8 / Fusion14 model definitions

세 profile 모두 causal raw 100 ms window를 flatten하고 `64→32→3` ReLU MLP에 입력했다. Training stride 10 ms, train per-run/class cap 200, inverse window-frequency class weight, Adam `lr=0.001`, batch 128, max 40 epochs, validation macro-F1 patience 6, seeds `20260827/28/29`를 동일 적용했다. Normalization은 profile별 train runs만 사용한 per-channel z-score다.

| Profile | Input | Parameters |
|---|---:|---:|
| IMU6 | 100×6 | 40,643 |
| FSR8 | 100×8 | 53,443 |
| Fusion14 | 100×14 | 91,843 |

No ratios, CoP, total-force feature, derivative, moving variance 또는 profile-specific tuning을 사용하지 않았다. 50 ms secondary sweep은 실행하지 않았다.

## 12. Classification metrics

아래는 early-target 100 ms window의 3-seed mean±std다. Recall은 Pilot holdout 기준이다.

| Profile | Validation macro F1 | Holdout macro F1 | NORMAL recall | SLIP recall | SINK recall | Holdout run-balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| IMU6 | 0.8879±0.0078 | 0.8739±0.0047 | 0.9784 | 0.8008 | 0.7853 | 0.9279 |
| FSR8 | 0.8900±0.0069 | 0.8496±0.0105 | 0.9678 | 0.7899 | 0.7394 | 0.9165 |
| Fusion14 | 0.9243±0.0061 | 0.9050±0.0077 | 0.9850 | 0.8876 | 0.7994 | 0.9470 |

Holdout confusion matrices `[actual NORMAL, SLIP, SINK] × [pred NORMAL, SLIP, SINK]`:

- IMU6: seed27 `[[3046,21,47],[19,274,45],[96,9,440]]`; seed28 `[[3054,25,35],[23,272,43],[120,10,415]]`; seed29 `[[3040,14,60],[30,266,42],[107,9,429]]`
- FSR8: seed27 `[[3045,1,68],[36,258,44],[126,12,407]]`; seed28 `[[3021,5,88],[18,276,44],[114,27,404]]`; seed29 `[[2975,13,126],[19,267,52],[130,17,398]]`
- Fusion14: seed27 `[[3067,14,33],[5,311,22],[97,14,434]]`; seed28 `[[3051,13,50],[7,291,40],[97,10,438]]`; seed29 `[[3084,3,27],[7,298,33],[96,14,435]]`

이는 historical established-state score를 대체하지 않는 별도 early-target comparison이다.

## 13. SINK Recall@20/30/50/100 ms

Full 9 SINK events, 1 ms causal stride, argmax, 10 consecutive ms confirmation의 3-seed event recall이다.

| Profile | @20 ms | @30 ms | @50 ms | @100 ms |
|---|---:|---:|---:|---:|
| IMU6 | 0.0000±0.0000 | 0.0000±0.0000 | 0.1111±0.0000 | 0.1111±0.0000 |
| FSR8 | 0.1111±0.0000 | 0.1111±0.0000 | 0.1111±0.0000 | 0.4074±0.2283 |
| Fusion14 | 0.1111±0.0000 | 0.1111±0.0000 | 0.1111±0.0000 | 0.4074±0.1048 |

FSR8 seed별 @100은 `0.6667/0.4444/0.1111`, Fusion14는 `0.3333/0.5556/0.3333`이다. FSR signal은 평균 개선을 보였지만 FSR-only seed variance가 크다.

## 14. Pre-t2 latency and margin

Positive-margin 7 SINK runs는 각 profile/seed 모두 결국 t2 전에 sustained SINK가 되어 `7/7`이었다. Pooled 21 run-seed events의 결과는 다음과 같다.

| Profile | Pre-t2 | Median latency from t1 | Median margin before t2 | Seed median latency | Seed median t2 margin |
|---|---:|---:|---:|---|---|
| IMU6 | 21/21 | 234 ms | 338 ms | 155/265/152 ms | 338/298/342 ms |
| FSR8 | 21/21 | 102 ms | 390 ms | 99/102/109 ms | 390/391/384 ms |
| Fusion14 | 21/21 | 101 ms | 390 ms | 101/98/103 ms | 368/396/377 ms |

Zero-margin `t1==t2` 2 runs은 pre-t2 question에서 분리했다. FSR/Fusion은 positive-margin detection을 약 132~133 ms 앞당겼지만 50 ms recall 개선은 없었다.

## 15. SLIP Recall@20/30/50/100 ms

| Profile | @20 ms | @30 ms | @50 ms | @100 ms |
|---|---:|---:|---:|---:|
| IMU6 | 0.6250±0.0000 | 0.6250±0.0000 | 0.6250±0.0000 | 0.7083±0.0589 |
| FSR8 | 0.6667±0.0589 | 0.7917±0.0589 | 0.7917±0.0589 | 1.0000±0.0000 |
| Fusion14 | 0.5833±0.0589 | 0.5833±0.0589 | 0.7500±0.1768 | 0.9167±0.0589 |

FSR 추가는 이 Pilot에서 SLIP@100을 훼손하지 않았다.

## 16. BENIGN soft-ground false positives

수치는 10 ms sustained any-hazard FP run count의 3-seed mean±std다.

| Profile | Total /16 | Concrete /4 | Marble /4 | Uniform Sand /4 | Mild Sink /2 | Moderate Sink /2 |
|---|---:|---:|---:|---:|---:|---:|
| IMU6 | 7.33±0.94 | 0.00 | 0.00 | 3.33±0.94 | 2.00 | 2.00 |
| FSR8 | 9.33±1.25 | 0.33 | 1.33 | 3.67±0.47 | 2.00 | 2.00 |
| Fusion14 | 5.67±1.25 | 0.00 | 0.00 | 1.67±1.25 | 2.00 | 2.00 |

Fusion은 total과 Uniform Sand FP를 줄였지만 mild/moderate transition을 분리하지 못했다. FSR8-only는 오히려 FP가 증가해 `soft load pattern = SINK` shortcut risk를 보였다.

## 17. Seed stability

Holdout macro-F1 mean/std와 worst seed는 IMU6 `0.8739±0.0047` / seed 20260828 `0.8703`, FSR8 `0.8496±0.0105` / seed 20260829 `0.8350`, Fusion14 `0.9050±0.0077` / seed 20260828 `0.8941`이다. SINK@100 worst는 IMU6/FSR8 `0.1111`, Fusion14 `0.3333`; Fusion worst early recall도 IMU보다 높다. FSR8-only의 SINK@100 std `0.2283`은 안정적인 단독 채택 근거가 부족함을 뜻한다.

## 18. Per-run failure

IMU6@100은 `sink_right_severe_s015_p030` 한 run만 세 seed 모두 검출했다. FSR8은 그 run 외에도 left 0.20/0.25 m/s와 right 0.20/0.25 m/s run에서 적어도 한 seed가 100 ms 내 검출해 개선이 한 side 또는 한두 run에만 집중되지는 않았다. Fusion14는 left 0.15/0.20/0.25와 right 0.15 run에 개선을 분산시켰다.

그러나 `sink_left_severe_s015_p030`은 FSR8 latency 147~169 ms, Fusion은 두 seed miss였고, several right-side runs도 Fusion latency 101~162 ms로 100 ms boundary를 자주 넘었다. Per-run/side coverage는 아직 작다.

## 19. Sensor ablation conclusion

`FSR adds observability: YES`, 단 idealized Pilot 범위다. FSR8/Fusion14는 IMU6 대비 SINK@100 평균을 `0.1111→0.4074`, pooled pre-t2 median latency를 `234→102/101 ms`로 개선했고 양 side 여러 run에서 효과가 나타났으며 SLIP@100도 악화시키지 않았다.

`Fusion benefit: YES`. Fusion은 FSR8과 같은 mean SINK@100을 더 낮은 seed variance로 달성하고, holdout macro-F1 `0.9050` 및 total/Uniform Sand FP에서 FSR8과 IMU6보다 나았다. 다만 early 20~50 ms recall과 benign mild/moderate Sink separation은 해결하지 못했다.

`Sensor architecture freeze: NOT YET`.

## 20. Sim-to-real FSR limitations

Virtual FSR은 exact MuJoCo normal contact load의 noiseless sum이다. 실제 FSR의 nonlinear resistance-to-force curve, hysteresis, drift, temperature, saturation, sensor-to-sensor spread, mounting/preload, sole compliance, shear sensitivity, ADC resolution, sample jitter와 failure mode를 포함하지 않는다. MuJoCo point/sphere contact와 current compliance terrain도 실제 granular/deformable ground가 아니다. 따라서 hardware accuracy나 calibration 요구량을 이 결과에서 추정할 수 없다.

## 21. Next recommendation

다음 단계는 Full Dataset 자동 생성이 아니라 sensor architecture decision checkpoint다. Small hardware-characterization 또는 calibrated sensor-noise/mounting variation study로 Fusion의 SINK@100 gain과 soft-transition FP가 유지되는지 검증하고, left/right/phase coverage를 늘린 독립 Pilot에서 재현해야 한다. Mild/moderate false firing을 줄이는 label/scenario observability도 함께 review한다. 그 뒤에만 IMU6/FSR8/Fusion14 중 final input을 고정하고 Full Dataset을 시작한다.
