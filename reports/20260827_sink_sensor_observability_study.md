# Sink Sensor Observability Study

## 1. Purpose

`SINK_SENSOR_OBSERVABILITY_STUDY`는 frozen deformable-support physical Sink onset `s1`을 runtime-facing `IMU6`, `FSR8`, `Fusion14`만으로 causal하게 관측할 수 있는지 평가한다. 이는 Sink-focused simulation observability study이며 final detector, Full Hazard Dataset, sensor architecture freeze 또는 E84 deployment가 아니다.

작업 시작 HEAD는 `fcad8f2eafd84d74d57c5ad87bd7479c97026b65`다. Dataset generator/source commit은 `9e021dc43a780612b6f3685a67fa22a7f859fe1a`이며, experiment config는 [`20260827_sink_sensor_observability_study.yaml`](../configs/experiment/20260827_sink_sensor_observability_study.yaml)이다.

## 2. Frozen physical Sink ground truth

Ground truth는 다음과 같이 결과를 보기 전에 고정했다.

```text
support_surface_spread_m = max(cell displacement) - min(cell displacement)
support_surface_spread_m >= 0.010 m for 20 consecutive 1 kHz samples
s1 = first sustained active sample
```

`d0`는 같은 foot/contact episode에서 deformable support를 처음 physical contact한 sample이다. Threshold, persistence, severity/speed/side별 기준, FSR/IMU/fall/old-t2 relabel은 사용하지 않았다. Frozen Slip 50 mm/3 ms와 historical outcome-based Sink diagnostics도 변경하지 않았다.

## 3. Dataset design

126개의 deterministic physical conditions를 simulation 전에 고정했다. 같은 condition을 seed만 바꾼 반복은 없고 `random_seed=null`이다.

| Group | Declared | SINK | BENIGN | INVALID |
|---|---:|---:|---:|---:|
| Rigid concrete/marble/uniform sand | 18 | 0 | 18 | 0 |
| Balanced deformable hard negative | 42 | 0 | 34 | 8 |
| Moderate uneven primary | 42 | 30 | 7 | 5 |
| Mild/severe uneven boundary | 24 | 20 | 4 | 0 |
| Total | 126 | 50 | 63 | 13 |

Frozen travel은 reference/mild/moderate/severe `4/20/40/65 mm`, stiffness는 `50000/12000/7000/4500 N/m`, damping은 `1000/490/374/300 N·s/m`다. Outcome에 따른 mechanics retuning은 없었다.

## 4. Split design

Run membership, speed와 patch-position 후보를 simulation 전에 확정했다. 모든 condition signature는 terrain, topology, severity, side, support pattern, speed, patch start와 patch width를 포함하며 duplicate는 0이다.

| Split | Declared | Valid | SINK | BENIGN | INVALID | Speeds (m/s) | Patch starts (m) |
|---|---:|---:|---:|---:|---:|---|---|
| Train | 76 | 71 | 31 | 40 | 5 | 0.12, 0.18, 0.24 | 0.30, 0.38 |
| Validation | 25 | 21 | 11 | 10 | 4 | 0.15, 0.27 | 0.34, 0.42 |
| Holdout | 25 | 21 | 8 | 13 | 4 | 0.21, 0.30 | 0.32, 0.40 |

Invalid 13개는 모두 `nonfoot_surface_contact` censor로 meaningful evaluation이 부족한 run이다. Outcome을 본 뒤 split을 이동하지 않았다. Holdout은 integrity와 counts만 확인하고 validation selection 전 waveform access를 막았으며, 선택 뒤 한 번 열었다.

## 5. Sensor contract

모든 run은 정렬된 `pelvis_imu [N,6] float32`와 `foot_fsr [N,8] float32`를 1 kHz로 저장한다. Profile은 `imu6`, `fsr8`, `fusion14=[pelvis_imu, foot_fsr]`다. Model tensor에는 support displacement/spread/velocity, contact, d0/s1, terrain/scenario, side/pattern, run ID, fall/censor와 historical t2가 없다. Sequence/timestamp는 alignment 전용이다.

## 6. Dataset coverage

Dataset은 `data/raw/sink_observability_20260827/`에 one-NPZ-per-run으로 저장했다. 총 1,008,000 sensor samples, 126 NPZ, 110,534,019 bytes(약 105.4 MiB)다. Manifest SHA-256은 `cbff615e3183024fc805b1cd3cae38c7d078d704ce8aff922700373c35f2b518`다.

전체 declared coverage는 concrete 6, marble 6, sand 114다. Severity는 none 18, mild 31, moderate 54, severe 23이고 side는 left 56, right 52, none 18이다. Support pattern은 balanced-soft 18, balanced-deformable 42, medial 32, lateral 18, localized 16이다. 모든 run은 8,000 samples이며 drop은 0이다.

## 7. Observed outcomes

Observed outcome은 `SINK 50 / BENIGN 63 / INVALID 13`이다. SINK side는 left 28/right 22, pattern은 medial 27/lateral 11/localized 12, severity는 mild 11/moderate 30/severe 9다. Readiness 기준 SINK ≥30, BENIGN ≥40, side별 ≥10, pattern별 ≥8을 모두 통과했다. Scenario가 uneven이더라도 no-s1 11개는 BENIGN이고, balanced plate에서 SINK로 강제된 run은 0이다.

## 8. Raw sensor sanity

ML 전에 train+validation SINK 42개만 사용해 s1-relative point statistics를 확인했다. Holdout waveform은 사용하지 않았다.

| s1 offset (ms) | Bilateral FSR load mean ± std (N) | IMU accel-z mean ± std (m/s²) |
|---:|---:|---:|
| -100 | 320.1 ± 58.3 | 9.018 ± 1.706 |
| -50 | 371.2 ± 54.0 | 10.105 ± 1.548 |
| -20 | 320.5 ± 80.2 | 8.589 ± 2.596 |
| 0 | 345.0 ± 59.9 | 9.413 ± 1.881 |
| +20 | 379.2 ± 130.9 | 10.527 ± 3.689 |
| +50 | 378.6 ± 63.9 | 10.864 ± 1.550 |
| +100 | 356.1 ± 48.2 | 10.551 ± 1.566 |

Both sensors change around s1, but the trajectory is not monotonic and between-run variation is large. Entropy, CoP optimization, FFT 또는 feature sweep은 수행하지 않았다.

## 9. Training protocol

Primary window는 100 ms, construction stride는 10 ms이며 class transition/excluded/censor를 가로지르는 window는 제외했다. Normalization은 각 profile의 train valid runs에서만 fit했다. Existing MLP와 unidirectional GRU를 같은 hidden size, Adam `1e-3`, batch 128, 최대 40 epochs, patience 6, inverse-frequency class weighting으로 학습했다. Seeds는 `20260827/20260828/20260829`다. Training target은 v3 raw `NORMAL=0`, `SINK=2`를 binary `NORMAL/SINK`로 매핑했다.

## 10. Sensor ablation

각 sensor profile에서 validation run-balanced macro F1가 가장 높은 family는 다음과 같다.

| Sensor | Best family | Validation run-balanced macro F1 | NORMAL recall | SINK recall |
|---|---|---:|---:|---:|
| IMU6 | MLP | 0.6986 | 0.9581 | 0.6944 |
| FSR8 | MLP | 0.6688 | 0.9330 | 0.7387 |
| Fusion14 | MLP | 0.6971 | 0.9717 | 0.6740 |

Fusion14는 IMU6 대비 primary run-balanced score 우위를 보이지 않았고 FSR8 단독도 낮았다. 이 Sink-only result로 final sensor architecture를 freeze하지 않는다.

## 11. Model comparison

| Profile | Family | Validation macro F1 | Run-balanced macro F1 | NORMAL recall | SINK recall | Run-balanced accuracy |
|---|---|---:|---:|---:|---:|---:|
| IMU6 | MLP | 0.7981 | 0.6986 | 0.9581 | 0.6944 | 0.9330 |
| IMU6 | GRU | 0.7793 | 0.6985 | 0.9385 | 0.7502 | 0.9224 |
| FSR8 | MLP | 0.7667 | 0.6688 | 0.9330 | 0.7387 | 0.9128 |
| FSR8 | GRU | 0.7263 | 0.6393 | 0.9046 | 0.7458 | 0.8888 |
| Fusion14 | MLP | 0.8201 | 0.6971 | 0.9717 | 0.6740 | 0.9434 |
| Fusion14 | GRU | 0.7783 | 0.6846 | 0.9306 | 0.7927 | 0.9164 |

GRU가 모든 sensor에서 score를 개선하지는 않았다. Fusion14 GRU는 가장 높은 SINK recall을 보였지만 primary run-balanced macro F1가 selection near-tie 범위 밖이었다.

## 12. Validation selection

Primary best는 IMU6 MLP 0.698624였지만 predeclared near-tie tolerance 0.005 안의 candidates에는 secondary minimum-class recall을 적용했다. IMU6 GRU는 run-balanced macro F1 0.698480, minimum class recall 0.7502로 선택됐다. 선택 protocol은 `imu6_gru_100ms` 3-seed logit ensemble이며 holdout 결과로 변경하지 않았다.

## 13. Holdout one-shot result

선택 뒤 holdout을 한 번 평가했다.

| Metric | Value | Gate |
|---|---:|---:|
| Accuracy | 0.9476 | diagnostic |
| Macro F1 | 0.7307 | ≥0.80, FAIL |
| NORMAL recall | 0.9529 | ≥0.90, PASS |
| SINK recall | 0.7862 | ≥0.80, FAIL |
| Run-balanced accuracy | 0.9398 | diagnostic |
| Run-balanced macro F1 | 0.6564 | diagnostic |
| Run-balanced macro recall | 0.8859 | diagnostic |

## 14. Established-state confusion

Rows are truth and columns are prediction.

|  | Pred NORMAL | Pred SINK |
|---|---:|---:|
| True NORMAL | 12,996 | 642 |
| True SINK | 96 | 353 |

SINK precision은 0.3548, SINK F1은 0.4889다. High overall accuracy는 NORMAL windows가 많기 때문이며 acceptance 판단을 대신하지 않는다.

## 15. Causal 1 ms replay

선택된 ensemble을 모든 valid complete run에서 1 ms stride로 replay했다. 각 inference는 endpoint까지의 과거 100 ms만 사용하고, 10 consecutive argmax SINK endpoints가 끝나는 시점을 model onset으로 정의했다. Holdout SINK 8/8은 censor 전에 sustained detection됐다. Positive run 중 pre-d0 sustained prediction은 3개였으며 causal false-positive evidence다.

## 16. s1-relative latency

Holdout latency는 `[-463, 72, 72, 77, 79, 80, 91, 92] ms`다. Median은 +78.0 ms, p95는 +91.65 ms로 latency ≤+100 ms gate를 통과했다. 한 mild-left run의 -463 ms onset은 d0 뒤 같은 episode에서 후속 s1보다 빨랐지만, balanced-control FP가 높아 credible precursor로 승인하지 않는다. `[-100,s1)` precursor는 0개다.

## 17. Recall@20/50/100

| Horizon | Holdout detected/events | Recall |
|---|---:|---:|
| By s1 | 1/8 | 0.125 |
| s1 +20 ms | 1/8 | 0.125 |
| s1 +50 ms | 1/8 | 0.125 |
| s1 +100 ms | 8/8 | 1.000 |

Recall@+100 ms gate는 통과했지만 separation이 대부분 +50 ms 이후에 나타났다. Validation Recall@+100 ms는 8/11=0.7273이어서 split variation도 크다.

## 18. Benign false positives

Holdout benign에서 10 ms sustained SINK FP를 run 단위로 집계했다. Duration은 SINK로 예측된 1 ms endpoints의 합이다.

| Group | FP runs / benign runs | Run rate | SINK duration (ms) | Pre-event FP runs |
|---|---:|---:|---:|---:|
| All benign | 9/13 | 0.692 | 5,533 | 6 |
| Concrete | 0/2 | 0.000 | 0 | 0 |
| Marble | 0/2 | 0.000 | 4 | 0 |
| Uniform sand | 2/2 | 1.000 | 355 | 2 |
| Balanced mild | 2/2 | 1.000 | 1,229 | 1 |
| Balanced moderate | 2/2 | 1.000 | 2,207 | 1 |
| Balanced severe | 2/2 | 1.000 | 1,515 | 1 |
| Uneven/no-s1 | 1/1 | 1.000 | 223 | 1 |

Overall benign FP ≤0.15와 balanced FP ≤0.15 gates를 모두 크게 실패했다. 특히 balanced severe 2/2 실패는 model이 level soft-ground motion을 uneven Sink와 충분히 구분하지 못했음을 보인다.

## 19. Severity/side/pattern breakdown

Established holdout SINK recall은 mild 0.6512, moderate 0.8464, severe 0.7000이다. Side는 left 0.7993/right 0.7586이다. Pattern은 medial 0.7137, lateral 0.8485, localized 0.9516이다. Run-balanced recalls도 각각 mild/moderate/severe 0.7624/0.8715/0.7000, left/right 0.8309/0.8093, medial/lateral/localized 0.7656/0.8485/0.9529다. Holdout positive speed는 0.21 m/s 한 stratum이므로 speed generalization은 판정할 수 없다.

Causal Recall@+100 ms는 both sides와 all three patterns에서 1.0이었다. 다만 holdout lateral은 positive 1 run뿐이므로 no-catastrophic-pattern gate 통과를 넓은 generalization evidence로 해석하지 않는다.

## 20. Fall/non-fall independence

Physical SINK 50개 중 censor/fall diagnostic이 있는 run은 7개, non-fall은 43개다. Mechanical recovery evidence는 25개, no-recovery는 25개다. 모두 같은 SINK target으로 처리했으며 s1 truth를 fall, recovery 또는 historical t2로 바꾸지 않았다. 따라서 이 study target은 fall prediction이 아니다.

## 21. Leakage/circularity audit

Leakage audit는 PASS다.

- Model input은 selected IMU6 또는 candidate raw IMU/FSR profile뿐이다.
- Support displacement/spread/velocity, exact contact, d0/s1, terrain/scenario, side/pattern, run ID, fall/censor와 t2는 input에 없다.
- Normalization fit run은 train valid runs뿐이다.
- Train/validation/holdout run overlap과 duplicate physical signature는 0이다.
- Causal windows는 endpoint 이후 sample을 사용하지 않으며 replay stride는 정확히 1 ms다.
- Holdout waveform은 validation selection 전에 열지 않았고 선택 후 한 번 평가했다.

FSR은 MuJoCo contact physics에서, Sink oracle은 support-body displacement에서 생성된다. 수치 변수를 직접 공유하지 않더라도 같은 simulation dynamics에서 상호작용하므로 이는 simulation observability evidence이지 real-sensor independent proof가 아니다.

## 22. Known limitations

- Deterministic controller와 engineering deformable-support proxy 한 종류에 한정된다.
- Holdout positive는 8 runs이고 lateral 1 run, severe 1 run이며 positive speed는 0.21 m/s뿐이다.
- Virtual FSR에 hardware noise, hysteresis, saturation, calibration과 mounting error가 없다.
- Near-overlapping windows는 독립 sample이 아니며 primary statistical unit은 run이다.
- Selected ensemble의 balanced/uniform-soft FP가 매우 높고 established SINK precision이 낮다.
- Hyperparameter/architecture/threshold search를 하지 않았으므로 최종 detector performance가 아니다.

## 23. Verdict

`SINK_SENSOR_OBSERVABILITY_PROMISING`

The selected runtime sensor protocol shows clear s1-related causal separation by +100 ms, but primary support is not established. Macro F1, SINK recall, total benign FP와 balanced deformable FP gates를 실패했다. Threshold/model retuning 또는 holdout 재평가로 verdict를 올리지 않는다. `FINAL_SENSOR_ARCHITECTURE_FROZEN`, Full Hazard readiness, real sensor validity 또는 E84 readiness를 선언하지 않는다.

## 24. Next recommendation

이 dataset과 report를 bounded evidence로 보존하고, balanced-soft shortcut과 split coverage를 review한 뒤 별도 승인 milestone에서만 다음 설계를 결정한다. Joint-state augmentation이나 새 runtime signal은 현재 결과만으로 자동 추가하지 않는다. SLIP+SINK Full Hazard Dataset, final sensor architecture freeze와 E84 deployment는 자동으로 시작하지 않는다.
