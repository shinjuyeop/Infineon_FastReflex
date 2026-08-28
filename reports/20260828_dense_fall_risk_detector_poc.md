# Dense Fall-Risk Dataset and Detector PoC

## 1. Why previous sparse audit failed

이전 `TEMPORAL_STABILITY_SEPARABILITY_AUDIT`은 84 simulations 중 train stable/fall `22/24`, validation `8/8`을 사용했고, 각 fixed offset에서 run당 window 하나와 offset별 별도 GRU를 학습했다. Privileged best AUROC `.875`, IMU6 `.844`였지만 이는 fall-risk signal 부재가 아니라 sparse protocol에서 reliable horizon을 입증하지 못한 결과로 보존한다. Historical MoS, single-state Mahalanobis와 sparse temporal report는 변경하지 않았다.

이번 작업 시작 시 `HEAD == origin/main == 2f90db2ef6be56bc4849dacde34812c25e6783fa`였고 worktree는 clean이었다. Dataset matrix, split, horizons, histories, dense sampling, GRU, seeds, threshold calibration과 gates는 simulation 전에 [`20260828_dense_fall_risk_detector_poc.yaml`](../configs/experiment/20260828_dense_fall_risk_detector_poc.yaml)에 동결했다. Config SHA-256은 `1b53009b865f9a79b9037ad96fe3107906e7a0320a995f6438501ced2833d7ae`다.

## 2. Dense supervised formulation

한 horizon `H`마다 하나의 GRU가 target contact 이후 전체 progression을 학습한다. Endpoint `t`의 label은 actual fall이 있을 때 `0 < t_fall - t <= H`이면 `FALL_RISK`, 그보다 이르면 `STABLE`이다. Observed stable run의 모든 valid pre-censor endpoint는 `STABLE`이다.

`t_instability`는 정의하거나 사용하지 않았다. Fall outcome/time은 label과 evaluation alignment에만 사용했고 model tensor에는 terrain, scenario role, patch, Slip/Sink, fall/time-to-fall, future/post-fall sample이 없다.

## 3. Dataset design

Dataset ID는 `fall_risk_dense_20260828`, local path는 Gitignored `data/raw/fall_risk_dense_20260828/`다. Eight source×target×design-role strata 각각 30개, primary transition 240개를 simulation 전에 만들었다. Index `01–18/19–24/25–30`은 각 stratum의 train/validation/holdout membership으로 고정했다. Secondary hard-ground controls는 Concrete 8 + Marble 8이다.

각 run은 one NPZ이며 timestamp, Pelvis IMU6, 40-D privileged full-state, gait phase, target-contact/fall/censor clock을 저장한다. Dataset 전체는 256 NPZ, `336,916,055 bytes`(약 323 MiB)다. Manifest SHA-256은 `6c452caebdbc1a6f5b2feb639d15aa3627c1c3e4e6e343dbf9adff3b60b39207`이다. Raw NPZ, manifest, checkpoints와 metrics JSON은 commit하지 않는다.

## 4. Speed-confound removal

Calibration의 individually observed 0.25 m/s Ice stable points `0.36/(0.70–0.75)`와 Ice fall points `0.33/0.70`, `0.34/0.75`, 그리고 Sand stable/fall 0.25 m/s domains를 근거로 primary 240개 모두를 정확히 `0.25 m/s`로 고정했다. 이전 sparse audit의 Ice stable/fall `0.25/0.15 m/s` confound는 제거됐다.

새 widths는 verified anchor의 local domain 안에서 deterministic하게 배치했고 기존 calibration/MoS/full-state/sparse-audit condition과 exact physical signature overlap은 0이다. Ice friction, Sand travel/stiffness/damping, controller와 policy는 변경하지 않았다.

## 5. Observed outcome coverage

Intended role을 label로 사용하지 않고 actual outcome만 집계했다.

| Cohort | Stable | Fall | Total |
|---|---:|---:|---:|
| Primary | 127 | 113 | 240 |
| Ice | 60 | 60 | 120 |
| Sand | 67 | 53 | 120 |
| Concrete-origin | 64 | 56 | 120 |
| Marble-origin | 63 | 57 | 120 |
| Hard controls | 16 | 0 | 16 |

Sand fall-design 7개가 actual stable로 flip했지만 run을 추가/제거하거나 split을 재배치하지 않았다. Valid primary `240/240`, invalid/nonfinite/pre-transition fall `0`, duplicate signature `0`, split overlap `0`이었다. 모든 readiness gate가 PASS했다.

## 6. Split

| Split | Designed | Observed stable | Observed fall | Access |
|---|---:|---:|---:|---|
| Train | 144 | 74 | 70 | normalization + fit |
| Validation | 48 | 27 | 21 | epoch/threshold/model selection |
| Holdout | 48 | 26 | 22 | sealed; not opened |

Validation hard controls는 8, sealed holdout hard controls는 8이다. Holdout metadata의 integrity/count/outcome만 readiness에서 확인했고 waveform은 validation selection 전후 모두 열지 않았다(`guard_open_count=0`).

## 7. Dense window contract

Training endpoint stride는 10 ms다. Fall run은 `[fall-H, fall)`의 positive 최대 `ceil(H/10)`개와 `fall-H-20 ms`보다 이른 deterministic early negative 동수를 제공한다. Stable run은 matched fall positive의 exact elapsed-since-contact endpoint를 최대 positive count만큼 제공한다.

| H | Train windows | Validation windows | Train independent runs | Validation independent runs |
|---:|---:|---:|---:|---:|
| 200 ms | 4,200 | 1,260 | 140 | 42 |
| 100 ms | 2,100 | 630 | 140 | 42 |
| 50 ms | 1,050 | 315 | 140 | 42 |

Window 수는 correlated samples이며 independent evidence로 주장하지 않는다. Run-level replay acceptance는 validation primary 48개 전체와 controls 8개를 사용한다.

## 8. Label semantics

Horizon boundary는 inclusive/exclusive `[t_fall-H, t_fall)`다. Positive input의 마지막 sample은 항상 `t_fall` 이전이며 post-fall sample은 없다. Fall run early negative에는 frozen 20 ms safety gap이 있고, stable matched endpoints를 임의로 앞당기지 않는다. Causality regression은 PASS다.

## 9. Privileged representation

`PRIVILEGED_FULL_STATE`는 이전 temporal audit의 40-D schema를 그대로 재사용한다: pelvis roll/pitch, angular/linear velocity, pelvis height, 12 lower-body q/dq pairs, COM velocity/height, exact support-phase one-hot이다. Absolute yaw, MoS, terrain과 fall diagnostic은 제외했다. GRU parameter count는 `7,170`; q/dq는 upper-bound privilege이며 runtime sensor 채택이 아니다.

## 10. Pelvis IMU6

`PELVIS_IMU6`는 1 kHz raw `accel_x/y/z + gyro_x/y/z` 6 channels만 사용한다. GRU parameter count는 `3,906`이다. Representation별 train-only mean/std는 144 train runs의 capped `272,844` samples로 fit했고 validation/holdout waveform은 normalization에 쓰지 않았다.

## 11. Training protocol

두 representation에 동일한 existing one-layer unidirectional GRU(hidden 32), Adam `1e-3`, batch 128, max 40 epochs, patience 6과 seeds `20260828/20260829/20260830`을 사용했다. Three-seed mean probability를 `P(FALL_RISK within H)`로 사용했다.

Canonical trainer 기본 macro-F1 selection은 유지하고 이 experiment에서만 validation cross-entropy minimum으로 best epoch를 선택했다. Best epochs는 privileged `4–17`, IMU6 `13–31` 범위였다. CNN/LSTM/Transformer/larger GRU와 hyperparameter sweep은 수행하지 않았다.

## 12. Threshold calibration

각 representation×H×history candidate의 validation replay를 1 ms causal stride로 실행했다. Threshold grid는 `.10–.90`, step `.02`, persistence는 10 consecutive samples이며 detection clock은 10번째 confirmation endpoint다.

Threshold가 먼저 stable transition FP ≤.15, hard-control FP ≤.10, fall-run premature FP ≤.15를 모두 만족해야 했다. Twelve candidates 모두 feasible threshold count가 `0`이었다. 아래 표의 threshold/metrics는 selection이 아니라 frozen diagnostic-best operating point다; threshold를 retune하지 않았다.

## 13. H=200 result

| Representation | History | Diagnostic threshold | Window AUROC | Fall recall | Stable spec. | Ice | Sand | Premature FP | Hard FP | Median lead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Privileged | 50 | .90 | .9720 | .429 | .481 | .500 | .333 | .571 | 1.000 | 183 ms |
| Privileged | 100 | .72 | .9745 | .286 | .000 | .250 | .333 | .714 | 1.000 | 196.5 ms |
| IMU6 | 50 | .86 | .9650 | .476 | .222 | .500 | .444 | .524 | .000 | 114 ms |
| IMU6 | 100 | .88 | .9582 | .619 | .222 | .500 | .778 | .381 | .000 | 114 ms |

Window ranking은 강하지만 full-run replay에서 recoverable/early trajectory가 sustained risk로 발화해 operational gates를 실패했다.

## 14. H=100 result

| Representation | History | Diagnostic threshold | Window AUROC | Fall recall | Stable spec. | Ice | Sand | Premature FP | Hard FP | Median lead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Privileged | 50 | .76 | .9836 | .143 | .222 | .250 | .000 | .857 | 1.000 | 100 ms |
| Privileged | 100 | .50 | .9866 | .143 | .000 | .250 | .000 | .857 | 1.000 | 100 ms |
| IMU6 | 50 | .90 | .9600 | .286 | .222 | .500 | .000 | .714 | 1.000 | 78 ms |
| IMU6 | 100 | .48 | .9637 | .143 | .000 | .250 | .000 | .857 | 1.000 | 100 ms |

## 15. H=50 result

| Representation | History | Diagnostic threshold | Window AUROC | Fall recall | Stable spec. | Ice | Sand | Premature FP | Hard FP | Median lead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Privileged | 50 | .70 | .9833 | .286 | .222 | .500 | .000 | .714 | 1.000 | 47.5 ms |
| Privileged | 100 | .68 | .9834 | .286 | .222 | .500 | .000 | .714 | .000 | 47.5 ms |
| IMU6 | 50 | .50 | .9221 | .286 | .000 | .500 | .000 | .714 | .000 | 46.5 ms |
| IMU6 | 100 | .90 | .9213 | .429 | .519 | .500 | .333 | .429 | .000 | 22 ms |

50 ms에서도 privileged/IMU6 모두 fall recall, stable specificity, Ice/Sand recall과 premature constraints를 함께 만족하지 못했다.

## 16. Selected horizon/history

| Representation | Selected H | Selected history | Selected threshold | Diagnostic best |
|---|---|---|---|---|
| Privileged | None | None | None | H=50, history=100, threshold=.68 |
| IMU6 | None | None | None | H=200, history=100, threshold=.88 |

Predeclared order `200→100→50 ms` 어디에서도 validation gate를 통과한 candidate가 없었다.

## 17. Privileged validation/holdout

Privileged diagnostic-best H50/history100은 validation 21 fall 중 valid 6, premature 15, missed 0이었고 stable specificity는 `6/27=.222`였다. Ice recall `.50`, Sand recall `0`, hard-control FP `0/8`, valid lead는 min/p10/median/p95/max `45/45/47.5/50/50 ms`였다.

Validation selection이 없으므로 privileged holdout waveform은 열지 않았고 metrics는 `N/A`다.

## 18. IMU validation/holdout

IMU6 diagnostic-best H200/history100은 validation 21 fall 중 valid 13, premature 8, missed 0이었고 stable specificity는 `6/27=.222`였다. Ice recall `.50`, Sand recall `.778`, hard-control FP `0/8`, valid lead는 `99/100.8/114/159/159 ms`였다.

IMU6도 validation selection이 없어 holdout waveform과 model evaluation을 열지 않았다.

## 19. Ice/Sand breakdown

Dataset Ice는 stable/fall `60/60`, Sand는 `67/53`으로 readiness를 통과했다. Diagnostic-best validation에서는 privileged Ice/Sand recall `.50/.00`, IMU6 `.50/.778`이었다. 두 candidate 모두 Ice stable 12/12를 false alert했고 Sand stable도 9/15를 false alert했다.

Sand signal은 IMU6 H200 ranking/recall에 일부 나타났지만 premature `.381`과 stable specificity `.222` 때문에 bounded operational signal로 채택할 수 없다.

## 20. Concrete/Marble robustness

Dataset Concrete-origin stable/fall은 `64/56`, Marble-origin `63/57`이다. Diagnostic-best valid detections는 privileged Concrete/Marble 각각 `3/3`, IMU6 `6/7`이었다. Source 수 자체는 균형이지만 overall false-alert semantics가 실패해 source robustness PASS로 세지 않는다.

## 21. Stable Ice Slip hard negatives

Ice stable `60/60`과 Ice fall `60/60` 모두 physical Slip diagnostic이 있었다. Validation diagnostic-best에서 privileged와 IMU6 모두 Ice stable `12/12`를 sustained FALL_RISK로 오인했다. Dense model이 Slip channel을 입력받지는 않았지만 recoverable Slip 이후 motion을 bounded fall risk와 분리하지 못했다.

## 22. Stable Sand hard negatives

Sand stable 67개는 actual deformation `20.139–40.172 mm`, 그중 outcome-flip 7개는 Sink diagnostic도 보였다. Sand fall 53개는 Sink `53/53`, deformation `40.084–40.236 mm`였다.

Validation diagnostic-best에서 privileged와 IMU6 모두 Sand stable `9/15`를 false alert했다. Stable deformation/recovery와 eventual fall progression이 current run-level operating point에서 충분히 분리되지 않았다.

## 23. Premature alerts

Horizon 이전 first sustained event는 eventual fall run이어도 premature false alarm으로 처리했다. Privileged diagnostic-best는 `15/21=.714`, IMU6는 `8/21=.381`이었다. Missed가 0이라는 사실은 성공이 아니다. 대부분의 fall run이 결국 alert됐지만 first causal alert가 horizon semantics보다 너무 일렀고 stable run에서도 같은 문제가 발생했다.

## 24. Prediction-lead distribution

Only valid in-horizon detections만 lead에 포함했다.

| Candidate | Min | P10 | Median | P95 | Max |
|---|---:|---:|---:|---:|---:|
| Privileged H50/history100 | 45 | 45 | 47.5 | 50 | 50 ms |
| IMU6 H200/history100 | 99 | 100.8 | 114 | 159 | 159 ms |

IMU6 median lead는 H의 50% gate를 넘지만 recall/specificity/premature gates를 실패한다. Lead 단일 metric으로 candidate를 구제하지 않았다.

## 25. Time-order sanity

Selection이 없으므로 representation별 diagnostic-best model을 동일 threshold에서 inference-only 비교했다.

| Representation | Mode | Fall recall | Stable spec. | Premature FP | Hard FP |
|---|---|---:|---:|---:|---:|
| Privileged | Original | .286 | .222 | .714 | .000 |
| Privileged | Reversed | .048 | .222 | .667 | .750 |
| Privileged | Last 20 ms only | .143 | .333 | .714 | .000 |
| IMU6 | Original | .619 | .222 | .381 | .000 |
| IMU6 | Reversed | .143 | .000 | .857 | .000 |
| IMU6 | Last 20 ms only | .333 | .778 | .286 | .000 |

두 representation 모두 `TEMPORAL_ORDER_AFFECTS_RUN_LEVEL_METRICS`다. GRU는 order를 전혀 무시하지는 않는다. Last-20-ms masking은 IMU6 specificity를 개선했지만 recall을 낮췄고 gate를 통과하지 못했으며 selection/retuning에 사용하지 않았다.

## 26. Historical comparison

| Study | Evidence | Result |
|---|---|---|
| Phase-aware MoS | scalar onset | FP 20.83%, fall coverage 18.75%; unsupported |
| Single-state full distance | instantaneous privileged state | FP 25%, coverage 56.25%, median lead 28 ms; not separable |
| Sparse temporal GRU | tens of run-windows per offset | privileged best AUROC .875, IMU6 .844; not supported |
| Dense temporal GRU | 240 fresh primary runs, progression windows | window AUROC high, but no false-alarm-feasible run-level threshold |

Dense data는 sparse sample-size limitation을 해소하고 ranking signal을 드러냈지만, sustained runtime detector operating point를 입증하지 못했다.

## 27. Fusion implication

Validated `FALL_RISK`가 없으므로 actual Terrain+Stability performance를 재평가하거나 runtime enum을 변경하지 않았다. Existing conceptual mapping ICE→SLIP_RISK, SAND→SINK_RISK, hard/unknown→GENERIC_INSTABILITY는 그대로이며 Fusion truth-table regression은 PASS다. Terrain FSR4+MLP+50 ms protected hashes는 전후 동일했고 재학습하지 않았다.

## 28. Limitations

- Evidence는 current deterministic G1 policy, MuJoCo engineering terrain과 current fall criterion에만 해당한다.
- 240 runs는 exact physical signatures가 독립이지만 stochastic resets가 아니라 verified anchors 주변의 correlated local conditions다.
- Window AUROC는 correlated dense windows의 secondary diagnostic이며 independent-run acceptance가 아니다.
- Sand fall-design 7개가 stable로 flip해 validation/holdout outcome balance가 designed matrix와 달라졌지만 membership은 보존했다.
- One fixed GRU architecture와 three seeds만 평가했다. Threshold grid 외 hyperparameter/model search는 하지 않았다.
- First sustained alert semantics는 recovery 후 재발화를 별도 valid event로 세지 않는다. 이는 predeclared bounded runtime contract다.
- Validation selection이 없어 holdout generalization evidence는 없다.
- Recovery controller가 변경되면 observed fall/no-fall relationship도 달라질 수 있다. Real-world universal fall prediction 결과가 아니다.

전체 repository regression은 `155 tests OK`, 기존 policy-dependent end-to-end smoke `1 skipped`로 통과했다. Terrain protected-file hash, Fusion truth table과 causal replay tests는 PASS다.

## 29. Verdict

`DENSE_FALL_RISK_NOT_SUPPORTED`

Dataset readiness는 PASS했지만 privileged full-state조차 50 ms horizon까지 validation gate를 통과하지 못했다. 높은 dense-window AUROC는 full-run replay의 stable false alerts와 premature fall alerts로 이어졌으며 feasible threshold가 없었다. 따라서 IMU6 detector, additional runtime sensor, production `FALL_RISK` state를 채택하지 않는다.

## 30. Next recommendation

다음 단계는 sensor augmentation이나 더 큰 model이 아니라 label/event formulation과 recoverable disturbance를 포함한 run-level decision semantics의 architecture review다. 특히 high window AUROC와 first-sustained replay failure의 원인을 분리해 검토해야 한다. 이 recommendation을 자동 시작하지 않는다.

이번 milestone에서 q/dq/FSR/Foot-IMU runtime augmentation, final sensor architecture freeze, Recovery, integrated production model과 E84를 수행하지 않았다.
