# First Pelvis IMU Classification PoC

## 1. 목적

이 실험은 1 kHz Pelvis IMU6의 50/100 ms causal raw sequence만으로 `NORMAL`, established `SLIP`, established hazardous `SINK` 상태가 구분 가능한지 확인하는 첫 Pilot PoC다. 최종 모델, onset 직후 검출, Full Dataset generalization 또는 deployment 가능성을 검증하는 실험이 아니다.

## 2. Dataset provenance와 integrity

- Dataset ID: `hazard_pilot_20260827`
- Local path: `data/raw/hazard_pilot_20260827`
- Schema: `hazard_dataset_contract_v1`
- Dataset source commit: `7e6fa1c168d8f9d419d49cf2fff96095e673761f`
- Manifest SHA-256: `2a539abe122f8d4e06429d39db2dc37ec6f7569a8360959e64681d9f0ac1cf3e`
- Policy SHA-256: `2a66ca6336eadb3c0b34b557763f3e06d01ff8fcf6260dd4cedbd69d6093fc28`

학습 전 canonical dataset validator로 metadata/manifest parse, manifest SHA, 40개 NPZ와 각 file SHA, unique run ID, contiguous sequence, 1,000 us timestamp delta, `[N,6]` `float32` IMU, finite/aligned arrays, policy SHA와 drop 0을 다시 확인했다. 결과는 40 runs, 320,000 samples, observed BENIGN/SLIP/SINK/DUAL/INVALID = 16/8/9/0/7이며 raw artifact는 수정하지 않았다.

## 3. Valid/invalid run

Observed outcome이 BENIGN/SLIP/SINK인 33개 run만 후보 pool에 포함했다. `INVALID` 7개는 raw artifact에는 보존하되 split, normalization, window, 학습과 평가에서 전부 제외했다.

## 4. Fixed run-disjoint split

Split은 waveform을 읽기 전에 manifest의 terrain, speed, patch, side와 first-contact metadata만 보고 고정했다. Train/validation/holdout 교집합은 모두 0이다.

| Split | BENIGN | SLIP | SINK | Total |
|---|---:|---:|---:|---:|
| Train | 10 | 5 | 5 | 20 |
| Validation | 3 | 1 | 2 | 6 |
| Holdout | 3 | 2 | 2 | 7 |

Train run IDs:

- `normal_concrete_s015_p000`, `normal_concrete_s020_p000`
- `normal_marble_s010_p000`, `normal_marble_s015_p000`, `normal_marble_s025_p000`
- `normal_sand_s015_p000`, `normal_sand_s020_p000`, `normal_sand_s025_p000`
- `normal_sink_right_mild_s015_p035`, `normal_sink_left_moderate_s015_p035`
- `slip_ice_s010_p030`, `slip_ice_s010_p035`, `slip_ice_s015_p030`, `slip_ice_s015_p035`, `slip_ice_s025_p035`
- `sink_left_severe_s015_p035`, `sink_left_severe_s025_p030`, `sink_left_severe_s025_p035`, `sink_right_severe_s015_p030`, `sink_right_severe_s020_p035`

Validation run IDs:

- `normal_concrete_s010_p000`, `normal_marble_s020_p000`, `normal_sink_left_mild_s015_p035`
- `slip_ice_s020_p035`
- `sink_left_severe_s015_p030`, `sink_right_severe_s015_p035`

Holdout run IDs:

- `normal_concrete_s025_p000`, `normal_sand_s010_p000`, `normal_sink_right_moderate_s015_p035`
- `slip_ice_s010_p040`, `slip_ice_s025_p030`
- `sink_left_severe_s020_p035`, `sink_right_severe_s025_p035`

## 5. Window extraction

각 window는 endpoint `t`에 대해 `[t-L+1, ..., t]`인 causal sequence다. 50/100 samples 전체가 `training_eligible=true`이고 동일한 `hazard_class_id` 0/1/2일 때만 사용했다. 따라서 onset을 가로지르거나 SINK `[t1,t2)` unresolved interval, censor 이후, invalid annotation을 포함한 window는 없다. Stride는 10 samples(10 ms)다.

Train은 available count가 run별 길이에 크게 좌우되므로 deterministic하게 run/class당 최대 200개를 시간축 전체에서 균등 선택하고 inverse train-window-frequency class weight를 사용했다. Validation과 holdout은 cap 없이 모든 eligible window를 사용했다.

## 6. Class/window count

| Window | Split | NORMAL | SLIP | SINK | Total |
|---|---|---:|---:|---:|---:|
| 50 ms | Train available | 9,711 | 925 | 593 | 11,229 |
| 50 ms | Train after cap | 3,652 | 521 | 551 | 4,724 |
| 50 ms | Validation | 2,977 | 241 | 172 | 3,390 |
| 50 ms | Holdout | 3,149 | 348 | 411 | 3,908 |
| 100 ms | Train available | 9,611 | 900 | 568 | 11,079 |
| 100 ms | Train after cap | 3,614 | 501 | 531 | 4,646 |
| 100 ms | Validation | 2,947 | 236 | 162 | 3,345 |
| 100 ms | Holdout | 3,114 | 338 | 401 | 3,853 |

## 7. Raw IMU sanity

통계는 33 valid run의 eligible class samples NORMAL 159,754, SLIP 15,488, SINK 12,149개로 계산했다. Acceleration 단위는 m/s², gyro 단위는 rad/s다.

| Class | Channel | Mean | Std | Median | P05 | P95 | Min | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NORMAL | accel_x | 0.0874 | 0.8932 | 0.1196 | -1.3192 | 1.2240 | -11.4535 | 5.9706 |
| NORMAL | accel_y | 0.0187 | 1.1994 | 0.1624 | -1.7294 | 1.5102 | -8.7740 | 9.0967 |
| NORMAL | accel_z | 9.7970 | 2.0659 | 9.5849 | 7.7612 | 12.3925 | -0.2778 | 43.4965 |
| NORMAL | gyro_x | 0.0016 | 0.1879 | -0.0052 | -0.2904 | 0.2819 | -1.5348 | 1.0742 |
| NORMAL | gyro_y | 0.0008 | 0.1415 | 0.0303 | -0.2580 | 0.1789 | -1.2242 | 1.0154 |
| NORMAL | gyro_z | 0.0495 | 0.0898 | 0.0438 | -0.0854 | 0.1997 | -0.4275 | 0.5711 |
| SLIP | accel_x | -0.2398 | 3.1946 | -0.3043 | -4.5029 | 5.0763 | -30.3097 | 16.2654 |
| SLIP | accel_y | -0.0005 | 4.6817 | 0.1425 | -7.2619 | 6.8774 | -43.0853 | 28.3807 |
| SLIP | accel_z | 9.2995 | 5.8886 | 8.7600 | 0.4457 | 20.6608 | -10.2108 | 42.1265 |
| SLIP | gyro_x | 0.1194 | 0.9815 | 0.0824 | -1.1126 | 1.5163 | -5.2001 | 4.9945 |
| SLIP | gyro_y | -0.0163 | 0.6345 | -0.0470 | -0.9937 | 0.8810 | -2.7791 | 4.1742 |
| SLIP | gyro_z | -0.1929 | 0.8318 | -0.1831 | -1.3837 | 0.9242 | -6.3295 | 5.0008 |
| SINK | accel_x | -0.1834 | 1.0925 | -0.0971 | -1.9566 | 1.4266 | -8.5506 | 3.2257 |
| SINK | accel_y | 0.0308 | 1.3398 | 0.0839 | -1.9918 | 2.1166 | -7.1892 | 8.9898 |
| SINK | accel_z | 9.8341 | 1.6471 | 9.7729 | 7.5121 | 12.2164 | 2.4933 | 21.7173 |
| SINK | gyro_x | -0.0067 | 0.2613 | -0.0207 | -0.3719 | 0.3978 | -1.5323 | 1.3004 |
| SINK | gyro_y | -0.0356 | 0.3288 | -0.0356 | -0.6051 | 0.4669 | -1.0958 | 0.7072 |
| SINK | gyro_z | 0.0519 | 0.1175 | 0.0457 | -0.1378 | 0.2409 | -0.4095 | 0.6180 |

핵심 관찰:

- 세 class range는 동일하지 않다. SLIP은 NORMAL보다 모든 channel의 분산이 크며, 특히 `accel_y` std는 4.6817 대 1.1994, `gyro_z`는 0.8318 대 0.0898이다.
- SINK와 NORMAL은 상당히 겹친다. 가장 뚜렷한 차이 중 하나는 `gyro_y` std 0.3288 대 0.1415이지만 단일 threshold로 분리될 정도는 아니다.
- established-slip t1 정렬 plot은 t1 주변/이후 acceleration과 angular-rate 변동 증가를 보인다. Sink physical t1/t2 정렬 변화는 더 작고 run/gait phase 분산과 겹친다.
- observed-outcome 내부 `median + 6*MAD` run-range audit에서 `normal_sink_left_moderate_s015_p035`의 `gyro_y` 한 건이 outlier였다. Class sample의 최대 단일-run 비율은 NORMAL 5.0%, SLIP 39.3%(`slip_ice_s025_p035`), SINK 27.0%(`sink_right_severe_s025_p035`)다. 한 run이 class 전체를 대표하지는 않지만 SLIP의 39.3% 집중은 Pilot limitation이며 train per-run cap이 필요한 근거다.
- eligible input에 non-finite 값은 0이고 반복 extreme 기반 clipping suspicion도 0이다. 대표 NORMAL trace는 `normal_concrete_s015_p000`의 1초 구간이다.

전체 통계와 run range는 local `raw_sanity.json`, 비교 그림은 `plots/`에 저장했다. 이 관찰을 handcrafted feature 선택으로 사용하지 않았다.

## 8. Train-only normalization

허용된 전처리는 train split의 eligible raw IMU 113,556 samples로 fit한 per-channel z-score뿐이다. Validation/holdout 통계는 fit에 사용하지 않았다.

- Mean: `[0.050608, -0.007150, 9.762341, 0.007166, -0.009553, 0.028361]`
- Std: `[1.162755, 1.735032, 2.518550, 0.306614, 0.225057, 0.224974]`

Filtering, smoothing, derivative, magnitude, FFT 또는 diagnostic feature는 사용하지 않았다.

## 9. Architecture와 parameter count

- MLP: `[B,L,6]` flatten → Linear(`L*6`, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 3). 50 ms는 21,443 parameters, 100 ms는 40,643 parameters다.
- GRU: one-layer unidirectional GRU(input 6, hidden 32, dropout 0) 마지막 hidden → Linear(32, 3). 두 window 모두 3,939 parameters다.

모든 후보는 CrossEntropyLoss, inverse-frequency train-only class weights, Adam 0.001, batch 128, 최대 40 epochs, validation macro F1 patience 6을 동일하게 사용했다.

## 10. Validation candidate 결과

| Candidate | Parameters | Macro F1 mean ± std | NORMAL recall | SLIP recall | SINK recall | Seed macro F1 (20260827 / 28 / 29) |
|---|---:|---:|---:|---:|---:|---|
| MLP 50 ms | 21,443 | 0.8858 ± 0.0050 | 0.9856 | 0.9433 | 0.7074 | 0.8791 / 0.8873 / 0.8911 |
| MLP 100 ms | 40,643 | **0.9071 ± 0.0023** | 0.9888 | 0.8503 | 0.8704 | 0.9043 / 0.9073 / 0.9098 |
| GRU 50 ms | 3,939 | 0.7822 ± 0.0181 | 0.9142 | 0.9308 | 0.7461 | 0.7583 / 0.7861 / 0.8023 |
| GRU 100 ms | 3,939 | 0.8526 ± 0.0248 | 0.9587 | 0.9393 | 0.8354 | 0.8188 / 0.8774 / 0.8618 |

MLP 100 ms의 seed std가 가장 작았다. GRU 100 ms는 50 ms보다 나았지만 seed 편차가 가장 컸다.

## 11. Selection

Primary는 3-seed validation macro F1 mean, secondary는 mean per-class recall의 최솟값, tertiary는 shorter window, parameter 수와 단순성이다. Macro F1 차이 0.005 이내를 near-tie로 미리 정의했다. `MLP 100 ms`만 최고 0.9071의 0.005 이내에 들어왔으므로 holdout 전에 그대로 선택했다. Holdout model seed도 config에 미리 고정한 `20260827`이다.

## 12. Holdout one-shot result

선택 이후 `MLP 100 ms / seed 20260827`만 Pilot internal holdout에서 한 번 평가했다. Holdout을 본 뒤 architecture, window, split, normalization 또는 hyperparameter를 변경하지 않았다.

- Accuracy: **0.9434**
- Macro precision: **0.8866**
- Macro recall: **0.8391**
- Macro F1: **0.8614**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.9639 | 0.9855 | 0.9746 | 3,114 |
| SLIP | 0.8662 | 0.7663 | 0.8132 | 338 |
| SINK | 0.8297 | 0.7656 | 0.7964 | 401 |

## 13. Confusion matrix

Rows are actual class and columns are predicted class in `NORMAL, SLIP, SINK` order.

| Actual \ Predicted | NORMAL | SLIP | SINK |
|---|---:|---:|---:|
| NORMAL | 3,069 | 22 | 23 |
| SLIP | 39 | 259 | 40 |
| SINK | 76 | 18 | 307 |

## 14. Run-balanced result

각 run의 window accuracy를 동일 가중한 run-balanced accuracy는 **0.9288**다. 각 run/class recall을 먼저 평균한 run-balanced recall은 NORMAL 0.9867, SLIP 0.7111, SINK 0.8388이며 그 macro average는 **0.8455**다. Window-weighted accuracy보다 낮아 긴 benign run이 overall accuracy를 높이는 효과가 있음을 보여준다.

## 15. Failure cases

가장 낮은 run accuracy는 `sink_right_severe_s025_p035`의 0.8017이다. 이 run의 SINK 318 windows 중 73개가 NORMAL, 18개가 SLIP으로 분류되어 전체 SINK→NORMAL 76건의 대부분을 차지했다. 다음은 confusion type별 최대 또는 대표 source run이다.

| Confusion | Representative run | Windows |
|---|---|---:|
| NORMAL → SLIP | `normal_sink_right_moderate_s015_p035` | 17 |
| NORMAL → SINK | `slip_ice_s010_p040`의 pre-event NORMAL | 17 |
| SLIP → NORMAL | `slip_ice_s025_p030` | 21 |
| SLIP → SINK | `slip_ice_s025_p030` | 24 |
| SINK → NORMAL | `sink_right_severe_s025_p035` | 73 |
| SINK → SLIP | `sink_right_severe_s025_p035` | 18 |

두 Slip holdout run accuracy도 `slip_ice_s010_p040` 0.8678, `slip_ice_s025_p030` 0.8770으로 benign controls보다 낮았다. Exact diagnostics는 사후 위치 확인에만 사용했고 model input에는 넣지 않았다.

## 16. Interpretation

이 Pilot에서 established-state 3-class 분류는 Pelvis IMU6 short sequence만으로 **가능성이 있다**. 선택된 MLP가 validation macro F1 0.9071, holdout macro F1 0.8614를 보였고 모든 class recall이 0.76 이상이었다. 그러나 run-balanced SLIP recall 0.7111과 특정 right severe Sink의 집중 miss는 scenario/run variation이 아직 충분히 일반화되지 않았음을 보여준다.

GRU는 parameter 수가 훨씬 작지만 같은 window의 MLP보다 낮았다. 따라서 이 Pilot은 recurrent temporal modeling이 우월하다는 evidence를 주지 않으며, 100 ms raw context와 단순 MLP가 가장 안정적이었다. 모델을 키우거나 새 architecture를 추가할 근거도 아직 없다.

## 17. Important limitations

- 33 valid-run Pilot 내부 split이며 fully unseen scenario family, random realization, initial gait phase 또는 final test generalization을 증명하지 않는다.
- Overlapping windows는 독립 sample이 아니다. Run-disjoint split과 run-balanced metric으로 누수를 줄였지만 run 수 자체가 작다.
- 이 PoC는 window 전체가 이미 동일 established class인 경우만 본다. onset-crossing window를 사용하지 않았으므로 Slip onset 직후, Sink physical t1 직후 또는 20/50 ms fast-reflex latency는 검증하지 않았다.
- Holdout은 project final test가 아니라 한 번 연 Pilot internal holdout이다.
- Full Dataset, CNN/LSTM comparison, sensor variation/noise, quantization과 E84 deployability는 범위 밖이다.

## 18. Artifacts와 reproducibility

- Experiment config: [`20260827_first_classification_poc.yaml`](../configs/experiment/20260827_first_classification_poc.yaml)
- Model configs: [`mlp.yaml`](../configs/model/mlp.yaml), [`gru.yaml`](../configs/model/gru.yaml)
- Local Gitignored artifacts: `artifacts/runs/20260827_first_classification_poc/`
- Stored outputs: `split.json`, `normalization.json`, `raw_sanity.json`, 3 plots, 12 best checkpoints, `metrics.json`, `confusion_matrix.csv`

## 19. 현재 checkpoint

제한된 상태 이름은 `FIRST_CLASSIFICATION_POC_COMPLETE`다. 이는 Pilot-only established-state separability evidence가 생겼다는 뜻이며 `MODEL_READY`, early detection ready 또는 Full Dataset ready를 뜻하지 않는다.

## 20. Next step

다음 별도 승인 작업은 established onset을 기준으로 onset-crossing causal history를 다루는 Time-to-Separation이다. 이 보고서에서는 Time-to-Separation, 20/30 ms sweep, Full Dataset 또는 full model-family comparison을 시작하지 않는다.
