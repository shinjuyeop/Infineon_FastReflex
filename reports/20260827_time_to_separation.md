# Hazard Time-to-Separation Analysis

## 1. 목적

이 분석은 첫 classification PoC에서 선택된 frozen `MLP 100 ms`가 1 kHz Pelvis IMU6만으로 SLIP/SINK physical event 전후 언제부터 stable hazard argmax를 내는지 확인한다. 여기서 separation time은 현재 Pilot, 현재 classifier와 10 ms diagnostic persistence에 한정된 관찰값이며 actual robot guarantee, final detector latency 또는 deployment latency가 아니다.

## 2. Dataset/model provenance

- Dataset: `hazard_pilot_20260827`, 40 runs / 320,000 samples
- Replay pool: BENIGN 16, observed SLIP 8, hazardous SINK 9; INVALID 7 제외
- Manifest SHA-256: `2a539abe122f8d4e06429d39db2dc37ec6f7569a8360959e64681d9f0ac1cf3e`
- First-PoC split SHA-256: `3b1b29a5e009783da2db0d1bdd198df24695d44c4b0cc55228bf28dfefda2a75`
- Train normalizer SHA-256: `ec9f20140f34a27eb21e82f72b78240114e371a95180020df7066336ceb1bcb9`
- Model: selected `MLP 100 ms`, class order `NORMAL / SLIP / SINK`
- Checkpoint seeds: `20260827`, `20260828`, `20260829`
- Checkpoint SHA-256: `73049b44…`, `e3211531…`, `c321f2d3…`

Canonical dataset validator로 metadata, manifest와 40 NPZ SHA, sample/timestamp/array alignment를 replay 전후 확인했다. Split, normalizer와 세 checkpoint도 config의 exact SHA로 replay 전후 검증했다.

## 3. 왜 retraining하지 않았는가

목적은 established-state classifier가 학습에 없던 onset/early interval에서 그대로 어떻게 행동하는지 보는 것이다. 새 label, threshold, feature, architecture, split 또는 optimizer를 도입하지 않았고 checkpoint를 다시 학습하지 않았다. 기존 MLP 50 ms/GRU 100 ms checkpoint도 architecture competition을 피하기 위해 secondary replay하지 않았다.

## 4. Event reference와 evidence boundary

- SLIP `t0`: first low-friction patch contact, `t1`: frozen ANY established Slip onset, `t3`: first censor/fall 또는 run end.
- SINK `t0`: first soft-patch contact, `t1`: physical sink onset, `t2`: frozen degradation onset, `t3`: first censor/fall.
- `t3`는 exclusive evidence boundary다. 이후 prediction은 trajectory에는 보존하지만 timing score에 사용하지 않는다.

SLIP `t1`은 최초 미끄럼이 아니라 50 mm anchor drift와 3 ms persistence가 끝난 established reference다. 따라서 `t0 <= prediction < t1`은 false positive가 아니라 `EARLY_SLIP_SIGNAL_CANDIDATE`로 구분한다. SINK에서는 `t1 <= prediction < t2`만 physical sink 뒤, frozen degradation 전의 positive-margin evidence다.

## 5. Causal replay 방법

각 valid run의 full raw IMU trace에서 endpoint `t`마다 `[t-99, ..., t]`만 입력하고 1 sample/1 ms stride로 inference했다. Future sample, terrain, speed, contact, drift, penetration, pelvis pose와 fall state는 model input에 사용하지 않았다. Exact diagnostics는 event alignment와 scoring에만 사용했다.

각 endpoint의 logits, softmax probability와 argmax를 세 seed별 compressed NPZ에 저장했다. `first_correct_endpoint`는 `t0` 이후 최초 target argmax다. `first_sustained_correct`는 target argmax가 10 consecutive ms가 되었음을 온라인에서 확인할 수 있는 10번째 endpoint다. 이 persistence는 분석용이며 deployment rule이 아니다.

## 6. SLIP t0/t1/t3와 early candidate

`t0~t1` 사이 sustained SLIP은 seeds별 5/8, 5/8, 6/8 runs에서 나타났다. 공통 early-candidate runs는 다음 다섯 개다.

- `slip_ice_s010_p035`
- `slip_ice_s015_p035`
- `slip_ice_s020_p035`
- `slip_ice_s025_p035`
- `slip_ice_s025_p030`

Seed `20260829`는 `slip_ice_s010_p040`도 t1 27 ms 전에 sustained SLIP을 냈다. 다만 이 run은 세 seed 모두 t0 이전 sustained SLIP false firing도 있었으므로 별도 incipient ground truth 없이 강한 early-onset evidence로 사용할 수 없다.

## 7. SLIP horizon recall

Recall은 한 run을 한 event opportunity로 계산한다. Seed 순서는 `20260827 / 20260828 / 20260829`다. Median latency도 각 horizon까지 검출된 run만 대상으로 하며 t1 이전 candidate는 음수다.

| Horizon from t1 | Detected by seed | Event recall mean ± std | Median sustained latency by seed (ms) | Pre-t0 sustained FP runs |
|---:|---|---:|---|---|
| 0 ms | 5/8 / 5/8 / 6/8 | 0.6667 ± 0.0589 | -50 / -34 / -38 | 1 / 1 / 1 |
| +20 ms | 5/8 / 5/8 / 6/8 | 0.6667 ± 0.0589 | -50 / -34 / -38 | 1 / 1 / 1 |
| +30 ms | 5/8 / 5/8 / 6/8 | 0.6667 ± 0.0589 | -50 / -34 / -38 | 1 / 1 / 1 |
| +50 ms | 5/8 / 5/8 / 6/8 | 0.6667 ± 0.0589 | -50 / -34 / -38 | 1 / 1 / 1 |
| +100 ms | 6/8 / 7/8 / 7/8 | **0.8333 ± 0.0589** | -42.5 / -30 / -35 | 1 / 1 / 1 |

모든 SLIP run은 결국 t3 전에 sustained SLIP이 됐다. 전체 8 run 기준 median latency는 -31.5 / -28.5 / -31.5 ms이고, 실제 censor가 있는 run의 median detection-to-censor margin은 1,112 / 1,112 / 1,101 ms다. 100 ms에서 남는 seed/run 차이는 짧은 horizon에 비해 작지만 continuous false-positive 문제와 분리해서 해석해야 한다.

## 8. SINK t0/t1/t2/t3

SINK primary horizon 결과는 SLIP과 다르다. 20 ms 이내 stable SINK는 zero-margin `sink_right_severe_s015_p030` 한 run에서만 seeds `20260827/29`에 나타났다. +30 ms에는 seed `20260828`도 같은 run을 검출했다. Positive-margin 7 runs 중 20/30/50/100 ms 내 t2 전 sustained detection은 0이었다.

그럼에도 positive-margin 7 runs는 세 seed 모두 결국 t2 전에 sustained SINK가 됐다. 즉 pre-degradation information은 존재하지만 현재 classifier에서 빠르게 stable해지지는 않는다.

- Positive-margin median latency from t1: 296 / 425 / 435 ms
- Median margin before t2: 197 / 68 / 72 ms
- All-SINK detected before t3: 각 seed 8/9
- All-SINK median latency including zero-margin: 292 / 419.5 / 419.5 ms

따라서 이 classifier의 SINK 출력은 대체로 physical t1 직후 20–100 ms가 아니라 t2에 가까워지는 posture/gait evolution을 보고 안정화되는 것으로 해석하는 편이 타당하다.

## 9. Zero-margin Sink cases

`t1 == t2`는 두 run이다.

- `sink_right_severe_s015_p030`: sustained latency -6 / +30 / -9 ms. Seed에 따라 t0~t1 candidate 또는 t2 직후 detection이다.
- `sink_left_severe_s015_p030`: 세 seed 모두 t3 전 10 ms sustained SINK가 없었고, 반대로 세 seed 모두 t0 이전 sustained SINK false firing이 있었다.

두 run은 pre-degradation eligible recall의 분모에서 제외했다. Full trajectory는 artifact에 보존했다.

## 10. SINK horizon recall

Pre-degradation recall은 positive-margin 7 runs 중 `t1+H` 이내이면서 `t2` 전 sustained detection 비율이다.

| Horizon from t1 | All-event detected by seed | All-event recall mean ± std | Pre-degradation recall | Median sustained latency by seed (ms) | Median margin before t2 |
|---:|---|---:|---:|---|---|
| 0 ms | 1/9 / 0/9 / 1/9 | 0.0741 ± 0.0524 | 0/7 / 0/7 / 0/7 | -6 / N/A / -9 | N/A |
| +20 ms | 1/9 / 0/9 / 1/9 | 0.0741 ± 0.0524 | 0/7 / 0/7 / 0/7 | -6 / N/A / -9 | N/A |
| +30 ms | 1/9 / 1/9 / 1/9 | 0.1111 ± 0.0000 | 0/7 / 0/7 / 0/7 | -6 / +30 / -9 | N/A |
| +50 ms | 1/9 / 1/9 / 1/9 | 0.1111 ± 0.0000 | 0/7 / 0/7 / 0/7 | -6 / +30 / -9 | N/A |
| +100 ms | 1/9 / 1/9 / 1/9 | **0.1111 ± 0.0000** | 0/7 / 0/7 / 0/7 | -6 / +30 / -9 | N/A |

Descriptive +300 ms에서 all-event recall은 5/9, 1/9, 1/9으로 seed variance가 매우 컸고, pre-degradation recall도 4/7, 0/7, 0/7이었다. 그러므로 +300 ms도 robust early horizon으로 볼 수 없다.

## 11. Split별 primary result

Primary interpretation은 validation과 이미 공개된 Pilot holdout에 둔다. 아래 값은 +100 ms event recall이며 seed 순서는 동일하다.

| Outcome | Train | Validation | Pilot holdout |
|---|---|---|---|
| SLIP | 4/5 / 5/5 / 4/5 | 1/1 / 1/1 / 1/1 | 1/2 / 1/2 / 2/2 |
| SINK | 1/5 / 1/5 / 1/5 | 0/2 / 0/2 / 0/2 | 0/2 / 0/2 / 0/2 |

Pilot holdout은 first PoC에서 이미 공개된 development split이며 sealed final test가 아니다.

## 12. BENIGN false-positive audit

`any hazard run FP`는 단 한 endpoint라도 hazard argmax가 있었던 run 수이고, sustained columns는 같은 hazard가 10 ms 지속된 run 수다. Seed 순서는 동일하다. 한 run에서 SLIP과 SINK sustained FP가 모두 발생할 수 있다.

| Scenario | Runs | Any hazard run FP | Sustained any hazard | Sustained SLIP | Sustained SINK |
|---|---:|---|---|---|---|
| Concrete | 4 | 4 / 4 / 4 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Marble | 4 | 4 / 4 / 4 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Uniform Sand | 4 | 4 / 4 / 4 | **4 / 4 / 4** | 0 / 4 / 2 | 4 / 2 / 1 |
| Benign mild Sink | 2 | 2 / 2 / 2 | 2 / 2 / 1 | 1 / 2 / 0 | 1 / 1 / 1 |
| Benign moderate Sink | 2 | 2 / 2 / 2 | **2 / 2 / 2** | 2 / 2 / 2 | 2 / 2 / 2 |
| All BENIGN | 16 | 16 / 16 / 16 | **8 / 8 / 7** | 3 / 8 / 4 | 7 / 5 / 4 |

Isolated 1 ms hazard argmax는 모든 BENIGN run에 있었고, 10 ms persistence 뒤에도 run-level FP rate는 50.0% / 50.0% / 43.75%다. 특히 Uniform Sand 4/4가 모든 seed에서 sustained hazard를 냈다. Concrete/Marble에는 sustained FP가 없었으므로 false firing은 soft-ground/benign-compliance domain에 집중된다. 현재 classifier/persistence를 continuous detector로 사용할 수 없다는 강한 제한이다.

## 13. Positive-run pre-event FP

SLIP 8개와 SINK 9개 run 모두 t0 이전에 isolated hazard window가 있었다. 10 ms persistence 뒤에는 다음 한 run씩만 남았으며 세 seed에서 동일했다.

- SLIP: `slip_ice_s010_p040` — pre-t0 sustained SLIP, 1/8 runs
- SINK: `sink_left_severe_s015_p030` — pre-t0 sustained SINK, 1/9 runs

따라서 해당 run의 t0 근처 early output은 causal signal과 기존 gait-pattern false firing을 구별할 ground truth가 부족하다.

## 14. Seed stability

- SLIP Recall@100 ms는 6/8, 7/8, 7/8로 mean 0.8333, std 0.0589다. Worst seed는 `20260827`이다.
- SINK Recall@100 ms는 모두 1/9로 수치상 안정적이지만, 안정적인 실패에 가깝다.
- Positive-margin SINK의 eventual t2-before detection은 모든 seed 7/7이나 median latency가 296→425→435 ms, median t2 margin이 197→68→72 ms로 initialization 영향이 크다.
- BENIGN sustained-any FP는 8/16, 8/16, 7/16이다. Seed `20260828`은 sustained SLIP FP 8/16으로 class-specific worst다.

특정 seed 하나의 빠른 SINK 결과를 강한 evidence로 해석하지 않았다.

## 15. Probability trajectory

`probability_trajectories.png`는 clean/recovery SLIP, left/right SINK, difficult SINK와 Uniform Sand를 세 seed mean/range로 비교한다. SLIP representative `slip_ice_s020_p035`는 t1 전부터 SLIP probability가 거의 1로 유지된다. 반면 Uniform Sand는 gait-periodic SINK/SLIP spikes를 반복해 benign FP 결과와 일치한다.

별도 raw IMU/probability plots는 t1 -100~+200 ms에서 future sample 없이 confidence와 IMU6를 함께 보여준다. Difficult SINK는 이 범위에서 NORMAL과 SINK confidence가 반복적으로 교차하고 10 ms stable condition은 아직 충족하지 않는다.

## 16. Difficult Sink run

`sink_right_severe_s025_p035`의 event time은 t0 1,508 ms, t1 1,540 ms, t2 2,495 ms, t3 censor 5,774 ms다.

- First single SINK argmax latency from t1: +43 / +35 / +24 ms
- First 10 ms sustained SINK confirmation: +480 / +460 / +440 ms
- Sustained confirmation margin before t2: 475 / 495 / 515 ms
- SINK probability at t1: 0.00027 / 0.000019 / 0.000068; NORMAL은 모두 0.999 이상
- SINK probability at t2: 0.9973 / 0.9993 / 0.9992

`[t1,t2)` argmax fraction은 NORMAL 46.9/46.4/44.2%, SLIP 0.6/4.6/1.2%, SINK 52.5/49.0/54.7%다. `[t2,t3)`에도 NORMAL 22.9/24.7/22.9%, SLIP 3.9/7.0/4.2%가 남는다. 따라서 prior holdout failure는 SINK signal이 전혀 없는 문제가 아니라, t1 직후의 강한 NORMAL과 이후 반복되는 NORMAL/일부 SLIP ambiguity가 window error로 누적된 결과다. SLIP output은 transient ambiguity이며 dominant explanation은 아니다.

## 17. Earliest promising horizon

**SLIP과 SINK를 함께 만족하는 promising horizon은 20/30/50/100 ms 중 없다.**

SLIP만 보면 +100 ms가 0.8333 ± 0.0589로 제한적인 후속 후보지만, SINK는 0.1111이고 validation/holdout SINK는 전부 0/2다. 동시에 BENIGN sustained FP가 7~8/16이므로 +100 ms를 detector gate나 latency contract로 freeze할 수 없다. Descriptive +300 ms SINK도 seed에 따라 5/9 대 1/9로 뒤집혀 대안이 아니다.

## 18. Interpretation

Pelvis IMU-only **early SLIP candidate**는 Pilot evidence가 있다. 여러 run에서 established t1보다 28~174 ms 앞서 stable SLIP이 시작되고 seeds 간 결과도 비교적 유사하다. 그러나 t1은 최초 motion onset이 아니고 positive/benign pre-event FP가 있으므로 physical early-onset latency라고 부를 수 없다.

Pelvis IMU-only **20–100 ms early SINK detection**은 현재 classifier에서 지지되지 않는다. Positive-margin SINK는 결국 모두 t2 전에 검출되지만 t1+296~435 ms 중앙값으로 늦고 t2 margin도 일부 seed에서 68~72 ms뿐이다. 이는 classifier가 immediate physical sink보다 진행 중인 posture/gait degradation을 관찰하는 쪽에 가깝다는 evidence다.

첫 PoC의 established-state holdout macro F1 0.8614는 continuous early detector 성능을 보장하지 않는다. Same-class established windows의 분류와 onset-crossing replay는 다른 문제이며, 높은 established accuracy가 early recall/benign FP로 직접 전이되지 않았다.

## 19. Limitations

- 8 SLIP, 9 SINK, 16 BENIGN의 deterministic simulation Pilot development evidence다.
- Holdout은 이미 공개된 Pilot split이며 fresh final test가 아니다.
- 10 ms persistence는 diagnostic일 뿐 tuning하거나 deployment threshold로 freeze하지 않았다.
- t1 이전 SLIP에 대응하는 independent incipient ground truth가 없다.
- SINK zero-margin 2 runs는 pre-degradation recall에서 제외했다.
- Uniform Sand/benign Sink FP는 normal-domain coverage와 onset-crossing training 부족 가능성을 보여주지만, 이번 분석에서 label/model을 변경해 원인을 검증하지 않았다.
- Simulation-to-real, target inference time, quantization과 E84 behavior는 검증하지 않았다.

## 20. Next recommendation

다음 별도 milestone은 이 결과를 반영한 Full Dataset design이다. Continuous benign soft-ground coverage, onset-relative interval provenance, scenario/randomization과 fresh test reservation을 먼저 설계한 뒤 full model comparison/retraining 여부를 결정해야 한다. 이번 milestone에서는 dataset 생성, relabeling, retraining, new architecture 또는 threshold tuning을 시작하지 않는다.
