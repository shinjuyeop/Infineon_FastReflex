# Model V2 Internal Failure Audit

## 1. Purpose

This read-only audit localized the frozen failed candidate `model_v2_data_only_gru20_20260901` without retraining, generation, tuning, or architecture changes. The verdict is:

```text
MODEL_V2_INTERNAL_FAILURE_AUDIT_ACTIONABLE
```

The primary actionable defect is not a generic 0.20 m/s or right-side observability failure. Six Slip failures are strong early responses under the unchanged primary timing rule; five occur inside frozen future-Slip precursor regions and one starts 5 ms before the 30 mm precursor. The genuine residual is delayed Marble Support: the response at I1 exceeds 0.99 but is only 3 ms long, below the frozen 5 ms persistence. Effective TRAIN contains the exact timing and event-local waveform, but only one unique delayed-Support waveform per source and sparse event endpoints. The smallest justified intervention is `V2_EXTRACTION_REBALANCE`, designed before any retraining.

## 2. Starting state

The audit started from a clean `main` at:

```text
HEAD = origin/main = ad338e86a626ce0a3f8ba3ec3b312db6469e98e1
Train data-only Model V2
```

The audit protocol was created before detailed probability/feature analysis at `configs/experiment/20260901_model_v2_internal_failure_audit.yaml`. Its SHA-256 is `9d2a4c5388ee576b589e60c815ae4af3a552bd2b2f92d4b8335e22e01a7ce102`.

## 3. Evidence boundary

Authorized waveforms were Unified TRAIN, V2_TRAIN, and the already-open V2_VALIDATION. Frozen training extraction/HNM artifacts and previously authorized development summaries were also inspected.

```text
Generalization VALIDATION V2 inference: NO
Current Unified HOLDOUT waveform reopened: NO
Current Unified HOLDOUT new inference: NO
Generalization HOLDOUT waveform opened: NO
Generalization HOLDOUT inference: NO
Generalization HOLDOUT guard count: 0
```

No Generalization VALIDATION V2 inference was run. Neither HOLDOUT waveform set was decoded or replayed.

## 4. V1/V2 preservation

All protected files were checked before diagnosis and were not modified.

| Protected item | SHA-256 / result |
|---|---|
| V1 Hazard freeze | `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2` exact |
| Feature schema | `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb` exact |
| V1 normalizer | `610a91c24a594a3cd1701b07549f4f3c4b7f4cb04eb354f9d41378214a6800a9` exact |
| V1 checkpoints | `e6bada49…`, `b04877d…`, `b6c782bd…`, 3/3 exact |
| V2 candidate freeze | `edb16b7d96fb38e680e36dcca5ecd7b1c1682ba410b94d21f40c2a38d4ed8725` exact |
| V2 normalizer | `e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a` exact |
| V2 checkpoints | `dd6c8581…`, `8e6709da…`, `811f486c…`, 3/3 exact |
| V2 dataset freeze / manifest | `fd81b647…` / `7a036d34…` exact |
| V2 NPZ aggregate | `5a8dfd54d1c08413dc6fb18d957269a5fbfea8cf526e4f29c78510886186e11c` exact |

V1 remains restorable. V2 remains the exact failed internal candidate; no `model_v2_freeze.json` was created and no artifact was promoted.

## 5. V2_VALIDATION reproduction

All 96 valid V2_VALIDATION runs were replayed through the canonical causal 80D, three-checkpoint ensemble, `0.99 / 5 ms` evaluation path. The newly computed result is structurally identical to the frozen `v2_validation_evaluation.json`.

| Metric | Reproduced | Frozen | Parity |
|---|---:|---:|---|
| Hazard recall | 55/64 = 85.94% | 55/64 | exact |
| Slip recall | 29/35 = 82.86% | 29/35 | exact |
| Support recall | 27/30 = 90.00% | 27/30 | exact |
| Confirmed no-hazard specificity | 26/26 = 100% | 26/26 | exact |
| Premature | 6/64 = 9.38% | 6/64 | exact |
| Right-only Support | 12/12 | 12/12 | exact |
| Staged Sand benign | 8/8 | 8/8 | exact |
| Speed Sand benign | 12/12 | 12/12 | exact |

Speed Slip was `.20 = 9/14`, `.25 = 11/11`, `.30 = 9/10`; Support was `.20 = 8/8`, `.25 = 11/14`, `.30 = 8/8`. Slip side recall was left `3/3`, right `0/3`, bilateral `26/29`. Evaluation parity is `PASS`.

The ignored diagnostic bundle contains 64 Hazard timelines and the required nine failure rows:

- `artifacts/runs/20260901_model_v2_internal_failure_audit/hazard_run_timelines.csv` — SHA `e94298a8b957218ed4f0a01a6864f7e0be58bded69476e54c3a1b59c19780169`.
- `artifacts/runs/20260901_model_v2_internal_failure_audit/failure_cases.csv` — one row per failed/premature Hazard run; SHA `8910b7ecebacc59a51c969d015524b7e4a0230a6f0303d8a1a9b7a30cd68b3f7`.
- `artifacts/runs/20260901_model_v2_internal_failure_audit/extraction_exposure.csv` — run-to-fit-window traceability; SHA `8cfd7b644bcf654a4ebb42de2733a9ea72e687f1c8113192a3e8e8df88ffb815`.
- `audit_summary.json` (SHA `c5dbac39b8ae0896999b1074625ed61d05b24dafc1157d938fb599dff3c015ab`) and three diagnostic PNGs in the same ignored run directory.

The JSON/CSV rows keep physical events separate from threshold crossing and `REFLEX_REQUIRED` onset. They include seed maxima, 0.99/0.95/0.90 excursions, event probabilities, censor/fall, contact phase, matched TRAIN counts/windows, and failure classification.

## 6. Failure localization

Exactly nine of 64 Hazard runs fail the frozen primary timing contract.

| Dimension | Exact failed/premature count |
|---|---|
| Result | 6 premature, 1 no-Reflex miss, 2 out-of-valid-window |
| Speed | .20: 5, .25: 3, .30: 1 |
| Side | bilateral: 3, right-only: 3, left-only: 3 |
| Source | Concrete: 4, Marble: 5 |
| Family | baseline immediate Ice: 1; Ice benign: 2; Ice precursor: 1; delayed Ice: 2; delayed Sand: 3 |

All six Slip failures are premature. All three non-Slip failures are the Marble half of the six-run `.25 m/s` delayed-Support cell.

## 7. Training coverage versus extraction exposure

The initial extractor produced 3,061 positives, but the deterministic TRAIN/monitor run partition placed 2,424 in optimizer fit and 637 in monitor. Fitted positives were 1,680 Slip and 744 Support.

| Target cell | TRAIN runs | Fit-positive windows | Share of subtype positives | V2_VAL | Correct |
|---|---:|---:|---:|---:|---:|
| .20 Slip | 37 | 435 | 25.89% Slip | 14 | 9 |
| .25 Slip | 73 | 870 | 51.79% Slip | 11 | 11 |
| .30 Slip | 31 | 375 | 22.32% Slip | 10 | 9 |
| left-only Slip | 13 | 135 | 8.04% Slip | 3 | 3 |
| right-only Slip | 8 | 75 | 4.46% Slip | 3 | 0 |
| bilateral Slip | 120 | 1,470 | 87.50% Slip | 29 | 26 |
| ordinary Support | 101 | 640 | 86.02% Support | 24 | 24 |
| delayed Support | 18 | 104 | 13.98% Support | 6 | 3 |
| left-only Support | 87 | 536 | 72.04% Support | 18 | 15 |
| right-only Support | 32 | 208 | 27.96% Support | 12 | 12 |

Run share and fit-window share are close for speed and event family. Thus a broad class/family weighting defect is not demonstrated. The narrower problem is temporal and diversity exposure: delayed Support contributes only eight isolated positive endpoints per fitted run, and the nine nominal runs per source collapse to one unique event-local waveform per source.

Fresh V2_TRAIN actual Slip sides were left/right/bilateral `8/8/87`. Effective TRAIN, after adding Unified TRAIN, was `13/8/120`; Unified added no right-only Slip. Within all 141 Slip runs, run shares were left/right/bilateral `9.22/5.67/85.11%`, while fitted Slip-positive shares were `8.04/4.46/87.50%`. Optimizer sample exposure follows those fit-window shares because each deterministic fit window appears once per epoch; no batch-level family sampler was used.

## 8. 0.20 m/s Slip

All 14 `.20 m/s` validation Slip runs are below. `I/M` denotes immediate or multi-contact physical progression; `benign` is the count of fully completed benign target episodes before first Slip. Contact is the first target contact and the episode containing first Slip.

| Run | Family/src/side | Contact / event episode | Dynamics | precursor → Reflex → Slip | max p / >=.99 | phase / peak drift | Result |
|---|---|---|---|---|---:|---|---|
| `bis_c_i07` | baseline/C/bilateral | 1764 / L:3 | I; benign 0 | 1810 → 1832 → 1829 | .99953 / 32 ms | left support / .0929 m | correct |
| `bis_c_i08` | baseline/C/bilateral | 1503 / L:4 | M; benign 3 | 2470 → 2476 → 2492 | .99991 / 67 | left support / .0970 | correct |
| `bis_m_i07` | baseline/M/bilateral | 1773 / L:5 | I; benign 0 | 1818 → 1827 → 1836 | .99977 / 52 | left support / .0876 | correct |
| `bis_m_i08` | baseline/M/bilateral | 1491 / L:7 | M; benign 11 | 2170 → 2165 → 2323 | .99994 / 75 | right support / .1616 | premature |
| `ibc_c_b07` | Ice-benign/C/right | 1504 / L:32 | M; benign 32 | 2466 → 2478 → 2627 | .99869 / 49 | left support / .0569 | premature |
| `ibc_c_b08` | Ice-benign/C/right | 1504 / L:34 | M; benign 35 | 2466 → 2478 → 2707 | .99869 / 43 | left support / .0907 | premature |
| `ibc_m_b07` | Ice-benign/M/bilateral | 1507 / L:5 | M; benign 3 | 2465 → 2469 → 2487 | .99908 / 27 | left support / .0639 | correct |
| `ibc_m_b08` | Ice-benign/M/bilateral | 1507 / L:5 | M; benign 3 | 2465 → 2469 → 2487 | .99908 / 45 | left support / .0636 | correct |
| `inp_c_p07` | precursor/C/bilateral | 1770 / L:4 | I; benign 0 | 1821 → 1838 → 1840 | .99965 / 35 | left support / .1829 | correct |
| `inp_c_p08` | precursor/C/right | 1504 / R:6 | M; benign 29 | 2466 → 2478 → 2627 | .99869 / 40 | left support / .0573 | premature |
| `inp_m_p07` | precursor/M/bilateral | 1773 / L:6 | I; benign 0 | 1820 → 1820 → 1838 | .99957 / 28 | left support / .1209 | correct |
| `inp_m_p08` | precursor/M/left | 1507 / L:5 | M; benign 3 | 2465 → 2469 → 2487 | .99908 / 23 | left support / .0740 | correct |
| `odi_c_d08` | delayed/C/bilateral | 1504 / R:6 | M; benign 20 | 2466 → 2478 → 2632 | .99934 / 31 | left support / .1351 | premature |
| `odi_m_d08` | delayed/M/bilateral | 1507 / L:5 | M; benign 3 | 2465 → 2469 → 2487 | .99936 / 29 | left support / .1251 | correct |

Effective TRAIN `.20` evidence is 37 Slip runs: 14 immediate and 23 multi-contact delayed; Concrete/Marble `21/16`; left/right/bilateral `5/6/26`; 435 fitted positives; 7,264 future-Slip precursor-masked samples; and 1,332 HNM windows over three rounds. The `.25/.30` comparisons are 73/31 runs, 870/375 fit positives, 19,406/14,809 masked precursor samples, and 2,628/1,116 HNM windows.

Run-to-window shares are proportional, so `.20` is not materially underrepresented after extraction. Its failures instead have a radically longer precursor-to-Slip interval: median `161 ms` versus `22 ms` for `.20` correct, and median 29 versus 3 benign contacts. Failure accel/gyro norm RMS is stronger (`14.71/1.28`) than correct (`10.92/0.67`), not weaker. Event-local normalized mean absolute amplitude is `1.532` versus `0.713`, with TRAIN-p01–p99 outside fraction `19.07%` versus `3.98%`; the largest shifts are gyro-y base/causal means and gyro-y deltas. The model nevertheless responds strongly and early.

Classification: `LOW_SPEED_EVENT_TIMING_SHIFT + PRECURSOR_TIMING_CONFLICT`, confidence `HIGH`. `LOW_SPEED_SIGNAL_AMPLITUDE_LIMIT` and broad `LOW_SPEED_TRAIN_COVERAGE_GAP` are rejected.

## 9. Right-only Slip

| Run | Family/source | precursor / Reflex / Slip | p at Slip | seed maxima | max p / >=.99 | accel/gyro RMS | feature abs / outside | Result |
|---|---|---|---:|---|---:|---:|---:|---|
| `ibc_c_b07` | Ice-benign/C | 2466 / 2478 / 2627 | .88595 | .99978/.99853/.99995 | .99869 / 49 ms | 15.70/1.29 | 2.039 / 28.42% | premature |
| `ibc_c_b08` | Ice-benign/C | 2466 / 2478 / 2707 | .99790 | .99978/.99853/.99994 | .99869 / 43 | 10.73/1.17 | .710 / 3.40% | premature |
| `inp_c_p08` | precursor/C | 2466 / 2478 / 2627 | .90527 | .99960/.99853/.99993 | .99869 / 40 | 15.53/1.28 | 2.015 / 28.20% | premature |

Effective TRAIN has 8 right-only Slip runs and 75 fitted Slip positives, versus left `13/135` and bilateral `120/1,470`. Right-only exposure is therefore small, but all three validation runs produce sustained, three-seed-high alerts inside future-Slip precursors. Right-only accel/gyro norm RMS medians `15.53/1.28` exceed the one matched `.20` left control (`10.57/.67`) and the eight correct `.20` bilateral controls (`11.19/.84`). A fully source/family-matched left control was unavailable, and the unmatched raw sign-correlation check was inconclusive; no sign-symmetry claim is made.

The 0/3 primary result therefore does not establish a right-side blind spot. It is caused by early timing, not absent signal or low probability. Classification: `PRECURSOR_TIMING_CONFLICT`, confidence `HIGH`. A latent `RIGHT_SLIP_SIDE_DISTRIBUTION_GAP` remains possible because fit share is only 4.46%, but it did not cause these three primary failures. `RIGHT_SLIP_OBSERVABILITY_LIMIT` is rejected.

## 10. Delayed Support

All six cases share `.25 m/s`, left-only Support, first target contact 1220, and `I1→Support = 56 ms`. At I1/Support the maximum spread/displacement is approximately `.0024/.0125 m` for Concrete and `.0024/.0126 m` for Marble.

| Run | Source | contact→I1 | I1 / Support | first crossing / Reflex | p(I1) / p(Support) | max p / >=.99 | Result |
|---|---|---:|---|---|---:|---:|---|
| `dss_c_s10` | Concrete | 1791 ms | 3011 / 3067 | 3011 / 3054 | .99435 / .98243 | .99714 / 7 ms | correct |
| `dss_c_s11` | Concrete | 1791 | 3011 / 3067 | 3011 / 3054 | .99435 / .98243 | .99714 / 7 | correct |
| `dss_c_s12` | Concrete | 1791 | 3011 / 3067 | 3011 / 3054 | .99435 / .98243 | .99714 / 7 | correct |
| `dss_m_s10` | Marble | 1792 | 3012 / 3068 | 3012 / none | .99785 / .98514 | .99785 / 3 | miss |
| `dss_m_s11` | Marble | 1792 | 3012 / 3068 | 3012 / 3667 | .99785 / .98514 | .99785 / 8 later | out of window |
| `dss_m_s12` | Marble | 1792 | 3012 / 3068 | 3012 / 3667 | .99785 / .98514 | .99785 / 8 later | out of window |

TRAIN timing matches validation exactly: Concrete/Marble contact→I1 is `1791/1792 ms`, and I1→Support is `56/56 ms`. TRAIN has nine delayed-Support runs per source, but the fit partition retains Concrete/Marble `7/6` runs and `56/48` positives. There are 1,013 extracted pre-I1 benign negatives (`539/474`) and 104 delayed-Support fit positives. Delayed Support is 15.13% of Support runs and 13.98% of Support fit positives: no gross family cap imbalance is present.

The decisive limitation is effective temporal/diversity exposure. All nine nominal runs of each source have one unique event-local waveform. The failed Marble local feature is an exact fitted-TRAIN match (RMS distance 0; only 0.0176% of feature values outside TRAIN p01–p99), yet the model produces only three consecutive `>=.99` samples at I1. The extractor gives eight isolated points per fitted delayed run across I1→Support and established-relative offsets; it does not ensure a contiguous positive neighborhood capable of learning the frozen 5 ms decision.

Classification: `POSITIVE_WINDOW_UNDEREXPOSURE + DELAYED_SUPPORT_SOURCE_CELL_GAP`, confidence `HIGH` for temporal endpoint underexposure and `MODERATE` for source-cell diversity. This is not a general Support problem: ordinary Support is 24/24 and Concrete delayed Support is 3/3.

## 11. Premature Ice

| Run | src/speed/side | precursor / crossing / Reflex / Slip | drift / velocity at Reflex | max pre-Slip drift | Outcome at Reflex | Primary / precursor-aware |
|---|---|---|---:|---:|---|---|
| `bis_m_i08` | M/.20/bilateral | 2170 / 2161 / 2165 / 2323 | .0257 m / .895 m/s | .0519 m | before frozen precursor | premature / pre-precursor by 5 ms |
| `ibc_c_b07` | C/.20/right | 2466 / 2468 / 2478 / 2627 | .0427 / 1.019 | .0521 | later Slip | premature / future-Slip precursor |
| `ibc_c_b08` | C/.20/right | 2466 / 2468 / 2478 / 2707 | .0427 / 1.019 | .0525 | later Slip | premature / future-Slip precursor |
| `inp_c_p08` | C/.20/right | 2466 / 2468 / 2478 / 2627 | .0427 / 1.019 | .0518 | later Slip | premature / future-Slip precursor |
| `odi_c_d08` | C/.20/bilateral | 2466 / 2468 / 2478 / 2632 | .0427 / 1.019 | .0517 | later Slip | premature / future-Slip precursor |
| `odi_m_d08` | M/.30/bilateral | 1889 / 1890 / 1894 / 2054 | .0349 / .850 | .0529 | later Slip | premature / future-Slip precursor |

Five of six alerts are inside the unchanged loaded exact-Ice `[30,50) mm` future-Slip precursor. The sixth Reflex begins 5 ms before the 30 mm crossing at 25.7 mm drift, and its first threshold crossing is 9 ms before. None is a fully observed benign release or a censored precursor. Under the primary contract all six remain failures and cannot retroactively make V2 pass. Under the already-frozen secondary view, five are physically meaningful future-Slip early alerts and one is a narrowly pre-precursor early alert.

Classification: `PRIMARY_METRIC_VS_PRECURSOR_SEMANTICS_TENSION`, confidence `HIGH`; the one pre-precursor case retains a genuine early-timing concern with `MODERATE` confidence. This does not make V2 a precursor detector: frozen secondary recall remains only 72/424 candidate episodes, with 352 precursor states missed.

## 12. Secondary 0.30/bilateral Slip

The sole `.30` miss is `odi_m_d08`, a bilateral delayed-Ice run. It is unanimous-high and fires 160 ms before Slip inside a `LATER_SLIP` precursor. The three bilateral failures are `bis_m_i08` at `.20`, `odi_c_d08` at `.20`, and `odi_m_d08` at `.30`; all are premature and all overlap the Ice timing mechanism. No additional bilateral or `.30` failure family is justified.

## 13. Seed and persistence behavior

Every one of the nine failure cases has all three seed maxima above `.99`; no failure is a single-seed outlier. The six Slip failures have ensemble `>=.99` streaks of `31–75 ms` and are `UNANIMOUS_HIGH_BUT_MISTIMED`. Delayed Marble Support has the same high I1 peak in all seeds. The first case reaches only 3 ms ensemble persistence; the other two later reach 8 ms at `+599 ms` after Support.

Persistence is not the root cause of Slip failure. For delayed Support, 5 ms exposes a temporally narrow learned response, but reducing persistence to 3 ms would be result-driven and would bypass the intended robustness contract. Training exposure should be corrected first. No persistence search was performed.

Every failure has maximum ensemble probability at least `.99785`; there is no low-confidence failure. The delayed-Support no-Reflex case is `NEAR_THRESHOLD/STRONG_BUT_TOO_BRIEF`; the remaining eight are `HIGH_CONFIDENCE_MISTIMED`. No threshold search was performed and threshold change is not justified.

## 14. HNM interaction

Each of 442 effective TRAIN runs contributed exactly 12 windows in each of three HNM rounds, for 15,912 total. Concrete and Marble each contributed 7,956. Relevant exposure is `.20 speed = 3,132` all-run HNM windows, `.20 Slip-bearing runs = 1,332`, right-only Slip-bearing runs = 288, and delayed-Support runs = 648. Family totals include baseline immediate Ice 1,296, delayed Ice 1,044, Ice precursor 1,152, and delayed Sand 648.

Forbidden-mask violations are exactly zero. The fixed run cap does not create a source imbalance and cannot explain Concrete 3/3 versus Marble 0/3 by count. Exact HNM endpoint/contact-phase provenance was not persisted; the audit does not invent it. The available evidence does not justify changing HNM.

## 15. Feature-space observations

No classifier or probe was trained. All comparisons use the frozen V2 normalizer and descriptive event-local causal 80D features.

- `.20` failures are higher-amplitude and more outlying than `.20` correct, principally in gyro-y base, rolling means, deltas, and variance. This accompanies the long multi-contact precursor structure; it is not an amplitude/observability limit because the model responds strongly.
- Delayed Marble Support has event-local normalized mean absolute amplitude `.287`, TRAIN-p01–p99 outside fraction `.0176%`, and nearest exact fitted-TRAIN RMS distance `0`. A feature-space coverage shift is not demonstrated at the scored event.
- Right-only Slip has strong raw norms and sustained probabilities. The single unmatched left control is insufficient for a reliable sign-reflection conclusion, so the audit uses no symmetry claim to justify its sensor decision.

## 16. Sensor implication

```text
10_CHANNEL_ARCHITECTURE_STILL_PLAUSIBLE
FINAL_SENSOR_ARCHITECTURE_NOT_READY
```

Right-only Support is 12/12. Right-only Slip has strong Pelvis IMU response and sustained three-seed alerts; its primary failure is timing. No new sensor observability limit is demonstrated. Sensor expansion is not justified, and the final sensor architecture remains unfrozen pending hardware/resource evidence.

## 17. Architecture implication

The unchanged GRU produced large V1→V2 improvements and now localizes failure to timing/extraction behavior. Current evidence answers the architecture questions as follows:

| Question | Answer | Evidence |
|---|---|---|
| Longer history justified? | NO | long-prelude Slip already causes sustained early response; delayed Support is exact local TRAIN match |
| LSTM justified? | NO | no coverage-controlled recurrent-memory failure was demonstrated |
| Larger GRU justified? | NO | right Support and ordinary Support are solved with 11,010 parameters |
| Threshold change justified? | NO | all failures reach >=.99785; no search authorized |
| Persistence change justified? | NO | would result-fit the 3 ms Marble peak; correct extraction first |
| Sensor expansion justified? | NO | right Slip signal and model response are strong |

`ARCHITECTURE_CHANGE_NOT_YET_JUSTIFIED` remains the correct status.

## 18. Failure attribution matrix

| Failure | Physical pattern | TRAIN coverage | Fit-window coverage | Model response | Likely cause | Confidence |
|---|---|---|---|---|---|---|
| .20 Slip | long multi-contact precursor; failure precursor→Slip median 161 ms | 37 runs, both sources, all sides | 435; 25.89% of Slip | strong 31–75 ms early | `LOW_SPEED_EVENT_TIMING_SHIFT + PRECURSOR_TIMING_CONFLICT` | HIGH |
| right-only Slip | three .20 Concrete future-Slip precursors | 8 runs; small but nonzero | 75; 4.46% | all seeds high, sustained 40–49 ms | `PRECURSOR_TIMING_CONFLICT`; latent side imbalance not causal here | HIGH |
| delayed Support | `.25`, left, Marble only; same timing/local waveform as TRAIN | 9/source, only 1 event-local waveform/source | Marble 48; eight isolated positives/run | 3 ms I1 peak or +599 ms late | `POSITIVE_WINDOW_UNDEREXPOSURE + SOURCE_CELL_DIVERSITY_GAP` | HIGH/MODERATE |
| premature Ice | five inside future-Slip precursor, one 5 ms before | future precursor explicitly masked from negatives | n/a primary score | unanimous high, sustained | `PRIMARY_METRIC_VS_PRECURSOR_SEMANTICS_TENSION` | HIGH |
| isolated .30 Slip | delayed, bilateral, later-Slip precursor | 31 `.30` Slip runs | 375 | unanimous high 160 ms early | same precursor timing mechanism | HIGH |
| bilateral Slip misses | two `.20`, one `.30`, all premature | 120 runs | 1,470 | unanimous high | same precursor timing mechanism | HIGH |

## 19. Smallest justified intervention

The selected action is:

```text
V2_EXTRACTION_REBALANCE
```

It should be designed, not implemented, in the next milestone. The conceptual correction is:

- preserve primary Slip/Support/I1 semantics, normalizer contract, architecture, threshold, and persistence;
- predeclare source/family-aware rare-positive quotas so delayed Support cannot be represented by only 48 Marble fit endpoints;
- ensure the delayed-Support positive extraction includes causal adjacent endpoint neighborhoods around I1 and the I1→Support transition, aligned with the already-frozen persistence contract;
- retain pre-I1 benign negatives and every precursor/censor forbidden mask;
- record fit/monitor and batch exposure by family, source, speed, side, and event timing before optimizer use.

No numeric weight or quota is chosen here. A design milestone must freeze exact values before retraining. Targeted augmentation is not the first action because the failed Marble event-local waveform and timing already exist exactly in fitted TRAIN; new data can be reconsidered only if extraction-balanced training still fails across predeclared diverse cells.

## 20. V2 decision table

| Possible next action | Evidence for | Evidence against | Decision |
|---|---|---|---|
| targeted .20 Slip augmentation | `.20` primary 9/14 | 37 runs/435 windows; all five failures are strong early | reject now |
| targeted right-only Slip augmentation | only 8 runs/75 windows | all three failures are sustained future-precursor alerts | reject as current root cause |
| delayed Support augmentation | one unique local waveform/source | failed waveform already exact in fitted TRAIN | defer |
| extraction rebalance | Marble peak only 3 ms; 48 source positives; isolated endpoint policy | family share broadly proportional | **select** |
| HNM adjustment | many negatives | equal 12/run/round; source-balanced; zero violations | reject |
| threshold change | Marble is near frozen boundary | all failures >=.99785; specificity impact untested | reject |
| persistence change | 3 ms would rescue one Marble trace | result-driven; early-Ice streaks already long | reject |
| longer GRU history | slow .20 dynamics | model already responds early and strongly | reject |
| LSTM | none specific | capacity-controlled evidence absent | reject |
| sensor expansion | right-only primary 0/3 | signal strong; model alerts; right Support 12/12 | reject |
| proceed to Generalization VALIDATION | most Slip failures are semantic tension | three genuine delayed-Support failures remain | reject now |

## 21. Limitations

- This remains simulator/policy/domain-specific evidence.
- Primary scores remain frozen; precursor interpretation does not rewrite them.
- V2_VALIDATION has only three right-only Slip and six delayed-Support cases.
- Nominal delayed-Support physical signatures are distinct, but event-local waveforms collapse to one per source; the proposed extraction correction still requires a predeclared design and fresh internal test.
- HNM endpoint/contact-phase provenance was not persisted, so only exact run/family/source/speed exposure and forbidden-mask evidence are reported.
- Feature comparisons are descriptive; no probe or post-result threshold optimization was used.
- Generalization VALIDATION remains an unconsumed external development gate, and Generalization HOLDOUT remains sealed final evidence.

## 22. Verdict

```text
MODEL_V2_INTERNAL_FAILURE_AUDIT_ACTIONABLE
ARCHITECTURE_CHANGE_NOT_YET_JUSTIFIED
10_CHANNEL_ARCHITECTURE_STILL_PLAUSIBLE
FINAL_SENSOR_ARCHITECTURE_NOT_READY
```

Training/generation counters for this milestone are all zero:

```text
optimizer steps = 0
checkpoint writes = 0
normalizer fits = 0
HNM rounds = 0
threshold searches = 0
persistence searches = 0
architecture searches = 0
seed searches = 0
new simulation runs = 0
```

Verification result:

- `PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest`: `78 passed, 1 skipped`.
- Bare `python -m pytest` was attempted as requested but the host's incompatible autoloaded `anyio` plugin imports missing `_pytest.scope`; this is the documented environment/plugin mismatch. Disabling third-party autoload and exposing `src` runs the complete suite successfully.
- `python -m compileall src scripts tests`: PASS.
- critical Ruff `E9,F63,F7,F82`: PASS.
- `git diff --check`: PASS.
- frozen V1 and V2 artifact verification: PASS.
- V2/Unified/Generalization NPZ hash verification: `412/412`, `256/256`, `72/72` PASS; only bytes were hashed for sealed splits.
- Unified HOLDOUT remained unopened; Generalization HOLDOUT guard remained 0.

The exactly one recommended next milestone is:

```text
MODEL_V2_EXTRACTION_REBALANCE_DESIGN
```

It was not started.
