# FSR Temporal Redistribution Analysis

## Scope

`FSR_TEMPORAL_REDISTRIBUTION_ANALYSIS`는 기존 `hazard_sensor_pilot_20260827`의 virtual FSR8만 읽은 analysis-only 작업이다. Simulation, dataset/NPZ/label 변경, model training, classifier와 threshold tuning은 수행하지 않았다.

- Manifest SHA-256: `3b15ed5412682fe6a233334caf202b271b6a8d89155bd817568d28a73f54a3e3`
- Primary: benign unilateral Sink 4 runs(mild 2, moderate 2) vs hazardous severe Sink 9 runs(left 5, right 4)
- Secondary: Uniform Sand 4 runs
- Statistical unit: one physical run
- Static parity: 이전 static artifact의 t0/t1 `20/50/100 ms` Top-1 metric, representation, AUROC exact match
- Local artifact: `artifacts/runs/20260827_fsr_temporal_redistribution_analysis/`

FSR-derived temporal quantities는 observability diagnostic이며 runtime feature가 아니다. CoP quantity는 quadrant center를 가정한 normalized load-center proxy이지 continuous physical CoP가 아니다.

## Method

Affected foot FSR4를 left/right Sink 모두 같은 affected-foot representation으로 canonicalize했다. Foot total이 frozen load-off threshold `2.5 N` 미만인 sample은 distribution에서 `NaN`으로 제외했다. Event initial은 `[event,event+9 ms]`, endpoint는 `[H-9,H]`의 valid distribution median이며, H가 짧으면 endpoint를 넘지 않게 window를 clip했다. Path는 median endpoint를 연결하지 않고 raw 1 ms trajectory의 adjacent valid pair만 누적했으며 invalid gap을 건너뛰지 않았다.

모든 temporal interval은 `[event,event+H]`로 causal하다. Event 전 instability baseline은 `[-100,-20] ms`만 사용했고 coverage가 부족하면 excess metric을 unavailable로 보존했다. Hazardous horizon에 t2가 들어오면 per-run row에 flag했고, 해당 row를 제외한 4 benign vs 7 positive-margin severe sensitivity도 별도 artifact로 저장했다.

## Static versus temporal

### t0 — first soft-patch contact

| Horizon | Best static | AUROC | Best temporal | Benign range | Hazardous range | AUROC | Overlap | Side AUC L/R |
|---:|---|---:|---|---:|---:|---:|---|---:|
| 20 ms | `cop_x_proxy` raw | 0.833 | `concentration_path` | 0.117–0.668 | 0.125–0.749 | 0.778 | yes | 0.800 / 1.000 |
| 50 ms | `front_left_share` raw | 0.667 | `front_delta` | -0.754–0.026 | -0.767–0.378 | 0.722 | yes | 0.600 / 0.750 |
| 100 ms | `bilateral_total_n` delta | 0.667 | `cop_path_efficiency` | 0.617–0.676 | 0.129–0.859 | 0.611 | yes | 0.800 / 0.500 |

t0+50에서 temporal AUROC가 static보다 0.056 높았지만 range가 크게 겹치고 양쪽 일관성도 약했다. +20과 +100에서는 temporal이 static보다 낮았다. 따라서 지면을 처음 밟은 뒤 100 ms 내 usable temporal separation은 없다. Uniform Sand에는 comparable finite-patch t0가 없으므로 인위적인 t0 alignment를 만들지 않았다.

### t1 — patch-linked physical Sink onset

| Horizon | Best static | AUROC | Best temporal | Benign range | Hazardous range | AUROC | Overlap | Side AUC L/R |
|---:|---|---:|---|---:|---:|---:|---|---:|
| 20 ms | `load_concentration` delta | 0.786 | `front_abs_change` | 0.0270–0.0334 | 0.0083–0.0189 | 1.000 | no | 1.000 / 1.000 |
| 50 ms | `medial_ratio` delta | 0.893 | `cop_path_efficiency` | 0.806–0.843 | 0.319–0.841 | 0.861 | yes | 0.800 / 1.000 |
| 100 ms | `affected_total_n` delta | 1.000 | `cop_path_length` | 0.587–0.626 | 0.644–1.030 | 1.000 | no | 1.000 / 1.000 |

t1+20 front absolute change는 현재 Pilot에서 완전 분리되고 Uniform Sand와도 겹치지 않았다. 그러나 같은 metric은 +30 ms AUROC 0.639, +50 ms 0.750, +100 ms 0.611로 즉시 약해졌다. +50 ms의 best temporal도 이전 static medial ratio보다 낮고 overlap이 남는다. +100 ms CoP path는 clear separation이지만 static보다 빠르지 않다.

Zero-margin severe 2 runs는 모든 t1 horizon에 t2 sample이 들어간다. 이를 제외한 pre-t2 sensitivity에서도 front absolute change t1+20은 AUROC 1.000(no overlap, L/R 1.000/1.000), CoP path t1+100도 AUROC 1.000(no overlap, L/R 1.000/1.000)이었다. 따라서 두 현상은 t2 contamination만으로 생기지는 않았지만, transient/terrain-specificity limitation은 그대로다.

## Temporal signal audit

Main table은 각 physical family의 20/50/100 ms 중 가장 유용한 t1 result를 보인다. Horizon을 함께 적어 서로 다른 시점을 같은 feature처럼 해석하지 않는다.

| Signal | Horizon | Benign range | Hazardous range | AUROC | Overlap | Primary-direction side AUC L/R | Uniform Sand audit |
|---|---:|---:|---:|---:|---|---:|---|
| `front_abs_change` | 20 | 0.0270–0.0334 | 0.0083–0.0189 | 1.000 | no | 1.000 / 1.000 | AUROC 1.000, no overlap |
| `quadrant_l1_change` | 20 | 0.198–0.259 | 0.073–0.246 | 0.944 | yes | 1.000 / 1.000 | AUROC 0.611, overlap |
| `entropy_abs_change` | 20 | 0.034–0.065 | 0.002–0.059 | 0.944 | yes | 1.000 / 1.000 | AUROC 0.889, overlap |
| `cop_displacement` | 20 | 0.210–0.224 | 0.052–0.223 | 0.944 | yes | 1.000 / 1.000 | AUROC 0.889, overlap |
| `cop_path_length` | 100 | 0.587–0.626 | 0.644–1.030 | 1.000 | no | 1.000 / 1.000 | primary-direction AUROC 0.000; Sand path가 더 큼 |

Other declared families:

- `quadrant_path_length`: t1+50 AUROC 0.722, benign `0.623–0.731`, hazardous `0.261–1.542`, overlap, side 0.900/0.500. 지속적 jitter는 early side-consistent signal이 아니다.
- `concentration_abs_change`: t1+20 AUROC 0.889, benign `0.0216–0.0568`, hazardous `0.0013–0.0539`, overlap, 양쪽 1.000/1.000.
- `max_share_abs_change`: t1+20 AUROC 0.833, benign `0.036–0.129`, hazardous `0.012–0.123`, overlap, 양쪽 1.000/1.000.
- `medial_abs_change`: t1+20 AUROC 0.889, benign `0.0996–0.1087`, hazardous `0.0249–0.1097`, overlap, side 1.000/0.750. t1+100 pre-t2 `medial_path`는 완전 분리하지만 static medial evidence보다 이르지 않다.

## Net redistribution versus jitter

Early t1+20의 상위 신호는 path/jitter가 아니라 net/absolute change였다. 더 중요한 점은 hazardous severe의 `front_abs_change`, `quadrant_l1_change`, `entropy_abs_change`, `cop_displacement`가 benign보다 대체로 **작았다**는 것이다. 즉 위험한 Sink가 즉시 더 크게/불안정하게 재배치된다는 가설보다, onset 직후 load distribution movement가 잠시 억제되는 transient pattern에 가깝다.

Jitter/path는 늦게 커졌다. t1+100 CoP path는 severe에서 더 컸지만 Uniform Sand range `3.609–5.882`가 severe `0.644–1.030`보다 훨씬 컸다. 같은 방향의 threshold라면 Uniform Sand가 더 hazardous-looking하다. General 100 ms reference window에서도 terrain별 median CoP path rate는 Concrete `0.0077`, Marble `0.0093`, Sand `0.0208/ms`, quadrant path rate는 `0.0079`, `0.0158`, `0.0250/ms`였다. 따라서 path magnitude는 strong terrain/contact-dynamics shortcut 가능성이 있다.

## Severity, side, and metadata audit

t1+20 front absolute change median은 mild `0.0305`, moderate `0.0293`, severe `0.0161`로 hazardous 방향(lower) monotonic이었다. Quadrant L1과 entropy absolute change도 같은 lower 방향 progression을 보였다. t1+100 CoP path는 `0.588 → 0.619 → 0.738`로 higher 방향 monotonic이었다. 모든 metric에 공통된 progression은 아니다.

Top t1+20 front signal의 severe speed Spearman correlation은 `-0.555`, contact phase `-0.037`, contact age `0.138`이었다. Quadrant L1은 각각 `-0.081`, `0.288`, `0.152`로 speed dependence가 작았지만 range overlap이 남았다. CoP displacement는 speed `-0.635`, t1+100 CoP path는 contact age `-0.588`로 shortcut 우려가 더 크다. 이 correlation은 작은 불균형 Pilot의 descriptive audit이며 유의성 검정이 아니다.

## Required questions

1. **Static보다 temporal이 더 좋은가?** 전체적으로 아니다. t1+20 한 지점은 좋아졌지만 +30에서 재현되지 않았고 +50은 static보다 낮으며 +100은 동률이다.
2. **t0 후 20/50/100 ms usable signal이 있는가?** 없다. 모두 overlap이고 AUROC 0.778/0.722/0.611이다.
3. **t1 후 20/50/100 ms에는?** +20 transient front candidate, +50 overlapping efficiency candidate, +100 clear CoP-path candidate가 있다.
4. **Best direction-independent metric은?** 단일 horizon 기준 t1+20 `front_abs_change`; foot-axis-independent family에서는 `quadrant_l1_change`/entropy absolute change가 AUROC 0.944지만 overlap한다.
5. **Net redistribution과 jitter 중 무엇이 중요한가?** Early에는 smaller net change, late에는 path 증가다. Early unstable jitter 가설은 지지되지 않는다.
6. **Concentration/entropy는 의미가 있는가?** Absolute change는 t1+20에 descriptive evidence가 있지만 path는 약하고 overlap/terrain confound가 남는다.
7. **CoP path는 의미가 있는가?** t1+100에는 강하지만 Uniform Sand가 훨씬 더 커 hazard-specific하지 않다.
8. **Static medial보다 일반화 가능해 보이는 metric이 있는가?** Early direction-independent 후보는 있으나 지속성/Uniform Sand 조건까지 만족해 static medial보다 낫다고 할 metric은 없다.
9. **Severity progression이 있는가?** Top front/quadrant/entropy net change와 late CoP path에는 monotonic median progression이 있다.
10. **Uniform Sand에서도 동일 신호가 나타나는가?** Early front suppression은 나타나지 않지만 path/jitter는 severe보다 더 크게 나타난다.
11. **Left/right 모두 일관적인가?** t1+20 front/quadrant/entropy와 t1+100 CoP path는 양쪽에서 일관적이다. Earlier path accumulation은 side consistency가 약하다.
12. **Speed/gait-phase shortcut 가능성이 있는가?** 있다. 특히 CoP displacement/path와 general terrain path rate가 speed/contact age/terrain에 의존할 가능성이 크다.
13. **Earliest meaningful separation이 앞당겨졌는가?** Nominal one-horizon candidate는 static t1+50에서 temporal t1+20으로 30 ms 빨라졌지만 반복성 기준을 충족하지 못한다. Criteria-based meaningful separation은 앞당겨지지 않았다.

## Conclusion

`FSR_TEMPORAL_REDISTRIBUTION_NO_ADDED_VALUE`

현재 Pilot에는 t1+20의 striking transient와 t1+100 path separation이 존재한다. 그러나 early transient는 다음 horizon에서 유지되지 않고, late jitter는 Uniform Sand에서 더 강하며, t1+50 temporal best는 기존 static medial ratio보다 낮다. 따라서 “hazardous Sink의 방향 독립적 급격한/불안정 redistribution이 static distribution보다 일반적인 early signal”이라는 가설에는 meaningful added value가 없다.

현실적인 temporal observation point를 하나 고르면 clear separation이 나타나는 **t1+100 ms**지만, 이는 static보다 빠르지 않고 hazard-specific threshold도 아니다. Temporal representation, detector, threshold와 sensor architecture는 모두 unfrozen이다. 다음 model training, Full Dataset, E84 작업은 수행하지 않는다.
