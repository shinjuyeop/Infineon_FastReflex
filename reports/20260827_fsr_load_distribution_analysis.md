# FSR Load Distribution Analysis

## Scope and provenance

`FSR_LOAD_DISTRIBUTION_ANALYSIS`는 기존 `hazard_sensor_pilot_20260827`의 virtual FSR8만 읽은 analysis-only 작업이다. Simulation, dataset 생성/수정, model training, classifier/threshold tuning은 수행하지 않았다.

- Dataset: `hazard_sensor_pilot_20260827`, 40 runs / 320,000 samples
- Manifest SHA-256: `3b15ed5412682fe6a233334caf202b271b6a8d89155bd817568d28a73f54a3e3`
- Primary statistical unit: one physical run
- Primary comparison: benign unilateral Sink 4 runs(mild 2, moderate 2) vs hazardous severe Sink 9 runs(left 5, right 4)
- Secondary control: Uniform Sand 4 runs; Concrete/Marble 각 4 runs는 일반 reference로만 사용
- Analysis source commit: `ced75afb7e6daad1ab3118f1141e0c73944791d3`
- Local artifact: `artifacts/runs/20260827_fsr_load_distribution_analysis/`

이 결과의 ratio와 load-center quantity는 observability diagnostic일 뿐 runtime input이 아니다. `cop_*_proxy`는 동일 간격의 quadrant center를 가정한 normalized load-center proxy이며 continuous physical CoP가 아니다.

## Method

Left/right unilateral transition을 affected/unaffected foot으로 canonicalize하되 affected side를 각 run row에 보존했다. Foot total이 frozen load-off 기준 `2.5 N` 미만이면 foot-internal ratio와 CoP proxy를 `NaN`으로 제외했으며 0으로 채우지 않았다. Bilateral metric에는 causal endpoint 구간의 affected/unaffected loaded-sample count를 함께 기록했다.

각 t0 patch-contact 및 t1 physical-Sink 기준 `0/20/50/100/150/200 ms`와 descriptive `300 ms` endpoint를 평가했다. Horizon 값은 future sample 없이 endpoint까지 최대 10 ms(`[H-9,H]`)의 valid-sample median이다. Delta baseline은 event `[-100,-20] ms`만 사용했다. 각 metric은 benign/hazardous run median과 range, range overlap, 방향을 보정한 run-level AUROC로 비교했다. 같은 metric의 raw/delta는 전체 표에 모두 보존하되 Top 5에는 metric당 하나만 포함했다.

## Horizon results

### t0 — first soft-patch contact

| Horizon | Best metric | Form | Benign range | Hazardous range | AUROC | Side AUROC (L/R) | Reading |
|---:|---|---|---:|---:|---:|---:|---|
| 20 ms | `cop_x_proxy` | raw | -0.466–-0.392 | -0.642–0.064 | 0.833 | 1.000 / 0.750 | overlap; transient candidate |
| 50 ms | `front_left_share` | raw | 0.125–0.164 | 0.058–0.289 | 0.667 | 0.600 / 0.750 | weak, overlapping |
| 100 ms | `bilateral_total_n` | delta | 26.504–42.847 N | 20.484–142.223 N | 0.667 | 0.600 / 0.750 | weak, overlapping absolute-load change |
| 150 ms | `front_right_share` | raw | 0.088–0.178 | 0.062–0.171 | 0.750 | 0.800 / 0.875 | overlap |
| 200 ms | `rear_left_share` | raw | 0.230–0.395 | 0.240–0.522 | 0.806 | 1.000 / 0.750 | late and overlapping |

t0+20의 front/rear-related CoP 신호는 다음 50/100 ms horizon에서 재현되지 않고 모든 range가 겹쳤다. 따라서 t0 기준 100 ms 이내에 의미 있고 반복되는 separation은 없다. Severe run의 t1은 t0 뒤 30–47 ms에 발생했으므로 t1+100 observation은 사용자 관점에서 대략 t0+130–147 ms이지만, transient gait trajectory 때문에 이를 t0+150 통계와 동일한 결과로 간주하지 않는다.

### t1 — patch-linked physical Sink onset

| Horizon | Best metric | Form | Benign range | Hazardous range | AUROC | Side AUROC (L/R) | Reading |
|---:|---|---|---:|---:|---:|---:|---|
| 20 ms | `load_concentration` | delta | -0.253–-0.158 | -0.379–-0.151 | 0.786 | 0.750 / 1.000 | overlap; hazardous delta 7/9 valid |
| 50 ms | `medial_ratio` | delta | 0.324–0.437 | 0.336–0.646 | 0.893 | 0.875 / 1.000 | first useful distribution candidate, still overlap; 7/9 valid |
| 100 ms | `affected_total_n` | delta | 329.269–332.328 N | 333.100–365.664 N | 1.000 | 1.000 / 1.000 | no overlap; `PILOT_SEPARATION_CANDIDATE` |
| 150 ms | `front_right_share` | raw | 0.096–0.186 | 0.045–0.173 | 0.778 | 0.800 / 1.000 | overlap |
| 200 ms | `cop_radius_proxy` | raw | 0.110–0.648 | 0.261–0.763 | 0.778 | 0.800 / 1.000 | overlap |

t1+50의 medial delta가 earliest meaningful descriptive candidate다. 그러나 clear Pilot separation은 t1+100의 absolute affected load에서 처음 나타난다. 이때 unaffected foot은 모든 primary run에서 unloaded여서 `bilateral_total_n`과 `affected_total_n`은 독립적인 두 신호가 아니라 같은 single-support load를 나타낸다. 완전 분리는 작은 4-vs-9 Pilot의 `PILOT_SEPARATION_CANDIDATE`일 뿐 threshold 해결을 뜻하지 않는다.

## Physical-signal audit at t1+100 ms

각 family에서 raw/delta 중 AUROC가 높은 form을 표시했다. 유리한 form만으로 결론을 만들지 않도록 나머지 form은 local `horizon_separation.csv`에 보존했다.

| Signal | Form | Benign range | Hazardous range | AUROC | Side AUROC (L/R) | Interpretation |
|---|---|---:|---:|---:|---:|---|
| affected total load | raw/delta | 329.269–332.328 N | 333.100–365.664 N | 1.000 | 1.000 / 1.000 | strongest, but absolute and terrain-overlapping |
| bilateral asymmetry | raw | 1.000–1.000 | 1.000–1.000 | 0.500 | 0.500 / 0.500 | single support indicator, not severity |
| affected load share | raw | 1.000–1.000 | 1.000–1.000 | 0.500 | 0.500 / 0.500 | no separation |
| front/rear (`front_ratio`) | raw | 0.174–0.340 | 0.156–0.345 | 0.583 | 0.500 / 0.750 | no useful separation |
| medial/lateral (`medial_ratio`) | delta | 0.380–0.474 | 0.400–0.649 | 0.929 | 1.000 / 1.000 | strongest distribution candidate; ranges overlap, 7 hazardous deltas valid |
| CoP (`cop_radius_proxy`) | delta | -0.334–0.034 | -0.780–0.073 | 0.714 | 0.750 / 0.667 | moderate, overlapping and phase-dependent |
| concentration | delta | -0.160–-0.147 | -0.370–-0.049 | 0.714 | 0.500 / 1.000 | inconsistent between sides |

## Controls, progression, and shortcut audit

Uniform Sand t1+100 worst-case pseudo-foot의 affected total range는 `341.715–344.566 N`으로 hazardous severe `333.100–365.664 N`과 겹쳤다(AUROC 0.778). 따라서 strongest absolute-load candidate는 hazard-specific하지 않다. 반면 raw medial ratio는 Uniform Sand `0.311–0.433`과 hazardous severe `0.555–0.714`를 이 Pilot에서 겹침 없이 분리했다(AUROC 1.000). 이는 late distribution evidence를 지지하지만, t1+50의 Uniform Sand comparison은 다시 겹쳤고 작은 control set이므로 threshold로 채택할 수 없다.

일반 terrain reference에서 run-median bilateral total의 terrain별 median은 Concrete `324.265 N`, Marble `324.023 N`, Uniform Sand `331.790 N`이었다. Bilateral asymmetry와 max-foot share의 run median은 세 terrain 모두 1.0으로 single-support gait phase를 주로 반영했다. Concrete/Marble은 event-matched primary control이 아니므로 Sink conclusion에는 사용하지 않았다.

t1+100에서 affected total의 severity median은 mild `329.943`, moderate `331.655`, severe `338.538 N`이었고 medial-ratio delta도 `0.426 → 0.431 → 0.598`로 monotonic했다. CoP-radius delta도 hazardous 방향으로 monotonic했지만 front-right share는 그렇지 않았다. Absolute load와 medial ratio는 left/right severe 양쪽에서 같은 분리 방향을 보였다.

Shortcut 가능성은 남는다. Severe speed Spearman correlation은 affected total `-0.367`, medial-ratio delta `0.591`, CoP-radius delta `-0.610`이었고, t0 contact-phase correlation은 각각 `0.014`, `0.199`, `-0.823`이었다. 특히 CoP 및 일부 quadrant 신호는 speed/gait phase 의존 가능성이 크다. 표본과 speed/phase 배치가 작고 불균형하므로 correlation은 descriptive audit일 뿐 유의성 검정이 아니다.

Observed benign envelope diagnostic도 horizon separation을 detector로 바로 바꿀 수 없음을 보여준다. Positive-margin severe 7 runs에서 t1+100 Top 5 중 `bilateral_total_n` 한 candidate만 한 run에서 t1+838 ms에 mild/moderate observed envelope를 t2 전에 10 ms 지속 이탈했다. 나머지 candidate/run에는 departure가 없었고 Uniform Sand를 추가한 broader envelope에서도 같았다. Horizon-specific 완전 분리와 phase-agnostic fixed threshold는 서로 다른 주장이다.

## Required questions

1. **Total force 자체는 분리하는가?** t1+100의 affected/bilateral total은 현재 4-vs-9 Pilot에서 완전 분리하지만 Uniform Sand와 겹치므로 hazard-specific하지 않다.
2. **Foot-internal distribution이 더 잘 분리하는가?** t1+50/+100 medial-ratio delta는 유용하나 absolute total보다 완전하지 않다. 다만 t1+100에서 Uniform Sand와의 구분은 total보다 낫다.
3. **Bilateral asymmetry는 유효한가?** 아니다. t1+100에 두 group 모두 1.0이며 single support를 나타낸다.
4. **Front/rear 변화가 있는가?** t0+20에 transient 신호가 있으나 50/100 ms에 유지되지 않고 t1+100도 약하다.
5. **Medial/lateral 변화가 있는가?** 있다. t1+50 AUROC 0.893, t1+100 delta AUROC 0.929로 가장 일관된 internal-distribution candidate다.
6. **CoP proxy가 유효한가?** t0+20에는 보이지만 반복되지 않는다. t1+100도 AUROC 0.714이고 phase correlation이 커 단독 근거로 부족하다.
7. **Load concentration이 유효한가?** t1+20의 earliest candidate지만 overlap, missing delta, side inconsistency 때문에 제한적이다.
8. **t0 earliest descriptive separation은 언제인가?** +20 ms에 transient CoP/front signal이 있으나 meaningful repeated separation은 100 ms 이내 없다.
9. **t1 earliest descriptive separation은 언제인가?** +20 ms concentration은 약한 후보, +50 ms medial ratio가 earliest meaningful candidate, +100 ms가 earliest clear Pilot separation이다.
10. **20/50/100 ms 중 실질적 후보는?** t1+100 ms가 가장 현실적이다. +50 ms medial ratio는 후속 검증 후보로 보존한다.
11. **Severity progression이 있는가?** affected total, medial-ratio delta, CoP-radius delta에는 monotonic median progression이 있으나 모든 distribution metric에 공통되지는 않는다.
12. **Uniform Sand와도 구분되는가?** absolute total은 아니며, t1+100 raw medial ratio는 현재 Pilot에서 구분된다.
13. **Left/right severe 모두 일관적인가?** affected total과 medial ratio는 양쪽에서 일관적이다. concentration과 일부 CoP/quadrant metric은 그렇지 않다.
14. **Speed/gait-phase shortcut 가능성이 있는가?** 있다. 특히 CoP/quadrant 계열 correlation이 커 더 넓고 균형 잡힌 조건이 필요하다.
15. **FSR distribution hypothesis를 지지하는가?** late physical-Sink 이후에는 제한적으로 지지하지만 ultra-fast detection 근거는 약하다.

## Conclusion

`FSR_LOAD_DISTRIBUTION_LATE_SEPARATION_ONLY`

FSR distribution에는 t1+50부터 descriptive evidence가 있고 t1+100 medial ratio는 Uniform Sand보다 hazard-specific한 모습을 보였다. 그러나 t0+20 신호는 재현되지 않았고, t1+100의 가장 강한 분리는 distribution이 아니라 absolute single-foot load이며, broader benign envelope의 pre-t2 sustained departure도 없었다. 따라서 20/50/100 ms 중 현실적인 observation point는 **t1+100 ms**이고 ultra-fast distribution detector를 지지하지 않는다.

이 결과는 raw FSR MLP가 early Sink에서 실패한 이유를 부분적으로 설명한다. 초기 raw load는 touchdown/terrain scale과 겹치고, 유용한 medial redistribution은 더 늦으며 baseline-relative representation과 low-load handling이 필요하다. 다만 classifier failure의 단일 원인으로 확정할 수 없다. Virtual FSR Pilot-only 결과이므로 sensor architecture, handcrafted runtime feature, threshold, detector는 모두 unfrozen이다. 다음 classifier training, Full Dataset, E84 작업은 이 분석에 포함하지 않는다.
