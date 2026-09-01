# Model V2 Dataset Design

## 1. Purpose

This milestone freezes the fresh data coverage, split, physical semantics, exclusions, and future evaluation contract for the first Hazard Model V2 experiment. It does not generate `model_v2_hazard_reflex_20260901`, fit a normalizer, train a model, mine negatives, or inspect either protected HOLDOUT.

The design verdict is `MODEL_V2_DATASET_DESIGN_READY`. The first V2 experiment is a `DATA_ONLY_INTERVENTION`: it changes training coverage and necessarily learned weights/TRAIN normalization while preserving the V1 sensor, causal feature, GRU, history, ensemble, threshold, and persistence contracts.

## 2. Model V1 preservation

Starting `HEAD` and `origin/main` were both `1451b96ac4ff3307078682c5468a6e481007d987` (`Study Ice near-hazard target semantics`), on `main`, with a clean tracked worktree. Read-only verification established `MODEL_V1_RESTORABLE = YES`.

| Protected item | Frozen identity | Verification |
|---|---|---|
| Hazard freeze | `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2` | exact canonical identity |
| Hazard feature schema | `fe5b6c1c5eca8207a01c62e156f1fe843f95f0c5001d179a12c4b2b16ddf8adb` | exact |
| Hazard normalizer | `610a91c24a594a3cd1701b07549f4f3c4b7f4cb04eb354f9d41378214a6800a9` | exact file hash |
| Hazard checkpoints | `e6bada49…`, `b04877dc…`, `b6c782bd…` | 3/3 exact file hashes |
| Terrain normalizer | `2c551181fbd23080c0e3008be8484bd0ee4139e594041ba6e46ab5b36ffb05de` | exact file hash |
| Terrain checkpoints | `21b0d122…`, `de6a55d3…`, `465803f4…` | 3/3 exact file hashes |

Hazard V1 remains Pelvis IMU6 at 1 kHz → causal 80D → `[20,80]` → one-layer unidirectional GRU hidden 32 → three-seed mean → threshold `0.99` for `5 ms`, with 11,010 parameters. Terrain V1 remains left-touchdown FSR4 → 50 ms MLP → three-seed advisory-only Terrain. No V1 source or artifact was modified or duplicated.

The Unified manifest remains `d023384aaaac22076a2be7c2fa242acac91a548d82102640c39868990cb0e9d6` with 256/256 NPZ byte hashes valid. The Generalization manifest remains `72f5dd300bb5be78b22b9a8c1ad9788cdf45e7cda9b4f46d79eb290140146f53` with 72/72 NPZ byte hashes valid. Calibration 78/78, Ice-resolution 48/48, and Ice-semantics 48/48 hashes also passed.

## 3. Historical baseline and failure

Historical results are not pooled, rescored, or rewritten.

| Evidence | Hazard | Slip | Support | No-hazard | Premature | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Unified fresh HOLDOUT | 26/26 | 13/13 | 13/13 | 26/26 | 0/26 | `UNIFIED_HAZARD_REFLEX_SUPPORTED_SINGLE_IMU` |
| Generalization VALIDATION | 13/26 | 7/12 | 6/14 | 5/10 | 7/26 | `ZERO_RETRAIN_GENERALIZATION_NOT_SUPPORTED` |

Generalization VALIDATION also retained Ice-benign specificity 3/4, Slip valid-latency p95 `+5.3 ms`, and Support established-latency p95 `-17.25 ms`. Those numbers remain the V1 development baseline.

## 4. Evidence boundary

Authorized design evidence is Unified TRAIN, previously authorized Unified VALIDATION, Generalization VALIDATION, the failure-mode audit, calibration pilots, Ice-resolution pilots, the Ice semantics DISCOVERY/CONFIRMATION physical results, and model-blind scenario metadata. Pilot and semantics corpora are context and signature exclusions; they are not automatically V2 TRAIN.

Generalization VALIDATION is external development evidence. It is explicitly excluded from training, normalization, HNM, and final-fresh claims. Generalization HOLDOUT contains 36 runs and remains sealed at guard count 0; only run IDs, membership, existence, hashes, and count were accessed. Unified HOLDOUT was not reopened and received no new inference.

## 5. V2 data-only hypothesis

The primary hypothesis is:

> Expanding TRAIN coverage across the localized missing physical scenario families, while preserving Pelvis IMU6 / causal 80D / GRU20, will materially improve generalization without a larger architecture or new Hazard sensors.

The controlled comparison keeps V1 architecture, feature schema, 20 ms history, hidden size 32, one unidirectional layer, three-seed ensemble concept, threshold `0.99`, and persistence `5 ms`. Only data coverage, the new effective-TRAIN normalizer, and learned weights change. Threshold/persistence calibration is reserved for a separate later milestone and cannot rescue the primary data-only comparison.

## 6. Frozen target semantics

`PRIMARY_HAZARD` remains established Slip OR established Support.

- Established Slip: touchdown-anchor tangential drift `>=50 mm` for 3 ms, any foot.
- Established Support: support-surface spread `>=10 mm` for 20 ms with contact-episode reset.
- I1: privileged positive loaded-foot spread-derivative reference for 20 ms; it defines the earliest valid Support alert boundary and is not a runtime input.
- Terrain, design intent, fall, and censor do not create a Hazard label. Actual physical outcome overrides intended scenario role.

Required stored annotations are `primary_hazard_target`, `ice_precursor_candidate`, `ice_precursor_future_outcome`, and `ice_precursor_censored`. The primary binary timeline is not rewritten.

## 7. Ice precursor handling

`ICE_PRECURSOR_CANDIDATE` is loaded exact-Ice drift in `[30,50) mm` before established Slip. It is distinct from established Hazard. No velocity threshold or phase-dependent production rule is added.

| Episode outcome | Primary timeline | Precursor interpretation | Negative/HNM handling |
|---|---|---|---|
| Same/next/later future Slip | unchanged established-Hazard timeline | candidate with future outcome | mask from ordinary negative sampling and never HNM as negative |
| Fully observed benign release | primary negative | precursor-target negative | eligible verified hard negative after full 1 s follow-up |
| Censored future | unchanged but ambiguous | `precursor_censored=true` | not a confirmed negative; exclude or predeclare down-weighting |

The first V2 primary model remains the original binary Hazard model. Precursor-local windows are retained as a separately annotated pool and an accepted-early secondary evaluation region. A multi-head architecture is not introduced in the first comparison.

## 8. Failure-to-data mapping

| V1 failure | Root cause | V2 data intervention | Planned family |
|---|---|---|---|
| Delayed Ice premature | delayed/multi-contact Ice absent plus target-boundary tension | exactly-one and multi-contact delayed Ice with episode/future annotation | `ONE_CONTACT_DELAYED_ICE_SLIP` |
| Ice-benign false positive | benign/near-slip Ice controls absent | narrow-domain no-Hazard Ice plus fully observed near-threshold release intents | `ICE_BENIGN_CONTROL`, `ICE_NEAR_HAZARD_PRECURSOR` |
| Delayed Sand pre-I1 false alert | static staged entry absent from negatives | explicit no-I1/no-Support staged entry controls; keep delayed positive pre-I1 interval negative | `STAGED_SAND_BENIGN_CONTROL`, `DELAYED_SAND_SUPPORT_ONSET` |
| Right Support 0/4 | no right-only Support TRAIN and unlearned signed mirroring | meaningful matched right Support grid with retained left reference | `RIGHT_SAND_SUPPORT_SPEED_MATRIX`, `LEFT_SAND_SUPPORT_SPEED_MATRIX` |
| 0.20/0.30 Slip degradation | all original Hazard TRAIN at 0.25 m/s | source-balanced Slip at all three speeds | `BASELINE_IMMEDIATE_ICE_SLIP_SPEED_MATRIX`, `ONE_CONTACT_DELAYED_ICE_SLIP` |
| Sand-benign false alerts | transition sensitivity plus speed gap | mild Sand controls across source, speed, and entry topology plus staged hard negatives | `SPEED_STRATIFIED_SAND_BENIGN`, `STAGED_SAND_BENIGN_CONTROL` |

## 9. Scenario-family definitions

Ten physical matrix families cover all thirteen required semantic roles without duplicating separate runs merely to rename the same axis.

| Family | Scientific purpose | Hazard role | Source → target | Speed domain | Designed topology | Actual-outcome rule |
|---|---|---|---|---|---|---|
| `HARD_GROUND_NORMAL_SPEED_MATRIX` | baseline hard control and local speed regression | no-Hazard intent | C→C, M→M | three local strata | N/A | unexpected Hazard retained/relabeled |
| `BASELINE_IMMEDIATE_ICE_SLIP_SPEED_MATRIX` | immediate Slip retention + speed Slip | Slip intent | C/M→Ice | .20/.25/.30 | full-width bilateral | natural actual side retained |
| `ONE_CONTACT_DELAYED_ICE_SLIP` | complete benign episode → distinct later Slip | Slip intent | C/M→Ice | .20/.25/.30 | full-width bilateral | exactly-one/multi intent audited; actual relation controls label |
| `ICE_BENIGN_CONTROL` | true narrow-domain Ice hard negative | no-Hazard intent | C/M→Ice | .20/.30 | full-width bilateral | accidental Slip retained/relabeled |
| `ICE_NEAR_HAZARD_PRECURSOR` | 30–50 mm future-Slip and benign-release contrast | mixed | C/M→Ice | .20/.25/.30 | full-width bilateral | actual future outcome overrides intent |
| `LEFT_SAND_SUPPORT_SPEED_MATRIX` | baseline/left/speed Support | Support intent | C/M→Sand | .20/.25/.30 | left | actual side retained |
| `RIGHT_SAND_SUPPORT_SPEED_MATRIX` | right Support and side-mirroring coverage | Support intent | C/M→Sand | .20/.25/.30 | right | actual side retained |
| `DELAYED_SAND_SUPPORT_ONSET` | benign interval → I1 → Support | Support intent | C/M→Sand | .25 only | staged left | pre-I1 remains negative |
| `STAGED_SAND_BENIGN_CONTROL` | static-entry hard negative | no-Hazard intent | C/M→Sand | .20/.25/.30 | staged left | unexpected Hazard retained/relabeled |
| `SPEED_STRATIFIED_SAND_BENIGN` | Sand benign across speeds and entry sides | no-Hazard intent | C/M→Sand | .20/.25/.30 | left/right | unexpected Hazard retained/relabeled |

The delayed Sand positive remains at `.25 m/s` because that staged timing relation was calibrated only there. Ice benign omits `.25 m/s` because that narrow cell often progressed to Slip. These are evidence-based limitations, not missing cells hidden by forced physics.

## 10. Source-terrain coverage

Every family expands equally over Concrete and Marble. Fresh V2 is exactly 206/206 by source: TRAIN 155/155 and V2_VALIDATION 51/51. Difficult Ice, right Support, staged Sand, and speed-control cells are not concentrated in one source.

Target terrain is Ice for three Slip/precursor families and the Ice control; Sand for the Support/Sand-control families; and the same hard terrain as source for the hard controls.

## 11. Speed coverage

Nominal fresh counts are `.20: 136`, `.25: 140`, `.30: 136`; TRAIN is 102/106/102 and V2_VALIDATION is 34/34/34. All non-hard families use the exact command speeds. Hard controls use three fresh command values per TRAIN stratum and one disjoint validation value:

```text
nominal .20: TRAIN .194/.200/.206; VALIDATION .209
nominal .25: TRAIN .244/.250/.256; VALIDATION .259
nominal .30: TRAIN .294/.300/.306; VALIDATION .309
```

In the effective Hazard TRAIN intent, exact-speed counts are `.20: 58`, `.25: 152`, `.30: 58`; `.25` is 56.7%, safely below the prohibited >80% dominance.

## 12. Side coverage

Support has 36 direct left and 36 direct right fresh TRAIN runs at matched sources/speeds/counts. Delayed Support adds 18 left-intent runs, yielding fresh Support intent 54 left / 36 right; the extra left count represents a distinct delayed mechanism rather than failure to balance the matched ordinary families.

Ice uses full-width mechanics and accepts natural left/right/bilateral outcomes. Right-only Ice is not required because calibration could not reliably create it. Matrix side columns below are designed physical topology, not a promise about the actual affected foot. The future manifest must store actual side and must not move or replace runs after outcome inspection.

## 13. Hard-negative design

The design makes known failures explicit instead of delegating them to future HNM:

- 24 fresh hard-ground normal controls;
- 32 narrow-domain Ice benign controls;
- 8 designed benign-release precursor contrasts at the demonstrated `.20/.30 m/s` narrow domain, with actual future outcome recorded;
- 48 staged static-Sand controls using the previously non-Support wide staged regime;
- 48 speed-stratified mild Sand controls split across left/right entry topology.

Static staged Sand remains negative until I1. A future positive in the same run does not make its pre-I1 static-entry interval positive. Fully observed benign 30–50 mm Ice release is a legitimate primary/precursor negative; future-Slip and censored precursor regions are not.

## 14. Positive-event design

Fresh TRAIN has 102 designed Slip and 90 designed Support runs. Slip contains immediate, exactly-one/multi-contact delayed, and future-Slip precursor-intent trajectories at all three speeds and both sources. Support contains matched left/right ordinary cases at all speeds plus `.25 m/s` delayed onset.

No single fresh family supplies most positives: each 36-run TRAIN matrix is 11.6% of fresh TRAIN and 7.8% of the effective pool; delayed Support contributes 18 runs. Actual oracle counts are recomputed after generation without deletion, split moves, or class-balance backfill.

## 15. Planned split

The only internal splits are `V2_TRAIN` and `V2_VALIDATION`; there is no internal V2 HOLDOUT. All 412 membership cells are frozen before simulation.

Run IDs follow:

```text
m2v2_{family_code}_{t|v}_{c|m}_{four_digit_round(1000*speed)}_{cell_id}
```

Expansion order is family → split → source → cell. Outcome and model output cannot change membership. V2_VALIDATION contains every family, both sources, every supported nominal speed, direct left/right Support, both delayed-Ice episode intents, both precursor outcome intents, and each hard-negative type. Validation geometries/commands are disjoint boundary or alternate cells, not copied TRAIN signatures.

## 16. Fresh-signature exclusion

The 9-field signature is `(source, target, speed, start, width, slip_pattern, sink_pattern, severity, support_pattern)`. All 12 available raw-data manifests are hash-pinned, including Unified 256, Generalization 72, calibration 78, Ice-resolution 48, Ice-semantics 48, and the historical manifests referenced by the Unified contract.

The exact config expansion was audited before generation:

| Audit | Result |
|---|---:|
| Planned run IDs | 412 unique / 412 |
| Planned physical signatures | 412 unique / 412 |
| V2_TRAIN / V2_VALIDATION exact overlap | 0 |
| Exact overlap with available canonical 9-field prior signatures | 0 |
| Cross-split near duplicates under the declared tolerances | 0 |

Broad transitions use exclusive proximity thresholds of 2 mm start and 4 mm width; the narrow Ice block uses 1 mm/1 mm; hard controls use 0.003 m/s command speed. Intentional narrow Ice contrasts are permitted only within the same split. Any manifest appearing before generation must be added and re-audited. Waveform similarity may never drive reassignment.

## 17. Planned run counts

The side columns count designed topology; `N/A` is used when side is not physically meaningful. Speed columns are nominal strata, with the hard-control command offsets described above.

| Family | TRAIN | V2_VALIDATION | Total | Concrete | Marble | 0.20 | 0.25 | 0.30 | Left | Right | Bilateral |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hard normal speed matrix | 18 | 6 | 24 | 12 | 12 | 8 | 8 | 8 | N/A | N/A | N/A |
| Immediate Ice Slip speed matrix | 36 | 12 | 48 | 24 | 24 | 16 | 16 | 16 | 0 | 0 | 48 |
| Delayed Ice Slip | 36 | 12 | 48 | 24 | 24 | 16 | 16 | 16 | 0 | 0 | 48 |
| Ice benign control | 24 | 8 | 32 | 16 | 16 | 16 | 0 | 16 | 0 | 0 | 32 |
| Ice near-hazard precursor | 34 | 10 | 44 | 22 | 22 | 16 | 12 | 16 | 0 | 0 | 44 |
| Left Sand Support speed matrix | 36 | 12 | 48 | 24 | 24 | 16 | 16 | 16 | 48 | 0 | 0 |
| Right Sand Support speed matrix | 36 | 12 | 48 | 24 | 24 | 16 | 16 | 16 | 0 | 48 | 0 |
| Delayed Sand Support | 18 | 6 | 24 | 12 | 12 | 0 | 24 | 0 | 24 | 0 | 0 |
| Staged Sand benign | 36 | 12 | 48 | 24 | 24 | 16 | 16 | 16 | 48 | 0 | 0 |
| Speed Sand benign | 36 | 12 | 48 | 24 | 24 | 16 | 16 | 16 | 24 | 24 | 0 |
| **Total** | **310** | **102** | **412** | **206** | **206** | **136** | **140** | **136** | **144** | **72** | **172** |

The remaining 24 topology counts are the side-N/A hard controls.

## 18. Effective training pool with Unified TRAIN

The selected `RETAIN_AND_AUGMENT` pool is Unified TRAIN 152 + fresh V2 TRAIN 310 = 462 runs. Fresh values below are design intents; generation must replace them with actual oracle counts without split changes.

| Dimension | Unified TRAIN | Fresh V2 TRAIN intent | Effective intent |
|---|---:|---:|---:|
| Total | 152 | 310 | 462 |
| Hazard / no-hazard | 76 / 76 | 192 / 118 | 268 / 194 |
| Slip / Support | 38 / 38 | 102 / 90 | 140 / 128 |
| Concrete / Marble | 76 / 76 | 155 / 155 | 231 / 231 |
| Exact .20 / .25 / .30 / other hard speeds | 0 / 114 / 0 / 38 | 98 / 102 / 98 / 12 | 98 / 216 / 98 / 50 |
| Hazard .20 / .25 / .30 | 0 / 76 / 0 | 58 / 76 / 58 | 58 / 152 / 58 |

Where affected side is known/planned, the effective 462 are left 97, right 36, bilateral 33, natural Ice side pending oracle 102, and no-hazard 194. This does not fabricate an Ice side before simulation. Original Unified is 152/462 = 32.9%; fresh baseline/regression matrices are 90/462 = 19.5%; fresh hard-case coverage is 220/462 = 47.6%.

The effective pool is not badly dominated by source, hazard subtype, side, or 0.25 m/s. Actual-outcome drift is a generation-readiness question, not permission for adaptive replacement.

## 19. Run-balanced sampling recommendation

Future training should sample a scenario family, then a run/event, then windows within that run. Recommended hierarchy:

1. balance primary Hazard and primary no-hazard draws;
2. balance Slip and Support within Hazard;
3. balance families rather than raw 1 kHz sample counts;
4. balance Concrete and Marble;
5. balance `.20/.25/.30` Hazard speeds;
6. balance direct left/right Support.

The four Unified groups remain separate baseline strata so their 38-run groups do not merge into one dominant historical bucket. Exact epoch draw counts belong in the future training config after actual physical counts are frozen. No sampler or class weight is implemented now.

## 20. Target extraction policy

Primary Slip windows retain V1's established-Slip local semantics; primary Support retains I1 through established Support semantics. Known benign Sand transition intervals stay negative.

For future-Slip Ice candidates, the stored primary timeline is unchanged, but `[30,50) mm` candidate windows are masked from explicit ordinary-negative sampling and HNM. They are not promoted to established positive. Fully observed benign releases remain negative. Censored/ambiguous candidate states are masked or use a predeclared down-weight, never silently called benign.

The initial V2 model remains one binary Hazard head. Precursor-aware accepted-early scoring is secondary and preserves causal episode provenance.

## 21. HNM policy

No HNM occurs in this milestone. If retained later, it is restricted to retained Unified TRAIN and V2_TRAIN. V2_VALIDATION, Generalization VALIDATION, Generalization HOLDOUT, and both Unified non-TRAIN splits are forbidden.

Established Slip regions, Support I1-positive regions, confirmed future-Slip precursor regions, and censored/ambiguous regions can never be mined as negatives. HNM focuses only on verified physically benign windows. The provisional first comparison retains the V1 mechanics of three post-Round-0 rounds, 1 ms replay, K=12/run, and 30 ms spacing, subject to the frozen new exclusions.

## 22. Primary evaluation contract

The original V1-compatible primary view remains authoritative for V1/V2 comparison. The exact future goals are:

| Gate | Goal |
|---|---:|
| Overall Hazard recall | >=0.90 |
| Slip recall | >=0.95 |
| Support recall | >=0.85 |
| Primary no-hazard specificity | >=0.95 |
| Ice-benign specificity | >=0.95 |
| Premature rate | <=0.10 |
| Slip valid-latency p95 | <=+40 ms |
| Support established-latency p95 | <=+50 ms |

V2_VALIDATION additionally requires every family to be present, precursor annotation integrity, staged-Sand and speed-Sand specificity each `>=0.95`, and right Support recall `>=0.85`. Generalization VALIDATION and the eventual one-shot Generalization HOLDOUT use the primary gates above without weakening them because V1 failed.

## 23. Precursor-aware secondary evaluation

The Ice-only secondary view reports:

- fraction of future-Slip candidate episodes with an alert;
- fraction of fully observed benign releases with an alert;
- precursor onset → Reflex and Reflex → established Slip timing;
- same/next/later episode behavior; and
- censored alerts in a separate non-benign category.

Categories are `VALID_ESTABLISHED_HAZARD_ALERT`, `ACCEPTABLE_ICE_PRECURSOR_ALERT`, `TRUE_BENIGN_FALSE_ALERT`, `PRECURSOR_MISS`, and `ESTABLISHED_HAZARD_MISS`. These are initially descriptive; there is no opaque combined accuracy or gate that can mask primary-contract failure.

## 24. Generalization VALIDATION role

Generalization VALIDATION is an external development/generalization gate. V1 and V2 may be compared on exactly the same runs and primary/secondary evaluation code. It may diagnose or choose among predeclared V2 training variants, but it is not training data, a normalizer source, HNM input, or later fresh-final evidence.

## 25. Generalization HOLDOUT preservation

Generalization HOLDOUT remains 36 sealed runs with guard count 0. It is opened once only after the final V2 corpus/split, architecture, feature schema, new TRAIN normalizer, checkpoints/ensemble, threshold, persistence, primary metric, precursor secondary metric, and model-selection decision are frozen.

The only authorized future transition is `0 → 1` for the final one-shot evaluation. No retuning follows. This milestone did not open, infer, visualize, or derive waveform statistics from it.

## 26. Model V2 experiment sequence

1. `MODEL_V2_DATASET_GENERATION`: expand this exact matrix, re-audit all exclusions, generate once, audit actual physics, and freeze the manifest.
2. `MODEL_V2_DATA_ONLY_TRAINING`: same GRU, new effective-TRAIN normalizer, fixed three seeds, `0.99 / 5 ms`, and TRAIN-only HNM if retained.
3. `MODEL_V2_GENERALIZATION_DEVELOPMENT_EVALUATION`: compare V1/V2 on Generalization VALIDATION under both declared views; no HOLDOUT.
4. `MODEL_V2_FINAL_FREEZE`: freeze every model/data/evaluation decision.
5. `MODEL_V2_GENERALIZATION_HOLDOUT`: open the sealed 36 once.

None of these milestones is started here.

## 27. Resource estimate

The design contains 412 runs × 8,000 samples = 3,296,000 samples. Recent corpora average about 0.315–0.367 MiB per NPZ, so raw NPZ storage is expected to be approximately 150–163 MiB including metadata margin.

The 48-run semantics and 72-run Generalization corpora have NPZ modification-time spans of 82.7 s and 139.0 s on this host. Allowing for staged topology and integrity work, a practical generation estimate is 15–25 minutes with comparable headless parallelism. This is a few hundred informative runs, not a redundant 1,000-run round target.

## 28. Limitations

- Counts by Hazard/no-hazard, subtype, and actual side are design intents until the physical oracle audits generated trajectories.
- Ice benign is deliberately narrow and excludes `.25 m/s`; accidental Slip must be retained.
- The staged benign wide-cell regime is motivated by previous no-I1/no-Support results but must be revalidated at all three speeds; unexpected Support is retained, not backfilled away.
- Delayed Sand remains `.25 m/s` because other speeds were not calibrated for its timing contract.
- Exact matrix freshness does not establish waveform independence; the predeclared parameter-distance audit reduces avoidable cross-split clones without post-simulation waveform moves.
- Precursor evidence is simulator-development evidence with correlated episodes and sparse benign releases; it is not a real-world probability calibration.
- The design does not evaluate hardware realism, E84 resources, quantization, HIL, or Recovery.

Invalid reasons are predeclared: nonfinite simulation, required target not encountered, malformed trace, sensor drop, or pretarget fall under a required-valid-encounter contract. Invalid rows remain in the manifest. Reserve grid is `none`; replacement, adaptive backfill, result-driven deletion, and class-balance backfill are prohibited.

## 29. Verdict

```text
MODEL_V2_DATASET_DESIGN_READY
```

Every localized V1 failure maps to an explicit intervention. Primary Hazard and Ice precursor semantics do not contradict each other. Source/speed/Support-side coverage is meaningful, the split and exact expansion rules are frozen, planned IDs/signatures are unique with zero prior/cross-split overlap, and the Generalization HOLDOUT remains sealed.

Config file SHA-256: `27076a9e85921d369587025dae828fca2603f9e6145cd5f679241e5486bf9232`.

Training/search counters remain:

```text
optimizer steps = 0
checkpoint writes = 0
normalizer fits = 0
HNM rounds = 0
threshold searches = 0
persistence searches = 0
architecture searches = 0
```

The architecture conclusion remains `ARCHITECTURE_CHANGE_NOT_YET_JUSTIFIED`. Pelvis IMU6 + left FSR4 = 10 physical channels remains a provisional candidate, not a final sensor freeze. Terrain V2 is not authorized.

## 30. Recommended next milestone

```text
MODEL_V2_DATASET_GENERATION
```

That milestone should consume this config as the authoritative pre-simulation matrix, re-audit every historical/new manifest, generate exactly once, and freeze actual physical outcomes. It is not started here.
