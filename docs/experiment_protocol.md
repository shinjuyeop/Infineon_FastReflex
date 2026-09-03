# Experiment Protocol

## 1. Current protocol

The supported candidates were selected under four non-negotiable boundaries:

```text
TRAIN-only preprocessing
TRAIN-only hard-negative mining
VALIDATION-only model/threshold selection
sealed one-shot HOLDOUT
```

Terrain and Hazard are evaluated independently. Terrain output cannot gate, authorize, suppress or delay `REFLEX_REQUIRED`.

## 2. Split discipline

Split assignment occurs at run/physical-condition level before simulation. Window-level random splitting is prohibited. Every derived sample inherits its source run split. Duplicate physical signatures and cross-split run IDs fail the dataset audit.

TRAIN may fit model weights, normalizers and hard negatives. VALIDATION may select declared candidates and the operating point only after HNM is complete. HOLDOUT may be opened once after the candidate freeze; no reselection follows.

The Unified and Generalization one-shot HOLDOUTs are consumed scientific evidence. Repository consolidation and ordinary verification do not reopen or reinterpret them. The 36-run Generalization HOLDOUT guard is permanently 1 after the exact final Generalization candidate's single authorized pass.

## 3. Unified Hazard training

The supported model consumes only causal Pelvis IMU-derived 80D features. Positive construction is the frozen union of Slip and Support intervals. Negative construction ends before established Slip, I1 activation, established Support, or fall censor.

Hard-negative mining is fixed:

```text
source split: TRAIN only
rounds after Round 0: 3
continuous replay stride: 1 ms
K: 12 per run
minimum spacing: 30 ms
I1-active region: never negative
```

The TRAIN set is deterministically divided into fit and internal monitor partitions. The monitor partition is still TRAIN; it does not grant early VALIDATION access. Candidate epoch selection uses monitor loss. Threshold selection occurred on VALIDATION only after HNM3.

Current architecture/operating point is frozen at GRU 20 ms, threshold 0.99 and persistence 5 ms. Consolidation does not rerun training or threshold search.

## 4. Hazard evaluation

Replay operates at 1 kHz from endpoint 19 through the pre-censor trace. Each model window is `[endpoint-19, endpoint]`; a future suffix cannot change a decided prefix.

A reflex begins on the fifth consecutive probability `>=0.99`. Physical scoring windows remain:

- Slip: established sample -30 through +40 ms
- Support: I1 precursor, when present, through established Support +50 ms

An alert before the earliest acceptable physical boundary is premature. A later alert cannot retroactively convert that first unjustified reflex into a hit. Primary no-hazard alerts are false positives. Fall/recovery is diagnostic/censor information, not a runtime target.

Current regression tests cover feature/schema parity, normalized tensor parity, `[20,80]` ordering, frozen probability/onset parity on a VALIDATION run, persistence boundary, physical labels and Terrain independence.

## 5. Generalization final-candidate and HOLDOUT protocol

The exact `model_v2_anchor_refined_gru20_20260902` ensemble is frozen under the role `final_generalization_candidate`. The role is an alias to the existing normalizer and three checkpoints; it does not duplicate or mutate artifacts. Generalization VALIDATION remains development evidence with historical primary verdict `GENERALIZATION_PRIMARY_GATES_FAIL` because Slip recall is 11/12, below the frozen 95% gate. The separate interpretation is `GENERALIZATION_DEVELOPMENT_SUPPORTED_WITH_ICE_TIMING_TENSION`; the sole primary failure is a sustained response inside the already-frozen loaded-Ice `[0.030,0.050) m` precursor, not a genuine detector miss.

The completed Generalization HOLDOUT operation retained the exact primary metrics, gates, and event windows used on Generalization VALIDATION. Ice-precursor outcomes remained a separately reported secondary diagnostic and did not rescue or rewrite a primary score. Frozen V1, final V2, and advisory Terrain ran on all 36 HOLDOUT runs in one shared pass, with each payload deserialized once. Final V2 achieved Hazard 25/28, Slip 11/14, Support 14/14, primary specificity 5/8, Ice-benign specificity 2/2, and premature 2/28. The primary and final verdicts are `GENERALIZATION_HOLDOUT_PRIMARY_GATES_FAIL` and `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`.

Readiness verification inspected only IDs, counts, split membership, file existence, stored hashes, file sizes, and guard metadata. The scientific evaluator atomically claimed guard `0 -> 1` before payload access and now refuses a second open. Post-open verification reads and hash-checks only saved summaries and immutable metadata; it cannot deserialize HOLDOUT payloads. There is no training, retuning, model selection, metric/relabeling change, family exclusion, or scientific rerun after consumption.

## 6. Terrain training and evaluation

Clean touchdown windows are run-disjoint. Normalization uses TRAIN events only. The selected candidate is FSR4, MLP, 50 ms and a three-seed mean-probability ensemble.

At runtime a prediction becomes available at touchdown +50 ms and is held until the next valid update. It may refine an already asserted Hazard cause, but it is never an input to Hazard feature extraction, GRU inference or persistence.

Current regression tests cover FSR4 channel/window parity, normalization/inference shape, exact update timestamp, held state, prediction provenance, protected hashes and the cause-refinement truth table.

## 7. Physical and simulator regression

The consolidation milestone does not change simulator behavior. Tests preserve:

- physics/viewer state parity;
- 0.5 ms physics and 1 kHz timestamps;
- established Slip oracle behavior;
- Support spread/loss and persistence behavior;
- causal physical diagnostics including I1 inputs;
- FSR channel/quadrant mapping and terrain geometry.

Potential scientific or timing bugs discovered during cleanup are not silently fixed. They require a separate declared milestone.

## 8. Protected artifact verification

Read-only verification checks:

- Unified Hazard freeze identity `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2`;
- Hazard normalizer and three final checkpoint SHA-256 values;
- 80D schema hash and GRU metadata/parameter count;
- Terrain normalizer and three checkpoint SHA-256 values;
- `terrain_used_as_gate: false` and no HOLDOUT opening.

Generated checkpoints are never rewritten by verification.

## 9. CLI and historical provenance

`collect`, `train`, and `evaluate` require explicit configs. The current supported evaluation configs verify the two frozen candidates. A historical experiment ID fails with a message directing the user to its recorded source commit. No historical ID falls through to a generic runner.

Dated configs, reports and Git history preserve the research path. This does not imply that every historical config remains runnable from the consolidated current tree.

## 10. Research-to-Deployment handoff

Research completion does not authorize quantization, export, E84 integration, HIL, Recovery or sensor freeze. Those actions require a reviewed deployment milestone and explicit artifact provenance. This repository's supported result is a Float research candidate and behavior contract, not a target-runtime claim.

## 11. Redesigned Sand generalization protocol

The redesigned Sand study is a development study, not a replacement final HOLDOUT. Its 176-run matrix is frozen before generation and creates fresh `REDESIGNED_DISCOVERY` and `REDESIGNED_CONFIRMATION` runs together under one domain. Invalid outcomes remain in their original split; adaptive replacement, backfill, split movement, and result-driven deletion are prohibited.

Generation and its first audit are model-blind. Phase diversity uses exact loaded-contact state 20 ms before first censor-valid target contact; the touchdown sample remains descriptive because touchdown structurally turns many entries into double support. Predeclared geometry variants replace the previous non-physical cohort IDs. Source/speed conditioning is permitted only through the frozen physical anchors and cannot use model results.

Confirmation stays sealed until all physical generation gates pass, Discovery physical/diversity analysis completes, and exactly one H1/H2/H3 interpretation plus its metrics is hash-frozen. While sealed, only integrity, planned-signature checks, and aggregate objective physical yield are allowed. Model replay, normalized 80D analysis, observability analysis, visualization, and hypothesis selection are forbidden. The historical consumed Generalization HOLDOUT remains at guard 1 and is never involved.

The generated redesigned study failed three localized Confirmation yield checks and remains physical calibration evidence only. Its 20 insufficient-follow-up records were all post-target fall censors before 9 seconds, not stable horizon censors, so observation duration, label semantics, and generation gates remain unchanged. Before another full study is frozen, only the broad-mild geometry/exposure boundary may be recalibrated with fresh model-blind scenarios; moderate Sand, source/speed axes, Support controls, phase measurement, topology semantics, and the sealed-Confirmation protocol remain fixed.

That recalibration used three pre-frozen, model-blind batches totaling the exact 72-run ceiling. The independent final batch passed 24/24 strict-benign and 4/4 per source-speed cell. The next study therefore uses one common joint start/width/exit envelope for transition-left, one common right envelope for five cells, and a single predeclared Concrete/.25 left-only exception. It does not use cumulative exposure as an input threshold because exposure is fall-censored. Full generation is a separate milestone.

The recalibrated 176-run matrix created fresh `MILD_RECALIBRATED_DISCOVERY` and `MILD_RECALIBRATED_CONFIRMATION` splits of 88 each. Counts, moderate/Support mechanics, nine-second duration, physical labels, diversity metrics, and generation gates remained unchanged. The one-pass, model-blind generation completed 176/176 with no replacement, backfill, or rerun; objective-valid yield was 169, and all 70 frozen physical-generation gates passed. Historical exact reuse, run-ID reuse, cross-split exact overlap, and forbidden planned parameter-near overlap remained zero.

Passing physical generation gates does not open Confirmation. The next authorized milestone may analyze only `MILD_RECALIBRATED_DISCOVERY`, replay the exact frozen final V2 there once, and freeze one exact H1/H2/H3 interpretation with its metric hash. `MILD_RECALIBRATED_CONFIRMATION` remains sealed from model inference, normalized 80D, observability, visualization, and hypothesis selection until that prerequisite is complete. The consumed historical Generalization HOLDOUT remains permanently closed.

That Discovery prerequisite is complete and valid. The analysis contract and implementation were hash-frozen before the successful replay; one anchor/vector per eligible run prevented window-count leakage. The final V2 replayed once on all 88 Discovery records, without training or tuning. Strict Sand was 67/69 specific, Support was 16/16, Pelvis-window reasonable separation passed 4/4, and realizable FSR material increment failed at 1/4. The systematic adverse-margin pattern localized to the coupled transition-left/right-single-precontact factor region, selecting exactly `DOMAIN_DIVERSITY_GAP_SUPPORTED` with interpretation SHA-256 `7c045cd98bb221a0f41911a9662b430548393be90ef192e3194bd619cc3f2ae5`.

The separate one-shot `MILD_RECALIBRATED_CONFIRMATION` replication is complete. Before opening, it froze the exact label, conjunctive rule, preprocessing, metric implementation, localization direction, final V2, and Discovery-pooled scaler hashes. Because the earlier Discovery artifact omitted scaler arrays, the arrays were deterministically reconstructed from Discovery only and accepted only after every mean/std hash matched; Confirmation never refit them. After guard `0 -> 1`, all 88 payloads were deserialized once and exact V2 replayed once. Strict Sand was 61/65 specific with 29/65 adverse, Support was 16/16, the topology/phase direction replicated, and FSR remained non-material at 1/4. Window reasonable separation failed at 3/4 because centroid separation was `.208030 < .75`, so the valid verdict is `DOMAIN_DIVERSITY_GAP_NOT_CONFIRMED`. No H2/H3 substitution is permitted. The split is consumed development evidence and cannot be reopened, rerun, trained on by default, tuned against, or treated as fresh final evidence.
