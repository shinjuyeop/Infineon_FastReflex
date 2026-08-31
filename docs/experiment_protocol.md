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

The current one-shot HOLDOUT is consumed scientific evidence. Repository consolidation and ordinary verification do not reopen or reinterpret it.

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

## 5. Terrain training and evaluation

Clean touchdown windows are run-disjoint. Normalization uses TRAIN events only. The selected candidate is FSR4, MLP, 50 ms and a three-seed mean-probability ensemble.

At runtime a prediction becomes available at touchdown +50 ms and is held until the next valid update. It may refine an already asserted Hazard cause, but it is never an input to Hazard feature extraction, GRU inference or persistence.

Current regression tests cover FSR4 channel/window parity, normalization/inference shape, exact update timestamp, held state, prediction provenance, protected hashes and the cause-refinement truth table.

## 6. Physical and simulator regression

The consolidation milestone does not change simulator behavior. Tests preserve:

- physics/viewer state parity;
- 0.5 ms physics and 1 kHz timestamps;
- established Slip oracle behavior;
- Support spread/loss and persistence behavior;
- causal physical diagnostics including I1 inputs;
- FSR channel/quadrant mapping and terrain geometry.

Potential scientific or timing bugs discovered during cleanup are not silently fixed. They require a separate declared milestone.

## 7. Protected artifact verification

Read-only verification checks:

- Unified Hazard freeze identity `91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2`;
- Hazard normalizer and three final checkpoint SHA-256 values;
- 80D schema hash and GRU metadata/parameter count;
- Terrain normalizer and three checkpoint SHA-256 values;
- `terrain_used_as_gate: false` and no HOLDOUT opening.

Generated checkpoints are never rewritten by verification.

## 8. CLI and historical provenance

`collect`, `train`, and `evaluate` require explicit configs. The current supported evaluation configs verify the two frozen candidates. A historical experiment ID fails with a message directing the user to its recorded source commit. No historical ID falls through to a generic runner.

Dated configs, reports and Git history preserve the research path. This does not imply that every historical config remains runnable from the consolidated current tree.

## 9. Research-to-Deployment handoff

Research completion does not authorize quantization, export, E84 integration, HIL, Recovery or sensor freeze. Those actions require a reviewed deployment milestone and explicit artifact provenance. This repository's supported result is a Float research candidate and behavior contract, not a target-runtime claim.
