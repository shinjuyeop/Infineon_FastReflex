# Deployment-Aware QAT Engineering Derivative

## Verdict

`DEPLOYMENT_AWARE_QAT_TRAIN_ACCEPTANCE_FAIL`

The single predeclared QAT family was implemented and trained from all three
exact frozen V2 Round-3 members, but it did not satisfy the frozen TRAIN-only
quantization-robustness and decision-parity gates. The run therefore stopped
before candidate freeze, V2 development/golden evaluation, or handoff export.
There is no QAT deployment candidate for E84 to accept from this cycle.

This result is an engineering failure, not a scientific result. The immutable
verdict remains `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`, and the cycle
does not create Model V3 or change any scientific label, split, architecture,
sensor, normalizer, threshold, or persistence contract.

## 1. Starting state

- Research branch: `main`.
- Research starting `HEAD`: `8616590a37fd82e12f0821597c198a30a2d2d6ba`.
- Starting `HEAD == origin/main`: yes.
- Starting tracked worktree: clean.
- Deployment branch/HEAD: `main` at
  `564d3db788f8a1fa4a02b405a8e8f227c1b2755f`.
- Deployment tracked worktree was clean and remained read-only. No tracked or
  generated Deployment file was modified by this work.

The Deployment contract was re-read from `docs/model_contract.md`,
`docs/deployment_pipeline.md`, and
`reports/int8_recurrent_error_localization_and_ptq_recovery.md` before QAT
design. Research was not used to reinterpret or rewrite M3/M3.1 evidence.

## 2. Scientific boundary

The candidate role was frozen as `DEPLOYMENT_QAT_ENGINEERING_CANDIDATE`, with
`scientific_candidate=false`, `scientific_release=false`,
`generalization_supported=false`, `real_robot_supported=false`, and
`safety_certified=false`.

The following historical state is unchanged:

- `MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED`
- `SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED`
- `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED`
- `BOUNDARY_RESOLUTION_INVALID`
- `TRAINING_OBJECTIVE_SAMPLING_TENSION`

Historical Generalization HOLDOUT, Generalization VALIDATION, Unified HOLDOUT,
consumed `FACTOR_VALIDATION`, protected Generalization payloads,
`BOUNDARY_RESOLUTION_VALIDATION`, failed boundary payloads, and the Sand
scientific-intervention corpora were rejected by the QAT config and code. Their
training, inference, feature reconstruction, visualization, and QAT access
counters are zero.

## 3. E84 M3/M3.1 failure contract

The frozen deployment diagnosis is that the first material full-INT8 error
occurs in the input projection before recurrent timestep 0. Similar early
projection error is amplified through recurrent hidden feedback, particularly
for seeds `20260829` and `20260830`. Their frozen 20-step hidden Jacobian-product
norms are `26.66` and `19.79`, versus `5.12` for seed `20260828`.
Sigmoid, tanh, classifier, softmax, and input saturation were not isolated as
the primary E84 cause.

The E84 TRAIN-selected 16-channel PTQ partition improved golden member maxima
to `0.2190 / 0.2614 / 0.1286` and ensemble max/p95 to `0.0833 / 0.0407`, but the
unchanged deployment-owned member maximum gate is `<=0.10`. M3.1 therefore
remains failed and M4 remains unauthorized.

## 4. QAT design and protocol freeze

The complete protocol was committed before optimizer step 1 in commit
`a1a126cc16a8213ad50746a558eab7d1d2b57443`; its config SHA-256 is
`e88034237aaba33175d2c1125e9e70f239876b9e6dbd33d4395bc7d4738409c6`.
No result-driven config change or hyperparameter search followed.

The sole intervention was explicit recurrent QAT:

- exact PyTorch reset-after equation `n + z * (h - n)`;
- 20 statically unrolled timesteps and fresh zero hidden state per window;
- mathematically equivalent two-block-per-gate 16-channel projection;
- INT8 `[-127,127]` symmetric per-output-channel weights;
- INT32 bias simulation with input-scale × weight-scale;
- INT8 `[-128,127]` affine per-tensor operation outputs;
- fake-quantized hidden state fed into the next recurrent timestep;
- fixed sigmoid/softmax scale `1/256` and tanh scale `1/128`;
- frozen E84 input scale `0.03241445869207382`, zero point `0`, derived from the
  TRAIN p99 bound `±4.132843623161313`;
- one model-blind observer pass over all 2,597 calibration windows, frozen
  before training.

Standard `torch.ao` GRU QAT was not used because it would not prove alignment
with the explicit E84 static lowering. The architecture and checkpoint state
layout remain `GRU(80,32,1) + Linear(32,2)`, 11,010 parameters.

## 5. Exact data usage

No simulator run or dataset generation occurred. Training reconstructed the
exact source Round-3 endpoint identities from the existing effective TRAIN:

| Data | Count / identity |
|---|---|
| Unified TRAIN runs | 152 |
| Valid V2_TRAIN runs | 290 |
| Effective TRAIN | 442 runs |
| Round-3 fit windows | 40,319 (`37,801` negative, `2,518` positive) |
| Frozen TRAIN monitor | 7,241 (`6,624` negative, `617` positive) |
| Fit window identity | `55cc0d053bd4f5a8c0d4be86ea90682d1f13f89ace2accb58e96c8cfafc91656` |
| Monitor identity | `68a8d6cf9b73a4c61c43d3e751a0cb44cb9c4e2ad3eeafec91c54a31b8dced62` |
| Calibration windows | 2,597 `[20,80] float32` |

The three frozen source HNM endpoint hashes reproduced exactly:

```text
4a0b0bb3610f1d7ce7eac035cc7a6c3f347b489e0ba2299186f1c04e4158d3fa
1ca59fc13213d0a8ee6987ad933d083848531d7ec829f84b1102dbd574699085
e1bcca00c28a553696198a36d48f7149d736f8884230addcf6e71f8f36e77d5f
```

This reproduction used only frozen source checkpoints and TRAIN. New HNM,
normalizer fit, model-output window selection, and Sand boundary loss mass were
all zero.

## 6. Fake-quant fidelity proof and diagnostics

Fake quantization was bypassed through the same explicit equation and compared
with each loaded frozen PyTorch GRU over all 2,597 calibration windows. Maximum
logit error was exactly `0.0` for each of the three seeds. This proves the Float
equation and state-dict parity of the explicit path, not TFLite INT8 parity.

The model-blind observer froze 548 operation/timestep tensor ranges per member.
Observed affine activation-scale min/median/max were:

| Seed | Min | Median | Max |
|---:|---:|---:|---:|
| 20260828 | 0.000765929 | 0.015625302 | 0.193813384 |
| 20260829 | 0.001085175 | 0.016189031 | 0.215676386 |
| 20260830 | 0.001328541 | 0.015819351 | 0.203960479 |

The frozen p99 input policy clips exactly 41,552 of 4,155,200 calibration
elements (`1.0%` by definition). The raw retained TRAIN-derived range remains
`[-40.36348, 314.62930]`; runtime preprocessing itself is unchanged.

Final QAT recurrent per-channel weight-scale ranges were:

| Seed | Input weight min/max | Recurrent weight min/max | Classifier min/max |
|---:|---:|---:|---:|
| 20260828 | 0.00164491 / 0.00401068 | 0.00175153 / 0.00370709 | 0.00233456 / 0.00236186 |
| 20260829 | 0.00182999 / 0.00474727 | 0.00198723 / 0.00525747 | 0.00317468 / 0.00337847 |
| 20260830 | 0.00175913 / 0.00438214 | 0.00173388 / 0.00678079 | 0.00248503 / 0.00287414 |

Research fake quant is intentionally labeled an approximation. Actual TFLite
calibration, integer kernels, graph conversion, and Vela mapping were not run in
Research and remain E84 authority.

## 7. Training protocol and ledger

Each student began from its corresponding exact frozen member. Adam used
`lr=1e-4`, batch size 128, no weight decay, no gradient clipping, maximum 12
epochs, patience 3, and deterministic shuffling. Epoch selection minimized the
fixed composite loss on the frozen TRAIN monitor only.

The loss was frozen as:

```text
0.20 * inverse-frequency CE(fake logits)
+ 1.00 * SmoothL1(fake logits, frozen teacher logits)
+ 1.00 * SmoothL1(student Float logits, frozen teacher logits)
+ 0.25 * MSE(fake hidden[0:20], frozen teacher hidden[0:20])
```

| Seed | Epochs completed | Best epoch | Optimizer steps | Best monitor composite |
|---:|---:|---:|---:|---:|
| 20260828 | 5 | 2 | 1,575 | 0.0496150 |
| 20260829 | 12 | 12 | 3,780 | 0.0501526 |
| 20260830 | 12 | 12 | 3,780 | 0.0524115 |
| Total | 29 | — | 9,135 | — |

Training did not diverge, and all three seeds were retained. Passing training
loss was not sufficient for candidate acceptance.

## 8. TRAIN-only quantization robustness

The reference is each exact frozen Float V2 member. `Original fake` applies the
same frozen fake-quant path to unchanged source weights; `QAT fake` applies it
to QAT weights. These values use all exact Round-3 fit windows and are not
actual E84 TFLite results.

| Seed | Original fake max / p95 | QAT Float max / p95 | QAT fake max / p95 |
|---:|---:|---:|---:|
| 20260828 | 0.6820 / 0.02128 | 0.3638 / 0.01495 | 0.6659 / 0.02321 |
| 20260829 | 0.8765 / 0.03715 | 0.1892 / 0.01407 | 0.5834 / 0.03256 |
| 20260830 | 0.8145 / 0.04079 | 0.2154 / 0.01678 | 0.5498 / 0.03450 |
| Ensemble | 0.7838 / 0.02235 | 0.1216 / 0.00975 | 0.5338 / 0.02033 |

Ensemble signed bias changed from `-0.000444` to `-0.000712`, within the
predeclared `0.01` budget. P95 absolute gates also passed. However:

- worst member max was `0.6659`, not `<=0.10`;
- ensemble max was `0.5338`, not `<=0.10`;
- worst-member max ratio was `0.760`, not `<=0.50` of original;
- ensemble p95 ratio was `0.910`, not `<=0.75` of original;
- ensemble threshold crossings differed on 93 fit endpoints;
- persistence counts differed on 93 fit endpoints.

Reflex-onset parity happened to remain exact on the sparse Round-3 fit endpoint
sequences, but this does not rescue the failed continuous and crossing gates.
The QAT effect was also seed-dependent: seed `20260828` p95 became worse while
the two originally unstable seeds improved but remained far outside the member
maximum contract.

## 9. Recurrent hidden-error analysis

Worst QAT-fake TRAIN windows retained material recurrent error:

| Seed | First projection max | First hidden `>=0.10` | t=19 hidden max | Classifier logit max |
|---:|---:|---:|---:|---:|
| 20260828 | 44.3489 | t=10 | 1.9610 | 1.9351 |
| 20260829 | 127.9752 | t=0 | 1.0789 | 1.7382 |
| 20260830 | 32.6873 | t=0 | 0.8803 | 1.3186 |

These TRAIN-fit maxima include input values beyond the frozen p99 range, hence
the very large first-projection discrepancies after input clipping. They are
not a reinterpretation of the E84 golden worst windows, which were within the
input range. Even with the two unstable seeds' probability improvement, hidden
feedback still produced order-one t=19 errors, so this cycle did not remove the
failure mode robustly.

## 10. Float behavior preservation

The config required candidate freeze before any development/golden behavior
check. Because TRAIN-only acceptance failed, the QAT candidate was not frozen
and no V2_VALIDATION or deployment golden inference was performed. Therefore:

```text
Float behavior preservation: NOT EVALUATED (correct fail-closed stop)
V2_VALIDATION access: 0
Deployment golden access: 0
Generalization VALIDATION/HOLDOUT access: 0
```

The TRAIN-only QAT Float-vs-source maxima (`0.3638 / 0.1892 / 0.2154`) also
indicate nontrivial boundary movement on some fit windows. This is diagnostic
TRAIN evidence only; it is not promoted to a development behavior claim.

## 11. Candidate freeze and export/handoff

No `candidate_freeze.json` was created. The trained checkpoints remain
Gitignored failed-cycle engineering artifacts and are not a candidate bundle.
No `artifacts/releases/model_v2_anchor_refined_gru20_qat_20260904/` directory
was created. The historical V2 release was not overwritten; its release
manifest remains
`d5d4e7225a35d7547e373b0ac62dbaf552d45c1a3290f214882a032355589dc7`.

Because there is no accepted candidate, candidate-manifest integrity and export
checksum tests are not applicable. Historical release integrity/checksum tests
remain applicable and unchanged.

## 12. Scientific non-claim and next action

This experiment does not support generalization, scientific Model V3, a new
decision boundary, real-robot behavior, production readiness, or safety. It
does not modify `SUPPORT_SIMULATION_GENERALIZATION_SUPPORTED` or any negative
scientific verdict.

There is no E84 QAT revalidation action for this failed candidate. E84 should
remain on the existing failed M3/M3.1 state and M4 remains unauthorized. The
smallest possible future engineering milestone is a separately predeclared
review of why fixed source-observer QAT left p99-clipped projection outliers and
seed `20260828` robustness unresolved. It must not be added to this cycle, and
must independently justify any change such as observer/range treatment. No
Jacobian penalty, architecture change, seed removal, loss redesign, threshold
search, or broader scientific work is implied.

## 13. Counters

| Counter | Value |
|---|---:|
| Candidate families | 1 |
| Source/trained seeds | 3 / 3 |
| Optimizer steps | 9,135 |
| Epochs completed | 29 |
| New simulator runs | 0 |
| New datasets | 0 |
| Normalizer fits | 0 |
| New HNM rounds | 0 |
| Hyperparameter searches | 0 |
| Threshold/persistence searches | 0 / 0 |
| Architecture/sensor/feature changes | 0 / 0 / 0 |
| Protected HOLDOUT opens/inference | 0 / 0 |
| Consumed FACTOR_VALIDATION access | 0 |
| Boundary-resolution payload access | 0 |
| Candidate freezes | 0 |
| Handoffs exported | 0 |
| Deployment repository modifications | 0 |

## 14. Artifacts and hashes

Generated failed-cycle evidence is Gitignored under
`artifacts/runs/20260904_deployment_aware_qat/`:

| Artifact | SHA-256 |
|---|---|
| Pretraining audit | `ef0c360a04ca1c1eed0f90c415acc30cb8a173813d9b1697df1e655a872f1e5d` |
| Training ledger | `0663d7b90d7e271b8fc005e5fe8e8b0dc907c381868ad9c4de40e50459407f19` |
| TRAIN quantization audit | `1e5c013d9c961deb4463f72e694d0cf3452701a92e5c2ffd551b457940761ce7` |
| Seed 20260828 checkpoint | `262d3846e3678b65d07ad1a432b65e5b72ebfff232e0ed1899c164ed82d6ae2e` |
| Seed 20260829 checkpoint | `98cfb31c2ede75244821b27d5602ff9735f4db5e448cfdc6eb03de6c2e84cc13` |
| Seed 20260830 checkpoint | `02429b3d732bb714653dc5eb9206ccfae8185c0105e152e090b6e2735b840ec7` |
| Seed 20260828 activation ranges | `1c267c6329c943be6dce39ff58efbebb90d20bb77326df17bac30d15f42ac757` |
| Seed 20260829 activation ranges | `0e9faeb8dd4313b00f266f21f966f905e32c303223cf131409a69f558d06dd60` |
| Seed 20260830 activation ranges | `0a87099380d77c99f7125bfd574436d20ddec329cffe87084c2ef54f0b98507d` |

The source checkpoint, normalizer, feature, architecture, calibration, and
release-manifest hashes match the frozen config. The exact final Git and test
state is recorded after documentation completion in the task handoff response.

## 15. Tests

Before training, focused QAT tests passed for the one-family/protected-data
contract, Float bypass equation/state-layout parity, finite fake-quant shapes,
and recurrent gradient flow. Final verification produced:

```bash
pytest                                                        # unavailable on PATH
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
# 218 passed, 1 skipped in 94.53s
python -m compileall -q .                                    # PASS
ruff check                                                    # PASS
git diff --check                                              # PASS
```

The first command's shell exit was `127` because no standalone `pytest`
executable is installed. The equivalent repository-supported `python -m
pytest` invocation passed the complete suite.
