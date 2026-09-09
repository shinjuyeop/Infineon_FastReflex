# Tests

The suite is organized by current responsibility rather than by every completed
research milestone.

| Area | Test files |
|---|---|
| Simulation | `test_simulation.py` |
| Dataset construction | `test_dataset.py`, `test_generation.py`, `test_factor_conditioned_dataset.py` |
| Hazard and Terrain | `test_hazard.py`, `test_terrain.py` |
| Stability pilot | `test_stability_pilot.py` (labels, latency, sampling phase coverage) |
| Training | `test_training.py`, `test_factor_conditioned_training.py`, `test_qat.py` |
| Protected evaluation | `test_generalization.py`, `test_readiness.py`, `test_holdout.py`, `test_boundary_validation.py` |
| Review tools | `test_visualization.py`, `test_export.py` |

`support.py` contains shared repository paths, fixture loaders, small comparison
helpers, dependency inspection, and the viewer test double. Domain calculations
and expected-value oracles stay inside their owning test so they remain independent
from the implementation being checked.

Historical Sand exploration, calibration, and intermediate-generation assertions
are preserved by Git history and their immutable reports. They are not repeated in
the current regression suite. The remaining tests retain reusable causality,
physical-oracle, integrity, one-shot guard, and protected-data boundaries.

Run the full suite from the repository root:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

The disabled plugin autoload keeps the result independent of unrelated pytest
plugins installed in the local environment.

## Recorded verification state

The 2026-09-09 run at `f35c41d` produced 126 passed, 1 skipped and 9 setup
errors. All nine errors came from `test_holdout.py`: the current
`src/fastreflex/evaluation/terrain.py` hash differs from the protected historical
source hash. The fixture substitutes historical CLI provenance only, so it
still rejects this Terrain source change. The full suite is not green.

Preserve frozen evidence and the consumed one-shot guard. Resolve historical
verification against its recorded source identity rather than replacing protected
hashes or rerunning HOLDOUT to accommodate current code. Tests also depend on
local ignored datasets/checkpoints and the verified walking policy where used;
a fresh clone does not supply these artifacts.
