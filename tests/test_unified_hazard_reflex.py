"""Contract tests for the final control-facing unified hazard reflex study."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from fastreflex.evaluation.reflex_event import (
    EVENT_TYPE_NONE,
    EVENT_TYPE_SLIP,
    EVENT_TYPE_SUPPORT,
    EventHoldoutGuard,
    EventRun,
    _load_yaml,
)
from fastreflex.evaluation.terrain_conditioned_reflex import (
    BranchReplay,
    TerrainGateTrace,
)
from fastreflex.evaluation.continuous_slip_reflex import SlipReplay
from fastreflex.evaluation.unified_hazard_reflex import (
    LABEL_NO_HAZARD,
    LABEL_PRECURSOR_ONLY,
    LABEL_SLIP,
    LABEL_SUPPORT,
    evaluate_unified_replays,
    generate_unified_specifications,
    i1_support_precursor_sample,
    physical_hazard_label,
    split_for_source_index,
    unified_negative_candidates,
    unified_positive_endpoints,
    validate_unified_design,
    verify_frozen_system,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiment/20260829_unified_hazard_reflex_system.yaml"


def _run(
    *,
    slip: int | None = None,
    support: int | None = None,
    spread_start: int | None = None,
    hard: bool = False,
) -> EventRun:
    samples = 500
    imu = np.zeros((samples, 6), dtype=np.float32)
    fsr = np.zeros((samples, 8), dtype=np.float32)
    spread = np.zeros((samples, 2), dtype=np.float32)
    loaded = np.zeros((samples, 2), dtype=bool)
    loaded[100:, 0] = True
    if spread_start is not None:
        spread[spread_start:, 0] = (
            np.arange(1, samples - spread_start + 1, dtype=np.float32) * 0.0001
        )
    event = (
        min(value for value in (slip, support) if value is not None)
        if slip is not None or support is not None
        else None
    )
    event_type = (
        EVENT_TYPE_SLIP
        if slip is not None and support is None
        else EVENT_TYPE_SUPPORT
        if support is not None and slip is None
        else "SLIP_AND_SUPPORT"
        if support is not None and slip is not None
        else EVENT_TYPE_NONE
    )
    return EventRun(
        run_id="run",
        split="validation",
        source_terrain="concrete",
        target_terrain="concrete"
        if hard
        else ("sand" if support is not None or spread_start is not None else "ice"),
        design_role="stable",
        first_contact_sample=100,
        first_touchdown_sample=100,
        censor_sample=samples,
        outcome_diagnostic="VALID_STABLE",
        fall_sample_diagnostic=None,
        features={
            "PELVIS_IMU6": imu,
            "PELVIS_IMU6_FSR8": np.concatenate((imu, fsr), axis=1),
        },
        timestamp_us=np.arange(samples, dtype=np.int64) * 1000,
        slip_event_samples_per_foot=(slip, None),
        support_event_samples_per_foot=(support, None),
        event_sample=event,
        event_type=event_type,
        hard_stable_control=hard,
        drift_m=np.zeros((samples, 2), dtype=np.float32),
        tangential_velocity_mps=np.zeros((samples, 2), dtype=np.float32),
        support_spread_m=spread,
        support_max_displacement_m=spread,
        loaded_contact=loaded,
        sink_pattern="uniform",
        support_pattern="balanced_soft",
    )


def _probability_replay(onset: int | None, samples: int = 500) -> SlipReplay:
    endpoints = np.arange(19, samples, dtype=np.int64)
    values = np.zeros(len(endpoints), dtype=np.float64)
    if onset is not None:
        # Persistence completes at ``onset`` after five samples above threshold.
        values[endpoints >= onset - 4] = 1.0
    return SlipReplay(endpoints=endpoints, probabilities=values)


def _branch_replay(onset: int | None, samples: int = 500) -> BranchReplay:
    replay = _probability_replay(onset, samples)
    return BranchReplay(
        endpoints=replay.endpoints,
        probabilities=replay.probabilities,
        terrain_state=np.zeros(len(replay.endpoints), dtype=np.int8),
    )


def _terrain(samples: int = 500) -> TerrainGateTrace:
    return TerrainGateTrace(
        state=np.zeros(samples, dtype=np.int8),
        update_samples=np.empty(0, dtype=np.int64),
        prediction_ids=np.empty(0, dtype=np.int64),
        prediction_probabilities=np.empty((0, 4), dtype=np.float32),
        first_target_valid_sample=None,
        clean_event_count=0,
    )


class UnifiedHazardReflexTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = _load_yaml(CONFIG)

    def test_frozen_256_run_design_is_fresh_and_split_before_simulation(self) -> None:
        specifications = generate_unified_specifications(self.document)
        audit = validate_unified_design(ROOT, self.document, specifications)
        self.assertTrue(audit["passed"])
        self.assertEqual(
            audit["total_split_counts"], {"train": 152, "validation": 52, "holdout": 52}
        )
        self.assertEqual(audit["duplicate_signatures"], 0)
        self.assertEqual(audit["prior_signature_overlap"], 0)
        self.assertEqual(split_for_source_index("concrete", 26), "validation")
        self.assertEqual(split_for_source_index("marble", 26), "holdout")

    def test_frozen_models_and_feature_schemas_are_unchanged(self) -> None:
        audit = verify_frozen_system(ROOT, self.document)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["slip_model"]["threshold"], 0.58)
        self.assertEqual(audit["support_model"]["threshold"], 0.94)
        self.assertEqual(audit["slip_model"]["history_ms"], 20)
        self.assertEqual(audit["support_model"]["history_ms"], 20)
        self.assertTrue(audit["slip80_strict_semantic_superset_of_support60"])

    def test_i1_is_causal_loaded_positive_derivative_persistence(self) -> None:
        run = _run(spread_start=120)
        self.assertEqual(i1_support_precursor_sample(run), 139)
        altered = _run(spread_start=120)
        altered.support_spread_m[200:] = 10.0
        self.assertEqual(i1_support_precursor_sample(altered), 139)

    def test_physical_labels_do_not_use_design_or_fall(self) -> None:
        self.assertEqual(physical_hazard_label(_run(slip=200), None), LABEL_SLIP)
        self.assertEqual(physical_hazard_label(_run(support=220), 150), LABEL_SUPPORT)
        self.assertEqual(
            physical_hazard_label(_run(slip=200, support=220), 150),
            "SLIP_AND_SUPPORT_HAZARD",
        )
        self.assertEqual(physical_hazard_label(_run(), None), LABEL_NO_HAZARD)
        self.assertEqual(physical_hazard_label(_run(), 150), LABEL_PRECURSOR_ONLY)
        fallen = replace(
            _run(slip=200),
            outcome_diagnostic="VALID_FALL",
            fall_sample_diagnostic=300,
        )
        self.assertEqual(physical_hazard_label(fallen, None), LABEL_SLIP)

    def test_phase_b_positive_and_negative_regions_do_not_overlap_i1(self) -> None:
        run = _run(support=240)
        positive = unified_positive_endpoints(run, 160, 20)
        negative = unified_negative_candidates(run, 160, 20)
        self.assertTrue(np.all(positive >= 160))
        self.assertTrue(np.all(negative < 160))
        self.assertLessEqual(len(positive), 20)

    def test_cross_trigger_is_a_correct_system_hazard_alert(self) -> None:
        run = _run(support=240)
        metrics = evaluate_unified_replays(
            {"run": run},
            {"run": _terrain()},
            {"run": _probability_replay(180)},
            {"run": _branch_replay(None)},
            slip_threshold=0.5,
            support_threshold=0.5,
            persistence_ms=5,
            precursor_samples={"run": 160},
        )
        self.assertEqual(metrics["support_hazard_recall"], 1.0)
        self.assertEqual(metrics["primary_no_hazard_runs"], 0)
        self.assertEqual(
            metrics["cause_attribution_diagnostic"][
                "support_hazards_triggered_by_slip_branch"
            ],
            1,
        )

    def test_terrain_state_cannot_change_the_or_reflex(self) -> None:
        run = _run(slip=200)
        unknown = _terrain()
        sand = _terrain()
        sand.state[:] = 4
        arguments = dict(
            runs={"run": run},
            slip_replays={"run": _probability_replay(190)},
            support_replays={"run": _branch_replay(None)},
            slip_threshold=0.5,
            support_threshold=0.5,
            persistence_ms=5,
            precursor_samples={"run": None},
        )
        first = evaluate_unified_replays(terrain={"run": unknown}, **arguments)
        second = evaluate_unified_replays(terrain={"run": sand}, **arguments)
        self.assertEqual(
            first["overall_hazard_recall"], second["overall_hazard_recall"]
        )
        self.assertEqual(
            first["rows"][0]["system_first_onset"],
            second["rows"][0]["system_first_onset"],
        )

    def test_pre_i1_alert_is_premature_and_no_hazard_alert_is_fp(self) -> None:
        hazard = _run(support=240)
        hazard_metrics = evaluate_unified_replays(
            {"run": hazard},
            {"run": _terrain()},
            {"run": _probability_replay(150)},
            None,
            slip_threshold=0.5,
            support_threshold=None,
            persistence_ms=5,
            precursor_samples={"run": 160},
        )
        self.assertEqual(hazard_metrics["system_premature_run_rate"], 1.0)
        normal = _run(hard=True)
        normal_metrics = evaluate_unified_replays(
            {"run": normal},
            {"run": _terrain()},
            {"run": _probability_replay(200)},
            None,
            slip_threshold=0.5,
            support_threshold=None,
            persistence_ms=5,
            precursor_samples={"run": None},
        )
        self.assertEqual(normal_metrics["primary_no_hazard_specificity"], 0.0)
        self.assertEqual(normal_metrics["hard_ground_specificity"], 0.0)

    def test_precursor_only_is_excluded_from_no_hazard_specificity(self) -> None:
        run = _run()
        metrics = evaluate_unified_replays(
            {"run": run},
            {"run": _terrain()},
            {"run": _probability_replay(200)},
            None,
            slip_threshold=0.5,
            support_threshold=None,
            persistence_ms=5,
            precursor_samples={"run": 150},
        )
        self.assertEqual(metrics["primary_no_hazard_runs"], 0)
        self.assertEqual(metrics["precursor_only_runs_excluded_from_specificity"], 1)

    def test_holdout_guard_is_one_shot(self) -> None:
        guard = EventHoldoutGuard()
        self.assertEqual(guard.open_count, 0)
        guard.open_once()
        self.assertEqual(guard.open_count, 1)
        with self.assertRaises(RuntimeError):
            guard.open_once()

    def test_config_freezes_phase_b_search_and_terrain_is_advisory(self) -> None:
        phase_b = self.document["phase_b"]
        self.assertEqual([row["history_ms"] for row in phase_b["candidates"]], [20, 50])
        self.assertEqual(phase_b["hnm"]["rounds"], 3)
        self.assertEqual(phase_b["hnm"]["top_k_per_run"], 12)
        self.assertEqual(
            phase_b["threshold"]["grid"], {"start": 0.10, "stop": 0.99, "step": 0.01}
        )
        self.assertFalse(self.document["terrain_advisory"]["terrain_can_block_reflex"])
        forbidden = set(self.document["common"]["runtime_model_forbidden"])
        self.assertTrue(
            {"terrain_identity", "slip_clock", "support_clock", "fall", "future_sample"}
            <= forbidden
        )

    def test_cli_has_canonical_unified_dispatch(self) -> None:
        cli = (ROOT / "scripts/fastreflex.py").read_text(encoding="utf-8")
        self.assertIn("UNIFIED_HAZARD_REFLEX_SYSTEM_VALIDATION", cli)
        self.assertIn("run_unified_hazard_reflex_system", cli)


if __name__ == "__main__":
    unittest.main()
