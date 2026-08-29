import unittest
from pathlib import Path

import numpy as np

from fastreflex.evaluation.reflex_event import EventHoldoutGuard, EventRun, _load_yaml
from fastreflex.evaluation.stability_temporal import _file_sha256
from fastreflex.evaluation.support_terrain_fusion import (
    PER_FOOT_UNAVAILABLE,
    POLICY_F0,
    POLICY_F1,
    current_sand_context,
    evaluate_policy,
    raw_support_alert,
    raw_policy_parity,
    recent_sand_context,
    select_validation_policy,
    support_risk_trace,
    terrain_interface_audit,
)
from fastreflex.evaluation.terrain_conditioned_reflex import (
    ICE,
    SAND,
    UNKNOWN,
    BranchReplay,
    TerrainGateTrace,
)
from fastreflex.evaluation.transition_scenarios import fusion_regression


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiment/20260829_causal_support_terrain_context_fusion.yaml"


def _run(*, event: int | None = 100, hard: bool = False) -> EventRun:
    samples = 200
    imu = np.zeros((samples, 6), dtype=np.float32)
    fsr = np.ones((samples, 8), dtype=np.float32)
    return EventRun(
        run_id="fusion_run",
        split="validation",
        source_terrain="concrete",
        target_terrain="sand" if not hard else "concrete",
        design_role="fall" if event is not None else "stable",
        first_contact_sample=20,
        first_touchdown_sample=30,
        censor_sample=samples,
        outcome_diagnostic="VALID_FALL" if event is not None else "VALID_STABLE",
        fall_sample_diagnostic=180 if event is not None else None,
        features={
            "PELVIS_IMU6": imu,
            "PELVIS_IMU6_FSR8": np.concatenate((imu, fsr), axis=1),
        },
        timestamp_us=np.arange(samples, dtype=np.int64) * 1000,
        slip_event_samples_per_foot=(None, None),
        support_event_samples_per_foot=(event, None),
        event_sample=event,
        event_type="SUPPORT" if event is not None else "NONE",
        hard_stable_control=hard,
        drift_m=np.zeros((samples, 2), dtype=np.float32),
        tangential_velocity_mps=np.zeros((samples, 2), dtype=np.float32),
        support_spread_m=np.zeros((samples, 2), dtype=np.float32),
        support_max_displacement_m=np.zeros((samples, 2), dtype=np.float32),
        loaded_contact=np.ones((samples, 2), dtype=bool),
        sink_pattern="transition_left",
        support_pattern="lateral_deformable",
    )


def _gate(states: np.ndarray) -> TerrainGateTrace:
    sand = np.flatnonzero(states == SAND)
    return TerrainGateTrace(
        state=states.astype(np.int8),
        update_samples=np.asarray((40, 92), dtype=np.int64),
        prediction_ids=np.asarray((3, 2), dtype=np.int8),
        prediction_probabilities=np.zeros((2, 4), dtype=np.float32),
        first_target_valid_sample=None if not len(sand) else int(sand[0]),
        clean_event_count=2,
    )


def _replay(probability: np.ndarray, states: np.ndarray) -> BranchReplay:
    endpoints = np.arange(20, 200, dtype=np.int64)
    return BranchReplay(
        endpoints=endpoints,
        probabilities=probability[endpoints].astype(np.float64),
        terrain_state=states[endpoints].astype(np.int8),
    )


class RawDetectorInvariantTest(unittest.TestCase):
    def test_frozen_contract_and_hashes_are_exact(self) -> None:
        document = _load_yaml(CONFIG)
        support = document["frozen_support"]
        self.assertEqual(support["history_ms"], 20)
        self.assertEqual(support["probability_threshold"], 0.94)
        self.assertEqual(support["persistence_ms"], 5)
        self.assertEqual(support["feature_dimension"], 60)
        for record in (support["normalizer"], *support["checkpoints"]):
            self.assertEqual(
                _file_sha256(ROOT / record["path"]), record["sha256"]
            )

    def test_raw_score_persistence_has_no_terrain_input_or_reset(self) -> None:
        probability = np.asarray((0.95,) * 8, dtype=np.float64)
        alert, onset = raw_support_alert(probability)
        self.assertEqual(np.flatnonzero(onset).tolist(), [4])
        terrain_a = np.asarray((SAND,) * 4 + (ICE,) * 4)
        terrain_b = terrain_a[::-1]
        self.assertTrue(np.array_equal(alert, raw_support_alert(probability)[0]))
        self.assertFalse(np.array_equal(terrain_a, terrain_b))
        self.assertTrue(np.all(alert[4:]))


class ContextPolicyTest(unittest.TestCase):
    def test_f0_is_exact_current_global_sand(self) -> None:
        states = np.asarray((UNKNOWN, SAND, SAND, ICE), dtype=np.int8)
        self.assertEqual(
            current_sand_context(states).tolist(), [False, True, True, False]
        )

    def test_f1_exact_50ms_grace_and_51ms_expiry(self) -> None:
        states = np.full(53, ICE, dtype=np.int8)
        states[0] = SAND
        context = recent_sand_context(states, grace_ms=50)
        self.assertTrue(context[50])
        self.assertFalse(context[51])
        self.assertFalse(context[52])

    def test_f1_non_sand_does_not_erase_reset_is_local_and_future_is_unused(self) -> None:
        states = np.asarray((SAND, ICE, ICE, ICE), dtype=np.int8)
        context = recent_sand_context(states, grace_ms=2)
        self.assertEqual(context.tolist(), [True, True, True, False])
        reset = recent_sand_context(np.asarray((ICE, ICE), dtype=np.int8), grace_ms=2)
        self.assertFalse(np.any(reset))
        future_changed = np.r_[states[:3], SAND]
        self.assertTrue(
            np.array_equal(
                context[:3], recent_sand_context(future_changed, grace_ms=2)[:3]
            )
        )

    def test_post_fusion_does_not_modify_raw_alert(self) -> None:
        raw = np.asarray((False, True, True, False))
        context = np.asarray((True, False, True, True))
        before = raw.copy()
        risk, onset = support_risk_trace(raw, context)
        self.assertTrue(np.array_equal(raw, before))
        self.assertEqual(risk.tolist(), [False, False, True, False])
        self.assertEqual(onset.tolist(), [False, False, True, False])

    def test_interface_now_exposes_reviewed_per_foot_provenance(self) -> None:
        audit = terrain_interface_audit()
        self.assertTrue(audit["F2_implementable"])
        self.assertIsNone(audit["F2_result"])
        self.assertTrue(audit["prediction_foot_identity"])


class EventAndSelectionTest(unittest.TestCase):
    def test_f0_suppression_and_f1_rescue_share_raw_event_clock(self) -> None:
        run = _run()
        probability = np.zeros(200, dtype=np.float64)
        probability[90:111] = 0.98
        states = np.full(200, ICE, dtype=np.int8)
        states[40:92] = SAND
        gate = _gate(states)
        replay = _replay(probability, states)
        f0 = evaluate_policy(
            POLICY_F0, {run.run_id: run}, {run.run_id: gate}, {run.run_id: replay}
        )
        f1 = evaluate_policy(
            POLICY_F1, {run.run_id: run}, {run.run_id: gate}, {run.run_id: replay}
        )
        self.assertEqual(f0["raw_detected_events"], 1)
        self.assertEqual(f1["raw_detected_events"], 1)
        self.assertEqual(
            f0["event_rows"][0]["raw_first_valid_sample"],
            f1["event_rows"][0]["raw_first_valid_sample"],
        )
        self.assertEqual(f0["context_suppression_count"], 1)
        self.assertEqual(f0["detected_events"], 0)
        self.assertEqual(f1["context_suppression_count"], 0)
        self.assertEqual(f1["detected_events"], 1)
        self.assertEqual(f1["event_rows"][0]["support_event_sample"], 100)
        self.assertTrue(raw_policy_parity(f0, f1))

    def test_system_premature_requires_raw_and_authorized_context(self) -> None:
        run = _run()
        probability = np.zeros(200, dtype=np.float64)
        probability[40:60] = 0.98
        sand = np.full(200, SAND, dtype=np.int8)
        metrics = evaluate_policy(
            POLICY_F0,
            {run.run_id: run},
            {run.run_id: _gate(sand)},
            {run.run_id: _replay(probability, sand)},
        )
        self.assertEqual(metrics["premature_event_runs"], 1)
        self.assertEqual(metrics["detected_events"], 0)
        self.assertEqual(
            metrics["premature_mechanisms"], {"raw_support_false_alert": 1}
        )

    def test_validation_selection_disqualifies_f0_historical_suppression(self) -> None:
        latency = {"median": 0.0, "p95": 0.0}
        base = {
            "support_recall": 1.0,
            "sand_benign_specificity": 1.0,
            "premature_event_run_rate": 0.0,
            "fusion_latency_ms": latency,
            "context_suppression_rate": 0.0,
            "hard_ground_specificity": 1.0,
        }
        gates = {
            "support_recall_min": 0.95,
            "sand_benign_specificity_min": 0.95,
            "premature_event_run_rate_max": 0.05,
            "median_fusion_latency_ms_max": 20,
            "p95_fusion_latency_ms_max": 50,
            "context_suppression_rate_max": 0.05,
            "hard_ground_specificity_min": 0.95,
        }
        selection = select_validation_policy(
            {POLICY_F0: base, POLICY_F1: base},
            {
                POLICY_F0: {"context_suppression_count": 7},
                POLICY_F1: {"context_suppression_count": 0},
            },
            gates,
        )
        self.assertEqual(selection["selected"], POLICY_F1)

    def test_holdout_guard_and_regressions(self) -> None:
        guard = EventHoldoutGuard()
        self.assertEqual(guard.open_count, 0)
        with self.assertRaises(RuntimeError):
            guard.require_open()
        guard.open_once()
        guard.require_open()
        self.assertEqual(guard.open_count, 1)
        with self.assertRaises(RuntimeError):
            guard.open_once()
        self.assertTrue(fusion_regression()["passed"])
        source = (
            ROOT / "src/fastreflex/evaluation/support_terrain_fusion.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("run_simulation(", source)
        self.assertNotIn("exact_geom", source)


if __name__ == "__main__":
    unittest.main()
