"""Regression tests for terrain-conditioned continuous reflex detectors."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from fastreflex.evaluation.reflex_event import (
    EVENT_TYPE_NONE,
    EVENT_TYPE_SLIP,
    EVENT_TYPE_SUPPORT,
    EventRun,
    persistent_threshold_events,
)
from fastreflex.evaluation.terrain_conditioned_reflex import (
    ICE,
    MARBLE,
    PHASE_A_CANDIDATES,
    PHASE_B_CANDIDATES,
    SAND,
    BranchHoldoutGuard,
    BranchReplay,
    TerrainGateTrace,
    _threshold_values,
    branch_is_active,
    branch_negative_candidates,
    branch_positive_endpoints,
    evaluate_branch_replays,
    extract_branch_features,
    feature_schema_for_components,
    fsr_feature_base,
    mine_hard_negative_endpoints,
    sustained_alert_trace,
)
from fastreflex.evaluation.transition_scenarios import fusion_regression


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "experiment"
    / "20260828_terrain_conditioned_reflex_detector.yaml"
)


def synthetic_run(
    *,
    run_id: str = "synthetic",
    target: str = "ice",
    event: int | None = 100,
    hard: bool = False,
) -> EventRun:
    samples = 200
    imu = np.zeros((samples, 6), dtype=np.float32)
    fsr = np.ones((samples, 8), dtype=np.float32)
    slip = (event, None) if target == "ice" and event is not None else (None, None)
    support = (event, None) if target == "sand" and event is not None else (None, None)
    event_type = (
        EVENT_TYPE_NONE
        if event is None
        else EVENT_TYPE_SLIP
        if target == "ice"
        else EVENT_TYPE_SUPPORT
    )
    return EventRun(
        run_id=run_id,
        split="validation",
        source_terrain="concrete",
        target_terrain=target,
        design_role="stable",
        first_contact_sample=0,
        first_touchdown_sample=0,
        censor_sample=samples,
        outcome_diagnostic="VALID_STABLE",
        fall_sample_diagnostic=None,
        features={
            "PELVIS_IMU6": imu,
            "PELVIS_IMU6_FSR8": np.concatenate((imu, fsr), axis=1),
        },
        timestamp_us=(np.arange(samples, dtype=np.int64) + 1) * 1000,
        slip_event_samples_per_foot=slip,
        support_event_samples_per_foot=support,
        event_sample=event,
        event_type=event_type,
        hard_stable_control=hard,
        drift_m=np.zeros((samples, 2), dtype=np.float32),
        tangential_velocity_mps=np.zeros((samples, 2), dtype=np.float32),
        support_spread_m=np.zeros((samples, 2), dtype=np.float32),
        support_max_displacement_m=np.zeros((samples, 2), dtype=np.float32),
        loaded_contact=np.ones((samples, 2), dtype=bool),
        sink_pattern="uniform",
        support_pattern="balanced_soft",
    )


def gate(state: int, samples: int = 200, first: int | None = 0) -> TerrainGateTrace:
    return TerrainGateTrace(
        state=np.full(samples, state, dtype=np.int8),
        update_samples=np.asarray([0], dtype=np.int64),
        prediction_ids=np.asarray([max(0, state - 1)], dtype=np.int8),
        prediction_probabilities=np.ones((1, 4), dtype=np.float32),
        first_target_valid_sample=first,
        clean_event_count=1,
    )


class TerrainGatingTests(unittest.TestCase):
    def test_branches_follow_frozen_output_only(self) -> None:
        ice = gate(ICE)
        sand = gate(SAND)
        marble = gate(MARBLE)
        self.assertTrue(np.all(branch_is_active(ice, "slip")))
        self.assertFalse(np.any(branch_is_active(ice, "support")))
        self.assertTrue(np.all(branch_is_active(sand, "support")))
        self.assertFalse(np.any(branch_is_active(marble, "slip")))

    def test_simulator_target_name_cannot_bypass_model_output(self) -> None:
        run = synthetic_run(target="ice")
        marble = gate(MARBLE)
        self.assertEqual(run.target_terrain, "ice")
        self.assertFalse(np.any(branch_is_active(marble, "slip")))
        self.assertEqual(len(branch_positive_endpoints(run, marble, "slip", 20)), 0)

    def test_wrong_terrain_output_can_create_system_fp(self) -> None:
        run = synthetic_run(run_id="hard", target="concrete", event=None, hard=True)
        replay = BranchReplay(
            endpoints=np.arange(200, dtype=np.int64),
            probabilities=np.ones(200),
            terrain_state=np.full(200, ICE, dtype=np.int8),
        )
        result = evaluate_branch_replays(
            {run.run_id: run},
            {run.run_id: gate(ICE)},
            {run.run_id: replay},
            "slip",
            0.5,
        )
        self.assertEqual(result["hard_ground_specificity"], 0.0)
        self.assertTrue(result["hard_rows"][0]["wrong_soft_terrain_output"])


class OracleAndFeatureTests(unittest.TestCase):
    def test_frozen_slip_and_support_persistence(self) -> None:
        values = np.zeros((30, 2), dtype=np.float64)
        valid = np.ones((30, 2), dtype=bool)
        episodes = np.zeros((30, 2), dtype=np.int64)
        values[10:13, 0] = 0.050
        _, slip = persistent_threshold_events(values, valid, episodes, 0.050, 3)
        self.assertEqual(int(np.flatnonzero(slip[:, 0])[0]), 12)
        values[:] = 0.0
        values[5:25, 1] = 0.010
        _, support = persistent_threshold_events(values, valid, episodes, 0.010, 20)
        self.assertEqual(int(np.flatnonzero(support[:, 1])[0]), 24)

    def test_pre_event_negative_and_positive_regions_are_branch_gated(self) -> None:
        run = synthetic_run(event=100)
        trace = gate(ICE)
        positive = branch_positive_endpoints(run, trace, "slip", 20)
        negative = branch_negative_candidates(run, trace, "slip", 20)
        self.assertEqual((int(positive[0]), int(positive[-1])), (80, 140))
        self.assertLessEqual(int(negative[-1]), 70)
        self.assertTrue(set(positive).isdisjoint(set(negative)))

    def test_fsr_features_include_spatial_and_bilateral_terms(self) -> None:
        base, names = fsr_feature_base(np.ones((20, 8), dtype=np.float32))
        self.assertEqual(base.shape, (20, 30))
        self.assertIn("left_front_minus_rear", names)
        self.assertIn("right_medial_ratio", names)
        self.assertIn("bilateral_left_ratio", names)

    def test_derived_features_are_causal_and_schema_is_fixed(self) -> None:
        run = synthetic_run()
        first, names = extract_branch_features(run, ("pelvis_imu6", "fsr8"))
        altered_imu = run.features["PELVIS_IMU6"].copy()
        altered_imu[101:] = 999.0
        altered = EventRun(
            **{
                **run.__dict__,
                "features": {
                    "PELVIS_IMU6": altered_imu,
                    "PELVIS_IMU6_FSR8": run.features["PELVIS_IMU6_FSR8"],
                },
            }
        )
        second, second_names = extract_branch_features(altered, ("pelvis_imu6", "fsr8"))
        np.testing.assert_array_equal(first[:101], second[:101])
        self.assertEqual(names, second_names)
        self.assertEqual(names, feature_schema_for_components(("pelvis_imu6", "fsr8")))
        forbidden = ("terrain", "fall", "event_time", "time_to_fall")
        self.assertFalse(
            any(any(token in name for token in forbidden) for name in names)
        )


class MiningAndEvaluationTests(unittest.TestCase):
    def test_hnm_top_k_separation_and_exclusion(self) -> None:
        endpoints = np.arange(0, 1000, 10, dtype=np.int64)
        probabilities = np.linspace(0.0, 1.0, len(endpoints))
        selected = mine_hard_negative_endpoints(
            endpoints,
            probabilities,
            top_k=8,
            minimum_separation_ms=50,
            excluded=(990,),
        )
        self.assertLessEqual(len(selected), 8)
        self.assertNotIn(990, selected)
        self.assertTrue(all(b - a >= 50 for a, b in zip(selected, selected[1:])))

    def test_five_ms_persistence_and_premature_does_not_hide_valid(self) -> None:
        probabilities = np.zeros(200)
        probabilities[50:55] = 1.0
        probabilities[95:100] = 1.0
        alert, onset = sustained_alert_trace(probabilities, np.ones(200, bool), 0.5, 5)
        np.testing.assert_array_equal(np.flatnonzero(onset), np.asarray([54, 99]))
        self.assertTrue(alert[54])
        run = synthetic_run(event=100)
        replay = BranchReplay(
            endpoints=np.arange(200, dtype=np.int64),
            probabilities=probabilities,
            terrain_state=np.full(200, ICE, dtype=np.int8),
        )
        result = evaluate_branch_replays(
            {run.run_id: run},
            {run.run_id: gate(ICE)},
            {run.run_id: replay},
            "slip",
            0.5,
        )
        row = result["event_rows"][0]
        self.assertTrue(row["any_premature_alert"])
        self.assertTrue(row["valid_detection"])
        self.assertEqual(row["latency_ms"], -1)

    def test_threshold_grid_and_holdout_guard_are_frozen(self) -> None:
        values = _threshold_values({"start": 0.10, "stop": 0.98, "step": 0.02})
        self.assertEqual((len(values), values[0], values[-1]), (45, 0.10, 0.98))
        guard = BranchHoldoutGuard()
        with self.assertRaises(RuntimeError):
            guard.require_open()
        guard.open_once()
        self.assertEqual(guard.open_count, 1)
        with self.assertRaises(RuntimeError):
            guard.open_once()


class ContractRegressionTests(unittest.TestCase):
    def test_exact_candidate_grids_and_phase_b_scope(self) -> None:
        self.assertEqual(set(PHASE_A_CANDIDATES["slip"]), {"S1", "S2", "S3"})
        self.assertEqual(set(PHASE_A_CANDIDATES["support"]), {"P1", "P2", "P3"})
        self.assertEqual(set(PHASE_B_CANDIDATES["slip"]), {"SF1", "SF2", "SF3"})
        self.assertEqual(set(PHASE_B_CANDIDATES["support"]), {"PF1", "PF2", "PF3"})
        self.assertTrue(
            all("foot_imu12" in value for value in PHASE_B_CANDIDATES["slip"].values())
        )

    def test_config_hnm_gates_and_no_final_freeze(self) -> None:
        document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        hnm = document["hard_negative_mining"]
        self.assertEqual((hnm["rounds"], hnm["top_k_per_run"]), (2, 8))
        self.assertEqual(hnm["minimum_separation_ms"], 50)
        self.assertEqual(document["validation"]["persistence_ms"], 5)
        self.assertEqual(document["terrain_branch"]["deployment_scheme"], "left_only")
        self.assertEqual(document["phase_b"]["trigger"], "failed_phase_a_branches_only")
        self.assertEqual(
            document["regression"]["final_sensor_architecture_freeze"], "prohibited"
        )

    def test_fusion_and_cli_regression(self) -> None:
        self.assertTrue(fusion_regression()["passed"])
        cli = (ROOT / "scripts" / "fastreflex.py").read_text(encoding="utf-8")
        self.assertIn("TERRAIN_CONDITIONED_REFLEX_DETECTOR_DEVELOPMENT", cli)
        self.assertIn("run_terrain_conditioned_reflex_detector", cli)


if __name__ == "__main__":
    unittest.main()
