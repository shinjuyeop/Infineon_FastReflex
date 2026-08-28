"""Contracts for the terrain-independent continuous Slip detector."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from fastreflex.evaluation.continuous_slip_reflex import (
    PHASE_A_CANDIDATES,
    PHASE_B_CANDIDATES,
    _negative_interval_mask,
    continuous_negative_candidates,
    continuous_positive_endpoints,
    continuous_sustained_alert,
    evaluate_continuous_replays,
    extract_continuous_slip_features,
    feature_schema_for_components,
    foot_imu_feature_base,
    mine_hard_negative_endpoints,
    reflex_decision,
    slip_event_sample,
    temporal_expansion,
    threshold_values,
    SlipReplay,
)
from fastreflex.evaluation.reflex_event import (
    EVENT_TYPE_NONE,
    EVENT_TYPE_SLIP,
    EVENT_TYPE_SUPPORT,
    EventHoldoutGuard,
    EventRun,
    persistent_threshold_events,
)
from fastreflex.evaluation.terrain_conditioned_reflex import (
    CONCRETE,
    ICE,
    SAND,
    UNKNOWN,
)
from fastreflex.evaluation.transition_scenarios import fusion_regression


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs" / "experiment" / "20260829_continuous_slip_reflex_detector.yaml"
)


def synthetic_run(
    *,
    run_id: str = "synthetic",
    slip: tuple[int | None, int | None] = (100, None),
    support: tuple[int | None, int | None] = (None, None),
    outcome: str = "VALID_STABLE",
    hard: bool = False,
) -> EventRun:
    samples = 220
    time = np.arange(samples, dtype=np.float32)
    imu = np.column_stack(
        (
            time,
            time * 0.1,
            np.ones(samples),
            time * 0.01,
            time * 0.02,
            time * 0.03,
        )
    ).astype(np.float32)
    fsr = np.tile(np.arange(1, 9, dtype=np.float32), (samples, 1))
    slip_values = [value for value in slip if value is not None]
    support_values = [value for value in support if value is not None]
    if slip_values:
        event_type = EVENT_TYPE_SLIP
        event = min(slip_values)
    elif support_values:
        event_type = EVENT_TYPE_SUPPORT
        event = min(support_values)
    else:
        event_type = EVENT_TYPE_NONE
        event = None
    loaded = np.zeros((samples, 2), dtype=bool)
    loaded[20:60, 0] = True
    loaded[60:100] = True
    loaded[100:150, 1] = True
    loaded[150:190] = True
    zeros2 = np.zeros((samples, 2), dtype=np.float32)
    return EventRun(
        run_id=run_id,
        split="train",
        source_terrain="concrete",
        target_terrain="ice" if slip_values else "sand",
        design_role="stable",
        first_contact_sample=20,
        first_touchdown_sample=20,
        censor_sample=samples,
        outcome_diagnostic=outcome,
        fall_sample_diagnostic=210 if outcome == "VALID_FALL" else None,
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
        drift_m=zeros2,
        tangential_velocity_mps=zeros2,
        support_spread_m=zeros2,
        support_max_displacement_m=zeros2,
        loaded_contact=loaded,
        sink_pattern="uniform",
        support_pattern="balanced_soft",
    )


class FrozenOracleTests(unittest.TestCase):
    def test_fifty_mm_three_ms_oracle_is_unchanged(self) -> None:
        drift = np.zeros((20, 2), dtype=np.float64)
        drift[7:10, 1] = 0.050
        valid = np.ones((20, 2), dtype=bool)
        episodes = np.zeros((20, 2), dtype=np.int64)
        _, onset = persistent_threshold_events(drift, valid, episodes, 0.050, 3)
        self.assertEqual(int(np.flatnonzero(onset[:, 1])[0]), 9)

    def test_any_slip_and_bilateral_semantics(self) -> None:
        self.assertEqual(slip_event_sample(synthetic_run(slip=(100, 110))), 100)
        self.assertEqual(slip_event_sample(synthetic_run(slip=(None, 105))), 105)
        self.assertIsNone(slip_event_sample(synthetic_run(slip=(None, None))))

    def test_fall_and_recovery_do_not_change_labels(self) -> None:
        stable = synthetic_run(outcome="VALID_STABLE")
        fall = synthetic_run(outcome="VALID_FALL")
        np.testing.assert_array_equal(
            continuous_positive_endpoints(stable, 20),
            continuous_positive_endpoints(fall, 20),
        )


class FeatureAndLabelTests(unittest.TestCase):
    def test_feature_dimensions_order_and_forbidden_metadata(self) -> None:
        self.assertEqual(len(feature_schema_for_components(("pelvis_imu6",))), 80)
        self.assertEqual(len(feature_schema_for_components(("fsr8",))), 240)
        self.assertEqual(len(feature_schema_for_components(("foot_imu12",))), 176)
        fusion = feature_schema_for_components(("pelvis_imu6", "fsr8"))
        self.assertEqual(len(fusion), 320)
        self.assertFalse(
            any(
                token in name
                for name in fusion
                for token in ("terrain", "fall", "recovery", "slip_clock")
            )
        )

    def test_derived_features_are_causal_finite_and_deterministic(self) -> None:
        run = synthetic_run()
        original, names = extract_continuous_slip_features(run, ("pelvis_imu6", "fsr8"))
        changed_imu = run.features["PELVIS_IMU6"].copy()
        changed_fusion = run.features["PELVIS_IMU6_FSR8"].copy()
        changed_imu[121:] = 999.0
        changed_fusion[121:] = 999.0
        changed = EventRun(
            **{
                **run.__dict__,
                "features": {
                    "PELVIS_IMU6": changed_imu,
                    "PELVIS_IMU6_FSR8": changed_fusion,
                },
            }
        )
        altered, altered_names = extract_continuous_slip_features(
            changed, ("pelvis_imu6", "fsr8")
        )
        np.testing.assert_array_equal(original[:121], altered[:121])
        self.assertEqual(names, altered_names)
        self.assertTrue(np.all(np.isfinite(original)))

    def test_temporal_transform_order_and_foot_bilateral_terms(self) -> None:
        base = np.arange(30, dtype=np.float32).reshape(10, 3)
        expanded, names = temporal_expansion(base, ("a", "b", "c"))
        self.assertEqual(expanded.shape, (10, 24))
        self.assertTrue(names[0].startswith("base_"))
        self.assertTrue(names[-1].startswith("causal_variance_10ms_"))
        foot, foot_names = foot_imu_feature_base(np.ones((10, 12), np.float32))
        self.assertEqual(foot.shape, (10, 22))
        self.assertIn("bilateral_accel_norm_difference", foot_names)

    def test_positive_negative_and_support_exclusion_boundaries(self) -> None:
        slip = synthetic_run(slip=(100, None))
        self.assertEqual(
            tuple(continuous_positive_endpoints(slip, 20)[[0, -1]]), (70, 140)
        )
        self.assertEqual(int(continuous_negative_candidates(slip, 20)[-1]), 60)
        support = synthetic_run(slip=(None, None), support=(120, None))
        self.assertEqual(int(continuous_negative_candidates(support, 20)[-1]), 90)
        self.assertEqual(len(continuous_positive_endpoints(support, 20)), 0)


class ContinuousReplayAndMiningTests(unittest.TestCase):
    def test_detector_and_reflex_do_not_wait_for_terrain(self) -> None:
        probability = np.zeros(100)
        probability[20:25] = 1.0
        alert, onset = continuous_sustained_alert(probability, 0.5, 5)
        self.assertEqual(int(np.flatnonzero(onset)[0]), 24)
        self.assertTrue(alert[24])
        self.assertEqual(
            reflex_decision(
                slip_alert=True, support_alert=False, terrain_state=UNKNOWN
            ),
            (True, "GENERIC_SLIP_DISTURBANCE"),
        )
        self.assertEqual(
            reflex_decision(
                slip_alert=True, support_alert=False, terrain_state=CONCRETE
            )[0],
            True,
        )
        self.assertEqual(
            reflex_decision(slip_alert=True, support_alert=False, terrain_state=ICE),
            (True, "SLIP_RISK"),
        )

    def test_sand_support_remains_conditioned(self) -> None:
        self.assertEqual(
            reflex_decision(slip_alert=False, support_alert=True, terrain_state=SAND),
            (True, "SUPPORT_RISK"),
        )
        self.assertEqual(
            reflex_decision(
                slip_alert=False, support_alert=True, terrain_state=UNKNOWN
            ),
            (False, "NORMAL"),
        )

    def test_premature_does_not_hide_later_valid_alert(self) -> None:
        run = synthetic_run()
        endpoints = np.arange(220, dtype=np.int64)
        probability = np.zeros(220)
        probability[40:45] = 1.0
        probability[95:100] = 1.0
        metrics = evaluate_continuous_replays(
            {run.run_id: run},
            {run.run_id: SlipReplay(endpoints, probability)},
            0.5,
            5,
        )
        row = metrics["event_rows"][0]
        self.assertTrue(row["premature_alert"])
        self.assertTrue(row["valid_detection"])
        self.assertEqual(row["latency_ms"], -1)

    def test_support_cross_trigger_is_not_system_false_reflex(self) -> None:
        run = synthetic_run(slip=(None, None), support=(100, None))
        probability = np.zeros(220)
        probability[95:100] = 1.0
        metrics = evaluate_continuous_replays(
            {run.run_id: run},
            {run.run_id: SlipReplay(np.arange(220), probability)},
            0.5,
            5,
        )
        self.assertEqual(metrics["hazard_cross_trigger_count"], 1)
        self.assertEqual(metrics["system_false_reflex_count"], 0)

    def test_hnm_is_deterministic_train_contract(self) -> None:
        endpoints = np.arange(0, 1000, 5, dtype=np.int64)
        probability = np.linspace(0.0, 1.0, len(endpoints))
        first = mine_hard_negative_endpoints(endpoints, probability)
        second = mine_hard_negative_endpoints(endpoints, probability)
        np.testing.assert_array_equal(first, second)
        self.assertLessEqual(len(first), 12)
        self.assertTrue(all(b - a >= 30 for a, b in zip(first, first[1:])))

    def test_negative_mask_excludes_physical_hazard_region(self) -> None:
        run = synthetic_run(slip=(None, None), support=(100, None))
        endpoints = np.arange(220)
        mask = _negative_interval_mask(run, endpoints)
        self.assertTrue(mask[69])
        self.assertFalse(mask[70])


class ProtocolRegressionTests(unittest.TestCase):
    def test_candidate_grids_and_hnm_protocol_are_exact(self) -> None:
        self.assertEqual(set(PHASE_A_CANDIDATES), {"A1", "A2", "A3"})
        self.assertEqual(set(PHASE_B_CANDIDATES), {"B1", "B2", "B3", "B4"})
        self.assertTrue(all("foot_imu12" in row for row in PHASE_B_CANDIDATES.values()))
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        hnm = document["hard_negative_mining"]
        self.assertEqual(hnm["training_rounds"], [0, 1, 2, 3])
        self.assertEqual((hnm["iterations"], hnm["top_k_per_run"]), (3, 12))
        self.assertEqual(hnm["minimum_separation_ms"], 30)
        self.assertEqual(document["validation"]["persistence_ms"], 5)

    def test_threshold_and_holdout_guard_are_frozen(self) -> None:
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        values = threshold_values(document["validation"]["threshold_grid"])
        self.assertEqual((len(values), values[0], values[-1]), (45, 0.10, 0.98))
        guard = EventHoldoutGuard()
        with self.assertRaises(RuntimeError):
            guard.require_open()
        guard.open_once()
        self.assertEqual(guard.open_count, 1)
        with self.assertRaises(RuntimeError):
            guard.open_once()

    def test_frozen_support_and_no_final_architecture_freeze(self) -> None:
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        support = document["frozen_support"]
        self.assertEqual(
            (
                support["components"],
                support["model_family"],
                support["history_ms"],
                support["probability_threshold"],
                support["persistence_ms"],
            ),
            (["pelvis_imu6"], "gru", 20, 0.94, 5),
        )
        self.assertEqual(
            document["phase_b"]["trigger"], "all_phase_a_candidates_fail_validation"
        )
        self.assertEqual(
            document["regression"]["final_sensor_architecture_freeze"], "prohibited"
        )

    def test_fusion_and_cli_regression(self) -> None:
        self.assertTrue(fusion_regression()["passed"])
        cli = (ROOT / "scripts" / "fastreflex.py").read_text(encoding="utf-8")
        self.assertIn("CONTINUOUS_SLIP_REFLEX_DETECTOR_DEVELOPMENT", cli)
        self.assertIn("run_continuous_slip_reflex_detector", cli)


if __name__ == "__main__":
    unittest.main()
