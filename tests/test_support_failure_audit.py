import tempfile
import unittest
from pathlib import Path

import numpy as np

from fastreflex.dataset.loader import Normalizer
from fastreflex.evaluation.reflex_event import EventRun
from fastreflex.evaluation.support_failure_audit import (
    _maximum_consecutive,
    assign_diagnostic_groups,
    assign_failure_modes,
    event_diagnostic_row,
    load_development_gates,
    validate_audit_splits,
)
from fastreflex.evaluation.terrain_conditioned_reflex import (
    SAND,
    BranchReplay,
    TerrainGateTrace,
    feature_schema_for_components,
)


def _run() -> EventRun:
    samples = 200
    imu = np.zeros((samples, 6), dtype=np.float32)
    fsr = np.ones((samples, 8), dtype=np.float32)
    return EventRun(
        run_id="dev_support",
        split="validation",
        source_terrain="concrete",
        target_terrain="sand",
        design_role="fall_domain",
        first_contact_sample=20,
        first_touchdown_sample=30,
        censor_sample=samples,
        outcome_diagnostic="fall",
        fall_sample_diagnostic=170,
        features={
            "PELVIS_IMU6": imu,
            "PELVIS_IMU6_FSR8": np.concatenate((imu, fsr), axis=1),
        },
        timestamp_us=np.arange(samples, dtype=np.int64) * 1000,
        slip_event_samples_per_foot=(None, None),
        support_event_samples_per_foot=(100, None),
        event_sample=100,
        event_type="SUPPORT",
        hard_stable_control=False,
        drift_m=np.zeros((samples, 2), dtype=np.float32),
        tangential_velocity_mps=np.zeros((samples, 2), dtype=np.float32),
        support_spread_m=np.zeros((samples, 2), dtype=np.float32),
        support_max_displacement_m=np.zeros((samples, 2), dtype=np.float32),
        loaded_contact=np.ones((samples, 2), dtype=bool),
        sink_pattern="left",
        support_pattern="uneven",
    )


def _gate(samples: int = 200) -> TerrainGateTrace:
    return TerrainGateTrace(
        state=np.full(samples, SAND, dtype=np.int8),
        update_samples=np.asarray((40,), dtype=np.int64),
        prediction_ids=np.asarray((3,), dtype=np.int8),
        prediction_probabilities=np.asarray(((0.0, 0.0, 0.0, 1.0),), dtype=np.float32),
        first_target_valid_sample=40,
        clean_event_count=1,
    )


class SupportFailureAuditTest(unittest.TestCase):
    def test_holdout_is_rejected_fail_closed(self) -> None:
        self.assertEqual(
            validate_audit_splits(("train", "validation")),
            ("train", "validation"),
        )
        with self.assertRaises(RuntimeError):
            validate_audit_splits(("validation", "holdout"))

    def test_only_development_gate_cache_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RuntimeError):
                load_development_gates(Path(folder) / "holdout", {})

    def test_maximum_consecutive_threshold_duration(self) -> None:
        self.assertEqual(
            _maximum_consecutive(np.asarray((False, True, True, False, True))), 2
        )
        self.assertEqual(_maximum_consecutive(np.zeros(3, dtype=bool)), 0)

    def test_event_diagnostic_reproduces_persistence_onset(self) -> None:
        run = _run()
        endpoints = np.arange(20, 200, dtype=np.int64)
        probability = np.zeros(len(endpoints), dtype=np.float64)
        probability[(endpoints >= 98) & (endpoints <= 102)] = 0.95
        replay = BranchReplay(
            endpoints=endpoints,
            probabilities=probability,
            terrain_state=np.full(len(endpoints), SAND, dtype=np.int8),
        )
        schema = feature_schema_for_components(("pelvis_imu6",))
        normalizer = Normalizer(
            mean=np.zeros(60, dtype=np.float32),
            std=np.ones(60, dtype=np.float32),
            sample_count=1,
            fit_run_ids=(run.run_id,),
            epsilon=1.0e-8,
        )
        row = event_diagnostic_row(
            run,
            _gate(),
            replay,
            normalizer,
            schema,
            threshold=0.94,
            persistence_ms=5,
        )
        self.assertTrue(row["detected"])
        self.assertEqual(row["detector_first_valid_output_sample"], 102)
        self.assertEqual(row["latency_ms"], 2)
        self.assertEqual(row["maximum_consecutive_threshold_above_ms"], 5)
        self.assertTrue(row["persistence_satisfied"])

    def test_groups_and_failure_modes_are_diagnostic_only(self) -> None:
        rows = [
            {
                "detected": True,
                "missed": False,
                "max_score_minus_threshold": 0.05,
                "raw_feature_z_energy_p90": 3.0,
                "all_feature_abs_z_p90": 2.0,
                "terrain_to_support_margin_ms": 100,
            },
            {
                "detected": True,
                "missed": False,
                "max_score_minus_threshold": 0.01,
                "raw_feature_z_energy_p90": 2.0,
                "all_feature_abs_z_p90": 1.5,
                "terrain_to_support_margin_ms": 110,
            },
            {
                "detected": False,
                "missed": True,
                "max_score_minus_threshold": -0.20,
                "raw_feature_z_energy_p90": 0.5,
                "all_feature_abs_z_p90": 0.5,
                "terrain_to_support_margin_ms": 115,
                "active_gate_samples_in_valid_window": 81,
                "persistence_requirement_ms": 5,
                "gating_state_at_peak": "SAND",
                "gating_state_at_support": "SAND",
                "detector_max_score": 0.74,
                "active_gate_max_score": 0.74,
                "threshold": 0.94,
                "maximum_consecutive_threshold_above_ms": 0,
                "raw_maximum_consecutive_threshold_above_ms": 0,
                "raw_persistence_satisfied": False,
                "first_late_output_sample": None,
                "raw_first_late_output_sample": None,
            },
        ]
        grouping = assign_diagnostic_groups(rows)
        failure = assign_failure_modes(rows)
        self.assertTrue(grouping["low_margin_success_definition"].startswith("detected"))
        self.assertIn("SCORE_INSUFFICIENT", rows[-1]["failure_modes"])
        self.assertTrue(failure["classification_is_diagnostic_only"])


if __name__ == "__main__":
    unittest.main()
