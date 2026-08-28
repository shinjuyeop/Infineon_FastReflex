"""Ground-truth-only walking-stability experiment contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

from fastreflex.evaluation.walking_stability import (
    OracleRun,
    _coverage_summary,
    _viewer_replay,
    future_suffix_independence,
    valid_prefall_instability,
    validate_experiment_design,
)
from fastreflex.simulation.stability import (
    DOUBLE_SUPPORT,
    LEFT_SINGLE_SUPPORT,
    RIGHT_SINGLE_SUPPORT,
    PhaseEnvelope,
    StabilityDiagnostics,
    StableCalibrationRun,
    detect_instability,
    fit_phase_envelope,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "20260828_walking_stability_ground_truth_sanity.yaml"
)


def _diagnostics(margins: np.ndarray, phases: np.ndarray) -> StabilityDiagnostics:
    samples = len(margins)
    points = np.zeros((samples, 2, 4, 3), dtype=np.float64)
    return StabilityDiagnostics(
        com_xyz_m=np.zeros((samples, 3)),
        com_velocity_xyz_m_s=np.zeros((samples, 3)),
        support_height_m=np.ones(samples),
        foot_support_points_xyz_m=points,
        gait_phase=np.asarray(phases, dtype=np.int8),
        xcom_xy_m=np.zeros((samples, 2)),
        raw_margin_of_stability_m=np.asarray(margins, dtype=np.float64),
    )


class WalkingStabilityGroundTruthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with CONFIG.open("r", encoding="utf-8") as stream:
            cls.document = yaml.safe_load(stream)

    def test_config_freezes_oracle_and_fresh_matrix_before_execution(self) -> None:
        design = validate_experiment_design(self.document)
        self.assertTrue(design["passed"])
        self.assertEqual(design["calibration_runs"], 36)
        self.assertEqual(design["fresh_runs"], 40)
        self.assertTrue(design["calibration_validation_disjoint"])
        self.assertTrue(design["fresh_signatures_unique"])
        self.assertTrue(
            all(count == 5 for count in design["fresh_group_counts"].values())
        )
        oracle = self.document["stability_oracle"]
        self.assertEqual(oracle["phase_lower_quantile"], 0.005)
        self.assertEqual(oracle["fixed_margin_m"], 0.010)
        self.assertEqual(oracle["persistence_ms"], 20)
        self.assertEqual(oracle["runtime_inputs"], [])
        self.assertFalse(oracle["future_fall_dependency"])
        self.assertEqual(oracle["post_validation_retuning"], "prohibited")

    def test_calibration_declares_hard_ice_and_sand_stable_evidence(self) -> None:
        hard = self.document["calibration"]["hard_stable_runs"]
        transitions = self.document["calibration"]["transition_runs"]
        self.assertEqual(
            {run["target_terrain"] for run in hard}, {"concrete", "marble"}
        )
        stable = [
            run for run in transitions if run["prior_observed_outcome"] == "stable"
        ]
        self.assertEqual({run["target_terrain"] for run in stable}, {"ice", "sand"})
        self.assertEqual(
            {run["source_terrain"] for run in stable}, {"concrete", "marble"}
        )

    def test_envelope_uses_observed_stable_ice_and_sand_but_rejects_fall(self) -> None:
        phases = np.tile(
            (LEFT_SINGLE_SUPPORT, RIGHT_SINGLE_SUPPORT, DOUBLE_SUPPORT), 20
        )
        ice = StableCalibrationRun(
            "ice_stable",
            _diagnostics(np.linspace(-0.04, 0.02, len(phases)), phases),
            observed_stable=True,
            observed_fall=False,
            source_terrain="concrete",
            target_terrain="ice",
        )
        sand = StableCalibrationRun(
            "sand_stable",
            _diagnostics(np.linspace(-0.03, 0.03, len(phases)), phases),
            observed_stable=True,
            observed_fall=False,
            source_terrain="marble",
            target_terrain="sand",
        )
        first = fit_phase_envelope((ice, sand), 0.005)
        second = fit_phase_envelope((ice, sand), 0.005)
        self.assertEqual(first.calibration_run_ids, ("ice_stable", "sand_stable"))
        self.assertEqual(first.lower_bound_m, second.lower_bound_m)
        falling = StableCalibrationRun(
            "fall",
            ice.diagnostics,
            observed_stable=False,
            observed_fall=True,
            source_terrain="concrete",
            target_terrain="ice",
        )
        with self.assertRaises(ValueError):
            fit_phase_envelope((ice, sand, falling), 0.005)

    def test_future_suffix_does_not_change_decided_prefix(self) -> None:
        samples = 80
        diagnostics = _diagnostics(
            np.r_[np.zeros(20), np.full(40, -0.03), np.zeros(20)],
            np.full(samples, LEFT_SINGLE_SUPPORT),
        )
        envelope = PhaseEnvelope({LEFT_SINGLE_SUPPORT: 0.0}, 0.005, ("stable",))
        primary = detect_instability(
            diagnostics,
            envelope,
            0.010,
            20,
            eligible_from_sample=10,
        )
        result = SimpleNamespace(
            stability=diagnostics,
            runtime=SimpleNamespace(timestamp_us=(np.arange(samples) + 1) * 1000),
        )
        run = OracleRun(
            specification={"id": "fall"},
            result=result,
            outcome="VALID_FALL",
            first_contact_sample=10,
            ungated=primary,
            primary=primary,
        )
        regression = future_suffix_independence(run, envelope, 0.010, 20)
        self.assertTrue(regression["passed"])
        self.assertTrue(regression["future_fall_and_postfall_are_not_oracle_inputs"])

    def test_only_strictly_prefall_instability_is_a_valid_fall_detection(self) -> None:
        self.assertTrue(valid_prefall_instability(19, 20))
        self.assertFalse(valid_prefall_instability(20, 20))
        self.assertFalse(valid_prefall_instability(21, 20))
        self.assertFalse(valid_prefall_instability(None, 20))

    def test_acceptance_summary_uses_observed_outcome_and_prefall_validity(
        self,
    ) -> None:
        rows = []
        for index in range(10):
            rows.append(
                {
                    "run_id": f"stable_{index}",
                    "source_terrain": "concrete" if index < 5 else "marble",
                    "target_terrain": "ice" if index % 2 == 0 else "sand",
                    "observed_outcome": "stable",
                    "false_instability": index == 0,
                    "valid_prefall_detection": False,
                    "pretransition_false_instability": False,
                    "fall_lead_ms": None,
                }
            )
        for index in range(20):
            rows.append(
                {
                    "run_id": f"fall_{index}",
                    "source_terrain": "concrete" if index < 10 else "marble",
                    "target_terrain": "ice" if index % 2 == 0 else "sand",
                    "observed_outcome": "fall",
                    "false_instability": False,
                    "valid_prefall_detection": index < 17,
                    "pretransition_false_instability": False,
                    "fall_lead_ms": 300.0 if index < 17 else None,
                }
            )
        summary = _coverage_summary(rows)
        self.assertEqual(summary["stable_false_instability_run_rate"], 0.10)
        self.assertEqual(summary["fall_coverage"], 0.85)
        self.assertEqual(summary["fall_lead_ms"]["p50"], 300.0)

    def test_viewer_replay_is_evaluation_value_parity_only(self) -> None:
        rows = []
        for terrain in ("ice", "sand"):
            for outcome in ("stable", "fall"):
                rows.append(
                    {
                        "run_id": f"{terrain}_{outcome}",
                        "source_terrain": "concrete",
                        "target_terrain": terrain,
                        "observed_outcome": outcome,
                        "first_target_contact_ms": 1000.0,
                        "support_phase_at_instability": None,
                        "raw_mos_at_instability_m": None,
                        "residual_at_instability_m": None,
                        "minimum_raw_mos_m": -0.10,
                        "minimum_stability_residual_m": -0.01,
                        "t_instability_ms": None,
                        "t_fall_ms": None,
                    }
                )
        viewer, text = _viewer_replay(rows)
        self.assertTrue(viewer["passed"])
        self.assertFalse(viewer["physics_mutation"])
        self.assertIn("STABILITY_RESIDUAL_M", text)


if __name__ == "__main__":
    unittest.main()
