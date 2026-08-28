"""Privileged full-state stability ground-truth contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

from fastreflex.evaluation.stability_ground_truth import (
    CANDIDATE_FEATURE_NAMES,
    CANDIDATE_ORDER,
    FULL_STATE,
    LOWER_BODY_FEATURE_NAMES,
    LOWER_BODY_JOINT_NAMES,
    LOWER_BODY_STATE,
    PELVIS_STATE,
    CandidateDistanceModel,
    DistanceTrace,
    PhaseDistanceDistribution,
    StateDistanceRun,
    _load_calibration_specs,
    _viewer_replay,
    canonical_sha256,
    deterministic_phase_sample_indices,
    extract_candidate_features,
    fit_candidate_distance_model,
    future_suffix_independence,
    lower_body_state_addresses,
    mahalanobis_distance,
    regularize_covariance,
    score_candidate_distance,
    select_calibration_candidate,
    validate_experiment_design,
)
from fastreflex.evaluation.transition_scenarios import fusion_regression
from fastreflex.simulation.g1 import load_g1_model
from fastreflex.simulation.stability import (
    DOUBLE_SUPPORT,
    LEFT_SINGLE_SUPPORT,
    NO_SUPPORT,
    RIGHT_SINGLE_SUPPORT,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "20260828_full_state_stability_ground_truth_sanity.yaml"
)


def _manual_model(threshold: float = 2.5) -> CandidateDistanceModel:
    dimension = len(CANDIDATE_FEATURE_NAMES[PELVIS_STATE])
    identity = np.eye(dimension)
    distribution = PhaseDistanceDistribution(
        mean=np.zeros(dimension),
        standard_deviation=np.ones(dimension),
        covariance=identity,
        regularized_covariance=identity,
        precision=identity,
        distance_threshold=threshold,
        fit_sample_count=100,
    )
    return CandidateDistanceModel(
        candidate=PELVIS_STATE,
        feature_names=CANDIDATE_FEATURE_NAMES[PELVIS_STATE],
        phase_distributions={phase: distribution for phase in (1, 2, 3)},
        fit_run_ids=("stable",),
        stride_samples=10,
        per_run_per_phase_cap=256,
        covariance_lambda=0.05,
        covariance_epsilon=1.0e-6,
        threshold_quantile=0.995,
    )


class FullStateStabilityGroundTruthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with CONFIG.open("r", encoding="utf-8") as stream:
            cls.document = yaml.safe_load(stream)
        cls.calibration_specs, cls.source_document = _load_calibration_specs(
            cls.document, ROOT
        )

    def test_candidate_schema_order_dimensions_and_leakage_boundary(self) -> None:
        design = validate_experiment_design(
            self.document, self.calibration_specs, self.source_document
        )
        self.assertEqual(
            design["candidate_dimensions"],
            {PELVIS_STATE: 9, LOWER_BODY_STATE: 24, FULL_STATE: 37},
        )
        forbidden = {"terrain", "fall", "slip", "sink", "run_id", "timestamp", "mos"}
        for candidate in CANDIDATE_ORDER:
            configured = tuple(
                self.document["candidate_definitions"][candidate]["feature_order"]
            )
            self.assertEqual(configured, CANDIDATE_FEATURE_NAMES[candidate])
            self.assertFalse(
                any(token in feature for token in forbidden for feature in configured)
            )
            self.assertTrue(all("imu" not in feature for feature in configured))
        self.assertNotIn("raw_mos_m", CANDIDATE_FEATURE_NAMES[FULL_STATE])

    def test_lower_body_joint_selection_uses_named_legs_and_excludes_upper_body(
        self,
    ) -> None:
        model, _ = load_g1_model("concrete")
        qpos, qvel = lower_body_state_addresses(model)
        np.testing.assert_array_equal(qpos, np.arange(7, 19))
        np.testing.assert_array_equal(qvel, np.arange(6, 18))
        self.assertEqual(len(LOWER_BODY_JOINT_NAMES), 12)
        self.assertTrue(
            all("shoulder" not in name for name in LOWER_BODY_FEATURE_NAMES)
        )
        self.assertTrue(all("wrist" not in name for name in LOWER_BODY_FEATURE_NAMES))
        self.assertTrue(all("waist" not in name for name in LOWER_BODY_FEATURE_NAMES))

    def test_feature_extraction_preserves_declared_joint_order(self) -> None:
        samples = 3
        qpos = np.tile(np.arange(36, dtype=np.float64), (samples, 1))
        qvel = np.tile(np.arange(35, dtype=np.float64) + 100.0, (samples, 1))
        result = SimpleNamespace(
            runtime=SimpleNamespace(sequence=np.arange(samples)),
            state_trace=SimpleNamespace(robot_qpos=qpos, robot_qvel=qvel),
            diagnostics=SimpleNamespace(
                pelvis_roll_rad=np.zeros(samples),
                pelvis_pitch_rad=np.zeros(samples),
                pelvis_angular_velocity_rad_s=np.zeros((samples, 3)),
                pelvis_linear_velocity_m_s=np.zeros((samples, 3)),
                pelvis_world_z_m=np.ones(samples),
            ),
            stability=SimpleNamespace(
                gait_phase=np.full(samples, DOUBLE_SUPPORT),
                com_velocity_xyz_m_s=np.zeros((samples, 3)),
                support_height_m=np.ones(samples),
            ),
        )
        lower = extract_candidate_features(
            result, LOWER_BODY_STATE, np.arange(7, 19), np.arange(6, 18)
        )
        pelvis = extract_candidate_features(
            result, PELVIS_STATE, np.arange(7, 19), np.arange(6, 18)
        )
        full = extract_candidate_features(
            result, FULL_STATE, np.arange(7, 19), np.arange(6, 18)
        )
        np.testing.assert_array_equal(lower[0, :4], (7.0, 106.0, 8.0, 107.0))
        self.assertEqual(lower.shape, (samples, 24))
        self.assertEqual(pelvis.shape, (samples, 9))
        self.assertEqual(full.shape, (samples, 37))
        np.testing.assert_array_equal(full[:, :9], pelvis)
        np.testing.assert_array_equal(full[:, 9:33], lower)
        np.testing.assert_allclose(full[:, -4:-1], 0.0)
        np.testing.assert_allclose(full[:, -1], 1.0)

    def test_run_phase_cap_is_deterministic_evenly_spread_and_time_strided(
        self,
    ) -> None:
        phase = np.full(1000, LEFT_SINGLE_SUPPORT)
        first = deterministic_phase_sample_indices(phase, LEFT_SINGLE_SUPPORT, 10, 16)
        second = deterministic_phase_sample_indices(phase, LEFT_SINGLE_SUPPORT, 10, 16)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 16)
        self.assertTrue(np.all(first % 10 == 0))
        self.assertEqual(first[0], 0)
        self.assertEqual(first[-1], 990)

    def test_covariance_regularization_and_mahalanobis_are_exact(self) -> None:
        covariance = np.asarray(((2.0, 1.0), (1.0, 4.0)))
        regularized = regularize_covariance(covariance, 0.05, 1.0e-6)
        expected = np.asarray(((2.000001, 0.95), (0.95, 4.000001)))
        np.testing.assert_allclose(regularized, expected, atol=1.0e-12)
        distance = mahalanobis_distance(
            np.asarray(((3.0, 4.0),)),
            np.zeros(2),
            np.ones(2),
            np.eye(2),
        )
        self.assertAlmostEqual(float(distance[0]), 5.0)

    def test_stable_only_phase_fit_and_q995_threshold_are_deterministic(self) -> None:
        rng = np.random.default_rng(20260828)
        features = rng.normal(size=(1200, 9))
        phase = np.tile(
            np.repeat((LEFT_SINGLE_SUPPORT, RIGHT_SINGLE_SUPPORT, DOUBLE_SUPPORT), 100),
            4,
        )
        kwargs = dict(
            stride_samples=2,
            per_run_per_phase_cap=128,
            standard_deviation_floor=1.0e-8,
            covariance_lambda=0.05,
            covariance_epsilon=1.0e-6,
            threshold_quantile=0.995,
        )
        first = fit_candidate_distance_model(
            PELVIS_STATE, {"stable": features}, {"stable": phase}, **kwargs
        )
        second = fit_candidate_distance_model(
            PELVIS_STATE, {"stable": features}, {"stable": phase}, **kwargs
        )
        self.assertEqual(first.fit_run_ids, ("stable",))
        for phase_id in (LEFT_SINGLE_SUPPORT, RIGHT_SINGLE_SUPPORT, DOUBLE_SUPPORT):
            left = first.phase_distributions[phase_id]
            right = second.phase_distributions[phase_id]
            self.assertEqual(left.fit_sample_count, 128)
            self.assertEqual(left.distance_threshold, right.distance_threshold)
            np.testing.assert_array_equal(left.mean, right.mean)

    def test_contact_gate_resets_twenty_ms_persistence_and_keeps_diagnostic(
        self,
    ) -> None:
        model = _manual_model()
        features = np.zeros((70, 9))
        features[:, 0] = 4.0
        phase = np.full(70, LEFT_SINGLE_SUPPORT)
        ungated = score_candidate_distance(features, phase, model, 20)
        gated = score_candidate_distance(
            features, phase, model, 20, eligible_from_sample=30
        )
        self.assertEqual(np.flatnonzero(ungated.onset).tolist(), [19])
        self.assertEqual(np.flatnonzero(gated.onset).tolist(), [49])
        self.assertFalse(np.any(gated.candidate[:30]))

    def test_selection_qualification_and_near_tie_prefer_simplicity(self) -> None:
        def metrics(fp: float, coverage: float) -> dict[str, object]:
            return {
                "stable_false_instability_run_rate": fp,
                "fall_coverage": coverage,
                "by_terrain": {
                    "ice": {"fall_coverage": coverage},
                    "sand": {"fall_coverage": coverage},
                },
                "fall_lead_ms": {"p50": 400.0},
            }

        result = select_calibration_candidate(
            {
                PELVIS_STATE: metrics(0.05, 0.85),
                LOWER_BODY_STATE: metrics(0.02, 0.89),
                FULL_STATE: metrics(0.01, 0.90),
            },
            self.document["calibration"]["qualification"],
            self.document["calibration"]["near_tie"],
        )
        self.assertEqual(result["selected"], PELVIS_STATE)
        self.assertEqual(result["reason"], "near_tie_simplicity")

    def test_freeze_sha_and_future_suffix_are_deterministic(self) -> None:
        payload = {"candidate": PELVIS_STATE, "thresholds": [1.0, 2.0, 3.0]}
        self.assertEqual(canonical_sha256(payload), canonical_sha256(dict(payload)))
        model = _manual_model()
        features = np.zeros((80, 9))
        features[20:60, 0] = 4.0
        phase = np.full(80, LEFT_SINGLE_SUPPORT)
        primary = score_candidate_distance(
            features, phase, model, 20, eligible_from_sample=10
        )
        result = SimpleNamespace(
            stability=SimpleNamespace(gait_phase=phase),
            runtime=SimpleNamespace(timestamp_us=(np.arange(80) + 1) * 1000),
        )
        run = StateDistanceRun(
            specification={"id": "fall"},
            result=result,
            outcome="VALID_FALL",
            features=features,
            first_contact_sample=10,
            ungated=DistanceTrace(
                primary.distance,
                primary.threshold,
                primary.candidate,
                primary.active,
                primary.onset,
            ),
            primary=primary,
        )
        regression = future_suffix_independence(run, model, 20)
        self.assertTrue(regression["passed"])
        self.assertTrue(regression["future_fall_is_not_a_score_input"])

    def test_calibration_fresh_disjoint_terrain_fusion_and_viewer_contract(
        self,
    ) -> None:
        design = validate_experiment_design(
            self.document, self.calibration_specs, self.source_document
        )
        self.assertTrue(design["calibration_prior_validation_fresh_disjoint"])
        self.assertEqual(design["fresh_runs"], 48)
        self.assertTrue(fusion_regression()["passed"])
        rows = []
        for source, target, outcome in (
            ("concrete", "ice", "stable"),
            ("concrete", "ice", "fall"),
            ("marble", "ice", "fall"),
            ("concrete", "sand", "stable"),
            ("concrete", "sand", "fall"),
            ("marble", "sand", "fall"),
        ):
            rows.append(
                {
                    "run_id": f"{source}_{target}_{outcome}",
                    "source_terrain": source,
                    "target_terrain": target,
                    "transition": f"{source}->{target}",
                    "observed_outcome": outcome,
                    "first_target_contact_ms": 1000.0,
                    "physical_slip_onset_ms": None,
                    "physical_sink_onset_ms": None,
                    "replay_sample_ms": 2000.0,
                    "replay_sample_kind": "EPISODE_MAXIMUM_SCORE",
                    "support_phase_at_replay": "DOUBLE_SUPPORT",
                    "distance_at_replay": 4.0,
                    "threshold_at_replay": 3.0,
                    "t_instability_ms": None,
                    "t_fall_ms": None,
                }
            )
        viewer, text = _viewer_replay(rows)
        self.assertTrue(viewer["passed"])
        self.assertFalse(viewer["physics_mutation"])
        self.assertIn("PHASE_THRESHOLD=3.0", text)

    def test_canonical_cli_dispatches_the_full_state_experiment(self) -> None:
        cli = (ROOT / "scripts" / "fastreflex.py").read_text(encoding="utf-8")
        self.assertIn(
            'experiment_id == "FULL_STATE_STABILITY_GROUND_TRUTH_SANITY"', cli
        )
        self.assertIn("run_full_state_stability_ground_truth_sanity", cli)


if __name__ == "__main__":
    unittest.main()
