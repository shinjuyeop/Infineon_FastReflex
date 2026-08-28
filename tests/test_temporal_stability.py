"""Temporal walking fall-risk observability contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

from fastreflex.dataset.loader import Normalizer
from fastreflex.evaluation.stability_temporal import (
    IMU6_FEATURE_NAMES,
    PRIVILEGED_FULL_STATE,
    REPRESENTATION_FEATURE_NAMES,
    RUNTIME_IMU6,
    TEMPORAL_FULL_STATE_FEATURE_NAMES,
    HoldoutGuard,
    TemporalRun,
    _load_scenario_specs,
    binary_auprc,
    binary_auroc,
    binary_metrics,
    build_matched_pairs,
    causal_last_valid_fill,
    causal_window_indices,
    fit_train_normalizer,
    horizon_fixed_label,
    materialize_matched_windows,
    privileged_temporal_features,
    select_history_and_horizon,
    validate_temporal_design,
)
from fastreflex.evaluation.transition_scenarios import (
    VALID_FALL,
    VALID_STABLE,
    fusion_regression,
)
from fastreflex.simulation.stability import DOUBLE_SUPPORT, NO_SUPPORT


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "20260828_temporal_stability_separability_audit.yaml"
)


def _fake_run(
    run_id: str,
    outcome: str,
    *,
    split: str = "train",
    source: str = "concrete",
    target: str = "sand",
    speed: float = 0.25,
    contact: int = 100,
    fall: int | None = None,
    value: float = 1.0,
) -> TemporalRun:
    samples = 900
    features = {
        PRIVILEGED_FULL_STATE: np.full((samples, 40), value, dtype=np.float32),
        RUNTIME_IMU6: np.full((samples, 6), value, dtype=np.float32),
    }
    return TemporalRun(
        run_id=run_id,
        split=split,
        source_terrain=source,
        target_terrain=target,
        speed_mps=speed,
        outcome=outcome,
        first_contact_sample=contact,
        fall_sample=fall,
        gait_phase=np.full(samples, DOUBLE_SUPPORT, dtype=np.int8),
        features=features,
        timestamp_us=(np.arange(samples) + 1) * 1000,
        slip_sample=None,
        sink_sample=None,
        maximum_support_deformation_m=0.0,
        hard_stable_control=False,
    )


class TemporalStabilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with CONFIG.open("r", encoding="utf-8") as stream:
            cls.document = yaml.safe_load(stream)
        cls.primary, cls.controls = _load_scenario_specs(cls.document, ROOT)

    def test_design_freezes_run_disjoint_split_offsets_histories_and_schemas(
        self,
    ) -> None:
        design = validate_temporal_design(self.document, self.primary, self.controls)
        self.assertEqual(design["primary_runs"], 78)
        self.assertEqual(
            design["split_counts"], {"train": 46, "validation": 16, "holdout": 16}
        )
        self.assertEqual(
            design["representation_dimensions"],
            {PRIVILEGED_FULL_STATE: 40, RUNTIME_IMU6: 6},
        )
        self.assertEqual(
            tuple(self.document["offset_analysis"]["offsets_before_fall_ms"]),
            (500, 300, 200, 100, 50),
        )
        self.assertEqual(
            tuple(self.document["history"]["candidates_ms"]), (50, 100, 200)
        )
        self.assertFalse(design["fall_label_channels_in_tensor"])
        self.assertFalse(design["terrain_channels_in_tensor"])

    def test_feature_schemas_exclude_labels_diagnostics_and_terrain(self) -> None:
        self.assertEqual(len(TEMPORAL_FULL_STATE_FEATURE_NAMES), 40)
        self.assertEqual(len(IMU6_FEATURE_NAMES), 6)
        forbidden = (
            "terrain",
            "fall_time",
            "time_to_fall",
            "slip",
            "sink",
            "deformation",
            "run_id",
        )
        for names in REPRESENTATION_FEATURE_NAMES.values():
            self.assertFalse(
                any(token in feature for feature in names for token in forbidden)
            )

    def test_causal_window_ends_at_endpoint_and_ignores_future_suffix(self) -> None:
        values = np.arange(1000 * 6, dtype=np.float32).reshape(1000, 6)
        indices = causal_window_indices(499, 200)
        self.assertEqual(indices[0], 300)
        self.assertEqual(indices[-1], 499)
        original = values[indices].copy()
        changed = values.copy()
        changed[500:] = -999.0
        np.testing.assert_array_equal(original, changed[indices])

    def test_causal_last_valid_fill_does_not_backfill_from_future(self) -> None:
        values = np.asarray((np.nan, np.nan, 2.0, np.nan, 4.0))
        filled = causal_last_valid_fill(values)
        np.testing.assert_array_equal(filled, (0.0, 0.0, 2.0, 2.0, 4.0))
        changed = values.copy()
        changed[4] = 99.0
        np.testing.assert_array_equal(causal_last_valid_fill(changed)[:4], filled[:4])

    def test_privileged_features_reuse_order_and_add_phase_context(self) -> None:
        samples = 3
        result = SimpleNamespace(
            runtime=SimpleNamespace(sequence=np.arange(samples)),
            state_trace=SimpleNamespace(
                robot_qpos=np.tile(np.arange(36, dtype=np.float64), (samples, 1)),
                robot_qvel=np.tile(
                    np.arange(35, dtype=np.float64) + 100.0, (samples, 1)
                ),
            ),
            diagnostics=SimpleNamespace(
                pelvis_roll_rad=np.zeros(samples),
                pelvis_pitch_rad=np.zeros(samples),
                pelvis_angular_velocity_rad_s=np.zeros((samples, 3)),
                pelvis_linear_velocity_m_s=np.zeros((samples, 3)),
                pelvis_world_z_m=np.ones(samples),
            ),
            stability=SimpleNamespace(
                gait_phase=np.asarray((NO_SUPPORT, DOUBLE_SUPPORT, NO_SUPPORT)),
                com_velocity_xyz_m_s=np.zeros((samples, 3)),
                support_height_m=np.asarray((np.nan, 1.2, np.nan)),
            ),
        )
        features = privileged_temporal_features(
            result, np.arange(7, 19), np.arange(6, 18)
        )
        self.assertEqual(features.shape, (samples, 40))
        np.testing.assert_allclose(features[:, 36], (0.0, 1.2, 1.2))
        np.testing.assert_array_equal(
            features[:, -3:], ((0, 0, 0), (0, 0, 1), (0, 0, 0))
        )

    def test_binary_auroc_auprc_and_threshold_metrics_are_exact(self) -> None:
        targets = np.asarray((0, 0, 1, 1))
        scores = np.asarray((0.1, 0.4, 0.35, 0.8))
        self.assertAlmostEqual(binary_auroc(targets, scores), 0.75)
        self.assertAlmostEqual(binary_auprc(targets, scores), 5.0 / 6.0)
        metrics = binary_metrics(targets, scores, threshold=0.5)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertAlmostEqual(metrics["stable_specificity"], 1.0)
        self.assertAlmostEqual(metrics["fall_recall"], 0.5)

    def test_horizon_label_uses_outcome_only_as_target(self) -> None:
        self.assertEqual(horizon_fixed_label(None, 700, 200), 0)
        self.assertEqual(horizon_fixed_label(1000, 800, 200), 1)
        self.assertEqual(horizon_fixed_label(1000, 799, 200), 0)
        self.assertEqual(horizon_fixed_label(1000, 1000, 200), 0)

    def test_matching_preserves_exact_elapsed_phase_and_per_run_cap(self) -> None:
        runs = {
            "stable": _fake_run("stable", VALID_STABLE),
            "fall": _fake_run("fall", VALID_FALL, fall=700),
        }
        pairs, exclusions = build_matched_pairs(runs, tuple(runs), (200,), 50)
        self.assertFalse(exclusions)
        self.assertEqual(len(pairs[200]), 1)
        pair = pairs[200][0]
        self.assertEqual(pair.fall_endpoint_sample, 500)
        self.assertEqual(pair.stable_endpoint_sample, 500)
        self.assertEqual(pair.elapsed_since_contact_samples, 400)
        self.assertTrue(pair.endpoint_phase_matched)
        normalizer = Normalizer(
            mean=np.zeros(6, dtype=np.float32),
            std=np.ones(6, dtype=np.float32),
            sample_count=1,
            fit_run_ids=("stable", "fall"),
            epsilon=1.0e-8,
        )
        batch = materialize_matched_windows(
            runs, pairs[200], RUNTIME_IMU6, 50, normalizer
        )
        self.assertEqual(len(batch.windows), 2)
        self.assertEqual(len(set(batch.windows.run_ids)), 2)
        self.assertEqual(batch.rows[0]["elapsed_since_contact_ms"], 400)
        self.assertEqual(batch.rows[1]["elapsed_since_contact_ms"], 400)

    def test_normalizer_uses_train_runs_only_with_deterministic_cap(self) -> None:
        runs = {
            "train": _fake_run("train", VALID_STABLE, value=2.0),
            "validation": _fake_run(
                "validation", VALID_STABLE, split="validation", value=1000.0
            ),
        }
        first = fit_train_normalizer(runs, ("train",), RUNTIME_IMU6, 128, 1.0e-8)
        second = fit_train_normalizer(runs, ("train",), RUNTIME_IMU6, 128, 1.0e-8)
        self.assertEqual(first.fit_run_ids, ("train",))
        self.assertEqual(first.sample_count, 128)
        np.testing.assert_array_equal(first.mean, np.full(6, 2.0))
        np.testing.assert_array_equal(first.mean, second.mean)

    def test_selection_uses_farthest_reliable_then_shortest_history(self) -> None:
        rows = []
        for history in (50, 100, 200):
            for offset in (500, 300, 200, 100, 50):
                passed = offset <= 200 and history >= 100
                rows.append(
                    {
                        "history_ms": history,
                        "offset_ms": offset,
                        "metrics": {
                            "auroc": 0.95 if passed else 0.5,
                            "auprc": 0.95 if passed else 0.5,
                            "balanced_accuracy": 0.9 if passed else 0.5,
                        },
                        "validation_gates": {"all": passed},
                    }
                )
        selection = select_history_and_horizon(rows)
        self.assertEqual(selection["selected"], {"history_ms": 100, "offset_ms": 200})

    def test_holdout_guard_opens_exactly_once_and_is_required(self) -> None:
        runs = {
            "stable": _fake_run("stable", VALID_STABLE, split="holdout"),
            "fall": _fake_run("fall", VALID_FALL, split="holdout", fall=700),
        }
        guard = HoldoutGuard()
        with self.assertRaises(RuntimeError):
            build_matched_pairs(runs, tuple(runs), (200,), 50, holdout_guard=guard)
        guard.open_once()
        pairs, _ = build_matched_pairs(
            runs, tuple(runs), (200,), 50, holdout_guard=guard
        )
        self.assertEqual(len(pairs[200]), 1)
        self.assertEqual(guard.open_count, 1)
        with self.assertRaises(RuntimeError):
            guard.open_once()

    def test_terrain_protection_fusion_and_canonical_cli_contract(self) -> None:
        for relative in self.document["terrain_regression"]["protected_paths"]:
            self.assertTrue((ROOT / relative).is_file())
        self.assertTrue(fusion_regression()["passed"])
        cli = (ROOT / "scripts" / "fastreflex.py").read_text(encoding="utf-8")
        self.assertIn('experiment_id == "TEMPORAL_STABILITY_SEPARABILITY_AUDIT"', cli)
        self.assertIn("run_temporal_stability_separability_audit", cli)


if __name__ == "__main__":
    unittest.main()
