"""Contracts for the dense causal fall-risk dataset and detector PoC."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from fastreflex.dataset.loader import Normalizer, WindowSet
from fastreflex.evaluation.stability_dense import (
    DENSE_FEATURE_NAMES,
    DENSE_REPRESENTATIONS,
    DenseHoldoutGuard,
    PELVIS_IMU6,
    ReplayTrace,
    _load_run,
    _predict_replay,
    _run_to_npz,
    build_dense_windows,
    classify_detection,
    dense_early_negative_endpoints,
    dense_horizon_label,
    dense_positive_endpoints,
    evaluate_run_level,
    generate_dense_specifications,
    physical_signature,
    select_dense_candidate,
    select_validation_threshold,
    sustained_confirmation_sample,
    threshold_grid,
    validate_dense_design,
)
from fastreflex.evaluation.stability_temporal import (
    PRIVILEGED_FULL_STATE,
    RUNTIME_IMU6,
    TemporalRun,
    _protected_hashes,
)
from fastreflex.evaluation.transition_scenarios import (
    VALID_FALL,
    VALID_STABLE,
    fusion_regression,
)
from fastreflex.models.baselines import build_model
from fastreflex.training.trainer import train_model


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT / "configs/experiment/20260828_dense_fall_risk_detector_poc.yaml"
)


def _document() -> dict[str, object]:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _run(
    run_id: str,
    outcome: str,
    *,
    source: str = "concrete",
    target: str = "ice",
    fall: int | None = None,
    split: str = "train",
    control: bool = False,
) -> TemporalRun:
    length = 1000
    imu = np.arange(length * 6, dtype=np.float32).reshape(length, 6) / 1000.0
    privileged = np.arange(length * 40, dtype=np.float32).reshape(length, 40) / 1000.0
    return TemporalRun(
        run_id=run_id,
        split=split,
        source_terrain=source,
        target_terrain=target,
        speed_mps=0.25,
        outcome=outcome,
        first_contact_sample=0 if control else 100,
        fall_sample=fall,
        gait_phase=np.zeros(length, dtype=np.int8),
        features={PRIVILEGED_FULL_STATE: privileged, RUNTIME_IMU6: imu},
        timestamp_us=np.arange(length, dtype=np.int64) * 1000,
        slip_sample=300 if target == "ice" else None,
        sink_sample=300 if target == "sand" else None,
        maximum_support_deformation_m=0.02 if target == "sand" else 0.0,
        hard_stable_control=control,
    )


def _normalizer(channels: int) -> Normalizer:
    return Normalizer(
        mean=np.zeros(channels, dtype=np.float32),
        std=np.ones(channels, dtype=np.float32),
        sample_count=1,
        fit_run_ids=("fall", "stable"),
        epsilon=1.0e-8,
    )


class DenseStabilityTest(unittest.TestCase):
    def test_matrix_speed_split_signatures_and_prior_freshness_are_frozen(self) -> None:
        document = _document()
        primary, controls = generate_dense_specifications(document)
        design = validate_dense_design(document, primary, controls, REPOSITORY_ROOT)
        self.assertEqual(len(primary), 240)
        self.assertEqual(len(controls), 16)
        self.assertEqual(
            design["split_counts"], {"train": 144, "validation": 48, "holdout": 48}
        )
        self.assertEqual(design["primary_speed_values_mps"], [0.25])
        self.assertEqual(len({physical_signature(row) for row in primary}), 240)
        self.assertEqual(design["prior_signature_overlap"], 0)

    def test_duplicate_physical_signature_is_rejected_before_simulation(self) -> None:
        document = deepcopy(_document())
        document["dataset"]["deterministic_condition_schedule"]["ice_stable"][
            "concrete_width"
        ]["step"] = 0.0
        primary, controls = generate_dense_specifications(document)
        with self.assertRaisesRegex(ValueError, "signatures are duplicated"):
            validate_dense_design(document, primary, controls, REPOSITORY_ROOT)

    def test_horizon_label_boundary_and_stable_labels(self) -> None:
        self.assertEqual(dense_horizon_label(1000, 800, 200), 1)
        self.assertEqual(dense_horizon_label(1000, 999, 200), 1)
        self.assertEqual(dense_horizon_label(1000, 799, 200), 0)
        self.assertEqual(dense_horizon_label(1000, 1000, 200), 0)
        self.assertEqual(dense_horizon_label(None, 99999, 200), 0)
        with self.assertRaises(ValueError):
            dense_horizon_label(1000, 900, 300)

    def test_positive_and_early_negative_endpoints_are_prefall_and_ten_ms(self) -> None:
        run = _run("fall", VALID_FALL, fall=800)
        positive = dense_positive_endpoints(run, 200, 10)
        early = dense_early_negative_endpoints(run, 200, 50, 10, 20, len(positive))
        self.assertEqual((positive[0], positive[-1], len(positive)), (600, 790, 20))
        np.testing.assert_array_equal(np.diff(positive), np.full(19, 10))
        self.assertTrue(np.all(early < 580))
        self.assertTrue(
            all(
                dense_horizon_label(run.fall_sample, int(value), 200) == 0
                for value in early
            )
        )

    def test_dense_windows_cap_runs_match_elapsed_and_never_include_fall(self) -> None:
        runs = {
            "fall": _run("fall", VALID_FALL, fall=800),
            "stable": _run("stable", VALID_STABLE),
        }
        batch = build_dense_windows(
            runs,
            ("fall", "stable"),
            PELVIS_IMU6,
            200,
            50,
            10,
            20,
            _normalizer(6),
        )
        kinds = [row["kind"] for row in batch.rows]
        self.assertEqual(kinds.count("fall_positive"), 20)
        self.assertEqual(kinds.count("fall_early_negative"), 20)
        self.assertEqual(kinds.count("stable_matched_negative"), 20)
        self.assertEqual(len(batch.windows), 60)
        self.assertTrue(
            np.all(
                batch.windows.endpoint_samples[batch.windows.run_ids == "fall"] < 800
            )
        )
        positive_elapsed = [
            row["elapsed_since_contact_ms"]
            for row in batch.rows
            if row["kind"] == "fall_positive"
        ]
        stable_elapsed = [
            row["elapsed_since_contact_ms"]
            for row in batch.rows
            if row["kind"] == "stable_matched_negative"
        ]
        self.assertEqual(positive_elapsed, stable_elapsed)

    def test_feature_schemas_exclude_terrain_fall_and_diagnostics(self) -> None:
        forbidden = (
            "terrain",
            "fall",
            "time_to_fall",
            "slip",
            "sink",
            "deformation",
            "patch",
        )
        self.assertEqual(DENSE_REPRESENTATIONS, (PRIVILEGED_FULL_STATE, PELVIS_IMU6))
        self.assertEqual(len(DENSE_FEATURE_NAMES[PRIVILEGED_FULL_STATE]), 40)
        self.assertEqual(len(DENSE_FEATURE_NAMES[PELVIS_IMU6]), 6)
        self.assertFalse(
            any(
                token in name
                for names in DENSE_FEATURE_NAMES.values()
                for name in names
                for token in forbidden
            )
        )

    def test_threshold_grid_is_frozen(self) -> None:
        grid = threshold_grid()
        self.assertEqual((len(grid), grid[0], grid[-1]), (41, 0.10, 0.90))
        self.assertTrue(
            all(round(right - left, 2) == 0.02 for left, right in zip(grid, grid[1:]))
        )

    def test_ten_ms_persistence_reports_confirmation_endpoint(self) -> None:
        endpoints = np.arange(100, 120, dtype=np.int64)
        scores = np.asarray([0.0] * 3 + [0.8] * 10 + [0.0] * 7)
        self.assertEqual(sustained_confirmation_sample(endpoints, scores, 0.5, 10), 112)
        scores[7] = 0.0
        self.assertIsNone(sustained_confirmation_sample(endpoints, scores, 0.5, 10))

    def test_detection_classification_distinguishes_premature_valid_and_stable_fp(
        self,
    ) -> None:
        self.assertEqual(classify_detection(1000, 799, 200), "FALL_PREMATURE_FP")
        self.assertEqual(classify_detection(1000, 800, 200), "FALL_VALID_DETECTION")
        self.assertEqual(classify_detection(1000, 999, 200), "FALL_VALID_DETECTION")
        self.assertEqual(classify_detection(1000, None, 200), "FALL_MISSED")
        self.assertEqual(classify_detection(None, 500, 200), "STABLE_FP")
        self.assertEqual(classify_detection(None, None, 200), "STABLE_TN")

    def test_run_level_metrics_count_stable_and_hard_control_false_positives(
        self,
    ) -> None:
        runs = {
            "stable": _run("stable", VALID_STABLE),
            "fall": _run("fall", VALID_FALL, fall=800),
            "control": _run("control", VALID_STABLE, control=True, split="validation"),
        }
        endpoints = np.arange(700, 800, dtype=np.int64)
        traces = {
            run_id: ReplayTrace(endpoints=endpoints, probabilities=np.ones(100))
            for run_id in runs
        }
        metrics = evaluate_run_level(runs, traces, 100, 0.5, 10, ("control",))
        self.assertEqual(metrics["fall_recall"], 1.0)
        self.assertEqual(metrics["stable_fp_rate"], 1.0)
        self.assertEqual(metrics["hard_control_fp_rate"], 1.0)

    def test_threshold_selection_applies_false_alarm_feasibility_then_priority(
        self,
    ) -> None:
        def row(
            threshold: float, recall: float, stable_fp: float, premature: float
        ) -> dict[str, object]:
            return {
                "threshold": threshold,
                "metrics": {
                    "fall_recall": recall,
                    "stable_fp_rate": stable_fp,
                    "hard_control_fp_rate": 0.0,
                    "premature_fall_fp_rate": premature,
                    "ice_fall_recall": recall,
                    "sand_fall_recall": recall,
                    "lead_ms": {"median": 100.0},
                },
            }

        result = select_validation_threshold(
            (
                row(0.40, 1.0, 0.20, 0.0),
                row(0.50, 0.9, 0.10, 0.0),
                row(0.60, 0.9, 0.10, 0.0),
            ),
            {
                "stable_transition_fp_rate_max": 0.15,
                "hard_control_fp_rate_max": 0.10,
                "premature_fall_fp_rate_max": 0.15,
            },
        )
        self.assertEqual(result["selected"]["threshold"], 0.60)

    def test_candidate_selection_prefers_longest_horizon_then_shortest_history(
        self,
    ) -> None:
        def candidate(horizon: int, history: int, passed: bool) -> dict[str, object]:
            return {
                "horizon_ms": horizon,
                "history_ms": history,
                "validation_passed": passed,
                "validation_gates": {"a": passed},
                "operating_point": {
                    "threshold": 0.5,
                    "fall_recall": 1.0,
                    "stable_specificity": 1.0,
                    "lead_ms": {"median": horizon / 2},
                },
            }

        selection = select_dense_candidate(
            (
                candidate(200, 100, True),
                candidate(200, 50, True),
                candidate(100, 50, True),
            )
        )
        self.assertEqual(
            selection["selected"],
            {"horizon_ms": 200, "history_ms": 50, "threshold": 0.5},
        )

    def test_holdout_guard_requires_selection_and_opens_once(self) -> None:
        guard = DenseHoldoutGuard()
        with self.assertRaises(RuntimeError):
            guard.require_open()
        guard.open_once()
        guard.require_open()
        self.assertEqual(guard.open_count, 1)
        with self.assertRaises(RuntimeError):
            guard.open_once()

    def test_npz_round_trip_preserves_reproducible_model_inputs_and_clocks(
        self,
    ) -> None:
        run = _run("fall", VALID_FALL, fall=800)
        row = {
            "run_id": "fall",
            "split": "train",
            "source_terrain": "concrete",
            "target_terrain": "ice",
            "speed_mps": 0.25,
            "observed_outcome": VALID_FALL,
            "slip_sample": 300,
            "sink_sample": None,
            "maximum_support_deformation_m": 0.0,
            "hard_stable_control": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fall.npz"
            _run_to_npz(path, run)
            restored = _load_run(path, row)
        np.testing.assert_array_equal(
            restored.features[RUNTIME_IMU6], run.features[RUNTIME_IMU6]
        )
        np.testing.assert_array_equal(
            restored.features[PRIVILEGED_FULL_STATE],
            run.features[PRIVILEGED_FULL_STATE],
        )
        self.assertEqual(
            (restored.first_contact_sample, restored.fall_sample), (100, 800)
        )

    def test_replay_is_one_ms_causal_and_supports_inference_only_order_diagnostics(
        self,
    ) -> None:
        run = _run("fall", VALID_FALL, fall=180)
        model = build_model("gru", 50, 6, class_count=2)
        original = _predict_replay(
            run, PELVIS_IMU6, 50, _normalizer(6), (model,), "original"
        )
        reversed_trace = _predict_replay(
            run, PELVIS_IMU6, 50, _normalizer(6), (model,), "reversed"
        )
        endpoint_only = _predict_replay(
            run, PELVIS_IMU6, 50, _normalizer(6), (model,), "endpoint_only"
        )
        np.testing.assert_array_equal(
            np.diff(original.endpoints), np.ones(len(original.endpoints) - 1)
        )
        self.assertLess(original.endpoints[-1], run.fall_sample)
        np.testing.assert_array_equal(original.endpoints, reversed_trace.endpoints)
        np.testing.assert_array_equal(original.endpoints, endpoint_only.endpoints)
        self.assertTrue(np.all(np.isfinite(reversed_trace.probabilities)))

    def test_validation_loss_epoch_selection_is_opt_in_and_recorded(self) -> None:
        inputs = np.asarray(
            [
                np.full((10, 2), value, dtype=np.float32)
                for value in (-1.0, -0.8, -0.6, -0.4, 0.4, 0.6, 0.8, 1.0)
            ]
        )
        targets = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        windows = WindowSet(
            inputs=inputs,
            targets=targets,
            run_ids=np.asarray([f"r{index}" for index in range(8)]),
            endpoint_samples=np.arange(8, dtype=np.int64),
            available_by_class=(4, 4, 0),
        )
        _, result = train_model(
            "gru",
            10,
            windows,
            windows,
            20260828,
            batch_size=4,
            max_epochs=2,
            patience=2,
            class_names=("STABLE", "FALL_RISK"),
            selection_metric="validation_loss",
        )
        self.assertTrue(
            all("validation_cross_entropy" in row for row in result.history)
        )

    def test_terrain_fusion_and_canonical_cli_are_unchanged_except_dispatch(
        self,
    ) -> None:
        document = _document()
        for relative in document["terrain_regression"]["protected_paths"]:
            self.assertTrue((REPOSITORY_ROOT / relative).is_file())
        protected = document["terrain_regression"]["protected_paths"]
        self.assertEqual(
            _protected_hashes(REPOSITORY_ROOT, protected),
            _protected_hashes(REPOSITORY_ROOT, protected),
        )
        self.assertTrue(fusion_regression()["passed"])
        cli = (REPOSITORY_ROOT / "scripts/fastreflex.py").read_text(encoding="utf-8")
        self.assertIn("DENSE_FALL_RISK_DATASET_AND_DETECTOR_POC", cli)
        self.assertNotIn("q_dq_runtime", cli)


if __name__ == "__main__":
    unittest.main()
