from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from fastreflex.dataset.loader import (
    ManifestRecord,
    WindowSet,
    build_windows,
    fit_normalizer,
    validate_split,
)
from fastreflex.evaluation.metrics import classification_metrics, confusion_matrix
from fastreflex.models.baselines import build_model, parameter_count
from fastreflex.training.trainer import (
    load_checkpoint,
    predict_model,
    save_checkpoint,
    train_model,
)


def write_run(
    path: Path,
    labels: np.ndarray,
    eligible: np.ndarray,
    offset: float = 0.0,
) -> None:
    samples = len(labels)
    time = np.arange(samples, dtype=np.float32)[:, None]
    channels = np.arange(6, dtype=np.float32)[None, :]
    imu = time + channels + offset
    np.savez(
        path,
        pelvis_imu=imu,
        hazard_class_id=labels.astype(np.int8),
        training_eligible=eligible.astype(bool),
    )


def record(path: Path, run_id: str, outcome: str = "BENIGN") -> ManifestRecord:
    return ManifestRecord(
        run_id=run_id,
        path=path,
        observed_outcome=outcome,
        scenario_family="normal",
        terrain="concrete",
        speed_mps=0.15,
        patch_start_x=None,
        sink_side=None,
    )


def window_set(inputs: np.ndarray, targets: np.ndarray) -> WindowSet:
    run_ids = np.asarray([f"run_{index // 3}" for index in range(len(targets))])
    counts = tuple(int(value) for value in np.bincount(targets, minlength=3))
    return WindowSet(
        inputs=inputs.astype(np.float32),
        targets=targets.astype(np.int64),
        run_ids=run_ids,
        endpoint_samples=np.arange(len(targets), dtype=np.int64),
        available_by_class=counts,
    )


class TrainingTest(unittest.TestCase):
    def test_split_is_disjoint_and_invalid_is_excluded(self) -> None:
        records = {
            "normal": record(Path("normal.npz"), "normal", "BENIGN"),
            "slip": record(Path("slip.npz"), "slip", "SLIP"),
            "sink": record(Path("sink.npz"), "sink", "SINK"),
            "invalid": record(Path("invalid.npz"), "invalid", "INVALID"),
        }
        counts = validate_split(
            records,
            {"train": ["normal"], "validation": ["slip"], "holdout": ["sink"]},
        )
        self.assertEqual(counts["train"]["BENIGN"], 1)
        with self.assertRaisesRegex(ValueError, "excluded outcome"):
            validate_split(
                records,
                {
                    "train": ["normal", "invalid"],
                    "validation": ["slip"],
                    "holdout": ["sink"],
                },
            )
        with self.assertRaisesRegex(ValueError, "run-disjoint"):
            validate_split(
                records,
                {
                    "train": ["normal"],
                    "validation": ["normal", "slip"],
                    "holdout": ["sink"],
                },
            )

    def test_windows_are_causal_same_class_and_have_expected_shape(self) -> None:
        labels = np.asarray([0] * 120 + [-1] * 10 + [1] * 120)
        eligible = labels >= 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.npz"
            write_run(path, labels, eligible)
            records = {"run": record(path, "run", "SLIP")}
            windows_50 = build_windows(records, ["run"], 50, 10, normalizer=None)
            windows_100 = build_windows(records, ["run"], 100, 10, normalizer=None)
        self.assertEqual(windows_50.inputs.shape, (16, 50, 6))
        self.assertEqual(windows_50.selected_by_class, (8, 8, 0))
        self.assertEqual(windows_100.inputs.shape, (6, 100, 6))
        self.assertEqual(windows_100.selected_by_class, (3, 3, 0))
        self.assertEqual(windows_100.endpoint_samples.tolist(), [99, 109, 119, 229, 239, 249])
        self.assertTrue(np.all(windows_100.targets[:3] == 0))
        self.assertTrue(np.all(windows_100.targets[3:] == 1))

    def test_normalizer_uses_only_declared_training_run(self) -> None:
        labels = np.zeros(60, dtype=np.int8)
        eligible = np.ones(60, dtype=bool)
        with tempfile.TemporaryDirectory() as directory:
            train_path = Path(directory) / "train.npz"
            holdout_path = Path(directory) / "holdout.npz"
            write_run(train_path, labels, eligible, offset=0.0)
            write_run(holdout_path, labels, eligible, offset=10_000.0)
            records = {
                "train": record(train_path, "train"),
                "holdout": record(holdout_path, "holdout"),
            }
            normalizer = fit_normalizer(records, ["train"])
        self.assertEqual(normalizer.fit_run_ids, ("train",))
        self.assertLess(float(normalizer.mean.max()), 100.0)
        self.assertEqual(normalizer.sample_count, 60)

    def test_model_forward_shapes_and_parameter_counts(self) -> None:
        inputs = torch.zeros(4, 50, 6)
        mlp = build_model("mlp", 50)
        gru = build_model("gru", 50)
        self.assertEqual(tuple(mlp(inputs).shape), (4, 3))
        self.assertEqual(tuple(gru(inputs).shape), (4, 3))
        self.assertEqual(parameter_count(mlp), 21_443)
        self.assertEqual(parameter_count(gru), 3_939)

    def test_training_smoke_is_deterministic(self) -> None:
        rng = np.random.default_rng(7)
        targets = np.tile(np.arange(3), 12)
        inputs = rng.normal(size=(36, 5, 6)).astype(np.float32)
        inputs += targets[:, None, None] * 2.0
        windows = window_set(inputs, targets)
        first_model, first = train_model(
            "mlp", 5, windows, windows, 123, batch_size=12, max_epochs=3, patience=2
        )
        second_model, second = train_model(
            "mlp", 5, windows, windows, 123, batch_size=12, max_epochs=3, patience=2
        )
        self.assertEqual(first.best_epoch, second.best_epoch)
        np.testing.assert_array_equal(
            predict_model(first_model, windows), predict_model(second_model, windows)
        )

    def test_confusion_matrix_and_metrics(self) -> None:
        targets = np.asarray([0, 0, 1, 1, 2, 2])
        predictions = np.asarray([0, 1, 1, 1, 0, 2])
        expected = np.asarray([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        np.testing.assert_array_equal(confusion_matrix(targets, predictions), expected)
        metrics = classification_metrics(
            targets, predictions, ["a", "a", "b", "b", "c", "c"]
        )
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertIn("run_balanced_accuracy", metrics)

    def test_checkpoint_round_trip(self) -> None:
        rng = np.random.default_rng(11)
        targets = np.tile(np.arange(3), 4)
        windows = window_set(rng.normal(size=(12, 5, 6)), targets)
        model, result = train_model(
            "gru", 5, windows, windows, 17, batch_size=6, max_epochs=2, patience=1
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, "gru", 5, 17, result)
            loaded, metadata = load_checkpoint(path)
        self.assertEqual(metadata["seed"], 17)
        np.testing.assert_array_equal(
            predict_model(model, windows), predict_model(loaded, windows)
        )


if __name__ == "__main__":
    unittest.main()
