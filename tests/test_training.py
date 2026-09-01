"""Tests for generic training and current Hazard TRAIN-only contracts."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import torch

from fastreflex.dataset.hazard import (
    EVENT_TYPE_NONE,
    PELVIS_IMU6,
    PELVIS_IMU6_FSR8,
    HazardRun,
)
from fastreflex.dataset.loader import WindowSet
from fastreflex.dataset.generation import (
    HazardRunAnnotations,
    load_model_v2_runs,
)
from fastreflex.models.baselines import build_model, parameter_count
from fastreflex.training.hazard import (
    HNM_MINIMUM_SPACING_MS,
    HNM_REPLAY_STRIDE_MS,
    HNM_ROUNDS,
    HNM_TOP_K_PER_RUN,
    audit_hazard_extraction,
    fit_hazard_normalizer,
    initial_negative_endpoints,
    training_negative_candidates,
)
from fastreflex.training.trainer import (
    load_checkpoint,
    save_checkpoint,
    train_model,
)


def _run(split: str) -> HazardRun:
    samples = 60
    rng = np.random.default_rng(4)
    imu = rng.normal(size=(samples, 6)).astype(np.float32)
    fsr = np.zeros((samples, 8), dtype=np.float32)
    zeros = np.zeros((samples, 2), dtype=np.float32)
    return HazardRun(
        run_id=split,
        split=split,
        source_terrain="concrete",
        target_terrain="concrete",
        design_role="normal",
        first_contact_sample=0,
        first_touchdown_sample=0,
        censor_sample=samples,
        outcome_diagnostic="VALID_STABLE",
        fall_sample_diagnostic=None,
        features={
            PELVIS_IMU6: imu,
            PELVIS_IMU6_FSR8: np.concatenate((imu, fsr), axis=1),
        },
        timestamp_us=np.arange(samples, dtype=np.int64) * 1000,
        slip_event_samples_per_foot=(None, None),
        support_event_samples_per_foot=(None, None),
        event_sample=None,
        event_type=EVENT_TYPE_NONE,
        hard_stable_control=True,
        drift_m=zeros,
        tangential_velocity_mps=zeros,
        support_spread_m=zeros,
        support_max_displacement_m=zeros,
        loaded_contact=np.zeros((samples, 2), dtype=bool),
        sink_pattern="uniform",
        support_pattern="balanced_soft",
    )


def _windows(seed: int) -> WindowSet:
    rng = np.random.default_rng(seed)
    inputs = rng.normal(size=(20, 5, 3)).astype(np.float32)
    targets = np.asarray([0, 1] * 10, dtype=np.int64)
    return WindowSet(
        inputs=inputs,
        targets=targets,
        run_ids=np.asarray([f"run_{index // 2}" for index in range(20)]),
        endpoint_samples=np.arange(20, dtype=np.int64),
        available_by_class=(10, 10, 0),
    )


def _annotations() -> HazardRunAnnotations:
    samples = 60
    target = np.zeros((samples, 2), dtype=bool)
    target[20:50, 0] = True
    candidate = np.zeros((samples, 2), dtype=bool)
    codes = np.zeros((samples, 2), dtype=np.int8)
    censored = np.zeros((samples, 2), dtype=bool)
    i1 = np.zeros(samples, dtype=bool)
    candidate[25, 0] = True
    codes[25, 0] = 1
    candidate[30, 0] = True
    codes[30, 0] = 5
    censored[30, 0] = True
    candidate[40, 0] = True
    codes[40, 0] = 4
    i1[35] = True
    return HazardRunAnnotations(
        dataset_id="model_v2_hazard_reflex_20260901",
        scenario_family="ICE_BENIGN_CONTROL",
        nominal_speed_mps=0.2,
        actual_side="NONE",
        target_contact=target,
        established_slip_active=np.zeros((samples, 2), dtype=bool),
        i1_active=i1,
        ice_precursor_candidate=candidate,
        ice_precursor_future_outcome_code=codes,
        ice_precursor_censored=censored,
    )


class TrainingTest(unittest.TestCase):
    def test_supported_gru_architecture_is_80_32_one_layer_two_outputs(self) -> None:
        model = build_model("gru", 20, 80, class_count=2)
        self.assertEqual(model.gru.input_size, 80)
        self.assertEqual(model.gru.hidden_size, 32)
        self.assertEqual(model.gru.num_layers, 1)
        self.assertFalse(model.gru.bidirectional)
        self.assertEqual(model.classifier.out_features, 2)
        self.assertEqual(parameter_count(model), 11_010)
        self.assertEqual(tuple(model(torch.zeros(3, 20, 80)).shape), (3, 2))

    def test_normalizer_is_train_only_and_records_provenance(self) -> None:
        train = _run("train")
        normalizer = fit_hazard_normalizer({"train": train}, ("train",))
        self.assertEqual(normalizer.fit_run_ids, ("train",))
        self.assertEqual(normalizer.mean.shape, (80,))
        self.assertEqual(normalizer.std.shape, (80,))
        with self.assertRaises(ValueError):
            fit_hazard_normalizer({"validation": _run("validation")}, ("validation",))

    def test_hnm_constants_are_the_frozen_train_only_protocol(self) -> None:
        self.assertEqual(HNM_ROUNDS, 3)
        self.assertEqual(HNM_REPLAY_STRIDE_MS, 1)
        self.assertEqual(HNM_TOP_K_PER_RUN, 12)
        self.assertEqual(HNM_MINIMUM_SPACING_MS, 30)

    def test_v2_negative_precedence_masks_future_censored_and_i1(self) -> None:
        run = _run("train")
        annotation = _annotations()
        eligible = set(training_negative_candidates(run, None, annotation))
        self.assertNotIn(25, eligible)
        self.assertNotIn(30, eligible)
        self.assertNotIn(35, eligible)
        self.assertIn(40, eligible)
        selected = set(initial_negative_endpoints(run, None, annotation=annotation))
        self.assertIn(40, selected)
        self.assertFalse(selected & {25, 30, 35})
        audit = audit_hazard_extraction(
            {"train": run}, ("train",), {"train": None}, {"train": annotation}
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(
            set(audit["ordinary_negative_violations"].values()), {0}
        )

    def test_v2_validation_loader_fails_closed_until_candidate_freeze(self) -> None:
        manifest = {"runs": []}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(RuntimeError):
                load_model_v2_runs(root, manifest, "V2_VALIDATION")
            freeze = root / "candidate_freeze.json"
            freeze.write_text(
                json.dumps(
                    {
                        "candidate_frozen_before_validation": True,
                        "v2_validation_evaluated": False,
                        "generalization_holdout_guard_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            runs, annotations = load_model_v2_runs(
                root,
                manifest,
                "V2_VALIDATION",
                candidate_freeze_path=freeze,
            )
        self.assertEqual(runs, {})
        self.assertEqual(annotations, {})

    def test_generic_training_is_deterministic(self) -> None:
        train = _windows(1)
        validation = _windows(2)
        first, first_result = train_model(
            "mlp",
            5,
            train,
            validation,
            seed=9,
            batch_size=8,
            max_epochs=2,
            patience=2,
            class_names=("NORMAL", "HAZARD"),
        )
        second, second_result = train_model(
            "mlp",
            5,
            train,
            validation,
            seed=9,
            batch_size=8,
            max_epochs=2,
            patience=2,
            class_names=("NORMAL", "HAZARD"),
        )
        self.assertEqual(first_result.history, second_result.history)
        for left, right in zip(first.parameters(), second.parameters()):
            self.assertTrue(torch.equal(left, right))

    def test_checkpoint_round_trip_preserves_identity_and_output(self) -> None:
        train = _windows(3)
        model, result = train_model(
            "gru",
            5,
            train,
            _windows(4),
            seed=11,
            max_epochs=1,
            patience=1,
            class_names=("NORMAL", "HAZARD"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.pt"
            save_checkpoint(
                path,
                model,
                "gru",
                5,
                11,
                result,
                input_channels=3,
                class_names=("NORMAL", "HAZARD"),
            )
            loaded, metadata = load_checkpoint(path)
        self.assertEqual(metadata["family"], "gru")
        self.assertEqual(metadata["input_channels"], 3)
        self.assertEqual(metadata["class_names"], ["NORMAL", "HAZARD"])
        inputs = torch.from_numpy(train.inputs)
        self.assertTrue(torch.equal(model(inputs), loaded(inputs)))


if __name__ == "__main__":
    unittest.main()
