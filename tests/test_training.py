"""Tests for generic training and current Hazard TRAIN-only contracts."""

from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from fastreflex.dataset.hazard import (
    EVENT_TYPE_NONE,
    PELVIS_IMU6,
    PELVIS_IMU6_FSR8,
    HazardRun,
    canonical_sha256,
)
from fastreflex.dataset.loader import WindowSet
from fastreflex.dataset.loader import sha256_file
from fastreflex.evaluation.hazard import (
    verify_model_v2_anchor_refined_training_result,
    verify_model_v2_extraction_rebalanced_training_result,
)
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
    audit_model_v2_anchor_refined_extraction,
    audit_model_v2_rebalanced_extraction,
    model_v2_anchor_refinement_candidates,
    model_v2_anchor_refined_policy,
    audit_hazard_extraction,
    fit_hazard_normalizer,
    initial_negative_endpoints,
    model_v2_rebalance_policy,
    prepare_model_v2_training_data,
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
    def test_model_v2_anchor_refined_extraction_matches_frozen_design(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config_path = (
            root
            / "configs/experiment/20260902_model_v2_anchor_refined_training.yaml"
        )
        dataset_path = root / "data/raw/model_v2_hazard_reflex_20260901"
        if not dataset_path.is_dir():
            self.skipTest("frozen Model V2 dataset is not available")
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data = prepare_model_v2_training_data(root, document)
        audit = audit_model_v2_anchor_refined_extraction(
            data.runs,
            data.precursor_samples,
            data.annotations,
        )

        self.assertTrue(audit["passed"])
        self.assertEqual(
            canonical_sha256(model_v2_anchor_refinement_candidates()),
            "cd6af9180613101f61fe271f2055d1c8a69daf7acf1d73ef429c8e0f47c555cb",
        )
        self.assertEqual(
            canonical_sha256(model_v2_anchor_refined_policy()),
            "52004bc2ddc307316a7a888855a1bd8014e50b96aa45a178a10965e890f4b199",
        )
        self.assertEqual(
            audit["positive_window_ids_sha256"],
            "248719864bc1974ac54a21de63f04a6d5e6f55ef3e3c37092cf0ec757872d09e",
        )
        self.assertEqual(
            audit["negative_window_ids_sha256"],
            "392c1fda06953135bfed9a97c7cabb3915d4c50fdf9ff11b2ac8e6550448936c",
        )
        self.assertEqual(
            audit["masked_window_sha256"],
            "32bae2d81b05709771545b375dd6cffb95d2dfc17c2808a7adf3a7f70174c35a",
        )
        self.assertEqual(
            audit["monitor_endpoint_sha256"],
            "39d30234f674446f305b1b51d446977ba301e6db1e0591ac14dbe7172cbb1bf5",
        )
        self.assertEqual(
            audit["monitor_positive_sha256"],
            "e4cd285091e55c92c773512b44958273d7773a708bad876806cec6a8401f9c88",
        )
        self.assertEqual(audit["all_positive_count"], 3_135)
        self.assertEqual(audit["all_negative_count"], 32_209)
        self.assertEqual(
            audit["fit_positive_counts"],
            {
                "slip": 1_680,
                "ordinary_support": 640,
                "delayed_support": 198,
                "support": 838,
                "total": 2_518,
            },
        )
        self.assertEqual(audit["fit_negative_count"], 25_585)
        self.assertEqual(
            audit["monitor_positive_counts"],
            {
                "slip": 431,
                "ordinary_support": 167,
                "delayed_support_concrete": 8,
                "delayed_support_marble": 11,
                "total": 617,
            },
        )
        self.assertEqual(audit["monitor_negative_count"], 6_624)
        self.assertEqual(audit["fit_monitor_endpoint_overlap"], 0)
        self.assertEqual(audit["delayed_support"]["fit_represented_runs"], 18)
        self.assertEqual(
            audit["delayed_support"]["by_source"],
            {
                "concrete": {"eligible_runs": 9, "fit_positive_windows": 99},
                "marble": {"eligible_runs": 9, "fit_positive_windows": 99},
            },
        )
        self.assertEqual(set(audit["contradiction_audit"].values()), {0})

    def test_model_v2_rebalanced_extraction_matches_frozen_design(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config_path = (
            root
            / "configs/experiment/20260901_model_v2_extraction_rebalanced_training.yaml"
        )
        dataset_path = root / "data/raw/model_v2_hazard_reflex_20260901"
        if not dataset_path.is_dir():
            self.skipTest("frozen Model V2 dataset is not available")
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        data = prepare_model_v2_training_data(root, document)
        audit = audit_model_v2_rebalanced_extraction(
            data.runs,
            data.precursor_samples,
            data.annotations,
        )

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["effective_train_run_count"], 442)
        self.assertEqual(
            canonical_sha256(model_v2_rebalance_policy()),
            "3c7ce82ed905d932ec8f17d69d7e5edb5d79ee7602ba95ffe2a53d2407142cd2",
        )
        self.assertEqual(
            audit["positive_window_ids_sha256"],
            "498f5d1f4419e3bfa72fc2f9649326db26f00e7a9523d9b3ecc8032436a3e0bb",
        )
        self.assertEqual(
            audit["negative_window_ids_sha256"],
            "392c1fda06953135bfed9a97c7cabb3915d4c50fdf9ff11b2ac8e6550448936c",
        )
        self.assertEqual(
            audit["masked_window_sha256"],
            "32bae2d81b05709771545b375dd6cffb95d2dfc17c2808a7adf3a7f70174c35a",
        )
        self.assertEqual(audit["all_positive_count"], 3_188)
        self.assertEqual(audit["all_negative_count"], 32_209)
        self.assertEqual(
            audit["fit_positive_counts"],
            {
                "slip": 1_680,
                "ordinary_support": 640,
                "delayed_support": 270,
                "support": 910,
                "total": 2_590,
            },
        )
        self.assertEqual(audit["fit_negative_count"], 25_585)
        self.assertEqual(audit["monitor_positive_count"], 598)
        self.assertEqual(audit["monitor_negative_count"], 6_624)
        self.assertEqual(
            audit["delayed_support"],
            {
                "eligible_runs": 18,
                "fit_represented_runs": 18,
                "by_source": {
                    "concrete": {
                        "eligible_runs": 9,
                        "fit_positive_windows": 135,
                    },
                    "marble": {
                        "eligible_runs": 9,
                        "fit_positive_windows": 135,
                    },
                },
            },
        )
        self.assertEqual(
            audit["masked_sample_counts"],
            {
                "future_slip_precursor": 41_479,
                "censored_precursor": 1_734,
                "i1_positive": 68_388,
            },
        )
        self.assertEqual(set(audit["contradiction_audit"].values()), {0})

    def test_rebalanced_candidate_reuses_normalizer_and_isolates_artifacts(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        config_path = (
            root
            / "configs/experiment/20260901_model_v2_extraction_rebalanced_training.yaml"
        )
        artifact_path = (
            root
            / "artifacts/runs/20260901_model_v2_extraction_rebalanced_training"
        )
        if not (artifact_path / "training_result.json").is_file():
            self.skipTest("frozen extraction-rebalanced candidate is not available")
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        candidate = json.loads(
            (artifact_path / "candidate_freeze.json").read_text(encoding="utf-8")
        )
        result = json.loads(
            (artifact_path / "training_result.json").read_text(encoding="utf-8")
        )
        verified = verify_model_v2_extraction_rebalanced_training_result(
            root, config_path
        )

        self.assertTrue(verified["passed"])
        self.assertEqual(candidate["normalizer_fits"], 0)
        self.assertEqual(
            candidate["normalizer_sha256"],
            "e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a",
        )
        self.assertEqual(
            sha256_file(root / candidate["normalizer_path"]),
            candidate["normalizer_sha256"],
        )
        protected_paths = {
            str(row["path"])
            for row in (
                *document["protected_v1"]["checkpoints"],
                *document["baseline_v2"]["checkpoints"],
            )
        }
        self.assertFalse(protected_paths & set(candidate["checkpoint_sha256"]))
        for record in (
            *document["protected_v1"]["checkpoints"],
            *document["baseline_v2"]["checkpoints"],
        ):
            self.assertEqual(sha256_file(root / record["path"]), record["sha256"])
        self.assertFalse(result["generalization_validation_v2_inference"])
        self.assertFalse(result["unified_holdout_waveform_reopened"])
        self.assertFalse(result["generalization_holdout_waveform_opened"])
        self.assertFalse(result["generalization_holdout_inference"])
        self.assertEqual(result["generalization_holdout_guard_count"], 0)

    def test_anchor_refined_candidate_reuses_normalizer_and_isolates_artifacts(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        config_path = (
            root
            / "configs/experiment/20260902_model_v2_anchor_refined_training.yaml"
        )
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        artifact_path = root / str(document["artifacts"]["path"])
        if not (artifact_path / "training_result.json").is_file():
            self.skipTest("frozen anchor-refined candidate is not available")
        candidate = json.loads(
            (artifact_path / "candidate_freeze.json").read_text(encoding="utf-8")
        )
        result = json.loads(
            (artifact_path / "training_result.json").read_text(encoding="utf-8")
        )
        verified = verify_model_v2_anchor_refined_training_result(
            root, config_path
        )

        self.assertTrue(verified["passed"])
        self.assertEqual(candidate["normalizer_fits"], 0)
        self.assertEqual(
            candidate["normalizer_sha256"],
            "e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a",
        )
        protected_paths = {
            str(row["path"])
            for row in (
                *document["protected_v1"]["checkpoints"],
                *document["baseline_v2"]["checkpoints"],
                *document["rebalanced_v2"]["checkpoints"],
            )
        }
        self.assertFalse(protected_paths & set(candidate["checkpoint_sha256"]))
        self.assertFalse(result["generalization_validation_v2_inference"])
        self.assertFalse(result["unified_holdout_waveform_reopened"])
        self.assertFalse(result["unified_holdout_new_inference"])
        self.assertFalse(result["generalization_holdout_waveform_opened"])
        self.assertFalse(result["generalization_holdout_inference"])
        self.assertEqual(result["generalization_holdout_guard_count"], 0)

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
