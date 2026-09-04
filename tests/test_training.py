"""Tests for generic training and current Hazard TRAIN-only contracts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

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
from tests.support import (
    REPOSITORY_ROOT as ROOT,
    assert_attributes,
    assert_false_fields,
    assert_mapping_values,
    load_json,
    load_yaml,
)


MODEL_V2_DATASET = ROOT / "data/raw/model_v2_hazard_reflex_20260901"


def _extraction_audit(config_name: str, auditor) -> dict[str, object]:
    if not MODEL_V2_DATASET.is_dir():
        raise unittest.SkipTest("frozen Model V2 dataset is not available")
    document = load_yaml(ROOT / "configs/experiment" / config_name)
    data = prepare_model_v2_training_data(ROOT, document)
    return auditor(data.runs, data.precursor_samples, data.annotations)


def _candidate_contract(config_name: str, verifier, artifact_path: str | None = None):
    config_path = ROOT / "configs/experiment" / config_name
    document = load_yaml(config_path)
    artifact = ROOT / (
        artifact_path if artifact_path is not None else document["artifacts"]["path"]
    )
    if not (artifact / "training_result.json").is_file():
        raise unittest.SkipTest("frozen Model V2 candidate is not available")
    return (
        document,
        load_json(artifact / "candidate_freeze.json"),
        load_json(artifact / "training_result.json"),
        verifier(ROOT, config_path),
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


class ModelV2ExtractionTest(unittest.TestCase):
    def test_model_v2_anchor_refined_extraction_matches_frozen_design(self) -> None:
        audit = _extraction_audit(
            "20260902_model_v2_anchor_refined_training.yaml",
            audit_model_v2_anchor_refined_extraction,
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
        assert_mapping_values(
            audit,
            {
                "positive_window_ids_sha256": "248719864bc1974ac54a21de63f04a6d5e6f55ef3e3c37092cf0ec757872d09e",
                "negative_window_ids_sha256": "392c1fda06953135bfed9a97c7cabb3915d4c50fdf9ff11b2ac8e6550448936c",
                "masked_window_sha256": "32bae2d81b05709771545b375dd6cffb95d2dfc17c2808a7adf3a7f70174c35a",
                "monitor_endpoint_sha256": "39d30234f674446f305b1b51d446977ba301e6db1e0591ac14dbe7172cbb1bf5",
                "monitor_positive_sha256": "e4cd285091e55c92c773512b44958273d7773a708bad876806cec6a8401f9c88",
                "all_positive_count": 3_135,
                "all_negative_count": 32_209,
                "fit_positive_counts": {
                    "slip": 1_680,
                    "ordinary_support": 640,
                    "delayed_support": 198,
                    "support": 838,
                    "total": 2_518,
                },
                "fit_negative_count": 25_585,
                "monitor_positive_counts": {
                    "slip": 431,
                    "ordinary_support": 167,
                    "delayed_support_concrete": 8,
                    "delayed_support_marble": 11,
                    "total": 617,
                },
                "monitor_negative_count": 6_624,
                "fit_monitor_endpoint_overlap": 0,
            },
        )
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
        audit = _extraction_audit(
            "20260901_model_v2_extraction_rebalanced_training.yaml",
            audit_model_v2_rebalanced_extraction,
        )

        self.assertTrue(audit["passed"])
        self.assertEqual(
            canonical_sha256(model_v2_rebalance_policy()),
            "3c7ce82ed905d932ec8f17d69d7e5edb5d79ee7602ba95ffe2a53d2407142cd2",
        )
        assert_mapping_values(
            audit,
            {
                "effective_train_run_count": 442,
                "positive_window_ids_sha256": "498f5d1f4419e3bfa72fc2f9649326db26f00e7a9523d9b3ecc8032436a3e0bb",
                "negative_window_ids_sha256": "392c1fda06953135bfed9a97c7cabb3915d4c50fdf9ff11b2ac8e6550448936c",
                "masked_window_sha256": "32bae2d81b05709771545b375dd6cffb95d2dfc17c2808a7adf3a7f70174c35a",
                "all_positive_count": 3_188,
                "all_negative_count": 32_209,
                "fit_positive_counts": {
                    "slip": 1_680,
                    "ordinary_support": 640,
                    "delayed_support": 270,
                    "support": 910,
                    "total": 2_590,
                },
                "fit_negative_count": 25_585,
                "monitor_positive_count": 598,
                "monitor_negative_count": 6_624,
                "delayed_support": {
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
                "masked_sample_counts": {
                    "future_slip_precursor": 41_479,
                    "censored_precursor": 1_734,
                    "i1_positive": 68_388,
                },
            },
        )
        self.assertEqual(set(audit["contradiction_audit"].values()), {0})

    def test_frozen_candidates_reuse_normalizer_and_isolate_artifacts(self) -> None:
        cases = (
            (
                "rebalanced",
                "20260901_model_v2_extraction_rebalanced_training.yaml",
                verify_model_v2_extraction_rebalanced_training_result,
                "artifacts/runs/20260901_model_v2_extraction_rebalanced_training",
                ("protected_v1", "baseline_v2"),
                (
                    "generalization_validation_v2_inference",
                    "unified_holdout_waveform_reopened",
                    "generalization_holdout_waveform_opened",
                    "generalization_holdout_inference",
                ),
            ),
            (
                "anchor_refined",
                "20260902_model_v2_anchor_refined_training.yaml",
                verify_model_v2_anchor_refined_training_result,
                None,
                ("protected_v1", "baseline_v2", "rebalanced_v2"),
                (
                    "generalization_validation_v2_inference",
                    "unified_holdout_waveform_reopened",
                    "unified_holdout_new_inference",
                    "generalization_holdout_waveform_opened",
                    "generalization_holdout_inference",
                ),
            ),
        )
        for name, config, verifier, artifact, groups, false_fields in cases:
            with self.subTest(candidate=name):
                document, candidate, result, verified = _candidate_contract(
                    config, verifier, artifact
                )
                self.assertTrue(verified["passed"])
                self.assertEqual(candidate["normalizer_fits"], 0)
                self.assertEqual(
                    candidate["normalizer_sha256"],
                    "e0d796e8840e0cd38bc7d0ed222b668187a8a661748cf8506d4141657f88e92a",
                )
                self.assertEqual(
                    sha256_file(ROOT / candidate["normalizer_path"]),
                    candidate["normalizer_sha256"],
                )
                records = tuple(
                    record
                    for group in groups
                    for record in document[group]["checkpoints"]
                )
                protected_paths = {str(record["path"]) for record in records}
                self.assertFalse(protected_paths & set(candidate["checkpoint_sha256"]))
                for record in records:
                    self.assertEqual(
                        sha256_file(ROOT / record["path"]), record["sha256"]
                    )
                assert_false_fields(result, *false_fields)
                self.assertEqual(result["generalization_holdout_guard_count"], 0)


class TrainingCoreTest(unittest.TestCase):
    def test_supported_gru_architecture_is_80_32_one_layer_two_outputs(self) -> None:
        model = build_model("gru", 20, 80, class_count=2)
        assert_attributes(
            model.gru,
            {
                "input_size": 80,
                "hidden_size": 32,
                "num_layers": 1,
                "bidirectional": False,
            },
        )
        assert_attributes(model.classifier, {"out_features": 2})
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
        self.assertEqual(
            (
                HNM_ROUNDS,
                HNM_REPLAY_STRIDE_MS,
                HNM_TOP_K_PER_RUN,
                HNM_MINIMUM_SPACING_MS,
            ),
            (3, 1, 12, 30),
        )

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
        self.assertEqual(set(audit["ordinary_negative_violations"].values()), {0})

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
