"""Regression tests for censor-aware Sand calibration and redesign."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fastreflex.dataset.generation import _load_yaml
from fastreflex.dataset.sand_calibration import (
    _calibration_result_summary,
    collect_sand_benign_redesigned_study,
    collect_sand_calibration_batch,
    expand_sand_benign_redesign,
    load_sand_benign_redesigned_discovery_payload,
    validate_sand_benign_redesign,
    verify_sand_benign_redesigned_dataset,
)
from fastreflex.dataset.loader import sha256_file


ROOT = Path(__file__).resolve().parents[1]
REDESIGN = (
    ROOT / "configs/experiment/20260902_sand_benign_generalization_study_redesign.yaml"
)
REDESIGNED_DATASET = (
    ROOT / "data/raw/sand_benign_generalization_redesigned_study_20260902"
)
GENERATION_CONFIG = (
    ROOT
    / "configs/experiment/20260902_sand_benign_generalization_study_redesigned_generation.yaml"
)
FAILURE_REVIEW_CONFIG = (
    ROOT
    / "configs/experiment/20260902_sand_benign_redesigned_domain_failure_review.yaml"
)


def _row(group: str) -> dict[str, object]:
    return {
        "group": group,
        "designed_side": "LEFT",
        "slip_event_summary": {"first_sample": None},
        "i1_summary": {"first_sample": None},
        "support_event_summary": {"first_sample": None, "side": "NONE"},
        "target_contact_summary": {},
    }


def _arrays(*, censor: int = 500, fall: int = -1) -> dict[str, np.ndarray]:
    target = np.zeros((600, 2), dtype=bool)
    target[100:200, 0] = True
    phase = np.full(600, 3, dtype=np.int8)
    phase[80:100] = 1
    loaded = np.ones((600, 2), dtype=bool)
    return {
        "censor_sample": np.asarray(censor),
        "first_fall_sample": np.asarray(fall),
        "target_terrain_contact": target,
        "target_terrain_touchdown": np.zeros((600, 2), dtype=bool),
        "gait_phase": phase,
        "loaded_contact": loaded,
        "support_surface_max_displacement_m": np.zeros(600),
        "support_surface_spread_m": np.zeros((600, 2)),
    }


CONTRACT = {
    "precontact_phase_lookback_ms": 20,
    "benign_post_target_followup_ms": 100,
    "support_post_event_followup_ms": 100,
}


class SandCalibrationTest(unittest.TestCase):
    def test_phase_uses_precontact_sample_not_touchdown_sample(self) -> None:
        row = _row("sand_benign")
        arrays = _arrays()
        _calibration_result_summary(row, arrays, CONTRACT)
        self.assertTrue(row["valid"])
        self.assertEqual(row["objective_physical_outcome"], "STRICT_BENIGN")
        summary = row["target_contact_summary"]
        self.assertEqual(summary["contact_sample_phase"], "DOUBLE_SUPPORT")
        self.assertEqual(summary["precontact_phase"], "LEFT_SINGLE_SUPPORT")

    def test_fully_observed_support_survives_later_fall_censor(self) -> None:
        row = _row("ordinary_support_control")
        row["i1_summary"] = {"first_sample": 120}
        row["support_event_summary"] = {
            "first_sample": 200,
            "side": "LEFT_ONLY",
        }
        arrays = _arrays(censor=500, fall=500)
        _calibration_result_summary(row, arrays, CONTRACT)
        self.assertTrue(row["valid"])
        self.assertEqual(row["objective_physical_outcome"], "SUPPORT")

    def test_actual_slip_overrides_benign_intent_without_replacement(self) -> None:
        row = _row("near_hazard_sand_benign")
        row["slip_event_summary"] = {"first_sample": 150}
        arrays = _arrays(censor=250, fall=250)
        _calibration_result_summary(row, arrays, CONTRACT)
        self.assertTrue(row["valid"])
        self.assertEqual(row["objective_physical_outcome"], "SLIP")
        self.assertFalse(row["intent_match"])

    def test_expected_nine_second_trace_is_enforced(self) -> None:
        row = _row("broad_sand_benign")
        contract = {**CONTRACT, "expected_samples": 9000}
        _calibration_result_summary(row, _arrays(), contract)
        self.assertFalse(row["valid"])
        self.assertEqual(row["invalid_reason"], "nonfinite_or_malformed")

    def test_redesign_expands_to_fresh_balanced_matrix(self) -> None:
        document = _load_yaml(REDESIGN)
        rows = expand_sand_benign_redesign(document)
        self.assertEqual(len(rows), 176)
        self.assertEqual(
            sum(row["split"] == "REDESIGNED_DISCOVERY" for row in rows), 88
        )
        with patch(
            "fastreflex.dataset.sand_calibration._historical_signatures",
            return_value=(set(), []),
        ):
            audit = validate_sand_benign_redesign(ROOT, document)
        self.assertEqual(audit["historical_signature_overlap"], 0)
        self.assertEqual(audit["cross_split_exact_overlap"], 0)
        self.assertEqual(audit["cross_split_parameter_near_duplicates"], 0)
        self.assertEqual(
            audit["redesign_sha256"],
            document["design_hashes"]["SAND_BENIGN_GENERALIZATION_STUDY_REDESIGN_SHA"],
        )

    def test_execution_config_freezes_implementation_and_boundaries(self) -> None:
        generation = _load_yaml(GENERATION_CONFIG)["generation"]
        self.assertEqual(generation["planned_total_runs"], 176)
        self.assertEqual(generation["planned_discovery_runs"], 88)
        self.assertEqual(generation["planned_confirmation_runs"], 88)
        self.assertEqual(generation["redesign_config_sha256"], sha256_file(REDESIGN))
        for artifact in generation["implementation_artifacts"]:
            self.assertEqual(artifact["sha256"], sha256_file(ROOT / artifact["path"]))
        for artifact in generation["protected_artifacts"]:
            self.assertEqual(artifact["sha256"], sha256_file(ROOT / artifact["path"]))

    def test_calibration_collector_has_no_model_inference(self) -> None:
        source = inspect.getsource(collect_sand_calibration_batch)
        for forbidden in (
            "predict_proba(",
            "load_model(",
            "torch.load(",
            "onnxruntime.InferenceSession",
        ):
            self.assertNotIn(forbidden, source)

    def test_redesigned_collector_has_no_model_inference(self) -> None:
        source = inspect.getsource(collect_sand_benign_redesigned_study)
        for forbidden in (
            "predict_proba(",
            "load_model(",
            "torch.load(",
            "onnxruntime.InferenceSession",
        ):
            self.assertNotIn(forbidden, source)

    def test_redesigned_confirmation_loader_refuses_before_npz(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            manifest = {
                "dataset_id": ("sand_benign_generalization_redesigned_study_20260902"),
                "runs": [
                    {
                        "run_id": "sealed",
                        "split": "REDESIGNED_CONFIRMATION",
                        "file": "must_not_exist.npz",
                        "file_sha256": "unreachable",
                    }
                ],
            }
            path = dataset / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            (dataset / "manifest.sha256").write_text(
                f"{sha256_file(path)}  manifest.json\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "SEALED"):
                load_sand_benign_redesigned_discovery_payload(dataset, "sealed")

    def test_generated_redesigned_dataset_hashes_are_deterministic(self) -> None:
        if not REDESIGNED_DATASET.exists():
            self.skipTest("redesigned corpus has not been generated yet")
        verification = verify_sand_benign_redesigned_dataset(REDESIGNED_DATASET)
        self.assertTrue(verification["passed"])
        self.assertEqual(verification["run_count"], 176)

    def test_failure_review_contract_is_model_blind_and_frozen(self) -> None:
        review = _load_yaml(FAILURE_REVIEW_CONFIG)
        self.assertEqual(
            sha256_file(FAILURE_REVIEW_CONFIG),
            "ddae06af4167e75b02ad5060a644fc516f99a68d9f9234f27b9d229b258515da",
        )
        self.assertEqual(
            review["review"]["dataset_freeze_sha"],
            "87956c511684a78780d8bc7c1ac50552779de55b85a739e0221d1fa449f9416a",
        )
        self.assertEqual(len(review["review"]["failed_gate_ids"]), 3)
        self.assertEqual(review["review"]["old_holdout_guard"], 1)
        for counter in review["counters"].values():
            self.assertEqual(counter, 0)
        self.assertTrue(review["protocol_guards"]["no_model_inference"])
        self.assertTrue(
            review["protocol_guards"]["old_holdout_payload_access_forbidden"]
        )

    def test_redesigned_failure_is_fall_censored_not_horizon_censored(self) -> None:
        if not REDESIGNED_DATASET.exists():
            self.skipTest("redesigned corpus has not been generated yet")
        manifest = json.loads(
            (REDESIGNED_DATASET / "manifest.json").read_text(encoding="utf-8")
        )
        invalid = [row for row in manifest["runs"] if not row["valid"]]
        insufficient = [
            row
            for row in invalid
            if row["invalid_reason"] == "insufficient_post_target_observation"
        ]
        pretarget = [
            row for row in invalid if row["invalid_reason"] == "pretarget_fall"
        ]
        self.assertEqual((len(invalid), len(insufficient), len(pretarget)), (23, 20, 3))
        for row in insufficient:
            target = row["target_contact_summary"]["first_sample"]
            fall = row["fall_censor_summary"]["first_fall_sample"]
            self.assertIsNotNone(target)
            self.assertIsNotNone(fall)
            self.assertLess(target, fall)
            self.assertLess(fall, row["actual_samples"])
            self.assertEqual(
                row["fall_censor_summary"]["fall_reasons"],
                ["nonfoot_surface_contact"],
            )
            self.assertIsNone(row["slip_event_summary"]["first_sample"])
            self.assertIsNone(row["i1_summary"]["first_sample"])
            self.assertIsNone(row["support_event_summary"]["first_sample"])

        audit = json.loads(
            (REDESIGNED_DATASET / "physical_audit.json").read_text(encoding="utf-8")
        )
        failed = {
            gate_id
            for gate_id, result in audit["generation_gates"].items()
            if not result["passed"]
        }
        self.assertEqual(
            failed,
            {
                "yield/REDESIGNED_CONFIRMATION/broad_mild",
                "yield/REDESIGNED_CONFIRMATION/concrete/0.25/strict_sand",
                "yield/REDESIGNED_CONFIRMATION/strict_sand",
            },
        )
        freeze = json.loads(
            (REDESIGNED_DATASET / "dataset_freeze.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            freeze["generation_verdict"],
            "SAND_BENIGN_GENERALIZATION_STUDY_REDESIGNED_GENERATION_YIELD_INSUFFICIENT",
        )
        seal = json.loads(
            (REDESIGNED_DATASET / "confirmation_seal.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(seal["status"], "SEALED_FOR_REDESIGNED_CONFIRMATION")
        self.assertFalse(seal["model_inference"])


if __name__ == "__main__":
    unittest.main()
