"""Contracts for the one-cycle factor-conditioned Sand intervention."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import yaml

from fastreflex.dataset.generation import canonical_sha256, sha256_file
from fastreflex.dataset.sand_factor_conditioned import (
    _factor_conditioned_eligible,
    expand_factor_conditioned_design,
    validate_factor_conditioned_design,
    verify_factor_conditioned_dataset,
)
from fastreflex.features import feature_schema_hash
from fastreflex.training.hazard import (
    model_v2_anchor_refined_policy,
    prepare_model_v2_training_data,
    run_factor_conditioned_data_intervention,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/experiment/20260903_sand_factor_conditioned_data_intervention.yaml"
)
CONFIG_SHA256 = "540034673d1703adce000182b73e2dc4c4bf8856e534e7489ea66bef6522246e"


class SandFactorConditionedContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_fresh_split_and_deterministic_design_are_frozen(self) -> None:
        self.assertEqual(sha256_file(CONFIG), CONFIG_SHA256)
        first = expand_factor_conditioned_design(self.document)
        second = expand_factor_conditioned_design(self.document)
        self.assertEqual(first, second)
        audit = validate_factor_conditioned_design(ROOT, self.document)
        self.assertEqual(audit["run_count"], 162)
        self.assertEqual(
            audit["split_counts"],
            {"FACTOR_TRAIN": 108, "FACTOR_VALIDATION": 54},
        )
        self.assertEqual(audit["unique_run_ids"], 162)
        self.assertEqual(audit["unique_scenario_signatures"], 162)
        self.assertEqual(audit["cross_split_exact_overlap"], 0)
        self.assertEqual(audit["cross_split_parameter_near_overlap"], 0)

    def test_historical_contamination_audit_never_opens_payloads(self) -> None:
        with patch("numpy.load", side_effect=AssertionError("payload opened")):
            audit = validate_factor_conditioned_design(ROOT, self.document)
        contamination = audit["historical_contamination"]
        self.assertEqual(contamination["exact_total"], 0)
        self.assertEqual(contamination["near_total"], 0)
        self.assertEqual(contamination["run_id_reuse_total"], 0)
        changed = deepcopy(self.document)
        changed["scenario_matrix"]["profiles"]["FACTOR_VALIDATION"][
            "sand_mild_standard"
        ][0].update(patch_start_x_m=0.326, patch_width_m=0.842)
        with self.assertRaisesRegex(ValueError, "duplicate|overlap"):
            validate_factor_conditioned_design(ROOT, changed)

    def test_actual_physics_not_design_intent_controls_eligibility(self) -> None:
        base = {
            "valid": True,
            "intent_match": True,
            "objective_physical_outcome": "STRICT_BENIGN",
            "actual_benign_severity": "LOW",
            "group": "sand_benign_mild",
        }
        self.assertTrue(_factor_conditioned_eligible(base))
        wrong_severity = {**base, "actual_benign_severity": "MEDIUM"}
        self.assertFalse(_factor_conditioned_eligible(wrong_severity))
        support = {
            **base,
            "group": "delayed_support_control",
            "objective_physical_outcome": "SUPPORT",
            "actual_benign_severity": None,
        }
        self.assertTrue(_factor_conditioned_eligible(support))
        self.assertFalse(
            _factor_conditioned_eligible({**support, "intent_match": False})
        )

    def test_model_normalizer_and_runtime_decision_are_immutable(self) -> None:
        document = self.document
        self.assertEqual(
            canonical_sha256(document["architecture"]),
            "ae4753699399aa7e4a93639935e085156ef28ed100a77058d7308eb1549aa897",
        )
        self.assertEqual(document["architecture"]["input_shape"], [20, 80])
        self.assertEqual(document["architecture"]["hidden_size"], 32)
        self.assertEqual(feature_schema_hash(), document["features"]["schema_sha256"])
        self.assertEqual(
            canonical_sha256(model_v2_anchor_refined_policy()),
            document["training_protocol"]["extraction_policy_sha256"],
        )
        self.assertEqual(
            sha256_file(ROOT / document["normalizer"]["path"]),
            document["normalizer"]["sha256"],
        )
        self.assertEqual(
            document["runtime_decision"],
            {
                "ensemble": "mean_probability_all_three_predeclared_seeds",
                "threshold": 0.99,
                "persistence_ms": 5,
            },
        )

    def test_training_sources_and_candidate_freeze_order_are_closed(self) -> None:
        protocol = self.document["training_protocol"]
        self.assertEqual(
            protocol["data_sources"], ["Unified_TRAIN", "V2_TRAIN", "FACTOR_TRAIN"]
        )
        self.assertIn("Sand_Confirmation", protocol["forbidden_sources"])
        self.assertIn(
            "historical_Generalization_HOLDOUT", protocol["forbidden_sources"]
        )
        self.assertEqual(self.document["hnm"]["source"], "effective_TRAIN_only")
        self.assertEqual(self.document["hnm"]["rounds"], 3)
        self.assertEqual(
            self.document["training"]["seeds"], [20260828, 20260829, 20260830]
        )
        source = Path(
            run_factor_conditioned_data_intervention.__code__.co_filename
        ).read_text(encoding="utf-8")
        self.assertLess(
            source.index("candidate_freeze_sha = sha256_file(candidate_freeze_path)"),
            source.index("validation_runs, _, validation_rows"),
        )
        self.assertIn('"factor_validation_evaluated": False', source)
        self.assertIn('"v2_validation_evaluated": False', source)

    def test_historical_model_dataset_and_consumed_guard_are_intact(self) -> None:
        base = prepare_model_v2_training_data(ROOT, self.document)
        self.assertEqual(len(base.runs), 442)
        for record in self.document["reference_model"]["checkpoints"]:
            self.assertEqual(sha256_file(ROOT / record["path"]), record["sha256"])
        guard_record = self.document["historical_evidence_boundary"]
        guard_path = ROOT / guard_record["holdout_guard_path"]
        self.assertEqual(sha256_file(guard_path), guard_record["holdout_guard_sha256"])
        guard = json.loads(guard_path.read_text(encoding="utf-8"))
        self.assertEqual(guard["guard_after"], 1)
        self.assertEqual(guard["scientific_open_count"], 1)

    def test_failed_physical_freeze_blocks_training_before_payload_access(self) -> None:
        dataset_path = ROOT / self.document["factor_dataset"]["path"]
        verification = verify_factor_conditioned_dataset(dataset_path)
        self.assertTrue(verification["passed"])
        self.assertEqual(verification["run_count"], 162)
        with patch(
            "fastreflex.training.hazard.np.load",
            side_effect=AssertionError("training payload opened"),
        ):
            with self.assertRaisesRegex(RuntimeError, "training is prohibited"):
                run_factor_conditioned_data_intervention(ROOT, CONFIG, dry_run=True)
        artifact_path = ROOT / self.document["artifacts"]["path"]
        self.assertFalse((artifact_path / "pretraining_audit.json").exists())


if __name__ == "__main__":
    unittest.main()
