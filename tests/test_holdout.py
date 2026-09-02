"""Frozen and post-open contracts for the Generalization HOLDOUT evaluator."""

from __future__ import annotations

import ast
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import yaml

from fastreflex.dataset.loader import sha256_file
from fastreflex.evaluation.holdout import (
    EXPERIMENT_ID,
    FINAL_CANDIDATE_ID,
    PRIMARY_CONTRACT_SHA256,
    SECONDARY_CONTRACT_SHA256,
    STARTING_COMMIT,
    VERDICT_HIERARCHY_SHA256,
    _claim_guard,
    _verdicts,
    load_holdout_yaml,
    preflight_holdout_evaluation,
    verify_generalization_holdout_evaluation,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/20260902_model_v2_generalization_holdout_one_shot_evaluation.yaml"
)
EXPECTED_CONFIG_SHA256 = (
    "ec53c761f426aaeba5528916c60a6c3f69550007987cdf5f3754304cd4bbef0a"
)


class GeneralizationHoldoutEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = load_holdout_yaml(CONFIG)
        artifact_path = ROOT / cls.document["artifacts"]["path"]
        cls.result = json.loads(
            (artifact_path / "evaluation_result.json").read_text(encoding="utf-8")
        )
        cls.guard = json.loads(
            (artifact_path / "holdout_access_guard.json").read_text(encoding="utf-8")
        )
        with patch(
            "fastreflex.evaluation.generalization.np.load",
            side_effect=AssertionError("HOLDOUT payload access during verification"),
        ):
            cls.verification = verify_generalization_holdout_evaluation(ROOT, CONFIG)

    def test_execution_config_and_final_candidate_are_exact(self) -> None:
        self.assertEqual(self.document["experiment"]["id"], EXPERIMENT_ID)
        self.assertEqual(sha256_file(CONFIG), EXPECTED_CONFIG_SHA256)
        self.assertEqual(
            self.document["final_candidate"]["candidate_id"],
            FINAL_CANDIDATE_ID,
        )
        self.assertEqual(
            self.result["final_candidate_sha256"],
            "52644d3efb0e756002bd43a7f94aea0a16a65eb2ff014c72548f2b3b138b7bc2",
        )

    def test_v1_and_terrain_candidates_are_exact_and_read_only(self) -> None:
        self.assertEqual(
            self.document["baseline_v1"]["freeze_sha256"],
            "91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2",
        )
        self.assertTrue(self.document["baseline_v1"]["read_only"])
        self.assertTrue(self.document["terrain_v1"]["read_only"])
        self.assertTrue(self.document["terrain_v1"]["advisory_only"])
        self.assertFalse(self.document["terrain_v1"]["hazard_gate"])
        self.assertTrue(self.result["v1_hazard_inference"])
        self.assertTrue(self.result["terrain_v1_inference"])

    def test_primary_secondary_and_verdict_contracts_are_exact(self) -> None:
        self.assertEqual(
            PRIMARY_CONTRACT_SHA256,
            "feabfc4519e8ec28e59710810b6e587b7a8be1a128ecf57a028d32710c1b246e",
        )
        self.assertEqual(
            SECONDARY_CONTRACT_SHA256,
            "085d6f73156a5618767284faa2ccdcd29d3645694f56155431159d533b77130a",
        )
        self.assertEqual(
            VERDICT_HIERARCHY_SHA256,
            "e86fb11f457734c41cd7b9c66a827a22b587f7a1f95aa91130931f7586c8cba5",
        )
        self.assertFalse(
            self.document["secondary_contract"]["primary_score_replacement"]
        )

    def test_saved_result_verification_never_deserializes_holdout_payload(self) -> None:
        self.assertTrue(self.verification["passed"])
        self.assertFalse(self.verification["holdout_payload_deserialized"])
        self.assertEqual(self.verification["guard_after"], 1)
        self.assertEqual(self.verification["scientific_open_count"], 1)
        self.assertEqual(self.result["holdout_runs"], 36)
        self.assertEqual(self.result["payload_deserializations"], 36)
        self.assertEqual(self.result["payload_deserializations_per_run"], 1)

    def test_evaluator_rejects_any_role_other_than_final_candidate(self) -> None:
        document = deepcopy(self.document)
        document["final_candidate"]["role"] = "development_candidate"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wrong_role.yaml"
            path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            with patch(
                "fastreflex.evaluation.holdout._git_revision",
                return_value=STARTING_COMMIT,
            ):
                with self.assertRaisesRegex(RuntimeError, "exact final candidate"):
                    preflight_holdout_evaluation(ROOT, path)

    def test_evaluator_rejects_nonzero_or_existing_guard(self) -> None:
        with (
            patch(
                "fastreflex.evaluation.holdout._git_revision",
                return_value=STARTING_COMMIT,
            ),
            patch(
                "fastreflex.evaluation.generalization.np.load",
                side_effect=AssertionError("HOLDOUT payload access during guard check"),
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "HOLDOUT_EVALUATION_ABORTED_GUARD_STATE"
            ):
                preflight_holdout_evaluation(ROOT, CONFIG)
        self.assertEqual((self.guard["guard_before"], self.guard["guard_after"]), (0, 1))
        self.assertTrue(self.guard["second_scientific_open_forbidden"])
        self.assertFalse(self.result["second_open_attempted"])

    def test_guard_claim_is_atomic_and_refuses_second_claim(self) -> None:
        document = deepcopy(self.document)
        with tempfile.TemporaryDirectory() as temporary:
            artifact_path = Path(temporary) / "one_shot"
            document["artifacts"]["path"] = str(artifact_path)
            document["guard"]["record_path"] = str(
                artifact_path / "holdout_access_guard.json"
            )
            path = Path(temporary) / "one_shot.yaml"
            path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
            with (
                patch(
                    "fastreflex.evaluation.holdout._git_revision",
                    return_value=STARTING_COMMIT,
                ),
                patch(
                    "fastreflex.evaluation.generalization.np.load",
                    side_effect=AssertionError("HOLDOUT payload access during preflight"),
                ),
            ):
                preflight = preflight_holdout_evaluation(ROOT, path)
            guard, guard_path = _claim_guard(ROOT, path, preflight)
            self.assertEqual((guard["guard_before"], guard["guard_after"]), (0, 1))
            self.assertTrue(guard_path.is_file())
            with self.assertRaisesRegex(
                RuntimeError, "HOLDOUT_EVALUATION_ABORTED_GUARD_STATE"
            ):
                _claim_guard(ROOT, path, preflight)

    def test_evaluator_has_no_training_or_optimizer_import_path(self) -> None:
        paths = [
            ROOT / "src/fastreflex/evaluation/holdout.py",
            ROOT / "src/fastreflex/evaluation/generalization.py",
            ROOT / "src/fastreflex/evaluation/hazard.py",
            ROOT / "src/fastreflex/evaluation/terrain.py",
            ROOT / "src/fastreflex/models/checkpoint.py",
        ]
        imported: set[str] = set()
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported.update(
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module is not None
            )
            imported.update(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
        self.assertFalse(
            any(module.startswith("fastreflex.training") for module in imported)
        )
        self.assertFalse(self.document["execution"]["trainer_or_optimizer_path"])

    def test_verdict_logic_is_deterministic_on_non_holdout_fixture(self) -> None:
        result = {
            "gates": {
                "overall_hazard_recall": True,
                "slip_hazard_recall": True,
                "support_hazard_recall": True,
                "primary_no_hazard_specificity": True,
                "ice_benign_specificity": True,
                "system_premature_run_rate": True,
                "slip_p95_latency_ms": True,
                "support_p95_established_latency_ms": True,
            }
        }
        expected = _verdicts(result, [])
        self.assertEqual(expected, _verdicts(result, []))
        self.assertEqual(
            expected["final_holdout_verdict"],
            "MODEL_V2_GENERALIZATION_HOLDOUT_SUPPORTED",
        )


if __name__ == "__main__":
    unittest.main()
