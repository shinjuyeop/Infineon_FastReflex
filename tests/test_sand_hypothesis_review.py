from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml

from fastreflex.dataset.hazard import canonical_sha256
from fastreflex.dataset.loader import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiment/20260903_sand_generalization_hypothesis_review.yaml"
ARTIFACTS = ROOT / "artifacts/runs/20260903_sand_generalization_hypothesis_review"


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


class SandHypothesisReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_review_contract_is_read_only(self) -> None:
        document = self.document
        self.assertEqual(document["experiment"]["starting_commit"], "ce8070930988872566c90121fc5e88d8c23fc4a8")
        self.assertTrue(document["experiment"]["review_only"])
        self.assertFalse(document["experiment"]["new_simulation"])
        self.assertFalse(document["experiment"]["model_inference"])
        self.assertFalse(document["experiment"]["training_or_tuning"])
        self.assertEqual(document["candidate_next_hypotheses"]["maximum_primary_selection"], 1)
        self.assertTrue(document["historical_status"]["rewrite_prohibited"])
        self.assertEqual(set(document["counters"].values()), {0})

    def test_frozen_evidence_hashes_and_composites(self) -> None:
        for key in ("discovery_evidence", "confirmation_evidence"):
            section = self.document[key]
            artifact_path = ROOT / section["artifact_path"]
            evidence = {
                "config_sha256": sha256_file(ROOT / section["config"]["path"]),
                "report_sha256": sha256_file(ROOT / section["report"]["path"]),
                "files": {
                    name: sha256_file(artifact_path / name)
                    for name in sorted(section["files"])
                },
            }
            self.assertEqual(evidence["config_sha256"], section["config"]["sha256"])
            self.assertEqual(evidence["report_sha256"], section["report"]["sha256"])
            self.assertEqual(evidence["files"], section["files"])
            self.assertEqual(canonical_sha256(evidence), section["canonical_evidence_input_sha256"])

    def test_review_artifacts_are_hash_frozen(self) -> None:
        review = _load_json(ARTIFACTS / "hypothesis_review.json")
        files = {
            "representation_geometry_review_sha256": "representation_geometry_review.json",
            "localization_replication_review_sha256": "localization_replication_review.json",
            "metric_robustness_review_sha256": "metric_robustness_review.json",
            "next_hypothesis_decision_sha256": "next_hypothesis_decision.json",
        }
        for field, filename in files.items():
            self.assertEqual(sha256_file(ARTIFACTS / filename), review["component_hashes"][field])
        semantic = dict(review)
        expected = semantic.pop("SAND_GENERALIZATION_HYPOTHESIS_REVIEW_SHA")
        self.assertEqual(canonical_sha256(semantic), expected)
        self.assertEqual(sha256_file(CONFIG), review["review_config_sha256"])

    def test_selected_future_hypothesis_and_deployment_boundary(self) -> None:
        decision = _load_json(ARTIFACTS / "next_hypothesis_decision.json")
        self.assertEqual(
            decision["selected_next_hypothesis"],
            "FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS",
        )
        self.assertEqual(
            decision["review_verdict"],
            "SAND_GENERALIZATION_HYPOTHESIS_REVIEW_ACTIONABLE",
        )
        self.assertTrue(decision["deployment_parallelization"]["can_proceed"])
        self.assertTrue(
            decision["deployment_parallelization"]["reference_only_not_final_supported_model"]
        )
        self.assertFalse(decision["discovery_result_changed"])
        self.assertFalse(decision["confirmation_result_changed"])

    def test_historical_holdout_guard_is_untouched(self) -> None:
        holdout = self.document["historical_holdout"]
        self.assertEqual(holdout["guard"], 1)
        self.assertEqual(holdout["scientific_opens"], 1)
        self.assertEqual(holdout["payload_reads_now"], 0)
        self.assertEqual(holdout["inference_now"], 0)
        self.assertEqual(
            sha256_file(ROOT / holdout["guard_path"]),
            holdout["guard_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
