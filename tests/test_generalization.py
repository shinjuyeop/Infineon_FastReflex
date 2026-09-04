"""Contracts for the frozen Generalization development evaluation."""

from __future__ import annotations

import json
import unittest

from fastreflex.dataset.hazard import canonical_sha256
from fastreflex.dataset.loader import sha256_file
from fastreflex.evaluation.generalization import (
    EXPERIMENT_ID,
    HOLDOUT_COUNT,
    HOLDOUT_SPLIT,
    VALIDATION_COUNT,
    VALIDATION_SPLIT,
    _canonical_json,
    _load_yaml,
    load_generalization_manifest,
    load_generalization_split,
    verify_generalization_development_evaluation,
    verify_promoted_candidate,
)
from tests.support import REPOSITORY_ROOT as ROOT, load_json

CONFIG = (
    ROOT
    / "configs/experiment/20260902_model_v2_generalization_development_evaluation.yaml"
)


class GeneralizationEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = _load_yaml(CONFIG)

    def test_promoted_candidate_resolves_without_alternate_selection(self) -> None:
        candidate = verify_promoted_candidate(ROOT, self.document)
        self.assertTrue(candidate["passed"])
        self.assertEqual(
            candidate["candidate_id"],
            "model_v2_anchor_refined_gru20_20260902",
        )
        self.assertEqual(candidate["threshold"], 0.99)
        self.assertEqual(candidate["persistence_ms"], 5)
        self.assertEqual(
            self.document["candidate"]["alternate_candidate_selection"],
            "prohibited",
        )

    def test_manifest_resolves_frozen_split_membership(self) -> None:
        manifest = load_generalization_manifest(ROOT, self.document)
        validation_ids = [
            row["run_id"]
            for row in manifest["runs"]
            if row["split"] == VALIDATION_SPLIT
        ]
        holdout_ids = [
            row["run_id"] for row in manifest["runs"] if row["split"] == HOLDOUT_SPLIT
        ]
        self.assertEqual(len(validation_ids), VALIDATION_COUNT)
        self.assertEqual(len(holdout_ids), HOLDOUT_COUNT)
        self.assertEqual(
            canonical_sha256(validation_ids),
            self.document["dataset"]["validation_run_ids_canonical_sha256"],
        )
        self.assertEqual(
            canonical_sha256(holdout_ids),
            self.document["dataset"]["holdout_run_ids_canonical_sha256"],
        )

    def test_validation_resolution_and_holdout_fail_closed(self) -> None:
        data = load_generalization_split(ROOT, self.document, VALIDATION_SPLIT)
        self.assertEqual(len(data.runs), VALIDATION_COUNT)
        self.assertEqual(
            {run.split for run in data.runs.values()},
            {VALIDATION_SPLIT},
        )
        with self.assertRaisesRegex(RuntimeError, "explicit guard"):
            load_generalization_split(ROOT, self.document, HOLDOUT_SPLIT)

    def test_primary_and_precursor_semantics_are_frozen(self) -> None:
        self.assertEqual(self.document["experiment"]["id"], EXPERIMENT_ID)
        self.assertEqual(
            self.document["primary_evaluation"]["event_timing"]["slip_valid_window_ms"],
            [-30, 40],
        )
        secondary = self.document["secondary_evaluation"]
        self.assertEqual(
            (
                secondary["lower_m_inclusive"],
                secondary["upper_m_exclusive"],
            ),
            (0.03, 0.05),
        )
        self.assertEqual(secondary["future_followup_ms"], 1000)
        self.assertFalse(secondary["primary_scores_rewritten"])

    def test_run_level_json_and_frozen_hash_are_deterministic(self) -> None:
        sample = {
            "rows": [
                {"run_id": "b", "value": 2},
                {"run_id": "a", "value": 1},
            ]
        }
        encoded = _canonical_json(sample)
        self.assertEqual(encoded, _canonical_json(json.loads(encoded)))
        artifact_path = ROOT / str(self.document["artifacts"]["path"])
        freeze_path = artifact_path / "evaluation_freeze.json"
        if not freeze_path.is_file():
            return
        result = verify_generalization_development_evaluation(ROOT, CONFIG)
        self.assertTrue(result["passed"])
        freeze = load_json(freeze_path)
        self.assertEqual(
            sha256_file(artifact_path / "run_level_results.json"),
            freeze["artifact_sha256"]["run_level_results.json"],
        )
