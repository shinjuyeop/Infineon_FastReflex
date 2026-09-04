"""Contracts for final-candidate freeze and one-shot HOLDOUT readiness."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastreflex.dataset.loader import sha256_file
from fastreflex.evaluation.generalization import (
    HOLDOUT_SPLIT,
    _load_yaml,
    load_generalization_split,
)
from fastreflex.evaluation.readiness import (
    FINAL_CANDIDATE_ID,
    FINAL_CANDIDATE_PATH,
    FUTURE_EXPERIMENT_ID,
    authorize_one_shot_holdout,
    claim_one_shot_guard,
    load_readiness_yaml,
    run_final_candidate_holdout_readiness_review,
    verify_contracts,
    verify_final_candidate,
    verify_protected_datasets,
)
from tests.support import (
    REPOSITORY_ROOT as ROOT,
    assert_false_fields,
    imported_modules,
)

CONFIG = (
    ROOT
    / "configs/experiment/20260902_model_v2_final_candidate_holdout_readiness_review.yaml"
)
DEVELOPMENT_CONFIG = (
    ROOT
    / "configs/experiment/20260902_model_v2_generalization_development_evaluation.yaml"
)
EXPECTED_CONFIG_SHA256 = (
    "0206dd12078ffa191cc6424a35cdead889c9614a1ecb38a9a830a1d464368ec8"
)
EXPECTED_FINAL_CANDIDATE_SHA256 = (
    "52644d3efb0e756002bd43a7f94aea0a16a65eb2ff014c72548f2b3b138b7bc2"
)
EXPECTED_READINESS_SHA256 = (
    "0167d72942ee402b7bdcb83f5bcd3e69c62f4db8044c1bf62bfd8607487eb7c6"
)


class FinalCandidateReadinessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.review = load_readiness_yaml(CONFIG)

    def test_final_candidate_is_exact_alias_without_artifact_copy(self) -> None:
        verified = verify_final_candidate(ROOT, self.review)
        final = load_readiness_yaml(ROOT / FINAL_CANDIDATE_PATH)

        self.assertTrue(verified["passed"])
        self.assertEqual(verified["candidate_id"], FINAL_CANDIDATE_ID)
        self.assertEqual(
            verified["candidate_freeze_sha256"],
            "95dab53275aa77ded36b479c3633ab879a1331cb2c96bd0eb014de2be99bd85f",
        )
        self.assertEqual(
            verified["final_candidate_sha256"],
            EXPECTED_FINAL_CANDIDATE_SHA256,
        )
        self.assertFalse(verified["artifact_copy_or_mutation"])
        self.assertFalse(final["source_candidate"]["artifacts_duplicated"])
        self.assertTrue(final["no_new_weights"])
        self.assertTrue(final["no_checkpoint_write"])
        self.assertTrue(final["no_normalizer_write"])
        self.assertTrue(final["no_architecture_mutation"])

    def test_primary_and_secondary_contracts_are_immutable(self) -> None:
        verified = verify_contracts(ROOT, self.review)

        self.assertTrue(verified["passed"])
        self.assertTrue(verified["primary_unchanged"])
        self.assertTrue(verified["secondary_unchanged"])
        self.assertFalse(verified["secondary_can_rescue_primary"])
        self.assertEqual(
            verified["canonical_sha256"],
            {
                "primary": "feabfc4519e8ec28e59710810b6e587b7a8be1a128ecf57a028d32710c1b246e",
                "secondary": "085d6f73156a5618767284faa2ccdcd29d3645694f56155431159d533b77130a",
                "verdict_hierarchy": "e86fb11f457734c41cd7b9c66a827a22b587f7a1f95aa91130931f7586c8cba5",
                "post_open_rules": "4baabe28158f7319177a9d5501e2be31eabad110706e2e32811638dfa657175a",
            },
        )

    def test_generalization_holdout_loader_remains_fail_closed(self) -> None:
        development = _load_yaml(DEVELOPMENT_CONFIG)
        with self.assertRaisesRegex(RuntimeError, "explicit guard"):
            load_generalization_split(ROOT, development, HOLDOUT_SPLIT)

    def test_readiness_forbids_holdout_access(self) -> None:
        holdout = self.review["holdout"]
        sealed = self.review["sealed_evidence"]

        self.assertFalse(holdout["authorized_now"])
        self.assertTrue(holdout["no_access_this_milestone"])
        self.assertEqual(holdout["guard_before"], 0)
        assert_false_fields(
            sealed,
            "Generalization_HOLDOUT_waveform_opened",
            "Generalization_HOLDOUT_Hazard_inference",
            "Generalization_HOLDOUT_Terrain_inference",
            "Generalization_HOLDOUT_visualization",
        )
        self.assertEqual(sealed["Generalization_HOLDOUT_guard_count"], 0)

    def test_one_shot_authorization_rejects_wrong_candidate_or_guard(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "candidate is not approved"):
            authorize_one_shot_holdout(
                ROOT,
                CONFIG,
                candidate_id="unapproved_candidate",
                guard_count=0,
            )
        with self.assertRaisesRegex(RuntimeError, "guard count 0"):
            authorize_one_shot_holdout(
                ROOT,
                CONFIG,
                candidate_id=FINAL_CANDIDATE_ID,
                guard_count=1,
            )

    def test_one_shot_guard_can_be_claimed_exactly_once(self) -> None:
        authorization = authorize_one_shot_holdout(
            ROOT,
            CONFIG,
            candidate_id=FINAL_CANDIDATE_ID,
            guard_count=0,
        )
        self.assertEqual(authorization.experiment_id, FUTURE_EXPERIMENT_ID)
        self.assertTrue(authorization.v1_and_v2_same_pass)
        self.assertTrue(authorization.terrain_in_same_pass)
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "holdout_access_guard.json"
            claim = claim_one_shot_guard(ledger, authorization)
            self.assertEqual((claim["guard_before"], claim["guard_after"]), (0, 1))
            with self.assertRaises(FileExistsError):
                claim_one_shot_guard(ledger, authorization)

    def test_readiness_path_has_no_trainer_or_optimizer_import(self) -> None:
        source_path = ROOT / "src/fastreflex/evaluation/readiness.py"
        modules = imported_modules(source_path)
        self.assertFalse(
            any(
                module.startswith("fastreflex.training")
                or module.startswith("torch")
                or module.startswith("numpy")
                for module in modules
            )
        )

    def test_final_metadata_and_readiness_freeze_are_deterministic(self) -> None:
        first = run_final_candidate_holdout_readiness_review(ROOT, CONFIG)
        second = run_final_candidate_holdout_readiness_review(ROOT, CONFIG)

        self.assertEqual(first, second)
        self.assertEqual(sha256_file(CONFIG), EXPECTED_CONFIG_SHA256)
        self.assertEqual(
            sha256_file(ROOT / FINAL_CANDIDATE_PATH),
            EXPECTED_FINAL_CANDIDATE_SHA256,
        )
        self.assertEqual(
            first["holdout_readiness_review_sha256"],
            EXPECTED_READINESS_SHA256,
        )

    def test_all_protected_datasets_have_exact_integrity(self) -> None:
        verified = verify_protected_datasets(ROOT, self.review)

        self.assertTrue(verified["passed"])
        self.assertEqual(verified["generalization_validation_count"], 36)
        self.assertEqual(verified["generalization_holdout_count"], 36)
        self.assertFalse(verified["holdout_payload_deserialized"])
        self.assertEqual(
            {
                dataset_id: row["exact_files"]
                for dataset_id, row in verified["datasets"].items()
            },
            {
                "unified_hazard_reflex_20260829": 256,
                "model_v2_hazard_reflex_20260901": 412,
                "generalization_hazard_reflex_20260831": 72,
                "ice_near_hazard_semantics_20260901": 48,
            },
        )
