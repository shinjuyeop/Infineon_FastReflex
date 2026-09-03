from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import yaml

from fastreflex.dataset.hazard import canonical_sha256
from fastreflex.dataset.loader import sha256_file
from fastreflex.evaluation.sand import separability_metrics
from fastreflex.evaluation.sand_confirmation import (
    _load_confirmation_payload,
    _raw_scaler,
    reconstruct_discovery_scalers,
    separability_with_frozen_scaler,
    verify_confirmation_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/20260903_sand_benign_generalization_study_mild_recalibrated_confirmation_analysis.yaml"
)
ARTIFACTS = (
    ROOT
    / "artifacts/runs/20260903_sand_benign_generalization_study_mild_recalibrated_confirmation_analysis"
)


class SandConfirmationAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    def test_frozen_contract_and_inputs(self) -> None:
        document = self.document
        self.assertEqual(
            document["experiment"]["id"],
            "SAND_BENIGN_MILD_RECALIBRATED_CONFIRMATION_ANALYSIS",
        )
        self.assertTrue(document["experiment"]["protocol_frozen_before_confirmation_open"])
        self.assertEqual(
            document["discovery"]["frozen_hypothesis"],
            "DOMAIN_DIVERSITY_GAP_SUPPORTED",
        )
        self.assertEqual(document["model"]["threshold"], 0.99)
        self.assertEqual(document["model"]["persistence_ms"], 5)
        self.assertEqual(
            sha256_file(ROOT / document["implementation"]["path"]),
            document["implementation"]["sha256"],
        )
        verification = verify_confirmation_inputs(ROOT, CONFIG, document)
        self.assertTrue(verification["passed"])
        self.assertEqual(verification["old_holdout_guard"], 1)
        self.assertEqual(verification["confirmation_payload_reads_before_authorization"], 0)

    def test_discovery_scaler_values_reconstruct_to_frozen_hashes(self) -> None:
        result = reconstruct_discovery_scalers(ROOT, self.document)
        self.assertEqual(result["eligible_run_count"], 85)
        self.assertEqual(result["discovery_model_replay"], 0)
        self.assertEqual(result["confirmation_payload_reads"], 0)
        self.assertEqual(set(result["scalers"]), {"current", "window", "fsr", "combined", "oracle"})

    def test_fixed_scaler_metric_body_matches_discovery_fit(self) -> None:
        values = np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.2],
                [0.2, 0.1],
                [3.0, 3.0],
                [3.1, 3.2],
                [3.2, 3.1],
            ],
            dtype=np.float64,
        )
        labels = [0, 0, 0, 1, 1, 1]
        run_ids = list("abcdef")
        original = separability_metrics(values, labels, run_ids=run_ids)
        fixed = separability_with_frozen_scaler(
            values,
            labels,
            run_ids=run_ids,
            scaler=_raw_scaler(values),
        )
        for field in (
            "centroid_separation",
            "balanced_1nn_agreement",
            "balanced_5nn_agreement",
            "median_nearest_opposite_to_same_ratio",
            "local_opposite_class_mixing",
            "bidirectional_95pct_radius_inclusion",
            "distance_quantiles",
            "pca",
        ):
            self.assertEqual(fixed[field], original[field])

    def test_confirmation_loader_requires_claimed_guard(self) -> None:
        row = {"split": "MILD_RECALIBRATED_CONFIRMATION", "run_id": "sealed"}
        with self.assertRaisesRegex(RuntimeError, "guard has not been claimed"):
            _load_confirmation_payload(Path("unused"), row, {"open_count": 0})

    def test_frozen_result_hashes_after_consumption(self) -> None:
        interpretation_path = ARTIFACTS / "confirmation_interpretation.json"
        if not interpretation_path.exists():
            self.assertFalse((ARTIFACTS / "confirmation_access_guard.json").exists())
            return
        interpretation = json.loads(interpretation_path.read_text(encoding="utf-8"))
        files = {
            "discovery_scalers_sha256": "discovery_scalers.json",
            "pelvis_confirmation_analysis_sha256": "pelvis_confirmation_analysis.json",
            "fsr_contact_confirmation_analysis_sha256": "fsr_contact_confirmation_analysis.json",
            "privileged_oracle_confirmation_analysis_sha256": "privileged_oracle_confirmation_analysis.json",
            "v2_confirmation_replay_sha256": "v2_confirmation_replay.json",
            "confirmation_factor_localization_sha256": "confirmation_factor_localization.json",
            "discovery_confirmation_replication_sha256": "discovery_confirmation_replication.json",
            "confirmation_decision_sha256": "confirmation_decision.json",
            "confirmation_access_guard_sha256": "confirmation_access_guard.json",
        }
        for field, filename in files.items():
            self.assertEqual(sha256_file(ARTIFACTS / filename), interpretation[field])
        semantic = dict(interpretation)
        expected = semantic.pop("SAND_BENIGN_CONFIRMATION_INTERPRETATION_SHA")
        self.assertEqual(canonical_sha256(semantic), expected)
        self.assertEqual(
            sha256_file(CONFIG), interpretation["confirmation_analysis_config_sha256"]
        )
        self.assertEqual(
            interpretation["selected_confirmation_verdict"],
            "DOMAIN_DIVERSITY_GAP_NOT_CONFIRMED",
        )
        self.assertEqual(
            interpretation["analysis_validity"],
            "SAND_BENIGN_GENERALIZATION_STUDY_CONFIRMATION_ANALYSIS_VALID",
        )
        self.assertEqual(
            interpretation["future_use_status"],
            "CONSUMED_FOR_FROZEN_H1_REPLICATION",
        )
        self.assertEqual(interpretation["confirmation_open_count"], 1)
        self.assertEqual(interpretation["confirmation_payload_deserializations"], 88)
        self.assertEqual(interpretation["confirmation_v2_replay_count"], 1)
        self.assertEqual(interpretation["old_holdout_payload_reads"], 0)


if __name__ == "__main__":
    unittest.main()
