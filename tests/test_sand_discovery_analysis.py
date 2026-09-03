from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

from fastreflex.dataset.loader import sha256_file
from fastreflex.dataset.sand_mild_calibration import (
    load_mild_recalibrated_discovery_payload,
)
from fastreflex.evaluation.sand import (
    benign_anchor,
    factor_localization,
    fsr_contact_vector,
    privileged_oracle_vector,
    separability_metrics,
    verify_analysis_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/20260903_sand_benign_generalization_study_mild_recalibrated_discovery_analysis.yaml"
)
DATASET = (
    ROOT
    / "data/raw/sand_benign_generalization_mild_recalibrated_study_20260903"
)
ARTIFACTS = (
    ROOT
    / "artifacts/runs/20260903_sand_benign_generalization_study_mild_recalibrated_discovery_analysis"
)


class SandDiscoveryAnalysisTest(unittest.TestCase):
    def test_analysis_contract_and_inputs_are_frozen(self) -> None:
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(
            document["experiment"]["id"],
            "SAND_BENIGN_MILD_RECALIBRATED_DISCOVERY_ANALYSIS",
        )
        self.assertTrue(document["experiment"]["protocol_frozen_before_v2_discovery_replay"])
        self.assertEqual(document["model"]["threshold"], 0.99)
        self.assertEqual(document["model"]["persistence_ms"], 5)
        self.assertEqual(document["model"]["ensemble_seeds"], [20260828, 20260829, 20260830])
        self.assertEqual(
            sha256_file(ROOT / document["implementation"]["path"]),
            document["implementation"]["sha256"],
        )
        verification = verify_analysis_inputs(ROOT, CONFIG, document)
        self.assertTrue(verification["passed"])
        self.assertEqual(verification["old_holdout_guard"], 1)
        self.assertEqual(verification["confirmation_payload_deserializations"], 0)

    def test_confirmation_is_rejected_before_npz_open(self) -> None:
        manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
        run_id = next(
            row["run_id"]
            for row in manifest["runs"]
            if row["split"] == "MILD_RECALIBRATED_CONFIRMATION"
        )
        with mock.patch("numpy.load", side_effect=AssertionError("NPZ opened")):
            with self.assertRaisesRegex(RuntimeError, "SEALED"):
                load_mild_recalibrated_discovery_payload(DATASET, run_id)

    def test_anchor_and_sensor_vectors_are_causal_and_exact_shape(self) -> None:
        samples = 120
        imu = np.zeros((samples, 6), dtype=np.float32)
        imu[50:70, 0] = 1.0
        imu[70:90, 0] = 3.0
        target = np.zeros((samples, 2), dtype=bool)
        target[50:70, 0] = True
        payload = {
            "pelvis_imu6": imu,
            "target_terrain_contact": target,
            "censor_sample": np.asarray(samples),
            "foot_fsr8": np.ones((samples, 8), dtype=np.float32),
            "support_surface_spread_m": np.zeros((samples, 2), dtype=np.float32),
            "support_surface_max_displacement_m": np.zeros(
                (samples, 2), dtype=np.float32
            ),
            "loaded_contact": np.ones((samples, 2), dtype=bool),
        }
        anchor = benign_anchor(payload, baseline_ms=20)
        self.assertEqual(anchor, 89)
        self.assertEqual(fsr_contact_vector(payload, anchor).shape, (39,))
        self.assertEqual(privileged_oracle_vector(payload, anchor).shape, (16,))

    def test_separability_and_localization_are_deterministic(self) -> None:
        values = np.asarray([[0.0], [0.1], [0.2], [3.0], [3.1], [3.2]])
        labels = [0, 0, 0, 1, 1, 1]
        first = separability_metrics(values, labels, run_ids=list("abcdef"))
        second = separability_metrics(values, labels, run_ids=list("abcdef"))
        self.assertEqual(first, second)
        self.assertEqual(first["balanced_1nn_agreement"], 1.0)
        self.assertGreater(first["centroid_separation"], 10.0)

        rows = []
        for index in range(32):
            adverse = index < 8 or 16 <= index < 18
            rows.append(
                {
                    "reflex": adverse,
                    "adverse_margin": adverse,
                    "factors": {
                        "source": "CONCRETE" if index % 2 == 0 else "MARBLE",
                        "speed": "0.20" if index % 3 else "0.25",
                        "severity": "A" if index < 16 else "B",
                    },
                }
            )
        localization = factor_localization(
            rows,
            factors=["severity"],
            minimum_level_n=8,
            fraction_range_min=0.25,
            cramers_v_min=0.20,
        )
        self.assertTrue(localization["metadata_localization"])

    def test_frozen_interpretation_hashes_when_analysis_exists(self) -> None:
        interpretation_path = ARTIFACTS / "discovery_interpretation.json"
        if not interpretation_path.exists():
            return
        interpretation = json.loads(interpretation_path.read_text(encoding="utf-8"))
        files = {
            "pelvis_analysis_sha256": "pelvis_analysis.json",
            "fsr_contact_analysis_sha256": "fsr_contact_analysis.json",
            "privileged_oracle_analysis_sha256": "privileged_oracle_analysis.json",
            "v2_discovery_replay_sha256": "v2_discovery_replay.json",
            "factor_localization_sha256": "factor_localization.json",
            "discovery_hypothesis_decision_sha256": "discovery_hypothesis_decision.json",
        }
        for field, filename in files.items():
            self.assertEqual(sha256_file(ARTIFACTS / filename), interpretation[field])
        semantic = dict(interpretation)
        expected = semantic.pop("SAND_BENIGN_DISCOVERY_INTERPRETATION_SHA")
        from fastreflex.dataset.hazard import canonical_sha256

        self.assertEqual(canonical_sha256(semantic), expected)
        self.assertEqual(
            interpretation["confirmation_status"],
            "SEALED_FOR_MILD_RECALIBRATED_CONFIRMATION",
        )
        self.assertEqual(interpretation["confirmation_payload_deserializations"], 0)


if __name__ == "__main__":
    unittest.main()
