"""Regression tests for censor-aware Sand calibration and redesign."""

from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from fastreflex.dataset.generation import _load_yaml
from fastreflex.dataset.sand_calibration import (
    _calibration_result_summary,
    collect_sand_calibration_batch,
    expand_sand_benign_redesign,
    validate_sand_benign_redesign,
)


ROOT = Path(__file__).resolve().parents[1]
REDESIGN = (
    ROOT / "configs/experiment/20260902_sand_benign_generalization_study_redesign.yaml"
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

    def test_calibration_collector_has_no_model_inference(self) -> None:
        source = inspect.getsource(collect_sand_calibration_batch)
        for forbidden in (
            "predict_proba(",
            "load_model(",
            "torch.load(",
            "onnxruntime.InferenceSession",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
