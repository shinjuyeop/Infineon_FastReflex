"""Contracts for model-blind Hazard dataset generation and annotations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fastreflex.dataset.generation import (
    PRECURSOR_OUTCOME_CODES,
    _load_yaml,
    _outcome_summary,
    _write_deterministic_npz,
    annotate_ice_precursors,
    expand_model_v2_design,
    i1_trace_from_diagnostics,
    i1_union_trace_from_diagnostics,
    validate_model_v2_design,
)
from fastreflex.dataset.loader import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "configs/experiment/20260901_model_v2_dataset_design.yaml"


def _precursor_fixture(
    outcome: str,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    samples = 1400
    exact = np.zeros((samples, 2), dtype=bool)
    loaded = np.zeros((samples, 2), dtype=bool)
    episode = np.full((samples, 2), -1, dtype=np.int32)
    drift = np.zeros((samples, 2), dtype=np.float64)
    velocity = np.zeros((samples, 2), dtype=np.float64)
    slip = np.zeros((samples, 2), dtype=bool)
    onset = np.zeros((samples, 2), dtype=bool)
    exact[10:100, 0] = True
    loaded[10:100, 0] = True
    episode[10:100, 0] = 0
    drift[20:100, 0] = 0.035
    velocity[20:100, 0] = 0.4
    censor = samples
    if outcome == "SAME_EPISODE_SLIP":
        onset[80, 0] = True
        slip[80:100, 0] = True
    elif outcome == "NEXT_EPISODE_SLIP":
        exact[150:250, 1] = True
        loaded[150:250, 1] = True
        episode[150:250, 1] = 0
        onset[200, 1] = True
        slip[200:250, 1] = True
    elif outcome == "LATER_SLIP":
        onset[300, 1] = True
        slip[300:320, 1] = True
    elif outcome == "CENSORED":
        censor = 500
    elif outcome != "BENIGN_RELEASE":
        raise ValueError(outcome)
    return annotate_ice_precursors(
        exact_ice_contact=exact,
        loaded_contact=loaded,
        contact_episode_id=episode,
        drift_m=drift,
        velocity_mps=velocity,
        established_slip=slip,
        established_slip_onset=onset,
        censor_sample=censor,
    )


class GenerationTest(unittest.TestCase):
    def test_frozen_model_v2_matrix_resolves_and_excludes_history(self) -> None:
        document = _load_yaml(DESIGN)
        specifications = expand_model_v2_design(document)
        audit = validate_model_v2_design(ROOT, document, specifications)
        self.assertEqual(len(specifications), 412)
        self.assertEqual(
            audit["split_counts"], {"V2_TRAIN": 310, "V2_VALIDATION": 102}
        )
        self.assertEqual(audit["unique_run_ids"], 412)
        self.assertEqual(audit["unique_physical_signatures"], 412)
        self.assertEqual(audit["internal_duplicate_signatures"], 0)
        self.assertEqual(audit["train_validation_overlap"], 0)
        self.assertEqual(audit["cross_split_near_duplicates"], 0)
        self.assertEqual(set(audit["overlap_by_reference"].values()), {0})
        self.assertEqual(
            audit["matrix_sha256"],
            "6d109808ac20c52bc913901dd61e2eaf1541c1b8d0163e81497b63964c239bd8",
        )

    def test_frozen_ice_precursor_future_outcomes(self) -> None:
        for outcome in (
            "SAME_EPISODE_SLIP",
            "NEXT_EPISODE_SLIP",
            "LATER_SLIP",
            "BENIGN_RELEASE",
            "CENSORED",
        ):
            with self.subTest(outcome=outcome):
                episodes, candidate, codes, censored = _precursor_fixture(outcome)
                self.assertEqual(len(episodes), 1)
                self.assertEqual(episodes[0]["future_outcome"], outcome)
                self.assertTrue(candidate[20, 0])
                self.assertEqual(codes[20, 0], PRECURSOR_OUTCOME_CODES[outcome])
                self.assertEqual(bool(censored[20, 0]), outcome == "CENSORED")

    def test_i1_persistence_resets_at_contact_episode_boundary(self) -> None:
        spread = np.zeros((80, 2), dtype=np.float64)
        spread[1:20, 0] = np.arange(1, 20) * 0.0001
        spread[20:50, 0] = 0.003 + np.arange(30) * 0.0001
        loaded = np.zeros((80, 2), dtype=bool)
        loaded[:50, 0] = True
        episodes = np.full((80, 2), -1, dtype=np.int32)
        episodes[:20, 0] = 0
        episodes[20:50, 0] = 1
        active = i1_trace_from_diagnostics(spread, loaded, episodes, 80)
        self.assertFalse(active[:40, 0].any())
        self.assertTrue(active[40, 0])
        self.assertFalse(active[:, 1].any())

        union = i1_union_trace_from_diagnostics(spread, loaded, 0, 80)
        self.assertFalse(union[:20].any())
        self.assertTrue(union[20])

    def test_npz_serialization_is_byte_reproducible(self) -> None:
        arrays = {
            "timestamp_us": np.arange(1, 11, dtype=np.int64) * 1000,
            "pelvis_imu6": np.zeros((10, 6), dtype=np.float32),
        }
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.npz"
            second = Path(temporary) / "second.npz"
            _write_deterministic_npz(first, arrays)
            _write_deterministic_npz(second, arrays)
            self.assertEqual(sha256_file(first), sha256_file(second))
            with np.load(first, allow_pickle=False) as payload:
                np.testing.assert_array_equal(
                    payload["timestamp_us"], arrays["timestamp_us"]
                )

    def test_run_balance_separates_confirmed_and_ambiguous_no_hazard(self) -> None:
        base = {
            "valid": True,
            "actual_hazard_label": "NO_HAZARD",
            "actual_subtype": "NONE",
            "intent_match": True,
            "intent_mismatch": False,
            "ice_precursor_censored": False,
            "i1_summary": {"first_sample": None},
        }
        confirmed = dict(base)
        i1_only = {**base, "i1_summary": {"first_sample": 100}}
        censored = {**base, "ice_precursor_censored": True}
        summary = _outcome_summary((confirmed, i1_only, censored))
        self.assertEqual(summary["no_hazard"], 3)
        self.assertEqual(summary["confirmed_no_hazard"], 1)
        self.assertEqual(summary["i1_only"], 1)
        self.assertEqual(summary["ambiguous_or_censored"], 2)


if __name__ == "__main__":
    unittest.main()
