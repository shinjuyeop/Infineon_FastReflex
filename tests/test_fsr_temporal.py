from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from fastreflex.evaluation.fsr_distribution import SinkRun
from fastreflex.evaluation.fsr_temporal import (
    _temporal_separation_rows,
    continuous_path_length,
    derive_temporal_trace,
    normalized_entropy,
    pre_event_path_rate,
    temporal_horizon_metrics,
)


class FsrTemporalAnalysisTest(unittest.TestCase):
    def _trace(self) -> tuple[np.ndarray, object]:
        fsr = np.asarray(
            [
                [2, 2, 2, 2, 4, 4, 4, 4],
                [4, 2, 1, 1, 4, 4, 4, 4],
                [0, 0, 0, 0, 4, 4, 4, 4],
                [1, 1, 2, 4, 4, 4, 4, 4],
                [8, 0, 0, 0, 4, 4, 4, 4],
            ],
            dtype=np.float32,
        )
        return fsr, derive_temporal_trace(fsr, "left")

    def test_normalized_shares_entropy_and_affected_side_canonicalization(self) -> None:
        fsr, left = self._trace()
        np.testing.assert_allclose(np.nansum(left.shares, axis=1)[[0, 1, 3, 4]], 1.0)
        self.assertTrue(np.isnan(left.shares[2]).all())
        self.assertAlmostEqual(normalized_entropy(left.shares)[0], 1.0)

        mirrored = np.column_stack((fsr[:, 4:], fsr[:, :4]))
        right = derive_temporal_trace(mirrored, "right")
        np.testing.assert_allclose(left.shares, right.shares, equal_nan=True)

    def test_l1_path_does_not_bridge_invalid_gap(self) -> None:
        _, trace = self._trace()
        path, pair_count, valid_count = continuous_path_length(
            trace.shares, 0, 3, "l1"
        )
        self.assertAlmostEqual(path, 0.5)
        self.assertEqual(pair_count, 1)
        self.assertEqual(valid_count, 3)

    def test_temporal_changes_paths_and_cop_are_causal(self) -> None:
        fsr, trace = self._trace()
        values, diagnostics = temporal_horizon_metrics(
            trace, 0, 3, 1, 1, -100, -20, 20
        )
        self.assertAlmostEqual(values["quadrant_l1_change"], 0.5)
        self.assertAlmostEqual(values["quadrant_path_length"], 0.5)
        self.assertAlmostEqual(values["concentration_path"], 0.09375)
        self.assertGreater(values["entropy_path"], 0.0)
        self.assertAlmostEqual(values["max_share_abs_change"], 0.25)
        self.assertAlmostEqual(values["cop_displacement"], np.sqrt(0.3125))
        self.assertAlmostEqual(values["cop_path_length"], np.sqrt(0.3125))
        self.assertAlmostEqual(values["medial_delta"], 0.125)
        self.assertEqual(diagnostics["endpoint_stop_sample_exclusive"], 4)

        changed_future = fsr.copy()
        changed_future[4, :4] = [0, 8, 0, 0]
        future_trace = derive_temporal_trace(changed_future, "left")
        future_values, _ = temporal_horizon_metrics(
            future_trace, 0, 3, 1, 1, -100, -20, 20
        )
        self.assertEqual(values, future_values)

    def test_event_initial_window_is_clipped_to_horizon(self) -> None:
        _, trace = self._trace()
        values, diagnostics = temporal_horizon_metrics(
            trace, 0, 0, 10, 10, -100, -20, 20
        )
        self.assertEqual(values["quadrant_l1_change"], 0.0)
        self.assertEqual(values["quadrant_path_length"], 0.0)
        self.assertIsNone(values["quadrant_path_rate"])
        self.assertEqual(diagnostics["initial_stop_sample_exclusive"], 1)
        self.assertEqual(diagnostics["endpoint_stop_sample_exclusive"], 1)

    def test_pre_event_baseline_uses_only_declared_samples(self) -> None:
        values = np.zeros(200, dtype=np.float64)
        values[:81] = np.arange(81) % 2
        values[81:] = np.arange(119) * 100.0
        rate, count, used = pre_event_path_rate(
            values, 100, -100, -20, 20, "absolute"
        )
        self.assertEqual((rate, count, used), (1.0, 81, (0, 81)))

    def test_run_level_left_right_parity(self) -> None:
        runs = [
            SinkRun("b_l", Path("b_l"), "BENIGN", "mild", "left", 0.15, 0.0),
            SinkRun("b_r", Path("b_r"), "BENIGN", "moderate", "right", 0.15, 0.5),
            SinkRun("h_l", Path("h_l"), "SINK", "severe", "left", 0.20, 0.0),
            SinkRun("h_r", Path("h_r"), "SINK", "severe", "right", 0.20, 0.5),
        ]
        rows = [
            {
                "run_id": run.run_id,
                "group": "BENIGN_SINK" if run.outcome == "BENIGN" else "HAZARDOUS_SINK",
                "alignment": "t1",
                "horizon_ms": 50,
                "metric": "quadrant_l1_change",
                "value": 1.0 if run.outcome == "BENIGN" else 3.0,
            }
            for run in runs
        ]
        result = _temporal_separation_rows(rows, runs)[0]
        self.assertEqual(result["oriented_auc"], 1.0)
        self.assertEqual(result["left_primary_direction_auc"], 1.0)
        self.assertEqual(result["right_primary_direction_auc"], 1.0)
        self.assertTrue(result["left_right_direction_consistent"])


if __name__ == "__main__":
    unittest.main()
