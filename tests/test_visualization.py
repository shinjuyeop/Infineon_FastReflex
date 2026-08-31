"""Contracts for read-only supported-pipeline visualization."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from fastreflex.dataset.hazard import GROUPS, PELVIS_IMU6
from fastreflex.evaluation.hazard import reflex_required_trace
from fastreflex.simulation.g1 import run_simulation
from fastreflex.visualization import (
    ParityReport,
    compare_stored_runtime,
    format_viewer_overlay,
    prepare_visualization,
    reconstruct_simulation_config,
    representative_validation_runs,
    require_parity,
    resolve_visualization_run,
    visualization_run_ids,
)
from scripts.fastreflex import build_parser

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/raw/unified_hazard_reflex_20260829"
POLICY = ROOT / "artifacts/external/unitree_g1/g1_velocity_policy.onnx"
REPRESENTATIVES = {
    "ICE_SLIP_HAZARD": "uhr_ice_h_c20",
    "SAND_SUPPORT_HAZARD": "uhr_sand_h_c20",
    "SAND_BENIGN": "uhr_sand_b_c20",
    "HARD_GROUND_NORMAL": "uhr_hard_n_c20",
}


class _FakeViewer:
    def __init__(self) -> None:
        self.sync_count = 0
        self.texts: list[object] = []

    def __enter__(self) -> "_FakeViewer":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def is_running(self) -> bool:
        return True

    def sync(self, state_only: bool = False) -> None:
        self.sync_count += 1

    def set_texts(self, texts: object) -> None:
        self.texts.append(texts)


@unittest.skipUnless(DATASET.is_dir(), "local frozen dataset is absent")
class VisualizationResolutionTest(unittest.TestCase):
    def test_cli_and_authorized_run_resolution(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["visualize", "--run-id", "uhr_ice_h_c20", "--speed", "2.0"]
        )
        self.assertEqual(args.command, "visualize")
        self.assertEqual(args.speed, 2.0)
        self.assertFalse(args.show_debug)

        validation = resolve_visualization_run(ROOT, "uhr_ice_h_c20")
        train = resolve_visualization_run(ROOT, "uhr_ice_h_c01")
        self.assertEqual(validation.run.split, "validation")
        self.assertEqual(train.run.split, "train")
        self.assertEqual(validation.specification["id"], validation.run.run_id)

    def test_holdout_and_unknown_ids_fail_closed(self) -> None:
        with mock.patch("fastreflex.visualization.load_hazard_runs") as load_runs:
            with self.assertRaisesRegex(
                ValueError, "HOLDOUT visualization is prohibited"
            ):
                resolve_visualization_run(ROOT, "uhr_ice_h_c27")
            load_runs.assert_not_called()
        with self.assertRaisesRegex(ValueError, "unknown Unified Hazard run ID"):
            resolve_visualization_run(ROOT, "invented_demo_run")

    def test_representatives_use_neutral_lexicographic_rule(self) -> None:
        selected = representative_validation_runs(ROOT)
        self.assertEqual(selected, REPRESENTATIVES)
        self.assertEqual(tuple(selected), GROUPS)
        runs = visualization_run_ids(ROOT)
        for group, run_id in selected.items():
            self.assertEqual(run_id, min(runs[group]["validation"]))
            self.assertNotIn("holdout", runs[group])

    @unittest.skipUnless(POLICY.is_file(), "local verified policy is absent")
    def test_scenario_reconstruction_is_frozen_and_deterministic(self) -> None:
        resolved = resolve_visualization_run(ROOT, "uhr_sand_h_c20")
        first = reconstruct_simulation_config(resolved)
        second = reconstruct_simulation_config(resolved)
        self.assertEqual(first, second)
        self.assertEqual(first.terrain, "sand")
        self.assertEqual(first.sink_pattern, "transition_left")
        self.assertEqual(first.sink_support_pattern, "lateral_deformable")
        self.assertTrue(first.headless)

    def test_parity_failure_is_explicit(self) -> None:
        report = ParityReport({"timestamp_us": True, "pelvis_imu6": False}, 0.0)
        with self.assertRaisesRegex(RuntimeError, "viewer will not open: pelvis_imu6"):
            require_parity(report)


@unittest.skipUnless(
    DATASET.is_dir() and POLICY.is_file(),
    "local frozen simulation/model artifacts are absent",
)
class VisualizationIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = prepare_visualization(ROOT, "uhr_ice_h_c20")

    def test_exact_runtime_event_and_frozen_inference_parity(self) -> None:
        prepared = self.prepared
        self.assertTrue(prepared.parity.passed)
        self.assertEqual(prepared.parity.sensor_absolute_tolerance, 0.0)
        self.assertTrue(all(prepared.parity.checks.values()))
        np.testing.assert_array_equal(
            prepared.traces.reflex_required,
            reflex_required_trace(
                prepared.traces.hazard, len(prepared.resolved.run.timestamp_us)
            ),
        )
        self.assertEqual(
            prepared.traces.terrain.state.shape,
            prepared.traces.reflex_required.shape,
        )

    def test_gt_and_terrain_metadata_never_enter_hazard_tensor(self) -> None:
        run = self.prepared.resolved.run
        self.assertEqual(run.features[PELVIS_IMU6].shape[1], 6)
        self.assertEqual(
            set(run.features), {"PELVIS_IMU6", "PELVIS_IMU6_FSR8"}
        )
        original = self.prepared.traces.reflex_required.copy()
        altered_terrain = self.prepared.traces.terrain.state.copy()
        altered_terrain[:] = 4
        np.testing.assert_array_equal(original, self.prepared.traces.reflex_required)
        self.assertFalse(np.shares_memory(run.features[PELVIS_IMU6], altered_terrain))

    def test_overlay_separates_model_output_from_simulator_gt(self) -> None:
        model, diagnostics = format_viewer_overlay(
            self.prepared, 1914, show_debug=True
        )
        self.assertIn("MODEL OUTPUT", model)
        self.assertIn("Hazard probability", model)
        self.assertIn("Terrain state", model)
        self.assertIn("Cause refinement", model)
        self.assertIn("SIMULATOR GT / DIAGNOSTIC", diagnostics)
        self.assertIn("NEVER USED AS MODEL INPUT", diagnostics)
        self.assertIn("Tangential drift", diagnostics)
        self.assertIn("Support spread", diagnostics)

    def test_viewer_overlay_and_pacing_do_not_change_physics(self) -> None:
        fake_viewer = _FakeViewer()
        config = replace(self.prepared.simulation_config, headless=False)
        with mock.patch(
            "fastreflex.simulation.g1.launch_passive_viewer",
            return_value=fake_viewer,
        ), mock.patch("fastreflex.simulation.g1._pace_viewer"):
            viewer_result = run_simulation(
                config,
                viewer_overlay=lambda sample: format_viewer_overlay(
                    self.prepared, sample
                ),
                playback_speed=2.0,
            )
        self.assertGreater(fake_viewer.sync_count, 1)
        self.assertTrue(fake_viewer.texts)
        viewer_parity = compare_stored_runtime(
            self.prepared.resolved, viewer_result
        )
        self.assertTrue(viewer_parity.passed)
        for field in ("timestamp_us", "pelvis_imu", "foot_fsr"):
            np.testing.assert_equal(
                getattr(self.prepared.simulation.runtime, field),
                getattr(viewer_result.runtime, field),
            )


if __name__ == "__main__":
    unittest.main()
