"""Contracts for read-only supported-pipeline visualization."""

from __future__ import annotations

import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import numpy as np

from fastreflex.dataset.hazard import GROUPS, PELVIS_IMU6
from fastreflex.evaluation.hazard import reflex_required_trace
from fastreflex.visualization import (
    ParityReport,
    PlaybackEvents,
    PlaybackState,
    SnapshotPlaybackControl,
    format_viewer_overlay,
    play_snapshot_trace,
    playback_events,
    prepare_visualization,
    reconstruct_simulation_config,
    representative_validation_runs,
    require_parity,
    resolve_visualization_run,
    visualization_run_ids,
    visualize_prepared_run,
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
    def __init__(self, running_checks: int = 3) -> None:
        self.sync_count = 0
        self.texts: list[object] = []
        self.running_checks = running_checks
        self.is_running_count = 0

    def __enter__(self) -> "_FakeViewer":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def is_running(self) -> bool:
        self.is_running_count += 1
        return self.is_running_count <= self.running_checks

    def sync(self, state_only: bool = False) -> None:
        self.sync_count += 1

    def lock(self) -> nullcontext[None]:
        return nullcontext()

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
        self.assertEqual(args.mode, "analysis")
        self.assertFalse(args.show_debug)
        controls = parser.parse_args(
            [
                "visualize",
                "--run-id",
                "uhr_ice_h_c20",
                "--pause-at",
                "1.5",
                "--pause-on-reflex",
                "--single-step",
                "--mode",
                "demo",
            ]
        )
        self.assertEqual(controls.pause_at, 1.5)
        self.assertTrue(controls.pause_on_reflex)
        self.assertTrue(controls.single_step)
        self.assertEqual(controls.mode, "demo")

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

    def test_playback_state_machine_seek_events_and_end_hold(self) -> None:
        events = PlaybackEvents(
            first_reflex=20,
            physical_hazard=30,
            i1_precursor=None,
            terrain_updates=(5, 15, 25),
        )
        control = SnapshotPlaybackControl(40, events=events)
        self.assertEqual(control.view().state, PlaybackState.PLAYING)

        control.key_callback(SnapshotPlaybackControl.SPACE_KEY)
        self.assertEqual(control.view().state, PlaybackState.PAUSED)
        control.key_callback(SnapshotPlaybackControl.RIGHT_ARROW_KEY)
        self.assertEqual(control.view().current_sample, 1)
        control.key_callback(SnapshotPlaybackControl.D_KEY)
        self.assertEqual(control.view().current_sample, 11)
        control.key_callback(SnapshotPlaybackControl.A_KEY)
        self.assertEqual(control.view().current_sample, 1)
        control.key_callback(SnapshotPlaybackControl.LEFT_ARROW_KEY)
        control.key_callback(SnapshotPlaybackControl.LEFT_ARROW_KEY)
        self.assertEqual(control.view().current_sample, 0)

        control.key_callback(SnapshotPlaybackControl.R_KEY)
        self.assertEqual(control.view().current_sample, 20)
        control.key_callback(SnapshotPlaybackControl.H_KEY)
        self.assertEqual(control.view().current_sample, 30)
        control.key_callback(SnapshotPlaybackControl.I_KEY)
        self.assertEqual(control.view().current_sample, 30)
        self.assertIn("not present", control.view().notice)
        control.key_callback(SnapshotPlaybackControl.HOME_KEY)
        control.key_callback(SnapshotPlaybackControl.T_KEY)
        self.assertEqual(control.view().current_sample, 5)
        control.key_callback(SnapshotPlaybackControl.T_KEY)
        self.assertEqual(control.view().current_sample, 15)
        control.key_callback(SnapshotPlaybackControl.G_KEY)
        self.assertEqual(control.view().current_sample, 5)

        control.key_callback(SnapshotPlaybackControl.END_KEY)
        self.assertEqual(control.view().current_sample, 39)
        self.assertEqual(control.view().state, PlaybackState.PAUSED)
        control.key_callback(SnapshotPlaybackControl.HOME_KEY)
        control.key_callback(SnapshotPlaybackControl.SPACE_KEY)
        control.key_callback(SnapshotPlaybackControl.D_KEY)
        self.assertEqual(control.view().current_sample, 10)
        self.assertEqual(control.view().state, PlaybackState.PAUSED)

        control.key_callback(SnapshotPlaybackControl.SPACE_KEY)
        ended = control.advance_by(100)
        self.assertEqual(ended.current_sample, 39)
        self.assertEqual(ended.state, PlaybackState.ENDED_PAUSED)
        self.assertTrue(ended.ended_reached)
        self.assertEqual(control.advance_by(100).state, PlaybackState.ENDED_PAUSED)
        control.key_callback(SnapshotPlaybackControl.SPACE_KEY)
        self.assertEqual(control.view().current_sample, 0)
        self.assertEqual(control.view().state, PlaybackState.PLAYING)

    def test_auto_pause_uses_exact_earliest_sample(self) -> None:
        control = SnapshotPlaybackControl(50, auto_pause_sample=12)
        paused = control.advance_by(40)
        self.assertEqual(paused.current_sample, 12)
        self.assertEqual(paused.state, PlaybackState.PAUSED)
        control.key_callback(SnapshotPlaybackControl.SPACE_KEY)
        self.assertEqual(control.advance_by(100).state, PlaybackState.ENDED_PAUSED)

        at_start = SnapshotPlaybackControl(50, auto_pause_sample=0)
        self.assertEqual(at_start.view().state, PlaybackState.PAUSED)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            SnapshotPlaybackControl(50, speed=0.0)


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
        render_trace = prepared.simulation.render_trace
        self.assertIsNotNone(render_trace)
        assert render_trace is not None
        self.assertEqual(
            render_trace.integration_state.shape[0],
            len(prepared.resolved.run.timestamp_us),
        )
        self.assertGreater(render_trace.integration_state.shape[1], 71)
        self.assertTrue(prepared.simulation.metadata["render_trace_captured"])
        with self.assertRaisesRegex(ValueError, "--pause-at must be in"):
            visualize_prepared_run(prepared, pause_at_s=8.0)

    def test_gt_and_terrain_metadata_never_enter_hazard_tensor(self) -> None:
        run = self.prepared.resolved.run
        self.assertEqual(run.features[PELVIS_IMU6].shape[1], 6)
        self.assertEqual(set(run.features), {"PELVIS_IMU6", "PELVIS_IMU6_FSR8"})
        original = self.prepared.traces.reflex_required.copy()
        altered_terrain = self.prepared.traces.terrain.state.copy()
        altered_terrain[:] = 4
        np.testing.assert_array_equal(original, self.prepared.traces.reflex_required)
        self.assertFalse(np.shares_memory(run.features[PELVIS_IMU6], altered_terrain))

    def test_overlay_separates_model_output_from_simulator_gt(self) -> None:
        initial_model, initial_diagnostics = format_viewer_overlay(self.prepared, -1)
        self.assertIn("Simulation time: 0.000 s", initial_model)
        self.assertIn("Hazard probability: N/A", initial_model)
        self.assertIn("Terrain state: UNKNOWN", initial_model)
        self.assertIn("Tangential drift: N/A", initial_diagnostics)

        first_reflex = self.prepared.traces.first_reflex_sample
        self.assertIsNotNone(first_reflex)
        assert first_reflex is not None
        model, diagnostics = format_viewer_overlay(
            self.prepared, first_reflex, show_debug=True
        )
        self.assertIn("MODEL OUTPUT", model)
        self.assertIn("Hazard probability", model)
        self.assertIn("Terrain state", model)
        self.assertIn("Current reflex (REFLEX_REQUIRED): TRUE", model)
        self.assertIn("Reflex occurred: TRUE (display history only)", model)
        self.assertIn("Current advisory cause", model)
        self.assertIn("Cause at first reflex", model)
        self.assertIn("SIMULATOR GT / DIAGNOSTIC", diagnostics)
        self.assertIn("NEVER USED AS MODEL INPUT", diagnostics)
        self.assertIn("Tangential drift", diagnostics)
        self.assertIn("Support spread", diagnostics)
        self.assertIn("EVENT TIMING", diagnostics)
        self.assertIn("delta = detector - reference", diagnostics)
        self.assertIn("Reflex -> Slip: -24 ms", diagnostics)
        self.assertIn("Reflex -> Terrain: -456 ms", diagnostics)
        self.assertIn("R=Reflex", diagnostics)

        later_false = np.flatnonzero(
            (~self.prepared.traces.reflex_required)
            & (np.arange(len(self.prepared.traces.reflex_required)) > first_reflex)
            & (
                np.arange(len(self.prepared.traces.reflex_required))
                < self.prepared.resolved.run.censor_sample
            )
        )
        self.assertTrue(later_false.size)
        history_model, _ = format_viewer_overlay(self.prepared, int(later_false[0]))
        self.assertIn("Current reflex (REFLEX_REQUIRED): FALSE", history_model)
        self.assertIn("Reflex occurred: TRUE", history_model)

        censor = self.prepared.resolved.run.censor_sample
        censored_model, censored_diagnostics = format_viewer_overlay(
            self.prepared, min(censor, len(self.prepared.resolved.run.timestamp_us) - 1)
        )
        self.assertIn("Current reflex (REFLEX_REQUIRED): CENSORED", censored_model)
        self.assertIn("Reflex occurred: TRUE", censored_model)
        self.assertNotIn("First reflex: not reached", censored_model)
        self.assertIn("Hazard probability: N/A", censored_model)
        self.assertIn("Slip event:", censored_diagnostics)

        demo_model, demo_diagnostics = format_viewer_overlay(
            self.prepared,
            first_reflex,
            mode="demo",
        )
        self.assertIn("MODEL OUTPUT / DEMO", demo_model)
        self.assertIn("Physical reference: SLIP", demo_diagnostics)
        self.assertIn("Reflex -> Slip", demo_diagnostics)

    def test_event_destinations_are_exact_and_missing_i1_is_safe(self) -> None:
        events = playback_events(self.prepared)
        self.assertEqual(events.first_reflex, self.prepared.traces.first_reflex_sample)
        self.assertEqual(
            events.physical_hazard,
            min(
                value
                for value in self.prepared.resolved.run.slip_event_samples_per_foot
                if value is not None
            ),
        )
        self.assertIsNone(events.i1_precursor)
        self.assertEqual(
            events.terrain_updates,
            tuple(sorted(set(self.prepared.traces.terrain.update_samples.tolist()))),
        )

    def test_snapshot_playback_never_steps_or_changes_frozen_traces(self) -> None:
        playback = SnapshotPlaybackControl(
            len(self.prepared.resolved.run.timestamp_us),
            events=playback_events(self.prepared),
            start_paused=True,
        )
        fake_viewer = _FakeViewer()
        runtime_before = self.prepared.simulation.runtime.pelvis_imu.copy()
        reflex_before = self.prepared.traces.reflex_required.copy()
        with (
            mock.patch(
                "fastreflex.visualization.launch_passive_viewer",
                return_value=fake_viewer,
            ),
            mock.patch("fastreflex.visualization.time.sleep"),
            mock.patch(
                "fastreflex.visualization.mujoco.mj_step",
                side_effect=AssertionError("snapshot playback must not call mj_step"),
            ),
        ):
            final = play_snapshot_trace(
                self.prepared,
                playback,
                mode="analysis",
            )
        self.assertGreaterEqual(fake_viewer.sync_count, 1)
        self.assertTrue(fake_viewer.texts)
        self.assertEqual(final.state, PlaybackState.PAUSED)
        np.testing.assert_array_equal(
            self.prepared.simulation.runtime.pelvis_imu, runtime_before
        )
        np.testing.assert_array_equal(
            self.prepared.traces.reflex_required, reflex_before
        )

    def test_ended_paused_remains_open_until_viewer_closes(self) -> None:
        playback = SnapshotPlaybackControl(len(self.prepared.resolved.run.timestamp_us))
        ended = playback.advance_by(playback.total_samples)
        self.assertEqual(ended.state, PlaybackState.ENDED_PAUSED)
        fake_viewer = _FakeViewer(running_checks=5)
        with (
            mock.patch(
                "fastreflex.visualization.launch_passive_viewer",
                return_value=fake_viewer,
            ),
            mock.patch("fastreflex.visualization.time.sleep"),
        ):
            final = play_snapshot_trace(self.prepared, playback)
        self.assertEqual(final.state, PlaybackState.ENDED_PAUSED)
        self.assertEqual(final.current_sample, playback.total_samples - 1)
        self.assertEqual(fake_viewer.is_running_count, 6)
        self.assertIn("PLAYBACK: ENDED / PAUSED", str(fake_viewer.texts[-1]))

    def test_early_viewer_close_is_clean_after_headless_parity(self) -> None:
        fake_viewer = _FakeViewer(running_checks=1)
        first_reflex = self.prepared.traces.first_reflex_sample
        assert first_reflex is not None
        later_pause_s = float(
            self.prepared.resolved.run.timestamp_us[first_reflex + 100] / 1_000_000.0
        ) + 0.0004
        with (
            mock.patch(
                "fastreflex.visualization.launch_passive_viewer",
                return_value=fake_viewer,
            ),
            mock.patch("fastreflex.visualization.time.sleep"),
            mock.patch(
                "fastreflex.visualization.mujoco.mj_step",
                side_effect=AssertionError("viewer must not execute physics"),
            ),
        ):
            result = visualize_prepared_run(
                self.prepared,
                pause_at_s=later_pause_s,
                pause_on_reflex=True,
            )
        self.assertTrue(result["viewer_closed_cleanly"])
        self.assertTrue(result["scientific_parity_prechecked"])
        self.assertFalse(result["viewer_physics_executed"])
        self.assertIsNone(result["viewer_physics_parity"])
        self.assertEqual(result["pause_at_sample"], first_reflex + 100)
        self.assertEqual(result["auto_pause_sample"], first_reflex)
        self.assertFalse(result["holdout_opened"])


if __name__ == "__main__":
    unittest.main()
