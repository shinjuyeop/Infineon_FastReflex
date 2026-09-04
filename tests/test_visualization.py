"""Contracts for read-only supported-pipeline visualization."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from fastreflex.dataset.hazard import GROUPS, PELVIS_IMU6
from fastreflex.evaluation.hazard import reflex_required_trace
from fastreflex.visualization import (
    ParityReport,
    PlaybackEvents,
    PlaybackState,
    RECORDING_HEIGHT,
    RECORDING_WIDTH,
    SnapshotPlaybackControl,
    build_qualitative_recording_plan,
    build_recording_plan,
    compose_recording_frame,
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
from tests.support import (
    REPOSITORY_ROOT as ROOT,
    ViewerStub,
    assert_contains,
    assert_false_fields,
)

DATASET = ROOT / "data/raw/unified_hazard_reflex_20260829"
POLICY = ROOT / "artifacts/external/unitree_g1/g1_velocity_policy.onnx"
REPRESENTATIVES = {
    "ICE_SLIP_HAZARD": "uhr_ice_h_c20",
    "SAND_SUPPORT_HAZARD": "uhr_sand_h_c20",
    "SAND_BENIGN": "uhr_sand_b_c20",
    "HARD_GROUND_NORMAL": "uhr_hard_n_c20",
}


def _press(
    control: SnapshotPlaybackControl,
    keycode: int,
    *,
    sample: int | None = None,
    state: PlaybackState | None = None,
) -> None:
    control.key_callback(keycode)
    view = control.view()
    if sample is not None:
        assert view.current_sample == sample
    if state is not None:
        assert view.state is state


class QualitativeRecordingContractTest(unittest.TestCase):
    def test_plan_stops_at_fall_or_after_simulator_risk(self) -> None:
        config = mock.Mock(sensor_rate_hz=1000)
        result = mock.Mock()
        result.runtime.timestamp_us = np.arange(2000, dtype=np.int64) * 1000
        result.diagnostics.deformable_sink_onset = np.zeros(
            (2000, 2), dtype=bool
        )
        result.diagnostics.deformable_sink_onset[100, 0] = True
        result.diagnostics.sink_degradation_active = np.zeros(2000, dtype=bool)
        result.diagnostics.sink_degradation_active[250:500] = True
        result.metadata = {"first_fall_sample": 800}

        fall_plan = build_qualitative_recording_plan(config, result)
        self.assertEqual(fall_plan.end_reason, "at_first_fall")
        self.assertEqual(fall_plan.end_sample, 800)
        self.assertEqual(fall_plan.hold_frame_count, 60)
        self.assertEqual(fall_plan.sample_indices[-1], 800)

        result.metadata = {"first_fall_sample": None}
        risk_plan = build_qualitative_recording_plan(config, result)
        self.assertEqual(risk_plan.end_reason, "post_simulator_risk")
        self.assertEqual(risk_plan.end_sample, 1000)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            build_qualitative_recording_plan(config, result, playback_speed=0.0)


@unittest.skipUnless(DATASET.is_dir(), "local frozen dataset is absent")
class VisualizationResolutionTest(unittest.TestCase):
    def test_cli_and_authorized_run_resolution(self) -> None:
        parser = build_parser()
        cases = (
            (
                ("--speed", "2.0"),
                {
                    "command": "visualize",
                    "speed": 2.0,
                    "mode": "analysis",
                    "show_debug": False,
                },
            ),
            (
                (
                    "--pause-at",
                    "1.5",
                    "--pause-on-reflex",
                    "--single-step",
                    "--mode",
                    "demo",
                ),
                {
                    "pause_at": 1.5,
                    "pause_on_reflex": True,
                    "single_step": True,
                    "mode": "demo",
                },
            ),
            (
                (
                    "--speed",
                    "0.5",
                    "--mode",
                    "demo",
                    "--record",
                    "simulation/outputs/ice_hazard_demo.mp4",
                    "--stop-before-fall",
                ),
                {
                    "speed": 0.5,
                    "record": Path("simulation/outputs/ice_hazard_demo.mp4"),
                    "stop_before_fall": True,
                },
            ),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                parsed = parser.parse_args(
                    ["visualize", "--run-id", "uhr_ice_h_c20", *arguments]
                )
                for name, value in expected.items():
                    self.assertEqual(getattr(parsed, name), value)

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

        _press(control, SnapshotPlaybackControl.SPACE_KEY, state=PlaybackState.PAUSED)
        _press(control, SnapshotPlaybackControl.RIGHT_ARROW_KEY, sample=1)
        _press(control, SnapshotPlaybackControl.D_KEY, sample=11)
        _press(control, SnapshotPlaybackControl.A_KEY, sample=1)
        _press(control, SnapshotPlaybackControl.LEFT_ARROW_KEY)
        _press(control, SnapshotPlaybackControl.LEFT_ARROW_KEY, sample=0)

        _press(control, SnapshotPlaybackControl.R_KEY, sample=20)
        _press(control, SnapshotPlaybackControl.H_KEY, sample=30)
        _press(control, SnapshotPlaybackControl.I_KEY, sample=30)
        self.assertIn("not present", control.view().notice)
        _press(control, SnapshotPlaybackControl.HOME_KEY)
        _press(control, SnapshotPlaybackControl.T_KEY, sample=5)
        _press(control, SnapshotPlaybackControl.T_KEY, sample=15)
        _press(control, SnapshotPlaybackControl.G_KEY, sample=5)

        _press(
            control,
            SnapshotPlaybackControl.END_KEY,
            sample=39,
            state=PlaybackState.ENDED_PAUSED,
        )
        self.assertTrue(control.view().ended_reached)
        _press(control, SnapshotPlaybackControl.HOME_KEY)
        _press(control, SnapshotPlaybackControl.SPACE_KEY)
        _press(
            control,
            SnapshotPlaybackControl.D_KEY,
            sample=10,
            state=PlaybackState.PAUSED,
        )

        control.key_callback(SnapshotPlaybackControl.SPACE_KEY)
        ended = control.advance_by(100)
        self.assertEqual(ended.current_sample, 39)
        self.assertEqual(ended.state, PlaybackState.ENDED_PAUSED)
        self.assertTrue(ended.ended_reached)
        self.assertEqual(control.advance_by(100).state, PlaybackState.ENDED_PAUSED)
        _press(
            control,
            SnapshotPlaybackControl.SPACE_KEY,
            sample=0,
            state=PlaybackState.PLAYING,
        )

    def test_missing_event_jump_is_a_safe_pause(self) -> None:
        control = SnapshotPlaybackControl(40)
        control.advance_by(7)
        before = control.view().current_sample

        for keycode, name in (
            (SnapshotPlaybackControl.R_KEY, "first Reflex"),
            (SnapshotPlaybackControl.H_KEY, "physical Hazard"),
            (SnapshotPlaybackControl.I_KEY, "I1 precursor"),
            (SnapshotPlaybackControl.T_KEY, "next Terrain update"),
            (SnapshotPlaybackControl.G_KEY, "previous Terrain update"),
        ):
            if control.view().state is not PlaybackState.PLAYING:
                control.key_callback(SnapshotPlaybackControl.SPACE_KEY)
            self.assertEqual(control.view().state, PlaybackState.PLAYING)
            control.key_callback(keycode)
            view = control.view()
            self.assertEqual(view.current_sample, before)
            self.assertEqual(view.state, PlaybackState.PAUSED)
            self.assertIn(name, view.notice)

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
        cls.support_prepared = prepare_visualization(ROOT, "uhr_sand_h_c20")

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
        assert_contains(
            initial_model,
            "Simulation time: 0.000 s",
            "Hazard probability: N/A",
            "Terrain state: UNKNOWN",
        )
        assert_contains(initial_diagnostics, "Tangential drift: N/A")

        first_reflex = self.prepared.traces.first_reflex_sample
        self.assertIsNotNone(first_reflex)
        assert first_reflex is not None
        model, diagnostics = format_viewer_overlay(
            self.prepared, first_reflex, show_debug=True
        )
        assert_contains(
            model,
            "MODEL OUTPUT",
            "Hazard probability",
            "Terrain state",
            "Current reflex (REFLEX_REQUIRED): TRUE",
            "Reflex occurred: TRUE (display history only)",
            "Current advisory cause",
            "Cause at first reflex",
        )
        assert_contains(
            diagnostics,
            "SIMULATOR GT / DIAGNOSTIC",
            "NEVER USED AS MODEL INPUT",
            "Tangential drift",
            "Support spread",
            "EVENT TIMING",
            "delta = detector - reference",
            "Reflex -> Slip: -24 ms",
            "Reflex -> Terrain: -456 ms",
            "R=Reflex",
        )

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
        assert_contains(
            history_model,
            "Current reflex (REFLEX_REQUIRED): FALSE",
            "Reflex occurred: TRUE",
        )

        censor = self.prepared.resolved.run.censor_sample
        censored_model, censored_diagnostics = format_viewer_overlay(
            self.prepared, min(censor, len(self.prepared.resolved.run.timestamp_us) - 1)
        )
        assert_contains(
            censored_model,
            "Current reflex (REFLEX_REQUIRED): CENSORED",
            "Reflex occurred: TRUE",
            "Hazard probability: N/A",
        )
        self.assertNotIn("First reflex: not reached", censored_model)
        assert_contains(censored_diagnostics, "Slip event:")

        demo_model, demo_diagnostics = format_viewer_overlay(
            self.prepared,
            first_reflex,
            mode="demo",
        )
        assert_contains(demo_model, "MODEL OUTPUT / DEMO")
        assert_contains(demo_diagnostics, "Physical reference: SLIP", "Reflex -> Slip")

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

    def test_support_and_i1_destinations_are_exact(self) -> None:
        prepared = self.support_prepared
        events = playback_events(prepared)
        self.assertEqual(
            events.physical_hazard,
            min(
                value
                for value in prepared.resolved.run.support_event_samples_per_foot
                if value is not None
            ),
        )
        self.assertEqual(events.i1_precursor, prepared.traces.i1_sample)
        self.assertIsNotNone(events.i1_precursor)

        playback = SnapshotPlaybackControl(
            len(prepared.resolved.run.timestamp_us),
            events=events,
        )
        playback.key_callback(SnapshotPlaybackControl.I_KEY)
        self.assertEqual(playback.view().current_sample, events.i1_precursor)
        self.assertEqual(playback.view().state, PlaybackState.PAUSED)
        playback.key_callback(SnapshotPlaybackControl.H_KEY)
        self.assertEqual(playback.view().current_sample, events.physical_hazard)
        self.assertEqual(playback.view().state, PlaybackState.PAUSED)

    def test_recording_plan_and_demo_alert_are_deterministic(self) -> None:
        prepared = self.prepared
        plan = build_recording_plan(prepared, playback_speed=0.5)
        self.assertEqual(plan.end_reason, "post_first_fall")
        self.assertEqual(
            plan.end_sample,
            prepared.simulation.metadata["first_fall_sample"] + 750,
        )
        self.assertEqual(plan.sample_indices[0], 0)
        self.assertEqual(plan.sample_indices[-1], plan.end_sample)
        self.assertEqual(plan.hold_frame_count, 60)
        self.assertEqual(
            len(plan.sample_indices),
            plan.playback_frame_count + plan.hold_frame_count,
        )
        pre_fall = build_recording_plan(
            prepared,
            playback_speed=0.5,
            stop_before_fall=True,
        )
        self.assertEqual(pre_fall.end_reason, "pre_first_fall")
        self.assertEqual(
            pre_fall.end_sample,
            prepared.simulation.metadata["first_fall_sample"] - 1,
        )

        frame = np.zeros((RECORDING_HEIGHT, RECORDING_WIDTH, 3), dtype=np.uint8)
        first_reflex = prepared.traces.first_reflex_sample
        assert first_reflex is not None
        safe = compose_recording_frame(
            frame,
            prepared,
            first_reflex - 1,
            playback_speed=0.5,
        )
        danger = compose_recording_frame(
            frame,
            prepared,
            first_reflex,
            playback_speed=0.5,
        )
        self.assertEqual(safe.shape, frame.shape)
        self.assertEqual(danger.shape, frame.shape)
        self.assertGreater(int(danger[0, 0, 0]), int(safe[0, 0, 0]))

        with self.assertRaisesRegex(ValueError, "positive and finite"):
            build_recording_plan(prepared, playback_speed=0.0)

    def test_snapshot_playback_never_steps_or_changes_frozen_traces(self) -> None:
        playback = SnapshotPlaybackControl(
            len(self.prepared.resolved.run.timestamp_us),
            events=playback_events(self.prepared),
            start_paused=True,
        )
        fake_viewer = ViewerStub()
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
        rendered_panels = fake_viewer.texts[-1]
        self.assertEqual(len(rendered_panels), 4)
        assert_contains(rendered_panels[0][2], "MODEL OUTPUT")
        assert_contains(rendered_panels[1][2], "SIMULATOR GT / DIAGNOSTIC")
        assert_contains(rendered_panels[2][2], "TERRAIN ADVISORY (CONTINUED)")
        assert_contains(rendered_panels[3][2], "EVENT TIMING", "NOW")
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
        fake_viewer = ViewerStub(running_checks=5)
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
        fake_viewer = ViewerStub(running_checks=1)
        first_reflex = self.prepared.traces.first_reflex_sample
        assert first_reflex is not None
        later_pause_s = (
            float(
                self.prepared.resolved.run.timestamp_us[first_reflex + 100]
                / 1_000_000.0
            )
            + 0.0004
        )
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
        assert_false_fields(result, "viewer_physics_executed", "holdout_opened")
        self.assertIsNone(result["viewer_physics_parity"])
        self.assertEqual(result["pause_at_sample"], first_reflex + 100)
        self.assertEqual(result["auto_pause_sample"], first_reflex)
