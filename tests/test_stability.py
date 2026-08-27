"""Contracts for exact stability, pelvis-IMU detection, and state fusion."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import yaml

from fastreflex.evaluation.integrated_stability import _window_set
from fastreflex.simulation.g1 import RuntimeTrace, load_g1_model
from fastreflex.simulation.stability import (
    DOUBLE_SUPPORT,
    LEFT_SINGLE_SUPPORT,
    NO_SUPPORT,
    RIGHT_SINGLE_SUPPORT,
    ExactStabilitySample,
    HazardState,
    IMURuleCalibration,
    ParallelRuntimeState,
    PhaseEnvelope,
    StabilityDiagnostics,
    StabilityState,
    StableCalibrationRun,
    TerrainState,
    assign_gait_phase,
    derive_stability_diagnostics,
    detect_instability,
    fit_phase_envelope,
    format_runtime_status,
    read_exact_stability_sample,
    run_imu_rule,
    signed_support_margin,
    support_polygon,
)


ROOT = Path(__file__).resolve().parents[1]
INTEGRATED_CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "20260827_terrain_stability_integrated_sanity.yaml"
)


def _points(samples: int = 1) -> np.ndarray:
    left = np.asarray(
        ((-0.1, 0.02, 0.0), (-0.1, 0.08, 0.0), (0.1, 0.02, 0.0), (0.1, 0.08, 0.0))
    )
    right = left.copy()
    right[:, 1] -= 0.10
    return np.repeat(np.stack((left, right))[None, ...], samples, axis=0)


def _diagnostics(margins: np.ndarray, phases: np.ndarray) -> StabilityDiagnostics:
    samples = len(margins)
    return StabilityDiagnostics(
        com_xyz_m=np.zeros((samples, 3)),
        com_velocity_xyz_m_s=np.zeros((samples, 3)),
        support_height_m=np.ones(samples),
        foot_support_points_xyz_m=_points(samples),
        gait_phase=np.asarray(phases, dtype=np.int8),
        xcom_xy_m=np.zeros((samples, 2)),
        raw_margin_of_stability_m=np.asarray(margins, dtype=np.float64),
    )


class StabilityTest(unittest.TestCase):
    def test_exact_whole_body_com_matches_mass_weighted_robot_subtree(self) -> None:
        model, _ = load_g1_model("concrete")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        sample = read_exact_stability_sample(model, data)
        pelvis_id = model.body("pelvis").id
        descendants = np.flatnonzero(model.body_rootid == pelvis_id)
        mass = model.body_mass[descendants]
        expected = np.sum(data.xipos[descendants] * mass[:, None], axis=0) / np.sum(mass)
        np.testing.assert_allclose(sample.com_xyz_m, expected, atol=1.0e-12)

    def test_com_velocity_capture_is_current_state_causal(self) -> None:
        model, _ = load_g1_model("concrete")
        data = mujoco.MjData(model)
        data.qvel[0] = 0.20
        mujoco.mj_forward(model, data)
        first = read_exact_stability_sample(model, data)
        frozen = first.com_velocity_xyz_m_s.copy()
        data.qvel[0] = -0.30
        mujoco.mj_forward(model, data)
        second = read_exact_stability_sample(model, data)
        np.testing.assert_array_equal(first.com_velocity_xyz_m_s, frozen)
        self.assertGreater(first.com_velocity_xyz_m_s[0], 0.0)
        self.assertLess(second.com_velocity_xyz_m_s[0], 0.0)

    def test_support_polygon_and_left_right_mapping(self) -> None:
        points = _points()[0]
        left = support_polygon(points, np.asarray((True, False)))
        right = support_polygon(points, np.asarray((False, True)))
        double = support_polygon(points, np.asarray((True, True)))
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertIsNotNone(double)
        assert left is not None and right is not None and double is not None
        self.assertGreater(float(np.mean(left[:, 1])), 0.0)
        self.assertLess(float(np.mean(right[:, 1])), 0.0)
        self.assertLess(float(np.min(double[:, 1])), float(np.min(left[:, 1])))
        self.assertIsNone(support_polygon(points, np.asarray((False, False))))

    def test_xcom_and_signed_support_margin(self) -> None:
        height = 0.8
        omega = np.sqrt(9.81 / height)
        com = np.asarray(((0.0, 0.05, height),))
        velocity = np.asarray(((0.05 * omega, 0.0, 0.0),))
        diagnostics = derive_stability_diagnostics(
            com, velocity, _points(), np.asarray(((True, True),))
        )
        np.testing.assert_allclose(diagnostics.xcom_xy_m[0], (0.05, 0.05))
        self.assertGreater(diagnostics.raw_margin_of_stability_m[0], 0.0)
        square = np.asarray(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)))
        self.assertAlmostEqual(signed_support_margin((0.0, 0.0), square), 1.0)
        self.assertAlmostEqual(signed_support_margin((2.0, 0.0), square), -1.0)

    def test_gait_phase_assignment(self) -> None:
        phases = assign_gait_phase(
            np.asarray(((False, False), (True, False), (False, True), (True, True)))
        )
        np.testing.assert_array_equal(
            phases,
            (NO_SUPPORT, LEFT_SINGLE_SUPPORT, RIGHT_SINGLE_SUPPORT, DOUBLE_SUPPORT),
        )

    def test_stable_envelope_rejects_nonstable_or_fall_calibration(self) -> None:
        phases = np.tile(
            (LEFT_SINGLE_SUPPORT, RIGHT_SINGLE_SUPPORT, DOUBLE_SUPPORT), 10
        )
        stable = StableCalibrationRun(
            "stable",
            _diagnostics(np.linspace(-0.02, 0.03, len(phases)), phases),
            intended_stable=True,
            observed_fall=False,
        )
        envelope = fit_phase_envelope((stable,), 0.01)
        self.assertEqual(envelope.calibration_run_ids, ("stable",))
        falling = StableCalibrationRun(
            "fall", stable.diagnostics, intended_stable=True, observed_fall=True
        )
        with self.assertRaises(ValueError):
            fit_phase_envelope((stable, falling), 0.01)
        nonstable = StableCalibrationRun(
            "hazard", stable.diagnostics, intended_stable=False, observed_fall=False
        )
        with self.assertRaises(ValueError):
            fit_phase_envelope((nonstable,), 0.01)

    def test_instability_persistence_is_causal_and_has_no_fall_dependency(self) -> None:
        phases = np.full(50, LEFT_SINGLE_SUPPORT)
        margins = np.zeros(50)
        margins[10:35] = -0.02
        envelope = PhaseEnvelope({LEFT_SINGLE_SUPPORT: 0.0}, 0.01, ("stable",))
        original = detect_instability(
            _diagnostics(margins, phases), envelope, 0.01, 20
        )
        self.assertEqual(np.flatnonzero(original.onset).tolist(), [29])
        future = margins.copy()
        future[35:] = 100.0
        changed = detect_instability(
            _diagnostics(future, phases), envelope, 0.01, 20
        )
        np.testing.assert_array_equal(original.onset[:35], changed.onset[:35])

    def test_imu_rule_is_causal_and_resets_after_stable_persistence(self) -> None:
        imu = np.zeros((30, 6), dtype=np.float64)
        imu[:, 2] = 9.81
        imu[5:8, 3] = 2.0
        imu[15:18, 3] = 2.0
        calibration = IMURuleCalibration(
            acceleration_norm_center_m_s2=9.81,
            thresholds=np.asarray((1.0, 1.0, 1.0)),
            quantile=0.995,
            calibration_run_ids=("stable",),
        )
        trace = run_imu_rule(imu, calibration, persistence_samples=3, reset_samples=2)
        self.assertEqual(np.flatnonzero(trace.onset).tolist(), [7, 17])
        self.assertFalse(trace.active[9])
        changed = imu.copy()
        changed[20:, 3] = 9.0
        future = run_imu_rule(changed, calibration, 3, 2)
        np.testing.assert_array_equal(trace.onset[:20], future.onset[:20])

    def test_ai_windows_are_run_disjoint_causal_and_use_no_future_sample(self) -> None:
        imu = np.arange(120, dtype=np.float32).reshape(20, 6)
        fake = SimpleNamespace(
            simulation=SimpleNamespace(
                runtime=SimpleNamespace(
                    pelvis_imu=imu, timestamp_us=np.arange(20) * 1000
                ),
                metadata={"first_fall_sample": None},
            ),
            instability=SimpleNamespace(onset=np.zeros(20, dtype=bool)),
        )
        windows = _window_set(
            {"run": fake}, ("run",), 5, 5, 1, np.zeros(6), np.ones(6)
        )
        changed_imu = imu.copy()
        changed_imu[10:] = -999.0
        fake_changed = SimpleNamespace(
            simulation=SimpleNamespace(
                runtime=SimpleNamespace(
                    pelvis_imu=changed_imu, timestamp_us=np.arange(20) * 1000
                ),
                metadata={"first_fall_sample": None},
            ),
            instability=fake.instability,
        )
        changed = _window_set(
            {"run": fake_changed}, ("run",), 5, 5, 1, np.zeros(6), np.ones(6)
        )
        np.testing.assert_array_equal(windows.inputs[:2], changed.inputs[:2])
        self.assertTrue(np.all(windows.run_ids == "run"))

    def test_parallel_state_producers_and_fusion_truth_table(self) -> None:
        initial = ParallelRuntimeState()
        terrain = initial.update_terrain(TerrainState.ICE, 1000)
        self.assertEqual(terrain.stability_state, StabilityState.STABLE)
        self.assertEqual(terrain.hazard_state, HazardState.NORMAL)
        unstable = terrain.update_stability(StabilityState.UNSTABLE, 2000)
        self.assertEqual(unstable.terrain_state, TerrainState.ICE)
        self.assertEqual(unstable.hazard_state, HazardState.SLIP_RISK)
        self.assertTrue(unstable.recovery_required)
        sand = unstable.update_terrain(TerrainState.SAND, 3000)
        self.assertEqual(sand.hazard_state, HazardState.SINK_RISK)
        hard = unstable.update_terrain(TerrainState.CONCRETE, 3000)
        self.assertEqual(hard.hazard_state, HazardState.GENERIC_INSTABILITY)
        unknown = initial.update_stability(StabilityState.UNSTABLE, 1000)
        self.assertEqual(unknown.hazard_state, HazardState.GENERIC_INSTABILITY)
        self.assertTrue(unknown.recovery_required)

    def test_status_formatting_does_not_mutate_physics(self) -> None:
        model, _ = load_g1_model("concrete")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        before_qpos = data.qpos.copy()
        before_qvel = data.qvel.copy()
        state = ParallelRuntimeState().update_terrain(TerrainState.CONCRETE, 1000)
        text = format_runtime_status(
            true_terrain=TerrainState.CONCRETE,
            state=state,
            stability_gt=StabilityState.STABLE,
            stability_ai=None,
            timestamp_us=1000,
            event_times_us={},
        )
        self.assertIn("RECOVERY_REQUIRED", text)
        np.testing.assert_array_equal(data.qpos, before_qpos)
        np.testing.assert_array_equal(data.qvel, before_qvel)

    def test_runtime_trace_contains_no_privileged_oracle_channels(self) -> None:
        names = {field.name for field in fields(RuntimeTrace)}
        self.assertEqual(names, {"sequence", "timestamp_us", "pelvis_imu", "foot_fsr"})
        forbidden = {"com", "xcom", "margin", "support_polygon", "terrain_gt", "fall"}
        self.assertFalse(names & forbidden)

    def test_integrated_config_freezes_matrix_splits_and_acceptance(self) -> None:
        with INTEGRATED_CONFIG.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(
            config["experiment"]["id"], "TERRAIN_STABILITY_INTEGRATED_SANITY"
        )
        self.assertEqual(config["experiment"]["matrix_status"], "frozen_before_integrated_runs")
        runs = config["runs"]
        self.assertEqual(len(runs), 44)
        self.assertEqual(len({run["id"] for run in runs}), 44)
        self.assertEqual(sum(run["group"] == "ice_fall" for run in runs), 10)
        self.assertEqual(sum(run["group"] == "sand_fall" for run in runs), 12)
        validation = set(config["ai_baseline"]["split"]["validation"])
        holdout = set(config["ai_baseline"]["split"]["holdout"])
        self.assertFalse(validation & holdout)
        self.assertEqual(config["terrain_runtime"]["status"], "TERRAIN_RUNTIME_MODEL_PENDING")
        self.assertEqual(config["stability_oracle"]["fixed_margin_m"], 0.010)
        self.assertEqual(config["stability_oracle"]["persistence_ms"], 20)


if __name__ == "__main__":
    unittest.main()
