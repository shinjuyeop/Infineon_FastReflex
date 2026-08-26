"""Contract tests for the single canonical G1 simulation baseline."""

from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import replace
import io
import os
from pathlib import Path
import unittest
from unittest import mock

import mujoco
import numpy as np
import yaml

from fastreflex.simulation.g1 import (
    ACTUATOR_NAMES,
    IMU_CHANNELS,
    RuntimeTrace,
    launch_passive_viewer,
    load_g1_model,
    load_simulation_config,
    read_pelvis_imu,
    run_simulation,
    summarize_result,
)
from fastreflex.simulation.hazards import (
    LOAD_OFF_N,
    LOAD_ON_N,
    PRE_EVENT_BASELINE_SAMPLES,
    SINK_HAZARD_TILT_PERSISTENCE_SAMPLES,
    SINK_HAZARD_TILT_THRESHOLD_RAD,
    SINK_PHYSICAL_PERSISTENCE_SAMPLES,
    SINK_PHYSICAL_THRESHOLD_M,
    SLIP_PERSISTENCE_SAMPLES,
    SLIP_THRESHOLD_M,
    TOUCHDOWN_TRANSIENT_SAMPLES,
    derive_physical_diagnostics,
)
from fastreflex.simulation.terrain import (
    SINK_PATCH_GEOM_NAMES,
    SINK_PATTERNS,
    SINK_SEVERITIES,
    SINK_SEVERITY_PROFILES,
    SLIP_PATTERNS,
    TERRAIN_PROFILES,
    TRANSITION_GROUND_GEOM_NAMES,
    TRANSITION_PATCH_END_X_M,
    TRANSITION_PATCH_GEOM_NAMES,
    TRANSITION_PATCH_START_X_M,
)
from scripts.fastreflex import build_parser


ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_CONFIG = ROOT / "configs" / "simulator" / "g1.yaml"
DATASET_CONFIG = ROOT / "configs" / "dataset" / "hazard.yaml"
SINK_EXPERIMENT_CONFIG = (
    ROOT / "configs" / "experiment" / "20260826_sink_scenario_sanity.yaml"
)
SINK_TRANSITION_EXPERIMENT_CONFIG = (
    ROOT / "configs" / "experiment" / "20260826_sink_transition_criteria.yaml"
)
SLIP_TRANSITION_EXPERIMENT_CONFIG = (
    ROOT / "configs" / "experiment" / "20260826_slip_transition_sanity.yaml"
)


class FakeViewer:
    def __init__(self, running_checks: int | None = None) -> None:
        self.sync_count = 0
        self.running_checks = running_checks
        self.is_running_count = 0

    def __enter__(self) -> "FakeViewer":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def is_running(self) -> bool:
        self.is_running_count += 1
        return (
            self.running_checks is None
            or self.is_running_count <= self.running_checks
        )

    def sync(self, state_only: bool = False) -> None:
        self.sync_count += 1


class SimulationTest(unittest.TestCase):
    def test_config_model_and_pelvis_imu_contract(self) -> None:
        config = load_simulation_config(SIMULATOR_CONFIG)
        self.assertEqual(config.physics_timestep_s, 0.0005)
        self.assertEqual(config.sensor_rate_hz, 1000)
        self.assertEqual(config.physics_steps_per_sample, 2)
        self.assertIsNone(config.policy_path)
        self.assertEqual(config.slip_pattern, "uniform")
        self.assertEqual(config.sink_pattern, "uniform")
        self.assertEqual(config.sink_severity, "moderate")

        model, ground_ids = load_g1_model("concrete")
        self.assertEqual(len(ground_ids), 1)
        terrain_id = next(iter(ground_ids))
        self.assertEqual(model.geom(terrain_id).name, "terrain")
        self.assertEqual(
            tuple(model.actuator(index).name for index in range(model.nu)),
            ACTUATOR_NAMES,
        )
        self.assertEqual((model.nq, model.nv, model.nu), (36, 35, 29))

        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        imu_site_id = model.site("imu").id
        pelvis_id = model.body("pelvis").id
        self.assertEqual(int(model.site_bodyid[imu_site_id]), pelvis_id)
        np.testing.assert_array_equal(model.site_pos[imu_site_id], np.zeros(3))
        np.testing.assert_array_equal(
            model.site_quat[imu_site_id], (1.0, 0.0, 0.0, 0.0)
        )
        np.testing.assert_allclose(
            data.site_xmat[imu_site_id].reshape(3, 3), np.eye(3), atol=1e-12
        )

        # MJCF symmetry and the coincident, unrotated IMU frame establish pelvis
        # +x forward, +y left, +z up. Injection proves accel then gyro ordering.
        self.assertGreater(model.body("left_hip_pitch_link").pos[1], 0.0)
        self.assertLess(model.body("right_hip_pitch_link").pos[1], 0.0)
        self.assertGreater(
            model.geom("left_foot_contact_3").pos[0],
            model.geom("left_foot_contact_1").pos[0],
        )
        for name, values in (
            ("imu_acc", (1.0, 2.0, 3.0)),
            ("imu_gyro", (4.0, 5.0, 6.0)),
        ):
            sensor_id = model.sensor(name).id
            address = int(model.sensor_adr[sensor_id])
            data.sensordata[address : address + 3] = values
        imu = read_pelvis_imu(model, data)
        self.assertEqual(
            IMU_CHANNELS,
            ("accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"),
        )
        self.assertEqual(imu.dtype, np.float32)
        np.testing.assert_array_equal(imu, np.arange(1.0, 7.0, dtype=np.float32))
        self.assertTrue(np.all(np.isfinite(imu)))

    def test_terrain_profiles_apply_exactly(self) -> None:
        self.assertEqual(tuple(TERRAIN_PROFILES), ("concrete", "marble", "ice", "sand"))
        for name, profile in TERRAIN_PROFILES.items():
            with self.subTest(terrain=name):
                model, ground_ids = load_g1_model(name)
                terrain_id = next(iter(ground_ids))
                np.testing.assert_allclose(model.geom_friction[terrain_id], profile.friction)
                np.testing.assert_allclose(model.geom_solref[terrain_id], profile.solref)
                np.testing.assert_allclose(model.geom_solimp[terrain_id], profile.solimp)

    def test_sink_patch_topology_side_mapping_and_profile_application(self) -> None:
        baseline, baseline_ids = load_g1_model("sand")
        baseline_id = next(iter(baseline_ids))
        self.assertEqual(baseline.geom(baseline_id).name, "terrain")
        self.assertEqual(baseline.geom_type[baseline_id], mujoco.mjtGeom.mjGEOM_PLANE)
        self.assertEqual(float(baseline.geom_pos[baseline_id, 2]), 0.0)

        for pattern in ("asymmetric_left", "asymmetric_right"):
            soft_side = pattern.removeprefix("asymmetric_")
            for severity in SINK_SEVERITIES:
                with self.subTest(pattern=pattern, severity=severity):
                    model, ground_ids = load_g1_model("sand", pattern, severity)
                    self.assertEqual(
                        {model.geom(geom_id).name for geom_id in ground_ids},
                        set(SINK_PATCH_GEOM_NAMES),
                    )
                    self.assertEqual(len(ground_ids), 2)
                    self.assertEqual(
                        mujoco.mj_name2id(
                            model, mujoco.mjtObj.mjOBJ_GEOM, "terrain"
                        ),
                        -1,
                    )
                    for side in ("left", "right"):
                        geom_id = model.geom(f"terrain_{side}").id
                        self.assertEqual(
                            model.geom_type[geom_id], mujoco.mjtGeom.mjGEOM_BOX
                        )
                        top_z = (
                            float(model.geom_pos[geom_id, 2])
                            + float(model.geom_size[geom_id, 2])
                        )
                        self.assertEqual(top_z, 0.0)
                        expected = (
                            SINK_SEVERITY_PROFILES[severity]
                            if side == soft_side
                            else TERRAIN_PROFILES["sand"]
                        )
                        np.testing.assert_allclose(
                            model.geom_friction[geom_id], expected.friction
                        )
                        np.testing.assert_allclose(
                            model.geom_solref[geom_id], expected.solref
                        )
                        np.testing.assert_allclose(
                            model.geom_solimp[geom_id], expected.solimp
                        )

                    left_id = model.geom("terrain_left").id
                    right_id = model.geom("terrain_right").id
                    left_min_y = (
                        model.geom_pos[left_id, 1] - model.geom_size[left_id, 1]
                    )
                    right_max_y = (
                        model.geom_pos[right_id, 1] + model.geom_size[right_id, 1]
                    )
                    self.assertEqual(float(left_min_y), 0.0)
                    self.assertEqual(float(right_max_y), 0.0)

        config = load_simulation_config(SIMULATOR_CONFIG)
        with self.assertRaisesRegex(ValueError, "require terrain='sand'"):
            replace(config, sink_pattern="asymmetric_left").validate()

    def test_transition_patch_geometry_profiles_and_side_mapping(self) -> None:
        self.assertEqual(TRANSITION_PATCH_START_X_M, 0.35)
        self.assertEqual(TRANSITION_PATCH_END_X_M, 1.10)
        for pattern in ("transition_left", "transition_right"):
            soft_side = pattern.removeprefix("transition_")
            for severity in SINK_SEVERITIES:
                with self.subTest(pattern=pattern, severity=severity):
                    model, ground_ids = load_g1_model("sand", pattern, severity)
                    self.assertEqual(
                        {model.geom(geom_id).name for geom_id in ground_ids},
                        set(TRANSITION_GROUND_GEOM_NAMES),
                    )
                    self.assertEqual(model.geom_contype[model.geom("terrain_left").id], 0)
                    self.assertEqual(model.geom_contype[model.geom("terrain_right").id], 0)
                    for name in TRANSITION_GROUND_GEOM_NAMES:
                        geom_id = model.geom(name).id
                        self.assertEqual(model.geom_contype[geom_id], 1)
                        self.assertEqual(
                            float(model.geom_pos[geom_id, 2] + model.geom_size[geom_id, 2]),
                            0.0,
                        )

                    pre_id = model.geom("terrain_transition_pre").id
                    left_id = model.geom("terrain_transition_left").id
                    right_id = model.geom("terrain_transition_right").id
                    post_id = model.geom("terrain_transition_post").id
                    self.assertAlmostEqual(
                        float(model.geom_pos[pre_id, 0] + model.geom_size[pre_id, 0]),
                        TRANSITION_PATCH_START_X_M,
                    )
                    for patch_id in (left_id, right_id):
                        self.assertAlmostEqual(
                            float(model.geom_pos[patch_id, 0] - model.geom_size[patch_id, 0]),
                            TRANSITION_PATCH_START_X_M,
                        )
                        self.assertAlmostEqual(
                            float(model.geom_pos[patch_id, 0] + model.geom_size[patch_id, 0]),
                            TRANSITION_PATCH_END_X_M,
                        )
                    self.assertAlmostEqual(
                        float(model.geom_pos[post_id, 0] - model.geom_size[post_id, 0]),
                        TRANSITION_PATCH_END_X_M,
                    )
                    self.assertEqual(float(model.qpos0[0]), 0.0)
                    self.assertGreater(
                        TRANSITION_PATCH_START_X_M,
                        float(model.qpos0[0]),
                    )
                    self.assertEqual(
                        float(model.geom_pos[left_id, 1] - model.geom_size[left_id, 1]),
                        0.0,
                    )
                    self.assertEqual(
                        float(model.geom_pos[right_id, 1] + model.geom_size[right_id, 1]),
                        0.0,
                    )
                    for side in ("left", "right"):
                        geom_id = model.geom(f"terrain_transition_{side}").id
                        expected = (
                            SINK_SEVERITY_PROFILES[severity]
                            if side == soft_side
                            else TERRAIN_PROFILES["concrete"]
                        )
                        np.testing.assert_allclose(
                            model.geom_solref[geom_id], expected.solref
                        )
                        np.testing.assert_allclose(
                            model.geom_solimp[geom_id], expected.solimp
                        )

    def test_full_width_slip_transition_topology_and_profile(self) -> None:
        self.assertEqual(SLIP_PATTERNS, ("uniform", "transition"))
        model, ground_ids = load_g1_model(
            "ice",
            slip_pattern="transition",
        )
        self.assertEqual(
            {model.geom(geom_id).name for geom_id in ground_ids},
            set(TRANSITION_GROUND_GEOM_NAMES),
        )
        self.assertEqual(model.geom_contype[model.geom("terrain_left").id], 0)
        self.assertEqual(model.geom_contype[model.geom("terrain_right").id], 0)
        pre_id = model.geom("terrain_transition_pre").id
        left_id = model.geom("terrain_transition_left").id
        right_id = model.geom("terrain_transition_right").id
        post_id = model.geom("terrain_transition_post").id
        self.assertAlmostEqual(
            float(model.geom_pos[pre_id, 0] + model.geom_size[pre_id, 0]),
            TRANSITION_PATCH_START_X_M,
        )
        for patch_id in (left_id, right_id):
            self.assertAlmostEqual(
                float(model.geom_pos[patch_id, 0] - model.geom_size[patch_id, 0]),
                TRANSITION_PATCH_START_X_M,
            )
            self.assertAlmostEqual(
                float(model.geom_pos[patch_id, 0] + model.geom_size[patch_id, 0]),
                TRANSITION_PATCH_END_X_M,
            )
        self.assertAlmostEqual(
            float(model.geom_pos[post_id, 0] - model.geom_size[post_id, 0]),
            TRANSITION_PATCH_END_X_M,
        )
        self.assertEqual(
            float(model.geom_pos[left_id, 1] - model.geom_size[left_id, 1]),
            0.0,
        )
        self.assertEqual(
            float(model.geom_pos[right_id, 1] + model.geom_size[right_id, 1]),
            0.0,
        )
        for name in TRANSITION_GROUND_GEOM_NAMES:
            geom_id = model.geom(name).id
            self.assertEqual(model.geom_contype[geom_id], 1)
            self.assertEqual(
                float(model.geom_pos[geom_id, 2] + model.geom_size[geom_id, 2]),
                0.0,
            )
            expected = (
                TERRAIN_PROFILES["ice"]
                if name in TRANSITION_PATCH_GEOM_NAMES
                else TERRAIN_PROFILES["concrete"]
            )
            np.testing.assert_allclose(
                model.geom_friction[geom_id],
                expected.friction,
            )
            np.testing.assert_allclose(model.geom_solref[geom_id], expected.solref)
            np.testing.assert_allclose(model.geom_solimp[geom_id], expected.solimp)

        config = load_simulation_config(SIMULATOR_CONFIG)
        with self.assertRaisesRegex(ValueError, "requires terrain='ice'"):
            replace(config, slip_pattern="transition").validate()
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            replace(
                config,
                terrain="ice",
                slip_pattern="transition",
                sink_pattern="transition_left",
            ).validate()

    def test_viewer_cli_rates_and_availability_contract(self) -> None:
        parser = build_parser()
        viewer_args = parser.parse_args(["simulate", "--viewer"])
        self.assertTrue(viewer_args.viewer)
        self.assertFalse(viewer_args.headless)
        sink_args = parser.parse_args(
            [
                "simulate",
                "--terrain",
                "sand",
                "--sink-pattern",
                "asymmetric_right",
                "--sink-severity",
                "severe",
            ]
        )
        self.assertEqual(sink_args.sink_pattern, "asymmetric_right")
        self.assertEqual(sink_args.sink_severity, "severe")
        transition_args = parser.parse_args(
            ["simulate", "--terrain", "sand", "--sink-pattern", "transition_left"]
        )
        self.assertEqual(transition_args.sink_pattern, "transition_left")
        slip_transition_args = parser.parse_args(
            ["simulate", "--terrain", "ice", "--slip-pattern", "transition"]
        )
        self.assertEqual(slip_transition_args.slip_pattern, "transition")
        with redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as conflict:
                parser.parse_args(["simulate", "--headless", "--viewer"])
        self.assertEqual(conflict.exception.code, 2)
        self.assertIn("not allowed with argument --headless", error.getvalue())

        viewer_config = replace(
            load_simulation_config(SIMULATOR_CONFIG), headless=False
        )
        viewer_config.validate()
        model, _ = load_g1_model(viewer_config.terrain)
        self.assertEqual(model.opt.timestep, 0.0005)
        self.assertEqual(viewer_config.sensor_rate_hz, 1000)
        with mock.patch(
            "fastreflex.simulation.g1.importlib.import_module",
            side_effect=ImportError("viewer unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "viewer is unavailable"):
                launch_passive_viewer(model, mujoco.MjData(model))

    def test_physical_diagnostics_and_dataset_threshold_parity(self) -> None:
        with DATASET_CONFIG.open("r", encoding="utf-8") as stream:
            contract = yaml.safe_load(stream)["labels"]
        self.assertEqual(contract["physical_load"]["on_threshold_n"], LOAD_ON_N)
        self.assertEqual(contract["physical_load"]["off_threshold_n"], LOAD_OFF_N)
        self.assertEqual(
            contract["physical_load"]["touchdown_transient_ms"],
            TOUCHDOWN_TRANSIENT_SAMPLES,
        )
        self.assertEqual(
            contract["established_slip"]["touchdown_anchor_drift_m"],
            SLIP_THRESHOLD_M,
        )
        self.assertEqual(
            contract["established_slip"]["persistence_ms"],
            SLIP_PERSISTENCE_SAMPLES,
        )
        self.assertEqual(
            contract["established_slip"]["primary_aggregation"],
            "any_foot",
        )
        self.assertEqual(
            contract["sink_physical"]["first_loaded_penetration_change_m"],
            SINK_PHYSICAL_THRESHOLD_M,
        )
        self.assertEqual(
            contract["sink_physical"]["persistence_ms"],
            SINK_PHYSICAL_PERSISTENCE_SAMPLES,
        )
        self.assertEqual(
            contract["sink_hazard"]["criteria_status"],
            "SINK_HAZARD_CRITERIA_FROZEN",
        )
        self.assertEqual(
            contract["sink_hazard"]["pelvis_tilt_threshold_rad"],
            SINK_HAZARD_TILT_THRESHOLD_RAD,
        )
        self.assertEqual(
            contract["sink_hazard"]["persistence_ms"],
            SINK_HAZARD_TILT_PERSISTENCE_SAMPLES,
        )

        samples = 50
        contact = np.ones((samples, 2), dtype=bool)
        force = np.full((samples, 2), 10.0)
        xyz = np.zeros((samples, 2, 3))
        velocity = np.zeros((samples, 2, 3))
        penetration = np.full((samples, 2), 0.001)
        pre_fall = np.ones(samples, dtype=bool)
        xyz[12:15, 0, 0] = SLIP_THRESHOLD_M
        penetration[20:40, 1] += SINK_PHYSICAL_THRESHOLD_M
        pelvis_z = np.full(samples, 0.8)
        orientation = np.tile((1.0, 0.0, 0.0, 0.0), (samples, 1))
        angular_velocity = np.zeros((samples, 3))
        linear_velocity = np.zeros((samples, 3))
        linear_velocity[:, 0] = 0.1
        fall_active = np.zeros(samples, dtype=bool)
        soft_patch_contact = np.zeros((samples, 2), dtype=bool)
        soft_patch_contact[10:45, 1] = True
        low_friction_patch_contact = np.zeros((samples, 2), dtype=bool)
        low_friction_patch_contact[10:45, 0] = True

        diagnostics = derive_physical_diagnostics(
            contact,
            force,
            xyz,
            velocity,
            penetration,
            pre_fall,
            pelvis_z,
            orientation,
            angular_velocity,
            linear_velocity,
            0.15,
            fall_active,
            soft_patch_contact=soft_patch_contact,
            low_friction_patch_contact=low_friction_patch_contact,
        )
        self.assertEqual(diagnostics.touchdown[0].tolist(), [True, True])
        self.assertFalse(diagnostics.established_slip[:14, 0].any())
        self.assertTrue(diagnostics.established_slip[14, 0])
        self.assertTrue(diagnostics.established_slip_onset[14, 0])
        self.assertTrue(diagnostics.established_slip_after_patch_onset[14, 0])
        self.assertTrue(diagnostics.any_established_slip_onset[14])
        self.assertTrue(diagnostics.any_established_slip_after_patch_onset[14])
        self.assertTrue(diagnostics.low_friction_patch_contact_onset[10, 0])
        self.assertFalse(diagnostics.sink_physical_active[:39, 1].any())
        self.assertTrue(diagnostics.sink_physical_active[39, 1])
        self.assertTrue(diagnostics.sink_physical_onset[39, 1])
        self.assertTrue(diagnostics.soft_patch_contact_onset[10, 1])
        self.assertTrue(diagnostics.sink_physical_after_patch_onset[39, 1])
        self.assertEqual(diagnostics.sink_physical_episode_id[39, 1], 0)
        np.testing.assert_allclose(diagnostics.pelvis_tilt_rad, 0.0)
        np.testing.assert_allclose(diagnostics.pelvis_forward_velocity_m_s, 0.1)
        np.testing.assert_allclose(diagnostics.forward_velocity_error_m_s, 0.05)
        self.assertEqual(
            np.count_nonzero(diagnostics.pre_event_baseline_valid), 10
        )
        self.assertEqual(PRE_EVENT_BASELINE_SAMPLES, 1000)
        np.testing.assert_allclose(
            diagnostics.pelvis_z_drop_from_pre_event_m[10:], 0.0
        )
        np.testing.assert_allclose(
            diagnostics.forward_velocity_drop_from_pre_event_m_s[10:], 0.0
        )
        self.assertTrue(np.all(np.isfinite(diagnostics.foot_world_xyz)))
        self.assertTrue(np.all(np.isfinite(diagnostics.foot_world_velocity_xyz)))
        self.assertTrue(np.all(np.isfinite(diagnostics.contact_penetration_m)))
        self.assertEqual(
            tuple(RuntimeTrace.__dataclass_fields__),
            ("sequence", "timestamp_us", "pelvis_imu"),
        )

        tilted_orientation = orientation.copy()
        tilt = SINK_HAZARD_TILT_THRESHOLD_RAD + 0.01
        tilted_orientation[20:, 0] = np.cos(tilt / 2.0)
        tilted_orientation[20:, 1] = np.sin(tilt / 2.0)
        tilted = derive_physical_diagnostics(
            contact,
            force,
            xyz,
            velocity,
            penetration,
            pre_fall,
            pelvis_z,
            tilted_orientation,
            angular_velocity,
            linear_velocity,
            0.15,
            fall_active,
            soft_patch_contact=soft_patch_contact,
            low_friction_patch_contact=low_friction_patch_contact,
        )
        expected_t2 = 20 + SINK_HAZARD_TILT_PERSISTENCE_SAMPLES - 1
        self.assertTrue(tilted.sink_degradation_onset[expected_t2])
        self.assertTrue(tilted.sink_hazard_onset[expected_t2])
        self.assertFalse(tilted.sink_hazard_active[:expected_t2].any())

    def test_sink_sanity_experiment_config_is_bounded_and_symmetric(self) -> None:
        with SINK_EXPERIMENT_CONFIG.open("r", encoding="utf-8") as stream:
            experiment = yaml.safe_load(stream)
        self.assertEqual(
            experiment["experiment"]["id"], "SINK_HAZARD_SCENARIO_SANITY"
        )
        self.assertEqual(experiment["common"]["duration_s"], 10.0)
        runs = experiment["runs"]
        self.assertEqual(len(runs), 8)
        self.assertEqual(
            {run["id"] for run in runs[:2]},
            {"concrete_control", "uniform_sand_control"},
        )
        asymmetric = runs[2:]
        self.assertEqual(
            {
                (run["sink_pattern"], run["sink_severity"])
                for run in asymmetric
            },
            {
                (f"asymmetric_{side}", severity)
                for side in ("left", "right")
                for severity in SINK_SEVERITIES
            },
        )
        self.assertFalse(
            experiment["interpretation"]["primary_sink_labels_generated"]
        )

    def test_sink_transition_experiment_config_is_finite_and_bounded(self) -> None:
        with SINK_TRANSITION_EXPERIMENT_CONFIG.open("r", encoding="utf-8") as stream:
            experiment = yaml.safe_load(stream)
        self.assertEqual(
            experiment["experiment"]["id"], "SINK_HAZARD_TRANSITION_AND_CRITERIA"
        )
        self.assertEqual(experiment["common"]["duration_s"], 8.0)
        self.assertEqual(
            experiment["geometry"]["patch_start_x_m"],
            TRANSITION_PATCH_START_X_M,
        )
        self.assertEqual(
            experiment["geometry"]["patch_end_x_m"],
            TRANSITION_PATCH_END_X_M,
        )
        self.assertEqual(
            experiment["timeline"]["t2_degradation"]["threshold_rad"],
            SINK_HAZARD_TILT_THRESHOLD_RAD,
        )
        runs = experiment["runs"]
        self.assertEqual(len(runs), 8)
        self.assertEqual(
            {
                (run["sink_pattern"], run["sink_severity"])
                for run in runs[2:]
            },
            {
                (f"transition_{side}", severity)
                for side in ("left", "right")
                for severity in SINK_SEVERITIES
            },
        )
        self.assertFalse(
            experiment["interpretation"]["primary_sink_labels_generated"]
        )
        self.assertEqual(
            experiment["interpretation"]["sink_hazard_status"],
            "SINK_HAZARD_CRITERIA_FROZEN",
        )

    def test_slip_transition_experiment_config_is_finite_and_bounded(self) -> None:
        with SLIP_TRANSITION_EXPERIMENT_CONFIG.open("r", encoding="utf-8") as stream:
            experiment = yaml.safe_load(stream)
        self.assertEqual(
            experiment["experiment"]["id"],
            "SLIP_HAZARD_TRANSITION_SANITY",
        )
        self.assertEqual(experiment["common"]["duration_s"], 8.0)
        self.assertEqual(
            experiment["geometry"]["patch_start_x_m"],
            TRANSITION_PATCH_START_X_M,
        )
        self.assertEqual(
            experiment["geometry"]["patch_end_x_m"],
            TRANSITION_PATCH_END_X_M,
        )
        self.assertEqual(experiment["geometry"]["patch_width"], "full")
        runs = experiment["runs"]
        self.assertEqual(len(runs), 5)
        self.assertEqual(
            {run["command_speed_mps"] for run in runs[1:]},
            {0.10, 0.15, 0.20, 0.25},
        )
        self.assertTrue(
            all(run["slip_pattern"] == "transition" for run in runs[1:])
        )
        self.assertFalse(
            experiment["interpretation"]["terrain_identity_is_label"]
        )

    @unittest.skipUnless(
        os.environ.get("FASTREFLEX_G1_POLICY"),
        "end-to-end policy smoke requires the user-supplied ONNX artifact",
    )
    def test_walking_smoke_sampling_and_runtime_separation(self) -> None:
        config = replace(
            load_simulation_config(SIMULATOR_CONFIG),
            duration_s=0.2,
            policy_path=Path(os.environ["FASTREFLEX_G1_POLICY"]),
        )
        with mock.patch(
            "fastreflex.simulation.g1.importlib.import_module",
            side_effect=ImportError("display-free regression"),
        ):
            result = run_simulation(config)
        self.assertEqual(result.runtime.pelvis_imu.shape, (200, 6))
        self.assertEqual(result.runtime.pelvis_imu.dtype, np.float32)
        self.assertEqual(result.runtime.timestamp_us.dtype, np.int64)
        np.testing.assert_array_equal(
            np.diff(result.runtime.timestamp_us), np.full(199, 1000)
        )
        np.testing.assert_array_equal(result.runtime.sequence, np.arange(200))
        self.assertTrue(np.all(np.isfinite(result.runtime.pelvis_imu)))
        self.assertEqual(
            set(vars(result.runtime)), {"sequence", "timestamp_us", "pelvis_imu"}
        )
        self.assertEqual(result.metadata["dropped_samples"], 0)
        self.assertIsNone(result.metadata["first_fall_sample"])

        diagnostics = result.diagnostics
        self.assertEqual(diagnostics.foot_world_xyz.shape, (200, 2, 3))
        self.assertEqual(diagnostics.foot_world_velocity_xyz.shape, (200, 2, 3))
        self.assertEqual(diagnostics.contact_penetration_m.shape, (200, 2))
        self.assertEqual(diagnostics.sink_physical_active.shape, (200, 2))
        self.assertEqual(diagnostics.pelvis_orientation_wxyz.shape, (200, 4))
        self.assertEqual(diagnostics.pelvis_angular_velocity_rad_s.shape, (200, 3))
        self.assertEqual(diagnostics.pelvis_linear_velocity_m_s.shape, (200, 3))
        self.assertEqual(diagnostics.pelvis_tilt_rad.shape, (200,))
        self.assertEqual(diagnostics.fall_active.shape, (200,))
        self.assertTrue(np.all(np.isfinite(diagnostics.foot_world_xyz)))
        self.assertTrue(np.all(np.isfinite(diagnostics.foot_world_velocity_xyz)))
        self.assertTrue(np.all(np.isfinite(diagnostics.contact_penetration_m)))

        transition_result = run_simulation(
            replace(
                config,
                terrain="sand",
                sink_pattern="transition_left",
                sink_severity="severe",
                duration_s=4.0,
            )
        )
        transition_summary = summarize_result(transition_result)
        t0 = transition_summary["first_soft_patch_contact_sample_per_foot"][0]
        t1 = transition_summary[
            "first_sink_physical_after_patch_sample_per_foot"
        ][0]
        t2 = transition_summary["first_sink_hazard_sample"]
        t3 = transition_summary["first_fall_sample"]
        self.assertIsNotNone(t0)
        self.assertIsNotNone(t1)
        self.assertIsNotNone(t2)
        self.assertIsNotNone(t3)
        self.assertGreaterEqual(t0, 1500)
        self.assertLessEqual(t0, 3000)
        self.assertLess(t0, t1)
        self.assertLess(t1, t2)
        self.assertLess(t2, t3)
        self.assertIn(
            "nonfoot_surface_contact", transition_summary["first_fall_reasons"]
        )
        self.assertTrue(
            np.all(
                np.isfinite(
                    transition_result.diagnostics.pelvis_z_drop_from_pre_event_m[t0:]
                )
            )
        )

        slip_result = run_simulation(
            replace(
                config,
                terrain="ice",
                slip_pattern="transition",
                duration_s=3.5,
            )
        )
        slip_summary = summarize_result(slip_result)
        slip_t0 = slip_summary[
            "first_low_friction_patch_contact_sample_per_foot"
        ]
        slip_t1 = slip_summary[
            "first_established_slip_after_patch_sample_per_foot"
        ]
        any_slip_t1 = slip_summary[
            "first_any_established_slip_after_patch_sample"
        ]
        self.assertTrue(all(value is not None for value in slip_t0))
        self.assertTrue(all(value is not None for value in slip_t1))
        self.assertGreaterEqual(min(slip_t0), 1500)
        self.assertLessEqual(min(slip_t0), 3000)
        self.assertGreater(any_slip_t1, min(slip_t0))
        self.assertEqual(any_slip_t1, min(slip_t1))
        self.assertEqual(
            slip_summary["slip_transition_qualification"],
            "CLEAN_SLIP_EVENT",
        )
        self.assertIsNone(slip_summary["first_sink_hazard_sample"])
        self.assertEqual(
            set(vars(slip_result.runtime)),
            {"sequence", "timestamp_us", "pelvis_imu"},
        )

        fake_viewer = FakeViewer()
        viewer_config = replace(config, headless=False)
        with mock.patch(
            "fastreflex.simulation.g1.launch_passive_viewer",
            return_value=fake_viewer,
        ), mock.patch("fastreflex.simulation.g1._pace_viewer"):
            viewer_result = run_simulation(viewer_config)
        self.assertGreater(fake_viewer.sync_count, 1)
        self.assertEqual(viewer_result.metadata["physics_timestep_s"], 0.0005)
        self.assertEqual(viewer_result.metadata["sensor_rate_hz"], 1000)
        self.assertTrue(viewer_result.metadata["viewer"])
        self.assertFalse(viewer_result.metadata["terminated_by_viewer"])
        for field in RuntimeTrace.__dataclass_fields__:
            np.testing.assert_equal(
                getattr(result.runtime, field), getattr(viewer_result.runtime, field)
            )
        for field in type(diagnostics).__dataclass_fields__:
            np.testing.assert_equal(
                getattr(diagnostics, field), getattr(viewer_result.diagnostics, field)
            )

        closing_viewer = FakeViewer(running_checks=20)
        with mock.patch(
            "fastreflex.simulation.g1.launch_passive_viewer",
            return_value=closing_viewer,
        ), mock.patch("fastreflex.simulation.g1._pace_viewer"):
            partial_result = run_simulation(viewer_config)
        self.assertTrue(partial_result.metadata["terminated_by_viewer"])
        self.assertLess(
            partial_result.metadata["actual_samples"],
            partial_result.metadata["expected_samples"],
        )
        self.assertEqual(partial_result.metadata["dropped_samples"], 0)


if __name__ == "__main__":
    unittest.main()
