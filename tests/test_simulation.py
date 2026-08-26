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
)
from fastreflex.simulation.hazards import (
    LOAD_OFF_N,
    LOAD_ON_N,
    SINK_PERSISTENCE_SAMPLES,
    SINK_THRESHOLD_M,
    SLIP_PERSISTENCE_SAMPLES,
    SLIP_THRESHOLD_M,
    TOUCHDOWN_TRANSIENT_SAMPLES,
    derive_physical_diagnostics,
)
from fastreflex.simulation.terrain import TERRAIN_PROFILES
from scripts.fastreflex import build_parser


ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_CONFIG = ROOT / "configs" / "simulator" / "g1.yaml"
DATASET_CONFIG = ROOT / "configs" / "dataset" / "hazard.yaml"


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

        model, terrain_id = load_g1_model("concrete")
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
                model, terrain_id = load_g1_model(name)
                np.testing.assert_allclose(model.geom_friction[terrain_id], profile.friction)
                np.testing.assert_allclose(model.geom_solref[terrain_id], profile.solref)
                np.testing.assert_allclose(model.geom_solimp[terrain_id], profile.solimp)

    def test_viewer_cli_rates_and_availability_contract(self) -> None:
        parser = build_parser()
        viewer_args = parser.parse_args(["simulate", "--viewer"])
        self.assertTrue(viewer_args.viewer)
        self.assertFalse(viewer_args.headless)
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
            contract["established_sink"]["first_loaded_penetration_change_m"],
            SINK_THRESHOLD_M,
        )
        self.assertEqual(
            contract["established_sink"]["persistence_ms"],
            SINK_PERSISTENCE_SAMPLES,
        )

        samples = 50
        contact = np.ones((samples, 2), dtype=bool)
        force = np.full((samples, 2), 10.0)
        xyz = np.zeros((samples, 2, 3))
        velocity = np.zeros((samples, 2, 3))
        penetration = np.full((samples, 2), 0.001)
        pre_fall = np.ones(samples, dtype=bool)
        xyz[12:15, 0, 0] = SLIP_THRESHOLD_M
        penetration[20:40, 1] += SINK_THRESHOLD_M

        diagnostics = derive_physical_diagnostics(
            contact, force, xyz, velocity, penetration, pre_fall
        )
        self.assertEqual(diagnostics.touchdown[0].tolist(), [True, True])
        self.assertFalse(diagnostics.established_slip[:14, 0].any())
        self.assertTrue(diagnostics.established_slip[14, 0])
        self.assertFalse(diagnostics.established_sink[:39, 1].any())
        self.assertTrue(diagnostics.established_sink[39, 1])
        self.assertTrue(np.all(np.isfinite(diagnostics.foot_world_xyz)))
        self.assertTrue(np.all(np.isfinite(diagnostics.foot_world_velocity_xyz)))
        self.assertTrue(np.all(np.isfinite(diagnostics.contact_penetration_m)))
        self.assertEqual(
            tuple(RuntimeTrace.__dataclass_fields__),
            ("sequence", "timestamp_us", "pelvis_imu"),
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
        self.assertTrue(np.all(np.isfinite(diagnostics.foot_world_xyz)))
        self.assertTrue(np.all(np.isfinite(diagnostics.foot_world_velocity_xyz)))
        self.assertTrue(np.all(np.isfinite(diagnostics.contact_penetration_m)))

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
