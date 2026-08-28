"""Contract tests for terrain touchdown events and sensor ablation."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from fastreflex.dataset.terrain import (
    HoldoutGuard,
    TerrainWindowSet,
    _compare_parity_pair,
    build_terrain_windows,
    build_touchdown_event_rows,
    extract_terrain_sensor_profile,
    fit_terrain_normalizer,
    load_terrain_collection_config,
    select_capped_events,
    terrain_identity_touchdown,
)
from fastreflex.simulation.g1 import (
    TESTED_POLICY_SHA256,
    load_g1_model,
    load_simulation_config,
    read_pelvis_imu,
    run_simulation,
)
from fastreflex.simulation.sensors import (
    FOOT_IMU_CHANNELS,
    FOOT_IMU_SITE_NAMES,
    read_foot_imu,
)
from fastreflex.simulation.terrain import (
    TERRAIN_CLASS_ORDER,
    terrain_contact_class_by_geom_id,
)
from fastreflex.training.terrain import (
    select_minimum_sensor,
    select_shortest_horizon,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment" / "20260828_terrain_rebuild_sensor_ablation.yaml"
SIMULATOR_CONFIG = ROOT / "configs" / "simulator" / "g1.yaml"
LOCAL_POLICY = ROOT / "artifacts" / "external" / "unitree_g1" / "g1_velocity_policy.onnx"


def _event_rows(contact: np.ndarray) -> list[dict[str, object]]:
    timestamps = (np.arange(len(contact), dtype=np.int64) + 1) * 1000
    return build_touchdown_event_rows(
        "terrain_run",
        "train",
        "concrete",
        "ice",
        timestamps,
        contact,
        None,
        True,
        False,
    )


class TerrainRecognitionTest(unittest.TestCase):
    def test_foot_imu_sites_shape_order_and_finite(self) -> None:
        model, _ = load_g1_model("concrete")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        sample = read_foot_imu(model, data)
        self.assertEqual(sample.shape, (12,))
        self.assertEqual(len(FOOT_IMU_CHANNELS), 12)
        self.assertEqual(
            FOOT_IMU_CHANNELS[:6],
            (
                "left_accel_x",
                "left_accel_y",
                "left_accel_z",
                "left_gyro_x",
                "left_gyro_y",
                "left_gyro_z",
            ),
        )
        self.assertTrue(np.all(np.isfinite(sample)))
        for side, site_name in zip(("left", "right"), FOOT_IMU_SITE_NAMES):
            site_id = model.site(site_name).id
            self.assertEqual(
                int(model.site_bodyid[site_id]), model.body(f"{side}_ankle_roll_link").id
            )
            np.testing.assert_allclose(model.site_pos[site_id], (0.035, 0.0, -0.02))

    def test_left_right_sensor_mapping_and_profile_slicing(self) -> None:
        fsr = np.arange(16, dtype=np.float32).reshape(2, 8)
        imu = np.arange(24, dtype=np.float32).reshape(2, 12)
        np.testing.assert_array_equal(
            extract_terrain_sensor_profile(fsr, imu, "left", "fsr4"), fsr[:, :4]
        )
        np.testing.assert_array_equal(
            extract_terrain_sensor_profile(fsr, imu, "right", "fsr4"), fsr[:, 4:]
        )
        np.testing.assert_array_equal(
            extract_terrain_sensor_profile(fsr, imu, "left", "foot_imu6"), imu[:, :6]
        )
        np.testing.assert_array_equal(
            extract_terrain_sensor_profile(fsr, imu, "right", "foot_imu6"), imu[:, 6:]
        )
        fusion = extract_terrain_sensor_profile(fsr, imu, "right", "fusion10")
        np.testing.assert_array_equal(fusion[:, :4], fsr[:, 4:])
        np.testing.assert_array_equal(fusion[:, 4:], imu[:, 6:])

    def test_exact_geom_identity_mapping_for_ice_and_sand(self) -> None:
        ice, _ = load_g1_model(
            "ice", slip_pattern="transition", source_terrain="marble"
        )
        ice_map = terrain_contact_class_by_geom_id(
            ice, "ice", "transition", "uniform", "balanced_soft", "marble"
        )
        self.assertEqual(
            ice_map[ice.geom("terrain_transition_pre").id],
            TERRAIN_CLASS_ORDER.index("marble"),
        )
        self.assertEqual(
            ice_map[ice.geom("terrain_transition_left").id],
            TERRAIN_CLASS_ORDER.index("ice"),
        )
        sand, _ = load_g1_model(
            "sand",
            "transition_left",
            "mild",
            sink_support_pattern="balanced_deformable",
            source_terrain="concrete",
        )
        sand_map = terrain_contact_class_by_geom_id(
            sand,
            "sand",
            "uniform",
            "transition_left",
            "balanced_deformable",
            "concrete",
        )
        self.assertEqual(
            sand_map[sand.geom("terrain_deformable_balanced_left_entry_medial").id],
            TERRAIN_CLASS_ORDER.index("sand"),
        )
        self.assertEqual(
            sand_map[sand.geom("terrain_transition_right").id],
            TERRAIN_CLASS_ORDER.index("concrete"),
        )

    def test_touchdown_exact_label_and_causal_windows(self) -> None:
        contact = np.zeros((100, 2, 4), dtype=bool)
        contact[10:80, 0, 2] = True
        rows = _event_rows(contact)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["terrain_gt"], "ICE")
        self.assertEqual(rows[0]["touchdown_sample"], 10)
        self.assertTrue(rows[0]["window_20ms_valid"])
        self.assertTrue(rows[0]["window_30ms_valid"])
        self.assertTrue(rows[0]["window_50ms_valid"])
        changed = contact.copy()
        changed[80:, 0, 0] = True
        self.assertEqual(rows, _event_rows(changed)[:1])

    def test_mixed_contact_exclusion_is_fixed_at_twenty_percent(self) -> None:
        contact = np.zeros((100, 2, 4), dtype=bool)
        contact[10:80, 0, 2] = True
        contact[10:20, 0, 0] = True
        row = next(row for row in _event_rows(contact) if row["terrain_gt"] == "ICE")
        self.assertAlmostEqual(row["mixed_contact_ratio"], 0.20)
        self.assertFalse(row["window_50ms_valid"])
        self.assertEqual(row["exclusion_reason"], "AMBIGUOUS_BOUNDARY")

    def test_identity_touchdown_is_per_foot_and_per_class(self) -> None:
        contact = np.zeros((5, 2, 4), dtype=bool)
        contact[1:4, 0, 0] = True
        contact[2:, 0, 2] = True
        contact[3:, 1, 3] = True
        onset = terrain_identity_touchdown(contact)
        self.assertEqual(np.argwhere(onset).tolist(), [[1, 0, 0], [2, 0, 2], [3, 1, 3]])

    def test_run_matrix_and_split_are_frozen_and_disjoint(self) -> None:
        config = load_terrain_collection_config(CONFIG)
        self.assertEqual(len(config.runs), 144)
        self.assertEqual(
            {name: sum(run.split == name for run in config.runs) for name in ("train", "validation", "holdout")},
            {"train": 88, "validation": 28, "holdout": 28},
        )
        self.assertEqual(len({run.condition_signature for run in config.runs}), 144)

    def test_event_cap_and_left_only_filter_are_deterministic(self) -> None:
        rows = []
        for foot in ("left", "right"):
            for sample in (10, 20, 30):
                rows.append(
                    {
                        "event_id": f"run_{foot}_{sample}",
                        "run_id": "run",
                        "foot": foot,
                        "terrain_class_id": 0,
                        "touchdown_sample": sample,
                        "split": "train",
                        "window_50ms_valid": True,
                    }
                )
        selected = select_capped_events(rows, "train", 2)
        self.assertEqual(len(selected), 2)
        left = select_capped_events(rows, "train", 2, foot="left")
        self.assertEqual(len(left), 2)
        self.assertTrue(all(row["foot"] == "left" for row in left))

    def test_train_only_normalization_records_only_supplied_events(self) -> None:
        windows = TerrainWindowSet(
            inputs=np.asarray([[[1.0], [3.0]], [[5.0], [7.0]]], dtype=np.float32),
            targets=np.asarray([0, 1]),
            run_ids=np.asarray(["train_a", "train_b"], dtype=object),
            event_ids=np.asarray(["event_a", "event_b"], dtype=object),
            feet=np.asarray(["left", "right"], dtype=object),
            touchdown_samples=np.asarray([1, 2]),
        )
        normalizer = fit_terrain_normalizer(windows)
        self.assertEqual(normalizer.fit_event_ids, ("event_a", "event_b"))
        self.assertEqual(normalizer.fit_run_ids, ("train_a", "train_b"))
        self.assertAlmostEqual(float(normalizer.mean[0]), 4.0)

    def test_holdout_guard_opens_exactly_once(self) -> None:
        guard = HoldoutGuard()
        with self.assertRaisesRegex(RuntimeError, "sealed"):
            guard.require_open()
        guard.open_once()
        guard.require_open()
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            guard.open_once()

    def test_holdout_waveform_requires_guard_and_model_input_has_no_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs").mkdir()
            np.savez_compressed(
                root / "runs" / "holdout_run.npz",
                foot_fsr=np.ones((60, 8), dtype=np.float32),
                foot_imu=np.ones((60, 12), dtype=np.float32),
                terrain_gt=np.full(60, 3, dtype=np.int8),
            )
            row = {
                "event_id": "holdout_event",
                "run_id": "holdout_run",
                "foot": "left",
                "terrain_class_id": 3,
                "touchdown_sample": 0,
                "split": "holdout",
                "window_20ms_valid": True,
                "window_30ms_valid": True,
                "window_50ms_valid": True,
            }
            with self.assertRaisesRegex(RuntimeError, "guard"):
                build_terrain_windows(root, [row], "fusion10", 50)
            guard = HoldoutGuard()
            guard.open_once()
            windows = build_terrain_windows(
                root, [row], "fusion10", 50, holdout_guard=guard
            )
            self.assertEqual(windows.inputs.shape, (1, 50, 10))
            self.assertFalse(np.any(windows.inputs == 3.0))

    def test_horizon_selection_is_shortest_passing(self) -> None:
        rows = [
            {"horizon_ms": 20, "validation_macro_f1_mean": 0.89, "validation_worst_class_recall_mean": 0.9},
            {"horizon_ms": 30, "validation_macro_f1_mean": 0.91, "validation_worst_class_recall_mean": 0.86},
            {"horizon_ms": 50, "validation_macro_f1_mean": 0.95, "validation_worst_class_recall_mean": 0.9},
        ]
        selected, passed = select_shortest_horizon(rows, 0.90, 0.85)
        self.assertTrue(passed)
        self.assertEqual(selected["horizon_ms"], 30)

    def test_minimum_sensor_selection_prefers_fewer_channels_in_band(self) -> None:
        rows = [
            {"profile": "fsr4", "input_channels": 4, "validation_macro_f1_mean": 0.92, "validation_worst_class_recall_mean": 0.86},
            {"profile": "foot_imu6", "input_channels": 6, "validation_macro_f1_mean": 0.94, "validation_worst_class_recall_mean": 0.90},
            {"profile": "fusion10", "input_channels": 10, "validation_macro_f1_mean": 0.95, "validation_worst_class_recall_mean": 0.92},
        ]
        selected, reason = select_minimum_sensor(rows, 0.90, 0.85, 0.02)
        self.assertEqual(selected["profile"], "foot_imu6")
        self.assertTrue(reason["qualification_available"])

    @unittest.skipUnless(
        LOCAL_POLICY.is_file(), "observer parity requires the verified local ONNX policy"
    )
    def test_foot_imu_one_khz_alignment_pelvis_parity_and_physics_parity(self) -> None:
        self.assertEqual(
            __import__("hashlib").sha256(LOCAL_POLICY.read_bytes()).hexdigest(),
            TESTED_POLICY_SHA256,
        )
        base = load_simulation_config(SIMULATOR_CONFIG)
        config = replace(
            base,
            duration_s=0.10,
            policy_path=LOCAL_POLICY,
            terrain="concrete",
            source_terrain="concrete",
            slip_pattern="uniform",
            sink_pattern="uniform",
            headless=True,
        )
        result = run_simulation(config)
        self.assertEqual(result.runtime.foot_imu.shape, (100, 12))
        np.testing.assert_array_equal(np.diff(result.runtime.timestamp_us), 1000)
        model, _ = load_g1_model("concrete")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertEqual(read_pelvis_imu(model, data).shape, (6,))
        parity = _compare_parity_pair(config, "unit")
        self.assertTrue(parity["passed"])
        self.assertEqual(parity["qpos_max_abs_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
