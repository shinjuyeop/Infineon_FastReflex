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
    SUPPORT_BASELINE_MIN_QUADRANTS,
    SUPPORT_BASELINE_PRESENCE_RATIO,
    SUPPORT_BASELINE_SAMPLES,
    SUPPORT_LOSS_PERSISTENCE_SAMPLES,
    SUPPORT_LOSS_THRESHOLD_RATIO,
    SUPPORT_TOTAL_LOAD_MIN_RATIO,
    SURFACE_SPREAD_PERSISTENCE_SAMPLES,
    SURFACE_SPREAD_THRESHOLD_M,
    TOUCHDOWN_TRANSIENT_SAMPLES,
    derive_physical_diagnostics,
    support_penetration_diagnostics,
    read_exact_foot_sample,
    support_loss_diagnostics,
    surface_displacement_diagnostics,
    uneven_support_oracle,
)
from fastreflex.simulation.sensors import (
    FSR_CHANNELS,
    fsr_quadrant_index,
    read_virtual_fsr,
)
from fastreflex.simulation.terrain import (
    DEFORMABLE_CELL_ORDER,
    DEFORMABLE_SUPPORT_PATTERNS,
    DEFORMABLE_SUPPORT_PROFILES,
    SINK_PATCH_GEOM_NAMES,
    SINK_SEVERITIES,
    SINK_SEVERITY_PROFILES,
    SINK_SUPPORT_PATTERNS,
    SLIP_PATTERNS,
    TERRAIN_PROFILES,
    TRANSITION_GROUND_GEOM_NAMES,
    TRANSITION_PATCH_END_X_M,
    TRANSITION_PATCH_GEOM_NAMES,
    TRANSITION_PATCH_START_X_M,
    TRANSITION_PATCH_WIDTH_M,
    deformable_support_layout,
    read_deformable_support_sample,
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
SINK_PHYSICAL_REDEFINITION_CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "20260827_sink_physical_hazard_redefinition.yaml"
)
SINK_SUPPORT_LOSS_CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "20260827_sink_support_loss_oracle_sanity.yaml"
)
SINK_DEFORMABLE_SUPPORT_CONFIG = (
    ROOT
    / "configs"
    / "experiment"
    / "20260827_sink_deformable_support_proxy_sanity.yaml"
)
LOCAL_POLICY = (
    ROOT
    / "artifacts"
    / "external"
    / "unitree_g1"
    / "g1_velocity_policy.onnx"
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
    def test_deformable_support_geometry_joint_and_pattern_contract(self) -> None:
        selected_by_pattern = {
            "medial_deformable": {"entry_medial", "exit_medial"},
            "lateral_deformable": {"entry_lateral", "exit_lateral"},
            "localized_deformable": {"entry_medial"},
        }
        for side_index, side in enumerate(("left", "right")):
            for pattern in DEFORMABLE_SUPPORT_PATTERNS:
                with self.subTest(side=side, pattern=pattern):
                    model, ground_ids = load_g1_model(
                        "sand",
                        f"transition_{side}",
                        "moderate",
                        sink_support_pattern=pattern,
                    )
                    self.assertEqual((model.nq, model.nv, model.nu), (46, 45, 29))
                    layout = deformable_support_layout(model, pattern)
                    self.assertEqual(
                        np.count_nonzero(layout.geom_ids[side_index] >= 0), 4
                    )
                    self.assertFalse(
                        np.any(layout.geom_ids[1 - side_index] >= 0)
                    )
                    data = mujoco.MjData(model)
                    mujoco.mj_forward(model, data)
                    for cell_index, cell in enumerate(DEFORMABLE_CELL_ORDER):
                        geom_id = int(layout.geom_ids[side_index, cell_index])
                        self.assertIn(geom_id, ground_ids)
                        self.assertAlmostEqual(
                            float(
                                data.geom_xpos[geom_id, 2]
                                + model.geom_size[geom_id, 2]
                            ),
                            0.0,
                        )
                        qpos_address = int(
                            layout.qpos_addresses[side_index, cell_index]
                        )
                        joint_ids = np.flatnonzero(
                            model.jnt_qposadr == qpos_address
                        )
                        self.assertEqual(joint_ids.size, 1)
                        joint_id = int(joint_ids[0])
                        self.assertEqual(
                            int(model.jnt_type[joint_id]),
                            int(mujoco.mjtJoint.mjJNT_SLIDE),
                        )
                        np.testing.assert_array_equal(
                            model.jnt_axis[joint_id], (0.0, 0.0, 1.0)
                        )
                        body_id = int(model.jnt_bodyid[joint_id])
                        self.assertEqual(model.body_gravcomp[body_id], 1.0)
                        expected_profile = DEFORMABLE_SUPPORT_PROFILES["moderate"]
                        if pattern != "balanced_deformable" and cell not in (
                            selected_by_pattern[pattern]
                        ):
                            expected_profile = DEFORMABLE_SUPPORT_PROFILES[
                                "reference"
                            ]
                        self.assertAlmostEqual(
                            -float(model.jnt_range[joint_id, 0]),
                            expected_profile.travel_m,
                        )
                        self.assertEqual(float(model.jnt_range[joint_id, 1]), 0.0)
                        self.assertEqual(
                            float(model.jnt_stiffness[joint_id]),
                            expected_profile.stiffness_n_per_m,
                        )
                    unique_joints = np.unique(layout.qpos_addresses[side_index])
                    self.assertEqual(
                        len(unique_joints),
                        1 if pattern == "balanced_deformable" else 4,
                    )

                    model.body_gravcomp[:] = 1.0
                    for _ in range(200):
                        mujoco.mj_step(model, data)
                    unloaded = read_deformable_support_sample(data, layout)
                    np.testing.assert_allclose(
                        unloaded.displacement_m, 0.0, atol=1.0e-12
                    )

    def test_passive_deformable_support_load_and_recovery(self) -> None:
        steady_displacement = []
        for severity in ("mild", "moderate", "severe"):
            model, _ = load_g1_model(
                "sand",
                "transition_left",
                severity,
                sink_support_pattern="balanced_deformable",
            )
            joint_id = model.joint("deformable_balanced_left_slide").id
            qpos_address = int(model.jnt_qposadr[joint_id])
            dof_address = int(model.jnt_dofadr[joint_id])
            model.body_gravcomp[:] = 1.0
            for geom_id in range(model.ngeom):
                model.geom_contype[geom_id] = 0
                model.geom_conaffinity[geom_id] = 0
            data = mujoco.MjData(model)
            response = []
            for _ in range(2000):
                data.qfrc_applied[dof_address] = -100.0
                mujoco.mj_step(model, data)
                response.append(-float(data.qpos[qpos_address]))
            travel = -float(model.jnt_range[joint_id, 0])
            self.assertGreater(response[-1], 0.0)
            self.assertLessEqual(max(response), travel + 0.0005)
            steady_displacement.append(float(np.mean(response[-100:])))
            for _ in range(2000):
                data.qfrc_applied[dof_address] = 0.0
                mujoco.mj_step(model, data)
            self.assertLess(abs(float(data.qpos[qpos_address])), 1.0e-8)
        self.assertEqual(steady_displacement, sorted(steady_displacement))

    def test_surface_displacement_oracle_gating_persistence_and_causality(
        self,
    ) -> None:
        samples = 70
        displacement = np.zeros((samples, 2, 4), dtype=np.float64)
        displacement[10:, 0, 2] = 0.012
        velocity = np.zeros_like(displacement)
        velocity[10, 0, 2] = 0.25
        cell_contact = np.zeros_like(displacement, dtype=bool)
        cell_contact[5:15, 0, 2] = True
        patch_contact = np.zeros((samples, 2), dtype=bool)
        patch_contact[5:15, 0] = True
        loaded = np.ones((samples, 2), dtype=bool)
        episodes = np.zeros((samples, 2), dtype=np.int32)
        pre_fall = np.ones(samples, dtype=bool)
        diagnostics = surface_displacement_diagnostics(
            displacement,
            velocity,
            cell_contact,
            patch_contact,
            loaded,
            episodes,
            pre_fall,
        )
        np.testing.assert_allclose(
            diagnostics["support_surface_spread_m"][10:, 0], 0.012
        )
        self.assertEqual(
            np.flatnonzero(diagnostics["deformable_sink_onset"][:, 0]).tolist(),
            [29],
        )
        self.assertTrue(diagnostics["deformable_patch_episode_active"][20, 0])
        self.assertTrue(diagnostics["deformable_sink_active"][40, 0])
        self.assertEqual(
            diagnostics["support_surface_max_downward_velocity_m_s"][10, 0],
            0.25,
        )

        no_patch = surface_displacement_diagnostics(
            displacement,
            velocity,
            cell_contact,
            np.zeros_like(patch_contact),
            loaded,
            episodes,
            pre_fall,
        )
        self.assertFalse(no_patch["deformable_sink_active"].any())
        future = displacement.copy()
        future[40:] = 0.0
        changed = surface_displacement_diagnostics(
            future,
            velocity,
            cell_contact,
            patch_contact,
            loaded,
            episodes,
            pre_fall,
        )
        np.testing.assert_array_equal(
            diagnostics["deformable_sink_onset"][:40],
            changed["deformable_sink_onset"][:40],
        )
        changed_episode = episodes.copy()
        changed_episode[20:, 0] = 1
        reset = surface_displacement_diagnostics(
            displacement,
            velocity,
            cell_contact,
            patch_contact,
            loaded,
            changed_episode,
            pre_fall,
        )
        self.assertFalse(reset["deformable_sink_active"][20:, 0].any())

    def test_deformable_support_experiment_config_is_predeclared(self) -> None:
        with SINK_DEFORMABLE_SUPPORT_CONFIG.open(
            "r", encoding="utf-8"
        ) as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(
            config["experiment"]["id"],
            "SINK_DEFORMABLE_SUPPORT_PROXY_SANITY",
        )
        self.assertEqual(
            config["mechanics"]["parameter_status"],
            "frozen_after_mechanical_stabilization_before_robot_matrix",
        )
        self.assertFalse(config["mechanics"]["robot_results_used_for_selection"])
        oracle = config["surface_displacement_oracle"]
        self.assertEqual(oracle["spread_threshold_m"], SURFACE_SPREAD_THRESHOLD_M)
        self.assertEqual(
            oracle["persistence_ms"], SURFACE_SPREAD_PERSISTENCE_SAMPLES
        )
        runs = config["runs"]
        self.assertEqual(len(runs), 32)
        self.assertEqual(
            sum(run["group"] == "rigid_benign" for run in runs), 6
        )
        self.assertEqual(
            sum(run["group"] == "balanced_benign" for run in runs), 8
        )
        self.assertEqual(
            sum(run["group"] == "primary_uneven" for run in runs), 12
        )
        self.assertEqual(
            sum(run["group"] == "outcome_diversity" for run in runs), 6
        )

    def test_uneven_support_geometry_profiles_and_no_step_or_hole(self) -> None:
        self.assertEqual(
            SINK_SUPPORT_PATTERNS,
            (
                "balanced_soft",
                "medial_soft",
                "lateral_soft",
                "localized_soft",
            ),
        )
        for side in ("left", "right"):
            for pattern in SINK_SUPPORT_PATTERNS[1:]:
                with self.subTest(side=side, pattern=pattern):
                    model, ground_ids = load_g1_model(
                        "sand",
                        f"transition_{side}",
                        "severe",
                        sink_support_pattern=pattern,
                    )
                    enabled_names = {model.geom(geom_id).name for geom_id in ground_ids}
                    cells = tuple(
                        f"terrain_uneven_{side}_{segment}_{region}"
                        for segment in ("entry", "exit")
                        for region in ("medial", "lateral")
                    )
                    self.assertTrue(set(cells).issubset(enabled_names))
                    self.assertNotIn(f"terrain_transition_{side}", enabled_names)
                    for geom_id in ground_ids:
                        self.assertEqual(
                            float(
                                model.geom_pos[geom_id, 2]
                                + model.geom_size[geom_id, 2]
                            ),
                            0.0,
                        )
                    entry = model.geom(cells[0]).id
                    exit_cell = model.geom(cells[2]).id
                    self.assertAlmostEqual(
                        float(model.geom_pos[entry, 0] - model.geom_size[entry, 0]),
                        TRANSITION_PATCH_START_X_M,
                    )
                    self.assertAlmostEqual(
                        float(model.geom_pos[entry, 0] + model.geom_size[entry, 0]),
                        float(
                            model.geom_pos[exit_cell, 0]
                            - model.geom_size[exit_cell, 0]
                        ),
                    )
                    self.assertAlmostEqual(
                        float(
                            model.geom_pos[exit_cell, 0]
                            + model.geom_size[exit_cell, 0]
                        ),
                        TRANSITION_PATCH_END_X_M,
                    )
                    for segment in ("entry", "exit"):
                        medial = model.geom(
                            f"terrain_uneven_{side}_{segment}_medial"
                        ).id
                        lateral = model.geom(
                            f"terrain_uneven_{side}_{segment}_lateral"
                        ).id
                        medial_bounds = sorted(
                            (
                                float(
                                    model.geom_pos[medial, 1]
                                    - model.geom_size[medial, 1]
                                ),
                                float(
                                    model.geom_pos[medial, 1]
                                    + model.geom_size[medial, 1]
                                ),
                            )
                        )
                        lateral_bounds = sorted(
                            (
                                float(
                                    model.geom_pos[lateral, 1]
                                    - model.geom_size[lateral, 1]
                                ),
                                float(
                                    model.geom_pos[lateral, 1]
                                    + model.geom_size[lateral, 1]
                                ),
                            )
                        )
                        self.assertAlmostEqual(
                            min(
                                abs(medial_bounds[0] - lateral_bounds[1]),
                                abs(medial_bounds[1] - lateral_bounds[0]),
                            ),
                            0.0,
                        )
                    severe_cells = {
                        "medial_soft": {"entry_medial", "exit_medial"},
                        "lateral_soft": {"entry_lateral", "exit_lateral"},
                        "localized_soft": {"entry_medial"},
                    }[pattern]
                    for name in cells:
                        suffix = name.removeprefix(f"terrain_uneven_{side}_")
                        expected = SINK_SEVERITY_PROFILES[
                            "severe" if suffix in severe_cells else "moderate"
                        ]
                        geom_id = model.geom(name).id
                        np.testing.assert_array_equal(
                            model.geom_solref[geom_id], expected.solref
                        )
                        np.testing.assert_array_equal(
                            model.geom_solimp[geom_id], expected.solimp
                        )
        balanced, _ = load_g1_model("sand", "transition_left", "severe")
        explicit, _ = load_g1_model(
            "sand",
            "transition_left",
            "severe",
            sink_support_pattern="balanced_soft",
        )
        for name in TRANSITION_GROUND_GEOM_NAMES:
            np.testing.assert_array_equal(
                balanced.geom_solref[balanced.geom(name).id],
                explicit.geom_solref[explicit.geom(name).id],
            )
            np.testing.assert_array_equal(
                balanced.geom_solimp[balanced.geom(name).id],
                explicit.geom_solimp[explicit.geom(name).id],
            )

    def test_exact_quadrant_penetration_mapping_and_load_sum(self) -> None:
        model, ground_ids = load_g1_model("concrete")
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        data.qpos[2] = 0.78
        mujoco.mj_forward(model, data)
        exact = read_exact_foot_sample(model, data, ground_ids)
        self.assertEqual(exact.quadrant_contact.shape, (2, 4))
        self.assertEqual(exact.quadrant_normal_force_n.shape, (2, 4))
        self.assertEqual(exact.quadrant_penetration_m.shape, (2, 4))
        self.assertTrue(np.all(exact.quadrant_contact))
        np.testing.assert_allclose(
            exact.quadrant_normal_force_n.sum(axis=1),
            exact.normal_force_n,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            np.nanmax(exact.quadrant_penetration_m, axis=1),
            exact.contact_penetration_m,
        )
        for side in ("left", "right"):
            body = model.body(f"{side}_ankle_roll_link").id
            expected_quadrants = set()
            for index in range(1, 5):
                geom = model.geom(f"{side}_foot_contact_{index}").id
                world_delta = data.geom_xpos[geom] - data.xpos[body]
                local = data.xmat[body].reshape(3, 3).T @ world_delta
                expected_quadrants.add(fsr_quadrant_index(local[0], local[1]))
            self.assertEqual(expected_quadrants, {0, 1, 2, 3})

    def test_loaded_only_support_spread_and_causal_persistence(self) -> None:
        samples = 40
        contact = np.ones((samples, 2, 4), dtype=bool)
        load = np.full((samples, 2, 4), 10.0)
        penetration = np.tile(
            np.asarray((0.001, 0.002, 0.004, 0.005)),
            (samples, 2, 1),
        )
        loaded = np.ones((samples, 2), dtype=bool)
        episodes = np.zeros((samples, 2), dtype=np.int32)
        pre_fall = np.ones(samples, dtype=bool)
        support = support_penetration_diagnostics(
            contact, load, penetration, loaded, episodes, pre_fall
        )
        self.assertTrue(
            np.isnan(
                support["support_penetration_spread_m"][
                    :TOUCHDOWN_TRANSIENT_SAMPLES
                ]
            ).all()
        )
        np.testing.assert_allclose(
            support["support_penetration_spread_m"][TOUCHDOWN_TRANSIENT_SAMPLES:],
            0.004,
        )
        unloaded = loaded.copy()
        unloaded[20, 0] = False
        invalid = support_penetration_diagnostics(
            contact, load, penetration, unloaded, episodes, pre_fall
        )
        self.assertTrue(
            np.isnan(invalid["support_penetration_spread_m"][20, 0])
        )

        spread = np.zeros((40, 2), dtype=float)
        valid = np.zeros((40, 2), dtype=bool)
        spread[10:, 0] = 0.006
        valid[10:, 0] = True
        active, onset = uneven_support_oracle(
            spread, valid, episodes, threshold_m=0.005, persistence_samples=20
        )
        self.assertEqual(np.flatnonzero(onset[:, 0]).tolist(), [29])
        self.assertFalse(active[:29, 0].any())
        future_changed = spread.copy()
        future_changed[30:, 0] = 0.0
        _, changed_onset = uneven_support_oracle(
            future_changed,
            valid,
            episodes,
            threshold_m=0.005,
            persistence_samples=20,
        )
        np.testing.assert_array_equal(changed_onset[:30], onset[:30])

    def test_support_loss_baseline_retention_weighting_and_persistence(self) -> None:
        samples = 70
        contact = np.ones((samples, 2, 4), dtype=bool)
        load = np.full((samples, 2, 4), 10.0)
        loaded = np.ones((samples, 2), dtype=bool)
        episodes = np.zeros((samples, 2), dtype=np.int32)
        pre_fall = np.ones(samples, dtype=bool)
        load[30:, 0, (1, 3)] = 0.0
        diagnostics = support_loss_diagnostics(
            contact, load, loaded, episodes, pre_fall
        )

        self.assertFalse(
            diagnostics["support_baseline_established"][:29, 0].any()
        )
        self.assertTrue(diagnostics["support_baseline_onset"][29, 0])
        np.testing.assert_array_equal(
            diagnostics["support_baseline_mask"][29, 0],
            np.ones(4, dtype=bool),
        )
        self.assertEqual(
            diagnostics["baseline_supported_quadrant_count"][29, 0], 4
        )
        self.assertEqual(
            diagnostics["support_retained_quadrant_count"][30, 0], 2
        )
        self.assertEqual(diagnostics["support_retention_ratio"][30, 0], 0.5)
        self.assertEqual(diagnostics["support_loss_ratio"][30, 0], 0.5)
        self.assertEqual(diagnostics["weighted_support_loss"][30, 0], 0.5)
        self.assertFalse(diagnostics["support_loss_active"][:49, 0].any())
        self.assertTrue(diagnostics["support_loss_onset"][49, 0])

    def test_support_loss_presence_boundary_reset_and_toe_off_gate(self) -> None:
        samples = 90
        contact = np.ones((samples, 2, 4), dtype=bool)
        load = np.full((samples, 2, 4), 10.0)
        loaded = np.ones((samples, 2), dtype=bool)
        episodes = np.zeros((samples, 2), dtype=np.int32)
        pre_fall = np.ones(samples, dtype=bool)
        # Baseline samples are 10:30. Exactly 10 supported samples are retained;
        # nine are excluded by the predeclared >=50% presence aggregation.
        load[10:20, 0, 2] = 0.0
        load[10:21, 0, 3] = 0.0
        diagnostics = support_loss_diagnostics(
            contact, load, loaded, episodes, pre_fall
        )
        np.testing.assert_array_equal(
            diagnostics["support_baseline_mask"][29, 0],
            np.asarray((True, True, True, False)),
        )
        self.assertEqual(
            diagnostics["baseline_supported_quadrant_count"][29, 0], 3
        )

        loaded[35, 0] = False
        reset = support_loss_diagnostics(
            contact, load, loaded, episodes, pre_fall
        )
        self.assertFalse(reset["support_baseline_established"][35, 0])
        self.assertFalse(reset["support_loss_active"][35, 0])
        self.assertTrue(reset["support_baseline_onset"][55, 0])

        toe_off_load = np.full((samples, 2, 4), 10.0)
        toe_off_load[30:, 0] = np.asarray((2.6, 2.6, 0.0, 0.0))
        toe_off = support_loss_diagnostics(
            contact, toe_off_load, np.ones_like(loaded), episodes, pre_fall
        )
        self.assertEqual(toe_off["support_loss_ratio"][30, 0], 0.5)
        self.assertFalse(toe_off["support_loss_valid"][30:, 0].any())
        self.assertFalse(toe_off["support_loss_active"][:, 0].any())

        next_episode = episodes.copy()
        next_episode[50:, 0] = 1
        changed = support_loss_diagnostics(
            contact, load, np.ones_like(loaded), next_episode, pre_fall
        )
        self.assertFalse(changed["support_baseline_established"][50:79, 0].any())
        self.assertTrue(changed["support_baseline_onset"][79, 0])

    def test_support_loss_is_causal_and_censored_at_fall(self) -> None:
        samples = 70
        contact = np.ones((samples, 2, 4), dtype=bool)
        load = np.full((samples, 2, 4), 10.0)
        load[30:, 0, (0, 2)] = 0.0
        loaded = np.ones((samples, 2), dtype=bool)
        episodes = np.zeros((samples, 2), dtype=np.int32)
        pre_fall = np.ones(samples, dtype=bool)
        original = support_loss_diagnostics(
            contact, load, loaded, episodes, pre_fall
        )
        future_load = load.copy()
        future_load[55:, 0] = 10.0
        changed = support_loss_diagnostics(
            contact, future_load, loaded, episodes, pre_fall
        )
        np.testing.assert_array_equal(
            original["support_loss_onset"][:55],
            changed["support_loss_onset"][:55],
        )
        pre_fall[45:] = False
        censored = support_loss_diagnostics(
            contact, load, loaded, episodes, pre_fall
        )
        self.assertFalse(censored["support_loss_active"].any())
        self.assertFalse(censored["support_baseline_established"][45:].any())

    def test_sink_physical_redefinition_config_is_bounded(self) -> None:
        with SINK_PHYSICAL_REDEFINITION_CONFIG.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(
            config["experiment"]["id"], "SINK_PHYSICAL_HAZARD_REDEFINITION"
        )
        runs = config["runs"]
        self.assertEqual(len(runs), 26)
        self.assertEqual(sum(run["role"] == "benign" for run in runs), 14)
        self.assertEqual(sum(run["role"] == "uneven" for run in runs), 12)
        self.assertEqual(len({run["id"] for run in runs}), 26)
        self.assertEqual(
            config["support_metric"]["freeze_status"],
            "criterion_not_freezable",
        )
        self.assertFalse(config["support_metric"]["future_outcome_dependency"])

    def test_support_loss_config_freezes_candidate_and_acceptance(self) -> None:
        with SINK_SUPPORT_LOSS_CONFIG.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        self.assertEqual(
            config["experiment"]["id"], "SINK_SUPPORT_LOSS_ORACLE_SANITY"
        )
        oracle = config["support_loss_oracle"]
        self.assertEqual(oracle["quadrant_load_cutoff_n"], LOAD_OFF_N)
        self.assertEqual(oracle["touchdown_transient_ms"], TOUCHDOWN_TRANSIENT_SAMPLES)
        self.assertEqual(oracle["baseline_window_ms"], SUPPORT_BASELINE_SAMPLES)
        self.assertEqual(
            oracle["baseline_presence_ratio"], SUPPORT_BASELINE_PRESENCE_RATIO
        )
        self.assertEqual(
            oracle["minimum_baseline_supported_quadrants"],
            SUPPORT_BASELINE_MIN_QUADRANTS,
        )
        self.assertEqual(
            oracle["support_loss_ratio_threshold"], SUPPORT_LOSS_THRESHOLD_RATIO
        )
        self.assertEqual(
            oracle["persistence_ms"], SUPPORT_LOSS_PERSISTENCE_SAMPLES
        )
        self.assertEqual(
            oracle["current_total_load_minimum_baseline_ratio"],
            SUPPORT_TOTAL_LOAD_MIN_RATIO,
        )
        runs = config["runs"]
        self.assertEqual(len(runs), 26)
        self.assertEqual(sum(run["role"] == "benign" for run in runs), 14)
        self.assertEqual(sum(run["role"] == "uneven" for run in runs), 12)
        with SINK_PHYSICAL_REDEFINITION_CONFIG.open(
            "r", encoding="utf-8"
        ) as stream:
            previous = yaml.safe_load(stream)
        self.assertEqual(runs, previous["runs"])
        self.assertEqual(config["provisional_acceptance"]["uneven_detection_min"], 9)
        self.assertEqual(oracle["freeze_status"], "criterion_not_freezable")

    @unittest.skipUnless(LOCAL_POLICY.is_file(), "local verified policy is absent")
    def test_balanced_medial_and_lateral_support_diagnostics_smoke(self) -> None:
        with SINK_PHYSICAL_REDEFINITION_CONFIG.open("r", encoding="utf-8") as stream:
            experiment = yaml.safe_load(stream)
        base = load_simulation_config(SIMULATOR_CONFIG)
        threshold = float(experiment["support_metric"]["candidate_threshold_m"])
        cases = (
            ("balanced_soft", "moderate", False, False),
            ("medial_soft", "severe", True, True),
            ("lateral_soft", "severe", False, True),
        )
        for pattern, severity, old_spread_expected, fall_expected in cases:
            with self.subTest(pattern=pattern):
                result = run_simulation(
                    replace(
                        base,
                        duration_s=8.0,
                        command_speed_mps=0.15,
                        policy_path=LOCAL_POLICY,
                        terrain="sand",
                        sink_pattern="transition_right",
                        sink_severity=severity,
                        sink_support_pattern=pattern,
                        headless=True,
                    )
                )
                spread = result.diagnostics.support_penetration_spread_m
                active, _ = uneven_support_oracle(
                    spread,
                    np.isfinite(spread),
                    result.diagnostics.contact_episode_id,
                    threshold,
                    20,
                )
                self.assertEqual(bool(np.any(active)), old_spread_expected)
                self.assertTrue(np.any(result.diagnostics.support_baseline_onset))
                self.assertTrue(
                    np.any(np.isfinite(result.diagnostics.support_loss_ratio))
                )
                self.assertEqual(
                    result.metadata["first_fall_sample"] is not None,
                    fall_expected,
                )
                self.assertEqual(result.metadata["dropped_samples"], 0)

    @unittest.skipUnless(LOCAL_POLICY.is_file(), "local verified policy is absent")
    def test_deformable_support_runtime_and_viewer_physics_parity(self) -> None:
        base = load_simulation_config(SIMULATOR_CONFIG)
        config = replace(
            base,
            duration_s=2.2,
            command_speed_mps=0.15,
            policy_path=LOCAL_POLICY,
            terrain="sand",
            sink_pattern="transition_left",
            sink_severity="moderate",
            sink_support_pattern="medial_deformable",
            headless=True,
        )
        headless = run_simulation(config)
        self.assertEqual(headless.runtime.pelvis_imu.shape, (2200, 6))
        self.assertEqual(headless.runtime.foot_fsr.shape, (2200, 8))
        np.testing.assert_array_equal(
            np.diff(headless.runtime.timestamp_us), np.full(2199, 1000)
        )
        self.assertEqual(headless.metadata["dropped_samples"], 0)
        displacement = headless.diagnostics.support_surface_displacement_m
        self.assertEqual(displacement.shape, (2200, 2, 4))
        d0 = np.flatnonzero(
            headless.diagnostics.soft_patch_contact_onset[:, 0]
        )
        self.assertTrue(d0.size)
        np.testing.assert_allclose(displacement[: d0[0], 0], 0.0, atol=1.0e-12)
        self.assertGreater(float(np.max(displacement[d0[0] :, 0])), 0.0)

        fake_viewer = FakeViewer()
        with mock.patch(
            "fastreflex.simulation.g1.launch_passive_viewer",
            return_value=fake_viewer,
        ), mock.patch("fastreflex.simulation.g1._pace_viewer"):
            viewer = run_simulation(replace(config, headless=False))
        self.assertGreater(fake_viewer.sync_count, 1)
        for field in RuntimeTrace.__dataclass_fields__:
            np.testing.assert_equal(
                getattr(headless.runtime, field), getattr(viewer.runtime, field)
            )
        for field in type(headless.diagnostics).__dataclass_fields__:
            np.testing.assert_equal(
                getattr(headless.diagnostics, field),
                getattr(viewer.diagnostics, field),
            )

    def test_virtual_fsr_channel_order_quadrants_and_contact_force_sum(self) -> None:
        self.assertEqual(
            FSR_CHANNELS,
            (
                "left_front_left", "left_front_right",
                "left_rear_left", "left_rear_right",
                "right_front_left", "right_front_right",
                "right_rear_left", "right_rear_right",
            ),
        )
        self.assertEqual(fsr_quadrant_index(0.0, 0.0), 0)
        self.assertEqual(fsr_quadrant_index(0.1, -0.1), 1)
        self.assertEqual(fsr_quadrant_index(-0.1, 0.1), 2)
        self.assertEqual(fsr_quadrant_index(-0.1, -0.1), 3)

        model, ground_ids = load_g1_model("concrete")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        np.testing.assert_array_equal(read_virtual_fsr(model, data, ground_ids), np.zeros(8))

        data.qpos[:] = model.qpos0
        data.qpos[2] = 0.78
        mujoco.mj_forward(model, data)
        before = {
            "qpos": data.qpos.copy(), "qvel": data.qvel.copy(),
            "ctrl": data.ctrl.copy(), "time": float(data.time),
            "imu": read_pelvis_imu(model, data).copy(),
        }
        fsr = read_virtual_fsr(model, data, ground_ids)
        exact = read_exact_foot_sample(model, data, ground_ids)
        self.assertEqual(fsr.dtype, np.float32)
        self.assertTrue(np.all(fsr >= 0.0))
        self.assertTrue(np.all(fsr > 0.0))
        np.testing.assert_allclose(
            fsr.reshape(2, 4).sum(axis=1),
            exact.normal_force_n,
            rtol=1.0e-6,
            atol=1.0e-3,
        )
        np.testing.assert_array_equal(data.qpos, before["qpos"])
        np.testing.assert_array_equal(data.qvel, before["qvel"])
        np.testing.assert_array_equal(data.ctrl, before["ctrl"])
        self.assertEqual(float(data.time), before["time"])
        np.testing.assert_array_equal(read_pelvis_imu(model, data), before["imu"])

    def test_config_model_and_pelvis_imu_contract(self) -> None:
        config = load_simulation_config(SIMULATOR_CONFIG)
        self.assertEqual(config.physics_timestep_s, 0.0005)
        self.assertEqual(config.sensor_rate_hz, 1000)
        self.assertEqual(config.physics_steps_per_sample, 2)
        self.assertIsNone(config.policy_path)
        self.assertEqual(config.slip_pattern, "uniform")
        self.assertEqual(config.sink_pattern, "uniform")
        self.assertEqual(config.sink_severity, "moderate")
        self.assertEqual(config.sink_support_pattern, "balanced_soft")
        self.assertEqual(config.patch_start_x_m, TRANSITION_PATCH_START_X_M)
        self.assertEqual(config.patch_width_m, TRANSITION_PATCH_WIDTH_M)

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

        shifted, _ = load_g1_model(
            "ice",
            slip_pattern="transition",
            patch_start_x_m=0.40,
            patch_width_m=0.75,
        )
        shifted_left = shifted.geom("terrain_transition_left").id
        shifted_pre = shifted.geom("terrain_transition_pre").id
        shifted_post = shifted.geom("terrain_transition_post").id
        self.assertAlmostEqual(
            float(
                shifted.geom_pos[shifted_pre, 0]
                + shifted.geom_size[shifted_pre, 0]
            ),
            0.40,
        )
        self.assertAlmostEqual(
            float(
                shifted.geom_pos[shifted_left, 0]
                - shifted.geom_size[shifted_left, 0]
            ),
            0.40,
        )
        self.assertAlmostEqual(
            float(
                shifted.geom_pos[shifted_left, 0]
                + shifted.geom_size[shifted_left, 0]
            ),
            1.15,
        )
        self.assertAlmostEqual(
            float(
                shifted.geom_pos[shifted_post, 0]
                - shifted.geom_size[shifted_post, 0]
            ),
            1.15,
        )

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
            [
                "simulate",
                "--terrain",
                "sand",
                "--sink-pattern",
                "transition_left",
                "--sink-support-pattern",
                "medial_soft",
            ]
        )
        self.assertEqual(transition_args.sink_pattern, "transition_left")
        self.assertEqual(transition_args.sink_support_pattern, "medial_soft")
        deformable_args = parser.parse_args(
            [
                "simulate",
                "--terrain",
                "sand",
                "--sink-pattern",
                "transition_right",
                "--sink-support-pattern",
                "lateral_deformable",
            ]
        )
        self.assertEqual(
            deformable_args.sink_support_pattern, "lateral_deformable"
        )
        slip_transition_args = parser.parse_args(
            ["simulate", "--terrain", "ice", "--slip-pattern", "transition"]
        )
        self.assertEqual(slip_transition_args.slip_pattern, "transition")
        shifted_args = parser.parse_args(
            [
                "simulate",
                "--terrain",
                "ice",
                "--slip-pattern",
                "transition",
                "--patch-start-x",
                "0.40",
                "--patch-width",
                "0.75",
            ]
        )
        self.assertEqual(shifted_args.patch_start_x, 0.40)
        self.assertEqual(shifted_args.patch_width, 0.75)
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
            ("sequence", "timestamp_us", "pelvis_imu", "foot_fsr", "foot_imu"),
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
