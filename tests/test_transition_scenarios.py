"""Contracts for transition prefix parity and frozen scenario calibration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import yaml

from fastreflex.evaluation.transition_scenarios import (
    _assert_unique_and_disjoint,
    _canonical_hash,
    _fusion_regression,
    _simulation_config,
    classify_scenario_outcome,
    compare_prefix_pair,
    construct_matched_reference,
    geometry_contact_audit,
    target_contact_mask,
)
from fastreflex.simulation.g1 import load_g1_model, load_simulation_config, run_simulation
from fastreflex.simulation.terrain import TERRAIN_PROFILES
from scripts.fastreflex import build_parser


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "experiment"
    / "20260828_transition_scenario_calibration.yaml"
)
SIMULATOR_CONFIG = ROOT / "configs" / "simulator" / "g1.yaml"
LOCAL_POLICY = (
    ROOT
    / "artifacts"
    / "external"
    / "unitree_g1"
    / "g1_velocity_policy.onnx"
)


def _document() -> dict[str, object]:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


class TransitionScenarioStaticTest(unittest.TestCase):
    def test_target_geometry_starts_at_boundary_without_step_or_hole(self) -> None:
        document = _document()
        for specification in document["prefix_parity"]["matched_pairs"]:
            with self.subTest(pair=specification["id"]):
                audit = geometry_contact_audit(specification)
                self.assertTrue(audit["passed"])
                self.assertTrue(audit["target_does_not_extend_before_boundary"])
                self.assertTrue(audit["same_height_no_step"])
                self.assertTrue(audit["complete_ground_no_hole"])

    def test_marble_transition_applies_marble_only_to_hard_prefix(self) -> None:
        model, _ = load_g1_model(
            "ice",
            slip_pattern="transition",
            source_terrain="marble",
        )
        pre = model.geom("terrain_transition_pre").id
        target = model.geom("terrain_transition_left").id
        np.testing.assert_array_equal(
            model.geom_friction[pre], TERRAIN_PROFILES["marble"].friction
        )
        np.testing.assert_array_equal(
            model.geom_friction[target], TERRAIN_PROFILES["ice"].friction
        )

    def test_cli_exposes_transition_source_terrain(self) -> None:
        args = build_parser().parse_args(
            [
                "simulate",
                "--source-terrain",
                "marble",
                "--terrain",
                "ice",
                "--slip-pattern",
                "transition",
            ]
        )
        self.assertEqual(args.source_terrain, "marble")

    def test_calibration_validation_disjoint_and_frozen_domains(self) -> None:
        document = _document()
        _assert_unique_and_disjoint(document)
        frozen_before = _canonical_hash(document["frozen_operating_points"])
        changed = deepcopy(document)
        changed["fresh_validation"]["runs"][0]["intended_role"] = "fall"
        self.assertEqual(
            frozen_before, _canonical_hash(changed["frozen_operating_points"])
        )
        escaped = deepcopy(document)
        escaped["fresh_validation"]["runs"][0]["patch_width_m"] = 0.79
        with self.assertRaisesRegex(ValueError, "escapes frozen"):
            _assert_unique_and_disjoint(escaped)

    def test_marble_reference_changes_only_source_terrain(self) -> None:
        document = _document()
        pair = document["prefix_parity"]["matched_pairs"][2]
        reference = construct_matched_reference(pair, 8.0, 0.75)
        self.assertEqual(reference["source_terrain"], "marble")
        self.assertEqual(reference["target_terrain"], pair["target_terrain"])
        self.assertEqual(reference["patch_start_x_m"], 8.0)
        for field in (
            "speed_mps",
            "slip_pattern",
            "sink_pattern",
            "sink_severity",
            "support_pattern",
        ):
            self.assertEqual(reference[field], pair[field])

    def test_fusion_truth_table_is_unchanged(self) -> None:
        audit = _fusion_regression()
        self.assertTrue(audit["passed"])
        self.assertFalse(audit["logic_changed"])


@unittest.skipUnless(LOCAL_POLICY.is_file(), "local verified policy is absent")
class TransitionScenarioRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = _document()
        cls.base = load_simulation_config(SIMULATOR_CONFIG)
        cls.results = {}
        for specification in cls.document["prefix_parity"]["matched_pairs"][:2]:
            transition_spec = dict(specification)
            reference_spec = construct_matched_reference(
                transition_spec, 8.0, 0.75
            )
            transition = run_simulation(
                _simulation_config(
                    cls.base, transition_spec, LOCAL_POLICY, duration_s=3.0
                ),
                capture_state_trace=True,
            )
            reference = run_simulation(
                _simulation_config(
                    cls.base, reference_spec, LOCAL_POLICY, duration_s=3.0
                ),
                capture_state_trace=True,
            )
            cls.results[str(specification["id"])] = (
                transition_spec,
                transition,
                reference,
            )

    def test_ice_and_sand_precontact_robot_controller_parity(self) -> None:
        tolerances = self.document["prefix_parity"]["tolerances"]
        for pair_id, (specification, transition, reference) in self.results.items():
            with self.subTest(pair=pair_id):
                row = compare_prefix_pair(
                    transition, reference, specification, tolerances
                )
                self.assertEqual(row["verdict"], "TRANSITION_PREFIX_PARITY_PASS")
                self.assertLessEqual(row["max_qpos_abs_diff"], 1.0e-12)
                self.assertLessEqual(row["max_qvel_abs_diff"], 1.0e-12)
                self.assertEqual(row["max_imu_abs_diff"], 0.0)
                self.assertEqual(row["max_controller_action_abs_diff"], 0.0)
                self.assertEqual(row["contact_mismatch_count"], 0)
                self.assertEqual(row["policy_update_mismatch_count"], 0)
                self.assertFalse(row["pretarget_dynamic_support_contact"])
                self.assertFalse(row["pretarget_fall"])

    def test_target_contact_detection_and_matched_initial_state(self) -> None:
        for pair_id, (specification, transition, reference) in self.results.items():
            with self.subTest(pair=pair_id):
                self.assertTrue(
                    np.any(
                        target_contact_mask(
                            transition, str(specification["target_terrain"])
                        )
                    )
                )
                self.assertFalse(
                    np.any(
                        target_contact_mask(
                            reference, str(specification["target_terrain"])
                        )
                    )
                )
                np.testing.assert_allclose(
                    transition.state_trace.robot_qpos[0],
                    reference.state_trace.robot_qpos[0],
                    atol=1.0e-12,
                    rtol=0.0,
                )
                np.testing.assert_allclose(
                    transition.state_trace.robot_qvel[0],
                    reference.state_trace.robot_qvel[0],
                    atol=1.0e-12,
                    rtol=0.0,
                )

    def test_observed_outcome_does_not_depend_on_intended_label(self) -> None:
        specification, transition, _ = self.results["concrete_ice_prefix"]
        stable_name = dict(specification, intended_role="stable")
        fall_name = dict(specification, intended_role="fall")
        self.assertEqual(
            classify_scenario_outcome(transition, stable_name),
            classify_scenario_outcome(transition, fall_name),
        )

    def test_state_capture_does_not_change_physics(self) -> None:
        specification, captured, _ = self.results["concrete_ice_prefix"]
        plain = run_simulation(
            _simulation_config(self.base, specification, LOCAL_POLICY, duration_s=3.0)
        )
        np.testing.assert_array_equal(captured.runtime.pelvis_imu, plain.runtime.pelvis_imu)
        np.testing.assert_array_equal(
            captured.diagnostics.physical_contact, plain.diagnostics.physical_contact
        )


if __name__ == "__main__":
    unittest.main()
