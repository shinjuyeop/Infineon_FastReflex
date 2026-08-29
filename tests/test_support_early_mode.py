"""Contracts for Support early-mode resolution."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import yaml

from fastreflex.evaluation.reflex_event import EventHoldoutGuard, EventRun
from fastreflex.evaluation.support_early_mode import (
    IncipientFit,
    classify_early_mode,
    fit_incipient_candidate,
    fit_phase_physical_envelope,
    incipient_score,
    mine_support_hard_negatives,
    persistent_onset,
    phase_ids,
    support_threshold_values,
    touchdown_samples,
    waveform_similarity,
)
from fastreflex.evaluation.stability_temporal import _file_sha256
from fastreflex.evaluation.terrain_conditioned_reflex import (
    feature_schema_for_components,
)


def synthetic_run(
    run_id: str,
    *,
    split: str = "train",
    support_event: int | None = None,
    spread: np.ndarray | None = None,
    outcome: str = "VALID_STABLE",
    target: str = "sand",
) -> EventRun:
    samples = 240
    imu = np.zeros((samples, 6), dtype=np.float32)
    fsr = np.ones((samples, 8), dtype=np.float32)
    loaded = np.zeros((samples, 2), dtype=bool)
    for start in (20, 120, 220):
        loaded[start : min(samples, start + 40), 0] = True
    for start in (70, 170):
        loaded[start : start + 40, 1] = True
    support_spread = (
        np.zeros((samples, 2), dtype=np.float32)
        if spread is None
        else np.asarray(spread, dtype=np.float32)
    )
    zeros = np.zeros((samples, 2), dtype=np.float32)
    return EventRun(
        run_id=run_id,
        split=split,
        source_terrain="concrete",
        target_terrain=target,
        design_role="test",
        first_contact_sample=20,
        first_touchdown_sample=20,
        censor_sample=samples,
        outcome_diagnostic=outcome,
        fall_sample_diagnostic=None,
        features={
            "PELVIS_IMU6": imu,
            "PELVIS_IMU6_FSR8": np.concatenate((imu, fsr), axis=1),
        },
        timestamp_us=np.arange(samples, dtype=np.int64) * 1000,
        slip_event_samples_per_foot=(None, None),
        support_event_samples_per_foot=(support_event, None),
        event_sample=support_event,
        event_type="SUPPORT" if support_event is not None else "NONE",
        hard_stable_control=False,
        drift_m=zeros,
        tangential_velocity_mps=zeros,
        support_spread_m=support_spread,
        support_max_displacement_m=zeros,
        loaded_contact=loaded,
        sink_pattern="transition_left",
        support_pattern="lateral_deformable",
    )


class SupportEarlyModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.config = yaml.safe_load(
            (
                cls.root
                / "configs/experiment/20260829_support_early_mode_resolution.yaml"
            ).read_text(encoding="utf-8")
        )

    @staticmethod
    def with_all_phases(run: EventRun) -> EventRun:
        loaded = run.loaded_contact.copy()
        loaded[30:35, 1] = True
        return replace(run, loaded_contact=loaded)

    def test_touchdown_and_gait_period_are_deterministic(self) -> None:
        run = synthetic_run("gait")
        left, right = touchdown_samples(
            run.loaded_contact, minimum_same_foot_separation_ms=30
        )
        self.assertEqual(left.tolist(), [20, 120, 220])
        self.assertEqual(right.tolist(), [70, 170])
        self.assertEqual(int(left[1] - left[0]), 100)

    def test_phase_mapping(self) -> None:
        loaded = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=bool)
        self.assertEqual(phase_ids(loaded).tolist(), [0, 1, 2, 3])

    def test_waveform_similarity_is_deterministic(self) -> None:
        values = np.arange(24, dtype=float).reshape(4, 6)
        first = waveform_similarity(values, values)
        second = waveform_similarity(values, values)
        self.assertEqual(first, second)
        self.assertAlmostEqual(float(first["normalized_l2"]), 0.0)
        self.assertAlmostEqual(float(first["cosine_similarity"]), 1.0)

    def test_privileged_physics_absent_from_runtime_schema(self) -> None:
        schema = feature_schema_for_components(("pelvis_imu6",))
        self.assertEqual(len(schema), 60)
        self.assertFalse(any("spread" in value or "terrain" in value for value in schema))

    def test_train_only_phase_envelope_excludes_validation_and_support(self) -> None:
        benign = self.with_all_phases(synthetic_run("train_benign"))
        validation_spread = np.ones((240, 2), dtype=np.float32)
        validation = synthetic_run("validation", split="validation", spread=validation_spread)
        event = synthetic_run("event", support_event=180, spread=validation_spread)
        result = fit_phase_physical_envelope(
            {run.run_id: run for run in (benign, validation, event)}
        )
        self.assertEqual(result["fit_run_ids"], ["train_benign"])
        self.assertTrue(result["validation_excluded"])
        self.assertFalse(result["fall_outcome_used"])
        self.assertEqual(result["quantile"], 0.995)

    def test_envelope_does_not_read_fall_outcome(self) -> None:
        stable = self.with_all_phases(
            synthetic_run("same", outcome="VALID_STABLE")
        )
        falling = replace(stable, outcome_diagnostic="VALID_FALL")
        first = fit_phase_physical_envelope({"same": stable})
        second = fit_phase_physical_envelope({"same": falling})
        self.assertEqual(first["phase_bounds"], second["phase_bounds"])

    def test_early_mode_classification(self) -> None:
        self.assertEqual(
            classify_early_mode(gait_alias=True, physical_precursor=False),
            "GAIT_ALIAS_FALSE_MODE",
        )
        self.assertEqual(
            classify_early_mode(gait_alias=False, physical_precursor=True),
            "PHYSICAL_PRECURSOR_MODE",
        )
        self.assertEqual(
            classify_early_mode(gait_alias=True, physical_precursor=True),
            "PHYSICAL_PRECURSOR_MODE",
        )
        self.assertEqual(
            classify_early_mode(gait_alias=False, physical_precursor=False),
            "MIXED_OR_UNRESOLVED",
        )

    def test_hnm_prioritizes_alias_and_respects_contract(self) -> None:
        endpoints = np.arange(0, 200, 10)
        scores = np.linspace(0.0, 1.0, len(endpoints))
        mined = mine_support_hard_negatives(
            endpoints,
            scores,
            positive_region=(80, 120),
            gait_alias_endpoints=(20,),
            top_k=16,
            minimum_separation_ms=30,
        )
        self.assertIn(20, mined)
        self.assertFalse(any(80 <= value <= 120 for value in mined))
        self.assertLessEqual(len(mined), 16)
        self.assertTrue(all(b - a >= 30 for a, b in zip(mined, mined[1:])))

    def test_threshold_grid_is_frozen(self) -> None:
        values = support_threshold_values()
        self.assertEqual(len(values), 50)
        self.assertEqual(values[0], 0.50)
        self.assertEqual(values[-1], 0.99)

    def test_incipient_fit_is_train_benign_only(self) -> None:
        benign = synthetic_run("benign")
        validation = synthetic_run(
            "validation", split="validation", spread=np.ones((240, 2))
        )
        event = synthetic_run(
            "event", support_event=180, spread=np.ones((240, 2))
        )
        fit = fit_incipient_candidate(
            "I1", {run.run_id: run for run in (benign, validation, event)}
        )
        self.assertEqual(fit.fit_run_ids, ("benign",))
        self.assertEqual(fit.quantile, 0.995)

    def test_incipient_score_uses_no_future_sample(self) -> None:
        spread = np.zeros((240, 2), dtype=np.float32)
        spread[100:, 0] = np.arange(140) * 0.001
        run = synthetic_run("causal", spread=spread)
        fit = IncipientFit("I2", 0.0, np.zeros(2), np.ones(2), ("x",), 1, 0.995)
        original = incipient_score(run, fit)
        changed = spread.copy()
        changed[180:, 0] += 99.0
        modified = incipient_score(replace(run, support_spread_m=changed), fit)
        np.testing.assert_array_equal(original[:180], modified[:180])

    def test_incipient_persistence_is_exactly_twenty_ms(self) -> None:
        score = np.zeros(80)
        score[30:50] = 1.0
        self.assertEqual(
            persistent_onset(
                score, 0.5, persistence_ms=20, first_sample=0, censor_sample=80
            ),
            49,
        )
        score[49] = 0.0
        self.assertIsNone(
            persistent_onset(
                score, 0.5, persistence_ms=20, first_sample=0, censor_sample=80
            )
        )

    def test_holdout_guard_opens_once(self) -> None:
        guard = EventHoldoutGuard()
        self.assertEqual(guard.open_count, 0)
        guard.open_once()
        self.assertEqual(guard.open_count, 1)
        with self.assertRaises(RuntimeError):
            guard.open_once()

    def test_config_freezes_oracle_hnm_and_persistence(self) -> None:
        support = self.config["frozen_support"]
        self.assertEqual(support["physical_oracle"], {
            "support_surface_spread_m": 0.010,
            "persistence_ms": 20,
        })
        self.assertEqual(support["history_ms"], 20)
        self.assertEqual(support["probability_threshold"], 0.94)
        self.assertEqual(support["persistence_ms"], 5)
        hnm = self.config["branch_a_retraining"]["hard_negative_mining"]
        self.assertEqual(hnm["rounds"], 3)
        self.assertEqual(hnm["top_k_per_run"], 16)
        self.assertEqual(hnm["minimum_separation_ms"], 30)
        self.assertTrue(hnm["prioritize_gait_alias"])

    def test_protected_hashes_and_canonical_cli_dispatch(self) -> None:
        declared = []
        support = self.config["frozen_support"]
        declared.append((support["normalizer"]["path"], support["normalizer"]["sha256"]))
        declared.extend((row["path"], row["sha256"]) for row in support["checkpoints"])
        terrain = self.config["protected"]["terrain"]
        declared.append((terrain["normalizer"]["path"], terrain["normalizer"]["sha256"]))
        declared.extend((row["path"], row["sha256"]) for row in terrain["checkpoints"])
        slip = self.config["protected"]["slip_freeze"]
        declared.append((slip["path"], slip["sha256"]))
        for relative, expected in declared:
            self.assertEqual(_file_sha256(self.root / relative), expected)
        cli = (self.root / "scripts/fastreflex.py").read_text(encoding="utf-8")
        self.assertIn('experiment_id == "SUPPORT_EARLY_MODE_RESOLUTION"', cli)

    def test_simulator_viewer_remains_physics_parity_only(self) -> None:
        simulation = (
            self.root / "src/fastreflex/simulation/g1.py"
        ).read_text(encoding="utf-8")
        self.assertIn("viewer.sync(state_only=True)", simulation)
        self.assertNotIn("support_early_mode", simulation)


if __name__ == "__main__":
    unittest.main()
