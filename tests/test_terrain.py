"""Current contract tests for advisory Terrain recognition."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from fastreflex.dataset.hazard import (
    EVENT_TYPE_NONE,
    PELVIS_IMU6,
    PELVIS_IMU6_FSR8,
    HazardRun,
)
from fastreflex.dataset.terrain import (
    build_touchdown_event_rows,
    load_terrain_collection_config,
    terrain_identity_touchdown,
)
from fastreflex.evaluation.terrain import (
    ICE,
    SAND,
    load_frozen_terrain_candidate,
    predict_terrain_window,
    refine_hazard_cause,
    replay_terrain,
    terrain_fsr4_window,
    terrain_predictions,
    verify_supported_terrain_candidate,
)
from fastreflex.simulation.g1 import RuntimeTrace, SimulationResult
from tests.support import REPOSITORY_ROOT as ROOT

CONFIG = ROOT / "configs/experiment/20260828_terrain_rebuild_sensor_ablation.yaml"


class _FixedModel(torch.nn.Module):
    def __init__(self, class_id: int) -> None:
        super().__init__()
        self.class_id = class_id

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((len(inputs), 4), dtype=torch.float32)
        logits[:, self.class_id] = 4.0
        return logits


def _run(samples: int = 100) -> HazardRun:
    imu = np.zeros((samples, 6), dtype=np.float32)
    fsr = np.arange(samples * 8, dtype=np.float32).reshape(samples, 8)
    zeros = np.zeros((samples, 2), dtype=np.float32)
    return HazardRun(
        run_id="terrain",
        split="validation",
        source_terrain="concrete",
        target_terrain="ice",
        design_role="stable",
        first_contact_sample=10,
        first_touchdown_sample=10,
        censor_sample=samples,
        outcome_diagnostic="VALID_STABLE",
        fall_sample_diagnostic=None,
        features={
            PELVIS_IMU6: imu,
            PELVIS_IMU6_FSR8: np.concatenate((imu, fsr), axis=1),
        },
        timestamp_us=np.arange(samples, dtype=np.int64) * 1000,
        slip_event_samples_per_foot=(None, None),
        support_event_samples_per_foot=(None, None),
        event_sample=None,
        event_type=EVENT_TYPE_NONE,
        hard_stable_control=False,
        drift_m=zeros,
        tangential_velocity_mps=zeros,
        support_spread_m=zeros,
        support_max_displacement_m=zeros,
        loaded_contact=np.zeros((samples, 2), dtype=bool),
        sink_pattern="uniform",
        support_pattern="balanced_soft",
    )


class TerrainTest(unittest.TestCase):
    def test_fsr4_window_is_exactly_touchdown_through_49ms(self) -> None:
        fsr = np.arange(100 * 8, dtype=np.float32).reshape(100, 8)
        left = terrain_fsr4_window(fsr, 10, "left")
        right = terrain_fsr4_window(fsr, 10, "right")
        self.assertEqual(left.shape, (50, 4))
        self.assertTrue(np.array_equal(left, fsr[10:60, :4]))
        self.assertTrue(np.array_equal(right, fsr[10:60, 4:]))

    def test_frozen_normalization_and_ensemble_output_are_exact(self) -> None:
        window = np.arange(200, dtype=np.float32).reshape(50, 4)
        models = [_FixedModel(2), _FixedModel(2), _FixedModel(2)]
        prediction, probability = predict_terrain_window(
            window,
            models,
            np.arange(4, dtype=np.float32),
            np.full(4, 2.0, dtype=np.float32),
        )
        self.assertEqual(prediction, 2)
        self.assertEqual(probability.dtype, np.float32)
        self.assertEqual(probability.shape, (4,))

    def test_frozen_candidate_probability_matches_preconsolidation_math(self) -> None:
        model_path = (
            ROOT
            / "artifacts/runs/20260828_terrain_rebuild_sensor_ablation"
            / "selected_models"
        )
        models, mean, std = load_frozen_terrain_candidate(model_path)
        window = (
            np.random.default_rng(20260831)
            .uniform(0.0, 120.0, size=(50, 4))
            .astype(np.float32)
        )
        prediction, probability = predict_terrain_window(window, models, mean, std)
        normalized = ((window - mean) / std).astype(np.float32)[None]
        tensor = torch.from_numpy(normalized)
        with torch.no_grad():
            legacy = np.mean(
                [
                    torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
                    for model in models
                ],
                axis=0,
            ).astype(np.float32)
        self.assertTrue(np.array_equal(probability, legacy))
        self.assertEqual(prediction, int(np.argmax(legacy)))

    def test_clean_touchdown_and_fifty_ms_held_state_timing(self) -> None:
        run = _run()
        contact = np.zeros((100, 2, 4), dtype=bool)
        contact[10:80, 0, 2] = True
        rows = build_touchdown_event_rows(
            "terrain",
            "validation",
            "concrete",
            "ice",
            run.timestamp_us,
            contact,
            None,
            False,
            False,
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["window_50ms_valid"])
        result = SimulationResult(
            runtime=RuntimeTrace(
                sequence=np.arange(100, dtype=np.int64),
                timestamp_us=run.timestamp_us,
                pelvis_imu=run.features[PELVIS_IMU6],
                foot_fsr=run.features[PELVIS_IMU6_FSR8][:, 6:],
            ),
            diagnostics=None,  # type: ignore[arg-type]
            metadata={},
            exact_terrain_contact=contact,
        )
        trace = replay_terrain(
            result,
            run,
            [_FixedModel(2)] * 3,
            np.zeros(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
        )
        self.assertTrue(np.all(trace.state[:60] == 0))
        self.assertTrue(np.all(trace.state[60:] == ICE))
        self.assertEqual(trace.update_samples.tolist(), [60])
        self.assertEqual(trace.first_target_valid_sample, 60)
        prediction = terrain_predictions(trace)[0]
        self.assertEqual(prediction.prediction_timestamp, 60)
        self.assertEqual(prediction.touchdown_foot, "LEFT")

    def test_touchdown_is_per_foot_and_per_class_and_prefix_causal(self) -> None:
        contact = np.zeros((80, 2, 4), dtype=bool)
        contact[10:70, 0, 0] = True
        contact[20:70, 0, 2] = True
        contact[30:70, 1, 3] = True
        onset = terrain_identity_touchdown(contact)
        self.assertEqual(
            np.argwhere(onset).tolist(), [[10, 0, 0], [20, 0, 2], [30, 1, 3]]
        )
        changed = contact.copy()
        changed[70:, :, :] = True
        self.assertTrue(
            np.array_equal(onset[:70], terrain_identity_touchdown(changed)[:70])
        )

    def test_current_dataset_matrix_remains_run_disjoint(self) -> None:
        config = load_terrain_collection_config(CONFIG)
        self.assertEqual(len(config.runs), 144)
        self.assertEqual(
            {
                split: sum(run.split == split for run in config.runs)
                for split in ("train", "validation", "holdout")
            },
            {"train": 88, "validation": 28, "holdout": 28},
        )
        self.assertEqual(len({run.condition_signature for run in config.runs}), 144)

    def test_cause_refinement_never_authorizes_or_blocks_reflex(self) -> None:
        self.assertEqual(refine_hazard_cause(False, ICE), "NORMAL")
        self.assertEqual(refine_hazard_cause(True, ICE), "SLIP_RISK")
        self.assertEqual(refine_hazard_cause(True, SAND), "SUPPORT_RISK")
        self.assertEqual(refine_hazard_cause(True, 0), "GENERIC_DISTURBANCE")
        self.assertEqual(refine_hazard_cause(True, 1), "GENERIC_DISTURBANCE")

    def test_supported_terrain_hashes_are_unchanged_and_advisory_only(self) -> None:
        audit = verify_supported_terrain_candidate(ROOT)
        self.assertTrue(audit["passed"])
        self.assertEqual(
            (audit["input"], audit["model_family"], audit["observation_ms"]),
            ("FSR4", "mlp", 50),
        )
        self.assertTrue(audit["advisory_only"])
        self.assertFalse(audit["hazard_gate"])
