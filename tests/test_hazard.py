"""Current contract tests for the supported Unified Hazard pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from fastreflex.dataset.hazard import (
    EVENT_TYPE_NONE,
    EVENT_TYPE_SLIP,
    EVENT_TYPE_SUPPORT,
    LABEL_NO_HAZARD,
    LABEL_PRECURSOR_ONLY,
    LABEL_SLIP,
    LABEL_SUPPORT,
    PELVIS_IMU6,
    PELVIS_IMU6_FSR8,
    HazardRun,
    HoldoutGuard,
    generate_hazard_specifications,
    i1_support_precursor_sample,
    load_hazard_manifest,
    load_hazard_runs,
    load_yaml,
    physical_hazard_label,
    split_for_source_index,
    validate_hazard_design,
)
from fastreflex.dataset.loader import Normalizer
from fastreflex.evaluation.hazard import (
    HazardReplay,
    evaluate_hazard_replays,
    load_hazard_normalizer,
    predict_hazard_window_members,
    predict_hazard_windows,
    reflex_onset_samples,
    reflex_required_trace,
    replay_hazard_run,
    verify_supported_candidate,
)
from fastreflex.evaluation.terrain import TerrainTrace
from fastreflex.features import (
    HAZARD_FEATURE_SCHEMA_SHA256,
    extract_hazard_features,
    feature_schema_hash,
    hazard_feature_schema,
)
from fastreflex.training.hazard import (
    build_hazard_windows,
    mine_hard_negative_endpoints,
    unified_negative_candidates,
    unified_positive_endpoints,
)
from fastreflex.training.trainer import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiment/20260829_unified_hazard_reflex_system.yaml"


def _run(
    *,
    split: str = "train",
    slip: int | None = None,
    support: int | None = None,
    spread_start: int | None = None,
    hard: bool = False,
    imu: np.ndarray | None = None,
) -> HazardRun:
    samples = 500
    pelvis = (
        np.zeros((samples, 6), dtype=np.float32)
        if imu is None
        else np.asarray(imu, dtype=np.float32)
    )
    fsr = np.zeros((samples, 8), dtype=np.float32)
    spread = np.zeros((samples, 2), dtype=np.float32)
    loaded = np.zeros((samples, 2), dtype=bool)
    loaded[100:, 0] = True
    if spread_start is not None:
        spread[spread_start:, 0] = (
            np.arange(1, samples - spread_start + 1, dtype=np.float32) * 0.0001
        )
    event = (
        min(value for value in (slip, support) if value is not None)
        if slip is not None or support is not None
        else None
    )
    event_type = (
        EVENT_TYPE_SLIP
        if slip is not None and support is None
        else EVENT_TYPE_SUPPORT
        if support is not None and slip is None
        else "SLIP_AND_SUPPORT"
        if support is not None and slip is not None
        else EVENT_TYPE_NONE
    )
    zeros = np.zeros((samples, 2), dtype=np.float32)
    return HazardRun(
        run_id="run",
        split=split,
        source_terrain="concrete",
        target_terrain=(
            "concrete"
            if hard
            else ("sand" if support is not None or spread_start is not None else "ice")
        ),
        design_role="stable",
        first_contact_sample=100,
        first_touchdown_sample=100,
        censor_sample=samples,
        outcome_diagnostic="VALID_STABLE",
        fall_sample_diagnostic=None,
        features={
            PELVIS_IMU6: pelvis,
            PELVIS_IMU6_FSR8: np.concatenate((pelvis, fsr), axis=1),
        },
        timestamp_us=np.arange(samples, dtype=np.int64) * 1000,
        slip_event_samples_per_foot=(slip, None),
        support_event_samples_per_foot=(support, None),
        event_sample=event,
        event_type=event_type,
        hard_stable_control=hard,
        drift_m=zeros,
        tangential_velocity_mps=zeros,
        support_spread_m=spread,
        support_max_displacement_m=spread,
        loaded_contact=loaded,
        sink_pattern="uniform",
        support_pattern="balanced_soft",
    )


def _reference_features(imu6: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Locked copy of the selected pre-consolidation 80D implementation."""
    values = np.asarray(imu6, dtype=np.float32)
    accel, gyro = values[:, :3], values[:, 3:]
    base = np.concatenate(
        (
            values,
            np.column_stack(
                (
                    np.linalg.norm(accel, axis=1),
                    np.linalg.norm(gyro, axis=1),
                    np.linalg.norm(accel[:, :2], axis=1),
                    np.linalg.norm(gyro[:, :2], axis=1),
                )
            ).astype(np.float32),
        ),
        axis=1,
    )

    def delta(lag: int) -> np.ndarray:
        result = np.zeros_like(base)
        if lag < len(base):
            result[lag:] = base[lag:] - base[:-lag]
        return result

    def rolling(width: int) -> tuple[np.ndarray, np.ndarray]:
        array = np.asarray(base, dtype=np.float64)
        prefix = np.vstack((np.zeros((1, 10)), np.cumsum(array, axis=0)))
        square = np.vstack((np.zeros((1, 10)), np.cumsum(array * array, axis=0)))
        ends = np.arange(1, len(array) + 1)
        starts = np.maximum(0, ends - width)
        counts = (ends - starts)[:, None]
        mean = (prefix[ends] - prefix[starts]) / counts
        variance = (square[ends] - square[starts]) / counts - mean * mean
        return mean.astype(np.float32), np.maximum(variance, 0.0).astype(np.float32)

    mean5, variance5 = rolling(5)
    mean10, variance10 = rolling(10)
    return (
        np.concatenate(
            (base, delta(1), delta(5), delta(10), mean5, mean10, variance5, variance10),
            axis=1,
        ).astype(np.float32, copy=False),
        hazard_feature_schema(),
    )


def _probability_replay(onset: int | None, samples: int = 500) -> HazardReplay:
    endpoints = np.arange(19, samples, dtype=np.int64)
    values = np.zeros(len(endpoints), dtype=np.float64)
    if onset is not None:
        values[endpoints >= onset - 4] = 1.0
    return HazardReplay(endpoints=endpoints, probabilities=values)


def _terrain(value: int, samples: int = 500) -> TerrainTrace:
    return TerrainTrace(
        state=np.full(samples, value, dtype=np.int8),
        update_samples=np.empty(0, dtype=np.int64),
        prediction_ids=np.empty(0, dtype=np.int64),
        prediction_probabilities=np.empty((0, 4), dtype=np.float32),
        first_target_valid_sample=None,
        clean_event_count=0,
    )


class HazardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = load_yaml(CONFIG)

    def test_feature_values_dtype_order_and_schema_match_selected_implementation(self) -> None:
        rng = np.random.default_rng(20260831)
        imu = rng.normal(size=(257, 6)).astype(np.float32)
        old_values, old_schema = _reference_features(imu)
        new_values = extract_hazard_features(imu)
        self.assertEqual(new_values.shape, (257, 80))
        self.assertEqual(new_values.dtype, np.float32)
        self.assertTrue(np.array_equal(old_values, new_values))
        self.assertEqual(old_schema, hazard_feature_schema())
        self.assertEqual(feature_schema_hash(), HAZARD_FEATURE_SCHEMA_SHA256)

    def test_features_are_prefix_causal_and_normalized_tensor_is_exact(self) -> None:
        rng = np.random.default_rng(7)
        imu = rng.normal(size=(80, 6)).astype(np.float32)
        altered = imu.copy()
        altered[60:] = 1_000.0
        first = extract_hazard_features(imu)
        second = extract_hazard_features(altered)
        self.assertTrue(np.array_equal(first[:60], second[:60]))
        normalizer = Normalizer(
            mean=np.linspace(-1.0, 1.0, 80, dtype=np.float32),
            std=np.linspace(0.5, 1.5, 80, dtype=np.float32),
            sample_count=60,
            fit_run_ids=("train",),
            epsilon=1.0e-6,
        )
        self.assertTrue(
            np.array_equal(
                normalizer.transform(first),
                ((first - normalizer.mean) / normalizer.std).astype(np.float32),
            )
        )

    def test_gru_windows_are_exactly_20_by_80_and_end_at_endpoint(self) -> None:
        imu = np.arange(500 * 6, dtype=np.float32).reshape(500, 6) / 100.0
        run = _run(slip=240, imu=imu)
        normalizer = Normalizer(
            mean=np.zeros(80, dtype=np.float32),
            std=np.ones(80, dtype=np.float32),
            sample_count=500,
            fit_run_ids=("run",),
            epsilon=1.0e-6,
        )
        windows = build_hazard_windows(
            {"run": run}, {"run"}, {"run": None}, normalizer
        )
        self.assertEqual(windows.inputs.shape[1:], (20, 80))
        endpoint = int(windows.endpoint_samples[0])
        expected = extract_hazard_features(imu)[endpoint - 19 : endpoint + 1]
        self.assertTrue(np.array_equal(windows.inputs[0], expected))

    def test_retained_member_probabilities_average_to_canonical_ensemble(self) -> None:
        torch.manual_seed(20260902)
        models = [
            torch.nn.Sequential(
                torch.nn.Flatten(), torch.nn.Linear(20 * 80, 2)
            )
            for _ in range(3)
        ]
        windows = np.random.default_rng(20260902).normal(
            size=(4, 20, 80)
        ).astype(np.float32)
        members = predict_hazard_window_members(models, windows)
        ensemble = predict_hazard_windows(models, windows)
        self.assertEqual(members.shape, (3, 4))
        self.assertTrue(np.array_equal(ensemble, np.mean(members, axis=0)))

    def test_frozen_design_labels_and_i1_semantics_are_unchanged(self) -> None:
        specifications = generate_hazard_specifications(self.document)
        audit = validate_hazard_design(ROOT, self.document, specifications)
        self.assertEqual(audit["total_split_counts"], {"train": 152, "validation": 52, "holdout": 52})
        self.assertEqual(split_for_source_index("concrete", 26), "validation")
        self.assertEqual(split_for_source_index("marble", 26), "holdout")
        self.assertEqual(i1_support_precursor_sample(_run(spread_start=120)), 139)
        self.assertEqual(physical_hazard_label(_run(slip=200), None), LABEL_SLIP)
        self.assertEqual(physical_hazard_label(_run(support=220), 150), LABEL_SUPPORT)
        self.assertEqual(physical_hazard_label(_run(), None), LABEL_NO_HAZARD)
        self.assertEqual(physical_hazard_label(_run(), 150), LABEL_PRECURSOR_ONLY)
        fallen = replace(_run(slip=200), outcome_diagnostic="VALID_FALL", fall_sample_diagnostic=300)
        self.assertEqual(physical_hazard_label(fallen, None), LABEL_SLIP)

    def test_positive_negative_and_hnm_contracts_are_exact(self) -> None:
        run = _run(support=240)
        positive = unified_positive_endpoints(run, 160)
        negative = unified_negative_candidates(run, 160)
        self.assertTrue(np.all(positive >= 160))
        self.assertTrue(np.all(negative < 160))
        self.assertLessEqual(len(positive), 20)
        candidates = np.arange(100, 500, dtype=np.int64)
        scores = np.linspace(0.0, 1.0, len(candidates))
        selected = mine_hard_negative_endpoints(candidates, scores)
        self.assertLessEqual(len(selected), 12)
        self.assertTrue(np.all(np.diff(selected) >= 30))
        self.assertTrue(np.array_equal(selected, mine_hard_negative_endpoints(candidates, scores)))

    def test_five_ms_persistence_timestamp_is_frozen(self) -> None:
        replay = _probability_replay(190)
        self.assertEqual(reflex_onset_samples(replay, threshold=0.99).tolist(), [190])
        trace = reflex_required_trace(replay, 500, threshold=0.99)
        self.assertFalse(trace[189])
        self.assertTrue(trace[190])

    def test_frozen_probability_and_event_timestamp_match_preconsolidation_math(self) -> None:
        dataset_path = ROOT / "data/raw/unified_hazard_reflex_20260829"
        freeze_path = (
            ROOT
            / "artifacts/runs/20260829_unified_hazard_reflex_system"
            / "selection_before_holdout.json"
        )
        if not dataset_path.is_dir() or not freeze_path.is_file():
            self.skipTest("local frozen TRAIN/VALIDATION artifacts are unavailable")
        manifest = load_hazard_manifest(dataset_path)
        validation_row = next(
            row for row in manifest["runs"] if row["split"] == "validation"
        )
        selected_manifest = {**manifest, "runs": [validation_row]}
        run = load_hazard_runs(
            dataset_path, selected_manifest, ("validation",)
        )[str(validation_row["run_id"])]
        selection = json.loads(freeze_path.read_text(encoding="utf-8"))["selection"]
        normalizer = load_hazard_normalizer(ROOT / str(selection["normalizer_path"]))
        models = [
            load_checkpoint(ROOT / relative)[0]
            for relative in selection["checkpoint_sha256"]
        ]
        canonical = replay_hazard_run(run, normalizer, models)

        features, _ = _reference_features(run.features[PELVIS_IMU6])
        endpoints = np.arange(19, run.censor_sample, dtype=np.int64)
        offsets = np.arange(19, -1, -1, dtype=np.int64)
        legacy_chunks = []
        for first in range(0, len(endpoints), 512):
            selected = endpoints[first : first + 512]
            windows = normalizer.transform(
                features[selected[:, None] - offsets[None, :]]
            )
            tensor = torch.from_numpy(windows)
            with torch.no_grad():
                legacy_chunks.append(
                    np.mean(
                        [
                            torch.softmax(model(tensor), dim=1)[:, 1].numpy()
                            for model in models
                        ],
                        axis=0,
                    ).astype(np.float64)
                )
        legacy = HazardReplay(endpoints, np.concatenate(legacy_chunks))
        self.assertTrue(np.array_equal(canonical.endpoints, legacy.endpoints))
        self.assertTrue(
            np.allclose(canonical.probabilities, legacy.probabilities, rtol=0.0, atol=1.0e-7)
        )
        self.assertTrue(
            np.array_equal(reflex_onset_samples(canonical), reflex_onset_samples(legacy))
        )

    def test_terrain_cannot_change_reflex_or_metrics(self) -> None:
        run = _run(slip=200)
        replay = _probability_replay(190)
        arguments = dict(
            runs={"run": run},
            replays={"run": replay},
            precursor_samples={"run": None},
            threshold=0.99,
        )
        unknown = evaluate_hazard_replays(terrain={"run": _terrain(0)}, **arguments)
        sand = evaluate_hazard_replays(terrain={"run": _terrain(4)}, **arguments)
        self.assertEqual(unknown["overall_hazard_recall"], sand["overall_hazard_recall"])
        self.assertEqual(unknown["rows"][0]["system_first_onset"], sand["rows"][0]["system_first_onset"])
        self.assertFalse(unknown["terrain_used_as_gate"])

    def test_no_hazard_and_precursor_only_specificity_boundaries(self) -> None:
        normal = _run(hard=True)
        metrics = evaluate_hazard_replays(
            {"run": normal}, {"run": _probability_replay(200)}, precursor_samples={"run": None}
        )
        self.assertEqual(metrics["primary_no_hazard_specificity"], 0.0)
        precursor = _run()
        excluded = evaluate_hazard_replays(
            {"run": precursor}, {"run": _probability_replay(200)}, precursor_samples={"run": 150}
        )
        self.assertEqual(excluded["primary_no_hazard_runs"], 0)
        self.assertEqual(excluded["precursor_only_runs_excluded_from_specificity"], 1)

    def test_holdout_guard_is_one_shot(self) -> None:
        guard = HoldoutGuard()
        guard.open_once()
        self.assertEqual(guard.open_count, 1)
        with self.assertRaises(RuntimeError):
            guard.open_once()

    def test_supported_freeze_and_all_protected_hashes_are_unchanged(self) -> None:
        audit = verify_supported_candidate(ROOT, self.document)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["freeze_sha256"], "91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2")
        self.assertEqual(audit["parameters"], 11_010)
        self.assertEqual((audit["history_ms"], audit["threshold"], audit["persistence_ms"]), (20, 0.99, 5))
        self.assertFalse(audit["holdout_opened"])

    def test_cli_fails_closed_for_historical_configs(self) -> None:
        current = subprocess.run(
            [sys.executable, str(ROOT / "scripts/fastreflex.py"), "evaluate", "--config", str(CONFIG)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(current.returncode, 0, current.stderr)
        historical = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/fastreflex.py"),
                "evaluate",
                "--config",
                str(ROOT / "configs/experiment/20260828_walking_stability_ground_truth_sanity.yaml"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(historical.returncode, 0)
        self.assertIn("historical and is not runnable", historical.stderr)


if __name__ == "__main__":
    unittest.main()
