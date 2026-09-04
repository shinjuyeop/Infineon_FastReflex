"""Canonical Hazard dataset boundary tests."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fastreflex.dataset.hazard import (
    HoldoutGuard,
    generate_hazard_specifications,
    load_hazard_manifest,
    load_hazard_runs,
    load_yaml,
    physical_signature,
    validate_hazard_design,
)
from tests.support import REPOSITORY_ROOT as ROOT

CONFIG = ROOT / "configs/experiment/20260829_unified_hazard_reflex_system.yaml"


def _write_run(path: Path) -> str:
    samples = 40
    zeros2 = np.zeros((samples, 2), dtype=np.float32)
    np.savez_compressed(
        path,
        timestamp_us=np.arange(samples, dtype=np.int64) * 1000,
        pelvis_imu6=np.zeros((samples, 6), dtype=np.float32),
        foot_fsr8=np.zeros((samples, 8), dtype=np.float32),
        tangential_anchor_drift_m=zeros2,
        tangential_velocity_mps=zeros2,
        support_surface_spread_m=zeros2,
        support_surface_max_displacement_m=zeros2,
        loaded_contact=np.zeros((samples, 2), dtype=bool),
        first_target_contact_sample=np.asarray(0, dtype=np.int64),
        first_target_touchdown_sample=np.asarray(0, dtype=np.int64),
        censor_sample=np.asarray(samples, dtype=np.int64),
        first_slip_event_sample_per_foot=np.asarray([-1, -1], dtype=np.int64),
        first_support_event_sample_per_foot=np.asarray([-1, -1], dtype=np.int64),
        first_reflex_event_sample=np.asarray(-1, dtype=np.int64),
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(run_id: str, split: str, file_hash: str) -> dict[str, object]:
    return {
        "run_id": run_id,
        "file": f"{run_id}.npz",
        "file_sha256": file_hash,
        "split": split,
        "source_terrain": "concrete",
        "target_terrain": "concrete",
        "design_role_diagnostic_only": "hard_ground_normal",
        "observed_outcome_diagnostic_only": "VALID_STABLE",
        "fall_sample_diagnostic_only": None,
        "hard_stable_control": True,
        "event_sample": None,
        "event_type": "NONE",
        "sink_pattern": "uniform",
        "support_pattern": "balanced_soft",
    }


class DatasetTest(unittest.TestCase):
    def test_frozen_matrix_is_unique_split_before_simulation_and_fresh(self) -> None:
        document = load_yaml(CONFIG)
        specifications = generate_hazard_specifications(document)
        result = validate_hazard_design(ROOT, document, specifications)
        self.assertTrue(result["passed"])
        self.assertEqual(result["runs"], 256)
        self.assertEqual(result["duplicate_signatures"], 0)
        self.assertEqual(result["prior_signature_overlap"], 0)
        self.assertEqual(
            result["total_split_counts"],
            {"train": 152, "validation": 52, "holdout": 52},
        )
        self.assertEqual(len({physical_signature(row) for row in specifications}), 256)

    def test_manifest_integrity_and_run_tensor_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "train.npz"
            file_hash = _write_run(path)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "dataset_id": "fixture",
                        "runs": [_row("train", "train", file_hash)],
                    }
                ),
                encoding="utf-8",
            )
            manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            (root / "manifest.sha256").write_text(
                f"{manifest_hash}  manifest.json\n", encoding="utf-8"
            )
            manifest = load_hazard_manifest(root)
            runs = load_hazard_runs(root, manifest, ("train",))
            self.assertEqual(set(runs), {"train"})
            self.assertEqual(runs["train"].features["PELVIS_IMU6"].shape, (40, 6))
            self.assertEqual(runs["train"].features["PELVIS_IMU6_FSR8"].shape, (40, 14))
            self.assertEqual(runs["train"].timestamp_us.dtype, np.int64)

    def test_holdout_waveforms_fail_closed_without_one_shot_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "holdout.npz"
            file_hash = _write_run(path)
            manifest = {"runs": [_row("holdout", "holdout", file_hash)]}
            with self.assertRaises(RuntimeError):
                load_hazard_runs(root, manifest, ("holdout",))
            guard = HoldoutGuard()
            guard.open_once()
            self.assertEqual(
                set(
                    load_hazard_runs(root, manifest, ("holdout",), holdout_guard=guard)
                ),
                {"holdout"},
            )

    def test_manifest_and_run_hash_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "train.npz"
            file_hash = _write_run(path)
            manifest = {
                "dataset_id": "fixture",
                "runs": [_row("train", "train", file_hash)],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (root / "manifest.sha256").write_text(
                "0" * 64 + "  manifest.json\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_hazard_manifest(root)
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaises(ValueError):
                load_hazard_runs(root, manifest, ("train",))
