from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from fastreflex.dataset.collector import (
    MANIFEST_FIELDS,
    REPOSITORY_ROOT,
    _write_manifest,
    _write_metadata,
    build_run_arrays,
    collect_dataset,
    load_collection_config,
    validate_run_arrays,
    write_run_npz,
)
from fastreflex.simulation.g1 import (
    TESTED_POLICY_SHA256,
    RuntimeTrace,
    SimulationResult,
    sha256_file,
)
from fastreflex.simulation.hazards import (
    SINK_HAZARD_TILT_THRESHOLD_RAD,
    SINK_PHYSICAL_THRESHOLD_M,
    SLIP_THRESHOLD_M,
    derive_physical_diagnostics,
)
from scripts.fastreflex import build_parser


EXPERIMENT_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "experiment"
    / "20260827_hazard_pilot_dataset.yaml"
)


def synthetic_result(kind: str) -> SimulationResult:
    samples = 80
    contact = np.ones((samples, 2), dtype=bool)
    force = np.full((samples, 2), 10.0)
    xyz = np.zeros((samples, 2, 3))
    velocity = np.zeros((samples, 2, 3))
    penetration = np.full((samples, 2), 0.001)
    pre_fall = np.ones(samples, dtype=bool)
    pelvis_z = np.full(samples, 0.8)
    orientation = np.tile((1.0, 0.0, 0.0, 0.0), (samples, 1))
    angular_velocity = np.zeros((samples, 3))
    linear_velocity = np.zeros((samples, 3))
    linear_velocity[:, 0] = 0.15
    fall_active = np.zeros(samples, dtype=bool)
    soft_patch_contact = np.zeros((samples, 2), dtype=bool)
    low_friction_patch_contact = np.zeros((samples, 2), dtype=bool)
    if kind == "slip":
        low_friction_patch_contact[10:, 0] = True
        xyz[12:, 0, 0] = SLIP_THRESHOLD_M
    elif kind == "sink":
        soft_patch_contact[10:, 1] = True
        penetration[20:, 1] += SINK_PHYSICAL_THRESHOLD_M
        tilt = SINK_HAZARD_TILT_THRESHOLD_RAD + 0.01
        orientation[42:, 0] = np.cos(tilt / 2.0)
        orientation[42:, 1] = np.sin(tilt / 2.0)
    else:
        raise ValueError(kind)
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
    runtime = RuntimeTrace(
        sequence=np.arange(samples, dtype=np.int64),
        timestamp_us=np.arange(1, samples + 1, dtype=np.int64) * 1000,
        pelvis_imu=np.zeros((samples, 6), dtype=np.float32),
    )
    return SimulationResult(
        runtime=runtime,
        diagnostics=diagnostics,
        metadata={
            "first_fall_sample": None,
            "first_fall_reasons": (),
            "dropped_samples": 0,
            "policy_sha256": TESTED_POLICY_SHA256,
        },
    )


class DatasetTest(unittest.TestCase):
    def test_collect_cli_uses_canonical_experiment_config(self) -> None:
        args = build_parser().parse_args(["collect"])
        self.assertEqual(args.command, "collect")
        self.assertEqual(args.config.resolve(), EXPERIMENT_CONFIG)
        self.assertIsNone(args.policy)

    def test_experiment_config_matrix_and_run_ids(self) -> None:
        config = load_collection_config(EXPERIMENT_CONFIG)
        self.assertEqual(config.dataset_id, "hazard_pilot_20260827")
        self.assertEqual(len(config.runs), 40)
        self.assertEqual(len({run.run_id for run in config.runs}), 40)
        self.assertEqual(
            sum(run.intended_role == "NORMAL" for run in config.runs), 16
        )
        self.assertEqual(
            sum(run.intended_role == "SLIP" for run in config.runs), 12
        )
        self.assertEqual(
            sum(run.intended_role == "SINK" for run in config.runs), 12
        )
        self.assertEqual(
            {run.patch_start_x_m for run in config.runs if run.intended_role == "SLIP"},
            {0.30, 0.35, 0.40},
        )

    def test_npz_round_trip_dtype_shape_and_sha_reproducibility(self) -> None:
        arrays = build_run_arrays(synthetic_result("slip"))
        validate_run_arrays(arrays, 80)
        self.assertEqual(arrays["pelvis_imu"].shape, (80, 6))
        self.assertEqual(arrays["pelvis_imu"].dtype, np.float32)
        self.assertEqual(arrays["sequence"].dtype, np.int64)
        self.assertEqual(arrays["sample_valid"].dtype, np.bool_)
        self.assertEqual(arrays["hazard_class_id"].dtype, np.int8)
        self.assertTrue(np.all(arrays["hazard_class_id"][:10] == 0))
        self.assertTrue(np.all(arrays["hazard_class_id"][10:14] == -1))
        self.assertTrue(np.all(arrays["hazard_class_id"][14:] == 1))
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.npz"
            second = Path(directory) / "second.npz"
            first_hash = write_run_npz(first, arrays)
            second_hash = write_run_npz(second, arrays)
            self.assertEqual(first_hash, sha256_file(first))
            self.assertEqual(first_hash, second_hash)
            with np.load(first, allow_pickle=False) as stored:
                self.assertEqual(set(stored.files), set(arrays))

    def test_sink_t1_to_t2_is_unresolved_and_aligned(self) -> None:
        arrays = build_run_arrays(synthetic_result("sink"))
        validate_run_arrays(arrays, 80)
        sink_t1 = int(
            arrays["first_sink_physical_onset_sample_per_foot"][1]
        )
        sink_t2 = int(arrays["first_sink_degradation_onset_sample"])
        self.assertEqual(sink_t1, 39)
        self.assertEqual(sink_t2, 61)
        self.assertTrue(np.all(arrays["hazard_class_id"][:10] == 0))
        self.assertTrue(np.all(arrays["hazard_class_id"][10:sink_t2] == -1))
        self.assertTrue(np.all(arrays["hazard_class_id"][sink_t2:] == 2))
        self.assertFalse(arrays["training_eligible"][sink_t1:sink_t2].any())

    def test_invalid_runtime_fails_closed(self) -> None:
        arrays = build_run_arrays(synthetic_result("slip"))
        invalid = {name: value.copy() for name, value in arrays.items()}
        invalid["pelvis_imu"][3, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_run_arrays(invalid, 80)

    def test_manifest_and_metadata_generation(self) -> None:
        config = load_collection_config(EXPERIMENT_CONFIG)
        row = {field: "" for field in MANIFEST_FIELDS}
        row.update(
            run_id="normal_concrete_s015_p000",
            file="runs/normal_concrete_s015_p000.npz",
            observed_outcome="BENIGN",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            metadata = root / "metadata.json"
            _write_manifest(manifest, [row])
            _write_metadata(metadata, config, "a" * 40, sha256_file(manifest))
            with manifest.open("r", encoding="utf-8", newline="") as stream:
                parsed_rows = list(csv.DictReader(stream))
            with metadata.open("r", encoding="utf-8") as stream:
                parsed_metadata = json.load(stream)
            self.assertEqual(parsed_rows[0]["observed_outcome"], "BENIGN")
            self.assertEqual(parsed_metadata["dataset_id"], config.dataset_id)
            self.assertEqual(parsed_metadata["source_commit"], "a" * 40)
            self.assertFalse(parsed_metadata["diagnostic_fields_are_runtime_input"])

    def test_existing_output_fails_without_overwrite(self) -> None:
        config = load_collection_config(EXPERIMENT_CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            (output_root / config.dataset_id).mkdir()
            policy = output_root / "policy.onnx"
            policy.touch()
            with mock.patch(
                "fastreflex.dataset.collector.sha256_file",
                return_value=TESTED_POLICY_SHA256,
            ), mock.patch(
                "fastreflex.dataset.collector._git_source_commit",
                return_value="a" * 40,
            ):
                with self.assertRaisesRegex(FileExistsError, "refusing overwrite"):
                    collect_dataset(
                        EXPERIMENT_CONFIG,
                        policy,
                        output_root=output_root,
                        progress=lambda _: None,
                    )

    def test_raw_dataset_path_is_git_ignored(self) -> None:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "data/raw/hazard_pilot_20260827"],
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
