"""Reviewed frozen-Float handoff packaging for the deployment repository."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import yaml

from fastreflex.dataset.loader import sha256_file
from fastreflex.evaluation.hazard import (
    load_hazard_normalizer,
    predict_hazard_window_members,
    sustained_reflex,
)
from fastreflex.features import (
    HAZARD_BASE_FEATURES,
    HAZARD_FEATURE_SCHEMA_SHA256,
    HAZARD_TEMPORAL_TRANSFORMS,
    extract_hazard_features,
    feature_schema_hash,
    hazard_feature_schema,
    imu_feature_base,
)
from fastreflex.models.baselines import parameter_count
from fastreflex.models.checkpoint import load_checkpoint


DEFAULT_CONFIG = Path("configs/model/deployment_engineering_reference.yaml")
REQUIRED_RELEASE_FILES = frozenset(
    {
        "golden_inputs/decision_probe.npz",
        "golden_inputs/runtime_chain.npz",
        "golden_manifest.json",
        "golden_outputs/decision_probe.npz",
        "golden_outputs/runtime_chain.npz",
        "label_map.json",
        "metrics.json",
        "model_manifest.json",
        "models/member_seed20260828.pt",
        "models/member_seed20260829.pt",
        "models/member_seed20260830.pt",
        "normalizer.json",
        "preprocessing.json",
        "sensor_schema.json",
    }
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write NPZ without wall-clock ZIP metadata so release hashes are stable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, values in sorted(arrays.items()):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.asarray(values), allow_pickle=False
            )
            info = zipfile.ZipInfo(
                filename=f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                buffer.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _repository_commit(root: Path) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _checked_source(root: Path, record: Mapping[str, object]) -> Path:
    path = root / str(record["path"])
    if not path.is_file():
        raise ValueError(f"required frozen artifact is missing: {path}")
    actual = sha256_file(path)
    if actual != str(record["sha256"]):
        raise ValueError(f"frozen artifact checksum changed: {path}")
    return path


def _consecutive_counts(passes: np.ndarray) -> np.ndarray:
    counts = np.zeros(len(passes), dtype=np.int64)
    count = 0
    for index, value in enumerate(passes):
        count = count + 1 if bool(value) else 0
        counts[index] = count
    return counts


def _runtime_documents(
    document: Mapping[str, object], packaging_commit: str
) -> dict[str, object]:
    release = document["release"]
    candidate = document["candidate"]
    architecture = document["architecture"]
    runtime = document["runtime"]
    checkpoints = [
        {
            "seed": int(record["seed"]),
            "path": f"models/member_seed{record['seed']}.pt",
            "sha256": str(record["sha256"]),
            "source_path": str(record["path"]),
        }
        for record in candidate["checkpoints"]
    ]
    model_manifest = {
        "schema_version": 1,
        "candidate_id": str(release["id"]),
        "engineering_role": str(release["role"]),
        "status": str(release["status"]),
        "release_model": False,
        "framework": "pytorch",
        "framework_version_at_export": torch.__version__,
        "numpy_version_at_export": np.__version__,
        "model_format": "fastreflex_pytorch_checkpoint_dict",
        "provenance": {
            "source_repository": str(release["source_repository"]),
            "candidate_source_commit": str(release["candidate_source_commit"]),
            "candidate_record_commit": str(release["candidate_record_commit"]),
            "scientific_verdict_commit": str(release["scientific_verdict_commit"]),
            "packaging_source_commit": packaging_commit,
            "packaged_date": str(release["date"]),
            "timezone": str(release["timezone"]),
            "candidate_freeze": dict(candidate["freeze"]),
            "candidate_evaluation_freeze": dict(candidate["evaluation_freeze"]),
        },
        "architecture": dict(architecture),
        "runtime": dict(runtime),
        "ensemble_members": checkpoints,
        "normalizer": {
            "path": "normalizer.json",
            "sha256": str(candidate["normalizer"]["sha256"]),
            "source_path": str(candidate["normalizer"]["path"]),
        },
        "class_index": {"0": "NORMAL", "1": "HAZARD_REFLEX_REQUIRED"},
        "golden_manifest": "golden_manifest.json",
        "scientific_status": {
            "verdict": "MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED",
            "simulation_status": "SIMULATION_GENERALIZATION_EVIDENCE_NOT_SUPPORTED",
            "engineering_only": True,
            "real_robot_supported": False,
            "safety_certified": False,
        },
    }
    sensor_schema = {
        "schema_version": 1,
        "sensor": "PELVIS_IMU6",
        "sample_rate_hz": 1000,
        "sample_period_ms": 1,
        "shape": ["samples", 6],
        "dtype": "float32",
        "frame": "pelvis_local",
        "reference_location": "pelvis frame origin",
        "reference_orientation": "identity relative to pelvis body frame",
        "channels": [
            {"index": 0, "name": "accel_x", "unit": "m/s^2"},
            {"index": 1, "name": "accel_y", "unit": "m/s^2"},
            {"index": 2, "name": "accel_z", "unit": "m/s^2"},
            {"index": 3, "name": "gyro_x", "unit": "rad/s"},
            {"index": 4, "name": "gyro_y", "unit": "rad/s"},
            {"index": 5, "name": "gyro_z", "unit": "rad/s"},
        ],
        "reference_sensor": "MuJoCo accelerometer and gyro at site imu",
        "software_filtering": "none before causal feature extraction",
        "hardware_mapping_status": "NOT_VALIDATED",
        "hardware_requirement": (
            "axis, sign, scale, bandwidth, timestamp, and acceleration/gravity "
            "convention must reproduce this schema before target validation"
        ),
    }
    feature_schema = list(hazard_feature_schema())
    preprocessing = {
        "schema_version": 1,
        "input": "PELVIS_IMU6 float32 [samples,6]",
        "output": "float32 [samples,80]",
        "base_feature_order": list(HAZARD_BASE_FEATURES),
        "temporal_transform_order": list(HAZARD_TEMPORAL_TRANSFORMS),
        "feature_order": feature_schema,
        "feature_schema_sha256": HAZARD_FEATURE_SCHEMA_SHA256,
        "base_math": {
            "raw": "input channels cast to float32",
            "accel_norm": "sqrt(accel_x^2 + accel_y^2 + accel_z^2)",
            "gyro_norm": "sqrt(gyro_x^2 + gyro_y^2 + gyro_z^2)",
            "horizontal_accel_norm": "sqrt(accel_x^2 + accel_y^2)",
            "horizontal_gyro_norm": "sqrt(gyro_x^2 + gyro_y^2)",
        },
        "delta": {
            "definition": "current minus lagged sample",
            "lags_samples": [1, 5, 10],
            "unavailable_prefix": "exact_zero",
        },
        "rolling": {
            "definition": "trailing samples including current endpoint",
            "widths_samples": [5, 10],
            "startup": "use all available samples from index zero",
            "accumulator_dtype": "float64",
            "variance": "population variance (ddof=0), clamped to at least zero",
            "output_dtype": "float32",
        },
        "normalization": {
            "method": "per_channel_zscore",
            "formula": "(feature - stored_mean) / stored_std",
            "parameter_dtype": "float32",
            "output_dtype": "float32",
            "normalizer_path": "normalizer.json",
            "normalizer_sha256": str(candidate["normalizer"]["sha256"]),
            "runtime_epsilon_or_clamp": "none; stored std is used exactly",
        },
        "window": {
            "shape": [20, 80],
            "order": "oldest_to_current",
            "endpoint": "inclusive",
            "first_endpoint_index": 19,
            "stride_samples": 1,
        },
    }
    label_map = {
        "schema_version": 1,
        "logit_and_softmax_class_index": {
            "0": "NORMAL",
            "1": "HAZARD_REFLEX_REQUIRED",
        },
        "hazard_probability": "softmax(logits, dim=1)[1] per member",
        "ensemble_probability": "arithmetic mean of three member probabilities",
        "control_output": {
            "false": "NORMAL",
            "true": "REFLEX_REQUIRED",
        },
        "terrain_used_as_hazard_gate": False,
    }
    return {
        "model_manifest.json": model_manifest,
        "sensor_schema.json": sensor_schema,
        "preprocessing.json": preprocessing,
        "label_map.json": label_map,
        "metrics.json": {
            "schema_version": 1,
            "candidate_id": str(release["id"]),
            "engineering_role": str(release["role"]),
            **dict(document["scientific_status"]),
        },
    }


def export_reference_release(
    repository_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    output_path: Path | None = None,
) -> Path:
    """Package the exact frozen candidate without training or protected access."""
    root = repository_root.resolve()
    config = config_path if config_path.is_absolute() else root / config_path
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    if int(document.get("schema_version", 0)) != 1:
        raise ValueError("unsupported deployment-reference config schema")
    release = document["release"]
    candidate = document["candidate"]
    runtime = document["runtime"]
    golden = document["golden"]
    destination = (
        output_path
        if output_path is not None
        else root / str(release["output_path"])
    ).resolve()
    if destination.exists():
        raise ValueError(f"release output already exists: {destination}")
    if feature_schema_hash() != str(runtime["feature_schema_sha256"]):
        raise ValueError("current feature schema differs from handoff config")

    _checked_source(root, candidate["freeze"])
    _checked_source(root, candidate["evaluation_freeze"])
    normalizer_source = _checked_source(root, candidate["normalizer"])
    for source in document["scientific_status"]["metrics_sources"]:
        _checked_source(root, source)

    checkpoint_sources: list[Path] = []
    models: list[torch.nn.Module] = []
    for record in candidate["checkpoints"]:
        path = _checked_source(root, record)
        model, metadata = load_checkpoint(path)
        if (
            int(metadata["seed"]) != int(record["seed"])
            or metadata["family"] != "gru"
            or int(metadata["window_samples"]) != 20
            or int(metadata["input_channels"]) != 80
            or metadata["class_names"]
            != ["NORMAL", "HAZARD_REFLEX_REQUIRED"]
            or parameter_count(model) != 11_010
        ):
            raise ValueError(f"checkpoint metadata contract changed: {path}")
        checkpoint_sources.append(path)
        models.append(model)

    source_run = root / str(golden["source_path"])
    if sha256_file(source_run) != str(golden["source_file_sha256"]):
        raise ValueError("golden source run checksum changed")
    with np.load(source_run, allow_pickle=False) as payload:
        if payload["pelvis_imu6"].dtype != np.float32:
            raise ValueError("golden source IMU dtype changed")
        start = int(golden["slice_start_sample"])
        stop = int(golden["slice_stop_sample_exclusive"])
        raw = payload["pelvis_imu6"][start:stop].copy()
        timestamp_us = payload["timestamp_us"][start:stop].copy()
    if raw.shape != (stop - start, 6) or len(raw) < 20:
        raise ValueError("golden source slice contract changed")

    base, _ = imu_feature_base(raw)
    features = extract_hazard_features(raw)
    normalizer = load_hazard_normalizer(normalizer_source)
    normalized = normalizer.transform(features)
    endpoints = np.arange(19, len(raw), dtype=np.int64)
    offsets = np.arange(19, -1, -1, dtype=np.int64)
    windows = normalized[endpoints[:, None] - offsets[None, :]].astype(
        np.float32, copy=False
    )
    tensor = torch.from_numpy(windows)
    with torch.no_grad():
        member_logits = np.stack(
            [model(tensor).cpu().numpy() for model in models]
        ).astype(np.float32, copy=False)
    member_probabilities = predict_hazard_window_members(models, windows)
    ensemble = np.mean(member_probabilities, axis=0)
    threshold_crossing = ensemble >= float(runtime["threshold"])
    reflex_required, reflex_onset = sustained_reflex(
        ensemble,
        threshold=float(runtime["threshold"]),
        persistence_ms=int(runtime["persistence_samples"]),
    )
    if not np.any(reflex_required) or not np.any(~reflex_required):
        raise ValueError("golden chain must exercise both control decisions")

    probe = np.asarray(
        [
            0.989,
            0.99,
            0.99,
            0.99,
            0.99,
            np.nextafter(0.99, 0.0),
            0.99,
            0.99,
            0.99,
            0.99,
            0.99,
            0.98,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float64,
    )
    probe_passes = probe >= float(runtime["threshold"])
    probe_alert, probe_onset = sustained_reflex(
        probe,
        threshold=float(runtime["threshold"]),
        persistence_ms=int(runtime["persistence_samples"]),
    )

    destination.mkdir(parents=True)
    shutil.copyfile(normalizer_source, destination / "normalizer.json")
    (destination / "models").mkdir()
    for record, source in zip(candidate["checkpoints"], checkpoint_sources):
        shutil.copyfile(
            source,
            destination / "models" / f"member_seed{record['seed']}.pt",
        )
    for name, payload in _runtime_documents(
        document, _repository_commit(root)
    ).items():
        _write_json(destination / name, payload)

    _write_deterministic_npz(
        destination / "golden_inputs/runtime_chain.npz",
        {
            "raw_pelvis_imu6": raw,
            "source_sample_indices": np.arange(start, stop, dtype=np.int64),
            "timestamp_us": timestamp_us,
        },
    )
    _write_deterministic_npz(
        destination / "golden_outputs/runtime_chain.npz",
        {
            "base_features": base,
            "causal_features": features,
            "consecutive_threshold_count": _consecutive_counts(
                threshold_crossing
            ),
            "ensemble_hazard_probability": ensemble,
            "member_hazard_probability": member_probabilities,
            "member_logits": member_logits,
            "model_windows": windows,
            "normalized_features": normalized,
            "reflex_onset": reflex_onset,
            "reflex_required": reflex_required,
            "threshold_crossing": threshold_crossing,
            "window_endpoints": endpoints,
        },
    )
    _write_deterministic_npz(
        destination / "golden_inputs/decision_probe.npz",
        {"ensemble_hazard_probability": probe},
    )
    _write_deterministic_npz(
        destination / "golden_outputs/decision_probe.npz",
        {
            "consecutive_threshold_count": _consecutive_counts(probe_passes),
            "reflex_onset": probe_onset,
            "reflex_required": probe_alert,
            "threshold_crossing": probe_passes,
        },
    )
    _write_json(
        destination / "golden_manifest.json",
        {
            "schema_version": 1,
            "purpose": "runtime_chain_parity_only",
            "scientific_evidence": False,
            "protected_holdout_access": False,
            "source": {
                "dataset_id": str(golden["source_dataset_id"]),
                "dataset_manifest_sha256": str(
                    golden["source_dataset_manifest_sha256"]
                ),
                "run_id": str(golden["source_run_id"]),
                "split": str(golden["source_split"]),
                "source_file_sha256": str(golden["source_file_sha256"]),
                "slice_start_sample": start,
                "slice_stop_sample_exclusive": stop,
            },
            "runtime_chain": [
                "raw_pelvis_imu6",
                "base_features",
                "causal_features",
                "normalized_features",
                "model_windows",
                "member_logits",
                "member_hazard_probability",
                "ensemble_hazard_probability",
                "threshold_crossing",
                "consecutive_threshold_count",
                "reflex_required",
                "reflex_onset",
            ],
            "numeric_tolerance": {"absolute": 1e-6, "relative": 1e-6},
            "discrete_parity": "exact",
            "decision_probe": (
                "separate synthetic probability trace exercises inclusive "
                "threshold, reset, fifth-sample assertion, and held assertion"
            ),
        },
    )

    files = {
        str(path.relative_to(destination)): sha256_file(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    if set(files) != REQUIRED_RELEASE_FILES:
        raise RuntimeError("release file set differs from the reviewed contract")
    _write_json(
        destination / "release_manifest.json",
        {
            "schema_version": 1,
            "release_id": str(release["id"]),
            "file_count": len(files),
            "files": files,
        },
    )
    verify_release(destination)
    return destination


def verify_release(release_path: Path) -> dict[str, object]:
    """Fail closed on missing, extra, traversing, or checksum-mismatched files."""
    root = release_path.resolve()
    manifest_path = root / "release_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("release_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("unsupported release manifest schema")
    records = manifest.get("files")
    if not isinstance(records, dict) or set(records) != REQUIRED_RELEASE_FILES:
        raise ValueError("release manifest file set differs from contract")
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != REQUIRED_RELEASE_FILES | {"release_manifest.json"}:
        raise ValueError("release directory contains missing or extra files")
    for relative, expected in records.items():
        selected = (root / relative).resolve()
        if root not in selected.parents or not selected.is_file():
            raise ValueError(f"invalid release path: {relative}")
        if sha256_file(selected) != str(expected):
            raise ValueError(f"release checksum mismatch: {relative}")
    model = json.loads((root / "model_manifest.json").read_text(encoding="utf-8"))
    if (
        model.get("engineering_role")
        != "DEPLOYMENT_ENGINEERING_REFERENCE_MODEL"
        or model.get("status") != "NON_RELEASE_ENGINEERING_REFERENCE"
        or model.get("release_model") is not False
        or model.get("scientific_status", {}).get("verdict")
        != "MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED"
    ):
        raise ValueError("release scientific/non-release status changed")
    return {
        "release_id": manifest["release_id"],
        "release_manifest_sha256": sha256_file(manifest_path),
        "files_verified": len(records),
        "status": "PASS",
    }
