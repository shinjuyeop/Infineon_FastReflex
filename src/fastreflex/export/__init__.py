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
        "golden_outputs/deployment_runtime_chain.npz",
        "golden_outputs/runtime_chain.npz",
        "float_numerical_contract.json",
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
            np.lib.format.write_array(buffer, np.asarray(values), allow_pickle=False)
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


def _infer_members(
    models: list[torch.nn.Module], windows: np.ndarray, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Run stateless windows in fixed chunks and retain Float32 softmax results."""
    if batch_size <= 0:
        raise ValueError("inference batch size must be positive")
    tensor = torch.from_numpy(windows)
    logits: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for model in models:
            member_logits: list[np.ndarray] = []
            member_probabilities: list[np.ndarray] = []
            for first in range(0, len(windows), batch_size):
                output = model(tensor[first : first + batch_size])
                member_logits.append(output.cpu().numpy())
                member_probabilities.append(
                    torch.softmax(output, dim=1)[:, 1].cpu().numpy()
                )
            logits.append(np.concatenate(member_logits))
            probabilities.append(np.concatenate(member_probabilities))
    return (
        np.stack(logits).astype(np.float32, copy=False),
        np.stack(probabilities).astype(np.float32, copy=False).astype(np.float64),
    )


def _maximum_absolute(actual: np.ndarray, reference: np.ndarray) -> float:
    difference = np.abs(
        actual.astype(np.float64, copy=False) - reference.astype(np.float64, copy=False)
    )
    return 0.0 if not difference.size else float(np.max(difference))


def _within_tolerance(
    actual: np.ndarray,
    reference: np.ndarray,
    tolerance: Mapping[str, object],
) -> bool:
    return bool(
        np.allclose(
            actual,
            reference,
            atol=float(tolerance["absolute"]),
            rtol=float(tolerance["relative"]),
        )
    )


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
        output_path if output_path is not None else root / str(release["output_path"])
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
            or metadata["class_names"] != ["NORMAL", "HAZARD_REFLEX_REQUIRED"]
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
        # PyTorch may prepare an internal GRU weight layout on the first CPU
        # call. Warm it without retaining state, then freeze logits and
        # probabilities from the same stable call.
        for model in models:
            model(tensor)
        logits = [model(tensor) for model in models]
        member_logits = np.stack([value.cpu().numpy() for value in logits]).astype(
            np.float32, copy=False
        )
        member_probabilities = np.stack(
            [torch.softmax(value, dim=1)[:, 1].cpu().numpy() for value in logits]
        ).astype(np.float64)
    ensemble = np.mean(member_probabilities, axis=0)
    threshold_crossing = ensemble >= float(runtime["threshold"])
    reflex_required, reflex_onset = sustained_reflex(
        ensemble,
        threshold=float(runtime["threshold"]),
        persistence_ms=int(runtime["persistence_samples"]),
    )
    if not np.any(reflex_required) or not np.any(~reflex_required):
        raise ValueError("golden chain must exercise both control decisions")

    numerical_settings = document.get("float_numerical_contract")
    if not isinstance(numerical_settings, dict):
        raise ValueError("Float numerical contract config is missing")
    if numerical_settings.get("verdict") != "FLOAT_EXPORT_NUMERICAL_CONTRACT_RESOLVED":
        raise ValueError("Float numerical contract verdict changed")
    canonical_batch_size = int(numerical_settings["canonical_batch_size"])
    if canonical_batch_size != 1:
        raise ValueError("deployment Float reference must use batch size one")
    continuous_parity = numerical_settings["continuous_parity"]
    required_tolerances = {
        "preprocessing_and_model_windows",
        "member_logits",
        "member_hazard_probability",
        "ensemble_hazard_probability",
    }
    if set(continuous_parity) != required_tolerances:
        raise ValueError("Float numerical parity layers changed")
    for tolerance in continuous_parity.values():
        if (
            set(tolerance) != {"absolute", "relative"}
            or float(tolerance["absolute"]) < 0.0
            or float(tolerance["relative"]) < 0.0
        ):
            raise ValueError("invalid Float numerical tolerance")

    deployment_logits, deployment_member_probabilities = _infer_members(
        models, windows, canonical_batch_size
    )
    deployment_ensemble = np.mean(
        deployment_member_probabilities, axis=0, dtype=np.float64
    )
    deployment_threshold_crossing = deployment_ensemble >= float(runtime["threshold"])
    deployment_reflex_required, deployment_reflex_onset = sustained_reflex(
        deployment_ensemble,
        threshold=float(runtime["threshold"]),
        persistence_ms=int(runtime["persistence_samples"]),
    )
    deployment_counts = _consecutive_counts(deployment_threshold_crossing)

    evidence_batch_sizes = [
        int(value) for value in numerical_settings["evidence_batch_sizes"]
    ]
    if (
        evidence_batch_sizes[0] != canonical_batch_size
        or evidence_batch_sizes[-1] != len(windows)
        or any(value <= 0 for value in evidence_batch_sizes)
        or evidence_batch_sizes != sorted(set(evidence_batch_sizes))
    ):
        raise ValueError("Float variability batch-size evidence changed")
    batch_size_sweep: dict[str, object] = {}
    for batch_size in evidence_batch_sizes:
        swept_logits, swept_probabilities = _infer_members(models, windows, batch_size)
        swept_ensemble = np.mean(swept_probabilities, axis=0, dtype=np.float64)
        for layer, actual, reference in (
            ("member_logits", swept_logits, deployment_logits),
            (
                "member_hazard_probability",
                swept_probabilities,
                deployment_member_probabilities,
            ),
            (
                "ensemble_hazard_probability",
                swept_ensemble,
                deployment_ensemble,
            ),
        ):
            if not _within_tolerance(actual, reference, continuous_parity[layer]):
                raise ValueError(
                    f"batch size {batch_size} exceeds the reviewed {layer} tolerance"
                )
        swept_crossing = swept_ensemble >= float(runtime["threshold"])
        swept_reflex, swept_onset = sustained_reflex(
            swept_ensemble,
            threshold=float(runtime["threshold"]),
            persistence_ms=int(runtime["persistence_samples"]),
        )
        discrete_exact = bool(
            np.array_equal(swept_crossing, deployment_threshold_crossing)
            and np.array_equal(_consecutive_counts(swept_crossing), deployment_counts)
            and np.array_equal(swept_reflex, deployment_reflex_required)
            and np.array_equal(swept_onset, deployment_reflex_onset)
        )
        if not discrete_exact:
            raise ValueError(
                f"batch size {batch_size} changed a discrete runtime result"
            )
        batch_size_sweep[str(batch_size)] = {
            "member_logits_max_absolute": _maximum_absolute(
                swept_logits, deployment_logits
            ),
            "member_hazard_probability_max_absolute": _maximum_absolute(
                swept_probabilities, deployment_member_probabilities
            ),
            "ensemble_hazard_probability_max_absolute": _maximum_absolute(
                swept_ensemble, deployment_ensemble
            ),
            "discrete_runtime_outputs_exact": discrete_exact,
        }
    sweep_logit_maximum = max(
        float(record["member_logits_max_absolute"])
        for record in batch_size_sweep.values()
    )
    sweep_probability_maximum = max(
        float(record["member_hazard_probability_max_absolute"])
        for record in batch_size_sweep.values()
    )
    sweep_ensemble_maximum = max(
        float(record["ensemble_hazard_probability_max_absolute"])
        for record in batch_size_sweep.values()
    )
    repeated_logits, repeated_probabilities = _infer_members(
        models, windows, canonical_batch_size
    )
    repeated_ensemble = np.mean(repeated_probabilities, axis=0, dtype=np.float64)
    repeated_batch_one_exact = bool(
        np.array_equal(repeated_logits, deployment_logits)
        and np.array_equal(repeated_probabilities, deployment_member_probabilities)
        and np.array_equal(repeated_ensemble, deployment_ensemble)
    )
    if not repeated_batch_one_exact:
        raise ValueError("canonical batch-one Float reference is not deterministic")

    for layer, actual, reference in (
        (
            "member_logits",
            member_logits,
            deployment_logits,
        ),
        (
            "member_hazard_probability",
            member_probabilities,
            deployment_member_probabilities,
        ),
        (
            "ensemble_hazard_probability",
            ensemble,
            deployment_ensemble,
        ),
    ):
        if not _within_tolerance(actual, reference, continuous_parity[layer]):
            raise ValueError(
                f"historical batch execution exceeds the reviewed {layer} tolerance"
            )
    if not (
        np.array_equal(threshold_crossing, deployment_threshold_crossing)
        and np.array_equal(_consecutive_counts(threshold_crossing), deployment_counts)
        and np.array_equal(reflex_required, deployment_reflex_required)
        and np.array_equal(reflex_onset, deployment_reflex_onset)
    ):
        raise ValueError("valid Float batching changed a discrete runtime result")

    threshold = float(runtime["threshold"])
    margins = np.abs(deployment_ensemble - threshold)
    above = deployment_threshold_crossing
    closest_index = int(np.argmin(margins))
    ensemble_tolerance = continuous_parity["ensemble_hazard_probability"]
    permitted_at_closest = float(ensemble_tolerance["absolute"]) + float(
        ensemble_tolerance["relative"]
    ) * abs(float(deployment_ensemble[closest_index]))
    minimum_margin = float(np.min(margins))
    if minimum_margin <= permitted_at_closest:
        raise ValueError(
            "Float contract tolerance reaches the nearest threshold margin"
        )

    packaging_commit = _repository_commit(root)
    float_contract = {
        "schema_version": 1,
        "verdict": str(numerical_settings["verdict"]),
        "candidate_id": str(release["id"]),
        "engineering_role": str(release["role"]),
        "scientific_verdict": "MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED",
        "scope": "deployment_float_parity_only_not_scientific_evidence",
        "protected_holdout_access": False,
        "provenance": {
            "contract_source_commit": packaging_commit,
            "candidate_source_commit": str(release["candidate_source_commit"]),
            "candidate_record_commit": str(release["candidate_record_commit"]),
            "scientific_verdict_commit": str(release["scientific_verdict_commit"]),
            "feature_schema_sha256": str(runtime["feature_schema_sha256"]),
            "normalizer_sha256": str(candidate["normalizer"]["sha256"]),
            "checkpoint_sha256": [
                str(record["sha256"]) for record in candidate["checkpoints"]
            ],
        },
        "canonical_execution": {
            "input_shape": [1, 20, 80],
            "input_dtype": "float32",
            "output_shape_per_member": [1, 2],
            "member_logit_dtype": "float32",
            "batch_size": 1,
            "one_causal_endpoint_per_invocation": True,
            "window_order": "oldest_to_current",
            "hidden_state": "zero_initialized_for_every_invocation_no_carry",
            "member_softmax_compute_dtype": "float32",
            "member_probability_storage_dtype": "float64",
            "ensemble_mean_dtype": "float64",
            "golden_generation": (
                "sequential independent PyTorch calls, one [1,20,80] "
                "window per member invocation"
            ),
            "golden_output": "golden_outputs/deployment_runtime_chain.npz",
        },
        "continuous_parity": {
            "preprocessing_and_model_windows": {
                **dict(continuous_parity["preprocessing_and_model_windows"]),
                "layers": [
                    "base_features",
                    "causal_features",
                    "normalized_features",
                    "model_windows",
                ],
                "rationale": (
                    "retains the reviewed M1 independent-host Float allowance"
                ),
            },
            "member_logits": {
                **dict(continuous_parity["member_logits"]),
                "rationale": (
                    f"absolute-only bound exceeds the {sweep_logit_maximum:.17g} "
                    "maximum produced by faithful Float32 PyTorch batch-size changes; "
                    "relative error is unsuitable for logits near zero"
                ),
            },
            "member_hazard_probability": {
                **dict(continuous_parity["member_hazard_probability"]),
                "rationale": (
                    "retains the reviewed M1 bound; the Research batch-size "
                    f"maximum is {sweep_probability_maximum:.17g}"
                ),
            },
            "ensemble_hazard_probability": {
                **dict(continuous_parity["ensemble_hazard_probability"]),
                "rationale": (
                    "retains the reviewed M1 bound; three-member Float64 "
                    "averaging reduces the observed batch-size maximum to "
                    f"{sweep_ensemble_maximum:.17g}"
                ),
            },
        },
        "exact_parity": {
            "shape_and_dtype": "exact_for_every_layer",
            "threshold_comparison": "probability_greater_than_or_equal_to_0.99",
            "threshold_crossing": "exact",
            "consecutive_threshold_count": "exact",
            "persistence": "five_consecutive_1ms_samples_reset_on_failure",
            "reflex_required": "exact",
            "reflex_onset": "exact",
        },
        "variability_evidence": {
            "source": "non-protected V2_VALIDATION golden slice",
            "window_count": len(windows),
            "reference_batch_size": canonical_batch_size,
            "batch_size_sweep": batch_size_sweep,
            "batch_one_repeated_exact": repeated_batch_one_exact,
            "historical_batch_121_vs_batch_1": {
                "member_logits_max_absolute": _maximum_absolute(
                    member_logits, deployment_logits
                ),
                "member_hazard_probability_max_absolute": _maximum_absolute(
                    member_probabilities, deployment_member_probabilities
                ),
                "ensemble_hazard_probability_max_absolute": _maximum_absolute(
                    ensemble, deployment_ensemble
                ),
                "discrete_runtime_outputs_exact": True,
            },
        },
        "threshold_sensitivity": {
            "threshold": threshold,
            "closest_window_index": closest_index,
            "closest_probability": float(deployment_ensemble[closest_index]),
            "minimum_absolute_margin": minimum_margin,
            "minimum_above_margin": float(
                np.min(deployment_ensemble[above] - threshold)
            ),
            "minimum_below_margin": float(
                np.min(threshold - deployment_ensemble[~above])
            ),
            "permitted_numerical_error_at_closest_probability": permitted_at_closest,
            "margin_to_permitted_error_ratio": minimum_margin / permitted_at_closest,
            "crossing_count": int(np.count_nonzero(above)),
            "reflex_onset_window_indices": np.flatnonzero(
                deployment_reflex_onset
            ).tolist(),
        },
        "m3_inheritance": {
            "float_reference": "canonical batch-one deployment golden",
            "float_contract_must_pass_before_int8_evaluation": True,
            "threshold_persistence_and_final_decisions_remain_exact": True,
            "int8_tolerances": "must_be_defined_and_justified_in_m3",
        },
    }

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
    for name, payload in _runtime_documents(document, packaging_commit).items():
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
            "consecutive_threshold_count": _consecutive_counts(threshold_crossing),
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
        destination / "golden_outputs/deployment_runtime_chain.npz",
        {
            "base_features": base,
            "causal_features": features,
            "consecutive_threshold_count": deployment_counts,
            "ensemble_hazard_probability": deployment_ensemble,
            "member_hazard_probability": deployment_member_probabilities,
            "member_logits": deployment_logits,
            "model_windows": windows,
            "normalized_features": normalized,
            "reflex_onset": deployment_reflex_onset,
            "reflex_required": deployment_reflex_required,
            "threshold_crossing": deployment_threshold_crossing,
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
            "historical_reference": {
                "path": "golden_outputs/runtime_chain.npz",
                "execution_shape": [121, 20, 80],
                "status": "preserved M1 batch-N evidence, not canonical deployment execution",
            },
            "deployment_reference": {
                "path": "golden_outputs/deployment_runtime_chain.npz",
                "execution_shape": [1, 20, 80],
                "one_causal_endpoint_per_invocation": True,
                "contract": "float_numerical_contract.json",
            },
            "decision_probe": (
                "separate synthetic probability trace exercises inclusive "
                "threshold, reset, fifth-sample assertion, and held assertion"
            ),
        },
    )
    _write_json(destination / "float_numerical_contract.json", float_contract)

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
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
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
        model.get("engineering_role") != "DEPLOYMENT_ENGINEERING_REFERENCE_MODEL"
        or model.get("status") != "NON_RELEASE_ENGINEERING_REFERENCE"
        or model.get("release_model") is not False
        or model.get("scientific_status", {}).get("verdict")
        != "MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED"
    ):
        raise ValueError("release scientific/non-release status changed")
    contract = json.loads(
        (root / "float_numerical_contract.json").read_text(encoding="utf-8")
    )
    canonical = contract.get("canonical_execution", {})
    exact = contract.get("exact_parity", {})
    if (
        contract.get("schema_version") != 1
        or contract.get("verdict") != "FLOAT_EXPORT_NUMERICAL_CONTRACT_RESOLVED"
        or contract.get("candidate_id") != manifest["release_id"]
        or contract.get("protected_holdout_access") is not False
        or canonical.get("input_shape") != [1, 20, 80]
        or canonical.get("batch_size") != 1
        or canonical.get("one_causal_endpoint_per_invocation") is not True
        or canonical.get("hidden_state")
        != "zero_initialized_for_every_invocation_no_carry"
        or exact.get("threshold_crossing") != "exact"
        or exact.get("consecutive_threshold_count") != "exact"
        or exact.get("reflex_required") != "exact"
        or exact.get("reflex_onset") != "exact"
    ):
        raise ValueError("deployment Float numerical contract changed")
    return {
        "release_id": manifest["release_id"],
        "release_manifest_sha256": sha256_file(manifest_path),
        "files_verified": len(records),
        "status": "PASS",
    }
