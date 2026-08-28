"""Dense causal fall-risk dataset generation and bounded detector PoC."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import gc
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import yaml

from fastreflex.dataset.loader import Normalizer, WindowSet
from fastreflex.evaluation.stability_ground_truth import lower_body_state_addresses
from fastreflex.evaluation.stability_temporal import (
    BINARY_CLASS_NAMES,
    IMU6_FEATURE_NAMES,
    PRIVILEGED_FULL_STATE,
    REPRESENTATION_FEATURE_NAMES,
    RUNTIME_IMU6,
    TEMPORAL_FULL_STATE_FEATURE_NAMES,
    TemporalRun,
    _file_sha256,
    _protected_hashes,
    binary_auprc,
    binary_auroc,
    fit_train_normalizer,
    predict_fall_probability,
    simulate_temporal_cohort,
)
from fastreflex.evaluation.transition_scenarios import (
    SIGNATURE_FIELDS,
    VALID_FALL,
    VALID_STABLE,
    fusion_regression,
)
from fastreflex.models.baselines import parameter_count
from fastreflex.simulation.g1 import (
    PHYSICS_TIMESTEP_S,
    SENSOR_RATE_HZ,
    load_g1_model,
    load_simulation_config,
)
from fastreflex.training.trainer import load_checkpoint, save_checkpoint, train_model


PELVIS_IMU6 = "PELVIS_IMU6"
DENSE_REPRESENTATIONS = (PRIVILEGED_FULL_STATE, PELVIS_IMU6)
STORAGE_FEATURE_KEY = {
    PRIVILEGED_FULL_STATE: PRIVILEGED_FULL_STATE,
    PELVIS_IMU6: RUNTIME_IMU6,
}
DENSE_FEATURE_NAMES = {
    PRIVILEGED_FULL_STATE: TEMPORAL_FULL_STATE_FEATURE_NAMES,
    PELVIS_IMU6: IMU6_FEATURE_NAMES,
}
DENSE_SIGNATURE_FIELDS = ("source_terrain", *SIGNATURE_FIELDS)


@dataclass(frozen=True)
class DenseBatch:
    """Dense causal windows and provenance kept outside the input tensor."""

    windows: WindowSet
    rows: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ReplayTrace:
    """One run's 1 kHz endpoint probability trace."""

    endpoints: np.ndarray
    probabilities: np.ndarray


class DenseHoldoutGuard:
    """Permit holdout waveform loading once, after validation selection."""

    def __init__(self) -> None:
        self._opened = False
        self._open_count = 0

    def open_once(self) -> None:
        if self._opened or self._open_count:
            raise RuntimeError("dense holdout may be opened exactly once")
        self._opened = True
        self._open_count = 1

    def require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("dense holdout waveform access is sealed")

    @property
    def open_count(self) -> int:
        return self._open_count


def _json_default(value: object) -> object:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=_json_default)
        stream.write("\n")


def _load_yaml(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _split_for_index(index: int) -> str:
    if 1 <= index <= 18:
        return "train"
    if 19 <= index <= 24:
        return "validation"
    if 25 <= index <= 30:
        return "holdout"
    raise ValueError("dense stratum index must be in [1,30]")


def physical_signature(specification: Mapping[str, object]) -> tuple[object, ...]:
    """Return the exact physical condition signature, including source terrain."""
    return tuple(specification[field] for field in DENSE_SIGNATURE_FIELDS)


def _scheduled_condition(
    document: Mapping[str, object],
    source: str,
    target: str,
    role: str,
    index: int,
) -> dict[str, object]:
    schedules = document["dataset"]["deterministic_condition_schedule"]
    schedule = schedules[f"{target}_{role}"]
    source_key = f"{source}_width"
    if target == "ice" and role == "stable":
        width = float(schedule[source_key]["first"]) + (index - 1) * float(
            schedule[source_key]["step"]
        )
        start = float(schedule["patch_start_x_m"])
        sink_pattern = str(schedule["sink_pattern"])
    else:
        anchor_index = (index - 1) % 2
        local_index = (index - 1) // 2
        anchor = schedule["alternating_anchors"][anchor_index]
        start = float(anchor["patch_start_x_m"])
        width = float(anchor[f"{source}_width_first"]) + local_index * float(
            anchor["width_step"]
        )
        sink_pattern = str(
            anchor["sink_pattern"]
            if "sink_pattern" in anchor
            else schedule["sink_pattern"]
        )
    return {
        "patch_start_x_m": round(start, 5),
        "patch_width_m": round(width, 5),
        "slip_pattern": str(schedule["slip_pattern"]),
        "sink_pattern": sink_pattern,
        "sink_severity": str(schedule["sink_severity"]),
        "support_pattern": str(schedule["support_pattern"]),
    }


def generate_dense_specifications(
    document: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Expand the frozen 8x30 matrix and deterministic hard controls."""
    primary: list[dict[str, object]] = []
    for source in document["dataset"]["matrix"]["sources"]:
        for target in document["dataset"]["matrix"]["targets"]:
            for role in document["dataset"]["matrix"]["design_roles"]:
                for index in range(1, 31):
                    condition = _scheduled_condition(
                        document, str(source), str(target), str(role), index
                    )
                    primary.append(
                        {
                            "id": (
                                f"dfr_{str(source)[0]}_{target}_{str(role)[0]}"
                                f"{index:02d}"
                            ),
                            "split": _split_for_index(index),
                            "design_role": str(role),
                            "source_terrain": str(source),
                            "target_terrain": str(target),
                            "speed_mps": float(document["common"]["primary_speed_mps"]),
                            "hard_stable_control": False,
                            **condition,
                        }
                    )
    controls: list[dict[str, object]] = []
    speeds = [
        float(value) for value in document["dataset"]["hard_controls"]["speed_mps"]
    ]
    for source in document["dataset"]["hard_controls"]["sources"]:
        for index, speed in enumerate(speeds, start=1):
            controls.append(
                {
                    "id": f"dfr_control_{str(source)[0]}_{index:02d}",
                    "split": "validation" if index <= 4 else "holdout",
                    "design_role": "stable_control",
                    "source_terrain": str(source),
                    "target_terrain": str(source),
                    "speed_mps": speed,
                    "patch_start_x_m": 0.35,
                    "patch_width_m": 0.75,
                    "slip_pattern": "uniform",
                    "sink_pattern": "uniform",
                    "sink_severity": "moderate",
                    "support_pattern": "balanced_soft",
                    "hard_stable_control": True,
                }
            )
    return primary, controls


def _recursive_scenario_rows(value: object) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    if isinstance(value, Mapping):
        if all(field in value for field in DENSE_SIGNATURE_FIELDS):
            rows.append(value)
        for child in value.values():
            rows.extend(_recursive_scenario_rows(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            rows.extend(_recursive_scenario_rows(child))
    return rows


def validate_dense_design(
    document: Mapping[str, object],
    primary: Sequence[Mapping[str, object]],
    controls: Sequence[Mapping[str, object]],
    repository_root: Path,
) -> dict[str, object]:
    """Fail before simulation if any predeclared dense contract drifted."""
    if document["experiment"]["id"] != "DENSE_FALL_RISK_DATASET_AND_DETECTOR_POC":
        raise ValueError("unsupported dense fall-risk experiment")
    if tuple(document["dense_windows"]["horizons_ms"]) != (200, 100, 50):
        raise ValueError("dense horizon contract changed")
    if tuple(document["dense_windows"]["histories_ms"]) != (50, 100):
        raise ValueError("dense history contract changed")
    if int(document["dense_windows"]["training_endpoint_stride_ms"]) != 10:
        raise ValueError("dense training stride changed")
    if tuple(document["model"]["seeds"]) != (20260828, 20260829, 20260830):
        raise ValueError("dense seed contract changed")
    if document["model"]["family"] != "gru" or bool(
        document["model"]["architecture"]["bidirectional"]
    ):
        raise ValueError("dense PoC requires the existing unidirectional GRU")
    if int(document["threshold_calibration"]["persistence_ms"]) != 10:
        raise ValueError("dense replay persistence changed")
    if len(primary) != 240 or len(controls) != 16:
        raise ValueError("dense matrix must contain 240 primary and 16 controls")
    ids = [str(row["id"]) for row in [*primary, *controls]]
    if len(ids) != len(set(ids)):
        raise ValueError("dense run ids are duplicated")
    signatures = [physical_signature(row) for row in primary]
    duplicate_count = len(signatures) - len(set(signatures))
    if duplicate_count:
        raise ValueError("dense physical condition signatures are duplicated")
    if any(float(row["speed_mps"]) != 0.25 for row in primary):
        raise ValueError("primary dense speed confound was not removed")
    split_sets = {
        split: {str(row["id"]) for row in primary if row["split"] == split}
        for split in ("train", "validation", "holdout")
    }
    if {key: len(value) for key, value in split_sets.items()} != {
        "train": 144,
        "validation": 48,
        "holdout": 48,
    }:
        raise ValueError("dense split counts changed")
    if any(
        split_sets[left] & split_sets[right]
        for left, right in (
            ("train", "validation"),
            ("train", "holdout"),
            ("validation", "holdout"),
        )
    ):
        raise ValueError("dense split is not run-disjoint")
    strata: dict[str, dict[str, int]] = {}
    for split in split_sets:
        counts: dict[str, int] = {}
        for row in primary:
            if row["split"] != split:
                continue
            key = (
                f"{row['source_terrain']}_{row['target_terrain']}_{row['design_role']}"
            )
            counts[key] = counts.get(key, 0) + 1
        expected = 18 if split == "train" else 6
        if len(counts) != 8 or any(count != expected for count in counts.values()):
            raise ValueError(f"{split} does not contain its frozen 8-stratum matrix")
        strata[split] = counts

    previous = set()
    for relative in document["source"]["freshness_reference_configs"]:
        source = _load_yaml(repository_root / str(relative))
        previous.update(
            physical_signature(row) for row in _recursive_scenario_rows(source)
        )
    overlap = sorted(set(signatures) & previous, key=str)
    if overlap:
        raise ValueError("dense conditions overlap a prior exact physical signature")
    if tuple(DENSE_FEATURE_NAMES[PRIVILEGED_FULL_STATE]) != tuple(
        REPRESENTATION_FEATURE_NAMES[PRIVILEGED_FULL_STATE]
    ):
        raise ValueError("privileged temporal schema drifted")
    forbidden = tuple(
        str(value) for value in document["common"]["forbidden_model_inputs"]
    )
    for names in DENSE_FEATURE_NAMES.values():
        if any(token in name for token in forbidden for name in names):
            raise ValueError("dense model schema contains a forbidden channel")
    return {
        "passed": True,
        "primary_runs": len(primary),
        "hard_controls": len(controls),
        "split_counts": {key: len(value) for key, value in split_sets.items()},
        "strata": strata,
        "duplicate_signatures": duplicate_count,
        "prior_signature_overlap": len(overlap),
        "primary_speed_values_mps": sorted(
            {float(row["speed_mps"]) for row in primary}
        ),
        "representation_dimensions": {
            name: len(DENSE_FEATURE_NAMES[name]) for name in DENSE_REPRESENTATIONS
        },
        "terrain_or_fall_channels_in_tensor": False,
    }


def dense_horizon_label(
    fall_sample: int | None, endpoint_sample: int, horizon_ms: int
) -> int:
    """Label whether an actual fall occurs strictly after now and within H."""
    if horizon_ms not in (200, 100, 50):
        raise ValueError("dense horizon must be one of 200/100/50 ms")
    if fall_sample is None:
        return 0
    delta = int(fall_sample) - int(endpoint_sample)
    return int(0 < delta <= horizon_ms)


def dense_positive_endpoints(
    run: TemporalRun, horizon_ms: int, stride_ms: int
) -> np.ndarray:
    """Return pre-fall endpoints in [fall-H, fall) at the frozen stride."""
    if run.fall_sample is None:
        return np.empty(0, dtype=np.int64)
    endpoints = np.arange(
        run.fall_sample - horizon_ms, run.fall_sample, stride_ms, dtype=np.int64
    )
    return endpoints[
        (endpoints >= run.first_contact_sample) & (endpoints < run.fall_sample)
    ]


def _evenly_spaced(values: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or not len(values):
        return np.empty(0, dtype=np.int64)
    if len(values) <= count:
        return values.astype(np.int64, copy=True)
    positions = np.linspace(0, len(values) - 1, count, dtype=np.int64)
    return values[positions].astype(np.int64, copy=False)


def dense_early_negative_endpoints(
    run: TemporalRun,
    horizon_ms: int,
    history_ms: int,
    stride_ms: int,
    safety_gap_ms: int,
    count: int,
) -> np.ndarray:
    """Sample deterministic fall-run negatives strictly before H+safety gap."""
    if run.fall_sample is None:
        return np.empty(0, dtype=np.int64)
    first = run.first_contact_sample + history_ms - 1
    stop = run.fall_sample - horizon_ms - safety_gap_ms
    candidates = np.arange(first, stop, stride_ms, dtype=np.int64)
    endpoints = _evenly_spaced(candidates, count)
    if any(
        dense_horizon_label(run.fall_sample, int(value), horizon_ms)
        for value in endpoints
    ):
        raise ValueError("fall-run early negative crossed the horizon boundary")
    return endpoints


def _causal_indices(endpoint: int, history_ms: int) -> np.ndarray:
    first = int(endpoint) - int(history_ms) + 1
    if first < 0:
        raise ValueError("endpoint has insufficient causal history")
    return np.arange(first, int(endpoint) + 1, dtype=np.int64)


def _normalizer_transform(normalizer: Normalizer, values: np.ndarray) -> np.ndarray:
    result = normalizer.transform(values)
    if not np.all(np.isfinite(result)):
        raise ValueError("normalized dense window is nonfinite")
    return result.astype(np.float32, copy=False)


def build_dense_windows(
    runs: Mapping[str, TemporalRun],
    run_ids: Sequence[str],
    representation: str,
    horizon_ms: int,
    history_ms: int,
    stride_ms: int,
    safety_gap_ms: int,
    normalizer: Normalizer,
) -> DenseBatch:
    """Build capped dense progression windows with elapsed-time stable matching."""
    selected = [runs[str(run_id)] for run_id in run_ids if str(run_id) in runs]
    falling = sorted(
        (run for run in selected if run.outcome == VALID_FALL),
        key=lambda run: run.run_id,
    )
    stable = sorted(
        (run for run in selected if run.outcome == VALID_STABLE),
        key=lambda run: run.run_id,
    )
    inputs: list[np.ndarray] = []
    targets: list[int] = []
    sources: list[str] = []
    endpoints_all: list[int] = []
    rows: list[dict[str, object]] = []
    stable_usage: dict[str, int] = {run.run_id: 0 for run in stable}
    cap = int(np.ceil(horizon_ms / stride_ms))

    def append(
        run: TemporalRun, endpoint: int, label: int, kind: str, matched: str | None
    ) -> None:
        indices = _causal_indices(endpoint, history_ms)
        if indices[0] < run.first_contact_sample:
            raise ValueError("dense primary window begins before target contact")
        raw = run.features[STORAGE_FEATURE_KEY[representation]][indices]
        if run.fall_sample is not None and indices[-1] >= run.fall_sample:
            raise ValueError(
                "dense positive/negative window contains fall or post-fall state"
            )
        inputs.append(_normalizer_transform(normalizer, raw))
        targets.append(label)
        sources.append(run.run_id)
        endpoints_all.append(endpoint)
        rows.append(
            {
                "run_id": run.run_id,
                "endpoint_sample": endpoint,
                "label": label,
                "kind": kind,
                "source_terrain": run.source_terrain,
                "target_terrain": run.target_terrain,
                "observed_outcome": run.outcome,
                "matched_fall_run_id": matched,
                "elapsed_since_contact_ms": endpoint - run.first_contact_sample,
            }
        )

    by_stratum_stable: dict[tuple[str, str], list[TemporalRun]] = {}
    for run in stable:
        by_stratum_stable.setdefault(
            (run.source_terrain, run.target_terrain), []
        ).append(run)
    fall_position: dict[tuple[str, str], int] = {}
    for fall_run in falling:
        positive = dense_positive_endpoints(fall_run, horizon_ms, stride_ms)
        positive = positive[positive >= fall_run.first_contact_sample + history_ms - 1][
            :cap
        ]
        early = dense_early_negative_endpoints(
            fall_run,
            horizon_ms,
            history_ms,
            stride_ms,
            safety_gap_ms,
            len(positive),
        )[:cap]
        for endpoint in positive:
            append(fall_run, int(endpoint), 1, "fall_positive", None)
        for endpoint in early:
            append(fall_run, int(endpoint), 0, "fall_early_negative", None)

        key = (fall_run.source_terrain, fall_run.target_terrain)
        candidates = by_stratum_stable.get(key, [])
        if not candidates:
            continue
        start = fall_position.get(key, 0)
        ordered = candidates[start:] + candidates[:start]
        stable_run = next(
            (
                candidate
                for candidate in ordered
                if stable_usage[candidate.run_id] < cap
            ),
            None,
        )
        if stable_run is None:
            continue
        fall_position[key] = (candidates.index(stable_run) + 1) % len(candidates)
        elapsed = positive - fall_run.first_contact_sample
        stable_endpoints = stable_run.first_contact_sample + elapsed
        valid = stable_endpoints < len(stable_run.gait_phase)
        stable_endpoints = stable_endpoints[valid]
        remaining = cap - stable_usage[stable_run.run_id]
        for endpoint in stable_endpoints[:remaining]:
            append(
                stable_run, int(endpoint), 0, "stable_matched_negative", fall_run.run_id
            )
            stable_usage[stable_run.run_id] += 1
    if not inputs or len(set(targets)) != 2:
        raise ValueError("dense windows must contain both classes")
    target_array = np.asarray(targets, dtype=np.int64)
    return DenseBatch(
        windows=WindowSet(
            inputs=np.stack(inputs).astype(np.float32),
            targets=target_array,
            run_ids=np.asarray(sources, dtype=str),
            endpoint_samples=np.asarray(endpoints_all, dtype=np.int64),
            available_by_class=(
                int(np.count_nonzero(target_array == 0)),
                int(np.count_nonzero(target_array == 1)),
                0,
            ),
        ),
        rows=tuple(rows),
    )


def threshold_grid(
    start: float = 0.10, stop: float = 0.90, step: float = 0.02
) -> tuple[float, ...]:
    """Return the exact predeclared validation-only threshold grid."""
    count = int(round((stop - start) / step)) + 1
    return tuple(round(start + index * step, 2) for index in range(count))


def sustained_confirmation_sample(
    endpoints: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    persistence_samples: int,
) -> int | None:
    """Return the causal confirmation endpoint of the first sustained event."""
    endpoint_values = np.asarray(endpoints, dtype=np.int64)
    risk = np.asarray(probabilities, dtype=np.float64) >= float(threshold)
    if endpoint_values.shape != risk.shape or persistence_samples <= 0:
        raise ValueError("invalid sustained-event inputs")
    count = 0
    previous: int | None = None
    for endpoint, active in zip(endpoint_values, risk):
        if previous is None or endpoint != previous + 1 or not active:
            count = 1 if active else 0
        else:
            count += 1
        if count >= persistence_samples:
            return int(endpoint)
        previous = int(endpoint)
    return None


def classify_detection(
    fall_sample: int | None, detection_sample: int | None, horizon_ms: int
) -> str:
    """Classify the first sustained event against bounded horizon semantics."""
    if fall_sample is None:
        return "STABLE_FP" if detection_sample is not None else "STABLE_TN"
    if detection_sample is None:
        return "FALL_MISSED"
    if detection_sample < fall_sample - horizon_ms:
        return "FALL_PREMATURE_FP"
    if detection_sample < fall_sample:
        return "FALL_VALID_DETECTION"
    return "FALL_MISSED"


def _lead_distribution(values: Sequence[int]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("min", "p10", "median", "p95", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def evaluate_run_level(
    runs: Mapping[str, TemporalRun],
    traces: Mapping[str, ReplayTrace],
    horizon_ms: int,
    threshold: float,
    persistence_ms: int,
    control_ids: Sequence[str] = (),
) -> dict[str, object]:
    """Evaluate causal first-event semantics with runs as independent units."""
    control_set = set(control_ids)
    stable_rows = []
    fall_rows = []
    control_rows = []
    leads = []
    for run_id, trace in traces.items():
        run = runs[run_id]
        detection = sustained_confirmation_sample(
            trace.endpoints, trace.probabilities, threshold, persistence_ms
        )
        classification = classify_detection(run.fall_sample, detection, horizon_ms)
        row = {
            "run_id": run_id,
            "source_terrain": run.source_terrain,
            "target_terrain": run.target_terrain,
            "detection_sample": detection,
            "classification": classification,
            "slip": run.slip_sample is not None,
            "sink": run.sink_sample is not None,
            "maximum_support_deformation_m": run.maximum_support_deformation_m,
        }
        if run_id in control_set:
            control_rows.append(row)
        elif run.outcome == VALID_STABLE:
            if detection is not None:
                above = trace.probabilities >= threshold
                row["fp_duration_ms"] = int(np.count_nonzero(above))
                row["first_fp_elapsed_since_contact_ms"] = (
                    detection - run.first_contact_sample
                )
            stable_rows.append(row)
        elif run.outcome == VALID_FALL:
            if classification == "FALL_VALID_DETECTION":
                assert run.fall_sample is not None and detection is not None
                lead = int(run.fall_sample - detection)
                row["prediction_lead_ms"] = lead
                leads.append(lead)
            fall_rows.append(row)
    stable_fp = sum(row["classification"] == "STABLE_FP" for row in stable_rows)
    valid = sum(row["classification"] == "FALL_VALID_DETECTION" for row in fall_rows)
    premature = sum(row["classification"] == "FALL_PREMATURE_FP" for row in fall_rows)
    control_fp = sum(row["classification"] == "STABLE_FP" for row in control_rows)

    def recall(field: str, value: str) -> float:
        selected = [row for row in fall_rows if row[field] == value]
        if not selected:
            return 0.0
        return float(
            sum(row["classification"] == "FALL_VALID_DETECTION" for row in selected)
            / len(selected)
        )

    stable_count = len(stable_rows)
    fall_count = len(fall_rows)
    control_count = len(control_rows)
    return {
        "threshold": float(threshold),
        "stable_runs": stable_count,
        "fall_runs": fall_count,
        "hard_control_runs": control_count,
        "fall_recall": 0.0 if not fall_count else valid / fall_count,
        "stable_specificity": 0.0
        if not stable_count
        else 1.0 - stable_fp / stable_count,
        "stable_fp_rate": 0.0 if not stable_count else stable_fp / stable_count,
        "premature_fall_fp_rate": 0.0 if not fall_count else premature / fall_count,
        "hard_control_fp_rate": 0.0
        if not control_count
        else control_fp / control_count,
        "ice_fall_recall": recall("target_terrain", "ice"),
        "sand_fall_recall": recall("target_terrain", "sand"),
        "concrete_origin_fall_recall": recall("source_terrain", "concrete"),
        "marble_origin_fall_recall": recall("source_terrain", "marble"),
        "lead_ms": _lead_distribution(leads),
        "stable_false_positive_runs": [
            row for row in stable_rows if row["classification"] == "STABLE_FP"
        ],
        "premature_fall_runs": [
            row for row in fall_rows if row["classification"] == "FALL_PREMATURE_FP"
        ],
        "valid_fall_runs": [
            row for row in fall_rows if row["classification"] == "FALL_VALID_DETECTION"
        ],
        "missed_fall_runs": [
            row for row in fall_rows if row["classification"] == "FALL_MISSED"
        ],
        "hard_control_false_positive_runs": [
            row for row in control_rows if row["classification"] == "STABLE_FP"
        ],
    }


def validation_gate_results(
    metrics: Mapping[str, object], gates: Mapping[str, object], horizon_ms: int
) -> dict[str, bool]:
    median = metrics["lead_ms"]["median"]
    return {
        "fall_recall": float(metrics["fall_recall"]) >= float(gates["fall_recall_min"]),
        "stable_specificity": float(metrics["stable_specificity"])
        >= float(gates["stable_specificity_min"]),
        "ice_fall_recall": float(metrics["ice_fall_recall"])
        >= float(gates["ice_fall_recall_min"]),
        "sand_fall_recall": float(metrics["sand_fall_recall"])
        >= float(gates["sand_fall_recall_min"]),
        "premature_fall_fp_rate": float(metrics["premature_fall_fp_rate"])
        <= float(gates["premature_fall_fp_rate_max"]),
        "hard_control_fp_rate": float(metrics["hard_control_fp_rate"])
        <= float(gates["hard_control_fp_rate_max"]),
        "median_prediction_lead": median is not None
        and float(median)
        >= float(gates["median_prediction_lead_fraction_min"]) * horizon_ms,
    }


def holdout_gate_results(
    metrics: Mapping[str, object], gates: Mapping[str, object]
) -> dict[str, bool]:
    return {
        "fall_recall": float(metrics["fall_recall"]) >= float(gates["fall_recall_min"]),
        "stable_specificity": float(metrics["stable_specificity"])
        >= float(gates["stable_specificity_min"]),
        "ice_fall_recall": float(metrics["ice_fall_recall"])
        >= float(gates["ice_fall_recall_min"]),
        "sand_fall_recall": float(metrics["sand_fall_recall"])
        >= float(gates["sand_fall_recall_min"]),
        "premature_fall_fp_rate": float(metrics["premature_fall_fp_rate"])
        <= float(gates["premature_fall_fp_rate_max"]),
        "hard_control_fp_rate": float(metrics["hard_control_fp_rate"])
        <= float(gates["hard_control_fp_rate_max"]),
    }


def select_validation_threshold(
    evaluations: Sequence[Mapping[str, object]],
    feasibility: Mapping[str, object],
) -> dict[str, object]:
    """Apply the frozen feasibility constraints and lexicographic priority."""
    feasible = [
        row
        for row in evaluations
        if float(row["metrics"]["stable_fp_rate"])
        <= float(feasibility["stable_transition_fp_rate_max"])
        and float(row["metrics"]["hard_control_fp_rate"])
        <= float(feasibility["hard_control_fp_rate_max"])
        and float(row["metrics"]["premature_fall_fp_rate"])
        <= float(feasibility["premature_fall_fp_rate_max"])
    ]

    def rank(row: Mapping[str, object]) -> tuple[float, float, float, float]:
        metrics = row["metrics"]
        median = metrics["lead_ms"]["median"]
        return (
            float(metrics["fall_recall"]),
            -1.0 if median is None else float(median),
            min(float(metrics["ice_fall_recall"]), float(metrics["sand_fall_recall"])),
            float(row["threshold"]),
        )

    if feasible:
        selected = max(feasible, key=rank)
        return {
            "selected": selected,
            "reason": "frozen_feasibility_then_recall_lead_terrain_threshold",
            "feasible_threshold_count": len(feasible),
        }
    diagnostic = max(
        evaluations,
        key=lambda row: (
            sum(
                (
                    float(row["metrics"]["stable_fp_rate"])
                    <= float(feasibility["stable_transition_fp_rate_max"]),
                    float(row["metrics"]["hard_control_fp_rate"])
                    <= float(feasibility["hard_control_fp_rate_max"]),
                    float(row["metrics"]["premature_fall_fp_rate"])
                    <= float(feasibility["premature_fall_fp_rate_max"]),
                )
            ),
            *rank(row),
        ),
    )
    return {
        "selected": None,
        "diagnostic_best": diagnostic,
        "reason": "no_threshold_met_frozen_false_alarm_constraints",
        "feasible_threshold_count": 0,
    }


def select_dense_candidate(
    candidates: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Select longest passing horizon, then the shorter passing history."""
    passing = [row for row in candidates if bool(row["validation_passed"])]
    if passing:
        for horizon in (200, 100, 50):
            at_horizon = [row for row in passing if int(row["horizon_ms"]) == horizon]
            if at_horizon:
                selected = min(at_horizon, key=lambda row: int(row["history_ms"]))
                return {
                    "selected": {
                        "horizon_ms": int(selected["horizon_ms"]),
                        "history_ms": int(selected["history_ms"]),
                        "threshold": float(selected["operating_point"]["threshold"]),
                    },
                    "reason": "longest_reliable_horizon_then_shortest_history",
                }
    diagnostic = max(
        candidates,
        key=lambda row: (
            sum(bool(value) for value in row["validation_gates"].values()),
            float(row["operating_point"]["fall_recall"]),
            float(row["operating_point"]["stable_specificity"]),
            -1.0
            if row["operating_point"]["lead_ms"]["median"] is None
            else float(row["operating_point"]["lead_ms"]["median"]),
            int(row["horizon_ms"]),
            -int(row["history_ms"]),
        ),
    )
    return {
        "selected": None,
        "reason": "no_validation_candidate_met_all_run_level_gates",
        "diagnostic_best": {
            "horizon_ms": int(diagnostic["horizon_ms"]),
            "history_ms": int(diagnostic["history_ms"]),
            "threshold": float(diagnostic["operating_point"]["threshold"]),
        },
    }


def _run_to_npz(path: Path, run: TemporalRun) -> None:
    np.savez_compressed(
        path,
        timestamp_us=run.timestamp_us,
        pelvis_imu6=run.features[RUNTIME_IMU6],
        privileged_full_state=run.features[PRIVILEGED_FULL_STATE],
        gait_phase=run.gait_phase,
        first_target_contact_sample=np.asarray(
            run.first_contact_sample, dtype=np.int64
        ),
        fall_sample=np.asarray(
            -1 if run.fall_sample is None else run.fall_sample, dtype=np.int64
        ),
        censor_sample=np.asarray(len(run.timestamp_us), dtype=np.int64),
    )


def _manifest_row(
    path: Path, run: TemporalRun, specification: Mapping[str, object]
) -> dict[str, object]:
    return {
        "run_id": run.run_id,
        "file": path.name,
        "file_sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
        "split": run.split,
        "source_terrain": run.source_terrain,
        "target_terrain": run.target_terrain,
        "speed_mps": run.speed_mps,
        "design_role": specification["design_role"],
        "observed_outcome": run.outcome,
        "hard_stable_control": run.hard_stable_control,
        "first_target_contact_sample": run.first_contact_sample,
        "fall_sample": run.fall_sample,
        "censor_sample": len(run.timestamp_us),
        "slip_sample": run.slip_sample,
        "sink_sample": run.sink_sample,
        "maximum_support_deformation_m": run.maximum_support_deformation_m,
        "physical_signature": list(physical_signature(specification)),
    }


def _load_run(path: Path, row: Mapping[str, object]) -> TemporalRun:
    with np.load(path, allow_pickle=False) as payload:
        imu = np.asarray(payload["pelvis_imu6"], dtype=np.float32)
        privileged = np.asarray(payload["privileged_full_state"], dtype=np.float32)
        timestamp = np.asarray(payload["timestamp_us"], dtype=np.int64)
        phase = np.asarray(payload["gait_phase"], dtype=np.int8)
        contact = int(payload["first_target_contact_sample"])
        fall_raw = int(payload["fall_sample"])
    if (
        imu.shape != (len(timestamp), 6)
        or privileged.shape != (len(timestamp), 40)
        or phase.shape != (len(timestamp),)
        or not np.all(np.isfinite(imu))
        or not np.all(np.isfinite(privileged))
    ):
        raise ValueError(f"dense run {row['run_id']} contains invalid tensors")
    return TemporalRun(
        run_id=str(row["run_id"]),
        split=str(row["split"]),
        source_terrain=str(row["source_terrain"]),
        target_terrain=str(row["target_terrain"]),
        speed_mps=float(row["speed_mps"]),
        outcome=str(row["observed_outcome"]),
        first_contact_sample=contact,
        fall_sample=None if fall_raw < 0 else fall_raw,
        gait_phase=phase,
        features={PRIVILEGED_FULL_STATE: privileged, RUNTIME_IMU6: imu},
        timestamp_us=timestamp,
        slip_sample=None if row["slip_sample"] is None else int(row["slip_sample"]),
        sink_sample=None if row["sink_sample"] is None else int(row["sink_sample"]),
        maximum_support_deformation_m=float(row["maximum_support_deformation_m"]),
        hard_stable_control=bool(row["hard_stable_control"]),
    )


def load_dataset_runs(
    dataset_path: Path,
    manifest: Mapping[str, object],
    splits: Sequence[str],
    *,
    holdout_guard: DenseHoldoutGuard | None = None,
) -> dict[str, TemporalRun]:
    """Load only requested split waveforms, enforcing the holdout seal."""
    if "holdout" in splits:
        if holdout_guard is None:
            raise RuntimeError("holdout loading requires an explicit guard")
        holdout_guard.require_open()
    runs = {}
    for row in manifest["runs"]:
        if str(row["split"]) not in splits:
            continue
        path = dataset_path / str(row["file"])
        if _file_sha256(path) != str(row["file_sha256"]):
            raise ValueError(f"dataset run integrity failed: {path.name}")
        run = _load_run(path, row)
        runs[run.run_id] = run
    return runs


def _cohort_summary(
    runs: Mapping[str, TemporalRun], invalid: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    primary = [run for run in runs.values() if not run.hard_stable_control]

    def counts(selected: Sequence[TemporalRun]) -> dict[str, int]:
        return {
            "stable": sum(run.outcome == VALID_STABLE for run in selected),
            "fall": sum(run.outcome == VALID_FALL for run in selected),
            "total": len(selected),
        }

    return {
        "primary": counts(primary),
        "hard_controls": counts(
            [run for run in runs.values() if run.hard_stable_control]
        ),
        "by_terrain": {
            terrain: counts([run for run in primary if run.target_terrain == terrain])
            for terrain in ("ice", "sand")
        },
        "by_source": {
            source: counts([run for run in primary if run.source_terrain == source])
            for source in ("concrete", "marble")
        },
        "by_split": {
            split: counts([run for run in primary if run.split == split])
            for split in ("train", "validation", "holdout")
        },
        "invalid": list(invalid),
        "invalid_count": len(invalid),
        "pretransition_fall_count": sum(
            row["outcome"] == "INVALID_PRETRANSITION" for row in invalid
        ),
    }


def readiness_results(
    cohort: Mapping[str, object],
    design: Mapping[str, object],
    gates: Mapping[str, object],
) -> dict[str, bool]:
    primary = cohort["primary"]
    ice = cohort["by_terrain"]["ice"]
    sand = cohort["by_terrain"]["sand"]
    concrete = cohort["by_source"]["concrete"]
    marble = cohort["by_source"]["marble"]
    return {
        "valid_runs": int(primary["total"]) >= int(gates["valid_runs_min"]),
        "observed_stable": int(primary["stable"]) >= int(gates["observed_stable_min"]),
        "observed_fall": int(primary["fall"]) >= int(gates["observed_fall_min"]),
        "ice_stable": int(ice["stable"]) >= int(gates["ice_stable_min"]),
        "ice_fall": int(ice["fall"]) >= int(gates["ice_fall_min"]),
        "sand_stable": int(sand["stable"]) >= int(gates["sand_stable_min"]),
        "sand_fall": int(sand["fall"]) >= int(gates["sand_fall_min"]),
        "concrete_origin_fall": int(concrete["fall"])
        >= int(gates["concrete_origin_fall_min"]),
        "marble_origin_fall": int(marble["fall"])
        >= int(gates["marble_origin_fall_min"]),
        "pretransition_fall": int(cohort["pretransition_fall_count"])
        <= int(gates["pretransition_fall_max"]),
        "duplicate_signature": int(design["duplicate_signatures"])
        <= int(gates["duplicate_signature_max"]),
        "split_overlap": True,
        "nonfinite_input": True,
        "label_leakage": bool(not design["terrain_or_fall_channels_in_tensor"]),
    }


def _checkpoint_paths(
    artifact_path: Path,
    representation: str,
    horizon_ms: int,
    history_ms: int,
    seeds: Sequence[int],
) -> list[Path]:
    return [
        artifact_path
        / "checkpoints"
        / f"{representation.lower()}_h{horizon_ms}_history{history_ms}_seed{seed}.pt"
        for seed in seeds
    ]


def _predict_replay(
    run: TemporalRun,
    representation: str,
    history_ms: int,
    normalizer: Normalizer,
    models: Sequence[torch.nn.Module],
    mode: str = "original",
    endpoint_keep_ms: int = 20,
    batch_size: int = 1024,
) -> ReplayTrace:
    start = run.first_contact_sample + history_ms - 1
    stop = run.fall_sample if run.fall_sample is not None else len(run.gait_phase)
    endpoints = np.arange(start, stop, dtype=np.int64)
    outputs: list[np.ndarray] = []
    offsets = np.arange(history_ms - 1, -1, -1, dtype=np.int64)
    features = run.features[STORAGE_FEATURE_KEY[representation]]
    for first in range(0, len(endpoints), batch_size):
        selected = endpoints[first : first + batch_size]
        indices = selected[:, None] - offsets[None, :]
        windows = _normalizer_transform(normalizer, features[indices])
        if mode == "reversed":
            windows = windows[:, ::-1].copy()
        elif mode == "endpoint_only":
            windows = windows.copy()
            windows[:, : max(0, history_ms - endpoint_keep_ms)] = 0.0
        elif mode != "original":
            raise ValueError(f"unsupported temporal diagnostic mode: {mode}")
        probabilities = []
        tensor = torch.from_numpy(windows)
        with torch.no_grad():
            for model in models:
                model.eval()
                probabilities.append(
                    torch.softmax(model(tensor), dim=1)[:, 1].cpu().numpy()
                )
        outputs.append(np.mean(np.stack(probabilities), axis=0))
    return ReplayTrace(
        endpoints=endpoints,
        probabilities=np.concatenate(outputs).astype(np.float64, copy=False),
    )


def _load_models(paths: Sequence[Path]) -> list[torch.nn.Module]:
    return [load_checkpoint(path)[0] for path in paths]


def _replay_many(
    runs: Mapping[str, TemporalRun],
    representation: str,
    history_ms: int,
    normalizer: Normalizer,
    checkpoint_paths: Sequence[Path],
    mode: str = "original",
    endpoint_keep_ms: int = 20,
) -> dict[str, ReplayTrace]:
    models = _load_models(checkpoint_paths)
    traces = {
        run_id: _predict_replay(
            run,
            representation,
            history_ms,
            normalizer,
            models,
            mode,
            endpoint_keep_ms,
        )
        for run_id, run in runs.items()
    }
    del models
    return traces


def _window_diagnostics(
    batch: DenseBatch, probabilities: np.ndarray
) -> dict[str, float]:
    return {
        "auroc": binary_auroc(batch.windows.targets, probabilities),
        "auprc": binary_auprc(batch.windows.targets, probabilities),
        "window_count": len(probabilities),
    }


def _candidate_by_choice(
    candidates: Sequence[Mapping[str, object]], choice: Mapping[str, object]
) -> Mapping[str, object]:
    found = [
        row
        for row in candidates
        if int(row["horizon_ms"]) == int(choice["horizon_ms"])
        and int(row["history_ms"]) == int(choice["history_ms"])
    ]
    if len(found) != 1:
        raise ValueError("dense selected candidate is missing or duplicated")
    return found[0]


def _train_candidates(
    document: Mapping[str, object],
    runs: Mapping[str, TemporalRun],
    normalizers: Mapping[str, Normalizer],
    primary_ids: Mapping[str, Sequence[str]],
    control_ids: Sequence[str],
    artifact_path: Path,
    progress: Callable[[str], None],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, dict[str, object]]]:
    config = document["model"]
    dense = document["dense_windows"]
    seeds = [int(value) for value in config["seeds"]]
    candidates_by_representation: dict[str, list[dict[str, object]]] = {}
    selections = {}
    for representation in DENSE_REPRESENTATIONS:
        candidates = []
        for horizon_ms in dense["horizons_ms"]:
            for history_ms in dense["histories_ms"]:
                train_batch = build_dense_windows(
                    runs,
                    primary_ids["train"],
                    representation,
                    int(horizon_ms),
                    int(history_ms),
                    int(dense["training_endpoint_stride_ms"]),
                    int(dense["fall_early_negative_safety_gap_ms"]),
                    normalizers[representation],
                )
                validation_batch = build_dense_windows(
                    runs,
                    primary_ids["validation"],
                    representation,
                    int(horizon_ms),
                    int(history_ms),
                    int(dense["training_endpoint_stride_ms"]),
                    int(dense["fall_early_negative_safety_gap_ms"]),
                    normalizers[representation],
                )
                checkpoints = _checkpoint_paths(
                    artifact_path,
                    representation,
                    int(horizon_ms),
                    int(history_ms),
                    seeds,
                )
                training_rows = []
                for seed, checkpoint in zip(seeds, checkpoints):
                    model, training = train_model(
                        "gru",
                        int(history_ms),
                        train_batch.windows,
                        validation_batch.windows,
                        seed,
                        batch_size=int(config["batch_size"]),
                        max_epochs=int(config["max_epochs"]),
                        patience=int(config["early_stopping_patience"]),
                        learning_rate=float(config["learning_rate"]),
                        class_names=BINARY_CLASS_NAMES,
                        selection_metric="validation_loss",
                    )
                    save_checkpoint(
                        checkpoint,
                        model,
                        "gru",
                        int(history_ms),
                        seed,
                        training,
                        input_channels=len(DENSE_FEATURE_NAMES[representation]),
                        class_names=BINARY_CLASS_NAMES,
                    )
                    training_rows.append(
                        {
                            "seed": seed,
                            "best_epoch": training.best_epoch,
                            "epochs_completed": training.epochs_completed,
                            "best_validation_cross_entropy": min(
                                row["validation_cross_entropy"]
                                for row in training.history
                            ),
                        }
                    )
                    del model
                models = _load_models(checkpoints)
                validation_probability_parts = []
                for model in models:
                    validation_probability_parts.append(
                        predict_fall_probability(model, validation_batch.windows)
                    )
                validation_window_scores = np.mean(
                    np.stack(validation_probability_parts), axis=0
                )
                del models
                replay_ids = [
                    *primary_ids["validation"],
                    *control_ids,
                ]
                replay_runs = {
                    run_id: runs[run_id] for run_id in replay_ids if run_id in runs
                }
                traces = _replay_many(
                    replay_runs,
                    representation,
                    int(history_ms),
                    normalizers[representation],
                    checkpoints,
                )
                evaluations = []
                for threshold in threshold_grid(
                    **document["threshold_calibration"]["grid"]
                ):
                    metrics = evaluate_run_level(
                        replay_runs,
                        traces,
                        int(horizon_ms),
                        threshold,
                        int(document["threshold_calibration"]["persistence_ms"]),
                        control_ids,
                    )
                    evaluations.append({"threshold": threshold, "metrics": metrics})
                calibration = select_validation_threshold(
                    evaluations, document["threshold_calibration"]["feasibility"]
                )
                operating = calibration["selected"] or calibration["diagnostic_best"]
                operating_metrics = operating["metrics"]
                gates = validation_gate_results(
                    operating_metrics, document["validation_gates"], int(horizon_ms)
                )
                candidate = {
                    "horizon_ms": int(horizon_ms),
                    "history_ms": int(history_ms),
                    "train_windows": len(train_batch.windows),
                    "validation_windows": len(validation_batch.windows),
                    "train_independent_runs": len(set(train_batch.windows.run_ids)),
                    "validation_independent_runs": len(
                        set(validation_batch.windows.run_ids)
                    ),
                    "parameter_count": parameter_count(
                        load_checkpoint(checkpoints[0])[0]
                    ),
                    "epoch_selection": "validation_cross_entropy_minimum",
                    "training": training_rows,
                    "window_diagnostics": _window_diagnostics(
                        validation_batch, validation_window_scores
                    ),
                    "threshold_calibration": {
                        "selected_threshold": None
                        if calibration["selected"] is None
                        else float(calibration["selected"]["threshold"]),
                        "feasible_threshold_count": calibration[
                            "feasible_threshold_count"
                        ],
                        "reason": calibration["reason"],
                    },
                    "operating_point": operating_metrics,
                    "validation_gates": gates,
                    "validation_passed": calibration["selected"] is not None
                    and all(gates.values()),
                    "checkpoint_paths": [
                        str(path.relative_to(artifact_path)) for path in checkpoints
                    ],
                }
                candidates.append(candidate)
                progress(
                    f"DENSE GRU {representation} H={horizon_ms} history={history_ms} "
                    f"threshold={operating['threshold']:.2f} recall={operating_metrics['fall_recall']:.3f} "
                    f"specificity={operating_metrics['stable_specificity']:.3f} "
                    f"passed={candidate['validation_passed']}"
                )
                del train_batch, validation_batch, traces
                gc.collect()
        candidates_by_representation[representation] = candidates
        selections[representation] = select_dense_candidate(candidates)
    return candidates_by_representation, selections


def _evaluate_holdout(
    document: Mapping[str, object],
    runs: Mapping[str, TemporalRun],
    candidates: Mapping[str, Sequence[Mapping[str, object]]],
    selections: Mapping[str, Mapping[str, object]],
    normalizers: Mapping[str, Normalizer],
    primary_ids: Sequence[str],
    control_ids: Sequence[str],
    artifact_path: Path,
) -> dict[str, object]:
    results = {}
    for representation in DENSE_REPRESENTATIONS:
        selected = selections[representation]["selected"]
        if selected is None:
            results[representation] = {
                "performed": False,
                "reason": "no_validation_selection",
            }
            continue
        candidate = _candidate_by_choice(candidates[representation], selected)
        paths = [artifact_path / str(value) for value in candidate["checkpoint_paths"]]
        selected_runs = {
            run_id: runs[run_id]
            for run_id in [*primary_ids, *control_ids]
            if run_id in runs
        }
        traces = _replay_many(
            selected_runs,
            representation,
            int(selected["history_ms"]),
            normalizers[representation],
            paths,
        )
        metrics = evaluate_run_level(
            selected_runs,
            traces,
            int(selected["horizon_ms"]),
            float(selected["threshold"]),
            int(document["threshold_calibration"]["persistence_ms"]),
            control_ids,
        )
        gates = holdout_gate_results(metrics, document["holdout"]["gates"])
        results[representation] = {
            "performed": True,
            **dict(selected),
            "metrics": metrics,
            "gates": gates,
            "passed": all(gates.values()),
        }
    return results


def _temporal_diagnostics(
    document: Mapping[str, object],
    runs: Mapping[str, TemporalRun],
    candidates: Mapping[str, Sequence[Mapping[str, object]]],
    selections: Mapping[str, Mapping[str, object]],
    normalizers: Mapping[str, Normalizer],
    validation_ids: Sequence[str],
    control_ids: Sequence[str],
    artifact_path: Path,
) -> dict[str, object]:
    results = {}
    selected_runs = {
        run_id: runs[run_id]
        for run_id in [*validation_ids, *control_ids]
        if run_id in runs
    }
    for representation in DENSE_REPRESENTATIONS:
        choice = (
            selections[representation]["selected"]
            or selections[representation]["diagnostic_best"]
        )
        candidate = _candidate_by_choice(candidates[representation], choice)
        paths = [artifact_path / str(value) for value in candidate["checkpoint_paths"]]
        modes = {}
        for mode in ("original", "reversed", "endpoint_only"):
            traces = _replay_many(
                selected_runs,
                representation,
                int(choice["history_ms"]),
                normalizers[representation],
                paths,
                mode,
                int(document["temporal_diagnostics"]["endpoint_keep_ms"]),
            )
            modes[mode] = evaluate_run_level(
                selected_runs,
                traces,
                int(choice["horizon_ms"]),
                float(choice["threshold"]),
                int(document["threshold_calibration"]["persistence_ms"]),
                control_ids,
            )
        original = modes["original"]
        reversed_metrics = modes["reversed"]
        results[representation] = {
            "candidate": dict(choice),
            "original": original,
            "reversed": reversed_metrics,
            "endpoint_only_last_20ms": modes["endpoint_only"],
            "time_order_diagnostic": (
                "MODEL_NOT_USING_TEMPORAL_ORDER_STRONGLY"
                if abs(
                    float(original["fall_recall"])
                    - float(reversed_metrics["fall_recall"])
                )
                <= 0.05
                and abs(
                    float(original["stable_specificity"])
                    - float(reversed_metrics["stable_specificity"])
                )
                <= 0.05
                else "TEMPORAL_ORDER_AFFECTS_RUN_LEVEL_METRICS"
            ),
            "selection_influence": False,
        }
    return results


def _dataset_manifest(
    document: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    dataset_path: Path,
) -> tuple[dict[str, object], str]:
    manifest = {
        "schema_version": 1,
        "dataset_id": document["dataset"]["dataset_id"],
        "created_at": f"{document['experiment']['date']}T00:00:00+09:00",
        "source_commit": document["experiment"]["source_commit_at_start"],
        "policy_sha256": document["source"]["policy_sha256"],
        "simulator_config_sha256": document["source"]["simulator_config_sha256"],
        "scenario_calibration_reference": {
            "path": document["source"]["scenario_calibration_config"],
            "sha256": document["source"]["scenario_calibration_sha256"],
        },
        "frozen_terrain_mechanics": {
            "ice_friction": [0.05, 0.001, 0.00001],
            "sand_mild": {
                "travel_m": 0.020,
                "stiffness_n_per_m": 12000.0,
                "damping_n_s_per_m": 490.0,
            },
            "sand_moderate": {
                "travel_m": 0.040,
                "stiffness_n_per_m": 7000.0,
                "damping_n_s_per_m": 374.0,
            },
        },
        "run_count": len(rows),
        "runs": list(rows),
    }
    path = dataset_path / "manifest.json"
    _write_json(path, manifest)
    sha = _file_sha256(path)
    (dataset_path / "manifest.sha256").write_text(
        f"{sha}  manifest.json\n", encoding="utf-8"
    )
    return manifest, sha


def _dataset_summary(
    dataset_path: Path, manifest: Mapping[str, object], manifest_sha: str
) -> dict[str, object]:
    return {
        "dataset_id": manifest["dataset_id"],
        "path": str(dataset_path),
        "run_files": len(manifest["runs"]),
        "size_bytes": sum(int(row["size_bytes"]) for row in manifest["runs"]),
        "manifest_sha256": manifest_sha,
    }


def _verdict(
    selections: Mapping[str, Mapping[str, object]],
    holdout: Mapping[str, Mapping[str, object]],
) -> str:
    supported = {
        representation: (
            selections[representation]["selected"] is not None
            and bool(holdout.get(representation, {}).get("passed", False))
        )
        for representation in DENSE_REPRESENTATIONS
    }
    supported_horizons = [
        int(selections[representation]["selected"]["horizon_ms"])
        for representation in DENSE_REPRESENTATIONS
        if supported[representation]
    ]
    if supported_horizons and max(supported_horizons) == 50:
        return "DENSE_FALL_RISK_LIMITED_TO_SHORT_HORIZON"
    if supported[PELVIS_IMU6]:
        return "DENSE_FALL_RISK_SUPPORTED_IMU6"
    if supported[PRIVILEGED_FULL_STATE]:
        return "DENSE_FALL_RISK_SUPPORTED_PRIVILEGED_ONLY"
    return "DENSE_FALL_RISK_NOT_SUPPORTED"


def run_dense_fall_risk_detector_poc(
    config_path: Path,
    repository_root: Path,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Generate the frozen dataset, then train/select/evaluate without leakage."""
    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    document = _load_yaml(config_path)
    primary_specs, control_specs = generate_dense_specifications(document)
    design = validate_dense_design(
        document, primary_specs, control_specs, repository_root
    )
    if PHYSICS_TIMESTEP_S != float(document["common"]["physics_timestep_s"]):
        raise ValueError("dense physics timestep differs from canonical value")
    if SENSOR_RATE_HZ != int(document["common"]["sensor_rate_hz"]):
        raise ValueError("dense sensor rate differs from canonical value")
    source = document["source"]
    for key, sha_key in (
        ("simulator_config", "simulator_config_sha256"),
        ("scenario_calibration_config", "scenario_calibration_sha256"),
        ("policy_path", "policy_sha256"),
    ):
        path = repository_root / str(source[key])
        if not path.is_file() or _file_sha256(path) != str(source[sha_key]):
            raise ValueError(f"dense provenance source changed: {key}")

    dataset_path = (repository_root / str(document["dataset"]["path"])).resolve()
    artifact_path = (repository_root / str(document["artifacts"]["path"])).resolve()
    dataset_path.relative_to(repository_root)
    artifact_path.relative_to(repository_root)
    if dataset_path.exists() and any(dataset_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite dense dataset: {dataset_path}")
    if artifact_path.exists() and any(artifact_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite dense artifacts: {artifact_path}")
    dataset_path.mkdir(parents=True, exist_ok=True)
    artifact_path.mkdir(parents=True, exist_ok=True)
    protected_paths = [
        str(value) for value in document["terrain_regression"]["protected_paths"]
    ]
    terrain_before = _protected_hashes(repository_root, protected_paths)
    base = load_simulation_config(repository_root / str(source["simulator_config"]))
    model, _ = load_g1_model("concrete")
    qpos_addresses, qvel_addresses = lower_body_state_addresses(model)
    adapter = {
        "common": {"duration_s": document["common"]["duration_s"]},
        "cohort": {"scenario_gate": document["common"]["scenario_gate"]},
    }
    generated, invalid = simulate_temporal_cohort(
        base,
        [*primary_specs, *control_specs],
        repository_root / str(source["policy_path"]),
        adapter,
        qpos_addresses,
        qvel_addresses,
        progress,
    )
    specs_by_id = {str(row["id"]): row for row in [*primary_specs, *control_specs]}
    manifest_rows = []
    for run_id in sorted(generated):
        run = generated[run_id]
        path = dataset_path / f"{run_id}.npz"
        _run_to_npz(path, run)
        manifest_rows.append(_manifest_row(path, run, specs_by_id[run_id]))
    cohort = _cohort_summary(generated, invalid)
    manifest, manifest_sha = _dataset_manifest(document, manifest_rows, dataset_path)
    dataset_summary = _dataset_summary(dataset_path, manifest, manifest_sha)
    readiness = readiness_results(cohort, design, document["readiness_gates"])
    del generated
    gc.collect()
    terrain_after_dataset = _protected_hashes(repository_root, protected_paths)
    if not all(readiness.values()):
        metrics = {
            "experiment": document["experiment"],
            "design": design,
            "dataset": dataset_summary,
            "cohort": cohort,
            "readiness": {"passed": False, "gates": readiness},
            "training": {"performed": False},
            "holdout": {"performed": False, "guard_open_count": 0},
            "terrain_regression": {
                "passed": terrain_before == terrain_after_dataset,
                "before": terrain_before,
                "after": terrain_after_dataset,
            },
            "fusion_regression": fusion_regression(),
            "verdict": "FALL_RISK_DATASET_NEEDS_REVISION",
        }
        _write_json(artifact_path / "metrics.json", metrics)
        return artifact_path, metrics

    primary_ids = {
        split: [str(row["id"]) for row in primary_specs if row["split"] == split]
        for split in ("train", "validation", "holdout")
    }
    control_ids = {
        split: [str(row["id"]) for row in control_specs if row["split"] == split]
        for split in ("validation", "holdout")
    }
    development_runs = load_dataset_runs(
        dataset_path, manifest, ("train", "validation")
    )
    normalizers = {
        representation: fit_train_normalizer(
            development_runs,
            [run_id for run_id in primary_ids["train"] if run_id in development_runs],
            STORAGE_FEATURE_KEY[representation],
            int(document["normalization"]["per_run_sample_cap"]),
            float(document["normalization"]["standard_deviation_floor"]),
        )
        for representation in DENSE_REPRESENTATIONS
    }
    _write_json(
        artifact_path / "train_only_normalization.json",
        {name: normalizer.to_dict() for name, normalizer in normalizers.items()},
    )
    candidates, selections = _train_candidates(
        document,
        development_runs,
        normalizers,
        primary_ids,
        control_ids["validation"],
        artifact_path,
        progress,
    )
    _write_json(
        artifact_path / "selection_before_holdout.json",
        {"selections": selections, "holdout_opened": False},
    )
    diagnostics = _temporal_diagnostics(
        document,
        development_runs,
        candidates,
        selections,
        normalizers,
        primary_ids["validation"],
        control_ids["validation"],
        artifact_path,
    )
    guard = DenseHoldoutGuard()
    if any(selections[name]["selected"] is not None for name in DENSE_REPRESENTATIONS):
        guard.open_once()
        holdout_runs = load_dataset_runs(
            dataset_path, manifest, ("holdout",), holdout_guard=guard
        )
        holdout = _evaluate_holdout(
            document,
            holdout_runs,
            candidates,
            selections,
            normalizers,
            primary_ids["holdout"],
            control_ids["holdout"],
            artifact_path,
        )
    else:
        holdout = {
            name: {"performed": False, "reason": "no_validation_selection"}
            for name in DENSE_REPRESENTATIONS
        }
    terrain_after = _protected_hashes(repository_root, protected_paths)
    verdict = _verdict(selections, holdout)
    metrics = {
        "experiment": document["experiment"],
        "design": design,
        "dataset": dataset_summary,
        "cohort": cohort,
        "readiness": {"passed": True, "gates": readiness},
        "normalization": {name: value.to_dict() for name, value in normalizers.items()},
        "validation": {
            "performed": True,
            "candidates": candidates,
            "selections": selections,
            "holdout_sealed_during_selection": True,
        },
        "holdout": {
            "performed": any(row["performed"] for row in holdout.values()),
            "guard_open_count": guard.open_count,
            "reselection_performed": False,
            "representations": holdout,
        },
        "temporal_diagnostics": diagnostics,
        "terrain_regression": {
            "passed": terrain_before == terrain_after,
            "before": terrain_before,
            "after": terrain_after,
            "retrained": False,
        },
        "fusion_regression": fusion_regression(),
        "causality": {
            "passed": True,
            "future_samples_in_tensor": False,
            "fall_or_time_to_fall_in_tensor": False,
            "post_fall_samples_in_tensor": False,
            "replay_stride_ms": 1,
            "confirmation_persistence_ms": 10,
        },
        "runtime_boundary": {
            "q_dq_runtime_augmentation": False,
            "fsr_stability_input": False,
            "foot_imu_stability_input": False,
            "runtime_enum_changed": False,
        },
        "verdict": verdict,
    }
    _write_json(artifact_path / "metrics.json", metrics)
    progress(
        json.dumps(
            {"selections": selections, "holdout": holdout, "verdict": verdict},
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    )
    return artifact_path, metrics
