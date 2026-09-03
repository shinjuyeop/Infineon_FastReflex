"""Discovery-only analysis for the calibrated Sand benign study."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fastreflex.dataset.hazard import HazardRun, canonical_sha256
from fastreflex.dataset.loader import sha256_file
from fastreflex.dataset.sand_mild_calibration import (
    load_mild_recalibrated_discovery_payload,
    load_mild_recalibrated_manifest,
    verify_mild_recalibrated_dataset,
)
from fastreflex.evaluation.hazard import (
    load_hazard_normalizer,
    reflex_onset_samples,
    replay_hazard_run,
)
from fastreflex.features import (
    HAZARD_FEATURE_SCHEMA_SHA256,
    extract_hazard_features,
    feature_schema_hash,
)
from fastreflex.models.checkpoint import load_checkpoint


DISCOVERY_SPLIT = "MILD_RECALIBRATED_DISCOVERY"
CONFIRMATION_SPLIT = "MILD_RECALIBRATED_CONFIRMATION"
STRICT_BENIGN = "STRICT_BENIGN"
SUPPORT_GROUPS = {"ordinary_support_control", "delayed_support_control"}
HISTORY_MS = 20
EPSILON = 1.0e-8


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.floating, float)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("analysis result contains a nonfinite float")
        return result
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    """Write deterministic, finite JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_value(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _optional_sample(value: object) -> int | None:
    return None if value is None or int(value) < 0 else int(value)


def _first_per_foot(onset: np.ndarray) -> tuple[int | None, int | None]:
    values = np.asarray(onset, dtype=bool)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("physical onset trace must have shape [samples,2]")
    result: list[int | None] = []
    for foot in range(2):
        indices = np.flatnonzero(values[:, foot])
        result.append(None if not len(indices) else int(indices[0]))
    return result[0], result[1]


def hazard_run_from_discovery(
    row: Mapping[str, Any], payload: Mapping[str, np.ndarray]
) -> HazardRun:
    """Adapt one verified recalibrated-Discovery payload to canonical replay."""
    timestamp = np.asarray(payload["timestamp_us"], dtype=np.int64)
    imu = np.asarray(payload["pelvis_imu6"], dtype=np.float32)
    fsr = np.asarray(payload["foot_fsr8"], dtype=np.float32)
    target = np.asarray(payload["target_terrain_contact"], dtype=bool)
    touchdown = np.asarray(payload["target_terrain_touchdown"], dtype=bool)
    loaded = np.asarray(payload["loaded_contact"], dtype=bool)
    spread = np.asarray(payload["support_surface_spread_m"], dtype=np.float32)
    displacement = np.asarray(
        payload["support_surface_max_displacement_m"], dtype=np.float32
    )
    drift = np.asarray(payload["tangential_anchor_drift_m"], dtype=np.float32)
    velocity = np.asarray(payload["tangential_velocity_mps"], dtype=np.float32)
    samples = len(timestamp)
    target_indices = np.flatnonzero(np.any(target, axis=1))
    touchdown_indices = np.flatnonzero(np.any(touchdown, axis=1))
    if not len(target_indices) or not len(touchdown_indices):
        raise ValueError(f"Discovery run lacks target contact: {row['run_id']}")
    censor = int(np.asarray(payload["censor_sample"]).item())
    first_fall = _optional_sample(np.asarray(payload["first_fall_sample"]).item())
    slip = _first_per_foot(np.asarray(payload["established_slip_onset"]))
    support_trace = np.asarray(payload["deformable_sink_onset"], dtype=bool)
    support = _first_per_foot(support_trace)
    event_values = [value for value in (*slip, *support) if value is not None]
    if imu.shape != (samples, 6) or fsr.shape != (samples, 8):
        raise ValueError(f"Discovery runtime tensor shape changed: {row['run_id']}")
    if any(
        array.shape != (samples, 2)
        for array in (loaded, spread, displacement, drift, velocity)
    ):
        raise ValueError(f"Discovery physical tensor shape changed: {row['run_id']}")
    if not np.all(np.isfinite(imu)) or not np.all(np.isfinite(fsr)):
        raise ValueError(f"Discovery runtime tensor is nonfinite: {row['run_id']}")
    fusion = np.concatenate((imu, fsr), axis=1).astype(np.float32, copy=False)
    return HazardRun(
        run_id=str(row["run_id"]),
        split=str(row["split"]),
        source_terrain=str(row["source_terrain"]),
        target_terrain=str(row["target_terrain"]),
        design_role=str(row["group"]),
        first_contact_sample=int(target_indices[0]),
        first_touchdown_sample=int(touchdown_indices[0]),
        censor_sample=censor,
        outcome_diagnostic=str(row["objective_physical_outcome"]),
        fall_sample_diagnostic=first_fall,
        features={"PELVIS_IMU6": imu, "PELVIS_IMU6_FSR8": fusion},
        timestamp_us=timestamp,
        slip_event_samples_per_foot=slip,
        support_event_samples_per_foot=support,
        event_sample=None if not event_values else min(event_values),
        event_type=str(row["actual_hazard_label"]),
        hard_stable_control=False,
        drift_m=drift,
        tangential_velocity_mps=velocity,
        support_spread_m=spread,
        support_max_displacement_m=displacement,
        loaded_contact=loaded,
        sink_pattern=str(row["sink_pattern"]),
        support_pattern=str(row["support_pattern"]),
    )


def benign_anchor(payload: Mapping[str, np.ndarray], *, baseline_ms: int = 200) -> int:
    """Return the earliest maximum frozen model-independent benign anchor."""
    imu = np.asarray(payload["pelvis_imu6"], dtype=np.float64)
    target = np.asarray(payload["target_terrain_contact"], dtype=bool)
    censor = int(np.asarray(payload["censor_sample"]).item())
    contacts = np.flatnonzero(np.any(target[:censor], axis=1))
    if not len(contacts):
        raise ValueError("benign anchor requires a pre-censor target contact")
    first = int(contacts[0])
    last = int(contacts[-1])
    baseline_start = max(0, first - baseline_ms)
    if first <= baseline_start:
        raise ValueError("benign anchor lacks a precontact baseline")
    baseline = np.mean(imu[baseline_start:first, :3], axis=0)
    lower = max(HISTORY_MS - 1, first)
    upper = min(censor - 1, last + 500)
    scores = []
    for endpoint in range(lower, upper + 1):
        window = imu[endpoint - HISTORY_MS + 1 : endpoint + 1, :3]
        scores.append(float(np.sqrt(np.mean((window - baseline) ** 2))))
    values = np.asarray(scores, dtype=np.float64)
    maximum = float(np.max(values))
    # isclose avoids platform-specific last-bit changes while preserving earliest.
    selected = int(np.flatnonzero(np.isclose(values, maximum, rtol=0.0, atol=1e-12))[0])
    return lower + selected


def support_anchor(row: Mapping[str, Any]) -> int:
    """Return the manifest-frozen I1 primary Support anchor."""
    value = row["i1_summary"]["first_sample"]
    if value is None or int(value) < HISTORY_MS - 1:
        raise ValueError(f"Support run lacks a valid I1 anchor: {row['run_id']}")
    return int(value)


def fsr_contact_vector(payload: Mapping[str, np.ndarray], anchor: int) -> np.ndarray:
    """Build the frozen 39D realizable FSR/contact summary."""
    fsr = np.asarray(payload["foot_fsr8"], dtype=np.float64)
    window = fsr[anchor - HISTORY_MS + 1 : anchor + 1]
    if window.shape != (HISTORY_MS, 8):
        raise ValueError("FSR anchor window is incomplete")
    left = np.sum(window[:, :4], axis=1)
    right = np.sum(window[:, 4:], axis=1)
    imbalance = (left - right) / (left + right + EPSILON)
    fsr_contact = np.column_stack((left > 0.0, right > 0.0)).astype(np.float64)
    return np.concatenate(
        (
            np.mean(window, axis=0),
            np.max(window, axis=0),
            np.std(window, axis=0),
            window[-1] - window[0],
            np.asarray(
                [np.mean(imbalance), np.std(imbalance), imbalance[-1]],
                dtype=np.float64,
            ),
            np.mean(fsr_contact, axis=0),
            fsr_contact[-1],
        )
    )


def privileged_oracle_vector(
    payload: Mapping[str, np.ndarray], anchor: int
) -> np.ndarray:
    """Build a separate 16D simulator-oracle diagnostic vector."""
    spread = np.asarray(payload["support_surface_spread_m"], dtype=np.float64)
    displacement = np.asarray(
        payload["support_surface_max_displacement_m"], dtype=np.float64
    )
    loaded = np.asarray(payload["loaded_contact"], dtype=bool).astype(np.float64)
    parts: list[np.ndarray] = []
    for values in (spread, displacement):
        window = values[anchor - HISTORY_MS + 1 : anchor + 1]
        parts.extend((np.mean(window, axis=0), np.max(window, axis=0), window[-1]))
    loaded_window = loaded[anchor - HISTORY_MS + 1 : anchor + 1]
    parts.extend((np.mean(loaded_window, axis=0), loaded_window[-1]))
    return np.concatenate(parts)


def _pooled_standardize(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    array = np.asarray(values, dtype=np.float64)
    mean = np.mean(array, axis=0)
    raw_std = np.std(array, axis=0)
    std = np.maximum(raw_std, EPSILON)
    result = (array - mean) / std
    return result, {
        "dimension": int(array.shape[1]),
        "epsilon": EPSILON,
        "constant_dimensions": int(np.sum(raw_std < EPSILON)),
        "mean_sha256": hashlib.sha256(mean.tobytes()).hexdigest(),
        "std_sha256": hashlib.sha256(std.tobytes()).hexdigest(),
    }


def _pairwise_distances(values: np.ndarray) -> np.ndarray:
    difference = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.sum(difference * difference, axis=2))


def _quantiles(values: Sequence[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("distance quantiles require at least one value")
    return {
        name: float(np.percentile(array, quantile))
        for name, quantile in (
            ("p05", 5),
            ("p25", 25),
            ("median", 50),
            ("p75", 75),
            ("p95", 95),
        )
    }


def _balanced_mean(values: np.ndarray, labels: np.ndarray) -> float:
    return float(
        np.mean([np.mean(values[labels == label]) for label in (0, 1)])
    )


def _pca_diagnostic(values: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    centered = values - np.mean(values, axis=0)
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    components = components.copy()
    for index in range(len(components)):
        pivot = int(np.argmax(np.abs(components[index])))
        if components[index, pivot] < 0.0:
            components[index] *= -1.0
    total = float(np.sum(singular * singular))
    explained = (singular * singular) / total if total > 0.0 else np.zeros_like(singular)
    component_count = min(2, len(components))
    projected = centered @ components[:component_count].T
    overlaps: list[float] = []
    intervals: list[dict[str, list[float]]] = []
    for component in range(component_count):
        bounds = []
        for label in (0, 1):
            selected = projected[labels == label, component]
            bounds.append(
                (
                    float(np.percentile(selected, 2.5)),
                    float(np.percentile(selected, 97.5)),
                )
            )
        intersection = max(0.0, min(x[1] for x in bounds) - max(x[0] for x in bounds))
        union = max(x[1] for x in bounds) - min(x[0] for x in bounds)
        overlaps.append(0.0 if union <= 0.0 else intersection / union)
        intervals.append(
            {"sand": list(bounds[0]), "support": list(bounds[1])}
        )
    return {
        "method": "full_SVD_largest_absolute_loading_positive",
        "explained_variance_ratio_first_two": [
            float(value) for value in explained[:component_count]
        ],
        "projection_95pct_interval_jaccard_overlap": overlaps,
        "projection_95pct_intervals": intervals,
    }


def separability_metrics(
    raw_values: np.ndarray,
    labels: Sequence[int],
    *,
    run_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute the frozen run-balanced two-class descriptive metrics."""
    values, scaling = _pooled_standardize(np.asarray(raw_values, dtype=np.float64))
    classes = np.asarray(labels, dtype=np.int64)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError("separability requires both Sand and Support")
    if run_ids is not None and len(set(run_ids)) != len(run_ids):
        raise ValueError("run-disjoint analysis received duplicate run IDs")
    distance = _pairwise_distances(values)
    centroids = [np.mean(values[classes == label], axis=0) for label in (0, 1)]
    within_squared = [
        np.mean(np.sum((values[classes == label] - centroids[label]) ** 2, axis=1))
        for label in (0, 1)
    ]
    within_rms = float(np.sqrt(np.mean(within_squared)))
    centroid_distance = float(np.linalg.norm(centroids[0] - centroids[1]))
    agreement: dict[str, float] = {}
    nearest_ratios = np.empty(len(values), dtype=np.float64)
    mixing = np.empty(len(values), dtype=np.float64)
    predictions: dict[int, np.ndarray] = {}
    for k in (1, 5):
        predicted = np.empty(len(values), dtype=np.int64)
        for query in range(len(values)):
            order = np.argsort(distance[query], kind="stable")
            neighbors = order[order != query][:k]
            predicted[query] = int(np.sum(classes[neighbors]) > k / 2)
        predictions[k] = predicted
        agreement[f"balanced_{k}nn_agreement"] = _balanced_mean(
            (predicted == classes).astype(np.float64), classes
        )
    for query in range(len(values)):
        same = np.flatnonzero((classes == classes[query]) & (np.arange(len(values)) != query))
        opposite = np.flatnonzero(classes != classes[query])
        nearest_ratios[query] = np.min(distance[query, opposite]) / max(
            float(np.min(distance[query, same])), EPSILON
        )
        order = np.argsort(distance[query], kind="stable")
        neighbors = order[order != query][:5]
        mixing[query] = np.mean(classes[neighbors] != classes[query])
    radii = [
        float(
            np.percentile(
                np.linalg.norm(values[classes == label] - centroids[label], axis=1),
                95,
            )
        )
        for label in (0, 1)
    ]
    sand_in_support = float(
        np.mean(np.linalg.norm(values[classes == 0] - centroids[1], axis=1) <= radii[1])
    )
    support_in_sand = float(
        np.mean(np.linalg.norm(values[classes == 1] - centroids[0], axis=1) <= radii[0])
    )
    within_sand = distance[np.ix_(classes == 0, classes == 0)]
    within_support = distance[np.ix_(classes == 1, classes == 1)]
    within_sand = within_sand[np.triu_indices(np.sum(classes == 0), 1)]
    within_support = within_support[np.triu_indices(np.sum(classes == 1), 1)]
    between = distance[np.ix_(classes == 0, classes == 1)].ravel()
    ratio_by_class = [float(np.median(nearest_ratios[classes == label])) for label in (0, 1)]
    return {
        "population": {
            "sand": int(np.sum(classes == 0)),
            "support": int(np.sum(classes == 1)),
            "one_vector_per_run": True,
            "run_ids_sha256": canonical_sha256(list(run_ids)) if run_ids else None,
        },
        "scaling": scaling,
        "centroid_separation": centroid_distance / max(within_rms, EPSILON),
        "centroid_distance": centroid_distance,
        "within_group_rms": within_rms,
        **agreement,
        "median_nearest_opposite_to_same_ratio": float(np.mean(ratio_by_class)),
        "nearest_opposite_to_same_ratio_class_medians": {
            "sand": ratio_by_class[0],
            "support": ratio_by_class[1],
        },
        "local_opposite_class_mixing": _balanced_mean(mixing, classes),
        "bidirectional_95pct_radius_inclusion": float(
            np.mean((sand_in_support, support_in_sand))
        ),
        "sand_in_support_95pct_radius": sand_in_support,
        "support_in_sand_95pct_radius": support_in_sand,
        "distance_quantiles": {
            "within_sand": _quantiles(within_sand),
            "within_support": _quantiles(within_support),
            "between": _quantiles(between),
        },
        "pca": _pca_diagnostic(values, classes),
    }


def _distribution(values: Sequence[float | int | None]) -> dict[str, float | None]:
    array = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(array):
        return {
            key: None for key in ("minimum", "median", "p75", "p90", "p95", "maximum")
        }
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _longest_streak(values: np.ndarray, threshold: float) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if float(value) >= threshold else 0
        best = max(best, current)
    return best


def _phase(row: Mapping[str, Any]) -> str:
    return str(row["physical_signature"]["precontact_phase_20ms"])


def _topology(row: Mapping[str, Any]) -> str:
    value = str(row["sink_pattern"])
    return value.removeprefix("transition_").upper()


def _start_stratum(row: Mapping[str, Any]) -> str:
    value = float(row["patch_start_x_m"])
    if value < 0.305:
        return "EARLY"
    if value < 0.355:
        return "MID"
    return "LATE"


def _width_stratum(row: Mapping[str, Any]) -> str:
    value = float(row["patch_width_m"])
    if value < 0.695:
        return "NARROW"
    if value < 0.760:
        return "MEDIUM"
    return "WIDE"


def _entry_timing_stratum(row: Mapping[str, Any]) -> str:
    value = int(row["physical_signature"]["first_target_contact_ms"])
    if value < 2500:
        return "EARLY"
    if value < 4500:
        return "MID"
    return "LATE"


def _exposure_stratum(row: Mapping[str, Any]) -> str:
    diagnostic = row.get("mild_physical_diagnostic")
    value = int(
        diagnostic["cumulative_loaded_sand_exposure_ms"]
        if isinstance(diagnostic, Mapping)
        else row["target_contact_summary"]["duration_ms_before_censor"]
    )
    if value < 2800:
        return "LOW"
    if value < 3600:
        return "MID"
    return "HIGH"


def factor_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return only factors frozen before model replay."""
    diagnostic = row.get("mild_physical_diagnostic")
    exposure = int(
        diagnostic["cumulative_loaded_sand_exposure_ms"]
        if isinstance(diagnostic, Mapping)
        else row["target_contact_summary"]["duration_ms_before_censor"]
    )
    return {
        "source": str(row["source_terrain"]).upper(),
        "speed": f"{float(row['speed_mps']):.2f}",
        "start_stratum": _start_stratum(row),
        "width_stratum": _width_stratum(row),
        "transition_topology": _topology(row),
        "actual_entry_phase": _phase(row),
        "actual_severity": str(row["actual_benign_severity"]),
        "realization_id": str(row["realization_id"]),
        "entry_timing_stratum": _entry_timing_stratum(row),
        "exposure_stratum": _exposure_stratum(row),
        "exposure_ms": exposure,
        "patch_start_x_m": float(row["patch_start_x_m"]),
        "patch_width_m": float(row["patch_width_m"]),
    }


def _cramers_v(levels: Sequence[str], adverse: Sequence[bool]) -> float:
    level_names = sorted(set(levels))
    table = np.asarray(
        [
            [
                sum(level == selected and flag == outcome for level, flag in zip(levels, adverse))
                for outcome in (False, True)
            ]
            for selected in level_names
        ],
        dtype=np.float64,
    )
    total = float(np.sum(table))
    expected = np.sum(table, axis=1, keepdims=True) @ np.sum(table, axis=0, keepdims=True) / total
    chi_square = float(np.sum(np.where(expected > 0.0, (table - expected) ** 2 / expected, 0.0)))
    denominator = total * min(table.shape[0] - 1, table.shape[1] - 1)
    return 0.0 if denominator <= 0.0 else float(np.sqrt(chi_square / denominator))


def factor_localization(
    benign_results: Sequence[Mapping[str, Any]],
    *,
    factors: Sequence[str],
    minimum_level_n: int,
    fraction_range_min: float,
    cramers_v_min: float,
) -> dict[str, Any]:
    """Apply the predeclared model-margin localization rule."""
    factor_results: dict[str, Any] = {}
    localized_factors: list[str] = []
    partial_signal = False
    for factor in factors:
        counts: dict[str, list[bool]] = defaultdict(list)
        for row in benign_results:
            counts[str(row["factors"][factor])].append(bool(row["adverse_margin"]))
        eligible = {key: value for key, value in counts.items() if len(value) >= minimum_level_n}
        if len(eligible) >= 2:
            fractions = {key: float(np.mean(value)) for key, value in eligible.items()}
            level_values = [key for key, values in eligible.items() for _ in values]
            adverse_values = [flag for values in eligible.values() for flag in values]
            fraction_range = max(fractions.values()) - min(fractions.values())
            cramers_v = _cramers_v(level_values, adverse_values)
            passed = fraction_range >= fraction_range_min and cramers_v >= cramers_v_min
            partial_signal = partial_signal or fraction_range >= fraction_range_min or cramers_v >= cramers_v_min
        else:
            fractions = {key: float(np.mean(value)) for key, value in eligible.items()}
            fraction_range = None
            cramers_v = None
            passed = False
        if passed:
            localized_factors.append(factor)
        factor_results[factor] = {
            "all_level_counts": {key: len(value) for key, value in sorted(counts.items())},
            "eligible_level_adverse_fraction": dict(sorted(fractions.items())),
            "adverse_fraction_range": fraction_range,
            "cramers_v": cramers_v,
            "passes": passed,
        }
    adverse_count = sum(bool(row["adverse_margin"]) for row in benign_results)
    false_reflex_count = sum(bool(row["reflex"]) for row in benign_results)
    adverse_fraction = adverse_count / len(benign_results)
    adverse_rows = [row for row in benign_results if row["adverse_margin"]]
    sources = Counter(str(row["factors"]["source"]) for row in adverse_rows)
    speeds = Counter(str(row["factors"]["speed"]) for row in adverse_rows)
    systematic = (
        adverse_fraction >= 0.20
        and len([value for value in sources.values() if value >= 2]) >= 2
        and len([value for value in speeds.values() if value >= 2]) >= 2
    )
    if localized_factors:
        status = "STRONGLY_METADATA_LOCALIZED"
    elif partial_signal:
        status = "PARTIALLY_METADATA_LOCALIZED"
    elif systematic:
        status = "BROAD_ACROSS_DOMAIN"
    elif false_reflex_count == 0 and adverse_count:
        status = "NO_FAILURE_BUT_LOW_MARGIN_BROAD"
    elif false_reflex_count == 0:
        status = "NO_FAILURE_AND_HEALTHY_MARGIN"
    else:
        status = "BROAD_ACROSS_DOMAIN"
    return {
        "adverse_definition": "reflex_or_max_probability_at_least_0.95",
        "adverse_count": adverse_count,
        "adverse_fraction": adverse_fraction,
        "false_reflex_count": false_reflex_count,
        "adverse_sources": dict(sorted(sources.items())),
        "adverse_speeds": dict(sorted(speeds.items())),
        "systematic_adverse_pattern": systematic,
        "factors": factor_results,
        "localized_factors": localized_factors,
        "metadata_localization": bool(localized_factors),
        "status": status,
    }


def _margin_bin(max_probability: float, reflex: bool) -> str:
    if reflex:
        return "REFLEX"
    if max_probability >= 0.99:
        return "GE_0.99_STREAK_LT_5MS"
    if max_probability >= 0.95:
        return "[0.95,0.99)"
    if max_probability >= 0.90:
        return "[0.90,0.95)"
    return "LT_0.90"


def summarize_benign(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    false_positive = sum(bool(row["reflex"]) for row in values)
    return {
        "n": len(values),
        "tn": len(values) - false_positive,
        "fp": false_positive,
        "specificity": None if not values else (len(values) - false_positive) / len(values),
        "max_probability": _distribution([float(row["max_probability"]) for row in values]),
        "reflex": false_positive,
        "near_threshold": sum(bool(row["adverse_margin"]) for row in values),
        "margin_bins": dict(sorted(Counter(str(row["margin_bin"]) for row in values).items())),
    }


def summarize_support(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    correct = sum(bool(row["support_correct"]) for row in values)
    return {
        "n": len(values),
        "correct": correct,
        "recall": None if not values else correct / len(values),
        "premature_pre_i1": sum(bool(row["pre_i1_reflex"]) for row in values),
        "i1_to_reflex_ms": _distribution([row["i1_to_reflex_ms"] for row in values]),
        "reflex_to_support_ms": _distribution(
            [row["reflex_to_support_ms"] for row in values]
        ),
    }


def _subset_summary(
    rows: Sequence[Mapping[str, Any]], factor: str
) -> dict[str, Any]:
    levels = sorted(set(str(row["factors"][factor]) for row in rows))
    return {
        level: summarize_benign(
            [row for row in rows if str(row["factors"][factor]) == level]
        )
        for level in levels
    }


def _factor_separability(
    vectors: np.ndarray,
    labels: np.ndarray,
    run_ids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    factors: Sequence[str],
) -> dict[str, Any]:
    """Describe predeclared Sand factor levels against all Support controls."""
    support_indices = np.flatnonzero(labels == 1)
    output: dict[str, Any] = {}
    for factor in factors:
        levels: dict[str, Any] = {}
        sand_values = sorted(
            set(str(rows[index]["factors"][factor]) for index in np.flatnonzero(labels == 0))
        )
        for level in sand_values:
            sand_indices = np.asarray(
                [
                    index
                    for index in np.flatnonzero(labels == 0)
                    if str(rows[index]["factors"][factor]) == level
                ],
                dtype=np.int64,
            )
            if len(sand_indices) < 2:
                levels[level] = {"sand_n": len(sand_indices), "metrics": None}
                continue
            indices = np.concatenate((sand_indices, support_indices))
            selected_labels = np.concatenate(
                (np.zeros(len(sand_indices), dtype=np.int64), np.ones(len(support_indices), dtype=np.int64))
            )
            levels[level] = {
                "sand_n": len(sand_indices),
                "metrics": separability_metrics(
                    vectors[indices],
                    selected_labels,
                    run_ids=[run_ids[index] for index in indices],
                ),
            }
        output[factor] = levels
    return output


def _verify_sha(root: Path, item: Mapping[str, Any], name: str) -> None:
    path = root / str(item["path"])
    actual = sha256_file(path)
    if actual != str(item["sha256"]):
        raise RuntimeError(f"{name} SHA mismatch: {actual}")


def verify_analysis_inputs(
    root: Path, config_path: Path, document: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed before any Discovery feature extraction or model replay."""
    dataset = root / str(document["dataset"]["path"])
    verification = verify_mild_recalibrated_dataset(dataset)
    if not verification["passed"]:
        raise RuntimeError("mild-recalibrated dataset integrity failed")
    freeze = json.loads((dataset / "dataset_freeze.json").read_text(encoding="utf-8"))
    expected = document["dataset"]["hashes"]
    checks = {
        "manifest": sha256_file(dataset / "manifest.json") == expected["manifest"],
        "pre_simulation_freeze": sha256_file(dataset / "pre_simulation_freeze.json")
        == expected["pre_simulation_freeze"],
        "physical_audit": sha256_file(dataset / "physical_audit.json")
        == expected["physical_audit"],
        "confirmation_seal": sha256_file(dataset / "confirmation_seal.json")
        == expected["confirmation_seal"],
        "semantic_dataset_freeze": freeze["MILD_RECALIBRATED_DATASET_FREEZE_SHA"]
        == expected["semantic_dataset_freeze"],
        "npz_aggregate": freeze["MILD_RECALIBRATED_NPZ_AGGREGATE_SHA"]
        == expected["npz_aggregate"],
        "physical_outcomes": freeze["MILD_RECALIBRATED_PHYSICAL_OUTCOME_SHA"]
        == expected["physical_outcomes"],
        "gate_results": freeze["MILD_RECALIBRATED_GENERATION_GATE_RESULT_SHA"]
        == expected["gate_results"],
        "physical_signatures": freeze["MILD_RECALIBRATED_PHYSICAL_SIGNATURE_SHA"]
        == expected["physical_signatures"],
        "discovery_split": freeze["MILD_RECALIBRATED_DISCOVERY_SPLIT_SHA"]
        == expected["discovery_split"],
        "confirmation_split": freeze["MILD_RECALIBRATED_CONFIRMATION_SPLIT_SHA"]
        == expected["confirmation_split"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen Sand analysis input mismatch: {checks}")
    _verify_sha(root, document["implementation"], "analysis implementation")
    _verify_sha(root, document["model"]["record"], "final candidate record")
    _verify_sha(root, document["model"]["candidate_freeze"], "candidate freeze")
    _verify_sha(root, document["model"]["normalizer"], "normalizer")
    for index, checkpoint in enumerate(document["model"]["checkpoints"]):
        _verify_sha(root, checkpoint, f"checkpoint {index}")
    if feature_schema_hash() != HAZARD_FEATURE_SCHEMA_SHA256:
        raise RuntimeError("feature schema implementation changed")
    if document["model"]["feature_schema_sha256"] != HAZARD_FEATURE_SCHEMA_SHA256:
        raise RuntimeError("analysis feature schema declaration changed")
    guard = root / str(document["boundaries"]["old_holdout_guard_path"])
    if sha256_file(guard) != document["boundaries"]["old_holdout_guard_sha256"]:
        raise RuntimeError("historical HOLDOUT guard changed")
    return {
        "passed": True,
        "dataset_checks": checks,
        "dataset_verification": verification,
        "analysis_config_sha256": sha256_file(config_path),
        "analysis_implementation_sha256": sha256_file(
            root / str(document["implementation"]["path"])
        ),
        "feature_schema_sha256": HAZARD_FEATURE_SCHEMA_SHA256,
        "old_holdout_guard_sha256": sha256_file(guard),
        "old_holdout_guard": 1,
        "confirmation_payload_deserializations": 0,
    }


def _decision_flags(
    pelvis: Mapping[str, Any], combined: Mapping[str, Any]
) -> dict[str, Any]:
    reasonable_checks = {
        "centroid_separation": pelvis["centroid_separation"] >= 0.75,
        "balanced_5nn_agreement": pelvis["balanced_5nn_agreement"] >= 0.80,
        "local_mixing": pelvis["local_opposite_class_mixing"] <= 0.30,
        "distance_ratio": pelvis["median_nearest_opposite_to_same_ratio"] >= 1.25,
    }
    strong_checks = {
        "centroid_separation": pelvis["centroid_separation"] <= 0.60,
        "balanced_5nn_agreement": pelvis["balanced_5nn_agreement"] <= 0.70,
        "local_mixing": pelvis["local_opposite_class_mixing"] >= 0.40,
        "distance_ratio": pelvis["median_nearest_opposite_to_same_ratio"] <= 1.10,
        "radius_inclusion": pelvis["bidirectional_95pct_radius_inclusion"] >= 0.75,
    }
    deltas = {
        "centroid_separation": combined["centroid_separation"] - pelvis["centroid_separation"],
        "balanced_5nn_agreement": combined["balanced_5nn_agreement"]
        - pelvis["balanced_5nn_agreement"],
        "local_mixing": combined["local_opposite_class_mixing"]
        - pelvis["local_opposite_class_mixing"],
        "distance_ratio": combined["median_nearest_opposite_to_same_ratio"]
        - pelvis["median_nearest_opposite_to_same_ratio"],
    }
    improvement_checks = {
        "centroid_separation": deltas["centroid_separation"] >= 0.25,
        "balanced_5nn_agreement": deltas["balanced_5nn_agreement"] >= 0.10,
        "local_mixing": deltas["local_mixing"] <= -0.15,
        "distance_ratio": deltas["distance_ratio"] >= 0.20,
    }
    no_degradation = (
        deltas["centroid_separation"] >= -0.05
        and deltas["balanced_5nn_agreement"] >= -0.05
        and deltas["local_mixing"] <= 0.05
        and deltas["distance_ratio"] >= -0.05
    )
    return {
        "reasonable_pelvis_checks": reasonable_checks,
        "reasonable_pelvis_separation": all(reasonable_checks.values()),
        "strong_mixing_checks": strong_checks,
        "strong_mixing_check_count": sum(strong_checks.values()),
        "strong_pelvis_mixing": sum(strong_checks.values()) >= 3,
        "realizable_fsr_deltas": deltas,
        "realizable_fsr_improvement_checks": improvement_checks,
        "realizable_fsr_improvement_count": sum(improvement_checks.values()),
        "realizable_fsr_no_metric_degradation_over_0.05": no_degradation,
        "realizable_fsr_material_increment": sum(improvement_checks.values()) >= 3
        and no_degradation,
    }


def _select_hypothesis(
    flags: Mapping[str, Any], localization: Mapping[str, Any], prerequisites: bool
) -> dict[str, Any]:
    systematic = bool(localization["systematic_adverse_pattern"])
    localized = bool(localization["metadata_localization"])
    material = bool(flags["realizable_fsr_material_increment"])
    reasonable = bool(flags["reasonable_pelvis_separation"])
    strong_mixing = bool(flags["strong_pelvis_mixing"])
    matches = {
        "DOMAIN_DIVERSITY_GAP_SUPPORTED": prerequisites
        and systematic
        and localized
        and reasonable
        and not material,
        "PELVIS_OBSERVABILITY_TENSION_SUPPORTED": prerequisites
        and systematic
        and not localized
        and strong_mixing
        and material,
        "MODEL_REPRESENTATION_OR_CAPACITY_TENSION_SUPPORTED": prerequisites
        and systematic
        and not localized
        and reasonable
        and not material,
    }
    selected = [name for name, value in matches.items() if value]
    verdict = selected[0] if len(selected) == 1 else "SAND_GENERALIZATION_STUDY_INCONCLUSIVE"
    return {
        "prerequisites_passed": prerequisites,
        "inputs": {
            "systematic_adverse_pattern": systematic,
            "metadata_localization": localized,
            "reasonable_pelvis_separation": reasonable,
            "strong_pelvis_mixing": strong_mixing,
            "realizable_fsr_material_increment": material,
        },
        "matches": matches,
        "unique_match_count": len(selected),
        "selected_hypothesis": verdict,
    }


def run_discovery_analysis(root: Path, config_path: Path) -> dict[str, Any]:
    """Execute the one authorized Discovery analysis and V2 replay."""
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if document["experiment"]["id"] != "SAND_BENIGN_MILD_RECALIBRATED_DISCOVERY_ANALYSIS":
        raise ValueError("unsupported Sand Discovery analysis config")
    artifact_dir = root / str(document["artifacts"]["path"])
    replay_path = artifact_dir / "v2_discovery_replay.json"
    if replay_path.exists():
        raise RuntimeError("the one authorized Discovery replay is already frozen")
    integrity = verify_analysis_inputs(root, config_path, document)
    write_json(
        artifact_dir / "pre_replay_freeze.json",
        {
            **integrity,
            "frozen_before_v2_discovery_replay": True,
            "v2_discovery_replay_count_before": 0,
            "confirmation_status": "SEALED_FOR_MILD_RECALIBRATED_CONFIRMATION",
            "confirmation_payload_deserializations": 0,
            "old_holdout_payload_reads": 0,
        },
    )
    dataset_path = root / str(document["dataset"]["path"])
    manifest = load_mild_recalibrated_manifest(dataset_path)
    manifest_rows = {
        str(row["run_id"]): row
        for row in manifest["runs"]
        if row["split"] == DISCOVERY_SPLIT
    }
    if len(manifest_rows) != 88:
        raise RuntimeError("Discovery split count changed")
    payloads: dict[str, dict[str, np.ndarray]] = {}
    runs: dict[str, HazardRun] = {}
    anchors: dict[str, int] = {}
    eligible_ids: list[str] = []
    labels: list[int] = []
    pelvis_current: list[np.ndarray] = []
    pelvis_windows: list[np.ndarray] = []
    fsr_vectors: list[np.ndarray] = []
    oracle_vectors: list[np.ndarray] = []
    vector_rows: list[dict[str, Any]] = []
    normalizer = load_hazard_normalizer(root / str(document["model"]["normalizer"]["path"]))
    for run_id, row in sorted(manifest_rows.items()):
        payload = load_mild_recalibrated_discovery_payload(dataset_path, run_id)
        payloads[run_id] = payload
        runs[run_id] = hazard_run_from_discovery(row, payload)
        eligible = row["objective_physical_outcome"] == STRICT_BENIGN or (
            row["group"] in SUPPORT_GROUPS
            and row["actual_hazard_label"] == "HAZARD"
            and row["actual_subtype"] == "SUPPORT"
            and row["valid"]
        )
        if not eligible:
            continue
        anchor = benign_anchor(payload) if row["objective_physical_outcome"] == STRICT_BENIGN else support_anchor(row)
        anchors[run_id] = anchor
        features = extract_hazard_features(payload["pelvis_imu6"])
        normalized = normalizer.transform(features).astype(np.float32, copy=False)
        window = normalized[anchor - HISTORY_MS + 1 : anchor + 1]
        if window.shape != (HISTORY_MS, 80):
            raise RuntimeError(f"incomplete Hazard analysis window: {run_id}")
        label = 0 if row["objective_physical_outcome"] == STRICT_BENIGN else 1
        eligible_ids.append(run_id)
        labels.append(label)
        pelvis_current.append(normalized[anchor].astype(np.float64))
        pelvis_windows.append(window.astype(np.float64).reshape(-1))
        fsr_vectors.append(fsr_contact_vector(payload, anchor))
        oracle_vectors.append(privileged_oracle_vector(payload, anchor))
        vector_rows.append(
            {
                "run_id": run_id,
                "label": "SAND" if label == 0 else "SUPPORT",
                "group": row["group"],
                "anchor_sample": anchor,
                "anchor_type": "BENIGN_ACCEL_DEVIATION_MAX" if label == 0 else "I1",
                "factors": factor_metadata(row) if label == 0 else {
                    "source": str(row["source_terrain"]).upper(),
                    "speed": f"{float(row['speed_mps']):.2f}",
                },
            }
        )
    label_array = np.asarray(labels, dtype=np.int64)
    current_array = np.asarray(pelvis_current, dtype=np.float64)
    window_array = np.asarray(pelvis_windows, dtype=np.float64)
    fsr_array = np.asarray(fsr_vectors, dtype=np.float64)
    oracle_array = np.asarray(oracle_vectors, dtype=np.float64)
    combined_array = np.concatenate((window_array, fsr_array), axis=1)
    pelvis_analysis = {
        "contract": {
            "population": "strict_benign_Sand_vs_actual_valid_Support",
            "benign_anchor": document["analysis"]["anchors"]["benign"],
            "support_anchor": document["analysis"]["anchors"]["support"],
            "frozen_normalizer_sha256": document["model"]["normalizer"]["sha256"],
            "feature_schema_sha256": HAZARD_FEATURE_SCHEMA_SHA256,
            "run_balancing": "one_anchor_vector_per_run",
        },
        "endpoint_count": len(eligible_ids),
        "window_count": len(eligible_ids),
        "anchors": anchors,
        "current_80d": separability_metrics(
            current_array, label_array, run_ids=eligible_ids
        ),
        "flattened_window_20x80": separability_metrics(
            window_array, label_array, run_ids=eligible_ids
        ),
        "factor_localized_window": _factor_separability(
            window_array,
            label_array,
            eligible_ids,
            vector_rows,
            document["analysis"]["descriptive_factors"],
        ),
    }
    fsr_analysis = {
        "contract": {
            "realizable": True,
            "fsr_dimension": int(fsr_array.shape[1]),
            "combined_dimension": int(combined_array.shape[1]),
            "exact_support_spread_included": False,
            "exact_loaded_contact_included": False,
            "classifier_or_probe_trained": False,
        },
        "fsr_contact_only": separability_metrics(
            fsr_array, label_array, run_ids=eligible_ids
        ),
        "pelvis_plus_fsr_contact": separability_metrics(
            combined_array, label_array, run_ids=eligible_ids
        ),
    }
    oracle_analysis = {
        "contract": {
            "privileged": True,
            "runtime_candidate": False,
            "sensor_claim_permitted": False,
            "dimension": int(oracle_array.shape[1]),
        },
        "privileged_oracle": separability_metrics(
            oracle_array, label_array, run_ids=eligible_ids
        ),
    }
    write_json(artifact_dir / "pelvis_analysis.json", pelvis_analysis)
    write_json(artifact_dir / "fsr_contact_analysis.json", fsr_analysis)
    write_json(artifact_dir / "privileged_oracle_analysis.json", oracle_analysis)
    model_independent_hashes = {
        name: sha256_file(artifact_dir / filename)
        for name, filename in (
            ("pelvis_analysis_sha256", "pelvis_analysis.json"),
            ("fsr_contact_analysis_sha256", "fsr_contact_analysis.json"),
            ("privileged_oracle_analysis_sha256", "privileged_oracle_analysis.json"),
        )
    }
    write_json(
        artifact_dir / "model_independent_analysis_freeze.json",
        {
            **model_independent_hashes,
            "completed_before_v2_discovery_replay": True,
            "discovery_payload_deserializations": 88,
            "confirmation_payload_deserializations": 0,
        },
    )

    # The one authorized exact frozen-V2 replay begins here.
    models = [
        load_checkpoint(root / str(item["path"]))[0]
        for item in document["model"]["checkpoints"]
    ]
    threshold = float(document["model"]["threshold"])
    persistence = int(document["model"]["persistence_ms"])
    replay_rows: list[dict[str, Any]] = []
    for run_id, run in sorted(runs.items()):
        row = manifest_rows[run_id]
        replay = replay_hazard_run(run, normalizer, models)
        probabilities = replay.probabilities
        crossings = replay.endpoints[probabilities >= threshold]
        onsets = reflex_onset_samples(replay, threshold, persistence)
        first_crossing = None if not len(crossings) else int(crossings[0])
        first_reflex = None if not len(onsets) else int(onsets[0])
        maximum = float(np.max(probabilities)) if len(probabilities) else 0.0
        streak = _longest_streak(probabilities, threshold)
        is_benign = row["objective_physical_outcome"] == STRICT_BENIGN
        is_support = (
            row["group"] in SUPPORT_GROUPS
            and row["actual_hazard_label"] == "HAZARD"
            and row["actual_subtype"] == "SUPPORT"
            and row["valid"]
        )
        i1 = _optional_sample(row["i1_summary"]["first_sample"])
        support = _optional_sample(row["support_event_summary"]["first_sample"])
        pre_i1 = bool(is_support and first_reflex is not None and i1 is not None and first_reflex < i1)
        support_correct = bool(
            is_support
            and first_reflex is not None
            and i1 is not None
            and support is not None
            and i1 <= first_reflex <= support + 50
        )
        result: dict[str, Any] = {
            "run_id": run_id,
            "group": row["group"],
            "physical_class": (
                "STRICT_SAND_BENIGN"
                if is_benign
                else "SUPPORT"
                if is_support
                else "INVALID_OR_NONPRIMARY"
            ),
            "eligible_primary_analysis": is_benign or is_support,
            "max_probability": maximum,
            "first_threshold_crossing": first_crossing,
            "first_reflex": first_reflex,
            "max_threshold_streak_ms": streak,
            "reflex": first_reflex is not None,
            "adverse_margin": bool(is_benign and (first_reflex is not None or maximum >= 0.95)),
            "margin_bin": _margin_bin(maximum, first_reflex is not None),
            "support_i1_sample": i1,
            "support_sample": support,
            "support_correct": support_correct,
            "pre_i1_reflex": pre_i1,
            "i1_to_reflex_ms": None if i1 is None or first_reflex is None else first_reflex - i1,
            "reflex_to_support_ms": None
            if support is None or first_reflex is None
            else support - first_reflex,
            "factors": factor_metadata(row) if is_benign else {
                "source": str(row["source_terrain"]).upper(),
                "speed": f"{float(row['speed_mps']):.2f}",
                "side": str(row["support_event_summary"]["side"]),
                "support_kind": "DELAYED" if row["group"] == "delayed_support_control" else "ORDINARY",
            },
        }
        replay_rows.append(result)
    benign_rows = [row for row in replay_rows if row["physical_class"] == "STRICT_SAND_BENIGN"]
    support_rows = [row for row in replay_rows if row["physical_class"] == "SUPPORT"]
    invalid_rows = [row for row in replay_rows if row["physical_class"] == "INVALID_OR_NONPRIMARY"]
    sand_summary = {
        "all_strict_sand": summarize_benign(benign_rows),
        "mild": summarize_benign(
            [row for row in benign_rows if row["factors"]["actual_severity"] == "LOW"]
        ),
        "moderate": summarize_benign(
            [row for row in benign_rows if row["factors"]["actual_severity"] == "MEDIUM"]
        ),
        "by_source": _subset_summary(benign_rows, "source"),
        "by_speed": _subset_summary(benign_rows, "speed"),
        "by_topology": _subset_summary(benign_rows, "transition_topology"),
        "by_phase": _subset_summary(benign_rows, "actual_entry_phase"),
        "by_entry_timing": _subset_summary(benign_rows, "entry_timing_stratum"),
        "by_exposure": _subset_summary(benign_rows, "exposure_stratum"),
    }
    support_summary = {
        "all": summarize_support(support_rows),
        "ordinary": summarize_support(
            [row for row in support_rows if row["factors"]["support_kind"] == "ORDINARY"]
        ),
        "delayed": summarize_support(
            [row for row in support_rows if row["factors"]["support_kind"] == "DELAYED"]
        ),
        "by_source": {
            level: summarize_support(
                [row for row in support_rows if row["factors"]["source"] == level]
            )
            for level in sorted(set(row["factors"]["source"] for row in support_rows))
        },
        "by_side": {
            level: summarize_support(
                [row for row in support_rows if row["factors"]["side"] == level]
            )
            for level in sorted(set(row["factors"]["side"] for row in support_rows))
        },
        "by_speed": {
            level: summarize_support(
                [row for row in support_rows if row["factors"]["speed"] == level]
            )
            for level in sorted(set(row["factors"]["speed"] for row in support_rows))
        },
    }
    replay_artifact = {
        "candidate_id": document["model"]["candidate_id"],
        "threshold": threshold,
        "persistence_ms": persistence,
        "ensemble_seeds": document["model"]["ensemble_seeds"],
        "discovery_runs_replayed": len(replay_rows),
        "discovery_split_replay_count": 1,
        "eligible_strict_sand": len(benign_rows),
        "eligible_support": len(support_rows),
        "invalid_or_nonprimary": len(invalid_rows),
        "sand": sand_summary,
        "support": support_summary,
        "run_level": replay_rows,
        "training_or_tuning": False,
        "confirmation_inference": False,
        "old_holdout_inference": False,
    }
    write_json(replay_path, replay_artifact)
    localization = factor_localization(
        benign_rows,
        factors=document["decision"]["metadata_localization"]["factors"],
        minimum_level_n=int(
            document["decision"]["metadata_localization"]["minimum_valid_runs_per_level"]
        ),
        fraction_range_min=float(
            document["decision"]["metadata_localization"]["adverse_fraction_range_min"]
        ),
        cramers_v_min=float(document["decision"]["metadata_localization"]["cramers_v_min"]),
    )
    localization["adverse_runs"] = [row for row in benign_rows if row["adverse_margin"]]
    write_json(artifact_dir / "factor_localization.json", localization)
    flags = _decision_flags(
        pelvis_analysis["flattened_window_20x80"],
        fsr_analysis["pelvis_plus_fsr_contact"],
    )
    decision = _select_hypothesis(flags, localization, prerequisites=True)
    decision["metric_flags"] = flags
    decision["hypothesis_rules"] = document["decision"]["hypothesis_rules"]
    decision["scientific_status_only"] = True
    write_json(artifact_dir / "discovery_hypothesis_decision.json", decision)
    hashes = {
        **model_independent_hashes,
        "v2_discovery_replay_sha256": sha256_file(replay_path),
        "factor_localization_sha256": sha256_file(artifact_dir / "factor_localization.json"),
        "discovery_hypothesis_decision_sha256": sha256_file(
            artifact_dir / "discovery_hypothesis_decision.json"
        ),
    }
    interpretation_body = {
        "analysis_config_sha256": sha256_file(config_path),
        "analysis_implementation_sha256": integrity["analysis_implementation_sha256"],
        "semantic_dataset_freeze_sha256": document["dataset"]["hashes"]["semantic_dataset_freeze"],
        "discovery_split_sha256": document["dataset"]["hashes"]["discovery_split"],
        "confirmation_split_sha256": document["dataset"]["hashes"]["confirmation_split"],
        "confirmation_seal_sha256": document["dataset"]["hashes"]["confirmation_seal"],
        **hashes,
        "selected_hypothesis": decision["selected_hypothesis"],
        "analysis_validity": "SAND_BENIGN_GENERALIZATION_STUDY_DISCOVERY_ANALYSIS_VALID",
        "confirmation_status": "SEALED_FOR_MILD_RECALIBRATED_CONFIRMATION",
        "confirmation_payload_deserializations": 0,
        "confirmation_model_inference": 0,
        "old_holdout_payload_reads": 0,
        "known_limitations": document["limitations"],
    }
    interpretation = {
        **interpretation_body,
        "SAND_BENIGN_DISCOVERY_INTERPRETATION_SHA": canonical_sha256(
            interpretation_body
        ),
    }
    write_json(artifact_dir / "discovery_interpretation.json", interpretation)
    return {
        "integrity": integrity,
        "pelvis": pelvis_analysis,
        "fsr": fsr_analysis,
        "oracle": oracle_analysis,
        "replay": replay_artifact,
        "localization": localization,
        "decision": decision,
        "interpretation": interpretation,
        "artifact_hashes": hashes,
    }


__all__ = [
    "benign_anchor",
    "factor_localization",
    "fsr_contact_vector",
    "hazard_run_from_discovery",
    "privileged_oracle_vector",
    "run_discovery_analysis",
    "separability_metrics",
    "summarize_benign",
    "summarize_support",
    "verify_analysis_inputs",
]
