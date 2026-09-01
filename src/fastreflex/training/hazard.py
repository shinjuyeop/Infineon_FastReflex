"""TRAIN-only construction and HNM for the supported Unified Hazard GRU."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from fastreflex.dataset.hazard import (
    HazardRun,
    canonical_sha256,
    load_hazard_runs,
    physical_hazard_label,
    slip_event_sample,
    support_event_sample,
)
from fastreflex.dataset.generation import (
    HazardRunAnnotations,
    load_model_v2_manifest,
    load_model_v2_runs,
)
from fastreflex.dataset.loader import Normalizer, WindowSet, sha256_file
from fastreflex.evaluation.hazard import (
    HISTORY_MS,
    evaluate_model_v2_validation,
    load_hazard_normalizer,
    reflex_onset_samples,
    replay_hazard_run,
    replay_hazard_runs,
)
from fastreflex.features import extract_hazard_features, hazard_feature_schema
from fastreflex.models.baselines import parameter_count
from fastreflex.training.trainer import (
    load_checkpoint,
    save_checkpoint,
    train_model,
)


EVENT_CLASS_NAMES = ("NORMAL", "HAZARD_REFLEX_REQUIRED")
HNM_ROUNDS = 3
HNM_TOP_K_PER_RUN = 12
HNM_MINIMUM_SPACING_MS = 30
HNM_REPLAY_STRIDE_MS = 1
MODEL_V2_DELAYED_SUPPORT_FAMILY = "DELAYED_SAND_SUPPORT_ONSET"


def model_v2_rebalance_policy() -> dict[str, object]:
    """Return the exact predeclared extraction-rebalance design payload."""
    return {
        "schema_version": 1,
        "positive_precedence": [
            "legacy_slip_positive",
            "delayed_support_i1_neighborhood",
            "delayed_support_interior_neighborhood",
            "delayed_support_established_neighborhood",
            "legacy_ordinary_support_positive",
        ],
        "legacy_slip": {
            "selection": "preserve_exact_baseline_fit_and_monitor_endpoint_identities",
            "nominal_offsets_ms": {"start": -30, "stop": 40, "stride": 5},
            "legacy_union_cap_per_run": 20,
            "rare_side_protection": "none",
        },
        "ordinary_support": {
            "selection": "preserve_exact_baseline_fit_and_monitor_endpoint_identities",
            "i1_to_support_even_points": 5,
            "support_offsets_ms": [-20, 0, 20, 40],
            "legacy_union_cap_per_run": 20,
        },
        "delayed_support": {
            "eligibility": "valid V2_TRAIN DELAYED_SAND_SUPPORT_ONSET with usable I1 and Support",
            "source_rule": "identical_for_concrete_and_marble",
            "i1_offsets_ms": [0, 1, 2, 3, 4],
            "interior_anchor": "floor((I1 + Support) / 2)",
            "interior_offsets_ms": [-2, -1, 0, 1, 2],
            "support_offsets_ms": [0, 1, 2, 3, 4],
            "deduplicate": True,
            "cap_per_run": 15,
            "fit_exposure": "all_valid_endpoints_from_every_eligible_run",
            "monitor_exposure": "none; other 598 baseline monitor positives remain",
            "pre_i1_positive": False,
        },
        "negative_endpoints": "preserve_exact_baseline_fit_and_monitor_endpoint_identities",
        "ice_masks": "preserve_exact_baseline_future_slip_and_censored_masks",
        "censor_masks": "preserve_exact_baseline_post_censor_and_post_fall_exclusion",
        "hnm": "unchanged_and_not_executed",
        "runtime_tensor": "strictly_causal_history_20ms_80D",
    }


@dataclass(frozen=True)
class HazardCandidate:
    history_ms: int
    normalizer: Normalizer
    checkpoint_paths: tuple[Path, ...]
    record: Mapping[str, object]


@dataclass(frozen=True)
class ModelV2TrainingData:
    runs: Mapping[str, HazardRun]
    precursor_samples: Mapping[str, int | None]
    annotations: Mapping[str, HazardRunAnnotations]
    input_audit: Mapping[str, object]
    composition: Mapping[str, object]
    v2_manifest: Mapping[str, object]


def _evenly_spaced(values: np.ndarray, count: int) -> np.ndarray:
    selected = np.asarray(values, dtype=np.int64)
    if count <= 0 or not len(selected):
        return np.empty(0, dtype=np.int64)
    if len(selected) <= count:
        return selected
    return selected[np.linspace(0, len(selected) - 1, count, dtype=np.int64)]


def unified_positive_endpoints(
    run: HazardRun,
    precursor: int | None,
    history_ms: int = HISTORY_MS,
    *,
    cap: int = 20,
) -> np.ndarray:
    """Frozen bounded union positives, using physical references only."""
    selected: set[int] = set()
    slip = slip_event_sample(run)
    support = support_event_sample(run)
    if slip is not None:
        selected.update(range(slip - 30, slip + 41, 5))
    if support is not None and precursor is not None:
        selected.update(int(value) for value in np.linspace(precursor, support, 5))
        selected.update(support + offset for offset in (-20, 0, 20, 40))
    elif support is not None:
        selected.update(support + offset for offset in (-20, 0, 20, 40))
    values = np.asarray(sorted(selected), dtype=np.int64)
    valid = (values >= history_ms - 1) & (values < run.censor_sample)
    if run.fall_sample_diagnostic is not None:
        valid &= values < int(run.fall_sample_diagnostic)
    return _evenly_spaced(values[valid], cap)


def _positive_role(run: HazardRun, endpoint: int) -> str:
    slip = slip_event_sample(run)
    return (
        "slip"
        if slip is not None and slip - 30 <= int(endpoint) <= slip + 40
        else "support"
    )


def delayed_support_positive_endpoints(
    run: HazardRun,
    precursor: int | None,
    annotation: HazardRunAnnotations,
    history_ms: int = HISTORY_MS,
) -> np.ndarray:
    """Resolve the frozen three-neighborhood delayed-Support positive rule."""
    if annotation.scenario_family != MODEL_V2_DELAYED_SUPPORT_FAMILY:
        return np.empty(0, dtype=np.int64)
    support = support_event_sample(run)
    if precursor is None or support is None or int(precursor) > support:
        raise ValueError("eligible delayed Support requires ordered I1 and Support")
    i1 = int(precursor)
    midpoint = (i1 + support) // 2
    selected = np.asarray(
        sorted(
            {
                *(i1 + offset for offset in (0, 1, 2, 3, 4)),
                *(midpoint + offset for offset in (-2, -1, 0, 1, 2)),
                *(support + offset for offset in (0, 1, 2, 3, 4)),
            }
        ),
        dtype=np.int64,
    )
    stop = run.censor_sample
    if run.fall_sample_diagnostic is not None:
        stop = min(stop, int(run.fall_sample_diagnostic))
    selected = selected[(selected >= history_ms - 1) & (selected < stop)][:15]
    if np.any(selected < i1):
        raise RuntimeError("delayed-Support positive precedes I1")
    if not all(i1 + offset in selected for offset in range(5)):
        raise RuntimeError("delayed-Support I1 neighborhood is shorter than 5 ms")
    return selected


def model_v2_rebalanced_positive_endpoints(
    run: HazardRun,
    precursor: int | None,
    annotation: HazardRunAnnotations,
    history_ms: int = HISTORY_MS,
) -> np.ndarray:
    """Preserve legacy positives except for the frozen delayed-Support role."""
    baseline = unified_positive_endpoints(run, precursor, history_ms, cap=20)
    if annotation.scenario_family != MODEL_V2_DELAYED_SUPPORT_FAMILY:
        return baseline
    slip = [int(value) for value in baseline if _positive_role(run, int(value)) == "slip"]
    delayed = delayed_support_positive_endpoints(
        run, precursor, annotation, history_ms
    )
    return np.asarray(sorted({*slip, *(int(value) for value in delayed)}), dtype=np.int64)


def model_v2_rebalanced_positive_plan(
    runs: Mapping[str, HazardRun],
    precursor_samples: Mapping[str, int | None],
    annotations: Mapping[str, HazardRunAnnotations],
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    """Assign exact proposed positives to fit/monitor without moving negatives."""
    fit_ids, monitor_ids = _train_monitor_partition(
        runs, sorted(runs), precursor_samples
    )
    fit_set = set(fit_ids)
    fit: dict[str, tuple[int, ...]] = {}
    monitor: dict[str, tuple[int, ...]] = {}
    for run_id in sorted(runs):
        run = runs[run_id]
        annotation = annotations[run_id]
        precursor = precursor_samples.get(run_id)
        baseline = unified_positive_endpoints(run, precursor, cap=20)
        if annotation.scenario_family == MODEL_V2_DELAYED_SUPPORT_FAMILY:
            support = delayed_support_positive_endpoints(
                run, precursor, annotation
            )
            fit[run_id] = tuple(int(value) for value in support)
            legacy_slip = tuple(
                int(value)
                for value in baseline
                if _positive_role(run, int(value)) == "slip"
            )
            if run_id in fit_set:
                fit[run_id] = tuple(sorted({*fit[run_id], *legacy_slip}))
            elif legacy_slip:
                monitor[run_id] = legacy_slip
            continue
        endpoints = tuple(int(value) for value in baseline)
        if run_id in fit_set:
            fit[run_id] = endpoints
        else:
            monitor[run_id] = endpoints
    return fit, monitor


def unified_negative_candidates(
    run: HazardRun, precursor: int | None, history_ms: int = HISTORY_MS
) -> np.ndarray:
    """Return only no-hazard endpoints; I1-active samples are never negative."""
    last = run.censor_sample - 1
    slip = slip_event_sample(run)
    support = support_event_sample(run)
    if slip is not None:
        last = min(last, slip - 31)
    if precursor is not None:
        last = min(last, int(precursor) - 1)
    elif support is not None:
        last = min(last, support - 1)
    if run.fall_sample_diagnostic is not None:
        last = min(last, int(run.fall_sample_diagnostic) - 1)
    first = history_ms - 1
    return (
        np.arange(first, last + 1, dtype=np.int64)
        if last >= first
        else np.empty(0, dtype=np.int64)
    )


def training_negative_candidates(
    run: HazardRun,
    precursor: int | None,
    annotation: HazardRunAnnotations | None = None,
    history_ms: int = HISTORY_MS,
) -> np.ndarray:
    """Apply V1 boundaries plus frozen V2 precursor/censor exclusions."""
    eligible = unified_negative_candidates(run, precursor, history_ms)
    if annotation is None:
        return eligible
    stop = run.censor_sample
    if run.fall_sample_diagnostic is not None:
        stop = min(stop, int(run.fall_sample_diagnostic))
    special = np.flatnonzero(annotation.benign_release_precursor)
    special = special[(special >= history_ms - 1) & (special < stop)]
    positive_region = np.zeros(len(run.timestamp_us), dtype=bool)
    slip = slip_event_sample(run)
    support = support_event_sample(run)
    if slip is not None:
        positive_region[max(0, slip - 30) : min(len(positive_region), slip + 41)] = True
    if support is not None:
        first = support if precursor is None else int(precursor)
        positive_region[max(0, first) : min(len(positive_region), support + 51)] = True
    special_forbidden = (
        np.any(annotation.established_slip_active, axis=1)
        | annotation.i1_active
        | positive_region
        | annotation.future_slip_precursor
        | annotation.censored_precursor
    )
    special = special[~special_forbidden[special]]
    eligible = np.unique(np.concatenate((eligible, special)))
    if not len(eligible):
        return eligible
    forbidden = (
        annotation.future_slip_precursor
        | annotation.censored_precursor
        | annotation.i1_active
    )
    return eligible[~forbidden[eligible]]


def gait_sampling_categories(run: HazardRun) -> dict[str, np.ndarray]:
    """Use contact phase only to diversify TRAIN negative sampling."""
    loaded = np.asarray(run.loaded_contact, dtype=bool)
    previous = np.vstack((np.zeros((1, 2), dtype=bool), loaded[:-1]))
    touchdown = np.any(loaded & ~previous, axis=1)
    release = np.any(~loaded & previous, axis=1)
    return {
        "touchdown_loading": np.flatnonzero(touchdown),
        "contact_release": np.flatnonzero(release),
        "left_support": np.flatnonzero(loaded[:, 0] & ~loaded[:, 1]),
        "right_support": np.flatnonzero(~loaded[:, 0] & loaded[:, 1]),
        "double_support": np.flatnonzero(np.all(loaded, axis=1)),
        "no_support": np.flatnonzero(~np.any(loaded, axis=1)),
    }


def initial_negative_endpoints(
    run: HazardRun,
    precursor: int | None,
    history_ms: int = HISTORY_MS,
    *,
    per_category: int = 12,
    annotation: HazardRunAnnotations | None = None,
    target_contact_cap: int = 12,
    benign_precursor_cap: int = 12,
) -> np.ndarray:
    eligible = training_negative_candidates(
        run, precursor, annotation, history_ms
    )
    if not len(eligible):
        return eligible
    allowed = set(int(value) for value in eligible)
    selected = [
        _evenly_spaced(
            np.asarray(
                [value for value in values if int(value) in allowed], dtype=np.int64
            ),
            per_category,
        )
        for values in gait_sampling_categories(run).values()
    ]
    if annotation is not None:
        selected.extend(
            (
                _evenly_spaced(
                    np.asarray(
                        [
                            value
                            for value in np.flatnonzero(
                                np.any(annotation.target_contact, axis=1)
                            )
                            if int(value) in allowed
                        ],
                        dtype=np.int64,
                    ),
                    target_contact_cap,
                ),
                _evenly_spaced(
                    np.asarray(
                        [
                            value
                            for value in np.flatnonzero(
                                annotation.benign_release_precursor
                            )
                            if int(value) in allowed
                        ],
                        dtype=np.int64,
                    ),
                    benign_precursor_cap,
                ),
            )
        )
    return np.unique(np.concatenate(selected)) if selected else np.empty(0, np.int64)


def mine_hard_negative_endpoints(
    candidates: np.ndarray,
    probabilities: np.ndarray,
    *,
    top_k: int = HNM_TOP_K_PER_RUN,
    minimum_separation_ms: int = HNM_MINIMUM_SPACING_MS,
    excluded: Sequence[int] = (),
) -> np.ndarray:
    """Choose highest-scoring TRAIN negatives with frozen spacing and K."""
    endpoints = np.asarray(candidates, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if endpoints.shape != scores.shape:
        raise ValueError("HNM endpoints and probabilities differ")
    prior = {int(value) for value in excluded}
    order = sorted(range(len(endpoints)), key=lambda i: (-scores[i], int(endpoints[i])))
    selected: list[int] = []
    for index in order:
        endpoint = int(endpoints[index])
        if endpoint in prior:
            continue
        if all(abs(endpoint - other) >= minimum_separation_ms for other in selected):
            selected.append(endpoint)
            if len(selected) >= top_k:
                break
    return np.asarray(sorted(selected), dtype=np.int64)


def fit_hazard_normalizer(
    runs: Mapping[str, HazardRun],
    run_ids: Sequence[str],
    *,
    per_run_sample_cap: int = 1000,
    standard_deviation_floor: float = 1.0e-6,
) -> Normalizer:
    """Fit feature normalization only from explicitly supplied TRAIN runs."""
    chunks: list[np.ndarray] = []
    fit_ids: list[str] = []
    for run_id in sorted(str(value) for value in run_ids):
        run = runs[run_id]
        if run.split != "train":
            raise ValueError("Hazard normalizer may use TRAIN runs only")
        features = extract_hazard_features(run.features["PELVIS_IMU6"])
        eligible = np.arange(0, run.censor_sample, dtype=np.int64)
        if run.fall_sample_diagnostic is not None:
            eligible = eligible[eligible < int(run.fall_sample_diagnostic)]
        eligible = _evenly_spaced(eligible, per_run_sample_cap)
        if len(eligible):
            chunks.append(features[eligible].astype(np.float64))
            fit_ids.append(run_id)
    if not chunks:
        raise ValueError("Hazard normalizer has no TRAIN samples")
    values = np.concatenate(chunks)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < standard_deviation_floor] = 1.0
    return Normalizer(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        sample_count=len(values),
        fit_run_ids=tuple(fit_ids),
        epsilon=float(standard_deviation_floor),
    )


def build_hazard_windows(
    runs: Mapping[str, HazardRun],
    run_ids: Sequence[str],
    precursor_samples: Mapping[str, int | None],
    normalizer: Normalizer,
    *,
    per_category: int = 12,
    positive_cap: int = 20,
    extra_negative_endpoints: Mapping[str, Sequence[int]] | None = None,
    annotations: Mapping[str, HazardRunAnnotations] | None = None,
    target_contact_cap: int = 12,
    benign_precursor_cap: int = 12,
    positive_endpoints: Mapping[str, Sequence[int]] | None = None,
    negative_run_ids: Sequence[str] | None = None,
) -> WindowSet:
    """Materialize causal [20,80] binary windows with provenance."""
    inputs: list[np.ndarray] = []
    targets: list[int] = []
    source_ids: list[str] = []
    endpoint_rows: list[int] = []
    extras = extra_negative_endpoints or {}
    metadata = annotations or {}
    negative_ids = {
        str(value)
        for value in (run_ids if negative_run_ids is None else negative_run_ids)
    }
    for run_id in sorted(str(value) for value in run_ids):
        run = runs[run_id]
        if run.split != "train":
            raise ValueError("Hazard training windows may use TRAIN runs only")
        features = extract_hazard_features(run.features["PELVIS_IMU6"])
        precursor = precursor_samples.get(run_id)
        positive = (
            unified_positive_endpoints(run, precursor, cap=positive_cap)
            if positive_endpoints is None
            else np.asarray(positive_endpoints.get(run_id, ()), dtype=np.int64)
        )
        negative = (
            initial_negative_endpoints(
                run,
                precursor,
                per_category=per_category,
                annotation=metadata.get(run_id),
                target_contact_cap=target_contact_cap,
                benign_precursor_cap=benign_precursor_cap,
            )
            if run_id in negative_ids
            else np.empty(0, dtype=np.int64)
        )
        allowed = set(
            int(value)
            for value in training_negative_candidates(
                run, precursor, metadata.get(run_id)
            )
        )
        if run_id in extras and run_id in negative_ids:
            extra = np.asarray(
                [int(value) for value in extras[run_id] if int(value) in allowed],
                dtype=np.int64,
            )
            negative = np.unique(np.concatenate((negative, extra)))
        if set(int(value) for value in positive) & set(int(value) for value in negative):
            raise RuntimeError("Hazard positive was used as a negative")
        for label, endpoints in ((1, positive), (0, negative)):
            for endpoint in endpoints:
                first = int(endpoint) - HISTORY_MS + 1
                if first < 0 or int(endpoint) >= run.censor_sample:
                    raise ValueError("Hazard window crossed a causal boundary")
                window = normalizer.transform(features[first : int(endpoint) + 1])
                if window.shape != (HISTORY_MS, 80):
                    raise RuntimeError("Hazard GRU input shape changed")
                inputs.append(window)
                targets.append(label)
                source_ids.append(run_id)
                endpoint_rows.append(int(endpoint))
    labels = np.asarray(targets, dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    if not inputs or np.any(counts == 0):
        raise ValueError("Hazard training windows require both classes")
    return WindowSet(
        inputs=np.stack(inputs).astype(np.float32),
        targets=labels,
        run_ids=np.asarray(source_ids, dtype=str),
        endpoint_samples=np.asarray(endpoint_rows, dtype=np.int64),
        available_by_class=(int(counts[0]), int(counts[1]), 0),
    )


def unified_training_annotations(
    manifest: Mapping[str, object], runs: Mapping[str, HazardRun]
) -> dict[str, HazardRunAnnotations]:
    """Attach run-level audit strata without adding new Unified anchors."""
    rows = {str(row["run_id"]): row for row in manifest["runs"]}
    result: dict[str, HazardRunAnnotations] = {}
    for run_id, run in runs.items():
        row = rows[run_id]
        samples = len(run.timestamp_us)
        empty_trace = np.zeros(samples, dtype=bool)
        empty_feet = np.zeros((samples, 2), dtype=bool)
        empty_codes = np.zeros((samples, 2), dtype=np.int8)
        slip = run.slip_event_samples_per_foot
        support = run.support_event_samples_per_foot
        active_sides = [
            side
            for side in range(2)
            if slip[side] is not None or support[side] is not None
        ]
        actual_side = (
            "NONE"
            if not active_sides
            else ("LEFT_ONLY" if active_sides == [0] else "RIGHT_ONLY")
            if active_sides in ([0], [1])
            else "BILATERAL"
        )
        result[run_id] = HazardRunAnnotations(
            dataset_id="unified_hazard_reflex_20260829",
            scenario_family=str(row["group"]),
            nominal_speed_mps=float(row["speed_mps"]),
            actual_side=actual_side,
            target_contact=empty_feet,
            established_slip_active=empty_feet,
            i1_active=empty_trace,
            ice_precursor_candidate=empty_feet,
            ice_precursor_future_outcome_code=empty_codes,
            ice_precursor_censored=empty_feet,
        )
    return result


def _negative_role(
    run: HazardRun, annotation: HazardRunAnnotations | None, endpoint: int
) -> str:
    if annotation is not None and annotation.benign_release_precursor[endpoint]:
        return "benign_near_threshold_ice"
    family = "" if annotation is None else annotation.scenario_family
    if family in ("HARD_GROUND_NORMAL_SPEED_MATRIX", "HARD_GROUND_NORMAL"):
        return "hard_normal"
    if family == "ICE_BENIGN_CONTROL":
        return "ice_benign"
    if family == "STAGED_SAND_BENIGN_CONTROL":
        return "staged_sand_benign"
    if family == "SPEED_STRATIFIED_SAND_BENIGN":
        return "speed_sand_benign"
    return "other_confirmed_benign"


def audit_hazard_extraction(
    runs: Mapping[str, HazardRun],
    run_ids: Sequence[str],
    precursor_samples: Mapping[str, int | None],
    annotations: Mapping[str, HazardRunAnnotations],
    *,
    per_category: int = 12,
    positive_cap: int = 20,
    target_contact_cap: int = 12,
    benign_precursor_cap: int = 12,
) -> dict[str, object]:
    """Dry-run endpoint extraction and prove forbidden-negative separation."""
    positive_roles: Counter[str] = Counter()
    negative_roles: Counter[str] = Counter()
    positive_runs: dict[str, set[str]] = {}
    negative_runs: dict[str, set[str]] = {}
    source_positive: Counter[str] = Counter()
    source_negative: Counter[str] = Counter()
    speed_positive: Counter[str] = Counter()
    speed_negative: Counter[str] = Counter()
    family_positive: Counter[str] = Counter()
    family_negative: Counter[str] = Counter()
    side_positive: Counter[str] = Counter()
    violations = Counter()
    masked = Counter()
    for run_id in sorted(str(value) for value in run_ids):
        run = runs[run_id]
        annotation = annotations[run_id]
        precursor = precursor_samples.get(run_id)
        positive = unified_positive_endpoints(
            run, precursor, cap=positive_cap
        )
        negative = initial_negative_endpoints(
            run,
            precursor,
            per_category=per_category,
            annotation=annotation,
            target_contact_cap=target_contact_cap,
            benign_precursor_cap=benign_precursor_cap,
        )
        slip = slip_event_sample(run)
        for endpoint in positive:
            role = (
                "slip_positive"
                if slip is not None and slip - 30 <= int(endpoint) <= slip + 40
                else "support_positive"
            )
            positive_roles[role] += 1
            positive_runs.setdefault(role, set()).add(run_id)
            source_positive[run.source_terrain] += 1
            speed_positive[f"{annotation.nominal_speed_mps:.2f}"] += 1
            family_positive[annotation.scenario_family] += 1
            side_positive[annotation.actual_side] += 1
        for endpoint in negative:
            endpoint = int(endpoint)
            role = _negative_role(run, annotation, endpoint)
            negative_roles[role] += 1
            negative_runs.setdefault(role, set()).add(run_id)
            source_negative[run.source_terrain] += 1
            speed_negative[f"{annotation.nominal_speed_mps:.2f}"] += 1
            family_negative[annotation.scenario_family] += 1
            if annotation.future_slip_precursor[endpoint]:
                violations["future_slip_precursor"] += 1
            if annotation.censored_precursor[endpoint]:
                violations["censored_precursor"] += 1
            if annotation.i1_active[endpoint]:
                violations["i1_positive"] += 1
            if endpoint >= run.censor_sample or (
                run.fall_sample_diagnostic is not None
                and endpoint >= run.fall_sample_diagnostic
            ):
                violations["post_censor_or_fall"] += 1
        masked["future_slip_precursor_samples"] += int(
            np.sum(annotation.future_slip_precursor[: run.censor_sample])
        )
        masked["censored_precursor_samples"] += int(
            np.sum(annotation.censored_precursor[: run.censor_sample])
        )
        masked["i1_positive_samples"] += int(
            np.sum(annotation.i1_active[: run.censor_sample])
        )
        if np.any(annotation.future_slip_precursor[: run.censor_sample]):
            masked["future_slip_precursor_runs"] += 1
        if np.any(annotation.censored_precursor[: run.censor_sample]):
            masked["censored_precursor_runs"] += 1
        if np.any(annotation.i1_active[: run.censor_sample]):
            masked["i1_positive_runs"] += 1
    violation_names = (
        "future_slip_precursor",
        "censored_precursor",
        "i1_positive",
        "post_censor_or_fall",
    )
    result = {
        "run_count": len(run_ids),
        "positive_windows": {
            "total": sum(positive_roles.values()),
            **dict(sorted(positive_roles.items())),
            "run_balanced": {
                key: len(value) for key, value in sorted(positive_runs.items())
            },
            "by_source": dict(sorted(source_positive.items())),
            "by_speed": dict(sorted(speed_positive.items())),
            "by_family": dict(sorted(family_positive.items())),
            "by_side": dict(sorted(side_positive.items())),
        },
        "negative_windows": {
            "total": sum(negative_roles.values()),
            **dict(sorted(negative_roles.items())),
            "run_balanced": {
                key: len(value) for key, value in sorted(negative_runs.items())
            },
            "by_source": dict(sorted(source_negative.items())),
            "by_speed": dict(sorted(speed_negative.items())),
            "by_family": dict(sorted(family_negative.items())),
        },
        "masked": dict(sorted(masked.items())),
        "ordinary_negative_violations": {
            name: int(violations[name]) for name in violation_names
        },
    }
    result["passed"] = not any(result["ordinary_negative_violations"].values())
    return result


def _train_monitor_partition(
    runs: Mapping[str, HazardRun],
    run_ids: Sequence[str],
    precursor_samples: Mapping[str, int | None],
) -> tuple[list[str], list[str]]:
    groups: dict[tuple[str, str, str], list[str]] = {}
    for run_id in sorted(str(value) for value in run_ids):
        run = runs[run_id]
        if run.split != "train":
            raise ValueError("Hazard HNM partition may use TRAIN runs only")
        groups.setdefault(
            (
                run.source_terrain,
                run.target_terrain,
                physical_hazard_label(run, precursor_samples.get(run_id)),
            ),
            [],
        ).append(run_id)
    monitor: list[str] = []
    for values in groups.values():
        count = max(1, int(round(len(values) * 0.20)))
        monitor.extend(
            values[int(index)]
            for index in np.linspace(0, len(values) - 1, count, dtype=np.int64)
        )
    monitor_set = set(monitor)
    return (
        sorted(run_id for run_id in run_ids if run_id not in monitor_set),
        sorted(monitor_set),
    )


def _endpoint_identity_hash(rows: Sequence[Mapping[str, object]]) -> str:
    return canonical_sha256(
        [
            f"{row['partition']}:{row['run_id']}:{row['endpoint_sample']}:{row['role']}"
            for row in rows
        ]
    )


def audit_model_v2_rebalanced_extraction(
    runs: Mapping[str, HazardRun],
    precursor_samples: Mapping[str, int | None],
    annotations: Mapping[str, HazardRunAnnotations],
    *,
    per_category: int = 12,
    target_contact_cap: int = 12,
    benign_precursor_cap: int = 12,
) -> dict[str, object]:
    """Reproduce every frozen extraction identity before optimizer access."""
    fit_ids, monitor_ids = _train_monitor_partition(
        runs, sorted(runs), precursor_samples
    )
    partitions = {run_id: "fit" for run_id in fit_ids} | {
        run_id: "monitor" for run_id in monitor_ids
    }
    fit_positive, monitor_positive = model_v2_rebalanced_positive_plan(
        runs, precursor_samples, annotations
    )
    positive_rows: list[dict[str, object]] = []
    negative_rows: list[dict[str, object]] = []
    mask_rows: list[dict[str, object]] = []
    violations: Counter[str] = Counter()
    eligible_delayed: list[str] = []

    for partition, endpoint_map in (
        ("fit", fit_positive),
        ("monitor", monitor_positive),
    ):
        for run_id, endpoints in endpoint_map.items():
            run = runs[run_id]
            annotation = annotations[run_id]
            precursor = precursor_samples.get(run_id)
            for endpoint in endpoints:
                value = int(endpoint)
                positive_rows.append(
                    {
                        "partition": partition,
                        "run_id": run_id,
                        "endpoint_sample": value,
                        "role": _positive_role(run, value),
                    }
                )
                first = value - HISTORY_MS + 1
                if first < 0 or value >= run.censor_sample or (
                    run.fall_sample_diagnostic is not None
                    and value >= run.fall_sample_diagnostic
                ):
                    violations["future_feature_leakage"] += 1
                if (
                    annotation.scenario_family == MODEL_V2_DELAYED_SUPPORT_FAMILY
                    and _positive_role(run, value) == "support"
                    and precursor is not None
                    and value < int(precursor)
                ):
                    violations["pre_i1_delayed_support_positive"] += 1

    for run_id in sorted(runs):
        run = runs[run_id]
        annotation = annotations[run_id]
        precursor = precursor_samples.get(run_id)
        if annotation.scenario_family == MODEL_V2_DELAYED_SUPPORT_FAMILY:
            eligible_delayed.append(run_id)
            delayed = delayed_support_positive_endpoints(
                run, precursor, annotation
            )
            if len(delayed) != 15:
                violations["eligible_delayed_support_short_neighborhood"] += 1
            if precursor is None or not all(
                int(precursor) + offset in delayed for offset in range(5)
            ):
                violations["persistence_neighborhood_shorter_than_5ms"] += 1
        negative = initial_negative_endpoints(
            run,
            precursor,
            per_category=per_category,
            annotation=annotation,
            target_contact_cap=target_contact_cap,
            benign_precursor_cap=benign_precursor_cap,
        )
        for endpoint in negative:
            value = int(endpoint)
            negative_rows.append(
                {
                    "partition": partitions[run_id],
                    "run_id": run_id,
                    "endpoint_sample": value,
                    "role": _negative_role(run, annotation, value),
                }
            )
            if annotation.future_slip_precursor[value]:
                violations["future_slip_precursor_ordinary_negative"] += 1
            if annotation.censored_precursor[value]:
                violations["censored_precursor_negative"] += 1
            if annotation.i1_active[value]:
                violations["i1_or_positive_negative"] += 1
            if value >= run.censor_sample or (
                run.fall_sample_diagnostic is not None
                and value >= run.fall_sample_diagnostic
            ):
                violations["post_censor_or_fall"] += 1
        mask_rows.append(
            {
                "run_id": run_id,
                "future_slip_precursor_samples": int(
                    np.sum(annotation.future_slip_precursor[: run.censor_sample])
                ),
                "future_slip_precursor_sha256": hashlib.sha256(
                    annotation.future_slip_precursor[: run.censor_sample].tobytes()
                ).hexdigest(),
                "censored_precursor_samples": int(
                    np.sum(annotation.censored_precursor[: run.censor_sample])
                ),
                "censored_precursor_sha256": hashlib.sha256(
                    annotation.censored_precursor[: run.censor_sample].tobytes()
                ).hexdigest(),
                "i1_positive_samples": int(
                    np.sum(annotation.i1_active[: run.censor_sample])
                ),
                "i1_positive_sha256": hashlib.sha256(
                    annotation.i1_active[: run.censor_sample].tobytes()
                ).hexdigest(),
                "censor_sample": run.censor_sample,
                "fall_sample": run.fall_sample_diagnostic,
            }
        )

    positive_rows.sort(
        key=lambda row: (
            str(row["partition"]),
            str(row["run_id"]),
            int(row["endpoint_sample"]),
            str(row["role"]),
        )
    )
    negative_rows.sort(
        key=lambda row: (
            str(row["partition"]),
            str(row["run_id"]),
            int(row["endpoint_sample"]),
            str(row["role"]),
        )
    )
    mask_rows.sort(key=lambda row: str(row["run_id"]))
    positive_ids = {
        (str(row["run_id"]), int(row["endpoint_sample"]))
        for row in positive_rows
    }
    negative_ids = {
        (str(row["run_id"]), int(row["endpoint_sample"]))
        for row in negative_rows
    }
    violations["i1_or_positive_negative"] += len(positive_ids & negative_ids)

    fit_positive_rows = [
        row for row in positive_rows if row["partition"] == "fit"
    ]
    monitor_positive_rows = [
        row for row in positive_rows if row["partition"] == "monitor"
    ]
    delayed_fit_rows = [
        row
        for row in fit_positive_rows
        if row["role"] == "support"
        and annotations[str(row["run_id"])].scenario_family
        == MODEL_V2_DELAYED_SUPPORT_FAMILY
    ]
    ordinary_fit = sum(
        row["role"] == "support" for row in fit_positive_rows
    ) - len(delayed_fit_rows)
    slip_fit = sum(row["role"] == "slip" for row in fit_positive_rows)
    violation_names = (
        "future_slip_precursor_ordinary_negative",
        "censored_precursor_negative",
        "i1_or_positive_negative",
        "post_censor_or_fall",
        "pre_i1_delayed_support_positive",
        "future_feature_leakage",
        "eligible_delayed_support_short_neighborhood",
        "persistence_neighborhood_shorter_than_5ms",
    )
    contradiction = {name: int(violations[name]) for name in violation_names}
    fit_negative = sum(row["partition"] == "fit" for row in negative_rows)
    return {
        "effective_train_run_count": len(runs),
        "extraction_policy_sha256": canonical_sha256(
            model_v2_rebalance_policy()
        ),
        "positive_window_ids_sha256": _endpoint_identity_hash(positive_rows),
        "negative_window_ids_sha256": _endpoint_identity_hash(negative_rows),
        "masked_window_sha256": canonical_sha256(mask_rows),
        "all_positive_count": len(positive_rows),
        "all_negative_count": len(negative_rows),
        "fit_positive_counts": {
            "slip": slip_fit,
            "ordinary_support": ordinary_fit,
            "delayed_support": len(delayed_fit_rows),
            "support": ordinary_fit + len(delayed_fit_rows),
            "total": len(fit_positive_rows),
        },
        "monitor_positive_count": len(monitor_positive_rows),
        "fit_negative_count": fit_negative,
        "monitor_negative_count": len(negative_rows) - fit_negative,
        "delayed_support": {
            "eligible_runs": len(eligible_delayed),
            "fit_represented_runs": len(
                {str(row["run_id"]) for row in delayed_fit_rows}
            ),
            "by_source": {
                source: {
                    "eligible_runs": sum(
                        runs[run_id].source_terrain == source
                        for run_id in eligible_delayed
                    ),
                    "fit_positive_windows": sum(
                        runs[str(row["run_id"])].source_terrain == source
                        for row in delayed_fit_rows
                    ),
                }
                for source in ("concrete", "marble")
            },
        },
        "masked_sample_counts": {
            "future_slip_precursor": sum(
                int(row["future_slip_precursor_samples"]) for row in mask_rows
            ),
            "censored_precursor": sum(
                int(row["censored_precursor_samples"]) for row in mask_rows
            ),
            "i1_positive": sum(
                int(row["i1_positive_samples"]) for row in mask_rows
            ),
        },
        "contradiction_audit": contradiction,
        "passed": not any(contradiction.values()),
    }


def _merge_endpoint_maps(
    *values: Mapping[str, Sequence[int]],
) -> dict[str, tuple[int, ...]]:
    keys = {key for mapping in values for key in mapping}
    return {
        key: tuple(
            sorted({int(value) for mapping in values for value in mapping.get(key, ())})
        )
        for key in keys
    }


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {key: None for key in ("min", "p10", "median", "p95", "max")}
    return {
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _event_side(run: HazardRun, event: str) -> str:
    samples = (
        run.slip_event_samples_per_foot
        if event == "slip"
        else run.support_event_samples_per_foot
    )
    active = [index for index, value in enumerate(samples) if value is not None]
    if active == [0]:
        return "left"
    if active == [1]:
        return "right"
    if active == [0, 1]:
        return "bilateral"
    return "none"


def positive_exposure_summary(
    windows: WindowSet,
    runs: Mapping[str, HazardRun],
    annotations: Mapping[str, HazardRunAnnotations],
) -> dict[str, int]:
    """Count every positive endpoint by the predeclared exposure cells."""
    counts: Counter[str] = Counter()
    for run_id_value, endpoint_value, target in zip(
        windows.run_ids, windows.endpoint_samples, windows.targets
    ):
        if int(target) != 1:
            continue
        run_id = str(run_id_value)
        endpoint = int(endpoint_value)
        run = runs[run_id]
        annotation = annotations[run_id]
        role = _positive_role(run, endpoint)
        counts[role] += 1
        if role == "support":
            kind = (
                "delayed_support"
                if annotation.scenario_family == MODEL_V2_DELAYED_SUPPORT_FAMILY
                else "ordinary_support"
            )
            counts[kind] += 1
            counts[f"{kind}_{run.source_terrain}"] += 1
            counts[f"support_{_event_side(run, 'support')}"] += 1
        else:
            counts[f"slip_speed_{annotation.nominal_speed_mps:.2f}"] += 1
            counts[f"slip_{_event_side(run, 'slip')}"] += 1
    counts["total"] = int(np.sum(windows.targets == 1))
    names = (
        "total",
        "slip",
        "support",
        "ordinary_support",
        "ordinary_support_concrete",
        "ordinary_support_marble",
        "delayed_support",
        "delayed_support_concrete",
        "delayed_support_marble",
        "support_left",
        "support_right",
        "support_bilateral",
        "slip_speed_0.20",
        "slip_speed_0.25",
        "slip_speed_0.30",
        "slip_left",
        "slip_right",
        "slip_bilateral",
    )
    return {name: int(counts[name]) for name in names}


def _mine_training_round(
    runs: Mapping[str, HazardRun],
    precursor_samples: Mapping[str, int | None],
    normalizer: Normalizer,
    checkpoint_paths: Sequence[Path],
    prior: Mapping[str, Sequence[int]],
    annotations: Mapping[str, HazardRunAnnotations],
) -> tuple[dict[str, tuple[int, ...]], dict[str, object]]:
    models = [load_checkpoint(path)[0] for path in checkpoint_paths]
    result: dict[str, tuple[int, ...]] = {}
    selected_scores: list[float] = []
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    speed_counts: Counter[str] = Counter()
    contributing_runs = 0
    duplicate_count = 0
    spacing_violations = 0
    forbidden_violations = Counter()
    for run_id, run in sorted(runs.items()):
        annotation = annotations[run_id]
        candidates = training_negative_candidates(
            run, precursor_samples.get(run_id), annotation, 1
        )
        replay = replay_hazard_run(run, normalizer, models)
        common, _, replay_indices = np.intersect1d(
            candidates, replay.endpoints, return_indices=True
        )
        scores = replay.probabilities[replay_indices]
        selected = mine_hard_negative_endpoints(
            common, scores, excluded=prior.get(run_id, ())
        )
        result[run_id] = tuple(int(value) for value in selected)
        if len(selected):
            contributing_runs += 1
            family_counts[annotation.scenario_family] += len(selected)
            source_counts[run.source_terrain] += len(selected)
            speed_counts[f"{annotation.nominal_speed_mps:.2f}"] += len(selected)
        duplicate_count += len(set(selected) & set(prior.get(run_id, ())))
        spacing_violations += int(np.sum(np.diff(selected) < HNM_MINIMUM_SPACING_MS))
        for endpoint in selected:
            endpoint = int(endpoint)
            if annotation.future_slip_precursor[endpoint]:
                forbidden_violations["future_slip_precursor"] += 1
            if annotation.censored_precursor[endpoint]:
                forbidden_violations["censored_precursor"] += 1
            if annotation.i1_active[endpoint]:
                forbidden_violations["i1_positive"] += 1
            if endpoint >= run.censor_sample or (
                run.fall_sample_diagnostic is not None
                and endpoint >= run.fall_sample_diagnostic
            ):
                forbidden_violations["post_censor_or_fall"] += 1
        lookup = {int(endpoint): float(score) for endpoint, score in zip(common, scores)}
        selected_scores.extend(lookup[int(endpoint)] for endpoint in selected)
    return result, {
        "runs_scored": len(runs),
        "mined_windows": sum(len(value) for value in result.values()),
        "runs_contributing": contributing_runs,
        "selected_by_family": dict(sorted(family_counts.items())),
        "selected_by_source": dict(sorted(source_counts.items())),
        "selected_by_speed": dict(sorted(speed_counts.items())),
        "selected_probability": _distribution(selected_scores),
        "train_only": True,
        "replay_stride_ms": HNM_REPLAY_STRIDE_MS,
        "top_k_per_run": HNM_TOP_K_PER_RUN,
        "minimum_spacing_ms": HNM_MINIMUM_SPACING_MS,
        "precursor_region_never_negative": True,
        "duplicate_mined_windows": duplicate_count,
        "spacing_violations": spacing_violations,
        "forbidden_mask_violations": {
            name: int(forbidden_violations[name])
            for name in (
                "future_slip_precursor",
                "censored_precursor",
                "i1_positive",
                "post_censor_or_fall",
            )
        },
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def train_hazard_candidate(
    repository_root: Path,
    runs: Mapping[str, HazardRun],
    precursor_samples: Mapping[str, int | None],
    artifact_path: Path,
    training_config: Mapping[str, object],
    annotations: Mapping[str, HazardRunAnnotations],
    checkpoint_prefix: str = "unified",
    progress: Callable[[str], None] = print,
    normalizer_override: Normalizer | None = None,
    normalizer_source_path: Path | None = None,
    fit_positive_endpoints: Mapping[str, Sequence[int]] | None = None,
    monitor_positive_endpoints: Mapping[str, Sequence[int]] | None = None,
    extraction_audit_override: Mapping[str, object] | None = None,
) -> HazardCandidate:
    """Train Round 0 plus exactly three TRAIN-only hard-negative rounds."""
    if any(run.split != "train" for run in runs.values()):
        raise ValueError("Hazard candidate training may receive TRAIN runs only")
    if set(runs) != set(annotations):
        raise ValueError("every Hazard TRAIN run requires frozen annotations")
    root = repository_root.resolve()
    fit_ids, monitor_ids = _train_monitor_partition(
        runs, sorted(runs), precursor_samples
    )
    extraction_audit = (
        audit_hazard_extraction(
            runs,
            sorted(runs),
            precursor_samples,
            annotations,
            per_category=int(
                training_config.get("initial_negative_per_gait_category", 12)
            ),
            positive_cap=int(training_config.get("positive_cap_per_run", 20)),
            target_contact_cap=int(
                training_config.get("target_contact_cap_per_run", 12)
            ),
            benign_precursor_cap=int(
                training_config.get("benign_precursor_cap_per_run", 12)
            ),
        )
        if extraction_audit_override is None
        else dict(extraction_audit_override)
    )
    if not extraction_audit["passed"]:
        raise RuntimeError("Hazard extraction violated a frozen negative mask")
    extraction_path = artifact_path / "extraction_audit.json"
    _write_json(extraction_path, extraction_audit)
    schema = hazard_feature_schema()
    if normalizer_override is None:
        normalizer = fit_hazard_normalizer(runs, sorted(runs))
        normalizer_path = artifact_path / "normalization" / "gru_history20.json"
        _write_json(
            normalizer_path,
            {
                **normalizer.to_dict(),
                "components": ["pelvis_imu6"],
                "feature_schema": list(schema),
                "feature_schema_sha256": canonical_sha256(schema),
                "train_only": True,
            },
        )
        normalizer_fits = 1
    else:
        if normalizer_source_path is None or not normalizer_source_path.is_file():
            raise ValueError("reused normalizer requires an existing source artifact")
        normalizer = normalizer_override
        normalizer_path = normalizer_source_path.resolve()
        normalizer_fits = 0
    fit_material_ids = sorted(
        set(fit_ids)
        | (set(fit_positive_endpoints) if fit_positive_endpoints is not None else set())
    )
    monitor_material_ids = sorted(
        set(monitor_ids)
        | (
            set(monitor_positive_endpoints)
            if monitor_positive_endpoints is not None
            else set()
        )
    )
    accumulated: dict[str, tuple[int, ...]] = {}
    round_records: list[dict[str, object]] = []
    final_paths: tuple[Path, ...] = ()
    for round_id in range(HNM_ROUNDS + 1):
        fit_windows = build_hazard_windows(
            runs,
            fit_material_ids,
            precursor_samples,
            normalizer,
            extra_negative_endpoints=accumulated,
            annotations=annotations,
            per_category=int(
                training_config.get("initial_negative_per_gait_category", 12)
            ),
            positive_cap=int(training_config.get("positive_cap_per_run", 20)),
            target_contact_cap=int(
                training_config.get("target_contact_cap_per_run", 12)
            ),
            benign_precursor_cap=int(
                training_config.get("benign_precursor_cap_per_run", 12)
            ),
            positive_endpoints=fit_positive_endpoints,
            negative_run_ids=fit_ids,
        )
        monitor_windows = build_hazard_windows(
            runs,
            monitor_material_ids,
            precursor_samples,
            normalizer,
            extra_negative_endpoints=accumulated,
            annotations=annotations,
            per_category=int(
                training_config.get("initial_negative_per_gait_category", 12)
            ),
            positive_cap=int(training_config.get("positive_cap_per_run", 20)),
            target_contact_cap=int(
                training_config.get("target_contact_cap_per_run", 12)
            ),
            benign_precursor_cap=int(
                training_config.get("benign_precursor_cap_per_run", 12)
            ),
            positive_endpoints=monitor_positive_endpoints,
            negative_run_ids=monitor_ids,
        )
        exposure = positive_exposure_summary(fit_windows, runs, annotations)
        paths: list[Path] = []
        epochs: list[int] = []
        epochs_completed: list[int] = []
        optimizer_steps: list[int] = []
        final_train_losses: list[float] = []
        best_validation_losses: list[float] = []
        for seed in training_config["seeds"]:
            path = (
                artifact_path
                / "checkpoints"
                / f"{checkpoint_prefix}_gru_history20_round{round_id}_seed{seed}.pt"
            )
            model, result = train_model(
                "gru",
                HISTORY_MS,
                fit_windows,
                monitor_windows,
                int(seed),
                batch_size=int(training_config["batch_size"]),
                max_epochs=int(training_config["max_epochs"]),
                patience=int(training_config["patience"]),
                learning_rate=float(training_config["learning_rate"]),
                class_names=EVENT_CLASS_NAMES,
                selection_metric="validation_loss",
            )
            save_checkpoint(
                path,
                model,
                "gru",
                HISTORY_MS,
                int(seed),
                result,
                input_channels=len(schema),
                class_names=EVENT_CLASS_NAMES,
            )
            paths.append(path)
            epochs.append(result.best_epoch)
            epochs_completed.append(result.epochs_completed)
            optimizer_steps.append(
                result.epochs_completed
                * math.ceil(len(fit_windows) / int(training_config["batch_size"]))
            )
            final_train_losses.append(float(result.history[-1]["train_loss"]))
            best_validation_losses.append(
                float(
                    result.history[result.best_epoch - 1][
                        "validation_cross_entropy"
                    ]
                )
            )
        final_paths = tuple(paths)
        record: dict[str, object] = {
            "round": round_id,
            "fit_windows": len(fit_windows),
            "monitor_windows": len(monitor_windows),
            "fit_class_counts": list(fit_windows.selected_by_class),
            "monitor_class_counts": list(monitor_windows.selected_by_class),
            "best_epochs": epochs,
            "epochs_completed": epochs_completed,
            "optimizer_steps": optimizer_steps,
            "final_train_loss": final_train_losses,
            "best_validation_cross_entropy": best_validation_losses,
            "positive_exposure_available": exposure,
            "positive_batch_exposure_by_seed": {
                str(seed): {
                    name: int(count) * int(epochs_completed[index])
                    for name, count in exposure.items()
                }
                for index, seed in enumerate(training_config["seeds"])
            },
            "checkpoint_sha256": {
                str(path.relative_to(root)): sha256_file(path) for path in paths
            },
        }
        progress(f"Hazard history=20 round={round_id} trained")
        if round_id < HNM_ROUNDS:
            mined, mining_record = _mine_training_round(
                runs,
                precursor_samples,
                normalizer,
                final_paths,
                accumulated,
                annotations,
            )
            accumulated = _merge_endpoint_maps(accumulated, mined)
            record["hard_negative_mining"] = mining_record
        round_records.append(record)
    return HazardCandidate(
        history_ms=HISTORY_MS,
        normalizer=normalizer,
        checkpoint_paths=final_paths,
        record={
            "model_family": "gru",
            "history_ms": HISTORY_MS,
            "feature_dimension": len(schema),
            "feature_schema_sha256": canonical_sha256(schema),
            "normalizer_path": str(normalizer_path.relative_to(root)),
            "normalizer_sha256": sha256_file(normalizer_path),
            "normalizer_sample_count": normalizer.sample_count,
            "normalizer_fit_run_ids_sha256": canonical_sha256(
                list(normalizer.fit_run_ids)
            ),
            "normalizer_fits": normalizer_fits,
            "extraction_audit_path": str(extraction_path.relative_to(root)),
            "extraction_audit_sha256": sha256_file(extraction_path),
            "checkpoint_sha256": {
                str(path.relative_to(root)): sha256_file(path) for path in final_paths
            },
            "parameters": parameter_count(load_checkpoint(final_paths[0])[0]),
            "rounds": round_records,
            "hnm_rounds": HNM_ROUNDS,
            "optimizer_steps": sum(
                sum(int(value) for value in row["optimizer_steps"])
                for row in round_records
            ),
            "checkpoint_writes": (HNM_ROUNDS + 1)
            * len(training_config["seeds"]),
            "validation_access_before_hnm3": False,
        },
    )


def _verified_manifest_files(
    dataset_path: Path,
    expected_manifest_sha: str,
) -> dict[str, object]:
    manifest_path = dataset_path / "manifest.json"
    if sha256_file(manifest_path) != expected_manifest_sha:
        raise RuntimeError(f"frozen manifest changed: {dataset_path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified = 0
    hashes: dict[str, str] = {}
    for row in manifest["runs"]:
        path = dataset_path / str(row["file"])
        actual = sha256_file(path)
        if actual != str(row["file_sha256"]):
            raise RuntimeError(f"frozen NPZ changed: {path}")
        hashes[str(row["file"])] = actual
        verified += 1
    return {
        "manifest": manifest,
        "manifest_sha256": expected_manifest_sha,
        "npz_verified": verified,
        "npz_aggregate_sha256": canonical_sha256(hashes),
    }


def _verify_protected_records(root: Path, records: Sequence[Mapping[str, object]]) -> None:
    for record in records:
        path = root / str(record["path"])
        if sha256_file(path) != str(record["sha256"]):
            raise RuntimeError(f"protected artifact changed: {record['path']}")


def _effective_training_composition(
    runs: Mapping[str, HazardRun],
    precursor_samples: Mapping[str, int | None],
    annotations: Mapping[str, HazardRunAnnotations],
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    source: Counter[str] = Counter()
    hazard_speed: Counter[str] = Counter()
    side: Counter[str] = Counter()
    for run_id, run in runs.items():
        annotation = annotations[run_id]
        slip = slip_event_sample(run) is not None
        support = support_event_sample(run) is not None
        hazard = slip or support
        ambiguous = (
            not hazard
            and (
                precursor_samples.get(run_id) is not None
                or bool(np.any(annotation.censored_precursor[: run.censor_sample]))
            )
        )
        counts["hazard"] += int(hazard)
        counts["confirmed_no_hazard"] += int(not hazard and not ambiguous)
        counts["ambiguous"] += int(ambiguous)
        counts["slip"] += int(slip)
        counts["support"] += int(support)
        counts["dual_slip_support"] += int(slip and support)
        source[run.source_terrain] += 1
        side[annotation.actual_side] += 1
        if hazard:
            hazard_speed[f"{annotation.nominal_speed_mps:.2f}"] += 1
    return {
        "total": len(runs),
        **{key: int(value) for key, value in sorted(counts.items())},
        "source": dict(sorted(source.items())),
        "hazard_speed": dict(sorted(hazard_speed.items())),
        "actual_side": dict(sorted(side.items())),
    }


def prepare_model_v2_training_data(
    root: Path, document: Mapping[str, object]
) -> ModelV2TrainingData:
    """Verify every frozen input and open only authorized TRAIN waveforms."""
    protected = document["protected_v1"]
    _verify_protected_records(root, protected["checkpoints"])
    _verify_protected_records(root, protected["terrain_checkpoints"])
    _verify_protected_records(root, (protected["terrain_normalizer"],))
    v1_normalizer = root / "artifacts/runs/20260829_unified_hazard_reflex_system/normalization/gru_history20.json"
    if sha256_file(v1_normalizer) != str(protected["normalizer_sha256"]):
        raise RuntimeError("protected Hazard V1 normalizer changed")

    dataset_config = document["v2_dataset"]
    v2_path = root / str(dataset_config["path"])
    v2_integrity = _verified_manifest_files(
        v2_path, str(dataset_config["manifest_sha256"])
    )
    freeze_path = v2_path / "dataset_freeze.json"
    if sha256_file(freeze_path) != str(dataset_config["dataset_freeze_sha256"]):
        raise RuntimeError("Model V2 dataset freeze changed")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    manifest = v2_integrity["manifest"]
    expected = dataset_config["expected"]
    train_rows = [
        row for row in manifest["runs"] if row["split"] == "V2_TRAIN"
    ]
    validation_rows = [
        row for row in manifest["runs"] if row["split"] == "V2_VALIDATION"
    ]
    if (
        int(manifest["run_count"]) != int(expected["designed"])
        or int(manifest["valid_count"]) != int(expected["valid"])
        or int(manifest["invalid_count"]) != int(expected["invalid"])
        or sum(bool(row["valid"]) for row in train_rows)
        != int(expected["valid_train"])
        or sum(bool(row["valid"]) for row in validation_rows)
        != int(expected["valid_validation"])
        or v2_integrity["npz_aggregate_sha256"]
        != str(dataset_config["npz_aggregate_sha256"])
        or manifest["physical_signature_sha256"]
        != str(dataset_config["physical_signature_sha256"])
        or manifest["split_sha256"]["V2_TRAIN"]
        != str(dataset_config["train_split_sha256"])
        or manifest["split_sha256"]["V2_VALIDATION"]
        != str(dataset_config["validation_split_sha256"])
        or any(
            int(value) != 0
            for value in freeze["historical_exclusion_integrity"][
                "overlap_by_reference"
            ].values()
        )
    ):
        raise RuntimeError("Model V2 dataset freeze contract changed")

    unified_config = document["effective_train"]["unified"]
    unified_path = root / str(unified_config["path"])
    unified_integrity = _verified_manifest_files(
        unified_path, str(unified_config["manifest_sha256"])
    )
    unified_manifest = unified_integrity["manifest"]
    unified_runs = load_hazard_runs(unified_path, unified_manifest, ("train",))
    v2_manifest = load_model_v2_manifest(v2_path)
    v2_runs, v2_annotations = load_model_v2_runs(
        v2_path, v2_manifest, "V2_TRAIN"
    )
    unified_annotations = unified_training_annotations(
        unified_manifest, unified_runs
    )
    runs = {**unified_runs, **v2_runs}
    annotations = {**unified_annotations, **v2_annotations}
    unified_rows = {
        str(row["run_id"]): row
        for row in unified_manifest["runs"]
        if row["split"] == "train"
    }
    v2_train_rows = {
        str(row["run_id"]): row
        for row in v2_manifest["runs"]
        if row["split"] == "V2_TRAIN" and bool(row["valid"])
    }
    precursor_samples = {
        **{
            run_id: (
                None
                if row["support_precursor_sample"] is None
                else int(row["support_precursor_sample"])
            )
            for run_id, row in unified_rows.items()
        },
        **{
            run_id: (
                None
                if row["i1_summary"]["first_sample"] is None
                else int(row["i1_summary"]["first_sample"])
            )
            for run_id, row in v2_train_rows.items()
        },
    }
    unified_ids = list(unified_rows)
    v2_ids = list(v2_train_rows)
    effective_identity = [
        {"dataset_id": "unified_hazard_reflex_20260829", "run_id": run_id}
        for run_id in unified_ids
    ] + [
        {"dataset_id": "model_v2_hazard_reflex_20260901", "run_id": run_id}
        for run_id in v2_ids
    ]
    if (
        len(runs) != int(document["effective_train"]["total_run_count"])
        or canonical_sha256(unified_ids)
        != str(unified_config["run_ids_sha256"])
        or canonical_sha256(v2_ids)
        != str(
            document["effective_train"]["augmentation"][
                "valid_run_ids_sha256"
            ]
        )
        or canonical_sha256(effective_identity)
        != str(document["effective_train"]["effective_run_ids_sha256"])
        or set(unified_runs) & set(v2_runs)
    ):
        raise RuntimeError("effective TRAIN identity changed")
    composition = _effective_training_composition(
        runs, precursor_samples, annotations
    )
    return ModelV2TrainingData(
        runs=runs,
        precursor_samples=precursor_samples,
        annotations=annotations,
        input_audit={
            "model_v1_restorable": True,
            "v2_dataset_modified_after_freeze": False,
            "v2_npz_verified": v2_integrity["npz_verified"],
            "unified_npz_verified": unified_integrity["npz_verified"],
            "historical_signature_overlap": 0,
            "train_validation_overlap": 0,
            "cross_split_near_duplicates": 0,
            "v2_validation_optimizer_leakage": 0,
            "generalization_validation_training_leakage": 0,
            "holdout_training_leakage": 0,
            "generalization_holdout_guard_count": 0,
        },
        composition=composition,
        v2_manifest=v2_manifest,
    )


def _training_recipe(document: Mapping[str, object]) -> dict[str, object]:
    window = document["window_extraction"]
    return {
        **document["training"],
        "positive_cap_per_run": int(
            window.get(
                "positive_cap_per_run",
                window["slip_positive"]["legacy_union_cap_per_run"],
            )
        ),
        "initial_negative_per_gait_category": int(
            window["initial_negative"]["per_gait_category"]
        ),
        "target_contact_cap_per_run": int(
            window["initial_negative"]["explicit_target_contact_cap_per_run"]
        ),
        "benign_precursor_cap_per_run": int(
            window["initial_negative"]["benign_precursor_cap_per_run"]
        ),
    }


def _comparison_metrics(
    v1: Mapping[str, object], v2: Mapping[str, object]
) -> dict[str, object]:
    def metric(value: Mapping[str, object], path: Sequence[str]) -> float:
        current: object = value
        for key in path:
            current = current[key]  # type: ignore[index]
        return float(current)

    paths = {
        "overall_hazard_recall": ("primary", "overall_hazard_recall"),
        "slip_recall": ("primary", "slip_hazard_recall"),
        "support_recall": ("primary", "support_hazard_recall"),
        "confirmed_no_hazard_specificity": (
            "primary",
            "primary_no_hazard_specificity",
        ),
        "premature_rate": ("primary", "system_premature_run_rate"),
        "right_only_support_recall": (
            "side",
            "support",
            "RIGHT_ONLY",
            "recall",
        ),
        "staged_sand_specificity": (
            "families",
            "STAGED_SAND_BENIGN_CONTROL",
            "specificity",
        ),
        "ice_benign_specificity": (
            "families",
            "ICE_BENIGN_CONTROL",
            "specificity",
        ),
    }
    return {
        name: {
            "v1": metric(v1, path),
            "v2": metric(v2, path),
            "delta": metric(v2, path) - metric(v1, path),
        }
        for name, path in paths.items()
    }


def _longest_threshold_excursion(values: np.ndarray, threshold: float) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if float(value) >= threshold else 0
        longest = max(longest, current)
    return longest


def _probability_at_sample(replay, sample: int) -> float | None:
    matches = np.flatnonzero(replay.endpoints == int(sample))
    return None if not len(matches) else float(replay.probabilities[int(matches[0])])


def _delayed_support_validation_diagnostics(
    runs: Mapping[str, HazardRun],
    replays: Mapping[str, object],
    precursor_samples: Mapping[str, int | None],
    manifest_rows: Mapping[str, Mapping[str, object]],
    validation_result: Mapping[str, object],
    *,
    threshold: float,
    persistence_ms: int,
) -> dict[str, object]:
    primary_rows = {
        str(row["run_id"]): row for row in validation_result["primary"]["rows"]
    }
    rows: list[dict[str, object]] = []
    for run_id in sorted(runs):
        manifest_row = manifest_rows[run_id]
        if manifest_row["scenario_family"] != MODEL_V2_DELAYED_SUPPORT_FAMILY:
            continue
        run = runs[run_id]
        replay = replays[run_id]
        i1 = precursor_samples[run_id]
        support = support_event_sample(run)
        if i1 is None or support is None:
            raise RuntimeError("V2_VALIDATION delayed Support lost a physical anchor")
        crossings = replay.endpoints[replay.probabilities >= threshold]
        onsets = reflex_onset_samples(
            replay, threshold=threshold, persistence_ms=persistence_ms
        )
        primary = primary_rows[run_id]
        first_crossing = None if not len(crossings) else int(crossings[0])
        first_reflex = None if not len(onsets) else int(onsets[0])
        result = (
            "correct"
            if bool(primary["valid_detection"])
            else "premature"
            if bool(primary["premature"])
            else "miss"
            if first_reflex is None
            else "out_of_valid_window"
        )
        rows.append(
            {
                "run_id": run_id,
                "source": run.source_terrain,
                "i1_sample": int(i1),
                "support_sample": support,
                "first_threshold_crossing": first_crossing,
                "first_reflex": first_reflex,
                "probability_at_i1": _probability_at_sample(replay, int(i1)),
                "probability_at_support": _probability_at_sample(replay, support),
                "maximum_probability": float(np.max(replay.probabilities)),
                "maximum_consecutive_at_or_above_threshold_ms": (
                    _longest_threshold_excursion(replay.probabilities, threshold)
                ),
                "primary_result": result,
                "i1_to_reflex_ms": (
                    None if first_reflex is None else first_reflex - int(i1)
                ),
                "reflex_to_support_ms": (
                    None if first_reflex is None else support - first_reflex
                ),
                "established_support_latency_ms": (
                    None
                    if primary["support_valid_detection"] is None
                    else int(primary["support_valid_detection"]) - support
                ),
            }
        )
    by_source = {}
    for source in ("concrete", "marble"):
        selected = [row for row in rows if row["source"] == source]
        by_source[source] = {
            "correct": sum(row["primary_result"] == "correct" for row in selected),
            "runs": len(selected),
            "recall": (
                None
                if not selected
                else sum(row["primary_result"] == "correct" for row in selected)
                / len(selected)
            ),
            "probability_at_i1": _distribution(
                [row["probability_at_i1"] for row in selected]
            ),
            "maximum_consecutive_at_or_above_threshold_ms": _distribution(
                [
                    row["maximum_consecutive_at_or_above_threshold_ms"]
                    for row in selected
                ]
            ),
        }
    return {
        "rows": rows,
        "overall": {
            "correct": sum(row["primary_result"] == "correct" for row in rows),
            "runs": len(rows),
            "recall": sum(row["primary_result"] == "correct" for row in rows)
            / len(rows),
            "miss_or_out_of_window": sum(
                row["primary_result"] in ("miss", "out_of_valid_window")
                for row in rows
            ),
        },
        "by_source": by_source,
    }


def _validation_preservation_summary(
    runs: Mapping[str, HazardRun],
    replays: Mapping[str, object],
    annotations: Mapping[str, HazardRunAnnotations],
    precursor_samples: Mapping[str, int | None],
    manifest_rows: Mapping[str, Mapping[str, object]],
    validation_result: Mapping[str, object],
    delayed: Mapping[str, object],
) -> dict[str, object]:
    primary_rows = {
        str(row["run_id"]): row for row in validation_result["primary"]["rows"]
    }

    def recall(selected: Sequence[Mapping[str, object]]) -> dict[str, object]:
        return {
            "correct": sum(bool(row["valid_detection"]) for row in selected),
            "runs": len(selected),
            "recall": (
                None
                if not selected
                else sum(bool(row["valid_detection"]) for row in selected)
                / len(selected)
            ),
        }

    slip_rows = [
        row for row in primary_rows.values() if row["slip_sample"] is not None
    ]
    slip_timing = {"immediate": [], "delayed": []}
    for row in slip_rows:
        run_id = str(row["run_id"])
        if annotations[run_id].dataset_id == "unified_hazard_reflex_20260829":
            name = "immediate"
        else:
            classification = str(
                manifest_rows[run_id]["delayed_ice_summary"]["classification"]
            )
            name = "immediate" if classification == "IMMEDIATE_SLIP" else "delayed"
        slip_timing[name].append(row)

    ordinary_support = [
        row
        for run_id, row in primary_rows.items()
        if row["support_sample"] is not None
        and manifest_rows[run_id]["scenario_family"]
        != MODEL_V2_DELAYED_SUPPORT_FAMILY
    ]
    ordinary_side = {
        side: recall(
            [
                row
                for row in ordinary_support
                if manifest_rows[str(row["run_id"])]["support_event_summary"][
                    "side"
                ]
                == side
            ]
        )
        for side in ("LEFT_ONLY", "RIGHT_ONLY")
    }
    premature_categories: Counter[str] = Counter()
    for run_id, row in primary_rows.items():
        if not bool(row["premature"]):
            continue
        onset = row["system_first_onset"]
        if onset is None:
            continue
        sample = int(onset)
        annotation = annotations[run_id]
        if annotation.future_slip_precursor[sample]:
            name = "inside_future_slip_precursor"
        elif annotation.benign_release_precursor[sample]:
            name = "inside_benign_release"
        elif annotation.censored_precursor[sample]:
            name = "inside_censored_precursor"
        else:
            name = "before_or_outside_precursor"
        premature_categories[name] += 1

    return {
        "ordinary_support": {
            "overall": recall(ordinary_support),
            "by_side": ordinary_side,
        },
        "slip": {
            "overall": recall(slip_rows),
            "by_speed": validation_result["speed"],
            "by_side": validation_result["side"]["slip"],
            "by_timing": {
                name: recall(rows) for name, rows in slip_timing.items()
            },
        },
        "specificity": {
            "confirmed_no_hazard": {
                "rate": validation_result["primary"][
                    "primary_no_hazard_specificity"
                ],
                "runs": validation_result["primary"]["primary_no_hazard_runs"],
            },
            "hard_normal": validation_result["families"][
                "HARD_GROUND_NORMAL_SPEED_MATRIX"
            ],
            "ice_benign": validation_result["families"]["ICE_BENIGN_CONTROL"],
            "staged_sand": validation_result["families"][
                "STAGED_SAND_BENIGN_CONTROL"
            ],
            "speed_sand": validation_result["families"][
                "SPEED_STRATIFIED_SAND_BENIGN"
            ],
        },
        "delayed_support": delayed,
        "premature_ice_classification": {
            name: int(premature_categories[name])
            for name in (
                "inside_future_slip_precursor",
                "before_or_outside_precursor",
                "inside_benign_release",
                "inside_censored_precursor",
            )
        },
        "ice_precursor_secondary": validation_result["ice_precursor_secondary"],
        "primary_scores_rewritten": False,
    }


def _three_model_comparison(
    v1: Mapping[str, object],
    baseline: Mapping[str, object],
    rebalanced: Mapping[str, object],
    delayed: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    def model_metrics(
        result: Mapping[str, object], delayed_result: Mapping[str, object]
    ) -> dict[str, object]:
        primary = result["primary"]
        return {
            "overall_hazard": primary["overall_hazard_recall"],
            "slip": primary["slip_hazard_recall"],
            "support": primary["support_hazard_recall"],
            "confirmed_specificity": primary["primary_no_hazard_specificity"],
            "premature": primary["system_premature_run_rate"],
            "right_only_support": result["side"]["support"]["RIGHT_ONLY"][
                "recall"
            ],
            "delayed_support": delayed_result["overall"]["recall"],
            "marble_delayed_support": delayed_result["by_source"]["marble"][
                "recall"
            ],
            "staged_sand_specificity": result["families"][
                "STAGED_SAND_BENIGN_CONTROL"
            ]["specificity"],
            "speed_sand_specificity": result["families"][
                "SPEED_STRATIFIED_SAND_BENIGN"
            ]["specificity"],
        }

    return {
        "v1": model_metrics(v1, delayed["v1"]),
        "baseline_v2": model_metrics(baseline, delayed["baseline_v2"]),
        "rebalanced_v2": model_metrics(rebalanced, delayed["rebalanced_v2"]),
    }


def run_model_v2_data_only_training(
    root: Path,
    config_path: Path,
    *,
    dry_run: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Run the frozen data-only protocol and stop before external evidence."""
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if document["experiment"]["id"] != "MODEL_V2_DATA_ONLY_TRAINING":
        raise ValueError("unsupported Model V2 training config")
    config_sha = sha256_file(config_path)
    data = prepare_model_v2_training_data(root, document)
    recipe = _training_recipe(document)
    extraction = audit_hazard_extraction(
        data.runs,
        sorted(data.runs),
        data.precursor_samples,
        data.annotations,
        per_category=int(recipe["initial_negative_per_gait_category"]),
        positive_cap=int(recipe["positive_cap_per_run"]),
        target_contact_cap=int(recipe["target_contact_cap_per_run"]),
        benign_precursor_cap=int(recipe["benign_precursor_cap_per_run"]),
    )
    if not extraction["passed"]:
        raise RuntimeError("pretraining extraction audit failed")
    policy = {
        "window_extraction": document["window_extraction"],
        "sampling": document["sampling"],
        "hnm": document["hnm"],
    }
    artifact_path = root / str(document["artifacts"]["path"])
    pretraining_path = artifact_path / "pretraining_audit.json"
    pretraining = {
        "training_config_sha256": config_sha,
        "input_audit": data.input_audit,
        "effective_train_composition": data.composition,
        "extraction_policy_sha256": canonical_sha256(policy),
        "extraction_audit": extraction,
        "optimizer_steps": 0,
        "normalizer_fits": 0,
        "v2_validation_waveform_opened": False,
        "generalization_validation_v2_inference": False,
        "generalization_holdout_guard_count": 0,
    }
    if dry_run:
        _write_json(pretraining_path, pretraining)
        return {
            "status": "MODEL_V2_PRETRAINING_AUDIT_READY",
            "training_config_sha256": config_sha,
            "pretraining_audit_sha256": sha256_file(pretraining_path),
            "composition": data.composition,
            "extraction": extraction,
        }
    if not pretraining_path.is_file():
        raise RuntimeError("run the frozen dry-run extraction audit before training")
    frozen_pretraining = json.loads(pretraining_path.read_text(encoding="utf-8"))
    if frozen_pretraining != pretraining:
        raise RuntimeError("pretraining audit differs from current frozen extraction")
    forbidden_existing = [
        artifact_path / "normalization/gru_history20.json",
        artifact_path / "candidate_freeze.json",
        artifact_path / "v2_validation_evaluation.json",
    ]
    if any(path.exists() for path in forbidden_existing) or any(
        (artifact_path / "checkpoints").glob("*.pt")
    ):
        raise RuntimeError("Model V2 training artifacts already exist")

    candidate = train_hazard_candidate(
        root,
        data.runs,
        data.precursor_samples,
        artifact_path,
        recipe,
        data.annotations,
        checkpoint_prefix="model_v2_data_only",
        progress=progress,
    )
    training_record = {
        "training_config_sha256": config_sha,
        "effective_train_ids_sha256": document["effective_train"][
            "effective_run_ids_sha256"
        ],
        "candidate": candidate.record,
        "normalizer_fits": 1,
        "threshold_searches": 0,
        "persistence_searches": 0,
        "architecture_searches": 0,
        "seed_searches": 0,
    }
    training_record_path = artifact_path / "training_record.json"
    _write_json(training_record_path, training_record)
    hnm_provenance = {
        "rounds": [
            row["hard_negative_mining"]
            for row in candidate.record["rounds"]
            if "hard_negative_mining" in row
        ],
        "source": "effective_train_only",
        "forbidden_splits": [
            "V2_VALIDATION",
            "Generalization_VALIDATION",
            "Generalization_HOLDOUT",
            "Unified_HOLDOUT",
        ],
    }
    hnm_path = artifact_path / "hnm_provenance.json"
    _write_json(hnm_path, hnm_provenance)
    architecture_sha = canonical_sha256(document["architecture"])
    candidate_freeze = {
        "candidate_id": document["artifacts"]["candidate_id"],
        "source_commit": document["experiment"]["source_commit"],
        "training_config_sha256": config_sha,
        "v2_dataset_freeze_sha256": document["v2_dataset"][
            "dataset_freeze_sha256"
        ],
        "effective_train_ids_sha256": document["effective_train"][
            "effective_run_ids_sha256"
        ],
        "extraction_policy_sha256": canonical_sha256(policy),
        "normalizer_sha256": candidate.record["normalizer_sha256"],
        "checkpoint_sha256": candidate.record["checkpoint_sha256"],
        "ensemble_membership": list(document["training"]["seeds"]),
        "architecture": document["architecture"],
        "architecture_sha256": architecture_sha,
        "feature_schema_sha256": document["features"]["schema_sha256"],
        "threshold": document["runtime_decision"]["threshold"],
        "persistence_ms": document["runtime_decision"]["persistence_ms"],
        "hnm_provenance_sha256": sha256_file(hnm_path),
        "candidate_frozen_before_validation": True,
        "v2_validation_evaluated": False,
        "generalization_validation_v2_inference": False,
        "generalization_holdout_guard_count": 0,
    }
    candidate_freeze_path = artifact_path / "candidate_freeze.json"
    _write_json(candidate_freeze_path, candidate_freeze)
    candidate_freeze_sha = sha256_file(candidate_freeze_path)

    v2_path = root / str(document["v2_dataset"]["path"])
    validation_runs, validation_annotations = load_model_v2_runs(
        v2_path,
        data.v2_manifest,
        "V2_VALIDATION",
        candidate_freeze_path=candidate_freeze_path,
    )
    validation_rows = {
        str(row["run_id"]): row
        for row in data.v2_manifest["runs"]
        if row["split"] == "V2_VALIDATION" and bool(row["valid"])
    }
    validation_precursor = {
        run_id: (
            None
            if row["i1_summary"]["first_sample"] is None
            else int(row["i1_summary"]["first_sample"])
        )
        for run_id, row in validation_rows.items()
    }
    v2_replays = replay_hazard_runs(
        validation_runs, candidate.normalizer, candidate.checkpoint_paths
    )
    v2_result = evaluate_model_v2_validation(
        validation_runs,
        v2_replays,
        validation_precursor,
        validation_annotations,
        validation_rows,
        document["validation"]["gates"],
    )
    v2_result_path = artifact_path / "v2_validation_evaluation.json"
    _write_json(v2_result_path, v2_result)

    v1_normalizer = load_hazard_normalizer(
        root
        / "artifacts/runs/20260829_unified_hazard_reflex_system/normalization/gru_history20.json"
    )
    v1_paths = tuple(root / str(row["path"]) for row in document["protected_v1"]["checkpoints"])
    v1_replays = replay_hazard_runs(validation_runs, v1_normalizer, v1_paths)
    v1_result = evaluate_model_v2_validation(
        validation_runs,
        v1_replays,
        validation_precursor,
        validation_annotations,
        validation_rows,
        document["validation"]["gates"],
    )
    v1_result_path = artifact_path / "v1_on_v2_validation.json"
    _write_json(v1_result_path, v1_result)
    comparison = _comparison_metrics(v1_result, v2_result)
    verdict = (
        "MODEL_V2_INTERNAL_VALIDATION_SUPPORTED"
        if v2_result["all_gates_passed"]
        else "MODEL_V2_INTERNAL_VALIDATION_NOT_SUPPORTED"
    )
    result = {
        "training_verdict": "MODEL_V2_DATA_ONLY_TRAINING_COMPLETE",
        "internal_validation_verdict": verdict,
        "candidate_freeze_sha256": candidate_freeze_sha,
        "training_config_sha256": config_sha,
        "training_record_sha256": sha256_file(training_record_path),
        "v2_validation_result_sha256": sha256_file(v2_result_path),
        "v1_on_v2_validation_sha256": sha256_file(v1_result_path),
        "v1_vs_v2": comparison,
        "optimizer_steps": candidate.record["optimizer_steps"],
        "checkpoint_writes": candidate.record["checkpoint_writes"],
        "normalizer_fits": 1,
        "hnm_rounds": HNM_ROUNDS,
        "generalization_validation_v2_inference": False,
        "generalization_holdout_guard_count": 0,
    }
    result_path = artifact_path / "training_result.json"
    _write_json(result_path, result)
    if v2_result["all_gates_passed"]:
        model_freeze = {
            **candidate_freeze,
            "candidate_freeze_sha256": candidate_freeze_sha,
            "v2_validation_result_sha256": sha256_file(v2_result_path),
            "precursor_secondary_definition": "frozen_30_to_50mm_episode_view",
            "internal_validation_verdict": verdict,
            "generalization_candidate": True,
        }
        model_freeze_path = artifact_path / "model_v2_freeze.json"
        _write_json(model_freeze_path, model_freeze)
        result["model_v2_freeze_sha256"] = sha256_file(model_freeze_path)
        _write_json(result_path, result)
    return result


def run_model_v2_extraction_rebalanced_training(
    root: Path,
    config_path: Path,
    *,
    dry_run: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Execute the frozen extraction-only Model V2 intervention."""
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if document["experiment"]["id"] != "MODEL_V2_EXTRACTION_REBALANCED_TRAINING":
        raise ValueError("unsupported extraction-rebalanced training config")
    config_sha = sha256_file(config_path)
    design_path = root / str(document["design"]["path"])
    if sha256_file(design_path) != str(document["design"]["config_sha256"]):
        raise RuntimeError("frozen extraction-rebalance design config changed")
    design = yaml.safe_load(design_path.read_text(encoding="utf-8"))
    if (
        design["dry_run_freeze"]["extraction_rebalance_design_sha256"]
        != document["design"]["extraction_rebalance_design_sha256"]
        or canonical_sha256(model_v2_rebalance_policy())
        != document["design"]["extraction_policy_sha256"]
    ):
        raise RuntimeError("frozen extraction-rebalance identity changed")
    baseline = document["baseline_v2"]
    _verify_protected_records(
        root,
        (
            baseline["candidate_freeze"],
            baseline["normalizer"],
            *baseline["checkpoints"],
        ),
    )
    data = prepare_model_v2_training_data(root, document)
    recipe = _training_recipe(document)
    extraction = audit_model_v2_rebalanced_extraction(
        data.runs,
        data.precursor_samples,
        data.annotations,
        per_category=int(recipe["initial_negative_per_gait_category"]),
        target_contact_cap=int(recipe["target_contact_cap_per_run"]),
        benign_precursor_cap=int(recipe["benign_precursor_cap_per_run"]),
    )
    expected_positive = document["window_extraction"]["expected_fit_positive"]
    expected_fit = {
        "slip": int(expected_positive["slip"]),
        "ordinary_support": int(expected_positive["ordinary_support"]),
        "delayed_support": int(expected_positive["delayed_support"]),
        "support": int(expected_positive["ordinary_support"])
        + int(expected_positive["delayed_support"]),
        "total": int(expected_positive["total"]),
    }
    if (
        not extraction["passed"]
        or extraction["extraction_policy_sha256"]
        != document["design"]["extraction_policy_sha256"]
        or extraction["positive_window_ids_sha256"]
        != document["design"]["proposed_positive_window_ids_sha256"]
        or extraction["negative_window_ids_sha256"]
        != document["design"]["negative_window_ids_sha256"]
        or extraction["masked_window_sha256"]
        != document["design"]["masked_window_sha256"]
        or extraction["fit_positive_counts"] != expected_fit
        or int(extraction["fit_negative_count"])
        != int(document["window_extraction"]["expected_fit_negative"])
        or int(extraction["all_positive_count"])
        != int(document["window_extraction"]["expected_all_positive"])
        or int(extraction["all_negative_count"])
        != int(document["window_extraction"]["expected_all_negative"])
    ):
        raise RuntimeError("pretraining extraction differs from frozen design")
    normalizer_path = root / str(document["normalizer"]["path"])
    if sha256_file(normalizer_path) != str(document["normalizer"]["sha256"]):
        raise RuntimeError("frozen baseline V2 normalizer changed")
    normalizer = load_hazard_normalizer(normalizer_path)
    artifact_path = root / str(document["artifacts"]["path"])
    pretraining_path = artifact_path / "pretraining_audit.json"
    pretraining = {
        "execution_config_sha256": config_sha,
        "design_config_sha256": document["design"]["config_sha256"],
        "extraction_rebalance_design_sha256": document["design"][
            "extraction_rebalance_design_sha256"
        ],
        "input_audit": data.input_audit,
        "effective_train_composition": data.composition,
        "extraction_audit": extraction,
        "normalizer_sha256": sha256_file(normalizer_path),
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "normalizer_fits": 0,
        "hnm_rounds": 0,
        "v2_validation_waveform_opened": False,
        "generalization_validation_v2_inference": False,
        "generalization_holdout_guard_count": 0,
    }
    if dry_run:
        _write_json(pretraining_path, pretraining)
        return {
            "status": "MODEL_V2_EXTRACTION_REBALANCED_PRETRAINING_READY",
            "execution_config_sha256": config_sha,
            "pretraining_audit_sha256": sha256_file(pretraining_path),
            "extraction": extraction,
            "normalizer_fits": 0,
        }
    if not pretraining_path.is_file():
        raise RuntimeError("run frozen extraction dry-run before optimizer step 1")
    if json.loads(pretraining_path.read_text(encoding="utf-8")) != pretraining:
        raise RuntimeError("pretraining audit changed after freeze")
    forbidden_existing = (
        artifact_path / "candidate_freeze.json",
        artifact_path / "candidate_evaluation_freeze.json",
        artifact_path / "v2_validation_evaluation.json",
        artifact_path / "training_result.json",
    )
    if any(path.exists() for path in forbidden_existing) or any(
        (artifact_path / "checkpoints").glob("*.pt")
    ):
        raise RuntimeError("extraction-rebalanced training artifacts already exist")

    fit_positive, monitor_positive = model_v2_rebalanced_positive_plan(
        data.runs, data.precursor_samples, data.annotations
    )
    candidate = train_hazard_candidate(
        root,
        data.runs,
        data.precursor_samples,
        artifact_path,
        recipe,
        data.annotations,
        checkpoint_prefix=str(document["artifacts"]["checkpoint_prefix"]),
        progress=progress,
        normalizer_override=normalizer,
        normalizer_source_path=normalizer_path,
        fit_positive_endpoints=fit_positive,
        monitor_positive_endpoints=monitor_positive,
        extraction_audit_override=extraction,
    )
    if (
        candidate.record["normalizer_fits"] != 0
        or candidate.record["normalizer_sha256"]
        != document["normalizer"]["sha256"]
    ):
        raise RuntimeError("training did not reuse the frozen V2 normalizer")
    training_record = {
        "execution_config_sha256": config_sha,
        "effective_train_ids_sha256": document["effective_train"][
            "effective_run_ids_sha256"
        ],
        "extraction_rebalance_design_sha256": document["design"][
            "extraction_rebalance_design_sha256"
        ],
        "candidate": candidate.record,
        "normalizer_fits": 0,
        "threshold_searches": 0,
        "persistence_searches": 0,
        "architecture_searches": 0,
        "seed_searches": 0,
        "new_simulation_runs": 0,
    }
    training_record_path = artifact_path / "training_record.json"
    _write_json(training_record_path, training_record)

    exposure_rounds = [
        {
            "round": row["round"],
            "available_fit_endpoints": row["positive_exposure_available"],
            "actual_batch_exposure_by_seed": row[
                "positive_batch_exposure_by_seed"
            ],
            "epochs_completed": row["epochs_completed"],
        }
        for row in candidate.record["rounds"]
    ]
    aggregate_exposure: Counter[str] = Counter()
    for row in exposure_rounds:
        for seed_counts in row["actual_batch_exposure_by_seed"].values():
            aggregate_exposure.update(
                {name: int(value) for name, value in seed_counts.items()}
            )
    exposure_provenance = {
        "definition": "each_fit_endpoint_is_seen_once_per_completed_epoch",
        "sampler": "deterministic_shuffled_dataloader_drop_last_false",
        "adaptive_sampling": False,
        "rounds": exposure_rounds,
        "aggregate_batch_exposure": dict(sorted(aggregate_exposure.items())),
    }
    exposure_path = artifact_path / "exposure_provenance.json"
    _write_json(exposure_path, exposure_provenance)

    hnm_provenance = {
        "rounds": [
            {
                "hnm_round": int(row["round"]) + 1,
                **row["hard_negative_mining"],
            }
            for row in candidate.record["rounds"]
            if "hard_negative_mining" in row
        ],
        "source": "effective_train_only",
        "policy_unchanged": True,
        "selected_window_identities_may_differ_with_model_weights": True,
        "forbidden_splits": [
            "V2_VALIDATION",
            "Generalization_VALIDATION",
            "Generalization_HOLDOUT",
            "Unified_HOLDOUT",
        ],
    }
    hnm_path = artifact_path / "hnm_provenance.json"
    _write_json(hnm_path, hnm_provenance)

    architecture_sha = canonical_sha256(document["architecture"])
    if architecture_sha != str(document["baseline_v2"]["architecture_sha256"]):
        raise RuntimeError("Model V2 architecture identity changed")
    candidate_freeze = {
        "candidate_id": document["artifacts"]["candidate_id"],
        "source_commit": document["experiment"]["source_commit"],
        "execution_config_sha256": config_sha,
        "design_config_sha256": document["design"]["config_sha256"],
        "extraction_rebalance_design_sha256": document["design"][
            "extraction_rebalance_design_sha256"
        ],
        "extraction_policy_sha256": extraction["extraction_policy_sha256"],
        "positive_window_ids_sha256": extraction["positive_window_ids_sha256"],
        "negative_window_ids_sha256": extraction["negative_window_ids_sha256"],
        "masked_window_sha256": extraction["masked_window_sha256"],
        "v2_dataset_freeze_sha256": document["v2_dataset"][
            "dataset_freeze_sha256"
        ],
        "effective_train_ids_sha256": document["effective_train"][
            "effective_run_ids_sha256"
        ],
        "normalizer_path": str(normalizer_path.relative_to(root)),
        "normalizer_sha256": candidate.record["normalizer_sha256"],
        "normalizer_fits": 0,
        "checkpoint_sha256": candidate.record["checkpoint_sha256"],
        "ensemble_membership": list(document["training"]["seeds"]),
        "architecture": document["architecture"],
        "architecture_sha256": architecture_sha,
        "feature_schema_sha256": document["features"]["schema_sha256"],
        "threshold": document["runtime_decision"]["threshold"],
        "persistence_ms": document["runtime_decision"]["persistence_ms"],
        "hnm_provenance_sha256": sha256_file(hnm_path),
        "exposure_provenance_sha256": sha256_file(exposure_path),
        "training_record_sha256": sha256_file(training_record_path),
        "candidate_frozen_before_validation": True,
        "v2_validation_evaluated": False,
        "generalization_validation_v2_inference": False,
        "generalization_holdout_guard_count": 0,
    }
    candidate_freeze_path = artifact_path / "candidate_freeze.json"
    _write_json(candidate_freeze_path, candidate_freeze)
    candidate_freeze_sha = sha256_file(candidate_freeze_path)

    v2_path = root / str(document["v2_dataset"]["path"])
    validation_runs, validation_annotations = load_model_v2_runs(
        v2_path,
        data.v2_manifest,
        "V2_VALIDATION",
        candidate_freeze_path=candidate_freeze_path,
    )
    validation_rows = {
        str(row["run_id"]): row
        for row in data.v2_manifest["runs"]
        if row["split"] == "V2_VALIDATION" and bool(row["valid"])
    }
    validation_precursor = {
        run_id: (
            None
            if row["i1_summary"]["first_sample"] is None
            else int(row["i1_summary"]["first_sample"])
        )
        for run_id, row in validation_rows.items()
    }
    gates = document["validation"]["gates"]
    threshold = float(document["runtime_decision"]["threshold"])
    persistence_ms = int(document["runtime_decision"]["persistence_ms"])

    rebalanced_replays = replay_hazard_runs(
        validation_runs, candidate.normalizer, candidate.checkpoint_paths
    )
    rebalanced_result = evaluate_model_v2_validation(
        validation_runs,
        rebalanced_replays,
        validation_precursor,
        validation_annotations,
        validation_rows,
        gates,
        threshold=threshold,
        persistence_ms=persistence_ms,
    )
    rebalanced_path = artifact_path / "v2_validation_evaluation.json"
    _write_json(rebalanced_path, rebalanced_result)

    baseline_paths = tuple(
        root / str(row["path"]) for row in document["baseline_v2"]["checkpoints"]
    )
    baseline_replays = replay_hazard_runs(
        validation_runs, normalizer, baseline_paths
    )
    baseline_result = evaluate_model_v2_validation(
        validation_runs,
        baseline_replays,
        validation_precursor,
        validation_annotations,
        validation_rows,
        gates,
        threshold=threshold,
        persistence_ms=persistence_ms,
    )
    frozen_baseline_result = json.loads(
        (
            root
            / "artifacts/runs/20260901_model_v2_data_only_training/v2_validation_evaluation.json"
        ).read_text(encoding="utf-8")
    )
    if baseline_result != frozen_baseline_result:
        raise RuntimeError("baseline V2 validation replay changed")
    baseline_path = artifact_path / "baseline_v2_on_v2_validation.json"
    _write_json(baseline_path, baseline_result)

    v1_normalizer = load_hazard_normalizer(
        root
        / "artifacts/runs/20260829_unified_hazard_reflex_system/normalization/gru_history20.json"
    )
    v1_paths = tuple(
        root / str(row["path"]) for row in document["protected_v1"]["checkpoints"]
    )
    v1_replays = replay_hazard_runs(validation_runs, v1_normalizer, v1_paths)
    v1_result = evaluate_model_v2_validation(
        validation_runs,
        v1_replays,
        validation_precursor,
        validation_annotations,
        validation_rows,
        gates,
        threshold=threshold,
        persistence_ms=persistence_ms,
    )
    frozen_v1_result = json.loads(
        (
            root
            / "artifacts/runs/20260901_model_v2_data_only_training/v1_on_v2_validation.json"
        ).read_text(encoding="utf-8")
    )
    if v1_result != frozen_v1_result:
        raise RuntimeError("Model V1 validation replay changed")
    v1_path = artifact_path / "v1_on_v2_validation.json"
    _write_json(v1_path, v1_result)

    delayed = {
        "v1": _delayed_support_validation_diagnostics(
            validation_runs,
            v1_replays,
            validation_precursor,
            validation_rows,
            v1_result,
            threshold=threshold,
            persistence_ms=persistence_ms,
        ),
        "baseline_v2": _delayed_support_validation_diagnostics(
            validation_runs,
            baseline_replays,
            validation_precursor,
            validation_rows,
            baseline_result,
            threshold=threshold,
            persistence_ms=persistence_ms,
        ),
        "rebalanced_v2": _delayed_support_validation_diagnostics(
            validation_runs,
            rebalanced_replays,
            validation_precursor,
            validation_rows,
            rebalanced_result,
            threshold=threshold,
            persistence_ms=persistence_ms,
        ),
    }
    delayed_path = artifact_path / "delayed_support_comparison.json"
    _write_json(delayed_path, delayed)
    preservation = _validation_preservation_summary(
        validation_runs,
        rebalanced_replays,
        validation_annotations,
        validation_precursor,
        validation_rows,
        rebalanced_result,
        delayed["rebalanced_v2"],
    )
    preservation_path = artifact_path / "preservation_diagnostics.json"
    _write_json(preservation_path, preservation)
    comparison = _three_model_comparison(
        v1_result, baseline_result, rebalanced_result, delayed
    )
    comparison_path = artifact_path / "three_model_comparison.json"
    _write_json(comparison_path, comparison)

    baseline_delayed = delayed["baseline_v2"]
    rebalanced_delayed = delayed["rebalanced_v2"]
    target_improved = (
        int(rebalanced_delayed["overall"]["correct"])
        > int(baseline_delayed["overall"]["correct"])
        and int(rebalanced_delayed["by_source"]["marble"]["correct"])
        > int(baseline_delayed["by_source"]["marble"]["correct"])
    )
    solved_behavior_retained = (
        float(
            rebalanced_result["side"]["support"]["RIGHT_ONLY"]["recall"]
        )
        >= float(baseline_result["side"]["support"]["RIGHT_ONLY"]["recall"])
        and float(rebalanced_result["primary"]["primary_no_hazard_specificity"])
        >= float(baseline_result["primary"]["primary_no_hazard_specificity"])
        and float(
            rebalanced_result["families"]["STAGED_SAND_BENIGN_CONTROL"][
                "specificity"
            ]
        )
        >= float(
            baseline_result["families"]["STAGED_SAND_BENIGN_CONTROL"][
                "specificity"
            ]
        )
        and float(
            rebalanced_result["families"]["SPEED_STRATIFIED_SAND_BENIGN"][
                "specificity"
            ]
        )
        >= float(
            baseline_result["families"]["SPEED_STRATIFIED_SAND_BENIGN"][
                "specificity"
            ]
        )
    )
    intervention_verdict = (
        "V2_EXTRACTION_REBALANCE_EFFECTIVE"
        if target_improved and solved_behavior_retained
        else "V2_EXTRACTION_REBALANCE_NOT_EFFECTIVE"
    )
    internal_verdict = (
        "MODEL_V2_INTERNAL_VALIDATION_SUPPORTED"
        if rebalanced_result["all_gates_passed"]
        else "MODEL_V2_INTERNAL_VALIDATION_NOT_SUPPORTED"
    )
    if target_improved and not solved_behavior_retained:
        next_milestone = "MODEL_V2_REBALANCE_REGRESSION_AUDIT"
    elif intervention_verdict == "V2_EXTRACTION_REBALANCE_NOT_EFFECTIVE":
        next_milestone = "MODEL_V2_DELAYED_SUPPORT_FAILURE_AUDIT"
    elif rebalanced_result["all_gates_passed"]:
        next_milestone = "MODEL_V2_GENERALIZATION_DEVELOPMENT_EVALUATION"
    else:
        next_milestone = "MODEL_V2_CANDIDATE_READINESS_REVIEW"

    evaluation_freeze = {
        "candidate_id": candidate_freeze["candidate_id"],
        "candidate_freeze_sha256": candidate_freeze_sha,
        "execution_config_sha256": config_sha,
        "normalizer_sha256": candidate_freeze["normalizer_sha256"],
        "checkpoint_sha256": candidate_freeze["checkpoint_sha256"],
        "architecture_sha256": architecture_sha,
        "feature_schema_sha256": document["features"]["schema_sha256"],
        "threshold": threshold,
        "persistence_ms": persistence_ms,
        "hnm_provenance_sha256": sha256_file(hnm_path),
        "v2_validation_result_sha256": sha256_file(rebalanced_path),
        "delayed_support_comparison_sha256": sha256_file(delayed_path),
        "preservation_diagnostics_sha256": sha256_file(preservation_path),
        "three_model_comparison_sha256": sha256_file(comparison_path),
        "intervention_verdict": intervention_verdict,
        "internal_validation_verdict": internal_verdict,
        "generalization_validation_v2_inference": False,
        "generalization_holdout_guard_count": 0,
    }
    evaluation_freeze_path = artifact_path / "candidate_evaluation_freeze.json"
    _write_json(evaluation_freeze_path, evaluation_freeze)
    evaluation_freeze_sha = sha256_file(evaluation_freeze_path)
    result = {
        "training_verdict": "MODEL_V2_EXTRACTION_REBALANCED_TRAINING_COMPLETE",
        "intervention_verdict": intervention_verdict,
        "internal_validation_verdict": internal_verdict,
        "recommended_next_milestone": next_milestone,
        "candidate_freeze_sha256": candidate_freeze_sha,
        "candidate_evaluation_freeze_sha256": evaluation_freeze_sha,
        "execution_config_sha256": config_sha,
        "training_record_sha256": sha256_file(training_record_path),
        "exposure_provenance_sha256": sha256_file(exposure_path),
        "hnm_provenance_sha256": sha256_file(hnm_path),
        "v2_validation_result_sha256": sha256_file(rebalanced_path),
        "baseline_v2_on_v2_validation_sha256": sha256_file(baseline_path),
        "v1_on_v2_validation_sha256": sha256_file(v1_path),
        "target_improved": target_improved,
        "solved_behavior_retained": solved_behavior_retained,
        "optimizer_steps": candidate.record["optimizer_steps"],
        "checkpoint_writes": candidate.record["checkpoint_writes"],
        "normalizer_fits": 0,
        "hnm_rounds": HNM_ROUNDS,
        "threshold_searches": 0,
        "persistence_searches": 0,
        "architecture_searches": 0,
        "seed_searches": 0,
        "new_simulation_runs": 0,
        "v2_validation_optimizer_leakage": 0,
        "generalization_validation_training_leakage": 0,
        "holdout_training_leakage": 0,
        "generalization_validation_v2_inference": False,
        "unified_holdout_waveform_reopened": False,
        "generalization_holdout_waveform_opened": False,
        "generalization_holdout_inference": False,
        "generalization_holdout_guard_count": 0,
    }
    result_path = artifact_path / "training_result.json"
    _write_json(result_path, result)
    if rebalanced_result["all_gates_passed"]:
        model_freeze = {
            **evaluation_freeze,
            "candidate_evaluation_freeze_sha256": evaluation_freeze_sha,
            "generalization_candidate": True,
        }
        model_freeze_path = artifact_path / "model_v2_freeze.json"
        _write_json(model_freeze_path, model_freeze)
        result["model_v2_freeze_sha256"] = sha256_file(model_freeze_path)
        _write_json(result_path, result)
    return result
