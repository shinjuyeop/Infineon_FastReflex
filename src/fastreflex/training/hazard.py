"""TRAIN-only construction and HNM for the supported Unified Hazard GRU."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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
from fastreflex.features import (
    extract_hazard_features,
    feature_schema_hash,
    hazard_feature_schema,
)
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
MODEL_V2_ANCHOR_REFINED_MONITOR_SHA256 = (
    "39d30234f674446f305b1b51d446977ba301e6db1e0591ac14dbe7172cbb1bf5"
)


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


def model_v2_anchor_refinement_candidates() -> list[dict[str, object]]:
    """Return the five TRAIN-only candidates frozen before separability analysis."""
    midpoint = {
        "numerator": 1,
        "denominator": 2,
        "rounding": "floor",
        "offsets_ms": [-2, -1, 0, 1, 2],
    }
    return [
        {
            "id": "CURRENT_RULE",
            "i1_offsets_ms": [0, 1, 2, 3, 4],
            "midpoint": dict(midpoint),
            "late": {"anchor": "support", "offsets_ms": [0, 1, 2, 3, 4]},
            "cap_per_run": 15,
        },
        {
            "id": "DROP_SUPPORT_LOCAL",
            "i1_offsets_ms": [0, 1, 2, 3, 4],
            "midpoint": dict(midpoint),
            "late": None,
            "cap_per_run": 10,
        },
        {
            "id": "SINGLE_SUPPORT_ENDPOINT",
            "i1_offsets_ms": [0, 1, 2, 3, 4],
            "midpoint": dict(midpoint),
            "late": {"anchor": "support", "offsets_ms": [0]},
            "cap_per_run": 11,
        },
        {
            "id": "SPARSE_SUPPORT_LOCAL",
            "i1_offsets_ms": [0, 1, 2, 3, 4],
            "midpoint": dict(midpoint),
            "late": {"anchor": "support", "offsets_ms": [0, 4]},
            "cap_per_run": 12,
        },
        {
            "id": "LATE_PRE_SUPPORT_INTERIOR",
            "i1_offsets_ms": [0, 1, 2, 3, 4],
            "midpoint": dict(midpoint),
            "late": {
                "anchor": "i1_plus_floor_fraction_of_i1_to_support",
                "numerator": 3,
                "denominator": 4,
                "offsets_ms": [0],
            },
            "cap_per_run": 11,
        },
    ]


def model_v2_anchor_refined_policy() -> dict[str, object]:
    """Return the exact selected extraction and fixed-monitor contract."""
    selected = next(
        row
        for row in model_v2_anchor_refinement_candidates()
        if row["id"] == "LATE_PRE_SUPPORT_INTERIOR"
    )
    return {
        "delayed_support_fit_policy": selected,
        "slip_and_ordinary_support": "preserve_baseline_exact",
        "negative_and_masks": "preserve_baseline_exact",
        "monitor_endpoint_sha256": MODEL_V2_ANCHOR_REFINED_MONITOR_SHA256,
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


def _candidate_delayed_support_endpoints(
    run: HazardRun,
    precursor: int | None,
    annotation: HazardRunAnnotations,
    policy: Mapping[str, object],
    history_ms: int = HISTORY_MS,
) -> np.ndarray:
    """Resolve one predeclared delayed-Support candidate without offset search."""
    if annotation.scenario_family != MODEL_V2_DELAYED_SUPPORT_FAMILY:
        return np.empty(0, dtype=np.int64)
    support = support_event_sample(run)
    if precursor is None or support is None or int(precursor) > int(support):
        raise ValueError("eligible delayed Support requires ordered I1 and Support")
    i1 = int(precursor)
    support = int(support)
    midpoint_rule = policy["midpoint"]
    midpoint = i1 + (
        (support - i1) * int(midpoint_rule["numerator"])
        // int(midpoint_rule["denominator"])
    )
    selected = {
        *(i1 + int(offset) for offset in policy["i1_offsets_ms"]),
        *(
            midpoint + int(offset)
            for offset in midpoint_rule["offsets_ms"]
        ),
    }
    late = policy["late"]
    if late is not None:
        if late["anchor"] == "support":
            anchor = support
        elif late["anchor"] == "i1_plus_floor_fraction_of_i1_to_support":
            anchor = i1 + (
                (support - i1) * int(late["numerator"])
                // int(late["denominator"])
            )
        else:
            raise ValueError(f"unsupported delayed-Support anchor: {late['anchor']}")
        selected.update(anchor + int(offset) for offset in late["offsets_ms"])
    values = np.asarray(sorted(selected), dtype=np.int64)
    stop = run.censor_sample
    if run.fall_sample_diagnostic is not None:
        stop = min(stop, int(run.fall_sample_diagnostic))
    if (
        len(values) != int(policy["cap_per_run"])
        or np.any(values < i1)
        or np.any(values < history_ms - 1)
        or np.any(values >= stop)
    ):
        raise RuntimeError(f"invalid delayed-Support candidate: {policy['id']}")
    return values


def anchor_refined_delayed_support_positive_endpoints(
    run: HazardRun,
    precursor: int | None,
    annotation: HazardRunAnnotations,
    history_ms: int = HISTORY_MS,
) -> np.ndarray:
    """Resolve the frozen I1, midpoint, and 3/4-interval positive anchors."""
    selected = next(
        row
        for row in model_v2_anchor_refinement_candidates()
        if row["id"] == "LATE_PRE_SUPPORT_INTERIOR"
    )
    return _candidate_delayed_support_endpoints(
        run, precursor, annotation, selected, history_ms
    )


def model_v2_anchor_refined_positive_plan(
    runs: Mapping[str, HazardRun],
    precursor_samples: Mapping[str, int | None],
    annotations: Mapping[str, HazardRunAnnotations],
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    """Assign refined fit anchors and the predeclared candidate-invariant monitor."""
    fit_ids, monitor_ids = _train_monitor_partition(
        runs, sorted(runs), precursor_samples
    )
    fit_set = set(fit_ids)
    monitor_set = set(monitor_ids)
    baseline_monitor: dict[str, tuple[int, ...]] = {}
    for run_id in sorted(runs):
        if run_id in monitor_set:
            baseline_monitor[run_id] = tuple(
                int(value)
                for value in unified_positive_endpoints(
                    runs[run_id], precursor_samples.get(run_id), cap=20
                )
            )

    candidate_pairs: set[tuple[str, int]] = set()
    candidates = model_v2_anchor_refinement_candidates()
    for run_id in sorted(runs):
        annotation = annotations[run_id]
        if annotation.scenario_family != MODEL_V2_DELAYED_SUPPORT_FAMILY:
            continue
        for policy in candidates:
            endpoints = _candidate_delayed_support_endpoints(
                runs[run_id],
                precursor_samples.get(run_id),
                annotation,
                policy,
            )
            candidate_pairs.update((run_id, int(value)) for value in endpoints)

    monitor = {
        run_id: tuple(
            endpoint
            for endpoint in endpoints
            if (run_id, int(endpoint)) not in candidate_pairs
        )
        for run_id, endpoints in baseline_monitor.items()
    }
    fit: dict[str, tuple[int, ...]] = {}
    for run_id in sorted(runs):
        run = runs[run_id]
        annotation = annotations[run_id]
        precursor = precursor_samples.get(run_id)
        baseline = tuple(
            int(value) for value in unified_positive_endpoints(run, precursor, cap=20)
        )
        if annotation.scenario_family == MODEL_V2_DELAYED_SUPPORT_FAMILY:
            refined = tuple(
                int(value)
                for value in anchor_refined_delayed_support_positive_endpoints(
                    run, precursor, annotation
                )
            )
            legacy_slip = tuple(
                endpoint
                for endpoint in baseline
                if _positive_role(run, endpoint) == "slip"
            )
            fit[run_id] = tuple(
                sorted({*refined, *(legacy_slip if run_id in fit_set else ())})
            )
        elif run_id in fit_set:
            fit[run_id] = baseline
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


def _audit_model_v2_extraction(
    runs: Mapping[str, HazardRun],
    precursor_samples: Mapping[str, int | None],
    annotations: Mapping[str, HazardRunAnnotations],
    *,
    positive_plan: Callable[..., tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]],
    delayed_endpoint_selector: Callable[..., np.ndarray],
    extraction_policy_sha256: str,
    expected_delayed_endpoints_per_run: int,
    per_category: int = 12,
    target_contact_cap: int = 12,
    benign_precursor_cap: int = 12,
) -> dict[str, object]:
    """Reproduce one frozen Model V2 extraction before optimizer access."""
    fit_ids, monitor_ids = _train_monitor_partition(
        runs, sorted(runs), precursor_samples
    )
    partitions = {run_id: "fit" for run_id in fit_ids} | {
        run_id: "monitor" for run_id in monitor_ids
    }
    fit_positive, monitor_positive = positive_plan(
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
            delayed = delayed_endpoint_selector(
                run, precursor, annotation
            )
            if len(delayed) != expected_delayed_endpoints_per_run:
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
    monitor_negative_rows = [
        row for row in negative_rows if row["partition"] == "monitor"
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
    monitor_delayed_sources: Counter[str] = Counter()
    monitor_roles: Counter[str] = Counter()
    for row in monitor_positive_rows:
        role = str(row["role"])
        monitor_roles[role] += 1
        run_id = str(row["run_id"])
        if (
            role == "support"
            and annotations[run_id].scenario_family
            == MODEL_V2_DELAYED_SUPPORT_FAMILY
        ):
            monitor_delayed_sources[runs[run_id].source_terrain] += 1
    monitor_identity = sorted(
        [
            f"positive:{row['run_id']}:{row['endpoint_sample']}:{row['role']}"
            for row in monitor_positive_rows
        ]
        + [
            f"negative:{row['run_id']}:{row['endpoint_sample']}:{row['role']}"
            for row in monitor_negative_rows
        ]
    )
    fit_endpoint_ids = {
        (str(row["run_id"]), int(row["endpoint_sample"]))
        for row in (*fit_positive_rows, *(row for row in negative_rows if row["partition"] == "fit"))
    }
    monitor_endpoint_ids = {
        (str(row["run_id"]), int(row["endpoint_sample"]))
        for row in (*monitor_positive_rows, *monitor_negative_rows)
    }
    return {
        "effective_train_run_count": len(runs),
        "extraction_policy_sha256": extraction_policy_sha256,
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
        "monitor_positive_sha256": _endpoint_identity_hash(
            monitor_positive_rows
        ),
        "monitor_positive_counts": {
            "slip": int(monitor_roles["slip"]),
            "ordinary_support": int(monitor_roles["support"])
            - int(sum(monitor_delayed_sources.values())),
            "delayed_support_concrete": int(
                monitor_delayed_sources["concrete"]
            ),
            "delayed_support_marble": int(monitor_delayed_sources["marble"]),
            "total": len(monitor_positive_rows),
        },
        "fit_negative_count": fit_negative,
        "monitor_negative_count": len(monitor_negative_rows),
        "monitor_endpoint_sha256": canonical_sha256(monitor_identity),
        "fit_monitor_endpoint_overlap": len(
            fit_endpoint_ids & monitor_endpoint_ids
        ),
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


def audit_model_v2_rebalanced_extraction(
    runs: Mapping[str, HazardRun],
    precursor_samples: Mapping[str, int | None],
    annotations: Mapping[str, HazardRunAnnotations],
    *,
    per_category: int = 12,
    target_contact_cap: int = 12,
    benign_precursor_cap: int = 12,
) -> dict[str, object]:
    """Reproduce the frozen dense delayed-Support extraction."""
    return _audit_model_v2_extraction(
        runs,
        precursor_samples,
        annotations,
        positive_plan=model_v2_rebalanced_positive_plan,
        delayed_endpoint_selector=delayed_support_positive_endpoints,
        extraction_policy_sha256=canonical_sha256(model_v2_rebalance_policy()),
        expected_delayed_endpoints_per_run=15,
        per_category=per_category,
        target_contact_cap=target_contact_cap,
        benign_precursor_cap=benign_precursor_cap,
    )


def audit_model_v2_anchor_refined_extraction(
    runs: Mapping[str, HazardRun],
    precursor_samples: Mapping[str, int | None],
    annotations: Mapping[str, HazardRunAnnotations],
    *,
    per_category: int = 12,
    target_contact_cap: int = 12,
    benign_precursor_cap: int = 12,
) -> dict[str, object]:
    """Reproduce the frozen late-interior extraction and fixed monitor."""
    return _audit_model_v2_extraction(
        runs,
        precursor_samples,
        annotations,
        positive_plan=model_v2_anchor_refined_positive_plan,
        delayed_endpoint_selector=anchor_refined_delayed_support_positive_endpoints,
        extraction_policy_sha256=canonical_sha256(model_v2_anchor_refined_policy()),
        expected_delayed_endpoints_per_run=11,
        per_category=per_category,
        target_contact_cap=target_contact_cap,
        benign_precursor_cap=benign_precursor_cap,
    )


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
        "selected_endpoint_sha256": canonical_sha256(
            [
                f"{run_id}:{endpoint}"
                for run_id, endpoints in sorted(result.items())
                for endpoint in endpoints
            ]
        ),
        "runs_scored": len(runs),
        "mined_windows": sum(len(value) for value in result.values()),
        "runs_contributing": contributing_runs,
        "selected_by_family": dict(sorted(family_counts.items())),
        "selected_by_source": dict(sorted(source_counts.items())),
        "selected_by_speed": dict(sorted(speed_counts.items())),
        "selected_by_run": {
            run_id: list(endpoints)
            for run_id, endpoints in sorted(result.items())
            if endpoints
        },
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
    fixed_monitor_endpoint_set: bool = False,
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
            extra_negative_endpoints=(
                {} if fixed_monitor_endpoint_set else accumulated
            ),
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
        negative_count = int(fit_windows.selected_by_class[0])
        positive_count = int(fit_windows.selected_by_class[1])
        total_count = negative_count + positive_count
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
            "class_weights": {
                "positive": total_count / (2.0 * positive_count),
                "negative": total_count / (2.0 * negative_count),
                "positive_to_negative_ratio": negative_count / positive_count,
            },
            "best_epochs": epochs,
            "epochs_completed": epochs_completed,
            "optimizer_steps": optimizer_steps,
            "final_train_loss": final_train_losses,
            "best_validation_cross_entropy": best_validation_losses,
            "positive_exposure_available": exposure,
            "fit_window_exposure_by_dataset": dict(
                sorted(
                    Counter(
                        annotations[str(run_id)].dataset_id
                        for run_id in fit_windows.run_ids
                    ).items()
                )
            ),
            "fit_window_exposure_by_design_role": dict(
                sorted(
                    Counter(
                        runs[str(run_id)].design_role for run_id in fit_windows.run_ids
                    ).items()
                )
            ),
            "fit_window_exposure_by_scenario_family": dict(
                sorted(
                    Counter(
                        annotations[str(run_id)].scenario_family
                        for run_id in fit_windows.run_ids
                    ).items()
                )
            ),
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
            "fixed_monitor_endpoint_set": fixed_monitor_endpoint_set,
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
        late_interior = int(i1) + 3 * (support - int(i1)) // 4
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
                "late_interior_sample": late_interior,
                "support_sample": support,
                "first_threshold_crossing": first_crossing,
                "first_reflex": first_reflex,
                "probability_at_i1": _probability_at_sample(replay, int(i1)),
                "probability_at_late_interior": _probability_at_sample(
                    replay, late_interior
                ),
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


def _speed_sand_validation_diagnostics(
    runs: Mapping[str, HazardRun],
    replays: Mapping[str, object],
    seed_replays: Mapping[str, Mapping[str, object]],
    manifest_rows: Mapping[str, Mapping[str, object]],
    validation_result: Mapping[str, object],
    *,
    threshold: float,
    persistence_ms: int,
) -> dict[str, object]:
    """Record every frozen Speed-Sand control without selecting on its result."""
    primary_rows = {
        str(row["run_id"]): row for row in validation_result["primary"]["rows"]
    }
    rows: list[dict[str, object]] = []
    for run_id in sorted(runs):
        manifest_row = manifest_rows[run_id]
        if manifest_row["scenario_family"] != "SPEED_STRATIFIED_SAND_BENIGN":
            continue
        run = runs[run_id]
        replay = replays[run_id]
        primary = primary_rows[run_id]
        crossings = replay.endpoints[replay.probabilities >= threshold]
        onsets = reflex_onset_samples(
            replay, threshold=threshold, persistence_ms=persistence_ms
        )
        rows.append(
            {
                "run_id": run_id,
                "source": run.source_terrain,
                "speed_mps": float(manifest_row["nominal_speed_mps"]),
                "topology": run.sink_pattern,
                "maximum_probability": float(np.max(replay.probabilities)),
                "first_threshold_crossing": (
                    None if not len(crossings) else int(crossings[0])
                ),
                "first_reflex": None if not len(onsets) else int(onsets[0]),
                "seed_maximum_probability": {
                    seed: float(np.max(values[run_id].probabilities))
                    for seed, values in sorted(seed_replays.items())
                },
                "specific": not bool(primary["system_false_positive"]),
                "physical_benign_confirmed": (
                    primary["physical_label"] == "NO_HAZARD"
                    and slip_event_sample(run) is None
                    and support_event_sample(run) is None
                    and run.fall_sample_diagnostic is None
                ),
                "peak_support_spread_m": float(np.max(run.support_spread_m)),
                "peak_support_displacement_m": float(
                    np.max(run.support_max_displacement_m)
                ),
                "peak_drift_m": float(np.max(run.drift_m)),
            }
        )
    by_source = {
        source: {
            "specific": sum(
                bool(row["specific"])
                for row in rows
                if row["source"] == source
            ),
            "runs": sum(row["source"] == source for row in rows),
        }
        for source in ("concrete", "marble")
    }
    by_speed = {
        f"{speed:.2f}": {
            "specific": sum(
                bool(row["specific"])
                for row in rows
                if float(row["speed_mps"]) == speed
            ),
            "runs": sum(float(row["speed_mps"]) == speed for row in rows),
        }
        for speed in (0.20, 0.25, 0.30)
    }
    return {
        "rows": rows,
        "overall": {
            "specific": sum(bool(row["specific"]) for row in rows),
            "runs": len(rows),
        },
        "by_source": by_source,
        "by_speed": by_speed,
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


def _four_model_comparison(
    results: Mapping[str, Mapping[str, object]],
    delayed: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Compare all four immutable candidates on identical V2_VALIDATION."""
    def metrics(name: str) -> dict[str, object]:
        result = results[name]
        primary = result["primary"]
        delayed_result = delayed[name]
        return {
            "overall_hazard": primary["overall_hazard_recall"],
            "slip": primary["slip_hazard_recall"],
            "support": primary["support_hazard_recall"],
            "confirmed_specificity": primary["primary_no_hazard_specificity"],
            "premature": primary["system_premature_run_rate"],
            "delayed_support": delayed_result["overall"]["recall"],
            "marble_delayed_support": delayed_result["by_source"]["marble"][
                "recall"
            ],
            "right_only_support": result["side"]["support"]["RIGHT_ONLY"][
                "recall"
            ],
            "staged_sand_specificity": result["families"][
                "STAGED_SAND_BENIGN_CONTROL"
            ]["specificity"],
            "speed_sand_specificity": result["families"][
                "SPEED_STRATIFIED_SAND_BENIGN"
            ]["specificity"],
            "ice_benign_specificity": result["families"]["ICE_BENIGN_CONTROL"][
                "specificity"
            ],
            "hard_normal_specificity": result["families"][
                "HARD_GROUND_NORMAL_SPEED_MATRIX"
            ]["specificity"],
            "slip_0.20": result["speed"]["0.20"]["slip_recall"],
            "right_only_slip": result["side"]["slip"]["RIGHT_ONLY"]["recall"],
        }

    return {name: metrics(name) for name in results}


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


def _load_factor_conditioned_runs(
    root: Path,
    document: Mapping[str, object],
    split: str,
    *,
    candidate_freeze_path: Path | None = None,
    validation_authorization_path: Path | None = None,
) -> tuple[
    dict[str, HazardRun],
    dict[str, HazardRunAnnotations],
    dict[str, Mapping[str, object]],
    dict[str, int | None],
]:
    """Open only the authorized fresh split and adapt it to canonical Hazard data."""
    from fastreflex.dataset.sand_factor_conditioned import (
        verify_factor_conditioned_dataset,
    )
    from fastreflex.evaluation.sand import hazard_run_from_discovery

    dataset = document["factor_dataset"]
    dataset_path = root / str(dataset["path"])
    verification = verify_factor_conditioned_dataset(dataset_path)
    if not verification["passed"]:
        raise RuntimeError("factor-conditioned dataset integrity failed")
    manifest_path = dataset_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_id = str(dataset["dataset_id"])
    if (
        manifest["dataset_id"] != dataset_id
        or manifest["intervention_config_sha256"]
        != str(
            dataset.get(
                "generation_config_sha256",
                sha256_file(root / str(document["experiment"]["config_path"])),
            )
        )
    ):
        raise RuntimeError("factor-conditioned dataset provenance changed")
    if split == "FACTOR_VALIDATION":
        if candidate_freeze_path is None or not candidate_freeze_path.is_file():
            raise RuntimeError("FACTOR_VALIDATION is sealed until candidate freeze")
        freeze = json.loads(candidate_freeze_path.read_text(encoding="utf-8"))
        if not (
            freeze.get("candidate_frozen_before_factor_validation")
            and freeze.get("factor_validation_evaluated") is False
        ):
            raise RuntimeError("candidate freeze does not authorize factor validation")
        if (
            validation_authorization_path is None
            or not validation_authorization_path.is_file()
        ):
            raise RuntimeError("FACTOR_VALIDATION requires one-shot authorization")
        authorization = json.loads(
            validation_authorization_path.read_text(encoding="utf-8")
        )
        if not (
            authorization.get("authorized") is True
            and authorization.get("open_count_before") == 0
            and authorization.get("open_count_after") == 1
            and authorization.get("candidate_freeze_sha256")
            == sha256_file(candidate_freeze_path)
            and authorization.get("validation_seal_sha256")
            == str(dataset["validation_seal_sha256"])
        ):
            raise RuntimeError("FACTOR_VALIDATION authorization is invalid")
    elif split != "FACTOR_TRAIN":
        raise ValueError(f"unsupported factor split: {split}")
    all_rows = {
        str(row["run_id"]): row
        for row in manifest["runs"]
        if row["split"] == split
    }
    selected_rows = {
        run_id: row
        for run_id, row in all_rows.items()
        if (
            bool(row["training_eligible"])
            if split == "FACTOR_TRAIN"
            else bool(row["validation_eligible"])
            or (
                bool(row["valid"])
                and row["objective_physical_outcome"] in {"SLIP", "DUAL_HAZARD"}
            )
        )
    }
    expected = dataset.get("expected", {})
    expected_all = expected.get(
        "factor_train" if split == "FACTOR_TRAIN" else "factor_validation"
    )
    expected_selected = expected.get(
        "factor_train_eligible"
        if split == "FACTOR_TRAIN"
        else "payload_records_expected",
        document.get("validation_protocol", {}).get("payload_records_expected")
        if split == "FACTOR_VALIDATION"
        else None,
    )
    if expected_all is not None and len(all_rows) != int(expected_all):
        raise RuntimeError(f"factor-conditioned {split} identity changed")
    if expected_selected is not None and len(selected_rows) != int(expected_selected):
        raise RuntimeError(f"factor-conditioned {split} eligibility changed")
    runs: dict[str, HazardRun] = {}
    annotations: dict[str, HazardRunAnnotations] = {}
    precursors: dict[str, int | None] = {}
    for run_id, row in sorted(selected_rows.items()):
        path = dataset_path / str(row["file"])
        if sha256_file(path) != str(row["file_sha256"]):
            raise RuntimeError(f"factor-conditioned run changed: {run_id}")
        with np.load(path, allow_pickle=False) as source:
            payload = {name: np.asarray(source[name]) for name in source.files}
        adapted = hazard_run_from_discovery(row, payload)
        runs[run_id] = replace(
            adapted, split="train" if split == "FACTOR_TRAIN" else "validation"
        )
        annotations[run_id] = HazardRunAnnotations(
            dataset_id=dataset_id,
            scenario_family=str(row["scenario_family"]),
            nominal_speed_mps=float(row["nominal_speed_mps"]),
            actual_side=str(row["support_event_summary"]["side"]),
            target_contact=np.asarray(payload["target_terrain_contact"], dtype=bool),
            established_slip_active=np.asarray(payload["established_slip"], dtype=bool),
            i1_active=np.asarray(payload["i1_active"], dtype=bool),
            ice_precursor_candidate=np.asarray(
                payload["ice_precursor_candidate"], dtype=bool
            ),
            ice_precursor_future_outcome_code=np.asarray(
                payload["ice_precursor_future_outcome_code"], dtype=np.int8
            ),
            ice_precursor_censored=np.asarray(
                payload["ice_precursor_censored"], dtype=bool
            ),
        )
        value = row["i1_summary"]["first_sample"]
        precursors[run_id] = None if value is None else int(value)
    return runs, annotations, all_rows, precursors


def _frozen_normalizer_diagnostic(
    runs: Mapping[str, HazardRun], normalizer: Normalizer
) -> dict[str, object]:
    samples: list[np.ndarray] = []
    for run_id in sorted(runs):
        run = runs[run_id]
        features = extract_hazard_features(run.features["PELVIS_IMU6"])
        eligible = np.linspace(
            0,
            run.censor_sample - 1,
            min(1000, run.censor_sample),
            dtype=np.int64,
        )
        samples.append(normalizer.transform(features[eligible]).astype(np.float64))
    values = np.concatenate(samples)
    absolute = np.abs(values)
    return {
        "run_count": len(runs),
        "sample_count": int(values.shape[0]),
        "feature_dimension": int(values.shape[1]),
        "all_finite": bool(np.all(np.isfinite(values))),
        "absolute_z": {
            "p95": float(np.percentile(absolute, 95)),
            "p99": float(np.percentile(absolute, 99)),
            "p99_9": float(np.percentile(absolute, 99.9)),
            "maximum": float(np.max(absolute)),
        },
        "fraction_absolute_z_gt_20": float(np.mean(absolute > 20.0)),
        "fraction_absolute_z_gt_50": float(np.mean(absolute > 50.0)),
    }


def _factor_validation_comparison(
    reference: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, object]:
    metric_paths = {
        "strict_sand_specificity": ("sand", "overall", "specificity"),
        "mild_sand_specificity": ("sand", "mild", "specificity"),
        "moderate_sand_specificity": ("sand", "moderate", "specificity"),
        "false_reflex_count": ("sand", "overall", "false_reflex"),
        "adverse_margin_rate": ("sand", "overall", "adverse_rate"),
        "median_max_probability": ("sand", "overall", "max_probability", "median"),
        "p95_max_probability": ("sand", "overall", "max_probability", "p95"),
        "support_recall": ("support", "overall", "recall"),
        "ordinary_support_recall": ("support", "ordinary", "recall"),
        "delayed_support_recall": ("support", "delayed", "recall"),
        "right_support_recall": ("support", "right", "recall"),
    }

    def get(value: Mapping[str, object], path: Sequence[str]) -> float:
        current: object = value
        for key in path:
            current = current[key]  # type: ignore[index]
        return float(current)

    metrics = {
        name: {
            "reference_v2": get(reference, path),
            "factor_candidate": get(candidate, path),
            "delta": get(candidate, path) - get(reference, path),
        }
        for name, path in metric_paths.items()
    }
    factor_names = (
        "transition_left",
        "transition_right",
        "right_single_precontact",
        "left_single_precontact",
        "concrete",
        "marble",
        "speed_0.20",
        "speed_0.25",
        "speed_0.30",
        "mild",
        "moderate",
        "adverse_direction_manifold",
        "comparison_direction_manifold",
        "concrete_025_exception",
        "transition_left_right_single",
        "transition_right_left_single",
    )
    factors = {}
    for name in factor_names:
        ref = reference["sand"]["factors"][name]  # type: ignore[index]
        new = candidate["sand"]["factors"][name]  # type: ignore[index]
        factors[name] = {
            "runs": int(new["runs"]),
            "reference_adverse": int(ref["adverse"]),
            "candidate_adverse": int(new["adverse"]),
            "reference_false_reflex": int(ref["false_reflex"]),
            "candidate_false_reflex": int(new["false_reflex"]),
            "adverse_rate_delta": float(new["adverse_rate"])
            - float(ref["adverse_rate"]),
            "improved": int(new["adverse"]) < int(ref["adverse"])
            and int(new["false_reflex"]) <= int(ref["false_reflex"]),
        }
    source_speed = {}
    for name, ref in reference["sand"]["source_speed"].items():  # type: ignore[union-attr]
        new = candidate["sand"]["source_speed"][name]  # type: ignore[index]
        source_speed[name] = {
            "runs": int(new["runs"]),
            "reference_specific": int(ref["specific"]),
            "candidate_specific": int(new["specific"]),
            "reference_false_reflex": int(ref["false_reflex"]),
            "candidate_false_reflex": int(new["false_reflex"]),
            "reference_adverse": int(ref["adverse"]),
            "candidate_adverse": int(new["adverse"]),
            "reference_median_probability": ref["max_probability"]["median"],
            "candidate_median_probability": new["max_probability"]["median"],
            "improved": int(new["adverse"]) < int(ref["adverse"])
            and int(new["false_reflex"]) <= int(ref["false_reflex"]),
        }
    return {"metrics": metrics, "factors": factors, "source_speed": source_speed}


def _validate_factor_conditioned_training_contract(
    document: Mapping[str, object],
) -> None:
    """Reject any second intervention or evidence-boundary drift pre-training."""
    expected_sources = ["Unified_TRAIN", "V2_TRAIN", "FACTOR_TRAIN"]
    if document["training_protocol"]["data_sources"] != expected_sources:  # type: ignore[index]
        raise RuntimeError("factor-conditioned training-source whitelist changed")
    if document["training_source_contract"]["authorized"] != expected_sources:  # type: ignore[index]
        raise RuntimeError("authorized factor-conditioned sources changed")
    if document["training"]["seeds"] != [20260828, 20260829, 20260830]:  # type: ignore[index]
        raise RuntimeError("factor-conditioned seed family changed")
    runtime = document["runtime_decision"]
    if runtime != {  # type: ignore[comparison-overlap]
        "ensemble": "mean_probability_all_three_predeclared_seeds",
        "threshold": 0.99,
        "persistence_ms": 5,
    }:
        raise RuntimeError("factor-conditioned runtime decision changed")
    hnm = document["hnm"]
    if not (  # type: ignore[index]
        hnm["enabled"]
        and hnm["rounds"] == HNM_ROUNDS
        and hnm["source"] == "effective_TRAIN_only"
        and hnm["validation_access"] == "prohibited"
    ):
        raise RuntimeError("factor-conditioned HNM policy changed")
    architecture = document["architecture"]
    if architecture != {  # type: ignore[comparison-overlap]
        "model_family": "gru",
        "input": "pelvis_imu6",
        "input_shape": [20, 80],
        "hidden_size": 32,
        "layers": 1,
        "bidirectional": False,
        "dropout": 0.0,
        "output_classes": ["NORMAL", "HAZARD_REFLEX_REQUIRED"],
        "parameters": 11010,
    }:
        raise RuntimeError("factor-conditioned architecture changed")
    if (
        document["features"]["schema_sha256"] != feature_schema_hash()  # type: ignore[index]
        or document["normalizer"]["policy"] != "reuse_frozen_v2_no_refit"  # type: ignore[index]
        or not document["legacy_regression"]["enabled"]  # type: ignore[index]
    ):
        raise RuntimeError("feature, normalizer, or regression contract changed")


def run_factor_conditioned_data_intervention(
    root: Path,
    config_path: Path,
    *,
    dry_run: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Train one data-only GRU20 family, then consume only development evidence."""
    from fastreflex.dataset.sand_factor_conditioned import (
        validate_factor_conditioned_design,
        verify_factor_conditioned_dataset,
    )
    from fastreflex.evaluation.sand_factor_conditioned import (
        evaluate_factor_conditioned_validation,
    )

    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_id = document["experiment"]["id"]
    supported_ids = {
        "SAND_FACTOR_CONDITIONED_DATA_INTERVENTION",
        "SAND_FACTOR_CONDITIONED_MODEL_TRAINING",
    }
    if experiment_id not in supported_ids:
        raise ValueError("unsupported factor-conditioned training config")
    config_sha = sha256_file(config_path)
    if str(document["experiment"]["config_path"]) != str(config_path.relative_to(root)):
        raise RuntimeError("factor-conditioned config path changed")
    if experiment_id == "SAND_FACTOR_CONDITIONED_DATA_INTERVENTION":
        validate_factor_conditioned_design(root, document)
    else:
        _validate_factor_conditioned_training_contract(document)
    dataset_path = root / str(document["factor_dataset"]["path"])
    dataset_verification = verify_factor_conditioned_dataset(dataset_path)
    if not dataset_verification["passed"]:
        raise RuntimeError("factor-conditioned dataset failed freeze verification")
    dataset_freeze_path = dataset_path / "dataset_freeze.json"
    dataset_freeze = json.loads(dataset_freeze_path.read_text(encoding="utf-8"))
    factor_dataset = document["factor_dataset"]
    expected_verdict = str(
        factor_dataset.get(
            "generation_verdict", "FACTOR_CONDITIONED_DATASET_GENERATION_READY"
        )
    )
    if dataset_freeze["generation_verdict"] != expected_verdict:
        raise RuntimeError("physical-generation gates failed; training is prohibited")
    protected_factor_hashes = {
        "dataset_freeze_file_sha256": sha256_file(dataset_freeze_path),
        "dataset_freeze_semantic_sha256": dataset_freeze["FACTOR_DATASET_FREEZE_SHA"],
        "manifest_sha256": sha256_file(dataset_path / "manifest.json"),
        "physical_audit_sha256": sha256_file(dataset_path / "physical_audit.json"),
        "validation_seal_sha256": sha256_file(dataset_path / "validation_seal.json"),
        "npz_aggregate_sha256": dataset_freeze["FACTOR_NPZ_AGGREGATE_SHA"],
        "factor_train_split_sha256": dataset_freeze["FACTOR_TRAIN_SPLIT_SHA"],
        "factor_validation_split_sha256": dataset_freeze[
            "FACTOR_VALIDATION_SPLIT_SHA"
        ],
    }
    for name, actual in protected_factor_hashes.items():
        expected = factor_dataset.get(name)
        if expected is not None and actual != expected:
            raise RuntimeError(f"factor-conditioned protected hash changed: {name}")
    guard = document["historical_evidence_boundary"]
    guard_path = root / str(guard["holdout_guard_path"])
    if sha256_file(guard_path) != str(guard["holdout_guard_sha256"]):
        raise RuntimeError("historical HOLDOUT guard changed")
    guard_state = json.loads(guard_path.read_text(encoding="utf-8"))
    if (
        int(guard_state["guard_after"]) != int(guard["guard"])
        or int(guard_state["scientific_open_count"])
        != int(guard["scientific_opens"])
    ):
        raise RuntimeError("historical HOLDOUT consumed state changed")
    normalizer_path = root / str(document["normalizer"]["path"])
    if sha256_file(normalizer_path) != str(document["normalizer"]["sha256"]):
        raise RuntimeError("frozen V2 normalizer changed")
    normalizer = load_hazard_normalizer(normalizer_path)
    if canonical_sha256(document["architecture"]) != str(
        document["reference_model"]["architecture_sha256"]
    ):
        raise RuntimeError("GRU20 architecture changed")
    if canonical_sha256(model_v2_anchor_refined_policy()) != str(
        document["training_protocol"]["extraction_policy_sha256"]
    ):
        raise RuntimeError("anchor-refined extraction policy changed")
    _verify_protected_records(
        root,
        (
            document["reference_model"]["candidate_freeze"],
            document["reference_model"]["candidate_record"],
            *document["reference_model"]["checkpoints"],
        ),
    )
    base = prepare_model_v2_training_data(root, document)
    factor_runs, factor_annotations, factor_rows, factor_precursors = (
        _load_factor_conditioned_runs(root, document, "FACTOR_TRAIN")
    )
    if set(base.runs) & set(factor_runs):
        raise RuntimeError("fresh FACTOR_TRAIN reuses a historical run ID")
    runs = {**base.runs, **factor_runs}
    annotations = {**base.annotations, **factor_annotations}
    precursors = {**base.precursor_samples, **factor_precursors}
    recipe = _training_recipe(document)
    extraction = audit_model_v2_anchor_refined_extraction(
        runs,
        precursors,
        annotations,
        per_category=int(recipe["initial_negative_per_gait_category"]),
        target_contact_cap=int(recipe["target_contact_cap_per_run"]),
        benign_precursor_cap=int(recipe["benign_precursor_cap_per_run"]),
    )
    if not extraction["passed"]:
        raise RuntimeError("factor-conditioned extraction audit failed")
    diagnostic = _frozen_normalizer_diagnostic(factor_runs, normalizer)
    normalizer_gates = document["normalizer"]["train_only_diagnostic_gates"]
    normalizer_usable = bool(
        diagnostic["all_finite"]
        and float(diagnostic["absolute_z"]["maximum"])
        <= float(normalizer_gates["maximum_absolute_z"])
        and float(diagnostic["fraction_absolute_z_gt_20"])
        <= float(normalizer_gates["fraction_absolute_z_gt_20_max"])
    )
    if not normalizer_usable:
        raise RuntimeError("frozen normalizer is unusable for FACTOR_TRAIN")
    artifact_path = root / str(document["artifacts"]["path"])
    source_ledger_path = artifact_path / "training_source_ledger.json"
    source_ledger = {
        "schema_version": 1,
        "intervention_config_sha256": config_sha,
        "authorized_sources": list(
            document["training_source_contract"]["authorized"]
        ),
        "forbidden_sources": list(
            document["training_source_contract"]["forbidden"]
        ),
        "entries": list(document["training_source_contract"]["ledger"]),
        "actual_training_run_counts": {
            "Unified_TRAIN": int(
                document["effective_train"]["unified"]["run_count"]
            ),
            "V2_TRAIN": int(
                document["effective_train"]["augmentation"]["run_count"]
            ),
            "FACTOR_TRAIN": len(factor_runs),
        },
        "actual_training_run_total": len(runs),
        "factor_train_physical_outcomes": dict(
            sorted(
                Counter(
                    str(row["objective_physical_outcome"])
                    for row in factor_rows.values()
                ).items()
            )
        ),
        "forbidden_training_or_hnm_use_count": 0,
        "factor_validation_payload_reads": 0,
        "historical_holdout_payload_reads": 0,
    }
    if len(runs) != int(document["effective_train"]["combined_run_count"]):
        raise RuntimeError("combined factor-conditioned TRAIN identity changed")
    _write_json(source_ledger_path, source_ledger)
    pretraining_path = artifact_path / "pretraining_audit.json"
    effective_identity = [
        {"dataset_id": annotations[run_id].dataset_id, "run_id": run_id}
        for run_id in sorted(runs)
    ]
    pretraining = {
        "schema_version": 1,
        "status": "FACTOR_CONDITIONED_PRETRAINING_READY",
        "intervention_config_sha256": config_sha,
        "dataset_freeze_file_sha256": sha256_file(dataset_freeze_path),
        "dataset_freeze_semantic_sha256": dataset_freeze["FACTOR_DATASET_FREEZE_SHA"],
        "base_train_run_count": len(base.runs),
        "factor_train_run_count": len(factor_runs),
        "effective_train_run_count": len(runs),
        "effective_train_ids_sha256": canonical_sha256(effective_identity),
        "factor_train_ids_sha256": canonical_sha256(sorted(factor_runs)),
        "training_source_ledger_sha256": sha256_file(source_ledger_path),
        "factor_train_composition": dict(
            sorted(Counter(str(row["group"]) for row in factor_rows.values()).items())
        ),
        "extraction_audit": extraction,
        "frozen_normalizer_diagnostic": diagnostic,
        "frozen_normalizer_usable": normalizer_usable,
        "normalizer_sha256": sha256_file(normalizer_path),
        "feature_schema_sha256": feature_schema_hash(),
        "architecture_sha256": canonical_sha256(document["architecture"]),
        "threshold": document["runtime_decision"]["threshold"],
        "persistence_ms": document["runtime_decision"]["persistence_ms"],
        "monitor_source": "effective_train_deterministic_partition_only",
        "factor_validation_opened": False,
        "v2_validation_opened": False,
        "old_holdout_payload_reads": 0,
        "optimizer_steps": 0,
        "protocol_frozen_before_optimizer_step_1": True,
    }
    if dry_run:
        _write_json(pretraining_path, pretraining)
        return {
            "status": "FACTOR_CONDITIONED_PRETRAINING_READY",
            "intervention_config_sha256": config_sha,
            "pretraining_audit_sha256": sha256_file(pretraining_path),
            "training_source_ledger_sha256": sha256_file(source_ledger_path),
            "effective_train_run_count": len(runs),
            "factor_train_run_count": len(factor_runs),
            "normalizer_usable": normalizer_usable,
        }
    if (
        not pretraining_path.is_file()
        or json.loads(pretraining_path.read_text(encoding="utf-8")) != pretraining
    ):
        raise RuntimeError("run and preserve the factor pretraining dry-run first")
    forbidden_existing = (
        artifact_path / "candidate_freeze.json",
        artifact_path / "factor_validation_authorization.json",
        artifact_path / "factor_validation_paired_sand.json",
        artifact_path / "factor_validation_comparison.json",
        artifact_path / "historical_v2_validation_regression.json",
        artifact_path / "final_intervention_decision.json",
        artifact_path / "development_candidate_freeze.json",
        artifact_path / "failure_interpretation.json",
    )
    if any(path.exists() for path in forbidden_existing) or any(
        (artifact_path / "checkpoints").glob("*.pt")
    ):
        raise RuntimeError("factor-conditioned training artifacts already exist")
    fit_positive, monitor_positive = model_v2_anchor_refined_positive_plan(
        runs, precursors, annotations
    )
    candidate = train_hazard_candidate(
        root,
        runs,
        precursors,
        artifact_path,
        recipe,
        annotations,
        checkpoint_prefix=str(document["artifacts"]["checkpoint_prefix"]),
        progress=progress,
        normalizer_override=normalizer,
        normalizer_source_path=normalizer_path,
        fit_positive_endpoints=fit_positive,
        monitor_positive_endpoints=monitor_positive,
        extraction_audit_override=extraction,
        fixed_monitor_endpoint_set=True,
    )
    if candidate.record["normalizer_fits"] != 0:
        raise RuntimeError("factor-conditioned training refit the normalizer")
    training_record = {
        "schema_version": 1,
        "intervention_config_sha256": config_sha,
        "dataset_freeze_sha256": dataset_freeze["FACTOR_DATASET_FREEZE_SHA"],
        "effective_train_ids_sha256": pretraining["effective_train_ids_sha256"],
        "factor_train_ids_sha256": pretraining["factor_train_ids_sha256"],
        "training_source_ledger_sha256": sha256_file(source_ledger_path),
        "candidate": candidate.record,
        "normalizer_fits": 0,
        "threshold_searches": 0,
        "persistence_searches": 0,
        "architecture_searches": 0,
        "seed_searches": 0,
        "sensor_fusion_experiments": 0,
    }
    training_record_path = artifact_path / "training_record.json"
    _write_json(training_record_path, training_record)
    exposure = {
        "definition": "each_fit_endpoint_seen_once_per_completed_epoch",
        "rounds": [
            {
                "round": row["round"],
                "available_fit_endpoints": row["positive_exposure_available"],
                "actual_batch_exposure_by_seed": row["positive_batch_exposure_by_seed"],
                "epochs_completed": row["epochs_completed"],
                "fit_window_exposure_by_dataset": row["fit_window_exposure_by_dataset"],
                "fit_window_exposure_by_design_role": row[
                    "fit_window_exposure_by_design_role"
                ],
                "fit_window_exposure_by_scenario_family": row[
                    "fit_window_exposure_by_scenario_family"
                ],
            }
            for row in candidate.record["rounds"]
        ],
    }
    exposure_path = artifact_path / "training_exposure.json"
    _write_json(exposure_path, exposure)
    factor_metadata = {
        run_id: {
            "source": row["source_terrain"],
            "speed_mps": row["speed_mps"],
            "topology": row["sink_pattern"],
            "precontact_phase": row["target_contact_summary"]["precontact_phase"],
            "group": row["group"],
        }
        for run_id, row in factor_rows.items()
    }
    hnm_rounds = []
    for row in candidate.record["rounds"]:
        if "hard_negative_mining" not in row:
            continue
        mining = dict(row["hard_negative_mining"])
        selected_by_run = mining["selected_by_run"]
        counts = {
            name: dict(
                sorted(
                    Counter(
                        str(factor_metadata[run_id][field])
                        for run_id, endpoints in selected_by_run.items()
                        for _ in endpoints
                        if run_id in factor_metadata
                    ).items()
                )
            )
            for name, field in (
                ("source", "source"),
                ("speed", "speed_mps"),
                ("topology", "topology"),
                ("precontact_phase", "precontact_phase"),
                ("group", "group"),
            )
        }
        hnm_rounds.append(
            {
                "hnm_round": int(row["round"]) + 1,
                **mining,
                "factor_train_composition": counts,
            }
        )
    hnm_provenance = {
        "rounds": hnm_rounds,
        "source": "effective_train_only",
        "policy_unchanged": True,
        "forbidden_splits": [
            "FACTOR_VALIDATION",
            "V2_VALIDATION",
            "Generalization_VALIDATION",
            "Generalization_HOLDOUT",
            "MILD_RECALIBRATED_DISCOVERY",
            "MILD_RECALIBRATED_CONFIRMATION",
        ],
    }
    hnm_path = artifact_path / "hnm_provenance.json"
    _write_json(hnm_path, hnm_provenance)
    candidate_freeze = {
        "schema_version": 1,
        "candidate_id": document["artifacts"]["candidate_id"],
        "role": "DEVELOPMENT_FACTOR_CONDITIONED_CANDIDATE",
        "source_commit": document["experiment"]["source_commit"],
        "intervention_config_sha256": config_sha,
        "dataset_freeze_sha256": dataset_freeze["FACTOR_DATASET_FREEZE_SHA"],
        "training_split_sha256": pretraining["factor_train_ids_sha256"],
        "factor_train_split_sha256": factor_dataset[
            "factor_train_split_sha256"
        ],
        "factor_validation_split_sha256": factor_dataset[
            "factor_validation_split_sha256"
        ],
        "effective_train_ids_sha256": pretraining["effective_train_ids_sha256"],
        "normalizer_sha256": candidate.record["normalizer_sha256"],
        "feature_schema_sha256": candidate.record["feature_schema_sha256"],
        "architecture": document["architecture"],
        "architecture_sha256": canonical_sha256(document["architecture"]),
        "checkpoint_sha256": candidate.record["checkpoint_sha256"],
        "ensemble_membership": list(document["training"]["seeds"]),
        "hnm_provenance_sha256": sha256_file(hnm_path),
        "training_exposure_sha256": sha256_file(exposure_path),
        "training_record_sha256": sha256_file(training_record_path),
        "training_source_ledger_sha256": sha256_file(source_ledger_path),
        "threshold": document["runtime_decision"]["threshold"],
        "persistence_ms": document["runtime_decision"]["persistence_ms"],
        "candidate_frozen_before_factor_validation": True,
        "candidate_frozen_before_validation": True,
        "factor_validation_evaluated": False,
        "v2_validation_evaluated": False,
        "historical_v2_validation_evaluated": False,
        "generalization_holdout_guard_count": 0,
        "final_generalization_established": False,
        "old_holdout_payload_reads": 0,
        "optimizer_training_provenance": {
            "optimizer": document["training"]["optimizer"],
            "learning_rate": document["training"]["learning_rate"],
            "weight_decay": document["training"]["weight_decay"],
            "loss": document["training"]["loss"],
            "batch_size": document["training"]["batch_size"],
            "max_epochs": document["training"]["max_epochs"],
            "patience": document["training"]["patience"],
            "train_only_monitor": document["training"]["epoch_selection"],
        },
    }
    candidate_freeze_path = artifact_path / "candidate_freeze.json"
    _write_json(candidate_freeze_path, candidate_freeze)
    candidate_freeze_sha = sha256_file(candidate_freeze_path)

    validation_seal_path = dataset_path / "validation_seal.json"
    validation_seal = json.loads(validation_seal_path.read_text(encoding="utf-8"))
    if (
        sha256_file(validation_seal_path)
        != str(factor_dataset["validation_seal_sha256"])
        or validation_seal["status"] != "SEALED_FOR_FUTURE_FACTOR_VALIDATION"
        or not validation_seal["requires_frozen_future_candidate"]
    ):
        raise RuntimeError("FACTOR_VALIDATION seal cannot be authorized")
    validation_authorization = {
        "schema_version": 1,
        "authorized": True,
        "authorization_timestamp": datetime.now(
            ZoneInfo(str(document["experiment"]["timezone"]))
        ).isoformat(timespec="seconds"),
        "candidate_id": candidate_freeze["candidate_id"],
        "candidate_freeze_sha256": candidate_freeze_sha,
        "reference_candidate_id": document["reference_model"]["candidate_id"],
        "dataset_id": factor_dataset["dataset_id"],
        "dataset_freeze_sha256": dataset_freeze["FACTOR_DATASET_FREEZE_SHA"],
        "validation_split_sha256": factor_dataset[
            "factor_validation_split_sha256"
        ],
        "validation_seal_sha256": sha256_file(validation_seal_path),
        "validation_seal_before_state": validation_seal["status"],
        "validation_seal_after_state": (
            "AUTHORIZED_ONCE_FOR_FROZEN_FACTOR_CONDITIONED_CANDIDATE"
        ),
        "evaluation_config_sha256": config_sha,
        "open_count_before": 0,
        "open_count_after": 1,
        "candidate_mutation_after_authorization": False,
    }
    validation_authorization_path = (
        artifact_path / "factor_validation_authorization.json"
    )
    _write_json(validation_authorization_path, validation_authorization)

    validation_runs, _, validation_rows, validation_precursors = (
        _load_factor_conditioned_runs(
            root,
            document,
            "FACTOR_VALIDATION",
            candidate_freeze_path=candidate_freeze_path,
            validation_authorization_path=validation_authorization_path,
        )
    )
    reference_paths = tuple(
        root / str(row["path"]) for row in document["reference_model"]["checkpoints"]
    )
    candidate_replays = replay_hazard_runs(
        validation_runs, candidate.normalizer, candidate.checkpoint_paths
    )
    reference_replays = replay_hazard_runs(validation_runs, normalizer, reference_paths)
    validation_protocol = document["validation_protocol"]
    evaluation_arguments = {
        "threshold": float(document["runtime_decision"]["threshold"]),
        "persistence_ms": int(document["runtime_decision"]["persistence_ms"]),
        "adverse_threshold": float(validation_protocol["adverse_threshold"]),
    }
    reference_validation = evaluate_factor_conditioned_validation(
        validation_runs,
        reference_replays,
        validation_rows,
        validation_precursors,
        **evaluation_arguments,
    )
    candidate_validation = evaluate_factor_conditioned_validation(
        validation_runs,
        candidate_replays,
        validation_rows,
        validation_precursors,
        **evaluation_arguments,
    )
    reference_validation_path = artifact_path / "factor_validation_reference.json"
    candidate_validation_path = artifact_path / "factor_validation_candidate.json"
    _write_json(reference_validation_path, reference_validation)
    _write_json(candidate_validation_path, candidate_validation)
    reference_by_run = {
        str(row["run_id"]): row for row in reference_validation["run_results"]
    }
    candidate_by_run = {
        str(row["run_id"]): row for row in candidate_validation["run_results"]
    }
    paired_rows = []
    for run_id in sorted(reference_by_run):
        reference_row = reference_by_run[run_id]
        candidate_row = candidate_by_run[run_id]
        if reference_row["physical_outcome"] != "STRICT_BENIGN":
            continue
        paired_rows.append(
            {
                "run_id": run_id,
                "source": reference_row["source"],
                "speed_mps": reference_row["speed_mps"],
                "severity": reference_row["severity"],
                "topology": reference_row["topology"],
                "precontact_phase": reference_row["precontact_phase"],
                "factor_manifold": reference_row["factor_manifold"],
                "reference_max_probability": reference_row["max_probability"],
                "candidate_max_probability": candidate_row["max_probability"],
                "probability_delta": float(candidate_row["max_probability"])
                - float(reference_row["max_probability"]),
                "reference_longest_threshold_streak": reference_row[
                    "longest_threshold_streak"
                ],
                "candidate_longest_threshold_streak": candidate_row[
                    "longest_threshold_streak"
                ],
                "reference_reflex": reference_row["reflex"],
                "candidate_reflex": candidate_row["reflex"],
            }
        )
    paired_sand = {
        "schema_version": 1,
        "candidate_freeze_sha256": candidate_freeze_sha,
        "validation_authorization_sha256": sha256_file(
            validation_authorization_path
        ),
        "run_count": len(paired_rows),
        "rows": paired_rows,
        "tuning_after_view": False,
    }
    paired_path = artifact_path / "factor_validation_paired_sand.json"
    _write_json(paired_path, paired_sand)
    comparison = _factor_validation_comparison(
        reference_validation, candidate_validation
    )
    comparison["paired_sand_sha256"] = sha256_file(paired_path)
    comparison_path = artifact_path / "factor_validation_comparison.json"
    _write_json(comparison_path, comparison)

    legacy_runs, legacy_annotations = load_model_v2_runs(
        root / str(document["v2_dataset"]["path"]),
        base.v2_manifest,
        "V2_VALIDATION",
        candidate_freeze_path=candidate_freeze_path,
    )
    legacy_rows = {
        str(row["run_id"]): row
        for row in base.v2_manifest["runs"]
        if row["split"] == "V2_VALIDATION" and bool(row["valid"])
    }
    legacy_precursors = {
        run_id: (
            None
            if row["i1_summary"]["first_sample"] is None
            else int(row["i1_summary"]["first_sample"])
        )
        for run_id, row in legacy_rows.items()
    }
    legacy_candidate_replays = replay_hazard_runs(
        legacy_runs, candidate.normalizer, candidate.checkpoint_paths
    )
    legacy_candidate = evaluate_model_v2_validation(
        legacy_runs,
        legacy_candidate_replays,
        legacy_precursors,
        legacy_annotations,
        legacy_rows,
        document["legacy_regression"]["gates"],
        threshold=float(document["runtime_decision"]["threshold"]),
        persistence_ms=int(document["runtime_decision"]["persistence_ms"]),
    )
    reference_result_path = root / str(
        document["legacy_regression"]["reference_result_path"]
    )
    if sha256_file(reference_result_path) != str(
        document["legacy_regression"]["reference_result_sha256"]
    ):
        raise RuntimeError("historical reference V2 validation result changed")
    legacy_reference = json.loads(reference_result_path.read_text(encoding="utf-8"))
    legacy_comparison = _comparison_metrics(legacy_reference, legacy_candidate)
    legacy_result = {
        "reference_result_sha256": sha256_file(reference_result_path),
        "reference": legacy_reference,
        "candidate": legacy_candidate,
        "comparison": legacy_comparison,
        "tuning_after_view": False,
    }
    legacy_path = artifact_path / "historical_v2_validation_regression.json"
    _write_json(legacy_path, legacy_result)

    rules = document["decision_rules"]
    metrics = comparison["metrics"]
    specificity_gain = float(metrics["strict_sand_specificity"]["delta"])
    fp_reduction = -float(metrics["false_reflex_count"]["delta"])
    adverse_reduction = -float(metrics["adverse_margin_rate"]["delta"])
    improved_factors = sum(
        bool(value["improved"])
        for value in comparison["factors"].values()
        if int(value["reference_adverse"]) > 0
    )
    source_speed_cells_improved = sum(
        bool(value["improved"])
        for value in comparison["source_speed"].values()
        if int(value["reference_adverse"]) > 0
    )
    adverse_manifold = comparison["factors"]["adverse_direction_manifold"]
    adverse_manifold_rate_reduction = -float(
        adverse_manifold["adverse_rate_delta"]
    )
    fresh_support_preserved = bool(
        candidate_validation["support"]["overall"]["recall"]
        >= float(rules["support"]["overall_recall_min"])
        and candidate_validation["support"]["ordinary"]["recall"]
        >= float(rules["support"]["ordinary_recall_min"])
        and candidate_validation["support"]["delayed"]["recall"]
        >= float(rules["support"]["delayed_recall_min"])
        and candidate_validation["support"]["right"]["recall"]
        >= float(rules["support"]["right_recall_min"])
    )
    legacy_support_preserved = bool(
        legacy_candidate["primary"]["support_hazard_recall"]
        >= float(rules["legacy_regression"]["support_recall_min"])
        and legacy_candidate["side"]["support"]["RIGHT_ONLY"]["recall"]
        >= float(rules["legacy_regression"]["right_support_recall_min"])
        and legacy_candidate["families"]["DELAYED_SAND_SUPPORT_ONSET"]["recall"]
        >= float(rules["legacy_regression"]["delayed_support_recall_min"])
        and legacy_candidate["primary"]["primary_no_hazard_specificity"]
        >= float(rules["legacy_regression"]["specificity_min"])
        and legacy_candidate["primary"]["slip_hazard_recall"]
        >= float(legacy_reference["primary"]["slip_hazard_recall"])
        - float(rules["legacy_regression"]["slip_recall_drop_max"])
    )
    effective = bool(
        candidate_validation["sand"]["overall"]["specificity"]
        >= float(rules["effective"]["candidate_specificity_min"])
        and specificity_gain >= float(rules["effective"]["specificity_gain_min"])
        and fp_reduction >= int(rules["effective"]["false_reflex_reduction_min"])
        and adverse_reduction >= float(rules["effective"]["adverse_rate_reduction_min"])
        and adverse_manifold_rate_reduction
        >= float(
            rules["effective"]["adverse_manifold_adverse_rate_reduction_min"]
        )
        and improved_factors >= int(rules["effective"]["factor_groups_improved_min"])
        and source_speed_cells_improved
        >= int(rules["effective"]["source_speed_cells_improved_min"])
        and fresh_support_preserved
        and legacy_support_preserved
    )
    partial = bool(
        specificity_gain >= float(rules["partial"]["specificity_gain_min"])
        and adverse_reduction >= float(rules["partial"]["adverse_rate_reduction_min"])
        and fresh_support_preserved
        and legacy_support_preserved
    )
    verdict = (
        "FACTOR_CONDITIONED_DATA_INTERVENTION_EFFECTIVE"
        if effective
        else "FACTOR_CONDITIONED_DATA_INTERVENTION_PARTIALLY_EFFECTIVE"
        if partial
        else "FACTOR_CONDITIONED_DATA_INTERVENTION_NOT_EFFECTIVE"
    )
    if effective:
        next_milestone = str(rules["effective_next"])
    else:
        next_milestone = str(rules["otherwise_next"])
    residual_factor_count = sum(
        int(value["candidate_adverse"]) > 0 for value in comparison["factors"].values()
    )
    hypothesis_status = (
        "FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS_DEVELOPMENT_SUPPORTED"
        if effective
        else "FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS_PARTIALLY_SUPPORTED"
        if partial
        else "FACTOR_CONDITIONED_DATA_DOMAIN_HYPOTHESIS_NOT_SUPPORTED_BY_DEVELOPMENT"
    )
    decision = {
        "schema_version": 1,
        "intervention_verdict": verdict,
        "recommended_next_scientific_milestone": next_milestone,
        "specificity_gain": specificity_gain,
        "false_reflex_reduction": fp_reduction,
        "adverse_rate_reduction": adverse_reduction,
        "factor_groups_improved": improved_factors,
        "source_speed_cells_improved": source_speed_cells_improved,
        "adverse_manifold_adverse_rate_reduction": (
            adverse_manifold_rate_reduction
        ),
        "fresh_support_preserved": fresh_support_preserved,
        "legacy_regression_preserved": legacy_support_preserved,
        "hypothesis_status": hypothesis_status,
        "frozen_decision_rules": rules,
        "historical_final_model_verdict": "MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED",
        "new_candidate_final_generalization": "NOT_ESTABLISHED",
        "new_independent_external_and_final_evidence_required": True,
        "historical_holdout_reuse": False,
        "no_retraining_after_validation": True,
    }
    decision_path = artifact_path / "final_intervention_decision.json"
    _write_json(decision_path, decision)
    failure_audit_path: Path | None = None
    if not effective:
        ranked_residuals = sorted(
            (
                {
                    "factor": name,
                    "runs": int(value["runs"]),
                    "candidate_adverse": int(value["candidate_adverse"]),
                    "candidate_false_reflex": int(value["candidate_false_reflex"]),
                    "adverse_rate_delta": float(value["adverse_rate_delta"]),
                }
                for name, value in comparison["factors"].items()
            ),
            key=lambda value: (
                -value["candidate_adverse"],
                -value["candidate_false_reflex"],
                value["factor"],
            ),
        )
        failure_audit = {
            "schema_version": 1,
            "mode": "READ_ONLY_POST_VALIDATION_FAILURE_INTERPRETATION",
            "intervention_verdict": verdict,
            "recommended_next_scientific_milestone": next_milestone,
            "fresh_train_run_count": len(factor_runs),
            "fresh_train_composition": pretraining["factor_train_composition"],
            "frozen_normalizer_diagnostic": diagnostic,
            "residual_factor_count": residual_factor_count,
            "largest_residual_factors": ranked_residuals[:5],
            "fresh_support_preserved": fresh_support_preserved,
            "legacy_regression_preserved": legacy_support_preserved,
            "no_additional_training": True,
            "old_holdout_payload_reads": 0,
        }
        failure_audit_path = artifact_path / "failure_interpretation.json"
        _write_json(failure_audit_path, failure_audit)
    development_freeze_path: Path | None = None
    if effective:
        development_freeze = {
            **candidate_freeze,
            "schema_version": 1,
            "role": "DEVELOPMENT_FACTOR_CONDITIONED_CANDIDATE",
            "pre_evaluation_candidate_freeze_sha256": candidate_freeze_sha,
            "factor_validation_comparison_sha256": sha256_file(comparison_path),
            "factor_validation_authorization_sha256": sha256_file(
                validation_authorization_path
            ),
            "factor_validation_paired_sand_sha256": sha256_file(paired_path),
            "historical_v2_validation_regression_sha256": sha256_file(legacy_path),
            "final_intervention_decision_sha256": sha256_file(decision_path),
            "factor_validation_evaluated": True,
            "v2_validation_evaluated": True,
            "historical_v2_validation_evaluated": True,
            "final_generalization_established": False,
            "new_independent_external_and_final_evidence_required": True,
        }
        development_freeze_path = artifact_path / "development_candidate_freeze.json"
        _write_json(development_freeze_path, development_freeze)
    evaluation_freeze = {
        "schema_version": 1,
        "candidate_id": candidate_freeze["candidate_id"],
        "candidate_freeze_sha256": candidate_freeze_sha,
        "factor_validation_authorization_sha256": sha256_file(
            validation_authorization_path
        ),
        "factor_validation_reference_sha256": sha256_file(reference_validation_path),
        "factor_validation_candidate_sha256": sha256_file(candidate_validation_path),
        "factor_validation_paired_sand_sha256": sha256_file(paired_path),
        "factor_validation_comparison_sha256": sha256_file(comparison_path),
        "historical_v2_validation_regression_sha256": sha256_file(legacy_path),
        "final_intervention_decision_sha256": sha256_file(decision_path),
        "intervention_verdict": verdict,
        "recommended_next_scientific_milestone": next_milestone,
        "development_candidate_freeze_sha256": (
            None
            if development_freeze_path is None
            else sha256_file(development_freeze_path)
        ),
        "failure_interpretation_sha256": (
            None if failure_audit_path is None else sha256_file(failure_audit_path)
        ),
        "old_holdout_payload_reads": 0,
        "old_holdout_inference": 0,
        "fresh_validation_training_use": 0,
        "factor_validation_open_count": 1,
        "factor_validation_state": "CONSUMED_DEVELOPMENT_VALIDATION",
        "candidate_mutation_after_authorization": False,
    }
    evaluation_freeze_path = artifact_path / "evaluation_freeze.json"
    _write_json(evaluation_freeze_path, evaluation_freeze)
    result = {
        "training_status": "FACTOR_CONDITIONED_DATA_INTERVENTION_TRAINING_COMPLETE",
        "intervention_verdict": verdict,
        "recommended_next_scientific_milestone": next_milestone,
        "intervention_config_sha256": config_sha,
        "dataset_freeze_sha256": dataset_freeze["FACTOR_DATASET_FREEZE_SHA"],
        "training_record_sha256": sha256_file(training_record_path),
        "training_source_ledger_sha256": sha256_file(source_ledger_path),
        "training_exposure_sha256": sha256_file(exposure_path),
        "hnm_provenance_sha256": sha256_file(hnm_path),
        "candidate_freeze_sha256": candidate_freeze_sha,
        "factor_validation_authorization_sha256": sha256_file(
            validation_authorization_path
        ),
        "evaluation_freeze_sha256": sha256_file(evaluation_freeze_path),
        "factor_validation_reference_sha256": sha256_file(reference_validation_path),
        "factor_validation_candidate_sha256": sha256_file(candidate_validation_path),
        "factor_validation_paired_sand_sha256": sha256_file(paired_path),
        "factor_validation_comparison_sha256": sha256_file(comparison_path),
        "historical_v2_validation_regression_sha256": sha256_file(legacy_path),
        "final_intervention_decision_sha256": sha256_file(decision_path),
        "development_candidate_freeze_sha256": (
            None
            if development_freeze_path is None
            else sha256_file(development_freeze_path)
        ),
        "failure_interpretation_sha256": (
            None if failure_audit_path is None else sha256_file(failure_audit_path)
        ),
        "optimizer_steps": candidate.record["optimizer_steps"],
        "checkpoint_writes": candidate.record["checkpoint_writes"],
        "normalizer_fits": 0,
        "hnm_rounds": HNM_ROUNDS,
        "threshold_searches": 0,
        "persistence_searches": 0,
        "architecture_searches": 0,
        "seed_searches": 0,
        "sensor_fusion_experiments": 0,
        "old_holdout_payload_reads": 0,
        "old_holdout_inference": 0,
        "old_sand_confirmation_training_use": 0,
        "fresh_validation_training_use": 0,
        "new_simulation_runs": 0,
        "new_pilot_runs": 0,
        "candidate_families_trained": 1,
        "seeds_trained": len(document["training"]["seeds"]),
        "v1_inference": 0,
        "reference_v2_factor_validation_inference": 1,
        "candidate_factor_validation_inference": 1,
        "factor_validation_open_count": 1,
        "generalization_validation_candidate_inference": 0,
    }
    result_path = artifact_path / "training_result.json"
    _write_json(result_path, result)
    return result


def verify_factor_conditioned_intervention_result(
    root: Path, config_path: Path
) -> dict[str, object]:
    """Verify the completed intervention hash chain without inference."""
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifact_path = root / str(document["artifacts"]["path"])
    result_path = artifact_path / "training_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checks = {
        "intervention_config": sha256_file(config_path)
        == result["intervention_config_sha256"],
        "training_record": sha256_file(artifact_path / "training_record.json")
        == result["training_record_sha256"],
        "training_source_ledger": sha256_file(
            artifact_path / "training_source_ledger.json"
        )
        == result["training_source_ledger_sha256"],
        "training_exposure": sha256_file(artifact_path / "training_exposure.json")
        == result["training_exposure_sha256"],
        "hnm_provenance": sha256_file(artifact_path / "hnm_provenance.json")
        == result["hnm_provenance_sha256"],
        "candidate_freeze": sha256_file(artifact_path / "candidate_freeze.json")
        == result["candidate_freeze_sha256"],
        "validation_authorization": sha256_file(
            artifact_path / "factor_validation_authorization.json"
        )
        == result["factor_validation_authorization_sha256"],
        "evaluation_freeze": sha256_file(artifact_path / "evaluation_freeze.json")
        == result["evaluation_freeze_sha256"],
        "factor_validation": sha256_file(
            artifact_path / "factor_validation_comparison.json"
        )
        == result["factor_validation_comparison_sha256"],
        "factor_validation_reference": sha256_file(
            artifact_path / "factor_validation_reference.json"
        )
        == result["factor_validation_reference_sha256"],
        "factor_validation_candidate": sha256_file(
            artifact_path / "factor_validation_candidate.json"
        )
        == result["factor_validation_candidate_sha256"],
        "factor_validation_paired_sand": sha256_file(
            artifact_path / "factor_validation_paired_sand.json"
        )
        == result["factor_validation_paired_sand_sha256"],
        "historical_regression": sha256_file(
            artifact_path / "historical_v2_validation_regression.json"
        )
        == result["historical_v2_validation_regression_sha256"],
        "decision": sha256_file(artifact_path / "final_intervention_decision.json")
        == result["final_intervention_decision_sha256"],
        "conditional_development_freeze": (
            result["development_candidate_freeze_sha256"] is None
            or sha256_file(artifact_path / "development_candidate_freeze.json")
            == result["development_candidate_freeze_sha256"]
        ),
        "conditional_failure_interpretation": (
            result["failure_interpretation_sha256"] is None
            or sha256_file(artifact_path / "failure_interpretation.json")
            == result["failure_interpretation_sha256"]
        ),
        "final_checkpoints": all(
            sha256_file(root / path) == expected
            for path, expected in json.loads(
                (artifact_path / "candidate_freeze.json").read_text(encoding="utf-8")
            )["checkpoint_sha256"].items()
        ),
        "zero_forbidden_counters": all(
            int(result[name]) == 0
            for name in (
                "normalizer_fits",
                "threshold_searches",
                "persistence_searches",
                "architecture_searches",
                "seed_searches",
                "sensor_fusion_experiments",
                "old_holdout_payload_reads",
                "old_holdout_inference",
                "old_sand_confirmation_training_use",
                "fresh_validation_training_use",
            )
        ),
        "single_validation_open": result["factor_validation_open_count"] == 1,
        "single_candidate_family": result["candidate_families_trained"] == 1
        and result["seeds_trained"] == 3,
    }
    if not all(checks.values()):
        raise RuntimeError("factor-conditioned intervention verification failed")
    return {
        "status": "FACTOR_CONDITIONED_DATA_INTERVENTION_VERIFIED",
        "checks": checks,
        "intervention_verdict": result["intervention_verdict"],
        "candidate_freeze_sha256": result["candidate_freeze_sha256"],
        "evaluation_freeze_sha256": result["evaluation_freeze_sha256"],
    }


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


def run_model_v2_anchor_refined_training(
    root: Path,
    config_path: Path,
    *,
    dry_run: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Train and evaluate the frozen late-interior Model V2 candidate once."""
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if document["experiment"]["id"] != "MODEL_V2_ANCHOR_REFINED_TRAINING":
        raise ValueError("unsupported anchor-refined training config")
    config_sha = sha256_file(config_path)
    design_path = root / str(document["design"]["path"])
    if sha256_file(design_path) != str(document["design"]["config_sha256"]):
        raise RuntimeError("frozen anchor-refinement design config changed")
    design = yaml.safe_load(design_path.read_text(encoding="utf-8"))
    if (
        design["dry_run_freeze"]["anchor_refinement_design_sha256"]
        != document["design"]["anchor_refinement_design_sha256"]
        or canonical_sha256(model_v2_anchor_refinement_candidates())
        != document["design"]["candidate_list_sha256"]
        or canonical_sha256(model_v2_anchor_refined_policy())
        != document["design"]["extraction_policy_sha256"]
    ):
        raise RuntimeError("frozen anchor-refinement identity changed")
    _verify_protected_records(
        root,
        (
            document["baseline_v2"]["candidate_freeze"],
            document["baseline_v2"]["normalizer"],
            *document["baseline_v2"]["checkpoints"],
            document["rebalanced_v2"]["candidate_freeze"],
            document["rebalanced_v2"]["evaluation_freeze"],
            *document["rebalanced_v2"]["checkpoints"],
        ),
    )
    data = prepare_model_v2_training_data(root, document)
    recipe = _training_recipe(document)
    extraction = audit_model_v2_anchor_refined_extraction(
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
    expected_monitor = {
        key: int(document["monitor"][key])
        for key in (
            "slip",
            "ordinary_support",
            "delayed_support_concrete",
            "delayed_support_marble",
        )
    } | {"total": int(document["monitor"]["positive_total"])}
    if (
        not extraction["passed"]
        or extraction["extraction_policy_sha256"]
        != document["design"]["extraction_policy_sha256"]
        or extraction["positive_window_ids_sha256"]
        != document["design"]["positive_window_ids_sha256"]
        or extraction["negative_window_ids_sha256"]
        != document["design"]["negative_window_ids_sha256"]
        or extraction["masked_window_sha256"]
        != document["design"]["masked_window_sha256"]
        or extraction["monitor_endpoint_sha256"]
        != document["design"]["monitor_endpoint_sha256"]
        or extraction["monitor_positive_sha256"]
        != document["monitor"]["positive_sha256"]
        or extraction["fit_positive_counts"] != expected_fit
        or extraction["monitor_positive_counts"] != expected_monitor
        or int(extraction["fit_negative_count"])
        != int(document["window_extraction"]["expected_fit_negative"])
        or int(extraction["monitor_negative_count"])
        != int(document["monitor"]["negative_total"])
        or int(extraction["all_positive_count"])
        != int(document["window_extraction"]["expected_all_positive"])
        or int(extraction["all_negative_count"])
        != int(document["window_extraction"]["expected_all_negative"])
        or int(extraction["fit_monitor_endpoint_overlap"]) != 0
    ):
        raise RuntimeError("pretraining extraction differs from anchor design")

    normalizer_path = root / str(document["normalizer"]["path"])
    if sha256_file(normalizer_path) != str(document["normalizer"]["sha256"]):
        raise RuntimeError("frozen V2 normalizer changed")
    normalizer = load_hazard_normalizer(normalizer_path)
    artifact_path = root / str(document["artifacts"]["path"])
    pretraining_path = artifact_path / "pretraining_audit.json"
    pretraining = {
        "execution_config_sha256": config_sha,
        "design_config_sha256": document["design"]["config_sha256"],
        "anchor_refinement_design_sha256": document["design"][
            "anchor_refinement_design_sha256"
        ],
        "input_audit": data.input_audit,
        "effective_train_composition": data.composition,
        "extraction_audit": extraction,
        "normalizer_sha256": sha256_file(normalizer_path),
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "normalizer_fits": 0,
        "hnm_rounds": 0,
        "monitor_searches": 0,
        "v2_validation_waveform_opened": False,
        "generalization_validation_v2_inference": False,
        "generalization_holdout_guard_count": 0,
    }
    if dry_run:
        _write_json(pretraining_path, pretraining)
        return {
            "status": "MODEL_V2_ANCHOR_REFINED_PRETRAINING_READY",
            "execution_config_sha256": config_sha,
            "pretraining_audit_sha256": sha256_file(pretraining_path),
            "extraction": extraction,
            "normalizer_fits": 0,
        }
    if not pretraining_path.is_file():
        raise RuntimeError("run anchor-refined dry-run before optimizer step 1")
    if json.loads(pretraining_path.read_text(encoding="utf-8")) != pretraining:
        raise RuntimeError("anchor-refined pretraining audit changed after freeze")
    forbidden_existing = (
        artifact_path / "candidate_freeze.json",
        artifact_path / "candidate_evaluation_freeze.json",
        artifact_path / "v2_validation_evaluation.json",
        artifact_path / "training_result.json",
    )
    if any(path.exists() for path in forbidden_existing) or any(
        (artifact_path / "checkpoints").glob("*.pt")
    ):
        raise RuntimeError("anchor-refined training artifacts already exist")

    fit_positive, monitor_positive = model_v2_anchor_refined_positive_plan(
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
        fixed_monitor_endpoint_set=True,
    )
    if (
        candidate.record["normalizer_fits"] != 0
        or candidate.record["normalizer_sha256"] != document["normalizer"]["sha256"]
        or not candidate.record["fixed_monitor_endpoint_set"]
        or any(
            row["monitor_class_counts"]
            != [
                int(document["monitor"]["negative_total"]),
                int(document["monitor"]["positive_total"]),
                0,
            ]
            for row in candidate.record["rounds"]
        )
    ):
        raise RuntimeError("training did not preserve normalizer or monitor freeze")
    training_record = {
        "execution_config_sha256": config_sha,
        "effective_train_ids_sha256": document["effective_train"][
            "effective_run_ids_sha256"
        ],
        "anchor_refinement_design_sha256": document["design"][
            "anchor_refinement_design_sha256"
        ],
        "candidate": candidate.record,
        "normalizer_fits": 0,
        "threshold_searches": 0,
        "persistence_searches": 0,
        "architecture_searches": 0,
        "seed_searches": 0,
        "monitor_searches": 0,
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
            {"hnm_round": int(row["round"]) + 1, **row["hard_negative_mining"]}
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
        "anchor_refinement_design_sha256": document["design"][
            "anchor_refinement_design_sha256"
        ],
        "extraction_policy_sha256": extraction["extraction_policy_sha256"],
        "positive_window_ids_sha256": extraction["positive_window_ids_sha256"],
        "negative_window_ids_sha256": extraction["negative_window_ids_sha256"],
        "masked_window_sha256": extraction["masked_window_sha256"],
        "monitor_endpoint_sha256": extraction["monitor_endpoint_sha256"],
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

    model_specs = {
        "anchor_refined_v2": (candidate.normalizer, candidate.checkpoint_paths),
        "baseline_v2": (
            normalizer,
            tuple(
                root / str(row["path"])
                for row in document["baseline_v2"]["checkpoints"]
            ),
        ),
        "rebalanced_v2": (
            normalizer,
            tuple(
                root / str(row["path"])
                for row in document["rebalanced_v2"]["checkpoints"]
            ),
        ),
        "v1": (
            load_hazard_normalizer(
                root
                / "artifacts/runs/20260829_unified_hazard_reflex_system/normalization/gru_history20.json"
            ),
            tuple(
                root / str(row["path"])
                for row in document["protected_v1"]["checkpoints"]
            ),
        ),
    }
    model_replays: dict[str, Mapping[str, object]] = {}
    model_results: dict[str, Mapping[str, object]] = {}
    for name, (model_normalizer, checkpoint_paths) in model_specs.items():
        replays = replay_hazard_runs(
            validation_runs, model_normalizer, checkpoint_paths
        )
        model_replays[name] = replays
        model_results[name] = evaluate_model_v2_validation(
            validation_runs,
            replays,
            validation_precursor,
            validation_annotations,
            validation_rows,
            gates,
            threshold=threshold,
            persistence_ms=persistence_ms,
        )
    frozen_results = {
        "baseline_v2": root
        / "artifacts/runs/20260901_model_v2_data_only_training/v2_validation_evaluation.json",
        "rebalanced_v2": root
        / "artifacts/runs/20260901_model_v2_extraction_rebalanced_training/v2_validation_evaluation.json",
        "v1": root
        / "artifacts/runs/20260901_model_v2_data_only_training/v1_on_v2_validation.json",
    }
    for name, path in frozen_results.items():
        if model_results[name] != json.loads(path.read_text(encoding="utf-8")):
            raise RuntimeError(f"{name} V2_VALIDATION replay changed")
    result_paths: dict[str, Path] = {}
    for name, result_value in model_results.items():
        filename = (
            "v2_validation_evaluation.json"
            if name == "anchor_refined_v2"
            else f"{name}_on_v2_validation.json"
        )
        path = artifact_path / filename
        _write_json(path, result_value)
        result_paths[name] = path

    delayed = {
        name: _delayed_support_validation_diagnostics(
            validation_runs,
            model_replays[name],
            validation_precursor,
            validation_rows,
            model_results[name],
            threshold=threshold,
            persistence_ms=persistence_ms,
        )
        for name in model_results
    }
    delayed_path = artifact_path / "delayed_support_comparison.json"
    _write_json(delayed_path, delayed)

    seed_replays: dict[str, dict[str, Mapping[str, object]]] = {}
    for name in ("baseline_v2", "rebalanced_v2", "anchor_refined_v2"):
        model_normalizer, checkpoint_paths = model_specs[name]
        seed_replays[name] = {
            str(seed): replay_hazard_runs(
                validation_runs, model_normalizer, (checkpoint_path,)
            )
            for seed, checkpoint_path in zip(
                document["training"]["seeds"], checkpoint_paths
            )
        }
    speed_sand = {
        name: _speed_sand_validation_diagnostics(
            validation_runs,
            model_replays[name],
            seed_replays.get(name, {}),
            validation_rows,
            model_results[name],
            threshold=threshold,
            persistence_ms=persistence_ms,
        )
        for name in model_results
    }
    speed_sand_path = artifact_path / "speed_sand_comparison.json"
    _write_json(speed_sand_path, speed_sand)

    preservation = _validation_preservation_summary(
        validation_runs,
        model_replays["anchor_refined_v2"],
        validation_annotations,
        validation_precursor,
        validation_rows,
        model_results["anchor_refined_v2"],
        delayed["anchor_refined_v2"],
    )
    preservation_path = artifact_path / "preservation_diagnostics.json"
    _write_json(preservation_path, preservation)
    comparison = _four_model_comparison(model_results, delayed)
    comparison_path = artifact_path / "four_model_comparison.json"
    _write_json(comparison_path, comparison)

    assessment = document["intervention_assessment"]
    anchor_result = model_results["anchor_refined_v2"]
    anchor_delayed = delayed["anchor_refined_v2"]
    support_gain_retained = (
        int(anchor_delayed["overall"]["correct"])
        >= int(assessment["delayed_support_min_correct"])
        and int(anchor_delayed["by_source"]["marble"]["correct"])
        >= int(assessment["marble_delayed_support_min_correct"])
    )
    specificity_recovered = (
        int(speed_sand["anchor_refined_v2"]["overall"]["specific"])
        >= int(assessment["speed_sand_min_specific"])
    )
    solved_behavior_retained = (
        int(
            anchor_result["side"]["support"]["RIGHT_ONLY"]["detected"]
        )
        >= int(assessment["right_support_min_correct"])
        and int(
            anchor_result["families"]["STAGED_SAND_BENIGN_CONTROL"]["runs"]
            - anchor_result["families"]["STAGED_SAND_BENIGN_CONTROL"][
                "false_reflex"
            ]
        )
        >= int(assessment["staged_sand_min_specific"])
    )
    intervention_verdict = (
        "V2_ANCHOR_REFINEMENT_EFFECTIVE"
        if support_gain_retained
        and specificity_recovered
        and solved_behavior_retained
        else "V2_ANCHOR_REFINEMENT_NOT_EFFECTIVE"
    )
    internal_verdict = (
        "MODEL_V2_INTERNAL_VALIDATION_SUPPORTED"
        if anchor_result["all_gates_passed"]
        else "MODEL_V2_INTERNAL_VALIDATION_NOT_SUPPORTED"
    )
    failed_gates = sorted(
        name for name, passed in anchor_result["gates"].items() if not passed
    )
    if not support_gain_retained:
        next_milestone = "MODEL_V2_SUPPORT_TARGET_TRADEOFF_REVIEW"
    elif not specificity_recovered:
        next_milestone = "MODEL_V2_ANCHOR_REFINEMENT_FAILURE_AUDIT"
    elif intervention_verdict != "V2_ANCHOR_REFINEMENT_EFFECTIVE":
        next_milestone = "MODEL_V2_ANCHOR_REFINEMENT_FAILURE_AUDIT"
    elif anchor_result["all_gates_passed"]:
        next_milestone = "MODEL_V2_GENERALIZATION_DEVELOPMENT_EVALUATION"
    elif set(failed_gates) <= {
        "overall_hazard_recall",
        "slip_hazard_recall",
    }:
        next_milestone = "MODEL_V2_CANDIDATE_READINESS_REVIEW"
    else:
        next_milestone = "MODEL_V2_ANCHOR_REFINEMENT_FAILURE_AUDIT"

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
        "v2_validation_result_sha256": sha256_file(
            result_paths["anchor_refined_v2"]
        ),
        "baseline_v2_on_v2_validation_sha256": sha256_file(
            result_paths["baseline_v2"]
        ),
        "rebalanced_v2_on_v2_validation_sha256": sha256_file(
            result_paths["rebalanced_v2"]
        ),
        "v1_on_v2_validation_sha256": sha256_file(result_paths["v1"]),
        "delayed_support_comparison_sha256": sha256_file(delayed_path),
        "speed_sand_comparison_sha256": sha256_file(speed_sand_path),
        "preservation_diagnostics_sha256": sha256_file(preservation_path),
        "four_model_comparison_sha256": sha256_file(comparison_path),
        "intervention_verdict": intervention_verdict,
        "internal_validation_verdict": internal_verdict,
        "generalization_validation_v2_inference": False,
        "generalization_holdout_guard_count": 0,
    }
    evaluation_freeze_path = artifact_path / "candidate_evaluation_freeze.json"
    _write_json(evaluation_freeze_path, evaluation_freeze)
    evaluation_freeze_sha = sha256_file(evaluation_freeze_path)
    result = {
        "training_verdict": "MODEL_V2_ANCHOR_REFINED_TRAINING_COMPLETE",
        "intervention_verdict": intervention_verdict,
        "internal_validation_verdict": internal_verdict,
        "recommended_next_milestone": next_milestone,
        "candidate_freeze_sha256": candidate_freeze_sha,
        "candidate_evaluation_freeze_sha256": evaluation_freeze_sha,
        "execution_config_sha256": config_sha,
        "training_record_sha256": sha256_file(training_record_path),
        "exposure_provenance_sha256": sha256_file(exposure_path),
        "hnm_provenance_sha256": sha256_file(hnm_path),
        "v2_validation_result_sha256": sha256_file(
            result_paths["anchor_refined_v2"]
        ),
        "baseline_v2_on_v2_validation_sha256": sha256_file(
            result_paths["baseline_v2"]
        ),
        "rebalanced_v2_on_v2_validation_sha256": sha256_file(
            result_paths["rebalanced_v2"]
        ),
        "v1_on_v2_validation_sha256": sha256_file(result_paths["v1"]),
        "four_model_comparison_sha256": sha256_file(comparison_path),
        "delayed_support_comparison_sha256": sha256_file(delayed_path),
        "speed_sand_comparison_sha256": sha256_file(speed_sand_path),
        "support_gain_retained": support_gain_retained,
        "specificity_recovered": specificity_recovered,
        "solved_behavior_retained": solved_behavior_retained,
        "failed_gates": failed_gates,
        "optimizer_steps": candidate.record["optimizer_steps"],
        "checkpoint_writes": candidate.record["checkpoint_writes"],
        "normalizer_fits": 0,
        "hnm_rounds": HNM_ROUNDS,
        "threshold_searches": 0,
        "persistence_searches": 0,
        "architecture_searches": 0,
        "seed_searches": 0,
        "monitor_searches": 0,
        "new_simulation_runs": 0,
        "v2_validation_optimizer_leakage": 0,
        "generalization_validation_training_leakage": 0,
        "holdout_training_leakage": 0,
        "generalization_validation_v2_inference": False,
        "unified_holdout_waveform_reopened": False,
        "unified_holdout_new_inference": False,
        "generalization_holdout_waveform_opened": False,
        "generalization_holdout_inference": False,
        "generalization_holdout_guard_count": 0,
    }
    result_path = artifact_path / "training_result.json"
    _write_json(result_path, result)
    if anchor_result["all_gates_passed"]:
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
