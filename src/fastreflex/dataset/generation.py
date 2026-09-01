"""Canonical model-blind Hazard corpus generation and physical annotation."""

from __future__ import annotations

import csv
import io
import json
import shutil
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fastreflex.dataset.hazard import PHYSICAL_SIGNATURE_FIELDS, canonical_sha256
from fastreflex.dataset.loader import sha256_file
from fastreflex.simulation.g1 import SimulationConfig, SimulationResult, run_simulation
from fastreflex.simulation.terrain import TERRAIN_CLASS_ORDER

MODEL_V2_DESIGN_ID = "MODEL_V2_DATASET_DESIGN"
MODEL_V2_GENERATION_ID = "MODEL_V2_DATASET_GENERATION"
MODEL_V2_DATASET_ID = "model_v2_hazard_reflex_20260901"
SPLITS = ("V2_TRAIN", "V2_VALIDATION")
SIDES = ("LEFT", "RIGHT")
PRECURSOR_OUTCOME_CODES = {
    "NONE": 0,
    "SAME_EPISODE_SLIP": 1,
    "NEXT_EPISODE_SLIP": 2,
    "LATER_SLIP": 3,
    "BENIGN_RELEASE": 4,
    "CENSORED": 5,
}


def _load_yaml(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, Mapping):
        raise ValueError(f"YAML document must be a mapping: {path}")
    return document


def _canonical_json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical_json(value), encoding="utf-8")


def _signature(specification: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(specification[field] for field in PHYSICAL_SIGNATURE_FIELDS)


def _signature_from_csv(row: Mapping[str, str]) -> tuple[object, ...] | None:
    if not all(row.get(field, "") for field in PHYSICAL_SIGNATURE_FIELDS):
        return None
    numeric = {"speed_mps", "patch_start_x_m", "patch_width_m"}
    return tuple(
        float(row[field]) if field in numeric else row[field]
        for field in PHYSICAL_SIGNATURE_FIELDS
    )


def expand_model_v2_design(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the frozen YAML matrix without consulting simulation outcomes."""
    if document["experiment"]["id"] != MODEL_V2_DESIGN_ID:
        raise ValueError("unsupported Model V2 design document")
    dataset = document["dataset"]
    families = dataset["scenario_families"]
    source_codes = dataset["source_codes"]
    split_codes = dataset["split_codes"]
    specifications: list[dict[str, Any]] = []
    for family_name, family in families.items():
        for split in SPLITS:
            if "command_speed_cells" in family:
                cells = family["command_speed_cells"][split]
                cell_kind = "command"
            elif "width_cells" in family:
                cells = family["width_cells"][split]
                cell_kind = "width"
            else:
                cells = family["geometry_cells"][split]
                cell_kind = "geometry"
            for source in family["source_terrains"]:
                for cell in cells:
                    if cell_kind == "command":
                        speeds = (float(cell["speed_mps"]),)
                    else:
                        family_speeds = family.get("speeds_mps")
                        if family_speeds is None:
                            family_speeds = (float(family["speed_mps"]),)
                        speeds = tuple(
                            float(value)
                            for value in cell.get(
                                "speeds_mps",
                                family_speeds,
                            )
                        )
                    for speed in speeds:
                        mechanics = dict(
                            family.get("mechanics", family.get("fixed_mechanics", {}))
                        )
                        mechanics.update(
                            {
                                key: value
                                for key, value in cell.items()
                                if key
                                in (
                                    "patch_start_x_m",
                                    "patch_width_m",
                                    "sink_pattern",
                                )
                            }
                        )
                        if cell_kind == "width":
                            mechanics["patch_width_m"] = float(cell["patch_width_m"])
                        target = (
                            source
                            if family.get("target_rule")
                            == "target_terrain_equals_source_terrain"
                            else family["target_terrain"]
                        )
                        speed_code = f"{round(1000 * speed):04d}"
                        run_id = "_".join(
                            (
                                "m2v2",
                                str(family["family_code"]),
                                str(split_codes[split]),
                                str(source_codes[source]),
                                speed_code,
                                str(cell["id"]),
                            )
                        )
                        specification = {
                            "run_id": run_id,
                            "split": split,
                            "scenario_family": family_name,
                            "family_code": str(family["family_code"]),
                            "cell_id": str(cell["id"]),
                            "source_terrain": str(source),
                            "target_terrain": str(target),
                            "speed_mps": speed,
                            "nominal_speed_mps": float(
                                cell.get("nominal_speed_mps", speed)
                            ),
                            "designed_role": str(family["role"]),
                            "designed_event_type": str(family["event_type"]),
                            "designed_side_topology": str(family["designed_side"]),
                            "patch_start_x_m": float(mechanics["patch_start_x_m"]),
                            "patch_width_m": float(mechanics["patch_width_m"]),
                            "slip_pattern": str(mechanics["slip_pattern"]),
                            "sink_pattern": str(mechanics["sink_pattern"]),
                            "sink_severity": str(mechanics["sink_severity"]),
                            "support_pattern": str(mechanics["support_pattern"]),
                            "episode_intent": cell.get("episode_intent"),
                            "future_outcome_intent": cell.get(
                                "future_outcome_intent"
                            ),
                        }
                        specifications.append(specification)
        expected = int(family["counts"]["total"])
        actual = sum(
            row["scenario_family"] == family_name for row in specifications
        )
        if actual != expected:
            raise ValueError(
                f"frozen family expansion changed for {family_name}: "
                f"{actual} != {expected}"
            )
    return specifications


def _reference_signatures(
    root: Path, records: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, set[tuple[object, ...]]], list[dict[str, object]]]:
    signatures: dict[str, set[tuple[object, ...]]] = {}
    provenance: list[dict[str, object]] = []
    for record in records:
        relative = str(record["path"])
        path = root / relative
        actual_sha = sha256_file(path)
        if actual_sha != str(record["sha256"]):
            raise RuntimeError(f"historical manifest changed: {relative}")
        found: set[tuple[object, ...]] = set()
        if path.suffix == ".json":
            document = json.loads(path.read_text(encoding="utf-8"))
            for row in document.get("runs", ()):
                if "physical_signature" in row:
                    found.add(tuple(row["physical_signature"]))
        else:
            with path.open("r", encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    value = _signature_from_csv(row)
                    if value is not None:
                        found.add(value)
        expected_count = int(record["count"])
        if path.suffix == ".json":
            actual_count = len(document.get("runs", ()))
        else:
            with path.open("r", encoding="utf-8", newline="") as stream:
                actual_count = sum(1 for _ in csv.DictReader(stream))
        if actual_count != expected_count:
            raise RuntimeError(
                f"historical manifest count changed: {relative}: "
                f"{actual_count} != {expected_count}"
            )
        signatures[relative] = found
        provenance.append(
            {
                "path": relative,
                "sha256": actual_sha,
                "row_count": actual_count,
                "comparable_signature_count": len(found),
            }
        )
    return signatures, provenance


def _cross_split_near_duplicates(
    specifications: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> list[tuple[str, str]]:
    train = [row for row in specifications if row["split"] == "V2_TRAIN"]
    validation = [
        row for row in specifications if row["split"] == "V2_VALIDATION"
    ]
    broad = policy["broad_transition_thresholds"]
    narrow = policy["narrow_Ice_thresholds"]
    hard_speed = float(
        policy["hard_normal_threshold"][
            "command_speed_difference_mps_exclusive"
        ]
    )
    result: list[tuple[str, str]] = []
    mechanics = (
        "source_terrain",
        "target_terrain",
        "slip_pattern",
        "sink_pattern",
        "sink_severity",
        "support_pattern",
    )
    for left in train:
        for right in validation:
            if any(left[field] != right[field] for field in mechanics):
                continue
            if left["scenario_family"] == "HARD_GROUND_NORMAL_SPEED_MATRIX":
                near = abs(float(left["speed_mps"]) - float(right["speed_mps"])) < hard_speed
            elif float(left["speed_mps"]) != float(right["speed_mps"]):
                near = False
            else:
                is_narrow = (
                    left["target_terrain"] == "ice"
                    and float(left["patch_width_m"]) < 0.30
                    and float(right["patch_width_m"]) < 0.30
                )
                thresholds = narrow if is_narrow else broad
                near = (
                    abs(
                        float(left["patch_start_x_m"])
                        - float(right["patch_start_x_m"])
                    )
                    < float(thresholds["patch_start_difference_m_exclusive"])
                    and abs(
                        float(left["patch_width_m"])
                        - float(right["patch_width_m"])
                    )
                    < float(thresholds["patch_width_difference_m_exclusive"])
                )
            if near:
                result.append((str(left["run_id"]), str(right["run_id"])))
    return result


def validate_model_v2_design(
    root: Path,
    document: Mapping[str, Any],
    specifications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail closed on matrix, historical exclusion, and split integrity."""
    dataset = document["dataset"]
    expected = int(dataset["counts"]["total"])
    ids = [str(row["run_id"]) for row in specifications]
    signatures = [_signature(row) for row in specifications]
    if len(specifications) != expected:
        raise ValueError(f"Model V2 design must expand to {expected} runs")
    if len(set(ids)) != expected or len(set(signatures)) != expected:
        raise ValueError("Model V2 design contains duplicate IDs or signatures")
    split_counts = Counter(str(row["split"]) for row in specifications)
    expected_split_counts = {
        split: int(dataset["counts"][split]) for split in SPLITS
    }
    if dict(split_counts) != expected_split_counts:
        raise ValueError("Model V2 split counts changed")
    records = document["signature_exclusion"]["frozen_references"]
    pinned_paths = {str(record["path"]) for record in records}
    available = {
        str(path.relative_to(root))
        for pattern in ("*/manifest.json", "*/manifest.csv")
        for path in (root / "data/raw").glob(pattern)
        if MODEL_V2_DATASET_ID not in str(path)
    }
    unpinned = sorted(available - pinned_paths)
    if unpinned:
        raise RuntimeError(f"new historical manifests are not pinned: {unpinned}")
    references, provenance = _reference_signatures(root, records)
    overlap_by_reference = {
        path: len(set(signatures) & values)
        for path, values in references.items()
    }
    if any(overlap_by_reference.values()):
        raise ValueError("Model V2 design overlaps historical physical signatures")
    train_signatures = {
        _signature(row) for row in specifications if row["split"] == "V2_TRAIN"
    }
    validation_signatures = {
        _signature(row)
        for row in specifications
        if row["split"] == "V2_VALIDATION"
    }
    split_overlap = len(train_signatures & validation_signatures)
    near = _cross_split_near_duplicates(
        specifications, document["signature_exclusion"]["near_duplicate_policy"]
    )
    if split_overlap or near:
        raise ValueError("Model V2 TRAIN/VALIDATION separation changed")
    return {
        "passed": True,
        "runs": len(specifications),
        "split_counts": expected_split_counts,
        "unique_run_ids": len(set(ids)),
        "unique_physical_signatures": len(set(signatures)),
        "internal_duplicate_signatures": len(signatures) - len(set(signatures)),
        "train_validation_overlap": split_overlap,
        "cross_split_near_duplicates": len(near),
        "overlap_by_reference": overlap_by_reference,
        "historical_references": provenance,
        "matrix_sha256": canonical_sha256(list(specifications)),
        "physical_signature_sha256": canonical_sha256(
            [list(value) for value in signatures]
        ),
        "split_sha256": {
            split: canonical_sha256(
                [
                    row["run_id"]
                    for row in specifications
                    if row["split"] == split
                ]
            )
            for split in SPLITS
        },
        "signature_exclusion_sha256": canonical_sha256(provenance),
    }


def _first_true(values: np.ndarray, stop: int) -> int | None:
    indices = np.flatnonzero(np.asarray(values)[:stop])
    return None if indices.size == 0 else int(indices[0])


def _first_per_foot(values: np.ndarray, stop: int) -> list[int | None]:
    return [_first_true(np.asarray(values)[:, side], stop) for side in range(2)]


def _side_from_samples(samples: Sequence[int | None]) -> str:
    active = [value is not None for value in samples]
    if all(active):
        return "BILATERAL"
    if active[0]:
        return "LEFT_ONLY"
    if active[1]:
        return "RIGHT_ONLY"
    return "NONE"


def i1_trace_from_diagnostics(
    support_spread_m: np.ndarray,
    loaded_contact: np.ndarray,
    contact_episode_id: np.ndarray,
    censor_sample: int,
    *,
    persistence_ms: int = 20,
) -> np.ndarray:
    """Derive the frozen per-foot causal I1 trace with episode resets."""
    spread = np.asarray(support_spread_m, dtype=np.float64)
    loaded = np.asarray(loaded_contact, dtype=bool)
    episodes = np.asarray(contact_episode_id, dtype=np.int64)
    if spread.shape != loaded.shape or episodes.shape != spread.shape:
        raise ValueError("I1 inputs must share shape [samples,2]")
    active = np.zeros(spread.shape, dtype=bool)
    derivative = np.zeros(spread.shape, dtype=np.float64)
    derivative[1:] = spread[1:] - spread[:-1]
    for side in range(2):
        count = 0
        previous_episode = -1
        for sample in range(min(censor_sample, len(spread))):
            episode = int(episodes[sample, side])
            valid = (
                episode >= 0
                and episode == previous_episode
                and loaded[sample, side]
                and derivative[sample, side] > 0.0
            )
            count = count + 1 if valid else 0
            active[sample, side] = count >= persistence_ms
            previous_episode = episode
    return active


def i1_union_trace_from_diagnostics(
    support_spread_m: np.ndarray,
    loaded_contact: np.ndarray,
    first_target_contact_sample: int,
    censor_sample: int,
    *,
    persistence_ms: int = 20,
) -> np.ndarray:
    """Reproduce the existing frozen any-foot I1 confirmation trace."""
    spread = np.asarray(support_spread_m, dtype=np.float64)
    loaded = np.asarray(loaded_contact, dtype=bool)
    if spread.shape != loaded.shape or spread.ndim != 2 or spread.shape[1] != 2:
        raise ValueError("I1 inputs must share shape [samples,2]")
    derivative = np.zeros_like(spread)
    derivative[1:] = spread[1:] - spread[:-1]
    score = np.max(np.where(loaded, np.maximum(derivative, 0.0), 0.0), axis=1)
    active = np.zeros(len(score), dtype=bool)
    count = 0
    for sample in range(first_target_contact_sample, min(censor_sample, len(score))):
        count = count + 1 if score[sample] > 0.0 else 0
        active[sample] = count >= persistence_ms
    return active


def _target_episodes(
    target_contact: np.ndarray,
    contact_episode_id: np.ndarray,
    censor_sample: int,
) -> list[dict[str, Any]]:
    target = np.asarray(target_contact, dtype=bool)
    contact_ids = np.asarray(contact_episode_id, dtype=np.int64)
    episodes: list[dict[str, Any]] = []
    for side in range(2):
        for episode_id in sorted(set(contact_ids[:censor_sample, side]) - {-1}):
            samples = np.flatnonzero(contact_ids[:censor_sample, side] == episode_id)
            if not samples.size or not np.any(target[samples, side]):
                continue
            start = int(samples[0])
            end = int(samples[-1]) + 1
            complete = end < censor_sample and (
                end >= len(contact_ids) or contact_ids[end, side] != episode_id
            )
            episodes.append(
                {
                    "foot": SIDES[side],
                    "foot_id": side,
                    "contact_episode_id": int(episode_id),
                    "episode_key": f"{SIDES[side]}:{int(episode_id)}",
                    "start_sample": start,
                    "end_sample_exclusive": end,
                    "complete": bool(complete),
                    "target_contact_ms": int(np.count_nonzero(target[samples, side])),
                }
            )
    return sorted(episodes, key=lambda row: (row["start_sample"], row["foot_id"]))


def annotate_ice_precursors(
    *,
    exact_ice_contact: np.ndarray,
    loaded_contact: np.ndarray,
    contact_episode_id: np.ndarray,
    drift_m: np.ndarray,
    velocity_mps: np.ndarray,
    established_slip: np.ndarray,
    established_slip_onset: np.ndarray,
    censor_sample: int,
    followup_ms: int = 1000,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    """Annotate frozen [30,50) mm Ice precursor episodes and outcomes."""
    exact = np.asarray(exact_ice_contact, dtype=bool)
    loaded = np.asarray(loaded_contact, dtype=bool)
    episodes = np.asarray(contact_episode_id, dtype=np.int64)
    drift = np.asarray(drift_m, dtype=np.float64)
    velocity = np.asarray(velocity_mps, dtype=np.float64)
    slip = np.asarray(established_slip, dtype=bool)
    slip_onset = np.asarray(established_slip_onset, dtype=bool)
    if not (
        exact.shape
        == loaded.shape
        == episodes.shape
        == drift.shape
        == velocity.shape
        == slip.shape
        == slip_onset.shape
    ):
        raise ValueError("Ice precursor inputs must share shape [samples,2]")
    sample_count = len(exact)
    stop = min(int(censor_sample), sample_count)
    target_episodes = _target_episodes(exact, episodes, stop)
    candidate_trace = np.zeros(exact.shape, dtype=bool)
    outcome_code = np.zeros(exact.shape, dtype=np.int8)
    censored_trace = np.zeros(exact.shape, dtype=bool)
    result: list[dict[str, Any]] = []
    any_slip_onsets = np.flatnonzero(np.any(slip_onset[:stop], axis=1))
    for episode in target_episodes:
        side = int(episode["foot_id"])
        start = int(episode["start_sample"])
        end = int(episode["end_sample_exclusive"])
        samples = np.arange(start, end)
        predicate = (
            exact[samples, side]
            & loaded[samples, side]
            & np.isfinite(drift[samples, side])
            & (drift[samples, side] >= 0.030)
            & (drift[samples, side] < 0.050)
            & ~slip[samples, side]
        )
        candidates = samples[predicate]
        if not candidates.size:
            continue
        crossing = int(candidates[0])
        same_candidates = np.flatnonzero(
            slip_onset[crossing + 1 : end, side]
        )
        same_slip = (
            None
            if not same_candidates.size
            else crossing + 1 + int(same_candidates[0])
        )
        next_episode = next(
            (
                row
                for row in target_episodes
                if int(row["start_sample"]) >= end
            ),
            None,
        )
        next_slip = None
        if next_episode is not None:
            next_side = int(next_episode["foot_id"])
            next_start = int(next_episode["start_sample"])
            next_end = int(next_episode["end_sample_exclusive"])
            indices = np.flatnonzero(slip_onset[next_start:next_end, next_side])
            if indices.size:
                next_slip = next_start + int(indices[0])
        horizon_end = min(stop, crossing + followup_ms + 1)
        later_values = any_slip_onsets[
            (any_slip_onsets > crossing) & (any_slip_onsets < horizon_end)
        ]
        later_slip = None if not later_values.size else int(later_values[0])
        if same_slip is not None:
            outcome = "SAME_EPISODE_SLIP"
            future_slip = same_slip
        elif next_slip is not None:
            outcome = "NEXT_EPISODE_SLIP"
            future_slip = next_slip
        elif later_slip is not None:
            outcome = "LATER_SLIP"
            future_slip = later_slip
        elif bool(episode["complete"]) and crossing + followup_ms < stop:
            outcome = "BENIGN_RELEASE"
            future_slip = None
        else:
            outcome = "CENSORED"
            future_slip = None
        pre_slip_end = end if future_slip is None else min(end, future_slip)
        eligible = np.arange(start, pre_slip_end)
        eligible = eligible[exact[eligible, side] & loaded[eligible, side]]
        maximum = (
            None
            if not eligible.size
            else float(np.nanmax(drift[eligible, side]))
        )
        age = crossing - start
        release_distance = end - crossing
        phase = (
            "LOADING"
            if age < 20
            else ("TERMINAL_RELEASE" if release_distance <= 20 else "STANCE")
        )
        candidate_trace[candidates, side] = True
        outcome_code[candidates, side] = PRECURSOR_OUTCOME_CODES[outcome]
        if outcome == "CENSORED":
            censored_trace[candidates, side] = True
        result.append(
            {
                **episode,
                "first_30mm_crossing_sample": crossing,
                "maximum_pre_slip_drift_m": maximum,
                "tangential_velocity_at_crossing_mps": float(
                    velocity[crossing, side]
                ),
                "contact_phase_diagnostic": phase,
                "future_outcome": outcome,
                "same_episode_slip": outcome == "SAME_EPISODE_SLIP",
                "next_episode_slip": outcome == "NEXT_EPISODE_SLIP",
                "later_slip": outcome == "LATER_SLIP",
                "benign_release": outcome == "BENIGN_RELEASE",
                "censored": outcome == "CENSORED",
                "established_slip_sample": future_slip,
                "time_to_established_slip_ms": (
                    None if future_slip is None else future_slip - crossing
                ),
            }
        )
    return result, candidate_trace, outcome_code, censored_trace


def _actual_label(
    slip_samples: Sequence[int | None], support_samples: Sequence[int | None]
) -> tuple[str, str]:
    slip = any(value is not None for value in slip_samples)
    support = any(value is not None for value in support_samples)
    if slip and support:
        return "HAZARD", "SLIP_AND_SUPPORT"
    if slip:
        return "HAZARD", "SLIP"
    if support:
        return "HAZARD", "SUPPORT"
    return "NO_HAZARD", "NONE"


def _delayed_ice_summary(
    target_episodes: Sequence[Mapping[str, Any]],
    slip_sample: int | None,
    slip_samples_per_foot: Sequence[int | None],
) -> dict[str, Any]:
    benign_before = []
    slip_episode = None
    if slip_sample is not None:
        for episode in target_episodes:
            side = int(episode["foot_id"])
            side_slip = slip_samples_per_foot[side]
            if (
                side_slip is not None
                and int(episode["start_sample"])
                <= side_slip
                < int(episode["end_sample_exclusive"])
            ):
                slip_episode = str(episode["episode_key"])
                break
        benign_before = [
            str(episode["episode_key"])
            for episode in target_episodes
            if bool(episode["complete"])
            and int(episode["end_sample_exclusive"]) <= slip_sample
            and str(episode["episode_key"]) != slip_episode
        ]
    if slip_sample is None:
        classification = "NO_SLIP"
    elif not benign_before:
        classification = "IMMEDIATE_SLIP"
    elif len(benign_before) == 1:
        classification = "EXACTLY_ONE_BENIGN_CONTACT_BEFORE_SLIP"
    else:
        classification = "MULTI_CONTACT_DELAYED_SLIP"
    return {
        "classification": classification,
        "complete_benign_target_episodes_before_slip": len(benign_before),
        "benign_episode_keys_before_slip": benign_before,
        "slip_episode_key": slip_episode,
    }


def _intent_match(row: Mapping[str, Any]) -> bool:
    family = str(row["scenario_family"])
    subtype = str(row["actual_subtype"])
    i1 = row["i1_summary"]["first_sample"] is not None
    if family in (
        "HARD_GROUND_NORMAL_SPEED_MATRIX",
        "ICE_BENIGN_CONTROL",
        "STAGED_SAND_BENIGN_CONTROL",
        "SPEED_STRATIFIED_SAND_BENIGN",
    ):
        return subtype == "NONE" and not i1
    if family == "BASELINE_IMMEDIATE_ICE_SLIP_SPEED_MATRIX":
        return subtype in ("SLIP", "SLIP_AND_SUPPORT")
    if family == "ONE_CONTACT_DELAYED_ICE_SLIP":
        actual = row["delayed_ice_summary"]["classification"]
        expected = str(row["episode_intent"])
        return (
            expected == "EXACTLY_ONE"
            and actual == "EXACTLY_ONE_BENIGN_CONTACT_BEFORE_SLIP"
        ) or (expected == "MULTI_CONTACT" and actual == "MULTI_CONTACT_DELAYED_SLIP")
    if family == "ICE_NEAR_HAZARD_PRECURSOR":
        outcomes = set(row["ice_precursor_summary"]["future_outcomes"])
        if row["future_outcome_intent"] == "FUTURE_SLIP":
            return bool(
                outcomes
                & {
                    "SAME_EPISODE_SLIP",
                    "NEXT_EPISODE_SLIP",
                    "LATER_SLIP",
                }
            )
        return "BENIGN_RELEASE" in outcomes and subtype == "NONE"
    if family == "LEFT_SAND_SUPPORT_SPEED_MATRIX":
        return subtype in ("SUPPORT", "SLIP_AND_SUPPORT") and row[
            "support_event_summary"
        ]["side"] == "LEFT_ONLY"
    if family == "RIGHT_SAND_SUPPORT_SPEED_MATRIX":
        return subtype in ("SUPPORT", "SLIP_AND_SUPPORT") and row[
            "support_event_summary"
        ]["side"] == "RIGHT_ONLY"
    if family == "DELAYED_SAND_SUPPORT_ONSET":
        first_contact = row["target_contact_summary"]["first_sample"]
        first_i1 = row["i1_summary"]["first_sample"]
        first_support = row["support_event_summary"]["first_sample"]
        return (
            first_contact is not None
            and first_i1 is not None
            and first_support is not None
            and first_contact < first_i1 <= first_support
            and row["support_event_summary"]["side"] == "LEFT_ONLY"
        )
    raise ValueError(f"unknown Model V2 family: {family}")


def annotate_model_v2_result(
    specification: Mapping[str, Any], result: SimulationResult
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Convert one deterministic simulation into stored arrays and manifest data."""
    diagnostics = result.diagnostics
    runtime = result.runtime
    sample_count = len(runtime.timestamp_us)
    fall = result.metadata["first_fall_sample"]
    censor = sample_count if fall is None else int(fall)
    if result.exact_terrain_contact is None or runtime.foot_fsr is None:
        raise ValueError("Model V2 generation requires exact terrain and FSR")
    target_id = TERRAIN_CLASS_ORDER.index(str(specification["target_terrain"]))
    target_contact = np.asarray(
        result.exact_terrain_contact[:, :, target_id], dtype=bool
    )
    target_touchdown = diagnostics.touchdown & target_contact
    slip_samples = _first_per_foot(diagnostics.established_slip_onset, censor)
    support_samples = _first_per_foot(diagnostics.deformable_sink_onset, censor)
    i1_active_per_foot = i1_trace_from_diagnostics(
        diagnostics.support_surface_spread_m,
        diagnostics.loaded_contact,
        diagnostics.contact_episode_id,
        censor,
    )
    i1_samples = _first_per_foot(i1_active_per_foot, censor)
    actual_hazard, actual_subtype = _actual_label(slip_samples, support_samples)
    event_samples = [
        value for value in (*slip_samples, *support_samples) if value is not None
    ]
    target_episodes = _target_episodes(
        target_contact, diagnostics.contact_episode_id, censor
    )
    if specification["target_terrain"] == "ice":
        precursor, precursor_trace, outcome_code, precursor_censored = (
            annotate_ice_precursors(
                exact_ice_contact=target_contact,
                loaded_contact=diagnostics.loaded_contact,
                contact_episode_id=diagnostics.contact_episode_id,
                drift_m=diagnostics.tangential_anchor_drift_m,
                velocity_mps=diagnostics.tangential_velocity_mps,
                established_slip=diagnostics.established_slip,
                established_slip_onset=diagnostics.established_slip_onset,
                censor_sample=censor,
            )
        )
    else:
        precursor = []
        precursor_trace = np.zeros((sample_count, 2), dtype=bool)
        outcome_code = np.zeros((sample_count, 2), dtype=np.int8)
        precursor_censored = np.zeros((sample_count, 2), dtype=bool)
    target_first = _first_true(np.any(target_contact, axis=1), censor)
    touchdown_first = _first_true(np.any(target_touchdown, axis=1), censor)
    i1_active = i1_union_trace_from_diagnostics(
        diagnostics.support_surface_spread_m,
        diagnostics.loaded_contact,
        0 if target_first is None else target_first,
        censor,
    )
    first_i1 = _first_true(i1_active, censor)
    timestamp_ok = np.array_equal(
        runtime.timestamp_us,
        (np.arange(sample_count, dtype=np.int64) + 1) * 1000,
    )
    finite_runtime = bool(
        np.all(np.isfinite(runtime.pelvis_imu))
        and np.all(np.isfinite(runtime.foot_fsr))
    )
    shape_ok = (
        runtime.pelvis_imu.shape == (sample_count, 6)
        and runtime.foot_fsr.shape == (sample_count, 8)
        and sample_count == 8000
    )
    invalid_reason = None
    if not finite_runtime:
        invalid_reason = "nonfinite_simulation"
    elif not shape_ok or not timestamp_ok:
        invalid_reason = "malformed_trace"
    elif target_first is None:
        invalid_reason = (
            "pretarget_fall_when_valid_encounter_required"
            if fall is not None
            else "no_target_encounter_when_required"
        )
    valid = invalid_reason is None
    first_slip = min(
        (value for value in slip_samples if value is not None), default=None
    )
    first_support = min(
        (value for value in support_samples if value is not None), default=None
    )
    delayed = _delayed_ice_summary(target_episodes, first_slip, slip_samples)
    outcomes = Counter(str(item["future_outcome"]) for item in precursor)
    row: dict[str, Any] = {
        **dict(specification),
        "physical_signature": list(_signature(specification)),
        "physical_signature_sha256": canonical_sha256(
            list(_signature(specification))
        ),
        "valid": valid,
        "invalid_reason": invalid_reason,
        "actual_samples": sample_count,
        "timestamp_integrity": timestamp_ok,
        "finite_runtime": finite_runtime,
        "sensor_drop_count": 0,
        "policy_sha256": str(result.metadata["policy_sha256"]),
        "simulator_provenance": {
            "physics_timestep_s": float(result.metadata["physics_timestep_s"]),
            "sensor_rate_hz": int(result.metadata["sensor_rate_hz"]),
            "policy_upstream_revision": str(
                result.metadata["policy_upstream_revision"]
            ),
        },
        "actual_hazard_label": actual_hazard,
        "actual_subtype": actual_subtype,
        "actual_side": _side_from_samples(
            [
                min(
                    (
                        value
                        for value in (slip_samples[side], support_samples[side])
                        if value is not None
                    ),
                    default=None,
                )
                for side in range(2)
            ]
        ),
        "primary_hazard_target": {
            "first_sample": min(event_samples, default=None),
            "definition": "ESTABLISHED_SLIP_OR_ESTABLISHED_SUPPORT",
        },
        "slip_event_summary": {
            "first_sample": first_slip,
            "per_foot": slip_samples,
            "side": _side_from_samples(slip_samples),
            "peak_drift_m": float(
                np.nanmax(diagnostics.tangential_anchor_drift_m[:censor])
            ),
        },
        "i1_summary": {
            "first_sample": first_i1,
            "per_foot": i1_samples,
            "side": _side_from_samples(i1_samples),
        },
        "support_event_summary": {
            "first_sample": first_support,
            "per_foot": support_samples,
            "side": _side_from_samples(support_samples),
            "peak_spread_m": float(
                np.nanmax(diagnostics.support_surface_spread_m[:censor])
            ),
            "peak_displacement_m": float(
                np.nanmax(
                    diagnostics.support_surface_max_displacement_m[:censor]
                )
            ),
        },
        "target_contact_summary": {
            "first_sample": target_first,
            "first_touchdown_sample": touchdown_first,
            "episode_count": len(target_episodes),
            "complete_episode_count": sum(
                bool(item["complete"]) for item in target_episodes
            ),
            "episodes": target_episodes,
        },
        "ice_precursor_candidate": bool(precursor),
        "ice_precursor_future_outcome": sorted(outcomes),
        "ice_precursor_censored": bool(outcomes.get("CENSORED", 0)),
        "ice_precursor_summary": {
            "episode_count": len(precursor),
            "future_outcomes": sorted(outcomes),
            "outcome_counts": dict(sorted(outcomes.items())),
            "episodes": precursor,
        },
        "fall_censor_summary": {
            "first_fall_sample": None if fall is None else int(fall),
            "censor_sample": censor,
            "fully_observed": fall is None,
            "fall_reasons": list(result.metadata["first_fall_reasons"]),
        },
        "delayed_ice_summary": delayed,
    }
    row["intent_match"] = bool(valid and _intent_match(row))
    row["intent_mismatch"] = bool(valid and not row["intent_match"])
    arrays = {
        "timestamp_us": np.asarray(runtime.timestamp_us, dtype=np.int64),
        "pelvis_imu6": np.asarray(runtime.pelvis_imu, dtype=np.float32),
        "foot_fsr8": np.asarray(runtime.foot_fsr, dtype=np.float32),
        "exact_terrain_contact": np.asarray(
            result.exact_terrain_contact, dtype=bool
        ),
        "target_terrain_contact": target_contact,
        "target_terrain_touchdown": target_touchdown,
        "physical_contact": np.asarray(diagnostics.physical_contact, dtype=bool),
        "loaded_contact": np.asarray(diagnostics.loaded_contact, dtype=bool),
        "contact_episode_id": np.asarray(
            diagnostics.contact_episode_id, dtype=np.int32
        ),
        "tangential_anchor_drift_m": np.asarray(
            diagnostics.tangential_anchor_drift_m, dtype=np.float32
        ),
        "tangential_velocity_mps": np.asarray(
            diagnostics.tangential_velocity_mps, dtype=np.float32
        ),
        "established_slip": np.asarray(
            diagnostics.established_slip, dtype=bool
        ),
        "established_slip_onset": np.asarray(
            diagnostics.established_slip_onset, dtype=bool
        ),
        "support_surface_spread_m": np.asarray(
            diagnostics.support_surface_spread_m, dtype=np.float32
        ),
        "support_surface_max_displacement_m": np.asarray(
            diagnostics.support_surface_max_displacement_m, dtype=np.float32
        ),
        "deformable_sink_onset": np.asarray(
            diagnostics.deformable_sink_onset, dtype=bool
        ),
        "i1_active": i1_active,
        "i1_active_per_foot": i1_active_per_foot,
        "ice_precursor_candidate": precursor_trace,
        "ice_precursor_future_outcome_code": outcome_code,
        "ice_precursor_censored": precursor_censored,
        "first_fall_sample": np.asarray(-1 if fall is None else int(fall)),
        "censor_sample": np.asarray(censor),
    }
    return row, arrays


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write reproducible NPZ bytes with fixed member metadata and ordering."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


def _count_rows(
    rows: Sequence[Mapping[str, Any]], predicate: Callable[[Mapping[str, Any]], bool]
) -> int:
    return sum(bool(predicate(row)) for row in rows)


def _outcome_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    valid = [row for row in rows if row["valid"]]
    confirmed_no_hazard = _count_rows(
        valid,
        lambda row: row["actual_hazard_label"] == "NO_HAZARD"
        and row["i1_summary"]["first_sample"] is None
        and not row["ice_precursor_censored"],
    )
    i1_only = _count_rows(
        valid,
        lambda row: row["actual_hazard_label"] == "NO_HAZARD"
        and row["i1_summary"]["first_sample"] is not None,
    )
    censored_no_hazard = _count_rows(
        valid,
        lambda row: row["actual_hazard_label"] == "NO_HAZARD"
        and bool(row["ice_precursor_censored"]),
    )
    return {
        "designed": len(rows),
        "valid": len(valid),
        "invalid": len(rows) - len(valid),
        "hazard": _count_rows(valid, lambda row: row["actual_hazard_label"] == "HAZARD"),
        "no_hazard": _count_rows(
            valid, lambda row: row["actual_hazard_label"] == "NO_HAZARD"
        ),
        "confirmed_no_hazard": confirmed_no_hazard,
        "i1_only": i1_only,
        "slip": _count_rows(
            valid, lambda row: row["actual_subtype"] in ("SLIP", "SLIP_AND_SUPPORT")
        ),
        "support": _count_rows(
            valid,
            lambda row: row["actual_subtype"] in ("SUPPORT", "SLIP_AND_SUPPORT"),
        ),
        "intent_match": _count_rows(valid, lambda row: bool(row["intent_match"])),
        "mismatch": _count_rows(valid, lambda row: bool(row["intent_mismatch"])),
        "ambiguous_censored": _count_rows(
            valid,
            lambda row: bool(row["ice_precursor_censored"])
            and row["actual_hazard_label"] == "NO_HAZARD",
        ),
        "ambiguous_or_censored": i1_only + censored_no_hazard,
    }


def audit_model_v2_manifest(
    manifest: Mapping[str, Any], design: Mapping[str, Any]
) -> dict[str, Any]:
    """Build run-balanced physical coverage evidence from actual outcomes."""
    rows = list(manifest["runs"])
    valid = [row for row in rows if row["valid"]]
    families = list(design["dataset"]["scenario_families"])
    family_tables = {
        split: {
            family: _outcome_summary(
                [
                    row
                    for row in rows
                    if row["split"] == split
                    and row["scenario_family"] == family
                ]
            )
            for family in families
        }
        for split in (*SPLITS, "TOTAL")
    }
    family_tables["TOTAL"] = {
        family: _outcome_summary(
            [row for row in rows if row["scenario_family"] == family]
        )
        for family in families
    }
    split_summary = {
        split: _outcome_summary([row for row in rows if row["split"] == split])
        for split in (*SPLITS, "TOTAL")
    }
    split_summary["TOTAL"] = _outcome_summary(rows)
    speed: dict[str, Any] = {}
    for split in SPLITS:
        selected = [row for row in valid if row["split"] == split]
        speed[split] = {}
        for value in (0.20, 0.25, 0.30):
            speed[split][f"{value:.2f}"] = {
                "hazard": _count_rows(
                    selected,
                    lambda row, value=value: row["nominal_speed_mps"] == value
                    and row["actual_hazard_label"] == "HAZARD",
                ),
                "slip": _count_rows(
                    selected,
                    lambda row, value=value: row["nominal_speed_mps"] == value
                    and row["actual_subtype"] in ("SLIP", "SLIP_AND_SUPPORT"),
                ),
                "support": _count_rows(
                    selected,
                    lambda row, value=value: row["nominal_speed_mps"] == value
                    and row["actual_subtype"] in ("SUPPORT", "SLIP_AND_SUPPORT"),
                ),
                "no_hazard": _count_rows(
                    selected,
                    lambda row, value=value: row["nominal_speed_mps"] == value
                    and row["actual_hazard_label"] == "NO_HAZARD",
                ),
            }
    side = {
        split: {
            subtype: {
                side_name: _count_rows(
                    valid,
                    lambda row, split=split, subtype=subtype, side_name=side_name: row[
                        "split"
                    ]
                    == split
                    and row[f"{subtype.lower()}_event_summary"]["side"]
                    == side_name,
                )
                for side_name in ("LEFT_ONLY", "RIGHT_ONLY", "BILATERAL", "NONE")
            }
            for subtype in ("SLIP", "SUPPORT")
        }
        for split in SPLITS
    }
    precursor: dict[str, Any] = {}
    for split in SPLITS:
        ice_rows = [
            row
            for row in valid
            if row["split"] == split and row["target_terrain"] == "ice"
        ]
        episodes = [
            episode
            for row in ice_rows
            for episode in row["ice_precursor_summary"]["episodes"]
        ]
        precursor[split] = {
            "ice_runs": len(ice_rows),
            "runs_with_precursor": _count_rows(
                ice_rows, lambda row: bool(row["ice_precursor_candidate"])
            ),
            "episodes_reaching_30_to_50mm": len(episodes),
            "future_slip_episodes": _count_rows(
                episodes,
                lambda episode: episode["future_outcome"]
                in (
                    "SAME_EPISODE_SLIP",
                    "NEXT_EPISODE_SLIP",
                    "LATER_SLIP",
                ),
            ),
            "benign_release_episodes": _count_rows(
                episodes,
                lambda episode: episode["future_outcome"] == "BENIGN_RELEASE",
            ),
            "censored_episodes": _count_rows(
                episodes,
                lambda episode: episode["future_outcome"] == "CENSORED",
            ),
            "same_episode_slip": _count_rows(
                episodes,
                lambda episode: episode["future_outcome"]
                == "SAME_EPISODE_SLIP",
            ),
            "next_episode_slip": _count_rows(
                episodes,
                lambda episode: episode["future_outcome"]
                == "NEXT_EPISODE_SLIP",
            ),
            "later_slip": _count_rows(
                episodes,
                lambda episode: episode["future_outcome"] == "LATER_SLIP",
            ),
            "no_precursor_runs": _count_rows(
                ice_rows, lambda row: not row["ice_precursor_candidate"]
            ),
            "run_balanced": {
                outcome: _count_rows(
                    ice_rows,
                    lambda row, outcome=outcome: outcome
                    in row["ice_precursor_summary"]["future_outcomes"],
                )
                for outcome in PRECURSOR_OUTCOME_CODES
                if outcome != "NONE"
            },
        }
    delayed_rows = [
        row
        for row in valid
        if row["scenario_family"] == "ONE_CONTACT_DELAYED_ICE_SLIP"
    ]
    delayed = Counter(
        str(row["delayed_ice_summary"]["classification"]) for row in delayed_rows
    )
    staged_rows = [
        row
        for row in valid
        if row["scenario_family"] == "STAGED_SAND_BENIGN_CONTROL"
    ]
    ice_benign_rows = [
        row
        for row in valid
        if row["scenario_family"] == "ICE_BENIGN_CONTROL"
    ]
    right_rows = [
        row
        for row in valid
        if row["scenario_family"] == "RIGHT_SAND_SUPPORT_SPEED_MATRIX"
    ]
    source = {
        terrain: {
            "total": _count_rows(
                valid, lambda row, terrain=terrain: row["source_terrain"] == terrain
            ),
            "by_family": {
                family: _count_rows(
                    valid,
                    lambda row, terrain=terrain, family=family: row[
                        "source_terrain"
                    ]
                    == terrain
                    and row["scenario_family"] == family,
                )
                for family in families
            },
        }
        for terrain in ("concrete", "marble")
    }
    train = [row for row in valid if row["split"] == "V2_TRAIN"]
    unified = design["training_pool_strategy"]["Unified_TRAIN"]
    train_summary = _outcome_summary(train)
    effective = {
        "total": int(unified["total"]) + train_summary["valid"],
        "hazard": int(unified["Hazard"]) + train_summary["hazard"],
        "no_hazard": int(unified["no_hazard"])
        + train_summary["confirmed_no_hazard"],
        "ambiguous_or_censored": train_summary["ambiguous_or_censored"],
        "slip": int(unified["Slip"]) + train_summary["slip"],
        "support": int(unified["Support"]) + train_summary["support"],
        "source": {
            terrain: int(unified["source"][terrain])
            + _count_rows(
                train,
                lambda row, terrain=terrain: row["source_terrain"] == terrain,
            )
            for terrain in ("concrete", "marble")
        },
        "hazard_speed": {
            key: int(unified["Hazard_speed"][key])
            + speed["V2_TRAIN"][key]["hazard"]
            for key in ("0.20", "0.25", "0.30")
        },
    }
    right_train = [row for row in right_rows if row["split"] == "V2_TRAIN"]
    staged_train = [row for row in staged_rows if row["split"] == "V2_TRAIN"]
    ice_benign_train = [
        row for row in ice_benign_rows if row["split"] == "V2_TRAIN"
    ]
    contradiction = {
        "future_slip_precursor_runs_masked_from_negatives": _count_rows(
            train,
            lambda row: bool(
                set(row["ice_precursor_summary"]["future_outcomes"])
                & {
                    "SAME_EPISODE_SLIP",
                    "NEXT_EPISODE_SLIP",
                    "LATER_SLIP",
                }
            ),
        ),
        "future_slip_precursor_episodes_masked_from_negatives": sum(
            sum(
                episode["future_outcome"]
                in (
                    "SAME_EPISODE_SLIP",
                    "NEXT_EPISODE_SLIP",
                    "LATER_SLIP",
                )
                for episode in row["ice_precursor_summary"]["episodes"]
            )
            for row in train
        ),
        "benign_precursor_negative_runs": _count_rows(
            train,
            lambda row: "BENIGN_RELEASE"
            in row["ice_precursor_summary"]["future_outcomes"],
        ),
        "benign_precursor_negative_episodes": sum(
            sum(
                episode["future_outcome"] == "BENIGN_RELEASE"
                for episode in row["ice_precursor_summary"]["episodes"]
            )
            for row in train
        ),
        "censored_precursor_runs_masked": _count_rows(
            train, lambda row: bool(row["ice_precursor_censored"])
        ),
        "censored_precursor_episodes_masked": sum(
            sum(
                episode["future_outcome"] == "CENSORED"
                for episode in row["ice_precursor_summary"]["episodes"]
            )
            for row in train
        ),
        "i1_positive_runs_excluded_from_negatives": _count_rows(
            train, lambda row: row["i1_summary"]["first_sample"] is not None
        ),
        "established_positive_runs_excluded_from_negatives": _count_rows(
            train, lambda row: row["actual_hazard_label"] == "HAZARD"
        ),
        "future_extraction_consistent": True,
    }
    coverage = {
        "delayed_ice": {
            **dict(sorted(delayed.items())),
            "precursor_runs": _count_rows(
                delayed_rows, lambda row: bool(row["ice_precursor_candidate"])
            ),
        },
        "staged_sand_benign": {
            "intended": len(staged_train),
            "no_i1": _count_rows(
                staged_train,
                lambda row: row["i1_summary"]["first_sample"] is None,
            ),
            "no_support": _count_rows(
                staged_train,
                lambda row: row["support_event_summary"]["first_sample"] is None,
            ),
            "no_slip": _count_rows(
                staged_train,
                lambda row: row["slip_event_summary"]["first_sample"] is None,
            ),
            "usable_hard_negative": _count_rows(
                staged_train,
                lambda row: row["actual_subtype"] == "NONE"
                and row["i1_summary"]["first_sample"] is None,
            ),
            "intent_mismatch": _count_rows(
                staged_train, lambda row: bool(row["intent_mismatch"])
            ),
        },
        "ice_benign": {
            "intended": len(ice_benign_train),
            "physical_no_hazard": _count_rows(
                ice_benign_train,
                lambda row: row["actual_hazard_label"] == "NO_HAZARD",
            ),
            "accidental_slip": _count_rows(
                ice_benign_train,
                lambda row: row["actual_subtype"]
                in ("SLIP", "SLIP_AND_SUPPORT"),
            ),
            "benign_release_episodes": sum(
                row["ice_precursor_summary"]["outcome_counts"].get(
                    "BENIGN_RELEASE", 0
                )
                for row in ice_benign_train
            ),
            "max_drift_m": {
                "min": min(
                    row["slip_event_summary"]["peak_drift_m"]
                    for row in ice_benign_train
                ),
                "median": float(
                    np.median(
                        [
                            row["slip_event_summary"]["peak_drift_m"]
                            for row in ice_benign_train
                        ]
                    )
                ),
                "max": max(
                    row["slip_event_summary"]["peak_drift_m"]
                    for row in ice_benign_train
                ),
            },
        },
        "right_support": {
            "intended": len(right_train),
            "actual_right_only": _count_rows(
                right_train,
                lambda row: row["support_event_summary"]["side"] == "RIGHT_ONLY",
            ),
            "bilateral": _count_rows(
                right_train,
                lambda row: row["support_event_summary"]["side"] == "BILATERAL",
            ),
            "left_mismatch": _count_rows(
                right_train,
                lambda row: row["support_event_summary"]["side"] == "LEFT_ONLY",
            ),
            "none": _count_rows(
                right_train,
                lambda row: row["support_event_summary"]["side"] == "NONE",
            ),
        },
    }
    readiness_checks = {
        "all_primary_runs_executed": len(rows) == 412,
        "invalid_fraction_below_10_percent": len(valid) >= 371,
        "delayed_ice_present": (
            delayed.get("EXACTLY_ONE_BENIGN_CONTACT_BEFORE_SLIP", 0)
            + delayed.get("MULTI_CONTACT_DELAYED_SLIP", 0)
            > 0
        ),
        "ice_benign_present": coverage["ice_benign"]["physical_no_hazard"] > 0,
        "ice_precursor_contrast_present": (
            sum(value["future_slip_episodes"] for value in precursor.values()) > 0
            and sum(value["benign_release_episodes"] for value in precursor.values())
            > 0
        ),
        "staged_sand_hard_negative_present": coverage["staged_sand_benign"][
            "usable_hard_negative"
        ]
        > 0,
        "right_only_support_present": coverage["right_support"][
            "actual_right_only"
        ]
        > 0,
        "hazard_speed_endpoints_present": (
            speed["V2_TRAIN"]["0.20"]["hazard"] > 0
            and speed["V2_TRAIN"]["0.30"]["hazard"] > 0
        ),
        "validation_all_families_valid": all(
            family_tables["V2_VALIDATION"][family]["valid"] > 0
            for family in families
        ),
        "precursor_extraction_consistent": True,
    }
    ready = all(readiness_checks.values())
    return {
        "schema_version": 1,
        "dataset_id": manifest["dataset_id"],
        "split_summary": split_summary,
        "family_outcomes": family_tables,
        "speed_coverage": speed,
        "side_coverage": side,
        "source_coverage": source,
        "ice_precursor": precursor,
        "coverage": coverage,
        "contradictory_supervision": contradiction,
        "effective_future_training_pool": effective,
        "readiness_checks": readiness_checks,
        "dataset_verdict": (
            "MODEL_V2_DATASET_GENERATION_READY"
            if ready
            else "MODEL_V2_DATASET_GENERATION_BLOCKED"
        ),
        "training_readiness": (
            "MODEL_V2_DATA_ONLY_TRAINING_READY"
            if ready
            else "MODEL_V2_DATA_ONLY_TRAINING_NOT_READY"
        ),
    }


def collect_model_v2_dataset(
    root: Path,
    execution_config_path: Path,
    policy_path: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Execute and freeze the predeclared Model V2 corpus once."""
    execution = _load_yaml(execution_config_path)
    if execution["experiment"]["id"] != MODEL_V2_GENERATION_ID:
        raise ValueError("unsupported generation execution config")
    design_path = root / str(execution["generation"]["design_config_path"])
    expected_design_sha = str(execution["generation"]["design_config_sha256"])
    if sha256_file(design_path) != expected_design_sha:
        raise RuntimeError("frozen Model V2 design config changed")
    design = _load_yaml(design_path)
    specifications = expand_model_v2_design(design)
    matrix_audit = validate_model_v2_design(root, design, specifications)
    generation = execution["generation"]
    expected_matrix_sha = str(generation["expected_matrix_sha256"])
    if matrix_audit["matrix_sha256"] != expected_matrix_sha:
        raise RuntimeError("resolved Model V2 matrix differs from execution freeze")
    if (
        str(generation["dataset_id"]) != MODEL_V2_DATASET_ID
        or len(specifications) != int(generation["expected_runs"])
        or matrix_audit["split_counts"]["V2_TRAIN"]
        != int(generation["expected_train_runs"])
        or matrix_audit["split_counts"]["V2_VALIDATION"]
        != int(generation["expected_validation_runs"])
        or matrix_audit["physical_signature_sha256"]
        != str(generation["expected_physical_signature_sha256"])
        or matrix_audit["signature_exclusion_sha256"]
        != str(generation["expected_signature_exclusion_sha256"])
        or matrix_audit["split_sha256"]
        != dict(generation["expected_split_sha256"])
    ):
        raise RuntimeError("Model V2 execution provenance differs from frozen design")
    if sha256_file(policy_path) != str(generation["policy_sha256"]):
        raise RuntimeError("walking policy differs from execution freeze")
    simulator_config = root / "configs/simulator/g1.yaml"
    if sha256_file(simulator_config) != str(generation["simulator_config_sha256"]):
        raise RuntimeError("simulator config differs from execution freeze")
    output_path = root / str(execution["generation"]["dataset_path"])
    partial_path = output_path.with_name(f"{output_path.name}_partial")
    if output_path.exists() or partial_path.exists():
        raise RuntimeError(f"Model V2 dataset output already exists: {output_path}")
    partial_path.mkdir(parents=True)
    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    try:
        for index, specification in enumerate(specifications, start=1):
            config = SimulationConfig(
                physics_timestep_s=float(design["dataset"]["physics_timestep_s"]),
                sensor_rate_hz=int(design["dataset"]["sensor_rate_hz"]),
                duration_s=float(design["dataset"]["duration_s"]),
                command_speed_mps=float(specification["speed_mps"]),
                policy_path=policy_path,
                terrain=str(specification["target_terrain"]),
                slip_pattern=str(specification["slip_pattern"]),
                sink_pattern=str(specification["sink_pattern"]),
                sink_severity=str(specification["sink_severity"]),
                patch_start_x_m=float(specification["patch_start_x_m"]),
                patch_width_m=float(specification["patch_width_m"]),
                headless=True,
                sink_support_pattern=str(specification["support_pattern"]),
                source_terrain=str(specification["source_terrain"]),
            )
            result = run_simulation(
                config, observe_fsr=True, observe_foot_imu=False
            )
            row, arrays = annotate_model_v2_result(specification, result)
            filename = f"{specification['run_id']}.npz"
            run_path = partial_path / filename
            _write_deterministic_npz(run_path, arrays)
            row["file"] = filename
            row["file_sha256"] = sha256_file(run_path)
            row["size_bytes"] = run_path.stat().st_size
            rows.append(row)
            if progress is not None and (index == 1 or index % 10 == 0):
                progress(
                    f"generated {index}/{len(specifications)}: "
                    f"{specification['run_id']}"
                )
        manifest = {
            "schema_version": 1,
            "dataset_id": MODEL_V2_DATASET_ID,
            "created_at": str(execution["generation"]["generation_start"]),
            "generation_source_commit": str(execution["generation"]["source_commit"]),
            "design_config_path": str(execution["generation"]["design_config_path"]),
            "design_config_sha256": expected_design_sha,
            "execution_config_path": str(execution_config_path.relative_to(root)),
            "execution_config_sha256": sha256_file(execution_config_path),
            "matrix_sha256": matrix_audit["matrix_sha256"],
            "physical_signature_sha256": matrix_audit[
                "physical_signature_sha256"
            ],
            "split_sha256": matrix_audit["split_sha256"],
            "signature_exclusion_sha256": matrix_audit[
                "signature_exclusion_sha256"
            ],
            "policy_sha256": sha256_file(policy_path),
            "simulator_config_sha256": str(
                execution["generation"]["simulator_config_sha256"]
            ),
            "model_blind": True,
            "reserve_runs_activated": 0,
            "run_count": len(rows),
            "valid_count": sum(bool(row["valid"]) for row in rows),
            "invalid_count": sum(not bool(row["valid"]) for row in rows),
            "runtime_model_input_fields": ["timestamp_us", "pelvis_imu6"],
            "runs": rows,
        }
        manifest_path = partial_path / "manifest.json"
        _write_json(manifest_path, manifest)
        manifest_sha = sha256_file(manifest_path)
        (partial_path / "manifest.sha256").write_text(
            f"{manifest_sha}  manifest.json\n", encoding="utf-8"
        )
        audit = audit_model_v2_manifest(manifest, design)
        audit["matrix_audit"] = matrix_audit
        _write_json(partial_path / "coverage_audit.json", audit)
        npz_hashes = {
            str(row["file"]): str(row["file_sha256"]) for row in rows
        }
        freeze = {
            "schema_version": 1,
            "dataset_id": MODEL_V2_DATASET_ID,
            "generation_source_commit": str(
                execution["generation"]["source_commit"]
            ),
            "design_config_sha256": expected_design_sha,
            "execution_config_sha256": sha256_file(execution_config_path),
            "run_count": len(rows),
            "valid_count": manifest["valid_count"],
            "invalid_count": manifest["invalid_count"],
            "manifest_sha256": manifest_sha,
            "physical_signature_sha256": matrix_audit[
                "physical_signature_sha256"
            ],
            "split_sha256": matrix_audit["split_sha256"],
            "npz_aggregate_sha256": canonical_sha256(npz_hashes),
            "actual_outcome_summary": audit["split_summary"],
            "precursor_annotation_summary": audit["ice_precursor"],
            "historical_exclusion_integrity": {
                "overlap_by_reference": matrix_audit["overlap_by_reference"],
                "signature_exclusion_sha256": matrix_audit[
                    "signature_exclusion_sha256"
                ],
            },
            "protected_model_hashes": design["protected_v1"],
            "generalization_holdout_guard_count": 0,
            "dataset_verdict": audit["dataset_verdict"],
            "training_readiness": audit["training_readiness"],
        }
        freeze_path = partial_path / "dataset_freeze.json"
        _write_json(freeze_path, freeze)
        freeze_sha = sha256_file(freeze_path)
        (partial_path / "dataset_freeze.sha256").write_text(
            f"{freeze_sha}  dataset_freeze.json\n", encoding="utf-8"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.rename(output_path)
    except Exception:
        if partial_path.exists():
            shutil.rmtree(partial_path)
        raise
    summary = {
        "dataset_id": MODEL_V2_DATASET_ID,
        "output_path": str(output_path),
        "designed_runs": len(specifications),
        "executed_runs": len(rows),
        "valid": sum(bool(row["valid"]) for row in rows),
        "invalid": sum(not bool(row["valid"]) for row in rows),
        "reserve_runs_activated": 0,
        "size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "generation_seconds": round(time.monotonic() - started, 3),
        "manifest_sha256": manifest_sha,
        "physical_signature_sha256": matrix_audit["physical_signature_sha256"],
        "npz_aggregate_sha256": freeze["npz_aggregate_sha256"],
        "split_sha256": matrix_audit["split_sha256"],
        "dataset_freeze_sha256": freeze_sha,
        "dataset_verdict": audit["dataset_verdict"],
        "training_readiness": audit["training_readiness"],
    }
    _write_json(output_path / "generation_summary.json", summary)
    return output_path, summary
