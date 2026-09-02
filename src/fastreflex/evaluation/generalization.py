"""Frozen-candidate evaluation on the external Generalization VALIDATION split."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fastreflex.dataset.generation import (
    HazardRunAnnotations,
    annotate_ice_precursors,
)
from fastreflex.dataset.hazard import (
    EVENT_TYPE_BOTH,
    EVENT_TYPE_NONE,
    EVENT_TYPE_SLIP,
    EVENT_TYPE_SUPPORT,
    LABEL_BOTH,
    LABEL_NO_HAZARD,
    LABEL_SLIP,
    LABEL_SUPPORT,
    PELVIS_IMU6,
    PELVIS_IMU6_FSR8,
    HazardRun,
    HoldoutGuard,
    canonical_sha256,
)
from fastreflex.dataset.loader import sha256_file
from fastreflex.evaluation.hazard import (
    HazardReplay,
    evaluate_hazard_replays,
    evaluate_ice_precursor_secondary,
    load_hazard_normalizer,
    reflex_onset_samples,
    reflex_required_trace,
    replay_hazard_runs,
    replay_hazard_runs_with_members,
)
from fastreflex.simulation.terrain import TERRAIN_CLASS_ORDER


EXPERIMENT_ID = "MODEL_V2_GENERALIZATION_DEVELOPMENT_EVALUATION"
DATASET_ID = "generalization_hazard_reflex_20260831"
VALIDATION_SPLIT = "GENERALIZATION_VALIDATION"
HOLDOUT_SPLIT = "GENERALIZATION_HOLDOUT"
VALIDATION_COUNT = 36
HOLDOUT_COUNT = 36
THRESHOLD = 0.99
PERSISTENCE_MS = 5
FUTURE_SLIP_OUTCOMES = frozenset(
    ("SAME_EPISODE_SLIP", "NEXT_EPISODE_SLIP", "LATER_SLIP")
)


@dataclass(frozen=True)
class GeneralizationData:
    """Authorized waveforms plus privileged evaluation-only annotations."""

    runs: Mapping[str, HazardRun]
    annotations: Mapping[str, HazardRunAnnotations]
    manifest_rows: Mapping[str, Mapping[str, object]]


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


def _optional_sample(value: int) -> int | None:
    return None if value < 0 else int(value)


def _first(values: np.ndarray) -> int | None:
    selected = np.flatnonzero(values)
    return None if not len(selected) else int(selected[0])


def _distribution(values: Sequence[int | float | None]) -> dict[str, float | None]:
    selected = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(selected):
        return {key: None for key in ("min", "p10", "median", "p95", "max")}
    return {
        "min": float(np.min(selected)),
        "p10": float(np.percentile(selected, 10)),
        "median": float(np.median(selected)),
        "p95": float(np.percentile(selected, 95)),
        "max": float(np.max(selected)),
    }


def _longest_threshold_excursion(values: np.ndarray, threshold: float) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if float(value) >= threshold else 0
        longest = max(longest, current)
    return longest


def _check_file(root: Path, record: Mapping[str, object]) -> None:
    path = root / str(record["path"])
    if sha256_file(path) != str(record["sha256"]):
        raise RuntimeError(f"protected artifact changed: {record['path']}")


def verify_promoted_candidate(
    root: Path, document: Mapping[str, Any]
) -> dict[str, object]:
    """Resolve the exact promotion without loading a model or dataset waveform."""
    candidate = document["candidate"]
    promotion_record = candidate["promotion_record"]
    _check_file(root, promotion_record)
    _check_file(root, candidate["candidate_freeze"])
    _check_file(root, candidate["internal_evaluation_freeze"])
    _check_file(root, candidate["training_config"])
    _check_file(root, candidate["normalizer"])
    for checkpoint in candidate["checkpoints"]:
        _check_file(root, checkpoint)

    promotion = _load_yaml(root / str(promotion_record["path"]))
    frozen = json.loads(
        (root / str(candidate["candidate_freeze"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    checkpoint_hashes = {
        str(row["path"]): str(row["sha256"])
        for row in candidate["checkpoints"]
    }
    if (
        promotion["candidate"]["id"] != candidate["id"]
        or frozen["candidate_id"] != candidate["id"]
        or promotion["candidate"]["candidate_freeze"]["sha256"]
        != candidate["candidate_freeze"]["sha256"]
        or frozen["normalizer_sha256"] != candidate["normalizer"]["sha256"]
        or frozen["checkpoint_sha256"] != checkpoint_hashes
        or frozen["architecture_sha256"] != candidate["architecture"]["sha256"]
        or frozen["feature_schema_sha256"] != candidate["feature_schema_sha256"]
        or float(frozen["threshold"]) != THRESHOLD
        or int(frozen["persistence_ms"]) != PERSISTENCE_MS
        or list(frozen["ensemble_membership"])
        != list(candidate["ensemble_membership"])
    ):
        raise RuntimeError("development promotion does not resolve exactly")
    return {
        "candidate_id": str(candidate["id"]),
        "promotion_sha256": str(promotion_record["sha256"]),
        "candidate_freeze_sha256": str(candidate["candidate_freeze"]["sha256"]),
        "normalizer_sha256": str(candidate["normalizer"]["sha256"]),
        "checkpoint_sha256": checkpoint_hashes,
        "architecture_sha256": str(candidate["architecture"]["sha256"]),
        "feature_schema_sha256": str(candidate["feature_schema_sha256"]),
        "threshold": THRESHOLD,
        "persistence_ms": PERSISTENCE_MS,
        "passed": True,
    }


def load_generalization_manifest(
    root: Path, document: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Verify the model-blind dataset freeze using metadata only."""
    dataset = document["dataset"]
    path = root / str(dataset["manifest"]["path"])
    freeze_path = root / str(dataset["dataset_freeze"]["path"])
    if (
        sha256_file(path) != str(dataset["manifest"]["file_sha256"])
        or sha256_file(freeze_path)
        != str(dataset["dataset_freeze"]["file_sha256"])
    ):
        raise RuntimeError("Generalization dataset metadata changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    validation_ids = [
        str(row["run_id"])
        for row in manifest["runs"]
        if row["split"] == VALIDATION_SPLIT
    ]
    holdout_ids = [
        str(row["run_id"])
        for row in manifest["runs"]
        if row["split"] == HOLDOUT_SPLIT
    ]
    if (
        manifest["dataset_id"] != DATASET_ID
        or len(manifest["runs"]) != VALIDATION_COUNT + HOLDOUT_COUNT
        or len(validation_ids) != VALIDATION_COUNT
        or len(holdout_ids) != HOLDOUT_COUNT
        or canonical_sha256(validation_ids)
        != str(dataset["validation_run_ids_canonical_sha256"])
        or canonical_sha256(holdout_ids)
        != str(dataset["holdout_run_ids_canonical_sha256"])
        or freeze["split_membership_sha256"]
        != str(dataset["split_membership_sha256"])
        or freeze["validation_run_ids"] != validation_ids
        or freeze["holdout_run_ids"] != holdout_ids
        or bool(freeze["generalization_holdout_waveform_opened"])
    ):
        raise RuntimeError("Generalization split/freeze contract changed")
    return manifest


def _load_generalization_row(
    dataset_path: Path, row: Mapping[str, Any]
) -> tuple[HazardRun, HazardRunAnnotations, Mapping[str, object]]:
    path = dataset_path / str(row["file"])
    if sha256_file(path) != str(row["file_sha256"]):
        raise RuntimeError(f"Generalization run changed: {row['run_id']}")
    with np.load(path, allow_pickle=False) as payload:
        timestamp = np.asarray(payload["timestamp_us"], dtype=np.int64)
        imu = np.asarray(payload["pelvis_imu6"], dtype=np.float32)
        fsr = np.asarray(payload["foot_fsr8"], dtype=np.float32)
        exact = np.asarray(payload["exact_terrain_contact"], dtype=bool)
        target_touchdown = np.asarray(payload["target_terrain_touchdown"], dtype=bool)
        loaded = np.asarray(payload["loaded_contact"], dtype=bool)
        contact_episode_id = np.asarray(payload["contact_episode_id"], dtype=np.int32)
        drift = np.asarray(payload["tangential_anchor_drift_m"], dtype=np.float32)
        velocity = np.asarray(payload["tangential_velocity_mps"], dtype=np.float32)
        slip_active = np.asarray(payload["established_slip"], dtype=bool)
        slip_onset = np.asarray(payload["established_slip_onset"], dtype=bool)
        spread = np.asarray(payload["support_surface_spread_m"], dtype=np.float32)
        displacement = np.asarray(
            payload["support_surface_max_displacement_m"], dtype=np.float32
        )
        support_onset = np.asarray(payload["deformable_sink_onset"], dtype=bool)
        i1_active = np.asarray(payload["i1_active"], dtype=bool)
        contact = int(payload["first_target_contact_sample"])
        touchdown = int(payload["first_target_touchdown_sample"])
        censor = int(payload["censor_sample"])
        slip = tuple(
            _optional_sample(int(value))
            for value in np.asarray(
                payload["first_slip_event_sample_per_foot"], dtype=np.int64
            )
        )
        support = tuple(
            _optional_sample(int(value))
            for value in np.asarray(
                payload["first_support_event_sample_per_foot"], dtype=np.int64
            )
        )
        fall = _optional_sample(int(payload["first_fall_sample"]))

    samples = len(timestamp)
    expected_2d = (samples, 2)
    if (
        timestamp.shape != (samples,)
        or imu.shape != (samples, 6)
        or fsr.shape != (samples, 8)
        or exact.shape != (samples, 2, len(TERRAIN_CLASS_ORDER))
        or target_touchdown.shape != expected_2d
        or loaded.shape != expected_2d
        or contact_episode_id.shape != expected_2d
        or drift.shape != expected_2d
        or velocity.shape != expected_2d
        or slip_active.shape != expected_2d
        or slip_onset.shape != expected_2d
        or spread.shape != expected_2d
        or displacement.shape != expected_2d
        or support_onset.shape != expected_2d
        or i1_active.shape != (samples,)
        or not np.array_equal(
            timestamp, (np.arange(samples, dtype=np.int64) + 1) * 1000
        )
        or not np.all(np.isfinite(imu))
        or not np.all(np.isfinite(fsr))
        or np.any(fsr < 0.0)
        or not 0 <= contact < censor <= samples
    ):
        raise RuntimeError(f"invalid Generalization tensors: {row['run_id']}")

    target_index = TERRAIN_CLASS_ORDER.index(str(row["target_terrain"]))
    target_contact = exact[:, :, target_index]
    ice_contact = exact[:, :, TERRAIN_CLASS_ORDER.index("ice")]
    episodes, precursor, outcome_code, precursor_censored = annotate_ice_precursors(
        exact_ice_contact=ice_contact,
        loaded_contact=loaded,
        contact_episode_id=contact_episode_id,
        drift_m=drift,
        velocity_mps=velocity,
        established_slip=slip_active,
        established_slip_onset=slip_onset,
        censor_sample=censor,
    )
    event_values = [value for value in (*slip, *support) if value is not None]
    event_type = (
        EVENT_TYPE_BOTH
        if any(value is not None for value in slip)
        and any(value is not None for value in support)
        else EVENT_TYPE_SLIP
        if any(value is not None for value in slip)
        else EVENT_TYPE_SUPPORT
        if any(value is not None for value in support)
        else EVENT_TYPE_NONE
    )
    fusion = np.concatenate((imu, fsr), axis=1).astype(np.float32, copy=False)
    run = HazardRun(
        run_id=str(row["run_id"]),
        split=str(row["split"]),
        source_terrain=str(row["source_terrain"]),
        target_terrain=str(row["target_terrain"]),
        design_role=str(row["expected_design_intent"]),
        first_contact_sample=contact,
        first_touchdown_sample=touchdown,
        censor_sample=censor,
        outcome_diagnostic=str(row["classification"]),
        fall_sample_diagnostic=fall,
        features={PELVIS_IMU6: imu, PELVIS_IMU6_FSR8: fusion},
        timestamp_us=timestamp,
        slip_event_samples_per_foot=slip,  # type: ignore[arg-type]
        support_event_samples_per_foot=support,  # type: ignore[arg-type]
        event_sample=min(event_values, default=None),
        event_type=event_type,
        hard_stable_control=False,
        drift_m=drift,
        tangential_velocity_mps=velocity,
        support_spread_m=spread,
        support_max_displacement_m=displacement,
        loaded_contact=loaded,
        sink_pattern=str(row["sink_pattern"]),
        support_pattern=str(row["support_pattern"]),
    )
    actual_side = (
        str(row["slip_side"])
        if str(row["slip_side"]) != "NONE"
        else str(row["support_side"])
    )
    annotation = HazardRunAnnotations(
        dataset_id=DATASET_ID,
        scenario_family=str(row["scenario_family"]),
        nominal_speed_mps=float(row["speed_mps"]),
        actual_side=actual_side,
        target_contact=target_contact,
        established_slip_active=slip_active,
        i1_active=i1_active,
        ice_precursor_candidate=precursor,
        ice_precursor_future_outcome_code=outcome_code,
        ice_precursor_censored=precursor_censored,
    )
    enriched = {
        **row,
        "ice_precursor_summary": {
            "episode_count": len(episodes),
            "future_outcomes": sorted(
                {str(value["future_outcome"]) for value in episodes}
            ),
            "episodes": episodes,
        },
    }
    return run, annotation, enriched


def load_generalization_split(
    root: Path,
    document: Mapping[str, Any],
    split: str,
    *,
    holdout_guard: HoldoutGuard | None = None,
) -> GeneralizationData:
    """Load validation directly and fail closed for the sealed HOLDOUT."""
    if split not in (VALIDATION_SPLIT, HOLDOUT_SPLIT):
        raise ValueError(f"unsupported Generalization split: {split}")
    if split == HOLDOUT_SPLIT:
        if holdout_guard is None:
            raise RuntimeError("Generalization HOLDOUT requires an explicit guard")
        holdout_guard.require_open()
    manifest = load_generalization_manifest(root, document)
    dataset_path = root / str(document["dataset"]["path"])
    runs: dict[str, HazardRun] = {}
    annotations: dict[str, HazardRunAnnotations] = {}
    rows: dict[str, Mapping[str, object]] = {}
    for row in manifest["runs"]:
        if row["split"] != split:
            continue
        run, annotation, enriched = _load_generalization_row(dataset_path, row)
        runs[run.run_id] = run
        annotations[run.run_id] = annotation
        rows[run.run_id] = enriched
    expected = VALIDATION_COUNT if split == VALIDATION_SPLIT else HOLDOUT_COUNT
    if len(runs) != expected:
        raise RuntimeError(f"Generalization {split} count changed")
    return GeneralizationData(runs=runs, annotations=annotations, manifest_rows=rows)


def _result_name(row: Mapping[str, object]) -> str:
    if row["physical_label"] == LABEL_NO_HAZARD:
        return "FALSE_POSITIVE" if row["system_false_positive"] else "TRUE_NEGATIVE"
    if row["valid_detection"]:
        return "CORRECT"
    if row["premature"]:
        return "PREMATURE"
    if row["system_first_onset"] is None:
        return "MISS"
    return "OUT_OF_VALID_WINDOW"


def _correct(row: Mapping[str, object], mode: str) -> bool:
    if mode == "specificity":
        return not bool(row["system_false_positive"])
    return bool(row["valid_detection"])


def _summary(
    run_ids: Sequence[str],
    rows: Mapping[str, Mapping[str, object]],
    *,
    mode: str,
) -> dict[str, object]:
    selected = [rows[run_id] for run_id in run_ids]
    correct = sum(_correct(row, mode) for row in selected)
    return {
        "eligible": len(selected),
        "correct": correct,
        "rate": None if not selected else correct / len(selected),
        "premature": sum(bool(row["premature"]) for row in selected),
        "false_positive": sum(
            bool(row["system_false_positive"]) for row in selected
        ),
    }


def _gate_results(
    primary: Mapping[str, object],
    ice_benign_specificity: float | None,
    gates: Mapping[str, object],
) -> dict[str, bool]:
    slip_p95 = primary["slip_latency_ms"]["p95"]
    support_p95 = primary["support_established_latency_ms"]["p95"]
    return {
        "overall_hazard_recall": float(primary["overall_hazard_recall"])
        >= float(gates["overall_hazard_recall_min"]),
        "slip_hazard_recall": float(primary["slip_hazard_recall"])
        >= float(gates["slip_hazard_recall_min"]),
        "support_hazard_recall": float(primary["support_hazard_recall"])
        >= float(gates["support_hazard_recall_min"]),
        "primary_no_hazard_specificity": float(
            primary["primary_no_hazard_specificity"]
        )
        >= float(gates["primary_no_hazard_specificity_min"]),
        "ice_benign_specificity": ice_benign_specificity is not None
        and ice_benign_specificity >= float(gates["ice_benign_specificity_min"]),
        "system_premature_run_rate": float(
            primary["system_premature_run_rate"]
        )
        <= float(gates["system_premature_run_rate_max"]),
        "slip_p95_latency_ms": slip_p95 is not None
        and float(slip_p95) <= float(gates["slip_p95_latency_ms_max"]),
        "support_p95_established_latency_ms": support_p95 is not None
        and float(support_p95)
        <= float(gates["support_p95_established_latency_ms_max"]),
    }


def _model_result(
    data: GeneralizationData,
    replays: Mapping[str, HazardReplay],
    gates: Mapping[str, object],
) -> dict[str, object]:
    precursor_samples = {
        run_id: (
            None
            if row["i1_sample"] is None
            else int(row["i1_sample"])
        )
        for run_id, row in data.manifest_rows.items()
    }
    primary = evaluate_hazard_replays(
        data.runs,
        replays,
        precursor_samples=precursor_samples,
        threshold=THRESHOLD,
        persistence_ms=PERSISTENCE_MS,
    )
    primary_rows = {str(row["run_id"]): row for row in primary["rows"]}
    ice_benign_ids = [
        run_id
        for run_id, row in data.manifest_rows.items()
        if row["scenario_family"] == "ICE_BENIGN_CONTROL"
        and primary_rows[run_id]["physical_label"] == LABEL_NO_HAZARD
    ]
    ice_benign = _summary(ice_benign_ids, primary_rows, mode="specificity")
    gate_results = _gate_results(primary, ice_benign["rate"], gates)
    secondary = evaluate_ice_precursor_secondary(
        data.runs,
        replays,
        data.annotations,
        data.manifest_rows,
        threshold=THRESHOLD,
        persistence_ms=PERSISTENCE_MS,
    )
    return {
        "primary": primary,
        "ice_benign": ice_benign,
        "gates": gate_results,
        "all_primary_gates_pass": all(gate_results.values()),
        "ice_precursor_secondary": secondary,
    }


def _verify_v1_parity(
    root: Path,
    document: Mapping[str, Any],
    result: Mapping[str, object],
) -> None:
    historical_path = root / str(
        document["baseline_v1"]["historical_generalization_validation"][
            "artifact"
        ]["path"]
    )
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    expected = historical["evaluation"]["primary_metrics"]
    actual = dict(result["primary"])
    actual.pop("rows")
    if actual != expected:
        raise RuntimeError("V1 Generalization VALIDATION replay parity failed")


def _first_reflex(replay: HazardReplay) -> int | None:
    onsets = reflex_onset_samples(replay, THRESHOLD, PERSISTENCE_MS)
    return None if not len(onsets) else int(onsets[0])


def _first_crossing(replay: HazardReplay) -> int | None:
    selected = replay.endpoints[replay.probabilities >= THRESHOLD]
    return None if not len(selected) else int(selected[0])


def _precursor_relation(
    annotation: HazardRunAnnotations,
    onset: int | None,
    i1: int | None,
    physical_label: str,
) -> str:
    if onset is None:
        return "GENUINE_NO_REFLEX" if physical_label != LABEL_NO_HAZARD else "NO_ALERT"
    if annotation.future_slip_precursor[onset]:
        return "SUPPORTED_FUTURE_SLIP_PRECURSOR"
    if annotation.benign_release_precursor[onset]:
        return "BENIGN_RELEASE_PRECURSOR_ALERT"
    if annotation.censored_precursor[onset]:
        return "CENSORED_PRECURSOR_ALERT"
    if i1 is not None and onset < i1:
        return "PRE_I1_SAND_FALSE_ALERT"
    if physical_label == LABEL_NO_HAZARD:
        return "BENIGN_FALSE_REFLEX"
    return "OUTSIDE_PRECURSOR_RESPONSE"


def _failure_diagnostics(
    row: Mapping[str, object], relation: str
) -> tuple[str | None, str | None, bool]:
    if bool(row["valid_detection"]) or row["physical_label"] == LABEL_NO_HAZARD:
        return None, None, False
    if bool(row["premature"]):
        if relation == "SUPPORTED_FUTURE_SLIP_PRECURSOR":
            return (
                "ICE_PRECURSOR_TIMING_CONFLICT",
                "PRIMARY_FAIL_WITH_PHYSICAL_EARLY_RESPONSE",
                False,
            )
        if relation == "PRE_I1_SAND_FALSE_ALERT":
            return (
                "PRE_I1_SAND_FALSE_ALERT",
                "SUPPORT_PRE_I1_FALSE_RESPONSE",
                False,
            )
        return (
            "UNRELATED_EARLY_FALSE_ALERT",
            "INVALID_BENIGN_EARLY_RESPONSE",
            True,
        )
    return "OTHER", "GENUINE_DETECTION_FAILURE", True


def _run_level_results(
    data: GeneralizationData,
    v1: Mapping[str, object],
    v2: Mapping[str, object],
    v1_replays: Mapping[str, HazardReplay],
    v2_replays: Mapping[str, HazardReplay],
    v2_members: Sequence[Mapping[str, HazardReplay]],
    terrain_rows: Mapping[str, Mapping[str, object]],
    seeds: Sequence[int],
) -> list[dict[str, object]]:
    v1_rows = {str(row["run_id"]): row for row in v1["primary"]["rows"]}
    v2_rows = {str(row["run_id"]): row for row in v2["primary"]["rows"]}
    results: list[dict[str, object]] = []
    for run_id, row in sorted(data.manifest_rows.items()):
        annotation = data.annotations[run_id]
        v1_row = v1_rows[run_id]
        v2_row = v2_rows[run_id]
        v2_onset = v2_row["system_first_onset"]
        precursor = _first(np.any(annotation.ice_precursor_candidate, axis=1))
        i1 = None if row["i1_sample"] is None else int(row["i1_sample"])
        relation = _precursor_relation(
            annotation,
            None if v2_onset is None else int(v2_onset),
            i1,
            str(v2_row["physical_label"]),
        )
        failure_reason, response_classification, genuine_failure = (
            _failure_diagnostics(v2_row, relation)
        )
        member_maxima = {
            str(seed): float(np.max(replays[run_id].probabilities))
            for seed, replays in zip(seeds, v2_members)
        }
        high = sum(value >= THRESHOLD for value in member_maxima.values())
        results.append(
            {
                "run_id": run_id,
                "family": str(row["scenario_family"]),
                "variant": row["variant_id"],
                "source": str(row["source_terrain"]),
                "speed_mps": float(row["speed_mps"]),
                "physical_label": str(v2_row["physical_label"]),
                "actual_side": str(annotation.actual_side),
                "target_contact": int(row["first_target_contact_sample"]),
                "terrain_target_first_valid": terrain_rows[run_id][
                    "terrain_first_target_valid_sample"
                ],
                "terrain_target_available": bool(
                    terrain_rows[run_id]["terrain_target_available"]
                ),
                "precursor_onset": precursor,
                "i1": i1,
                "slip": v2_row["slip_sample"],
                "support": v2_row["support_sample"],
                "benign_contacts_before_slip": int(
                    row["qualifying_benign_target_episode_count_before_slip"]
                ),
                "physical_classification": str(row["classification"]),
                "maximum_target_drift_m": float(
                    row["maximum_finite_target_drift_m"]
                ),
                "v1_first_threshold_crossing": _first_crossing(v1_replays[run_id]),
                "v1_first_reflex": v1_row["system_first_onset"],
                "v1_result": _result_name(v1_row),
                "v1_max_probability": float(
                    np.max(v1_replays[run_id].probabilities)
                ),
                "v2_first_threshold_crossing": _first_crossing(v2_replays[run_id]),
                "v2_first_reflex": v2_onset,
                "v2_result": _result_name(v2_row),
                "v2_max_probability": float(
                    np.max(v2_replays[run_id].probabilities)
                ),
                "v2_max_consecutive_at_or_above_0_99_ms": (
                    _longest_threshold_excursion(
                        v2_replays[run_id].probabilities, THRESHOLD
                    )
                ),
                "v2_seed_maximum_probability": member_maxima,
                "v2_seed_pattern": (
                    "ALL_3_HIGH"
                    if high == 3
                    else "2_OF_3_HIGH"
                    if high == 2
                    else "1_OF_3_HIGH"
                    if high == 1
                    else "ALL_LOW"
                ),
                "primary_classification": _result_name(v2_row),
                "primary_failure_reason": failure_reason,
                "v2_precursor_aware_classification": relation,
                "physical_response_classification": response_classification,
                "genuine_detection_failure": genuine_failure,
            }
        )
        terrain_first = results[-1]["terrain_target_first_valid"]
        if terrain_first is None or v2_onset is None:
            results[-1]["v2_terrain_reflex_order"] = "UNAVAILABLE"
            results[-1]["v2_terrain_to_reflex_ms"] = None
        elif int(terrain_first) <= int(v2_onset):
            results[-1]["v2_terrain_reflex_order"] = "TERRAIN_BEFORE_REFLEX"
            results[-1]["v2_terrain_to_reflex_ms"] = int(v2_onset) - int(
                terrain_first
            )
        else:
            results[-1]["v2_terrain_reflex_order"] = "REFLEX_BEFORE_TERRAIN"
            results[-1]["v2_terrain_to_reflex_ms"] = int(v2_onset) - int(
                terrain_first
            )
    return results


def _family_results(
    data: GeneralizationData,
    v1_rows: Mapping[str, Mapping[str, object]],
    v2_rows: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    groups: dict[str, tuple[list[str], str]] = {}
    for family in (
        "ONE_CONTACT_DELAYED_ICE_SLIP",
        "ICE_BENIGN_CONTROL",
        "DELAYED_SAND_SUPPORT_ONSET",
        "RIGHT_SAND_SUPPORT",
    ):
        ids = [
            run_id
            for run_id, row in data.manifest_rows.items()
            if row["scenario_family"] == family
        ]
        mode = "specificity" if family == "ICE_BENIGN_CONTROL" else "recall"
        groups[family] = (ids, mode)
    for variant in ("ICE_SLIP", "SAND_SUPPORT", "SAND_BENIGN"):
        ids = [
            run_id
            for run_id, row in data.manifest_rows.items()
            if row["scenario_family"] == "SPEED_STRATIFIED_HAZARD"
            and row["variant_id"] == variant
        ]
        mode = "specificity" if variant == "SAND_BENIGN" else "recall"
        groups[f"SPEED_STRATIFIED_{variant}"] = (ids, mode)
    return {
        name: {
            "mode": mode,
            "v1": _summary(ids, v1_rows, mode=mode),
            "v2": _summary(ids, v2_rows, mode=mode),
        }
        for name, (ids, mode) in groups.items()
    }


def _speed_results(
    data: GeneralizationData,
    v1_rows: Mapping[str, Mapping[str, object]],
    v2_rows: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for speed in (0.20, 0.25, 0.30):
        speed_result: dict[str, object] = {}
        for variant in ("ICE_SLIP", "SAND_SUPPORT", "SAND_BENIGN"):
            ids = [
                run_id
                for run_id, row in data.manifest_rows.items()
                if row["scenario_family"] == "SPEED_STRATIFIED_HAZARD"
                and row["variant_id"] == variant
                and float(row["speed_mps"]) == speed
            ]
            mode = "specificity" if variant == "SAND_BENIGN" else "recall"
            speed_result[variant] = {
                "mode": mode,
                "v1": _summary(ids, v1_rows, mode=mode),
                "v2": _summary(ids, v2_rows, mode=mode),
            }
        result[f"{speed:.2f}"] = speed_result
    return result


def _side_results(
    data: GeneralizationData,
    v1_rows: Mapping[str, Mapping[str, object]],
    v2_rows: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for physical_type, label in (("slip", LABEL_SLIP), ("support", LABEL_SUPPORT)):
        result[physical_type] = {}
        for side in ("LEFT_ONLY", "RIGHT_ONLY", "BILATERAL"):
            ids = [
                run_id
                for run_id, row in v2_rows.items()
                if row["physical_label"] in (label, LABEL_BOTH)
                and data.annotations[run_id].actual_side == side
            ]
            result[physical_type][side] = {
                "v1": _summary(ids, v1_rows, mode="recall"),
                "v2": _summary(ids, v2_rows, mode="recall"),
            }
    return result


def _source_results(
    data: GeneralizationData,
    v1_rows: Mapping[str, Mapping[str, object]],
    v2_rows: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for source in ("concrete", "marble"):
        source_ids = [
            run_id
            for run_id, row in data.manifest_rows.items()
            if row["source_terrain"] == source
        ]
        categories = {
            "hazard": (
                [
                    run_id
                    for run_id in source_ids
                    if v2_rows[run_id]["physical_label"]
                    in (LABEL_SLIP, LABEL_SUPPORT, LABEL_BOTH)
                ],
                "recall",
            ),
            "slip": (
                [run_id for run_id in source_ids if v2_rows[run_id]["slip_sample"] is not None],
                "recall",
            ),
            "support": (
                [run_id for run_id in source_ids if v2_rows[run_id]["support_sample"] is not None],
                "recall",
            ),
            "specificity": (
                [
                    run_id
                    for run_id in source_ids
                    if v2_rows[run_id]["physical_label"] == LABEL_NO_HAZARD
                ],
                "specificity",
            ),
        }
        result[source] = {
            name: {
                "v1": _summary(ids, v1_rows, mode=mode),
                "v2": _summary(ids, v2_rows, mode=mode),
            }
            for name, (ids, mode) in categories.items()
        }
        family_result = {}
        for family in (
            "ONE_CONTACT_DELAYED_ICE_SLIP",
            "ICE_BENIGN_CONTROL",
            "DELAYED_SAND_SUPPORT_ONSET",
            "RIGHT_SAND_SUPPORT",
            "SPEED_STRATIFIED_HAZARD",
        ):
            ids = [
                run_id
                for run_id in source_ids
                if data.manifest_rows[run_id]["scenario_family"] == family
            ]
            mode = "specificity" if family == "ICE_BENIGN_CONTROL" else "recall"
            family_result[family] = {
                "mode": mode,
                "v1": _summary(ids, v1_rows, mode=mode),
                "v2": _summary(ids, v2_rows, mode=mode),
            }
        result[source]["families"] = family_result
    return result


def _precursor_run_counts(
    data: GeneralizationData,
    replays: Mapping[str, HazardReplay],
) -> dict[str, int]:
    future_runs = 0
    future_alert_runs = 0
    benign_runs = 0
    benign_alert_runs = 0
    censored_runs = 0
    censored_alert_runs = 0
    for run_id, row in data.manifest_rows.items():
        outcomes = {
            str(episode["future_outcome"])
            for episode in row["ice_precursor_summary"]["episodes"]
        }
        trace = reflex_required_trace(
            replays[run_id], len(data.runs[run_id].timestamp_us), THRESHOLD, PERSISTENCE_MS
        )
        annotation = data.annotations[run_id]
        if outcomes & FUTURE_SLIP_OUTCOMES:
            future_runs += 1
            future_alert_runs += int(np.any(trace & annotation.future_slip_precursor))
        if "BENIGN_RELEASE" in outcomes:
            benign_runs += 1
            benign_alert_runs += int(np.any(trace & annotation.benign_release_precursor))
        if "CENSORED" in outcomes:
            censored_runs += 1
            censored_alert_runs += int(np.any(trace & annotation.censored_precursor))
    return {
        "future_slip_runs": future_runs,
        "future_slip_alert_in_candidate_runs": future_alert_runs,
        "benign_release_runs": benign_runs,
        "benign_release_alert_runs": benign_alert_runs,
        "censored_runs": censored_runs,
        "censored_alert_runs": censored_alert_runs,
    }


def _failure_resolution(
    family: Mapping[str, Any], speed: Mapping[str, Any]
) -> list[dict[str, object]]:
    entries = [
        (
            "Delayed Ice",
            "missing delayed/multi-contact Ice coverage",
            "retained_and_augmented_delayed_Ice",
            family["ONE_CONTACT_DELAYED_ICE_SLIP"],
        ),
        (
            "Ice benign",
            "missing near-hazard benign coverage",
            "Ice_benign_and_precursor_aware_coverage",
            family["ICE_BENIGN_CONTROL"],
        ),
        (
            "Delayed Sand pre-I1",
            "benign staged-entry transient",
            "staged_Sand_negatives_and_refined_Support_anchors",
            family["DELAYED_SAND_SUPPORT_ONSET"],
        ),
        (
            "Right Sand Support",
            "right-side positive coverage absent",
            "balanced_right_Support_augmentation",
            family["RIGHT_SAND_SUPPORT"],
        ),
        (
            "0.20 Slip",
            "endpoint speed coverage absent",
            "source_balanced_speed_expansion",
            speed["0.20"]["ICE_SLIP"],
        ),
        (
            "0.30 Slip",
            "endpoint speed coverage absent",
            "source_balanced_speed_expansion",
            speed["0.30"]["ICE_SLIP"],
        ),
        (
            "Speed Sand benign",
            "transition hard-negative coverage narrow",
            "speed_stratified_Sand_benign_negatives",
            family["SPEED_STRATIFIED_SAND_BENIGN"],
        ),
    ]
    result = []
    for name, cause, intervention, values in entries:
        v1 = values["v1"]
        v2 = values["v2"]
        resolved = bool(v2["eligible"] and v2["correct"] == v2["eligible"])
        result.append(
            {
                "original_v1_failure": name,
                "why_it_existed": cause,
                "v2_intervention": intervention,
                "v1_external": v1,
                "v2_external": v2,
                "resolution": "RESOLVED" if resolved else "NOT_FULLY_RESOLVED",
                "notes": (
                    "All eligible external runs satisfy the original primary contract."
                    if resolved
                    else "At least one eligible run still fails the original primary contract."
                ),
            }
        )
    return result


def _development_verdict(
    v2: Mapping[str, Any], run_rows: Sequence[Mapping[str, object]]
) -> str:
    if bool(v2["all_primary_gates_pass"]):
        return "GENERALIZATION_DEVELOPMENT_SUPPORTED"
    primary_failures = [
        row
        for row in run_rows
        if row["physical_label"] != LABEL_NO_HAZARD
        and row["v2_result"] != "CORRECT"
    ]
    supported = [
        row
        for row in primary_failures
        if row["v2_precursor_aware_classification"]
        == "SUPPORTED_FUTURE_SLIP_PRECURSOR"
    ]
    non_slip_gates = {
        name: passed
        for name, passed in v2["gates"].items()
        if name not in ("slip_hazard_recall", "system_premature_run_rate")
    }
    if (
        all(non_slip_gates.values())
        and primary_failures
        and len(supported) == len(primary_failures)
        and not any(bool(row["genuine_detection_failure"]) for row in primary_failures)
    ):
        return "GENERALIZATION_DEVELOPMENT_SUPPORTED_WITH_ICE_TIMING_TENSION"
    return "GENERALIZATION_DEVELOPMENT_NOT_SUPPORTED"


def _data_coverage_verdict(
    development_verdict: str,
    failure_resolution: Sequence[Mapping[str, object]],
) -> str:
    if development_verdict in (
        "GENERALIZATION_DEVELOPMENT_SUPPORTED",
        "GENERALIZATION_DEVELOPMENT_SUPPORTED_WITH_ICE_TIMING_TENSION",
    ):
        return "DATA_COVERAGE_HYPOTHESIS_SUPPORTED"
    resolved = sum(row["resolution"] == "RESOLVED" for row in failure_resolution)
    if resolved >= 4:
        return "DATA_COVERAGE_HYPOTHESIS_PARTIALLY_SUPPORTED"
    return "DATA_COVERAGE_HYPOTHESIS_NOT_SUPPORTED"


def run_generalization_development_evaluation(
    root: Path, config_path: Path
) -> dict[str, object]:
    """Execute the one predeclared validation comparison and freeze its results."""
    document = _load_yaml(config_path)
    if document["experiment"]["id"] != EXPERIMENT_ID:
        raise ValueError("unsupported Generalization development config")
    config_sha = sha256_file(config_path)
    artifact_path = root / str(document["artifacts"]["path"])
    freeze_path = artifact_path / "evaluation_freeze.json"
    if freeze_path.exists():
        return verify_generalization_development_evaluation(root, config_path)

    candidate = verify_promoted_candidate(root, document)
    load_generalization_manifest(root, document)
    data = load_generalization_split(root, document, VALIDATION_SPLIT)

    _check_file(root, document["baseline_v1"]["normalizer"])
    _check_file(
        root,
        document["baseline_v1"]["historical_generalization_validation"][
            "artifact"
        ],
    )
    v1_normalizer = load_hazard_normalizer(
        root / str(document["baseline_v1"]["normalizer"]["path"])
    )
    for record in document["baseline_v1"]["checkpoints"]:
        _check_file(root, record)
    v1_checkpoints = tuple(
        root / str(record["path"])
        for record in document["baseline_v1"]["checkpoints"]
    )
    v1_replays = replay_hazard_runs(data.runs, v1_normalizer, v1_checkpoints)
    gates = document["primary_evaluation"]["gates"]
    v1_result = _model_result(data, v1_replays, gates)
    _verify_v1_parity(root, document, v1_result)

    artifact_path.mkdir(parents=True, exist_ok=False)
    v2_normalizer = load_hazard_normalizer(
        root / str(document["candidate"]["normalizer"]["path"])
    )
    v2_checkpoints = tuple(
        root / str(record["path"])
        for record in document["candidate"]["checkpoints"]
    )
    v2_replays, v2_members = replay_hazard_runs_with_members(
        data.runs, v2_normalizer, v2_checkpoints
    )
    v2_result = _model_result(data, v2_replays, gates)

    historical = json.loads(
        (
            root
            / str(
                document["baseline_v1"]["historical_generalization_validation"][
                    "artifact"
                ]["path"]
            )
        ).read_text(encoding="utf-8")
    )
    terrain_rows = {
        str(row["run_id"]): row for row in historical["evaluation"]["rows"]
    }
    run_rows = _run_level_results(
        data,
        v1_result,
        v2_result,
        v1_replays,
        v2_replays,
        v2_members,
        terrain_rows,
        document["candidate"]["ensemble_membership"],
    )
    v1_rows = {
        str(row["run_id"]): row for row in v1_result["primary"]["rows"]
    }
    v2_rows = {
        str(row["run_id"]): row for row in v2_result["primary"]["rows"]
    }
    family = _family_results(data, v1_rows, v2_rows)
    speed = _speed_results(data, v1_rows, v2_rows)
    side = _side_results(data, v1_rows, v2_rows)
    source = _source_results(data, v1_rows, v2_rows)
    precursor = {
        "v1": {
            **v1_result["ice_precursor_secondary"],
            "runs": _precursor_run_counts(data, v1_replays),
        },
        "v2": {
            **v2_result["ice_precursor_secondary"],
            "runs": _precursor_run_counts(data, v2_replays),
        },
        "primary_scores_rewritten": False,
    }
    failure_resolution = _failure_resolution(family, speed)
    primary_verdict = (
        "GENERALIZATION_PRIMARY_GATES_PASS"
        if v2_result["all_primary_gates_pass"]
        else "GENERALIZATION_PRIMARY_GATES_FAIL"
    )
    development_verdict = _development_verdict(v2_result, run_rows)
    data_coverage_verdict = _data_coverage_verdict(
        development_verdict, failure_resolution
    )

    artifacts: dict[str, object] = {
        "v1_generalization_validation_metrics.json": {
            **v1_result,
            "replay_parity": True,
        },
        "v2_generalization_validation_metrics.json": v2_result,
        "run_level_results.json": {"rows": run_rows},
        "family_results.json": family,
        "speed_results.json": speed,
        "side_results.json": side,
        "source_results.json": source,
        "ice_precursor_results.json": precursor,
        "failure_resolution_matrix.json": failure_resolution,
    }
    artifact_hashes: dict[str, str] = {}
    for name, value in artifacts.items():
        path = artifact_path / name
        _write_json(path, value)
        artifact_hashes[name] = sha256_file(path)

    freeze = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": config_sha,
        "source_sha256": sha256_file(Path(__file__)),
        "candidate": candidate,
        "dataset_id": DATASET_ID,
        "validation_count": len(data.runs),
        "v1_replay_parity": True,
        "promoted_v2_inference_runs": len(data.runs),
        "promoted_v2_inference_repetitions": 1,
        "intermediate_v2_candidate_inference": False,
        "primary_gate_verdict": primary_verdict,
        "development_verdict": development_verdict,
        "data_coverage_verdict": data_coverage_verdict,
        "artifact_sha256": artifact_hashes,
        "generalization_holdout_waveform_opened": False,
        "generalization_holdout_model_inference": False,
        "generalization_holdout_terrain_inference": False,
        "generalization_holdout_visualization": False,
        "generalization_holdout_guard_count": 0,
        "current_unified_holdout_waveform_reopened": False,
        "current_unified_holdout_new_inference": False,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "normalizer_fits": 0,
        "hnm_rounds": 0,
        "threshold_searches": 0,
        "persistence_searches": 0,
        "architecture_searches": 0,
        "seed_searches": 0,
        "new_simulation_runs": 0,
        "candidate_mutated": False,
        "dataset_mutated": False,
        "status": "MODEL_V2_GENERALIZATION_DEVELOPMENT_EVALUATION_COMPLETE",
    }
    _write_json(freeze_path, freeze)
    return {
        "status": freeze["status"],
        "config_sha256": config_sha,
        "evaluation_freeze_sha256": sha256_file(freeze_path),
        "v1_replay_parity": True,
        "primary_gate_verdict": primary_verdict,
        "development_verdict": development_verdict,
        "data_coverage_verdict": data_coverage_verdict,
        "generalization_holdout_guard_count": 0,
    }


def verify_generalization_development_evaluation(
    root: Path, config_path: Path
) -> dict[str, object]:
    """Verify frozen outputs without reopening any dataset waveform."""
    document = _load_yaml(config_path)
    artifact_path = root / str(document["artifacts"]["path"])
    freeze_path = artifact_path / "evaluation_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    candidate = verify_promoted_candidate(root, document)
    load_generalization_manifest(root, document)
    if (
        freeze["config_sha256"] != sha256_file(config_path)
        or freeze["candidate"] != candidate
        or freeze["validation_count"] != VALIDATION_COUNT
        or not freeze["v1_replay_parity"]
        or freeze["promoted_v2_inference_runs"] != VALIDATION_COUNT
        or freeze["promoted_v2_inference_repetitions"] != 1
        or freeze["intermediate_v2_candidate_inference"]
        or freeze["generalization_holdout_waveform_opened"]
        or freeze["generalization_holdout_model_inference"]
        or freeze["generalization_holdout_terrain_inference"]
        or freeze["generalization_holdout_visualization"]
        or freeze["generalization_holdout_guard_count"] != 0
        or freeze["candidate_mutated"]
        or freeze["dataset_mutated"]
        or any(
            sha256_file(artifact_path / name) != expected
            for name, expected in freeze["artifact_sha256"].items()
        )
    ):
        raise RuntimeError("Generalization development evaluation freeze changed")
    return {
        "status": str(freeze["status"]),
        "config_sha256": str(freeze["config_sha256"]),
        "evaluation_freeze_sha256": sha256_file(freeze_path),
        "v1_replay_parity": True,
        "primary_gate_verdict": str(freeze["primary_gate_verdict"]),
        "development_verdict": str(freeze["development_verdict"]),
        "data_coverage_verdict": str(freeze["data_coverage_verdict"]),
        "generalization_holdout_guard_count": 0,
        "passed": True,
    }
