"""One-shot Confirmation replication for the calibrated Sand study."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import yaml

import fastreflex.evaluation.sand as discovery
from fastreflex.dataset.hazard import canonical_sha256
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
from fastreflex.features import HAZARD_FEATURE_SCHEMA_SHA256, extract_hazard_features
from fastreflex.models.checkpoint import load_checkpoint


DISCOVERY_SPLIT = "MILD_RECALIBRATED_DISCOVERY"
CONFIRMATION_SPLIT = "MILD_RECALIBRATED_CONFIRMATION"
STRICT_BENIGN = "STRICT_BENIGN"
SUPPORT_GROUPS = {"ordinary_support_control", "delayed_support_control"}
HISTORY_MS = 20
EPSILON = 1.0e-8


def _verify_sha(root: Path, item: Mapping[str, Any], name: str) -> None:
    actual = sha256_file(root / str(item["path"]))
    if actual != str(item["sha256"]):
        raise RuntimeError(f"{name} SHA mismatch: {actual}")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _optional_sample(value: object) -> int | None:
    return None if value is None or int(value) < 0 else int(value)


def _raw_scaler(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mean = np.mean(array, axis=0)
    raw_std = np.std(array, axis=0)
    std = np.maximum(raw_std, EPSILON)
    return {
        "dimension": int(array.shape[1]),
        "epsilon": EPSILON,
        "constant_dimensions": int(np.sum(raw_std < EPSILON)),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "mean_sha256": hashlib.sha256(mean.tobytes()).hexdigest(),
        "std_sha256": hashlib.sha256(std.tobytes()).hexdigest(),
    }


def _scaler_arrays(scaler: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)
    if (
        mean.shape != std.shape
        or mean.ndim != 1
        or len(mean) != int(scaler["dimension"])
        or np.any(std < EPSILON)
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(std))
    ):
        raise ValueError("frozen Discovery scaler is malformed")
    if hashlib.sha256(mean.tobytes()).hexdigest() != scaler["mean_sha256"]:
        raise ValueError("frozen Discovery scaler mean changed")
    if hashlib.sha256(std.tobytes()).hexdigest() != scaler["std_sha256"]:
        raise ValueError("frozen Discovery scaler std changed")
    return mean, std


def _collect_vectors(
    rows: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, np.ndarray]],
    normalizer: Any,
) -> dict[str, Any]:
    run_ids: list[str] = []
    labels: list[int] = []
    anchors: dict[str, int] = {}
    current: list[np.ndarray] = []
    windows: list[np.ndarray] = []
    fsr: list[np.ndarray] = []
    oracle: list[np.ndarray] = []
    vector_rows: list[dict[str, Any]] = []
    for run_id, row in sorted(rows.items()):
        eligible = row["objective_physical_outcome"] == STRICT_BENIGN or (
            row["group"] in SUPPORT_GROUPS
            and row["actual_hazard_label"] == "HAZARD"
            and row["actual_subtype"] == "SUPPORT"
            and row["valid"]
        )
        if not eligible:
            continue
        payload = payloads[run_id]
        label = 0 if row["objective_physical_outcome"] == STRICT_BENIGN else 1
        anchor = (
            discovery.benign_anchor(payload)
            if label == 0
            else discovery.support_anchor(row)
        )
        features = extract_hazard_features(payload["pelvis_imu6"])
        normalized = normalizer.transform(features).astype(np.float32, copy=False)
        window = normalized[anchor - HISTORY_MS + 1 : anchor + 1]
        if window.shape != (HISTORY_MS, 80):
            raise RuntimeError(f"incomplete Confirmation analysis window: {run_id}")
        run_ids.append(run_id)
        labels.append(label)
        anchors[run_id] = anchor
        current.append(normalized[anchor].astype(np.float64))
        windows.append(window.astype(np.float64).reshape(-1))
        fsr.append(discovery.fsr_contact_vector(payload, anchor))
        oracle.append(discovery.privileged_oracle_vector(payload, anchor))
        vector_rows.append(
            {
                "run_id": run_id,
                "label": "SAND" if label == 0 else "SUPPORT",
                "group": row["group"],
                "factors": discovery.factor_metadata(row) if label == 0 else {
                    "source": str(row["source_terrain"]).upper(),
                    "speed": f"{float(row['speed_mps']):.2f}",
                },
            }
        )
    current_array = np.asarray(current, dtype=np.float64)
    window_array = np.asarray(windows, dtype=np.float64)
    fsr_array = np.asarray(fsr, dtype=np.float64)
    return {
        "run_ids": run_ids,
        "labels": np.asarray(labels, dtype=np.int64),
        "anchors": anchors,
        "current": current_array,
        "window": window_array,
        "fsr": fsr_array,
        "combined": np.concatenate((window_array, fsr_array), axis=1),
        "oracle": np.asarray(oracle, dtype=np.float64),
        "rows": vector_rows,
    }


def reconstruct_discovery_scalers(
    root: Path, document: Mapping[str, Any]
) -> dict[str, Any]:
    """Reconstruct omitted scaler values and verify their frozen hashes."""
    dataset_path = root / str(document["dataset"]["path"])
    manifest = load_mild_recalibrated_manifest(dataset_path)
    rows = {
        str(row["run_id"]): row
        for row in manifest["runs"]
        if row["split"] == DISCOVERY_SPLIT
    }
    payloads = {
        run_id: load_mild_recalibrated_discovery_payload(dataset_path, run_id)
        for run_id in sorted(rows)
    }
    normalizer = load_hazard_normalizer(root / str(document["model"]["normalizer"]["path"]))
    vectors = _collect_vectors(rows, payloads, normalizer)
    scalers = {
        name: _raw_scaler(vectors[name])
        for name in ("current", "window", "fsr", "combined", "oracle")
    }
    discovery_path = root / str(document["discovery"]["artifact_path"])
    expected = {
        "current": _load_json(discovery_path / "pelvis_analysis.json")["current_80d"]["scaling"],
        "window": _load_json(discovery_path / "pelvis_analysis.json")["flattened_window_20x80"]["scaling"],
        "fsr": _load_json(discovery_path / "fsr_contact_analysis.json")["fsr_contact_only"]["scaling"],
        "combined": _load_json(discovery_path / "fsr_contact_analysis.json")["pelvis_plus_fsr_contact"]["scaling"],
        "oracle": _load_json(discovery_path / "privileged_oracle_analysis.json")["privileged_oracle"]["scaling"],
    }
    for name, scaler in scalers.items():
        for field in (
            "dimension",
            "epsilon",
            "constant_dimensions",
            "mean_sha256",
            "std_sha256",
        ):
            if scaler[field] != expected[name][field]:
                raise RuntimeError(f"reconstructed Discovery {name} scaler mismatch: {field}")
    return {
        "schema_version": 1,
        "purpose": "exact_Discovery_scaler_value_recovery_for_frozen_Confirmation",
        "discovery_model_replay": 0,
        "confirmation_payload_reads": 0,
        "eligible_run_count": len(vectors["run_ids"]),
        "eligible_run_ids_sha256": canonical_sha256(vectors["run_ids"]),
        "scalers": scalers,
    }


def _metrics_on_scaled(
    scaled_values: np.ndarray,
    labels: Sequence[int],
    *,
    run_ids: Sequence[str],
    scaling: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the exact frozen metric body after a fixed Discovery scaler."""
    values = np.asarray(scaled_values, dtype=np.float64)
    classes = np.asarray(labels, dtype=np.int64)
    if set(classes.tolist()) != {0, 1}:
        raise ValueError("separability requires Sand and Support")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("run-disjoint analysis received duplicate run IDs")
    distance = discovery._pairwise_distances(values)
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
    for k in (1, 5):
        predicted = np.empty(len(values), dtype=np.int64)
        for query in range(len(values)):
            order = np.argsort(distance[query], kind="stable")
            neighbors = order[order != query][:k]
            predicted[query] = int(np.sum(classes[neighbors]) > k / 2)
        agreement[f"balanced_{k}nn_agreement"] = discovery._balanced_mean(
            (predicted == classes).astype(np.float64), classes
        )
    for query in range(len(values)):
        same = np.flatnonzero(
            (classes == classes[query]) & (np.arange(len(values)) != query)
        )
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
    ratio_by_class = [
        float(np.median(nearest_ratios[classes == label])) for label in (0, 1)
    ]
    return {
        "population": {
            "sand": int(np.sum(classes == 0)),
            "support": int(np.sum(classes == 1)),
            "one_vector_per_run": True,
            "run_ids_sha256": canonical_sha256(list(run_ids)),
        },
        "scaling": {
            "source": "frozen_Discovery_pooled_scaler",
            "dimension": scaling["dimension"],
            "epsilon": scaling["epsilon"],
            "constant_dimensions": scaling["constant_dimensions"],
            "mean_sha256": scaling["mean_sha256"],
            "std_sha256": scaling["std_sha256"],
        },
        "centroid_separation": centroid_distance / max(within_rms, EPSILON),
        "centroid_distance": centroid_distance,
        "within_group_rms": within_rms,
        **agreement,
        "median_nearest_opposite_to_same_ratio": float(np.mean(ratio_by_class)),
        "nearest_opposite_to_same_ratio_class_medians": {
            "sand": ratio_by_class[0],
            "support": ratio_by_class[1],
        },
        "local_opposite_class_mixing": discovery._balanced_mean(mixing, classes),
        "bidirectional_95pct_radius_inclusion": float(
            np.mean((sand_in_support, support_in_sand))
        ),
        "sand_in_support_95pct_radius": sand_in_support,
        "support_in_sand_95pct_radius": support_in_sand,
        "distance_quantiles": {
            "within_sand": discovery._quantiles(within_sand),
            "within_support": discovery._quantiles(within_support),
            "between": discovery._quantiles(between),
        },
        "pca": discovery._pca_diagnostic(values, classes),
    }


def separability_with_frozen_scaler(
    raw_values: np.ndarray,
    labels: Sequence[int],
    *,
    run_ids: Sequence[str],
    scaler: Mapping[str, Any],
) -> dict[str, Any]:
    """Transform with the frozen Discovery scaler, then run exact metrics."""
    mean, std = _scaler_arrays(scaler)
    values = np.asarray(raw_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(mean):
        raise ValueError("Confirmation representation differs from frozen scaler")
    return _metrics_on_scaled(
        (values - mean) / std,
        labels,
        run_ids=run_ids,
        scaling=scaler,
    )


def verify_confirmation_inputs(
    root: Path, config_path: Path, document: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify all immutable evidence before the Confirmation guard opens."""
    dataset_path = root / str(document["dataset"]["path"])
    dataset_verification = verify_mild_recalibrated_dataset(dataset_path)
    if not dataset_verification["passed"]:
        raise RuntimeError("mild-recalibrated dataset integrity failed")
    freeze = _load_json(dataset_path / "dataset_freeze.json")
    hashes = document["dataset"]["hashes"]
    checks = {
        "manifest": sha256_file(dataset_path / "manifest.json") == hashes["manifest"],
        "discovery_split": freeze["MILD_RECALIBRATED_DISCOVERY_SPLIT_SHA"]
        == hashes["discovery_split"],
        "confirmation_split": freeze["MILD_RECALIBRATED_CONFIRMATION_SPLIT_SHA"]
        == hashes["confirmation_split"],
        "confirmation_seal": sha256_file(dataset_path / "confirmation_seal.json")
        == hashes["confirmation_seal"],
        "semantic_dataset_freeze": freeze["MILD_RECALIBRATED_DATASET_FREEZE_SHA"]
        == hashes["semantic_dataset_freeze"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Confirmation dataset identity changed: {checks}")
    _verify_sha(root, document["implementation"], "Confirmation implementation")
    _verify_sha(root, document["metric_implementation"], "Discovery metric implementation")
    _verify_sha(root, document["model"]["record"], "final candidate record")
    _verify_sha(root, document["model"]["candidate_freeze"], "candidate freeze")
    _verify_sha(root, document["model"]["normalizer"], "normalizer")
    for index, checkpoint in enumerate(document["model"]["checkpoints"]):
        _verify_sha(root, checkpoint, f"checkpoint {index}")
    if document["model"]["feature_schema_sha256"] != HAZARD_FEATURE_SCHEMA_SHA256:
        raise RuntimeError("feature schema declaration changed")
    discovery_path = root / str(document["discovery"]["artifact_path"])
    for name, expected in document["discovery"]["artifact_hashes"].items():
        if sha256_file(discovery_path / name) != expected:
            raise RuntimeError(f"Discovery artifact changed: {name}")
    interpretation = _load_json(discovery_path / "discovery_interpretation.json")
    semantic = dict(interpretation)
    interpretation_sha = semantic.pop("SAND_BENIGN_DISCOVERY_INTERPRETATION_SHA")
    if canonical_sha256(semantic) != interpretation_sha:
        raise RuntimeError("Discovery interpretation semantic hash changed")
    if (
        interpretation_sha != document["discovery"]["interpretation_sha256"]
        or interpretation["selected_hypothesis"] != "DOMAIN_DIVERSITY_GAP_SUPPORTED"
        or interpretation["analysis_validity"]
        != "SAND_BENIGN_GENERALIZATION_STUDY_DISCOVERY_ANALYSIS_VALID"
    ):
        raise RuntimeError("frozen Discovery H1 changed")
    guard = root / str(document["boundaries"]["old_holdout_guard_path"])
    if sha256_file(guard) != document["boundaries"]["old_holdout_guard_sha256"]:
        raise RuntimeError("historical HOLDOUT guard changed")
    manifest = load_mild_recalibrated_manifest(dataset_path)
    confirmation = [row for row in manifest["runs"] if row["split"] == CONFIRMATION_SPLIT]
    counts = {
        "total": len(confirmation),
        "strict_mild": sum(
            row["objective_physical_outcome"] == STRICT_BENIGN
            and row["actual_benign_severity"] == "LOW"
            for row in confirmation
        ),
        "strict_moderate": sum(
            row["objective_physical_outcome"] == STRICT_BENIGN
            and row["actual_benign_severity"] == "MEDIUM"
            for row in confirmation
        ),
        "support": sum(
            row["actual_hazard_label"] == "HAZARD"
            and row["actual_subtype"] == "SUPPORT"
            and row["valid"]
            for row in confirmation
        ),
        "actual_slip": sum(row["actual_subtype"] == "SLIP" for row in confirmation),
        "invalid": sum(not row["valid"] for row in confirmation),
    }
    if counts != document["dataset"]["confirmation_population"]:
        raise RuntimeError(f"Confirmation population changed: {counts}")
    return {
        "passed": True,
        "dataset_checks": checks,
        "dataset_verification": dataset_verification,
        "confirmation_population": counts,
        "analysis_config_sha256": sha256_file(config_path),
        "confirmation_implementation_sha256": sha256_file(
            root / str(document["implementation"]["path"])
        ),
        "metric_implementation_sha256": sha256_file(
            root / str(document["metric_implementation"]["path"])
        ),
        "discovery_interpretation_sha256": interpretation_sha,
        "old_holdout_guard": 1,
        "old_holdout_payload_reads": 0,
        "confirmation_payload_reads_before_authorization": 0,
    }


def _load_confirmation_payload(
    dataset_path: Path,
    row: Mapping[str, Any],
    guard: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if row["split"] != CONFIRMATION_SPLIT:
        raise RuntimeError("authorized loader accepts Confirmation only")
    if guard.get("open_count") != 1 or not guard.get("consumed"):
        raise RuntimeError("Confirmation access guard has not been claimed")
    path = dataset_path / str(row["file"])
    if sha256_file(path) != str(row["file_sha256"]):
        raise RuntimeError(f"Confirmation run integrity failed: {row['run_id']}")
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


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


def _longest_streak(values: np.ndarray, threshold: float) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if float(value) >= threshold else 0
        best = max(best, current)
    return best


def _subset_summary(
    rows: Sequence[Mapping[str, Any]], factor: str
) -> dict[str, Any]:
    return {
        level: discovery.summarize_benign(
            [row for row in rows if str(row["factors"][factor]) == level]
        )
        for level in sorted(set(str(row["factors"][factor]) for row in rows))
    }


def _support_subsets(
    rows: Sequence[Mapping[str, Any]], factor: str
) -> dict[str, Any]:
    return {
        level: discovery.summarize_support(
            [row for row in rows if str(row["factors"][factor]) == level]
        )
        for level in sorted(set(str(row["factors"][factor]) for row in rows))
    }


def _factor_metrics_with_scaler(
    vectors: np.ndarray,
    labels: np.ndarray,
    run_ids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    factors: Sequence[str],
    scaler: Mapping[str, Any],
) -> dict[str, Any]:
    mean, std = _scaler_arrays(scaler)
    scaled = (np.asarray(vectors, dtype=np.float64) - mean) / std
    support_indices = np.flatnonzero(labels == 1)
    output: dict[str, Any] = {}
    for factor in factors:
        levels: dict[str, Any] = {}
        factor_levels = sorted(
            set(str(rows[index]["factors"][factor]) for index in np.flatnonzero(labels == 0))
        )
        for level in factor_levels:
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
                (
                    np.zeros(len(sand_indices), dtype=np.int64),
                    np.ones(len(support_indices), dtype=np.int64),
                )
            )
            levels[level] = {
                "sand_n": len(sand_indices),
                "metrics": _metrics_on_scaled(
                    scaled[indices],
                    selected_labels,
                    run_ids=[run_ids[index] for index in indices],
                    scaling=scaler,
                ),
            }
        output[factor] = levels
    return output


def _replication_direction(localization: Mapping[str, Any]) -> dict[str, Any]:
    topology = localization["factors"]["transition_topology"]
    phase = localization["factors"]["actual_entry_phase"]
    topology_fraction = topology["eligible_level_adverse_fraction"]
    phase_fraction = phase["eligible_level_adverse_fraction"]
    topology_direction = (
        topology["passes"]
        and topology_fraction.get("LEFT", -1.0) > topology_fraction.get("RIGHT", -1.0)
    )
    phase_direction = (
        phase["passes"]
        and phase_fraction.get("RIGHT_SINGLE_SUPPORT", -1.0)
        > phase_fraction.get("LEFT_SINGLE_SUPPORT", -1.0)
    )
    return {
        "transition_left_greater_than_transition_right": topology_direction,
        "right_single_greater_than_left_single": phase_direction,
        "topology_and_phase_physically_coupled": True,
        "same_frozen_direction_replicated": topology_direction and phase_direction,
    }


def run_confirmation_analysis(root: Path, config_path: Path) -> dict[str, Any]:
    """Consume Confirmation exactly once and test the frozen H1."""
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if document["experiment"]["id"] != "SAND_BENIGN_MILD_RECALIBRATED_CONFIRMATION_ANALYSIS":
        raise ValueError("unsupported Sand Confirmation analysis config")
    artifact_dir = root / str(document["artifacts"]["path"])
    guard_path = artifact_dir / "confirmation_access_guard.json"
    replay_path = artifact_dir / "v2_confirmation_replay.json"
    if guard_path.exists() or replay_path.exists():
        raise RuntimeError("the one-shot Sand Confirmation has already been consumed")
    integrity = verify_confirmation_inputs(root, config_path, document)
    scalers = reconstruct_discovery_scalers(root, document)
    discovery.write_json(artifact_dir / "discovery_scalers.json", scalers)
    discovery.write_json(
        artifact_dir / "pre_open_freeze.json",
        {
            **integrity,
            "discovery_scalers_sha256": sha256_file(
                artifact_dir / "discovery_scalers.json"
            ),
            "confirmation_guard_before": 0,
            "confirmation_payload_reads": 0,
            "v2_confirmation_replay_count": 0,
            "frozen_hypothesis": "DOMAIN_DIVERSITY_GAP_SUPPORTED",
            "no_hypothesis_substitution": True,
        },
    )
    normalizer = load_hazard_normalizer(root / str(document["model"]["normalizer"]["path"]))
    models = [
        load_checkpoint(root / str(item["path"]))[0]
        for item in document["model"]["checkpoints"]
    ]
    authorization = {
        "schema_version": 1,
        "dataset_id": document["dataset"]["id"],
        "split": CONFIRMATION_SPLIT,
        "purpose": "one_shot_frozen_DOMAIN_DIVERSITY_GAP_replication",
        "authorized_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "source_commit": document["experiment"]["starting_commit"],
        "confirmation_split_sha256": document["dataset"]["hashes"]["confirmation_split"],
        "confirmation_config_sha256": sha256_file(config_path),
        "discovery_interpretation_sha256": document["discovery"]["interpretation_sha256"],
        "final_candidate_record_sha256": document["model"]["record"]["sha256"],
        "metric_implementation_sha256": document["metric_implementation"]["sha256"],
        "guard_before": 0,
        "guard_after": 1,
        "open_count": 1,
        "consumed": True,
        "historical_holdout_guard_unchanged": 1,
    }
    discovery.write_json(guard_path, authorization)

    dataset_path = root / str(document["dataset"]["path"])
    manifest = load_mild_recalibrated_manifest(dataset_path)
    rows = {
        str(row["run_id"]): row
        for row in manifest["runs"]
        if row["split"] == CONFIRMATION_SPLIT
    }
    payloads = {
        run_id: _load_confirmation_payload(dataset_path, row, authorization)
        for run_id, row in sorted(rows.items())
    }
    vectors = _collect_vectors(rows, payloads, normalizer)
    labels = vectors["labels"]
    run_ids = vectors["run_ids"]
    frozen_scalers = scalers["scalers"]
    pelvis_analysis = {
        "contract": {
            "metric_implementation_sha256": document["metric_implementation"]["sha256"],
            "frozen_discovery_scalers_sha256": sha256_file(
                artifact_dir / "discovery_scalers.json"
            ),
            "one_anchor_vector_per_run": True,
            "confirmation_refit": False,
        },
        "endpoint_count": len(run_ids),
        "window_count": len(run_ids),
        "anchors": vectors["anchors"],
        "current_80d": separability_with_frozen_scaler(
            vectors["current"], labels, run_ids=run_ids, scaler=frozen_scalers["current"]
        ),
        "flattened_window_20x80": separability_with_frozen_scaler(
            vectors["window"], labels, run_ids=run_ids, scaler=frozen_scalers["window"]
        ),
        "factor_localized_window": _factor_metrics_with_scaler(
            vectors["window"],
            labels,
            run_ids,
            vectors["rows"],
            document["analysis"]["descriptive_factors"],
            frozen_scalers["window"],
        ),
    }
    fsr_analysis = {
        "contract": {
            "realizable": True,
            "confirmation_refit": False,
            "exact_support_spread_included": False,
            "exact_loaded_contact_included": False,
            "classifier_probe_or_fusion_training": False,
        },
        "fsr_contact_only": separability_with_frozen_scaler(
            vectors["fsr"], labels, run_ids=run_ids, scaler=frozen_scalers["fsr"]
        ),
        "pelvis_plus_fsr_contact": separability_with_frozen_scaler(
            vectors["combined"],
            labels,
            run_ids=run_ids,
            scaler=frozen_scalers["combined"],
        ),
    }
    oracle_analysis = {
        "contract": {
            "privileged": True,
            "runtime_candidate": False,
            "sensor_claim_permitted": False,
            "confirmation_refit": False,
        },
        "privileged_oracle": separability_with_frozen_scaler(
            vectors["oracle"], labels, run_ids=run_ids, scaler=frozen_scalers["oracle"]
        ),
    }
    discovery.write_json(artifact_dir / "pelvis_confirmation_analysis.json", pelvis_analysis)
    discovery.write_json(artifact_dir / "fsr_contact_confirmation_analysis.json", fsr_analysis)
    discovery.write_json(artifact_dir / "privileged_oracle_confirmation_analysis.json", oracle_analysis)

    threshold = float(document["model"]["threshold"])
    persistence = int(document["model"]["persistence_ms"])
    replay_rows: list[dict[str, Any]] = []
    for run_id, row in sorted(rows.items()):
        run = discovery.hazard_run_from_discovery(row, payloads[run_id])
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
        is_slip = row["actual_subtype"] == "SLIP"
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
        replay_rows.append(
            {
                "run_id": run_id,
                "group": row["group"],
                "physical_class": (
                    "STRICT_SAND_BENIGN"
                    if is_benign
                    else "SUPPORT"
                    if is_support
                    else "ACTUAL_SLIP"
                    if is_slip
                    else "INVALID_OR_NONPRIMARY"
                ),
                "eligible_primary_analysis": is_benign or is_support,
                "max_probability": maximum,
                "first_threshold_crossing": first_crossing,
                "first_reflex": first_reflex,
                "max_threshold_streak_ms": streak,
                "reflex": first_reflex is not None,
                "adverse_margin": bool(
                    is_benign and (first_reflex is not None or maximum >= 0.95)
                ),
                "margin_bin": _margin_bin(maximum, first_reflex is not None),
                "support_i1_sample": i1,
                "support_sample": support,
                "support_correct": support_correct,
                "pre_i1_reflex": pre_i1,
                "i1_to_reflex_ms": None
                if i1 is None or first_reflex is None
                else first_reflex - i1,
                "reflex_to_support_ms": None
                if support is None or first_reflex is None
                else support - first_reflex,
                "factors": discovery.factor_metadata(row) if is_benign else {
                    "source": str(row["source_terrain"]).upper(),
                    "speed": f"{float(row['speed_mps']):.2f}",
                    "side": str(row["support_event_summary"]["side"]),
                    "support_kind": "DELAYED"
                    if row["group"] == "delayed_support_control"
                    else "ORDINARY",
                },
            }
        )
    benign_rows = [row for row in replay_rows if row["physical_class"] == "STRICT_SAND_BENIGN"]
    support_rows = [row for row in replay_rows if row["physical_class"] == "SUPPORT"]
    slip_rows = [row for row in replay_rows if row["physical_class"] == "ACTUAL_SLIP"]
    invalid_rows = [row for row in replay_rows if row["physical_class"] == "INVALID_OR_NONPRIMARY"]
    sand_summary = {
        "all_strict_sand": discovery.summarize_benign(benign_rows),
        "mild": discovery.summarize_benign(
            [row for row in benign_rows if row["factors"]["actual_severity"] == "LOW"]
        ),
        "moderate": discovery.summarize_benign(
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
        "all": discovery.summarize_support(support_rows),
        "ordinary": discovery.summarize_support(
            [row for row in support_rows if row["factors"]["support_kind"] == "ORDINARY"]
        ),
        "delayed": discovery.summarize_support(
            [row for row in support_rows if row["factors"]["support_kind"] == "DELAYED"]
        ),
        "by_source": _support_subsets(support_rows, "source"),
        "by_side": _support_subsets(support_rows, "side"),
        "by_speed": _support_subsets(support_rows, "speed"),
    }
    replay_artifact = {
        "candidate_id": document["model"]["candidate_id"],
        "threshold": threshold,
        "persistence_ms": persistence,
        "ensemble_seeds": document["model"]["ensemble_seeds"],
        "confirmation_runs_replayed": len(replay_rows),
        "confirmation_split_replay_count": 1,
        "eligible_strict_sand": len(benign_rows),
        "eligible_support": len(support_rows),
        "actual_slip_excluded_from_benign": len(slip_rows),
        "invalid_or_nonprimary": len(invalid_rows),
        "sand": sand_summary,
        "support": support_summary,
        "actual_slip_descriptive": slip_rows,
        "run_level": replay_rows,
        "training_or_tuning": False,
        "hypothesis_substitution": False,
        "old_holdout_inference": False,
    }
    discovery.write_json(replay_path, replay_artifact)
    localization = discovery.factor_localization(
        benign_rows,
        factors=document["replication"]["metadata_localization"]["factors"],
        minimum_level_n=int(
            document["replication"]["metadata_localization"]["minimum_valid_runs_per_level"]
        ),
        fraction_range_min=float(
            document["replication"]["metadata_localization"]["adverse_fraction_range_min"]
        ),
        cramers_v_min=float(
            document["replication"]["metadata_localization"]["cramers_v_min"]
        ),
    )
    localization["adverse_runs"] = [row for row in benign_rows if row["adverse_margin"]]
    localization["frozen_direction"] = _replication_direction(localization)
    discovery.write_json(artifact_dir / "confirmation_factor_localization.json", localization)

    flags = discovery._decision_flags(
        pelvis_analysis["flattened_window_20x80"],
        fsr_analysis["pelvis_plus_fsr_contact"],
    )
    discovery_path = root / str(document["discovery"]["artifact_path"])
    discovery_pelvis = _load_json(discovery_path / "pelvis_analysis.json")["flattened_window_20x80"]
    discovery_replay = _load_json(discovery_path / "v2_discovery_replay.json")
    discovery_localization = _load_json(discovery_path / "factor_localization.json")
    direction = localization["frozen_direction"]
    support_pass = (
        support_summary["all"]["recall"] == 1.0
        and support_summary["all"]["premature_pre_i1"] == 0
    )
    h1_checks = {
        "frozen_Discovery_hypothesis_is_H1": True,
        "Confirmation_population_and_integrity": True,
        "systematic_adverse_pattern": localization["systematic_adverse_pattern"],
        "same_topology_and_phase_direction": direction["same_frozen_direction_replicated"],
        "reasonable_pelvis_separation": flags["reasonable_pelvis_separation"],
        "strong_pelvis_mixing_unsupported": not flags["strong_pelvis_mixing"],
        "realizable_fsr_material_increment_unsupported": not flags[
            "realizable_fsr_material_increment"
        ],
        "Support_controls_strong": support_pass,
        "no_hypothesis_substitution": True,
    }
    verdict = (
        "DOMAIN_DIVERSITY_GAP_CONFIRMED"
        if all(h1_checks.values())
        else "DOMAIN_DIVERSITY_GAP_NOT_CONFIRMED"
    )
    replication = {
        "frozen_Discovery_hypothesis": "DOMAIN_DIVERSITY_GAP_SUPPORTED",
        "discovery": {
            "strict_sand": discovery_replay["sand"]["all_strict_sand"],
            "mild": discovery_replay["sand"]["mild"],
            "moderate": discovery_replay["sand"]["moderate"],
            "support": discovery_replay["support"]["all"],
            "pelvis_window": discovery_pelvis,
            "localization": discovery_localization,
            "fsr_material_improvement_count": 1,
        },
        "confirmation": {
            "strict_sand": sand_summary["all_strict_sand"],
            "mild": sand_summary["mild"],
            "moderate": sand_summary["moderate"],
            "support": support_summary["all"],
            "pelvis_window": pelvis_analysis["flattened_window_20x80"],
            "localization": localization,
            "fsr_material_improvement_count": flags[
                "realizable_fsr_improvement_count"
            ],
        },
        "same_frozen_localization_direction": direction,
        "h1_checks": h1_checks,
        "confirmation_verdict": verdict,
        "alternate_hypothesis_selection": False,
    }
    discovery.write_json(
        artifact_dir / "discovery_confirmation_replication.json", replication
    )
    decision = {
        "frozen_hypothesis": "DOMAIN_DIVERSITY_GAP_SUPPORTED",
        "h1_checks": h1_checks,
        "all_h1_checks_passed": all(h1_checks.values()),
        "metric_flags": flags,
        "localization_direction": direction,
        "selected_confirmation_verdict": verdict,
        "H2_or_H3_substitution": False,
        "analysis_validity": "SAND_BENIGN_GENERALIZATION_STUDY_CONFIRMATION_ANALYSIS_VALID",
    }
    discovery.write_json(artifact_dir / "confirmation_decision.json", decision)
    component_hashes = {
        "discovery_scalers_sha256": sha256_file(artifact_dir / "discovery_scalers.json"),
        "pelvis_confirmation_analysis_sha256": sha256_file(
            artifact_dir / "pelvis_confirmation_analysis.json"
        ),
        "fsr_contact_confirmation_analysis_sha256": sha256_file(
            artifact_dir / "fsr_contact_confirmation_analysis.json"
        ),
        "privileged_oracle_confirmation_analysis_sha256": sha256_file(
            artifact_dir / "privileged_oracle_confirmation_analysis.json"
        ),
        "v2_confirmation_replay_sha256": sha256_file(replay_path),
        "confirmation_factor_localization_sha256": sha256_file(
            artifact_dir / "confirmation_factor_localization.json"
        ),
        "discovery_confirmation_replication_sha256": sha256_file(
            artifact_dir / "discovery_confirmation_replication.json"
        ),
        "confirmation_decision_sha256": sha256_file(
            artifact_dir / "confirmation_decision.json"
        ),
        "confirmation_access_guard_sha256": sha256_file(guard_path),
    }
    interpretation_body = {
        "confirmation_analysis_config_sha256": sha256_file(config_path),
        "confirmation_implementation_sha256": integrity[
            "confirmation_implementation_sha256"
        ],
        "metric_implementation_sha256": integrity["metric_implementation_sha256"],
        "semantic_dataset_freeze_sha256": document["dataset"]["hashes"][
            "semantic_dataset_freeze"
        ],
        "confirmation_split_sha256": document["dataset"]["hashes"][
            "confirmation_split"
        ],
        "discovery_interpretation_sha256": document["discovery"][
            "interpretation_sha256"
        ],
        **component_hashes,
        "frozen_Discovery_hypothesis": "DOMAIN_DIVERSITY_GAP_SUPPORTED",
        "selected_confirmation_verdict": verdict,
        "analysis_validity": "SAND_BENIGN_GENERALIZATION_STUDY_CONFIRMATION_ANALYSIS_VALID",
        "confirmation_open_count": 1,
        "confirmation_payload_deserializations": 88,
        "confirmation_v2_replay_count": 1,
        "old_holdout_payload_reads": 0,
        "old_holdout_inference": 0,
        "training_or_tuning": False,
        "future_use_status": "CONSUMED_FOR_FROZEN_H1_REPLICATION",
        "known_limitations": document["limitations"],
    }
    interpretation = {
        **interpretation_body,
        "SAND_BENIGN_CONFIRMATION_INTERPRETATION_SHA": canonical_sha256(
            interpretation_body
        ),
    }
    discovery.write_json(
        artifact_dir / "confirmation_interpretation.json", interpretation
    )
    return {
        "integrity": integrity,
        "scalers": scalers,
        "pelvis": pelvis_analysis,
        "fsr": fsr_analysis,
        "oracle": oracle_analysis,
        "replay": replay_artifact,
        "localization": localization,
        "replication": replication,
        "decision": decision,
        "interpretation": interpretation,
        "component_hashes": component_hashes,
    }


__all__ = [
    "reconstruct_discovery_scalers",
    "run_confirmation_analysis",
    "separability_with_frozen_scaler",
    "verify_confirmation_inputs",
]
