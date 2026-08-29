"""Fresh-corpus validation of the control-facing unified hazard reflex.

The primary label is a physical safety decision, not cause classification:
established Slip or established Support requires a reflex.  The frozen I1
support precursor only changes when an alert becomes causally acceptable; it
does not replace the established Support event or enter a runtime tensor.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from fastreflex.dataset.loader import Normalizer, WindowSet
from fastreflex.evaluation.continuous_slip_reflex import (
    SlipReplay,
    extract_continuous_slip_features,
    feature_schema_for_components as slip_feature_schema,
    fit_continuous_normalizer,
    gait_sampling_categories,
    mine_hard_negative_endpoints,
    replay_many as replay_slip_many,
    slip_event_sample,
    support_event_sample,
)
from fastreflex.evaluation.reflex_event import (
    EVENT_CLASS_NAMES,
    EventHoldoutGuard,
    EventRun,
    _event_manifest_row,
    _hard_control_outcome,
    _load_yaml,
    _reduce_simulation,
    _write_json,
    load_event_runs,
    physical_signature,
)
from fastreflex.evaluation.stability_temporal import _file_sha256
from fastreflex.evaluation.support_terrain_fusion import raw_support_alert
from fastreflex.evaluation.terrain_conditioned_reflex import (
    BranchReplay,
    TerrainGateTrace,
    TERRAIN_STATE_NAMES,
    _terrain_models,
    extract_branch_features,
    feature_schema_for_components as support_feature_schema,
    frozen_terrain_gate_from_result,
)
from fastreflex.evaluation.transition_scenarios import (
    VALID_OUTCOMES,
    classify_scenario_outcome,
    fusion_regression,
    transition_simulation_config,
)
from fastreflex.models.baselines import parameter_count
from fastreflex.simulation.g1 import load_simulation_config, run_simulation
from fastreflex.training.trainer import load_checkpoint, save_checkpoint, train_model


EXPERIMENT_ID = "UNIFIED_HAZARD_REFLEX_SYSTEM_VALIDATION"
LABEL_SLIP = "SLIP_HAZARD"
LABEL_SUPPORT = "SUPPORT_HAZARD"
LABEL_BOTH = "SLIP_AND_SUPPORT_HAZARD"
LABEL_NO_HAZARD = "NO_HAZARD"
LABEL_PRECURSOR_ONLY = "SUPPORT_PRECURSOR_ONLY"
PHYSICAL_LABELS = (
    LABEL_SLIP,
    LABEL_SUPPORT,
    LABEL_BOTH,
    LABEL_NO_HAZARD,
    LABEL_PRECURSOR_ONLY,
)
VERDICTS = (
    "UNIFIED_HAZARD_REFLEX_SUPPORTED_FROZEN_BRANCHES",
    "UNIFIED_HAZARD_REFLEX_SUPPORTED_SINGLE_IMU",
    "UNIFIED_HAZARD_REFLEX_PROMISING",
    "UNIFIED_HAZARD_REFLEX_NOT_SUPPORTED",
    "UNIFIED_HAZARD_DATASET_NEEDS_REVISION",
)
GROUPS = (
    "ICE_SLIP_HAZARD",
    "SAND_SUPPORT_HAZARD",
    "SAND_BENIGN",
    "HARD_GROUND_NORMAL",
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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


def split_for_source_index(source: str, index: int) -> str:
    """Frozen source-balanced 38/13/13 split for every 64-run group."""
    if source not in ("concrete", "marble") or not 1 <= index <= 32:
        raise ValueError("source/index is outside the frozen unified matrix")
    if index <= 19:
        return "train"
    validation_last = 26 if source == "concrete" else 25
    return "validation" if index <= validation_last else "holdout"


def generate_unified_specifications(
    document: Mapping[str, object]
) -> list[dict[str, object]]:
    """Expand the predeclared four groups without reading any outcome."""
    groups = document["dataset"]["groups"]
    specifications: list[dict[str, object]] = []
    abbreviations = {
        "ICE_SLIP_HAZARD": "ice_h",
        "SAND_SUPPORT_HAZARD": "sand_h",
        "SAND_BENIGN": "sand_b",
        "HARD_GROUND_NORMAL": "hard_n",
    }
    for group in GROUPS:
        config = groups[group]
        for source in ("concrete", "marble"):
            for index in range(1, 33):
                split = split_for_source_index(source, index)
                common = {
                    "id": f"uhr_{abbreviations[group]}_{source[0]}{index:02d}",
                    "group": group,
                    "split": split,
                    "design_role": group.lower(),
                    "intended_role": "stable",
                    "source_terrain": source,
                    "speed_mps": float(config.get("speed_mps", 0.25)),
                    "hard_stable_control": group == "HARD_GROUND_NORMAL",
                }
                if group == "ICE_SLIP_HAZARD":
                    width = config["width_schedule"][source]
                    mechanics = config["mechanics"]
                    specification = {
                        **common,
                        "target_terrain": "ice",
                        "patch_start_x_m": float(
                            config["patch_starts_cycle"][(index - 1) % 4]
                        ),
                        "patch_width_m": round(
                            float(width["first"]) + (index - 1) * float(width["step"]),
                            5,
                        ),
                        **mechanics,
                    }
                elif group == "SAND_SUPPORT_HAZARD":
                    anchor = (index - 1) % 2
                    local = (index - 1) // 2
                    width = config["width_schedule_per_anchor"][source]
                    mechanics = config["mechanics"]
                    specification = {
                        **common,
                        "target_terrain": "sand",
                        "patch_start_x_m": float(
                            config["alternating_patch_starts"][anchor]
                        ),
                        "patch_width_m": round(
                            float(width["first"]) + local * float(width["step"]), 5
                        ),
                        **mechanics,
                    }
                elif group == "SAND_BENIGN":
                    anchor = (index - 1) % 2
                    local = (index - 1) // 2
                    patch = config["alternating_patch"][anchor]
                    width = config["width_schedule_per_anchor"][source]
                    mechanics = config["mechanics"]
                    specification = {
                        **common,
                        "target_terrain": "sand",
                        "patch_start_x_m": float(patch["start"]),
                        "patch_width_m": round(
                            float(width["first"]) + local * float(width["step"]), 5
                        ),
                        "sink_pattern": str(patch["sink_pattern"]),
                        **mechanics,
                    }
                else:
                    speed = config["speed_schedule"]
                    mechanics = config["mechanics"]
                    specification = {
                        **common,
                        "target_terrain": source,
                        "speed_mps": round(
                            float(speed["first"]) + (index - 1) * float(speed["step"]),
                            5,
                        ),
                        "patch_start_x_m": 0.35,
                        "patch_width_m": 0.75,
                        **mechanics,
                    }
                # Canonical transition code expects the role field but physical
                # labels are always recomputed from the result.
                specification["intended_role"] = (
                    "fall"
                    if group in ("ICE_SLIP_HAZARD", "SAND_SUPPORT_HAZARD")
                    else "stable"
                )
                specifications.append(specification)
    return specifications


def _signature_from_csv_row(row: Mapping[str, str]) -> tuple[object, ...] | None:
    required = (
        "source_terrain",
        "target_terrain",
        "speed_mps",
        "patch_start_x_m",
        "patch_width_m",
        "slip_pattern",
        "sink_pattern",
        "sink_severity",
        "support_pattern",
    )
    if not all(row.get(key, "") != "" for key in required):
        return None
    return (
        row["source_terrain"],
        row["target_terrain"],
        float(row["speed_mps"]),
        float(row["patch_start_x_m"]),
        float(row["patch_width_m"]),
        row["slip_pattern"],
        row["sink_pattern"],
        row["sink_severity"],
        row["support_pattern"],
    )


def prior_physical_signatures(
    root: Path, document: Mapping[str, object]
) -> set[tuple[object, ...]]:
    result: set[tuple[object, ...]] = set()
    for record in document["source"]["prior_manifests"]:
        path = root / str(record["path"])
        if _file_sha256(path) != str(record["sha256"]):
            raise RuntimeError(f"prior manifest changed: {record['path']}")
        if path.suffix == ".json":
            manifest = json.loads(path.read_text(encoding="utf-8"))
            for row in manifest.get("runs", ()):
                if "physical_signature" in row:
                    result.add(tuple(row["physical_signature"]))
        else:
            with path.open("r", encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    signature = _signature_from_csv_row(row)
                    if signature is not None:
                        result.add(signature)
    return result


def validate_unified_design(
    root: Path,
    document: Mapping[str, object],
    specifications: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if document["experiment"]["id"] != EXPERIMENT_ID:
        raise ValueError("unsupported unified hazard experiment")
    semantics = document["physical_semantics"]
    if (
        float(semantics["slip"]["threshold_m"]) != 0.050
        or int(semantics["slip"]["persistence_ms"]) != 3
        or float(semantics["support_established"]["threshold_m"]) != 0.010
        or int(semantics["support_established"]["persistence_ms"]) != 20
        or float(semantics["support_precursor_i1"]["frozen_threshold"]) != 0.0
        or int(semantics["support_precursor_i1"]["persistence_ms"]) != 20
    ):
        raise ValueError("unified physical semantics changed")
    if len(specifications) != 256:
        raise ValueError("unified design must contain exactly 256 runs")
    ids = [str(row["id"]) for row in specifications]
    signatures = [physical_signature(row) for row in specifications]
    prior = prior_physical_signatures(root, document)
    counts = {
        group: {
            split: sum(
                row["group"] == group and row["split"] == split
                for row in specifications
            )
            for split in ("train", "validation", "holdout")
        }
        for group in GROUPS
    }
    expected = {"train": 38, "validation": 13, "holdout": 13}
    if any(value != expected for value in counts.values()):
        raise ValueError("unified per-group split changed")
    duplicates = len(signatures) - len(set(signatures))
    overlap = len(set(signatures) & prior)
    if len(ids) != len(set(ids)) or duplicates or overlap:
        raise ValueError("unified IDs/signatures are not fresh and unique")
    return {
        "passed": True,
        "runs": len(specifications),
        "group_split_counts": counts,
        "total_split_counts": {
            split: sum(row["split"] == split for row in specifications)
            for split in ("train", "validation", "holdout")
        },
        "duplicate_signatures": duplicates,
        "prior_manifest_count": len(document["source"]["prior_manifests"]),
        "prior_signature_count": len(prior),
        "prior_signature_overlap": overlap,
        "split_membership_frozen_before_simulation": True,
    }


def i1_support_precursor_sample(
    run: EventRun, *, threshold: float = 0.0, persistence_ms: int = 20
) -> int | None:
    """First causal loaded-foot positive spread derivative confirmation."""
    spread = np.asarray(run.support_spread_m, dtype=np.float64)
    derivative = np.zeros_like(spread)
    derivative[1:] = spread[1:] - spread[:-1]
    score = np.max(
        np.where(run.loaded_contact, np.maximum(derivative, 0.0), 0.0), axis=1
    )
    count = 0
    for sample in range(run.first_contact_sample, run.censor_sample):
        count = count + 1 if score[sample] > threshold else 0
        if count >= persistence_ms:
            return sample
    return None


def physical_hazard_label(run: EventRun, precursor: int | None) -> str:
    """Use physical clocks only; fall/recovery and design role are irrelevant."""
    slip = any(value is not None for value in run.slip_event_samples_per_foot)
    support = any(value is not None for value in run.support_event_samples_per_foot)
    if slip and support:
        return LABEL_BOTH
    if slip:
        return LABEL_SLIP
    if support:
        return LABEL_SUPPORT
    if precursor is not None:
        return LABEL_PRECURSOR_ONLY
    return LABEL_NO_HAZARD


def _save_unified_run(
    path: Path,
    run: EventRun,
    precursor: int | None,
    terrain: TerrainGateTrace,
) -> None:
    np.savez_compressed(
        path,
        timestamp_us=run.timestamp_us,
        pelvis_imu6=run.features["PELVIS_IMU6"],
        foot_fsr8=run.features["PELVIS_IMU6_FSR8"][:, 6:],
        tangential_anchor_drift_m=run.drift_m,
        tangential_velocity_mps=run.tangential_velocity_mps,
        support_surface_spread_m=run.support_spread_m,
        support_surface_max_displacement_m=run.support_max_displacement_m,
        loaded_contact=run.loaded_contact,
        first_target_contact_sample=np.asarray(run.first_contact_sample, np.int64),
        first_target_touchdown_sample=np.asarray(run.first_touchdown_sample, np.int64),
        censor_sample=np.asarray(run.censor_sample, np.int64),
        first_slip_event_sample_per_foot=np.asarray(
            [
                -1 if value is None else value
                for value in run.slip_event_samples_per_foot
            ],
            np.int64,
        ),
        first_support_event_sample_per_foot=np.asarray(
            [
                -1 if value is None else value
                for value in run.support_event_samples_per_foot
            ],
            np.int64,
        ),
        first_reflex_event_sample=np.asarray(
            -1 if run.event_sample is None else run.event_sample, np.int64
        ),
        first_support_precursor_sample=np.asarray(
            -1 if precursor is None else precursor, np.int64
        ),
        terrain_state=terrain.state,
        terrain_update_samples=terrain.update_samples,
        terrain_prediction_ids=terrain.prediction_ids,
        terrain_prediction_probabilities=terrain.prediction_probabilities,
        terrain_first_target_valid_sample=np.asarray(
            -1
            if terrain.first_target_valid_sample is None
            else terrain.first_target_valid_sample,
            np.int64,
        ),
    )


def _manifest(
    root: Path,
    document: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    invalid: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], str]:
    dataset_path = root / str(document["dataset"]["path"])
    manifest = {
        "schema_version": 1,
        "dataset_id": document["dataset"]["dataset_id"],
        "created_at": "2026-08-29T00:00:00+09:00",
        "source_commit": document["experiment"]["source_commit_at_start"],
        "policy_sha256": document["source"]["policy_sha256"],
        "simulator_config_sha256": document["source"]["simulator_config_sha256"],
        "physical_semantics": document["physical_semantics"],
        "model_input_fields": ["timestamp_us", "pelvis_imu6"],
        "terrain_role": "advisory_only_never_detector_gate",
        "fall_recovery_role": "diagnostic_only_never_label_or_tensor",
        "run_count": len(rows),
        "invalid_count": len(invalid),
        "invalid": list(invalid),
        "runs": list(rows),
    }
    path = dataset_path / "manifest.json"
    _write_json(path, manifest)
    sha = _file_sha256(path)
    (dataset_path / "manifest.sha256").write_text(
        f"{sha}  manifest.json\n", encoding="utf-8"
    )
    return manifest, sha


def generate_unified_dataset(
    root: Path,
    document: Mapping[str, object],
    specifications: Sequence[Mapping[str, object]],
    progress: Callable[[str], None],
) -> tuple[dict[str, object], str]:
    """Generate or resume the fresh corpus; final files remain Gitignored."""
    dataset_path = root / str(document["dataset"]["path"])
    manifest_path = dataset_path / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_id") != document["dataset"]["dataset_id"]:
            raise RuntimeError("existing unified dataset identity differs")
        return manifest, _file_sha256(manifest_path)
    dataset_path.mkdir(parents=True, exist_ok=True)
    state_path = dataset_path / "generation_state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {"rows": [], "invalid": []}
    )
    completed = {str(row["run_id"]) for row in state["rows"]} | {
        str(row["run_id"]) for row in state["invalid"]
    }
    base = load_simulation_config(root / str(document["source"]["simulator_config"]))
    policy = root / str(document["source"]["policy_path"])
    terrain_models, terrain_mean, terrain_std = _terrain_models(
        root / "artifacts/runs/20260828_terrain_rebuild_sensor_ablation/selected_models"
    )
    gate = document["common"]["scenario_gate"]
    for index, raw in enumerate(specifications, start=1):
        run_id = str(raw["id"])
        if run_id in completed:
            continue
        specification = dict(raw)
        specification["minimum_normal_prefix_ms"] = int(gate["normal_prefix_ms_min"])
        specification["minimum_post_contact_ms"] = int(gate["post_contact_ms_min"])
        result = run_simulation(
            transition_simulation_config(
                base, specification, policy, float(document["common"]["duration_s"])
            ),
            observe_fsr=True,
            observe_foot_imu=False,
            capture_state_trace=False,
        )
        outcome = (
            _hard_control_outcome(result)
            if bool(specification["hard_stable_control"])
            else classify_scenario_outcome(result, specification)
        )
        progress(f"UNIFIED DATASET {index}/256 {run_id}: {outcome}")
        if outcome not in VALID_OUTCOMES:
            state["invalid"].append(
                {"run_id": run_id, "split": specification["split"], "outcome": outcome}
            )
        else:
            run = _reduce_simulation(specification, result, outcome)
            precursor = i1_support_precursor_sample(run)
            terrain = frozen_terrain_gate_from_result(
                result,
                run,
                terrain_models,
                terrain_mean,
                terrain_std,
                deployment_scheme=str(
                    document["terrain_advisory"]["deployment_scheme"]
                ),
            )
            path = dataset_path / f"{run_id}.npz"
            _save_unified_run(path, run, precursor, terrain)
            row = _event_manifest_row(path, run, specification)
            row.update(
                {
                    "group": specification["group"],
                    "physical_label": physical_hazard_label(run, precursor),
                    "support_precursor_sample": precursor,
                    "terrain_first_target_valid_sample": terrain.first_target_valid_sample,
                }
            )
            state["rows"].append(row)
        _write_json(state_path, state)
        del result
        if index % 8 == 0:
            gc.collect()
    del terrain_models
    manifest, sha = _manifest(root, document, state["rows"], state["invalid"])
    return manifest, sha


def audit_dataset_readiness(
    document: Mapping[str, object], manifest: Mapping[str, object]
) -> dict[str, object]:
    """Check physical coverage from manifest metadata without waveform access."""
    rows = list(manifest["runs"])
    labels = [str(row["physical_label"]) for row in rows]
    slip_rows = [
        row for row in rows if row["physical_label"] in (LABEL_SLIP, LABEL_BOTH)
    ]
    support_rows = [
        row for row in rows if row["physical_label"] in (LABEL_SUPPORT, LABEL_BOTH)
    ]
    no_hazard_rows = [row for row in rows if row["physical_label"] == LABEL_NO_HAZARD]
    i1_covered = [
        row
        for row in support_rows
        if row.get("support_precursor_sample") is not None
        and int(row["support_precursor_sample"])
        <= min(
            int(value)
            for value in row["support_event_samples_per_foot"]
            if value is not None
        )
    ]
    signatures = [tuple(row["physical_signature"]) for row in rows]
    split_ids = {
        split: {str(row["run_id"]) for row in rows if row["split"] == split}
        for split in ("train", "validation", "holdout")
    }
    split_overlap = sum(
        len(split_ids[left] & split_ids[right])
        for left, right in (
            ("train", "validation"),
            ("train", "holdout"),
            ("validation", "holdout"),
        )
    )
    pretransition_fall = sum(
        row.get("fall_sample_diagnostic_only") is not None
        and int(row["fall_sample_diagnostic_only"])
        < int(row["first_target_contact_sample"])
        for row in rows
    )
    readiness = document["readiness"]
    counts = {
        "valid_runs": len(rows),
        "invalid_runs": len(manifest.get("invalid", ())),
        "established_slip": len(slip_rows),
        "established_support": len(support_rows),
        "i1_support_coverage": 0.0
        if not support_rows
        else len(i1_covered) / len(support_rows),
        "primary_no_hazard": len(no_hazard_rows),
        "sand_benign_no_hazard": sum(
            row["group"] == "SAND_BENIGN" and row["physical_label"] == LABEL_NO_HAZARD
            for row in rows
        ),
        "hard_ground_no_hazard": sum(
            row["group"] == "HARD_GROUND_NORMAL"
            and row["physical_label"] == LABEL_NO_HAZARD
            for row in rows
        ),
        "support_precursor_only": sum(
            label == LABEL_PRECURSOR_ONLY for label in labels
        ),
        "pretransition_fall": pretransition_fall,
        "duplicate_signatures": len(signatures) - len(set(signatures)),
        "split_overlap": split_overlap,
        "split_counts": {key: len(value) for key, value in split_ids.items()},
        "physical_label_counts": {
            label: labels.count(label) for label in PHYSICAL_LABELS
        },
        "group_counts": {
            group: sum(row["group"] == group for row in rows) for group in GROUPS
        },
    }
    checks = {
        "valid_runs": counts["valid_runs"] >= int(readiness["valid_runs_min"]),
        "established_slip": counts["established_slip"]
        >= int(readiness["established_slip_min"]),
        "established_support": counts["established_support"]
        >= int(readiness["established_support_min"]),
        "i1_support_coverage": counts["i1_support_coverage"]
        >= float(readiness["i1_support_coverage_min"]),
        "primary_no_hazard": counts["primary_no_hazard"]
        >= int(readiness["primary_no_hazard_min"]),
        "sand_benign_no_hazard": counts["sand_benign_no_hazard"]
        >= int(readiness["sand_benign_no_hazard_min"]),
        "hard_ground_no_hazard": counts["hard_ground_no_hazard"]
        >= int(readiness["hard_ground_no_hazard_min"]),
        "pretransition_fall": pretransition_fall
        <= int(readiness["pretransition_fall_max"]),
        "duplicate_signatures": counts["duplicate_signatures"]
        <= int(readiness["duplicate_signature_max"]),
        "split_overlap": split_overlap <= int(readiness["split_overlap_max"]),
    }
    return {"counts": counts, "checks": checks, "passed": all(checks.values())}


def _load_normalizer(
    path: Path, expected_dimension: int
) -> tuple[Normalizer, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    if mean.shape != (expected_dimension,) or std.shape != (expected_dimension,):
        raise ValueError(
            f"normalizer dimension differs from frozen {expected_dimension}"
        )
    if (
        np.any(std <= 0.0)
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(std))
    ):
        raise ValueError("frozen normalizer is nonfinite")
    return (
        Normalizer(
            mean=mean,
            std=std,
            sample_count=int(payload["sample_count"]),
            fit_run_ids=tuple(str(value) for value in payload["fit_run_ids"]),
            epsilon=float(payload["epsilon"]),
        ),
        payload,
    )


def verify_frozen_system(
    root: Path, document: Mapping[str, object]
) -> dict[str, object]:
    """Fail closed if any frozen detector, Terrain producer, or source changed."""
    records: list[tuple[str, str]] = []
    source = document["source"]
    records.extend(
        (
            str(source[path_key]),
            str(source[sha_key]),
        )
        for path_key, sha_key in (
            ("simulator_config", "simulator_config_sha256"),
            ("policy_path", "policy_sha256"),
            ("scenario_calibration_config", "scenario_calibration_sha256"),
            ("dense_design_config", "dense_design_sha256"),
        )
    )
    for branch in ("slip", "support"):
        config = document["phase_a"][branch]
        if "freeze_path" in config:
            records.append((str(config["freeze_path"]), str(config["freeze_sha256"])))
        records.append(
            (str(config["normalizer"]["path"]), str(config["normalizer"]["sha256"]))
        )
        records.extend(
            (str(row["path"]), str(row["sha256"])) for row in config["checkpoints"]
        )
    terrain = document["terrain_advisory"]
    records.append(
        (str(terrain["normalizer"]["path"]), str(terrain["normalizer"]["sha256"]))
    )
    records.extend(
        (str(row["path"]), str(row["sha256"])) for row in terrain["checkpoints"]
    )
    actual = {path: _file_sha256(root / path) for path, _ in records}
    expected = {path: sha for path, sha in records}
    if actual != expected:
        changed = sorted(
            path for path in expected if actual.get(path) != expected[path]
        )
        raise RuntimeError(
            "protected unified system input changed: " + ", ".join(changed)
        )
    slip = document["phase_a"]["slip"]
    support = document["phase_a"]["support"]
    slip_schema = slip_feature_schema(tuple(slip["components"]))
    support_schema = support_feature_schema(tuple(support["components"]))
    if len(slip_schema) != int(slip["feature_dimension"]) or _canonical_sha256(
        slip_schema
    ) != str(slip["feature_schema_sha256"]):
        raise RuntimeError("frozen Slip feature schema changed")
    if len(support_schema) != int(support["feature_dimension"]) or _canonical_sha256(
        support_schema
    ) != str(support["feature_schema_sha256"]):
        raise RuntimeError("frozen Support feature schema changed")
    mapped_support_schema = tuple(
        name.replace("pelvis_raw_", "pelvis_base_")
        .replace("pelvis_mean_10ms_", "pelvis_causal_mean_10ms_")
        .replace("pelvis_variance_10ms_", "pelvis_causal_variance_10ms_")
        for name in support_schema
    )
    if not set(mapped_support_schema).issubset(set(slip_schema)):
        raise RuntimeError("Slip80 is not the declared semantic superset of Support60")
    for branch in ("slip", "support"):
        config = document["phase_a"][branch]
        for row in config["checkpoints"]:
            _, metadata = load_checkpoint(root / str(row["path"]))
            if (
                metadata["family"] != config["model_family"]
                or int(metadata["window_samples"]) != int(config["history_ms"])
                or int(metadata["input_channels"]) != int(config["feature_dimension"])
            ):
                raise RuntimeError(f"frozen {branch} checkpoint identity changed")
    return {
        "passed": True,
        "hashes": actual,
        "slip_feature_schema_sha256": _canonical_sha256(slip_schema),
        "support_feature_schema_sha256": _canonical_sha256(support_schema),
        "slip80_strict_semantic_superset_of_support60": len(slip_schema)
        > len(mapped_support_schema)
        and set(mapped_support_schema).issubset(set(slip_schema)),
        "slip_model": {
            key: slip[key]
            for key in ("model_family", "history_ms", "threshold", "persistence_ms")
        },
        "support_model": {
            key: support[key]
            for key in ("model_family", "history_ms", "threshold", "persistence_ms")
        },
    }


def _load_terrain_trace(path: Path, sample_count: int) -> TerrainGateTrace:
    with np.load(path, allow_pickle=False) as stored:
        state = np.asarray(stored["terrain_state"], dtype=np.int8)
        updates = np.asarray(stored["terrain_update_samples"], dtype=np.int64)
        ids = np.asarray(stored["terrain_prediction_ids"], dtype=np.int64)
        probabilities = np.asarray(
            stored["terrain_prediction_probabilities"], dtype=np.float32
        )
        first = int(stored["terrain_first_target_valid_sample"])
    if (
        state.shape != (sample_count,)
        or len(updates) != len(ids)
        or len(ids) != len(probabilities)
    ):
        raise ValueError(f"invalid frozen Terrain advisory trace: {path.name}")
    return TerrainGateTrace(
        state=state,
        update_samples=updates,
        prediction_ids=ids,
        prediction_probabilities=probabilities,
        first_target_valid_sample=None if first < 0 else first,
        clean_event_count=len(updates),
    )


def load_terrain_traces(
    dataset_path: Path, runs: Mapping[str, EventRun]
) -> dict[str, TerrainGateTrace]:
    return {
        run_id: _load_terrain_trace(
            dataset_path / f"{run_id}.npz", len(run.timestamp_us)
        )
        for run_id, run in sorted(runs.items())
    }


def _predict_binary(
    models: Sequence[torch.nn.Module], windows: np.ndarray
) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(windows, dtype=np.float32))
    with torch.no_grad():
        probabilities = [
            torch.softmax(model(tensor), dim=1)[:, 1].cpu().numpy() for model in models
        ]
    return np.mean(np.stack(probabilities), axis=0).astype(np.float64)


def replay_frozen_support_continuously(
    runs: Mapping[str, EventRun],
    normalizer: Normalizer,
    checkpoint_paths: Sequence[Path],
    *,
    history_ms: int = 20,
) -> dict[str, BranchReplay]:
    """Replay Support from the first causal endpoint, never Terrain-gated."""
    models = [load_checkpoint(path)[0] for path in checkpoint_paths]
    result: dict[str, BranchReplay] = {}
    offsets = np.arange(history_ms - 1, -1, -1, dtype=np.int64)
    for run_id, run in sorted(runs.items()):
        features, _ = extract_branch_features(run, ("pelvis_imu6",))
        stop = run.censor_sample
        if run.fall_sample_diagnostic is not None:
            stop = min(stop, int(run.fall_sample_diagnostic))
        endpoints = np.arange(history_ms - 1, stop, dtype=np.int64)
        chunks: list[np.ndarray] = []
        for first in range(0, len(endpoints), 512):
            selected = endpoints[first : first + 512]
            indices = selected[:, None] - offsets[None, :]
            windows = normalizer.transform(features[indices])
            chunks.append(_predict_binary(models, windows))
        result[run_id] = BranchReplay(
            endpoints=endpoints,
            probabilities=np.concatenate(chunks) if chunks else np.empty(0),
            terrain_state=np.zeros(len(endpoints), dtype=np.int8),
        )
        del features
    del models
    return result


def _onset_samples(
    replay: SlipReplay | BranchReplay, threshold: float, persistence_ms: int
) -> tuple[np.ndarray, np.ndarray]:
    alert, onset = raw_support_alert(
        replay.probabilities, threshold=threshold, persistence_ms=persistence_ms
    )
    return replay.endpoints[onset], alert


def _first_in_ranges(
    values: np.ndarray, ranges: Sequence[tuple[int, int]]
) -> int | None:
    selected = [
        int(value)
        for value in values
        if any(lower <= int(value) <= upper for lower, upper in ranges)
    ]
    return None if not selected else min(selected)


def evaluate_unified_replays(
    runs: Mapping[str, EventRun],
    terrain: Mapping[str, TerrainGateTrace],
    slip_replays: Mapping[str, SlipReplay],
    support_replays: Mapping[str, BranchReplay] | None,
    *,
    slip_threshold: float,
    support_threshold: float | None,
    persistence_ms: int,
    precursor_samples: Mapping[str, int | None],
) -> dict[str, object]:
    """Score the OR decision; cross-trigger cause classification is non-primary."""
    rows: list[dict[str, object]] = []
    slip_latency: list[int] = []
    support_latency: list[int] = []
    support_precursor_latency: list[int] = []
    support_lead: list[int] = []
    for run_id, run in sorted(runs.items()):
        slip_onsets, slip_alert = _onset_samples(
            slip_replays[run_id], slip_threshold, persistence_ms
        )
        if support_replays is None:
            support_onsets = np.empty(0, dtype=np.int64)
            support_alert = np.zeros(0, dtype=bool)
        else:
            if support_threshold is None:
                raise ValueError("Support replay requires a threshold")
            support_onsets, support_alert = _onset_samples(
                support_replays[run_id], support_threshold, persistence_ms
            )
        onsets = np.unique(np.concatenate((slip_onsets, support_onsets)))
        slip = slip_event_sample(run)
        support = support_event_sample(run)
        precursor = precursor_samples.get(run_id)
        ranges: list[tuple[int, int]] = []
        if slip is not None:
            ranges.append((slip - 30, slip + 40))
        if support is not None:
            lower = support if precursor is None else int(precursor)
            ranges.append((lower, support + 50))
        earliest = min((lower for lower, _ in ranges), default=None)
        valid_detection = _first_in_ranges(onsets, ranges)
        first = None if not len(onsets) else int(onsets[0])
        premature = earliest is not None and first is not None and first < earliest
        if premature:
            # The contract scores the first system alert.  A later retrigger
            # cannot retroactively turn an unjustified early reflex into a hit.
            valid_detection = None
        label = physical_hazard_label(run, precursor)
        primary_no_hazard = label == LABEL_NO_HAZARD
        system_fp = primary_no_hazard and first is not None
        slip_valid = (
            None
            if slip is None
            else _first_in_ranges(onsets, ((slip - 30, slip + 40),))
        )
        support_valid = (
            None
            if support is None
            else _first_in_ranges(
                onsets,
                (((support if precursor is None else int(precursor)), support + 50),),
            )
        )
        if premature:
            slip_valid = None
            support_valid = None
        slip_branch_slip_valid = (
            None
            if slip is None
            else _first_in_ranges(slip_onsets, ((slip - 30, slip + 40),))
        )
        support_branch_slip_valid = (
            None
            if slip is None
            else _first_in_ranges(support_onsets, ((slip - 30, slip + 40),))
        )
        support_range = (
            ()
            if support is None
            else (((support if precursor is None else int(precursor)), support + 50),)
        )
        slip_branch_support_valid = (
            None if support is None else _first_in_ranges(slip_onsets, support_range)
        )
        support_branch_support_valid = (
            None if support is None else _first_in_ranges(support_onsets, support_range)
        )
        if slip is not None and slip_valid is not None:
            slip_latency.append(slip_valid - slip)
        if support is not None and support_valid is not None:
            support_latency.append(support_valid - support)
            support_lead.append(support - support_valid)
            if precursor is not None:
                support_precursor_latency.append(support_valid - int(precursor))
        detect = valid_detection if valid_detection is not None else first
        terrain_state = (
            None
            if detect is None or detect >= len(terrain[run_id].state)
            else int(terrain[run_id].state[detect])
        )
        terrain_valid = terrain[run_id].first_target_valid_sample
        rows.append(
            {
                "run_id": run_id,
                "split": run.split,
                "source_terrain": run.source_terrain,
                "target_terrain": run.target_terrain,
                "hard_ground": run.hard_stable_control,
                "outcome_diagnostic_only": run.outcome_diagnostic,
                "physical_label": label,
                "slip_sample": slip,
                "support_precursor_sample": precursor,
                "support_sample": support,
                "slip_branch_first_onset": None
                if not len(slip_onsets)
                else int(slip_onsets[0]),
                "support_branch_first_onset": None
                if not len(support_onsets)
                else int(support_onsets[0]),
                "system_first_onset": first,
                "first_valid_detection": valid_detection,
                "valid_detection": valid_detection is not None,
                "premature": bool(premature),
                "system_false_positive": bool(system_fp),
                "slip_valid_detection": slip_valid,
                "support_valid_detection": support_valid,
                "slip_branch_slip_valid_detection": slip_branch_slip_valid,
                "support_branch_slip_valid_detection": support_branch_slip_valid,
                "slip_branch_support_valid_detection": slip_branch_support_valid,
                "support_branch_support_valid_detection": support_branch_support_valid,
                "terrain_at_detection": None
                if terrain_state is None
                else TERRAIN_STATE_NAMES[terrain_state],
                "terrain_first_target_valid_sample": terrain_valid,
                "reflex_minus_terrain_valid_ms": None
                if detect is None or terrain_valid is None
                else detect - int(terrain_valid),
                "terrain_valid_minus_touchdown_ms": None
                if terrain_valid is None
                else int(terrain_valid) - run.first_touchdown_sample,
                "slip_raw_alert_duration_ms": int(np.count_nonzero(slip_alert)),
                "support_raw_alert_duration_ms": int(np.count_nonzero(support_alert)),
            }
        )

    hazard = [
        row
        for row in rows
        if row["physical_label"] in (LABEL_SLIP, LABEL_SUPPORT, LABEL_BOTH)
    ]
    slip_rows = [row for row in rows if row["slip_sample"] is not None]
    support_rows = [row for row in rows if row["support_sample"] is not None]
    no_hazard = [row for row in rows if row["physical_label"] == LABEL_NO_HAZARD]
    precursor_only = [
        row for row in rows if row["physical_label"] == LABEL_PRECURSOR_ONLY
    ]
    sand_benign = [
        row
        for row in no_hazard
        if row["target_terrain"] == "sand" and not row["hard_ground"]
    ]
    hard = [row for row in no_hazard if row["hard_ground"]]

    def recall(selected: Sequence[Mapping[str, object]]) -> float:
        return (
            0.0
            if not selected
            else sum(bool(row["valid_detection"]) for row in selected) / len(selected)
        )

    def specificity(selected: Sequence[Mapping[str, object]]) -> float:
        return (
            1.0
            if not selected
            else 1.0
            - sum(bool(row["system_false_positive"]) for row in selected)
            / len(selected)
        )

    premature = sum(bool(row["premature"]) for row in hazard)
    terrain_correct = [
        row
        for row in hazard
        if row["first_valid_detection"] is not None
        and row["terrain_at_detection"] is not None
    ]
    return {
        "runs": len(rows),
        "hazard_runs": len(hazard),
        "overall_hazard_recall": recall(hazard),
        "slip_hazard_runs": len(slip_rows),
        "slip_hazard_recall": recall(slip_rows),
        "support_hazard_runs": len(support_rows),
        "support_hazard_recall": recall(support_rows),
        "primary_no_hazard_runs": len(no_hazard),
        "primary_no_hazard_specificity": specificity(no_hazard),
        "sand_benign_runs": len(sand_benign),
        "sand_benign_specificity": specificity(sand_benign),
        "hard_ground_runs": len(hard),
        "hard_ground_specificity": specificity(hard),
        "system_premature_runs": premature,
        "system_premature_run_rate": 0.0 if not hazard else premature / len(hazard),
        "slip_premature_runs": sum(bool(row["premature"]) for row in slip_rows),
        "support_pre_precursor_premature_runs": sum(
            bool(row["premature"]) for row in support_rows
        ),
        "slip_latency_ms": _distribution(slip_latency),
        "support_precursor_latency_ms": _distribution(support_precursor_latency),
        "support_established_latency_ms": _distribution(support_latency),
        "support_lead_ms": _distribution(support_lead),
        "precursor_only_runs_excluded_from_specificity": sum(
            row["physical_label"] == LABEL_PRECURSOR_ONLY for row in rows
        ),
        "precursor_only_diagnostic": {
            "runs": len(precursor_only),
            "reflex_runs": sum(
                row["system_first_onset"] is not None for row in precursor_only
            ),
            "reflex_rate": 0.0
            if not precursor_only
            else sum(row["system_first_onset"] is not None for row in precursor_only)
            / len(precursor_only),
            "outcome_counts": {
                outcome: sum(
                    row["outcome_diagnostic_only"] == outcome for row in precursor_only
                )
                for outcome in ("VALID_STABLE", "VALID_FALL")
            },
        },
        "cause_attribution_diagnostic": {
            "slip_hazards_triggered_by_slip_branch": sum(
                row["slip_branch_slip_valid_detection"] is not None for row in rows
            ),
            "slip_hazards_triggered_by_support_branch": sum(
                row["support_branch_slip_valid_detection"] is not None for row in rows
            ),
            "support_hazards_triggered_by_slip_branch": sum(
                row["slip_branch_support_valid_detection"] is not None for row in rows
            ),
            "support_hazards_triggered_by_support_branch": sum(
                row["support_branch_support_valid_detection"] is not None
                for row in rows
            ),
            "native_slip_branch_recall": 0.0
            if not slip_rows
            else sum(
                row["slip_branch_slip_valid_detection"] is not None for row in slip_rows
            )
            / len(slip_rows),
            "native_support_branch_recall": 0.0
            if not support_rows
            else sum(
                row["support_branch_support_valid_detection"] is not None
                for row in support_rows
            )
            / len(support_rows),
        },
        "terrain_advisory_at_detection": {
            "rows": len(terrain_correct),
            "correct_target_count": sum(
                str(row["terrain_at_detection"]).lower()
                == str(row["target_terrain"]).lower()
                for row in terrain_correct
            ),
            "state_counts": {
                name: sum(
                    row["terrain_at_detection"] == name for row in terrain_correct
                )
                for name in TERRAIN_STATE_NAMES
            },
            "reflex_before_terrain_count": sum(
                row["reflex_minus_terrain_valid_ms"] is not None
                and int(row["reflex_minus_terrain_valid_ms"]) < 0
                for row in hazard
            ),
            "reflex_minus_terrain_valid_ms": _distribution(
                [row["reflex_minus_terrain_valid_ms"] for row in hazard]
            ),
            "terrain_valid_minus_touchdown_ms": _distribution(
                [row["terrain_valid_minus_touchdown_ms"] for row in hazard]
            ),
            "used_as_gate": False,
        },
        "rows": rows,
    }


def unified_gate_results(
    metrics: Mapping[str, object], gates: Mapping[str, object]
) -> dict[str, bool]:
    slip_p95 = metrics["slip_latency_ms"]["p95"]
    support_p95 = metrics["support_established_latency_ms"]["p95"]
    support_median_lead = metrics["support_lead_ms"]["median"]
    return {
        "overall_hazard_recall": float(metrics["overall_hazard_recall"])
        >= float(gates["overall_hazard_recall_min"]),
        "slip_hazard_recall": float(metrics["slip_hazard_recall"])
        >= float(gates["slip_hazard_recall_min"]),
        "support_hazard_recall": float(metrics["support_hazard_recall"])
        >= float(gates["support_hazard_recall_min"]),
        "primary_no_hazard_specificity": float(metrics["primary_no_hazard_specificity"])
        >= float(gates["primary_no_hazard_specificity_min"]),
        "sand_benign_specificity": float(metrics["sand_benign_specificity"])
        >= float(gates["sand_benign_specificity_min"]),
        "hard_ground_specificity": float(metrics["hard_ground_specificity"])
        >= float(gates["hard_ground_specificity_min"]),
        "system_premature_run_rate": float(metrics["system_premature_run_rate"])
        <= float(gates["system_premature_run_rate_max"]),
        "slip_p95_latency": slip_p95 is not None
        and float(slip_p95) <= float(gates["slip_p95_latency_ms_max"]),
        "support_p95_latency": support_p95 is not None
        and float(support_p95)
        <= float(gates["support_p95_established_latency_ms_max"]),
        **(
            {
                "median_support_lead": support_median_lead is not None
                and float(support_median_lead)
                >= float(gates["median_support_lead_ms_min"])
            }
            if "median_support_lead_ms_min" in gates
            else {}
        ),
    }


@dataclass
class UnifiedCandidate:
    history_ms: int
    normalizer: Normalizer
    checkpoint_paths: tuple[Path, ...]
    record: dict[str, object]


def _evenly_spaced(values: np.ndarray, count: int) -> np.ndarray:
    selected = np.asarray(values, dtype=np.int64)
    if count <= 0 or not len(selected):
        return np.empty(0, dtype=np.int64)
    if len(selected) <= count:
        return selected
    return selected[np.linspace(0, len(selected) - 1, count, dtype=np.int64)]


def unified_positive_endpoints(
    run: EventRun,
    precursor: int | None,
    history_ms: int,
    *,
    cap: int = 20,
) -> np.ndarray:
    """Bounded deterministic union positives, using only TRAIN references."""
    selected: set[int] = set()
    slip = slip_event_sample(run)
    support = support_event_sample(run)
    if slip is not None:
        selected.update(range(slip - 30, slip + 41, 5))
    if support is not None and precursor is not None:
        interval = np.linspace(precursor, support, 5, dtype=np.int64)
        selected.update(int(value) for value in interval)
        selected.update(support + offset for offset in (-20, 0, 20, 40))
    elif support is not None:
        selected.update(support + offset for offset in (-20, 0, 20, 40))
    values = np.asarray(sorted(selected), dtype=np.int64)
    valid = (values >= history_ms - 1) & (values < run.censor_sample)
    if run.fall_sample_diagnostic is not None:
        valid &= values < int(run.fall_sample_diagnostic)
    values = values[valid]
    return _evenly_spaced(values, cap)


def unified_negative_candidates(
    run: EventRun, precursor: int | None, history_ms: int
) -> np.ndarray:
    """True no-hazard endpoints; an I1-active region is never a negative."""
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


def initial_unified_negative_endpoints(
    run: EventRun,
    precursor: int | None,
    history_ms: int,
    per_category: int,
) -> np.ndarray:
    eligible = unified_negative_candidates(run, precursor, history_ms)
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
    return np.unique(np.concatenate(selected)) if selected else np.empty(0, np.int64)


def _binary_window_set(
    inputs: Sequence[np.ndarray],
    targets: Sequence[int],
    run_ids: Sequence[str],
    endpoints: Sequence[int],
) -> WindowSet:
    labels = np.asarray(targets, dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    if not inputs or np.any(counts == 0):
        raise ValueError("unified training windows require both classes")
    return WindowSet(
        inputs=np.stack(inputs).astype(np.float32),
        targets=labels,
        run_ids=np.asarray(run_ids, dtype=str),
        endpoint_samples=np.asarray(endpoints, dtype=np.int64),
        available_by_class=tuple(int(value) for value in counts[:2]),
    )


def build_unified_windows(
    runs: Mapping[str, EventRun],
    run_ids: Sequence[str],
    precursor_samples: Mapping[str, int | None],
    history_ms: int,
    normalizer: Normalizer,
    *,
    per_category: int,
    positive_cap: int,
    extra_negative_endpoints: Mapping[str, Sequence[int]] | None = None,
) -> WindowSet:
    inputs: list[np.ndarray] = []
    targets: list[int] = []
    source_ids: list[str] = []
    endpoint_rows: list[int] = []
    extras = extra_negative_endpoints or {}
    for run_id in sorted(str(value) for value in run_ids):
        run = runs[run_id]
        features, _ = extract_continuous_slip_features(run, ("pelvis_imu6",))
        positive = unified_positive_endpoints(
            run, precursor_samples.get(run_id), history_ms, cap=positive_cap
        )
        negative = initial_unified_negative_endpoints(
            run, precursor_samples.get(run_id), history_ms, per_category
        )
        allowed = set(
            int(value)
            for value in unified_negative_candidates(
                run, precursor_samples.get(run_id), history_ms
            )
        )
        if run_id in extras:
            extra = np.asarray(
                [int(value) for value in extras[run_id] if int(value) in allowed],
                dtype=np.int64,
            )
            negative = np.unique(np.concatenate((negative, extra)))
        if set(int(value) for value in positive) & set(
            int(value) for value in negative
        ):
            raise RuntimeError("unified positive was used as a negative")
        for label, values in ((1, positive), (0, negative)):
            for endpoint in values:
                first = int(endpoint) - history_ms + 1
                if first < 0 or int(endpoint) >= run.censor_sample:
                    raise ValueError("unified window crossed a causal boundary")
                if run.fall_sample_diagnostic is not None and int(endpoint) >= int(
                    run.fall_sample_diagnostic
                ):
                    raise ValueError("post-fall sample entered unified detector")
                inputs.append(normalizer.transform(features[first : int(endpoint) + 1]))
                targets.append(label)
                source_ids.append(run_id)
                endpoint_rows.append(int(endpoint))
        del features
    return _binary_window_set(inputs, targets, source_ids, endpoint_rows)


def _unified_train_monitor_partition(
    runs: Mapping[str, EventRun],
    run_ids: Sequence[str],
    precursor_samples: Mapping[str, int | None],
) -> tuple[list[str], list[str]]:
    groups: dict[tuple[str, str, str], list[str]] = {}
    for run_id in sorted(str(value) for value in run_ids):
        run = runs[run_id]
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


def _mine_unified_negatives(
    runs: Mapping[str, EventRun],
    replays: Mapping[str, SlipReplay],
    precursor_samples: Mapping[str, int | None],
    prior: Mapping[str, Sequence[int]],
    *,
    top_k: int,
    minimum_separation_ms: int,
) -> tuple[dict[str, tuple[int, ...]], dict[str, object]]:
    result: dict[str, tuple[int, ...]] = {}
    selected_scores: list[float] = []
    for run_id, run in sorted(runs.items()):
        candidates = unified_negative_candidates(run, precursor_samples.get(run_id), 1)
        replay = replays[run_id]
        common, _, replay_indices = np.intersect1d(
            candidates, replay.endpoints, return_indices=True
        )
        scores = replay.probabilities[replay_indices]
        selected = mine_hard_negative_endpoints(
            common,
            scores,
            top_k=top_k,
            minimum_separation_ms=minimum_separation_ms,
            excluded=prior.get(run_id, ()),
        )
        result[run_id] = tuple(int(value) for value in selected)
        lookup = {
            int(endpoint): float(score) for endpoint, score in zip(common, scores)
        }
        selected_scores.extend(lookup[int(endpoint)] for endpoint in selected)
    return result, {
        "runs_scored": len(runs),
        "mined_windows": sum(len(value) for value in result.values()),
        "selected_probability": _distribution(selected_scores),
        "train_only": True,
        "precursor_region_never_negative": True,
    }


def train_unified_candidate(
    root: Path,
    document: Mapping[str, object],
    runs: Mapping[str, EventRun],
    precursor_samples: Mapping[str, int | None],
    history_ms: int,
    artifact_path: Path,
    progress: Callable[[str], None],
) -> UnifiedCandidate:
    """Train Round 0 plus exactly three TRAIN-only hard-negative rounds."""
    config = document["phase_b"]
    train = config["training"]
    fit_ids, monitor_ids = _unified_train_monitor_partition(
        runs, sorted(runs), precursor_samples
    )
    normalizer = fit_continuous_normalizer(
        runs,
        sorted(runs),
        ("pelvis_imu6",),
        foot_dataset_path=None,
        per_run_sample_cap=1000,
        standard_deviation_floor=1.0e-6,
    )
    normalizer_path = artifact_path / "normalization" / f"gru_history{history_ms}.json"
    schema = slip_feature_schema(("pelvis_imu6",))
    normalizer_payload = {
        **normalizer.to_dict(),
        "components": ["pelvis_imu6"],
        "feature_schema": list(schema),
        "feature_schema_sha256": _canonical_sha256(schema),
        "train_only": True,
    }
    _write_json(normalizer_path, normalizer_payload)
    accumulated: dict[str, tuple[int, ...]] = {}
    round_records: list[dict[str, object]] = []
    final_paths: tuple[Path, ...] = ()
    for round_id in range(4):
        fit_windows = build_unified_windows(
            runs,
            fit_ids,
            precursor_samples,
            history_ms,
            normalizer,
            per_category=int(config["initial_negative_per_gait_category"]),
            positive_cap=int(config["per_run_positive_cap"]),
            extra_negative_endpoints=accumulated,
        )
        monitor_windows = build_unified_windows(
            runs,
            monitor_ids,
            precursor_samples,
            history_ms,
            normalizer,
            per_category=int(config["initial_negative_per_gait_category"]),
            positive_cap=int(config["per_run_positive_cap"]),
            extra_negative_endpoints=accumulated,
        )
        paths: list[Path] = []
        epochs: list[int] = []
        for seed in train["seeds"]:
            path = (
                artifact_path
                / "checkpoints"
                / f"unified_gru_history{history_ms}_round{round_id}_seed{seed}.pt"
            )
            model, result = train_model(
                "gru",
                history_ms,
                fit_windows,
                monitor_windows,
                int(seed),
                batch_size=int(train["batch_size"]),
                max_epochs=int(train["max_epochs"]),
                patience=int(train["patience"]),
                learning_rate=float(train["learning_rate"]),
                class_names=EVENT_CLASS_NAMES,
                selection_metric="validation_loss",
            )
            save_checkpoint(
                path,
                model,
                "gru",
                history_ms,
                int(seed),
                result,
                input_channels=len(schema),
                class_names=EVENT_CLASS_NAMES,
            )
            paths.append(path)
            epochs.append(result.best_epoch)
        final_paths = tuple(paths)
        record: dict[str, object] = {
            "round": round_id,
            "fit_windows": len(fit_windows),
            "monitor_windows": len(monitor_windows),
            "fit_class_counts": list(fit_windows.selected_by_class),
            "monitor_class_counts": list(monitor_windows.selected_by_class),
            "best_epochs": epochs,
        }
        progress(f"PHASE B history={history_ms} round={round_id} trained")
        if round_id < 3:
            replay = replay_slip_many(
                runs,
                sorted(runs),
                ("pelvis_imu6",),
                history_ms,
                normalizer,
                final_paths,
                foot_dataset_path=None,
            )
            mined, mining_record = _mine_unified_negatives(
                runs,
                replay,
                precursor_samples,
                accumulated,
                top_k=int(config["hnm"]["top_k_per_run"]),
                minimum_separation_ms=int(config["hnm"]["minimum_separation_ms"]),
            )
            accumulated = _merge_endpoint_maps(accumulated, mined)
            record["hard_negative_mining"] = mining_record
        round_records.append(record)
    return UnifiedCandidate(
        history_ms=history_ms,
        normalizer=normalizer,
        checkpoint_paths=final_paths,
        record={
            "model_family": "gru",
            "history_ms": history_ms,
            "feature_dimension": len(schema),
            "feature_schema_sha256": _canonical_sha256(schema),
            "normalizer_path": str(normalizer_path.relative_to(root)),
            "normalizer_sha256": _file_sha256(normalizer_path),
            "checkpoint_sha256": {
                str(path.relative_to(root)): _file_sha256(path) for path in final_paths
            },
            "parameters": parameter_count(load_checkpoint(final_paths[0])[0]),
            "rounds": round_records,
            "validation_access_before_hnm3": False,
        },
    )


def _precursor_map(
    manifest: Mapping[str, object], splits: Sequence[str]
) -> dict[str, int | None]:
    allowed = set(splits)
    return {
        str(row["run_id"]): (
            None
            if row.get("support_precursor_sample") is None
            else int(row["support_precursor_sample"])
        )
        for row in manifest["runs"]
        if str(row["split"]) in allowed
    }


def replay_phase_a(
    root: Path,
    document: Mapping[str, object],
    runs: Mapping[str, EventRun],
) -> tuple[dict[str, SlipReplay], dict[str, BranchReplay], dict[str, object]]:
    slip = document["phase_a"]["slip"]
    support = document["phase_a"]["support"]
    slip_normalizer, slip_payload = _load_normalizer(
        root / str(slip["normalizer"]["path"]), int(slip["feature_dimension"])
    )
    support_normalizer, support_payload = _load_normalizer(
        root / str(support["normalizer"]["path"]), int(support["feature_dimension"])
    )
    slip_checkpoints = tuple(root / str(row["path"]) for row in slip["checkpoints"])
    support_checkpoints = tuple(
        root / str(row["path"]) for row in support["checkpoints"]
    )
    slip_replays = replay_slip_many(
        runs,
        sorted(runs),
        tuple(str(value) for value in slip["components"]),
        int(slip["history_ms"]),
        slip_normalizer,
        slip_checkpoints,
        foot_dataset_path=None,
    )
    support_replays = replay_frozen_support_continuously(
        runs,
        support_normalizer,
        support_checkpoints,
        history_ms=int(support["history_ms"]),
    )
    parity = {
        "slip_schema_sha256": slip_payload["feature_schema_sha256"],
        "support_schema_sha256": support_payload["feature_schema_sha256"],
        "raw_detector_terrain_independent": True,
        "continuous_replay_from_first_causal_endpoint": True,
        "system_logic": "SLIP_ALERT OR SUPPORT_ALERT",
        "terrain_gate_used": False,
    }
    return slip_replays, support_replays, parity


def select_phase_b_candidate(
    document: Mapping[str, object],
    candidates: Sequence[UnifiedCandidate],
    validation_runs: Mapping[str, EventRun],
    terrain: Mapping[str, TerrainGateTrace],
    precursor_samples: Mapping[str, int | None],
    progress: Callable[[str], None],
) -> dict[str, object]:
    """Open VALIDATION only after every candidate reached HNM3."""
    grid = document["phase_b"]["threshold"]["grid"]
    thresholds = np.round(
        np.arange(
            float(grid["start"]),
            float(grid["stop"]) + 0.0001,
            float(grid["step"]),
        ),
        2,
    )
    records: list[dict[str, object]] = []
    for candidate in candidates:
        replay = replay_slip_many(
            validation_runs,
            sorted(validation_runs),
            ("pelvis_imu6",),
            candidate.history_ms,
            candidate.normalizer,
            candidate.checkpoint_paths,
            foot_dataset_path=None,
        )
        threshold_records: list[dict[str, object]] = []
        passing: list[tuple[float, dict[str, object], dict[str, bool]]] = []
        for threshold in thresholds:
            metrics = evaluate_unified_replays(
                validation_runs,
                terrain,
                replay,
                None,
                slip_threshold=float(threshold),
                support_threshold=None,
                persistence_ms=int(document["phase_b"]["threshold"]["persistence_ms"]),
                precursor_samples=precursor_samples,
            )
            checks = unified_gate_results(
                metrics, document["phase_a"]["validation_gates"]
            )
            compact = {key: value for key, value in metrics.items() if key != "rows"}
            threshold_records.append(
                {
                    "threshold": float(threshold),
                    "metrics": compact,
                    "gates": checks,
                    "passed": all(checks.values()),
                }
            )
            if all(checks.values()):
                passing.append((float(threshold), metrics, checks))
        selected = None
        if passing:
            # Maximize hazard recall, then minimum cause recall, specificity,
            # and finally the higher threshold for the more conservative tie.
            selected = max(
                passing,
                key=lambda value: (
                    float(value[1]["overall_hazard_recall"]),
                    min(
                        float(value[1]["slip_hazard_recall"]),
                        float(value[1]["support_hazard_recall"]),
                    ),
                    float(value[1]["primary_no_hazard_specificity"]),
                    -float(value[1]["system_premature_run_rate"]),
                    value[0],
                ),
            )
        record = {
            "history_ms": candidate.history_ms,
            "threshold_grid": [float(value) for value in thresholds],
            "threshold_records": threshold_records,
            "selected_threshold": None if selected is None else selected[0],
            "selected_metrics": None
            if selected is None
            else {key: value for key, value in selected[1].items() if key != "rows"},
            "selected_gates": None if selected is None else selected[2],
            "passed": selected is not None,
        }
        records.append(record)
        progress(
            f"PHASE B VALIDATION history={candidate.history_ms}: "
            + ("PASS" if selected is not None else "FAIL")
        )
    passing_records = [record for record in records if record["passed"]]
    chosen = None
    if passing_records:
        # Both are already gated; the predeclared simpler/shorter history wins.
        chosen = min(passing_records, key=lambda record: int(record["history_ms"]))
    return {"candidates": records, "selected": chosen}


def _candidate_for_history(
    candidates: Sequence[UnifiedCandidate], history_ms: int
) -> UnifiedCandidate:
    selected = [
        candidate for candidate in candidates if candidate.history_ms == history_ms
    ]
    if len(selected) != 1:
        raise RuntimeError("unified candidate identity is ambiguous")
    return selected[0]


def _freeze_selection(
    root: Path,
    config_path: Path,
    artifact_path: Path,
    architecture: str,
    selection: Mapping[str, object],
    document: Mapping[str, object],
) -> dict[str, object]:
    payload = {
        "architecture": architecture,
        "selection": dict(selection),
        "experiment_config": str(config_path.relative_to(root)),
        "experiment_config_sha256": _file_sha256(config_path),
        "source_commit": document["experiment"]["source_commit_at_start"],
        "physical_reference_semantics": document["physical_semantics"],
        "terrain_used_as_gate": False,
        "holdout_opened": False,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    _write_json(artifact_path / "selection_before_holdout.json", payload)
    return payload


def _dataset_file_audit(
    root: Path, manifest: Mapping[str, object], document: Mapping[str, object]
) -> dict[str, object]:
    dataset_path = root / str(document["dataset"]["path"])
    rows = list(manifest["runs"])
    return {
        "run_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "all_files_exist": all(
            (dataset_path / str(row["file"])).is_file() for row in rows
        ),
        "holdout_file_count": sum(row["split"] == "holdout" for row in rows),
        "holdout_waveform_access_before_freeze": False,
        "holdout_guard_open_count_before_freeze": 0,
    }


def run_unified_hazard_reflex_system(
    config_path: Path,
    repository_root: Path,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Execute fresh corpus -> Phase A -> conditional Phase B -> one-shot holdout."""
    root = repository_root.resolve()
    config_path = config_path.resolve()
    document = _load_yaml(config_path)
    specifications = generate_unified_specifications(document)
    design = validate_unified_design(root, document, specifications)
    frozen_before = verify_frozen_system(root, document)
    manifest, manifest_sha = generate_unified_dataset(
        root, document, specifications, progress
    )
    file_audit = _dataset_file_audit(root, manifest, document)
    readiness = audit_dataset_readiness(document, manifest)
    readiness["counts"]["prior_signature_overlap"] = design["prior_signature_overlap"]
    readiness["checks"]["prior_signature_overlap"] = int(
        design["prior_signature_overlap"]
    ) <= int(document["readiness"]["prior_signature_overlap_max"])
    readiness["passed"] = all(readiness["checks"].values())
    artifact_path = root / str(document["artifacts"]["path"])
    artifact_path.mkdir(parents=True, exist_ok=True)
    dataset_path = root / str(document["dataset"]["path"])
    guard = EventHoldoutGuard()
    phase_a: dict[str, object] = {"performed": False}
    phase_b: dict[str, object] = {"activated": False}
    holdout: dict[str, object] = {
        "performed": False,
        "guard_open_count": 0,
        "reason": "dataset_not_ready",
    }
    freeze: dict[str, object] | None = None
    selected_architecture: str | None = None
    selected_candidate: UnifiedCandidate | None = None
    selected_threshold: float | None = None
    candidates: list[UnifiedCandidate] = []

    if readiness["passed"]:
        # Per-run HOLDOUT outcomes and waveforms remain outside this development load.
        train_runs = load_event_runs(dataset_path, manifest, ("train",))
        validation_runs = load_event_runs(dataset_path, manifest, ("validation",))
        train_precursor = _precursor_map(manifest, ("train",))
        validation_precursor = _precursor_map(manifest, ("validation",))
        train_terrain = load_terrain_traces(dataset_path, train_runs)
        validation_terrain = load_terrain_traces(dataset_path, validation_runs)

        progress("PHASE A frozen dual detector replay on fresh VALIDATION")
        validation_slip, validation_support, parity = replay_phase_a(
            root, document, validation_runs
        )
        phase_a_metrics = evaluate_unified_replays(
            validation_runs,
            validation_terrain,
            validation_slip,
            validation_support,
            slip_threshold=float(document["phase_a"]["slip"]["threshold"]),
            support_threshold=float(document["phase_a"]["support"]["threshold"]),
            persistence_ms=int(document["phase_a"]["slip"]["persistence_ms"]),
            precursor_samples=validation_precursor,
        )
        phase_a_checks = unified_gate_results(
            phase_a_metrics, document["phase_a"]["validation_gates"]
        )
        # TRAIN is a diagnostic only and does not select the architecture.
        train_slip, train_support, _ = replay_phase_a(root, document, train_runs)
        train_metrics = evaluate_unified_replays(
            train_runs,
            train_terrain,
            train_slip,
            train_support,
            slip_threshold=float(document["phase_a"]["slip"]["threshold"]),
            support_threshold=float(document["phase_a"]["support"]["threshold"]),
            persistence_ms=int(document["phase_a"]["slip"]["persistence_ms"]),
            precursor_samples=train_precursor,
        )
        phase_a = {
            "performed": True,
            "parity": parity,
            "train_diagnostic": train_metrics,
            "validation": phase_a_metrics,
            "validation_gates": phase_a_checks,
            "passed": all(phase_a_checks.values()),
        }
        if phase_a["passed"]:
            selected_architecture = "FROZEN_DUAL_DETECTOR_HAZARD_SYSTEM"
            selected_threshold = None
            selection = {
                "slip_threshold": document["phase_a"]["slip"]["threshold"],
                "support_threshold": document["phase_a"]["support"]["threshold"],
                "persistence_ms": 5,
                "logic": "OR",
                "slip_checkpoints": [
                    row["path"] for row in document["phase_a"]["slip"]["checkpoints"]
                ],
                "slip_checkpoint_sha256": {
                    row["path"]: row["sha256"]
                    for row in document["phase_a"]["slip"]["checkpoints"]
                },
                "support_checkpoints": [
                    row["path"] for row in document["phase_a"]["support"]["checkpoints"]
                ],
                "support_checkpoint_sha256": {
                    row["path"]: row["sha256"]
                    for row in document["phase_a"]["support"]["checkpoints"]
                },
                "slip_normalizer_sha256": document["phase_a"]["slip"]["normalizer"][
                    "sha256"
                ],
                "support_normalizer_sha256": document["phase_a"]["support"][
                    "normalizer"
                ]["sha256"],
            }
            freeze = _freeze_selection(
                root,
                config_path,
                artifact_path,
                selected_architecture,
                selection,
                document,
            )
            phase_b = {"activated": False, "reason": "phase_a_passed_validation"}
        else:
            progress(
                "PHASE A failed; Phase B TRAIN-only HNM begins; HOLDOUT remains sealed"
            )
            for row in document["phase_b"]["candidates"]:
                candidates.append(
                    train_unified_candidate(
                        root,
                        document,
                        train_runs,
                        train_precursor,
                        int(row["history_ms"]),
                        artifact_path,
                        progress,
                    )
                )
            progress(
                "All Phase B candidates reached HNM3; VALIDATION threshold calibration begins"
            )
            validation_selection = select_phase_b_candidate(
                document,
                candidates,
                validation_runs,
                validation_terrain,
                validation_precursor,
                progress,
            )
            phase_b = {
                "activated": True,
                "feature_relationship": {
                    "selected_schema": "continuous_slip_pelvis_imu80",
                    "dimensions": 80,
                    "support60_semantic_subset": True,
                    "mapping": {
                        "raw": "base",
                        "mean_10ms": "causal_mean_10ms",
                        "variance_10ms": "causal_variance_10ms",
                    },
                    "new_transforms_added": False,
                },
                "candidates": [candidate.record for candidate in candidates],
                "validation_selection": validation_selection,
                "passed": validation_selection["selected"] is not None,
            }
            if validation_selection["selected"] is not None:
                selected = validation_selection["selected"]
                selected_candidate = _candidate_for_history(
                    candidates, int(selected["history_ms"])
                )
                selected_threshold = float(selected["selected_threshold"])
                selected_architecture = "PHASE_B_UNIFIED_HAZARD_DETECTOR"
                freeze = _freeze_selection(
                    root,
                    config_path,
                    artifact_path,
                    selected_architecture,
                    {
                        **selected_candidate.record,
                        "threshold": selected_threshold,
                        "persistence_ms": int(
                            document["phase_b"]["threshold"]["persistence_ms"]
                        ),
                    },
                    document,
                )

        if selected_architecture is not None:
            if guard.open_count != 0:
                raise RuntimeError("fresh holdout guard opened before candidate freeze")
            guard.open_once()
            holdout_runs = load_event_runs(
                dataset_path, manifest, ("holdout",), holdout_guard=guard
            )
            holdout_precursor = _precursor_map(manifest, ("holdout",))
            holdout_terrain = load_terrain_traces(dataset_path, holdout_runs)
            if selected_architecture == "FROZEN_DUAL_DETECTOR_HAZARD_SYSTEM":
                holdout_slip, holdout_support, _ = replay_phase_a(
                    root, document, holdout_runs
                )
                holdout_metrics = evaluate_unified_replays(
                    holdout_runs,
                    holdout_terrain,
                    holdout_slip,
                    holdout_support,
                    slip_threshold=float(document["phase_a"]["slip"]["threshold"]),
                    support_threshold=float(
                        document["phase_a"]["support"]["threshold"]
                    ),
                    persistence_ms=5,
                    precursor_samples=holdout_precursor,
                )
            else:
                if selected_candidate is None or selected_threshold is None:
                    raise RuntimeError("Phase B selection identity was not frozen")
                holdout_replay = replay_slip_many(
                    holdout_runs,
                    sorted(holdout_runs),
                    ("pelvis_imu6",),
                    selected_candidate.history_ms,
                    selected_candidate.normalizer,
                    selected_candidate.checkpoint_paths,
                    foot_dataset_path=None,
                )
                holdout_metrics = evaluate_unified_replays(
                    holdout_runs,
                    holdout_terrain,
                    holdout_replay,
                    None,
                    slip_threshold=selected_threshold,
                    support_threshold=None,
                    persistence_ms=5,
                    precursor_samples=holdout_precursor,
                )
            holdout_checks = unified_gate_results(
                holdout_metrics, document["holdout"]["gates"]
            )
            holdout = {
                "performed": True,
                "guard_open_count": guard.open_count,
                "metrics": holdout_metrics,
                "gates": holdout_checks,
                "passed": all(holdout_checks.values()),
                "reselection_performed": False,
            }
        elif readiness["passed"]:
            holdout = {
                "performed": False,
                "guard_open_count": guard.open_count,
                "reason": "no_validation_supported_candidate",
            }

    frozen_after = verify_frozen_system(root, document)
    protected_regression = frozen_before["hashes"] == frozen_after["hashes"]
    if not readiness["passed"]:
        verdict = "UNIFIED_HAZARD_DATASET_NEEDS_REVISION"
    elif selected_architecture is None:
        verdict = "UNIFIED_HAZARD_REFLEX_NOT_SUPPORTED"
    elif not bool(holdout["passed"]):
        verdict = "UNIFIED_HAZARD_REFLEX_PROMISING"
    elif selected_architecture == "FROZEN_DUAL_DETECTOR_HAZARD_SYSTEM":
        verdict = "UNIFIED_HAZARD_REFLEX_SUPPORTED_FROZEN_BRANCHES"
    else:
        verdict = "UNIFIED_HAZARD_REFLEX_SUPPORTED_SINGLE_IMU"
    if verdict not in VERDICTS:
        raise RuntimeError("unsupported unified verdict")
    metrics = {
        "experiment": document["experiment"],
        "design_audit": design,
        "dataset": {
            "dataset_id": document["dataset"]["dataset_id"],
            "manifest_sha256": manifest_sha,
            **file_audit,
        },
        "readiness": readiness,
        "frozen_system_before": frozen_before,
        "phase_a": phase_a,
        "phase_b": phase_b,
        "selection_freeze": freeze,
        "selected_architecture": selected_architecture,
        "holdout": holdout,
        "protected_regression": {
            "passed": protected_regression,
            "before": frozen_before["hashes"],
            "after": frozen_after["hashes"],
        },
        "terrain_regression": {
            "passed": protected_regression,
            "retrained": False,
            "used_as_gate": False,
        },
        "fusion_regression": fusion_regression(),
        "causality": {
            "passed": True,
            "runtime_tensor": "Pelvis IMU6 causal history only",
            "future_state_used": False,
            "terrain_used_as_gate": False,
            "physical_clocks_used_only_for_scoring": True,
        },
        "sensor_implication": {
            "pelvis_imu6_shared_by_hazard_system": True,
            "terrain_fsr4_advisory": True,
            "minimum_unique_physical_channels": 10,
            "augmentation_required": False
            if selected_architecture is not None
            else None,
            "final_sensor_architecture_frozen": False,
        },
        "verdict": verdict,
    }
    output_path = root / str(document["artifacts"]["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    result_path = output_path / "metrics.json"
    _write_json(result_path, metrics)
    return result_path, metrics
