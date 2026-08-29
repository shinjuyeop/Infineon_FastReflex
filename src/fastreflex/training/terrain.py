"""Terrain touchdown sensor ablation, selection, and sealed holdout evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import yaml

from fastreflex.dataset.terrain import (
    SENSOR_PROFILE_CHANNELS,
    TERRAIN_CLASS_NAMES,
    HoldoutGuard,
    TerrainNormalizer,
    TerrainWindowSet,
    build_terrain_windows,
    fit_terrain_normalizer,
    load_terrain_collection_config,
    read_event_index,
    select_capped_events,
    validate_terrain_dataset,
)
from fastreflex.evaluation.metrics import classification_metrics
from fastreflex.models.baselines import parameter_count
from fastreflex.training.trainer import (
    TrainingResult,
    load_checkpoint,
    save_checkpoint,
    train_model,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_windows(
    windows: TerrainWindowSet,
    normalizer: TerrainNormalizer,
) -> TerrainWindowSet:
    return TerrainWindowSet(
        inputs=normalizer.transform(windows.inputs),
        targets=windows.targets,
        run_ids=windows.run_ids,
        event_ids=windows.event_ids,
        feet=windows.feet,
        touchdown_samples=windows.touchdown_samples,
    )


def _predict_logits(model: torch.nn.Module, windows: TerrainWindowSet) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(windows.inputs)).cpu().numpy()
    if logits.shape != (len(windows), 4) or not np.all(np.isfinite(logits)):
        raise ValueError("terrain model produced invalid logits")
    return logits


def _ensemble_metrics(
    models: Sequence[torch.nn.Module],
    windows: TerrainWindowSet,
) -> tuple[dict[str, object], np.ndarray]:
    logits = np.mean([_predict_logits(model, windows) for model in models], axis=0)
    predictions = np.argmax(logits, axis=1).astype(np.int64)
    return (
        classification_metrics(
            windows.targets,
            predictions,
            windows.run_ids,
            TERRAIN_CLASS_NAMES,
        ),
        predictions,
    )


def _mean_seed_summary(seed_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    macro = np.asarray(
        [float(row["validation"]["macro_f1"]) for row in seed_rows],
        dtype=np.float64,
    )
    recall_by_class = {
        name: np.asarray(
            [float(row["validation"]["per_class"][name]["recall"]) for row in seed_rows],
            dtype=np.float64,
        )
        for name in TERRAIN_CLASS_NAMES
    }
    recall_mean = {name: float(values.mean()) for name, values in recall_by_class.items()}
    return {
        "validation_macro_f1_mean": float(macro.mean()),
        "validation_macro_f1_std": float(macro.std()),
        "validation_per_class_recall_mean": recall_mean,
        "validation_per_class_recall_std": {
            name: float(values.std()) for name, values in recall_by_class.items()
        },
        "validation_worst_class_recall_mean": min(recall_mean.values()),
    }


def _train_candidate(
    dataset_path: Path,
    train_rows: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]],
    profile: str,
    family: str,
    horizon_ms: int,
    seeds: Sequence[int],
    settings: Mapping[str, object],
    progress: Callable[[str], None],
) -> dict[str, object]:
    raw_train = build_terrain_windows(
        dataset_path, train_rows, profile, horizon_ms
    )
    normalizer = fit_terrain_normalizer(raw_train)
    train_windows = _normalized_windows(raw_train, normalizer)
    raw_validation = build_terrain_windows(
        dataset_path, validation_rows, profile, horizon_ms
    )
    validation_windows = _normalized_windows(raw_validation, normalizer)
    if min(train_windows.selected_by_class) == 0:
        raise ValueError("terrain train events do not cover all four classes")
    if min(validation_windows.selected_by_class) == 0:
        raise ValueError("terrain validation events do not cover all four classes")
    models: list[torch.nn.Module] = []
    training_results: list[TrainingResult] = []
    seed_rows: list[dict[str, object]] = []
    for seed in seeds:
        progress(f"[train] {profile}/{family}/{horizon_ms}ms seed={seed}")
        model, result = train_model(
            family,
            horizon_ms,
            train_windows,
            validation_windows,
            int(seed),
            batch_size=int(settings["batch_size"]),
            max_epochs=int(settings["max_epochs"]),
            patience=int(settings["patience"]),
            learning_rate=float(settings["learning_rate"]),
            class_names=TERRAIN_CLASS_NAMES,
        )
        models.append(model)
        training_results.append(result)
        seed_rows.append(
            {
                "seed": int(seed),
                "best_epoch": result.best_epoch,
                "epochs_completed": result.epochs_completed,
                "validation": result.best_validation,
            }
        )
    ensemble, ensemble_predictions = _ensemble_metrics(models, validation_windows)
    validation_by_foot = {}
    for foot in ("left", "right"):
        mask = validation_windows.feet == foot
        validation_by_foot[foot] = (
            classification_metrics(
                validation_windows.targets[mask],
                ensemble_predictions[mask],
                validation_windows.run_ids[mask],
                TERRAIN_CLASS_NAMES,
            )
            if np.count_nonzero(mask)
            else None
        )
    first_model = models[0]
    summary: dict[str, object] = {
        "candidate_id": f"{profile}_{family}_{horizon_ms}ms",
        "profile": profile,
        "family": family,
        "horizon_ms": horizon_ms,
        "input_channels": SENSOR_PROFILE_CHANNELS[profile],
        "parameter_count": parameter_count(first_model),
        "train_event_count": len(train_windows),
        "validation_event_count": len(validation_windows),
        "train_class_counts": dict(zip(TERRAIN_CLASS_NAMES, train_windows.selected_by_class)),
        "validation_class_counts": dict(zip(TERRAIN_CLASS_NAMES, validation_windows.selected_by_class)),
        "normalizer": normalizer.to_dict(),
        "seeds": seed_rows,
        "validation_ensemble": ensemble,
        "validation_by_foot": validation_by_foot,
        **_mean_seed_summary(seed_rows),
    }
    return {
        "summary": summary,
        "models": models,
        "training_results": training_results,
        "normalizer": normalizer,
        "train_windows": train_windows,
        "validation_windows": validation_windows,
    }


def _bilateral_validation_signature(summary: Mapping[str, object]) -> dict[str, object]:
    """Keep only deterministic fields needed to prove historical reconstruction."""
    seeds = summary["seeds"]
    return {
        "candidate_id": summary["candidate_id"],
        "profile": summary["profile"],
        "family": summary["family"],
        "horizon_ms": summary["horizon_ms"],
        "input_channels": summary["input_channels"],
        "parameter_count": summary["parameter_count"],
        "train_event_count": summary["train_event_count"],
        "validation_event_count": summary["validation_event_count"],
        "normalizer_mean": summary["normalizer"]["mean"],
        "normalizer_std": summary["normalizer"]["std"],
        "seed_results": [
            {
                "seed": row["seed"],
                "best_epoch": row["best_epoch"],
                "epochs_completed": row["epochs_completed"],
                "macro_f1": row["validation"]["macro_f1"],
                "confusion_matrix": row["validation"]["confusion_matrix"],
            }
            for row in seeds
        ],
        "validation_macro_f1_mean": summary["validation_macro_f1_mean"],
        "validation_worst_class_recall_mean": summary[
            "validation_worst_class_recall_mean"
        ],
        "ensemble_macro_f1": summary["validation_ensemble"]["macro_f1"],
        "ensemble_confusion_matrix": summary["validation_ensemble"][
            "confusion_matrix"
        ],
        "left_macro_f1": summary["validation_by_foot"]["left"]["macro_f1"],
        "right_macro_f1": summary["validation_by_foot"]["right"]["macro_f1"],
    }


def _nested_numeric_parity(
    actual: object, expected: object, tolerance: float, path: str = "root"
) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise RuntimeError(f"bilateral validation parity keys differ at {path}")
        for key in expected:
            _nested_numeric_parity(
                actual[key], expected[key], tolerance, f"{path}.{key}"
            )
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise RuntimeError(f"bilateral validation parity shape differs at {path}")
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
            _nested_numeric_parity(
                actual_value, expected_value, tolerance, f"{path}[{index}]"
            )
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not np.isclose(
            float(actual), float(expected), rtol=0.0, atol=tolerance
        ):
            raise RuntimeError(
                f"bilateral validation parity differs at {path}: "
                f"{actual!r} != {expected!r}"
            )
        return
    if actual != expected:
        raise RuntimeError(
            f"bilateral validation parity differs at {path}: "
            f"{actual!r} != {expected!r}"
        )


def reconstruct_bilateral_shared_candidate(
    contract: Mapping[str, object],
    repository_root: Path,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Rebuild the historical shared-foot candidate without opening Terrain holdout.

    This is deliberately narrower than the original sensor-ablation runner: it
    executes exactly the already-recorded FSR4/MLP/50 ms TRAIN/VALIDATION
    contract and rejects any validation difference before exposing checkpoints.
    """
    root = repository_root.resolve()
    original_config_path = root / str(contract["original_config"])
    original_events_path = root / str(contract["original_events"])
    original_metrics_path = root / str(contract["historical_metrics"])
    original_dataset_config_path = root / str(contract["original_dataset_config"])
    declared = (
        (original_config_path, str(contract["original_config_sha256"])),
        (original_events_path, str(contract["original_events_sha256"])),
        (original_metrics_path, str(contract["historical_metrics_sha256"])),
        (
            original_dataset_config_path,
            str(contract["original_dataset_config_sha256"]),
        ),
    )
    for path, expected_sha in declared:
        if _file_sha256(path) != expected_sha:
            raise RuntimeError(f"frozen bilateral reconstruction input changed: {path}")

    output = root / str(contract["artifact_path"])
    provenance_path = output / "reconstruction.json"
    checkpoint_paths = tuple(
        output / f"seed_{int(seed)}.pt" for seed in contract["seeds"]
    )
    normalizer_path = output / "normalization.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        files = (*checkpoint_paths, normalizer_path)
        if not all(path.is_file() for path in files):
            raise RuntimeError("bilateral reconstruction artifact is incomplete")
        for path in files:
            expected = provenance["artifact_hashes"][path.name]
            if _file_sha256(path) != expected:
                raise RuntimeError(f"bilateral reconstruction artifact changed: {path}")
        for path in checkpoint_paths:
            _, metadata = load_checkpoint(path)
            if (
                metadata["family"] != "mlp"
                or int(metadata["window_samples"]) != 50
                or int(metadata["input_channels"]) != 4
            ):
                raise RuntimeError("reconstructed Terrain checkpoint contract changed")
        return output, provenance

    original_config = yaml.safe_load(original_config_path.read_text(encoding="utf-8"))
    settings = original_config["training"]
    required_settings = {
        "profile": "fsr4",
        "family": "mlp",
        "observation_ms": 50,
        "input_channels": 4,
        "seeds": list(settings["seeds"]),
        "batch_size": int(settings["batch_size"]),
        "max_epochs": int(settings["max_epochs"]),
        "patience": int(settings["patience"]),
        "learning_rate": float(settings["learning_rate"]),
        "max_events_per_run_class": int(
            original_config["common"]["max_clean_events_per_class_per_run"]
        ),
    }
    for key, expected in required_settings.items():
        if contract[key] != expected:
            raise RuntimeError(f"bilateral reconstruction contract changed: {key}")

    dataset_path = root / str(contract["original_dataset"])
    rows = read_event_index(original_events_path)
    cap = int(contract["max_events_per_run_class"])
    train_rows = select_capped_events(
        rows, "train", cap, required_horizon_ms=50
    )
    validation_rows = select_capped_events(
        rows, "validation", cap, required_horizon_ms=50
    )
    candidate = _train_candidate(
        dataset_path,
        train_rows,
        validation_rows,
        "fsr4",
        "mlp",
        50,
        tuple(int(value) for value in contract["seeds"]),
        settings,
        progress,
    )
    historical = json.loads(original_metrics_path.read_text(encoding="utf-8"))[
        "deployment"
    ]["bilateral_shared"]["validation"]
    actual_signature = _bilateral_validation_signature(candidate["summary"])
    expected_signature = _bilateral_validation_signature(historical)
    _nested_numeric_parity(
        actual_signature,
        expected_signature,
        float(contract["parity_tolerance"]),
    )

    output.mkdir(parents=True, exist_ok=False)
    for model, result, seed, checkpoint in zip(
        candidate["models"],
        candidate["training_results"],
        contract["seeds"],
        checkpoint_paths,
    ):
        save_checkpoint(
            checkpoint,
            model,
            "mlp",
            50,
            int(seed),
            result,
            input_channels=4,
            class_names=TERRAIN_CLASS_NAMES,
        )
    _write_json(normalizer_path, candidate["normalizer"].to_dict())
    provenance = {
        "status": contract["reconstruction_label"],
        "terrain_holdout_access_count": 0,
        "support_task_used_for_training_or_selection": False,
        "historical_validation_parity": True,
        "parity_tolerance": float(contract["parity_tolerance"]),
        "validation_signature": actual_signature,
        "source_hashes": {str(path.relative_to(root)): sha for path, sha in declared},
        "artifact_hashes": {
            path.name: _file_sha256(path)
            for path in (*checkpoint_paths, normalizer_path)
        },
    }
    _write_json(provenance_path, provenance)
    return output, provenance


def _qualified(summary: Mapping[str, object], macro_min: float, recall_min: float) -> bool:
    return bool(
        float(summary["validation_macro_f1_mean"]) >= macro_min
        and float(summary["validation_worst_class_recall_mean"]) >= recall_min
    )


def select_minimum_sensor(
    summaries: Sequence[Mapping[str, object]],
    macro_min: float,
    recall_min: float,
    tolerance: float,
) -> tuple[Mapping[str, object], dict[str, object]]:
    """Apply qualification, near-best band, then minimum channel count."""
    qualified = [
        row for row in summaries if _qualified(row, macro_min, recall_min)
    ]
    pool = qualified if qualified else list(summaries)
    best = max(float(row["validation_macro_f1_mean"]) for row in pool)
    contenders = [
        row
        for row in pool
        if best - float(row["validation_macro_f1_mean"]) <= tolerance
    ]
    selected = min(
        contenders,
        key=lambda row: (
            int(row["input_channels"]),
            -float(row["validation_worst_class_recall_mean"]),
            -float(row["validation_macro_f1_mean"]),
            str(row["profile"]),
        ),
    )
    return selected, {
        "qualification_available": bool(qualified),
        "qualified_profiles": [str(row["profile"]) for row in qualified],
        "best_qualified_macro_f1": best if qualified else None,
        "near_best_tolerance": tolerance,
        "contenders": [str(row["profile"]) for row in contenders],
        "rule": "qualified_then_fewer_channels_within_two_percentage_points",
    }


def select_model_family(
    summaries: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    """Select on validation only; exact ties prefer the simpler MLP."""
    return max(
        summaries,
        key=lambda row: (
            float(row["validation_macro_f1_mean"]),
            float(row["validation_worst_class_recall_mean"]),
            str(row["family"]) == "mlp",
        ),
    )


def select_shortest_horizon(
    summaries: Sequence[Mapping[str, object]],
    macro_min: float,
    recall_min: float,
) -> tuple[Mapping[str, object], bool]:
    """Choose the first passing causal horizon, or retain 50 ms as revision evidence."""
    by_horizon = {int(row["horizon_ms"]): row for row in summaries}
    for horizon in (20, 30, 50):
        row = by_horizon[horizon]
        if _qualified(row, macro_min, recall_min):
            return row, True
    return by_horizon[50], False


def _recall_subset(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    class_id: int,
) -> dict[str, object]:
    relevant = mask & (targets == class_id)
    support = int(np.count_nonzero(relevant))
    correct = int(np.count_nonzero(relevant & (predictions == class_id)))
    return {
        "support": support,
        "correct": correct,
        "recall": float(correct / support) if support else None,
    }


def _holdout_robustness(
    windows: TerrainWindowSet,
    predictions: np.ndarray,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_event = {str(row["event_id"]): row for row in rows}
    ordered = [by_event[str(event_id)] for event_id in windows.event_ids]
    observed_fall = np.asarray([bool(row["observed_fall"]) for row in ordered])
    source = np.asarray([str(row["source_terrain"]) for row in ordered])
    ice_id = TERRAIN_CLASS_NAMES.index("ICE")
    sand_id = TERRAIN_CLASS_NAMES.index("SAND")
    result: dict[str, object] = {
        "ice": {
            "stable": _recall_subset(windows.targets, predictions, ~observed_fall, ice_id),
            "fall": _recall_subset(windows.targets, predictions, observed_fall, ice_id),
        },
        "sand": {
            "stable": _recall_subset(windows.targets, predictions, ~observed_fall, sand_id),
            "fall": _recall_subset(windows.targets, predictions, observed_fall, sand_id),
        },
        "source_terrain": {},
        "foot": {},
    }
    target_mask = np.isin(windows.targets, (ice_id, sand_id))
    for name in ("concrete", "marble"):
        mask = source == name
        subset = mask & target_mask
        support = int(np.count_nonzero(subset))
        result["source_terrain"][name] = {
            "support": support,
            "accuracy": (
                float(np.mean(predictions[subset] == windows.targets[subset]))
                if support
                else None
            ),
            "ice_recall": _recall_subset(windows.targets, predictions, mask, ice_id),
            "sand_recall": _recall_subset(windows.targets, predictions, mask, sand_id),
        }
    for foot in ("left", "right"):
        mask = windows.feet == foot
        if np.count_nonzero(mask):
            result["foot"][foot] = classification_metrics(
                windows.targets[mask],
                predictions[mask],
                windows.run_ids[mask],
                TERRAIN_CLASS_NAMES,
            )
        else:
            result["foot"][foot] = None
    return result


def _delay_summary(values_ms: Sequence[float], total_runs: int) -> dict[str, object]:
    values = np.asarray(values_ms, dtype=np.float64)
    return {
        "evaluable_runs": int(len(values)),
        "unavailable_runs": int(total_runs - len(values)),
        "median_ms": None if not len(values) else float(np.median(values)),
        "p95_ms": None if not len(values) else float(np.percentile(values, 95)),
        "max_ms": None if not len(values) else float(np.max(values)),
    }


def terrain_update_delays(
    dataset_path: Path,
    all_event_rows: Sequence[Mapping[str, object]],
    horizon_ms: int,
) -> dict[str, object]:
    """Compare sensor-placement delay, separate from model inference time."""
    with (dataset_path / "manifest.csv").open("r", encoding="utf-8", newline="") as stream:
        import csv

        manifest = list(csv.DictReader(stream))
    bilateral: list[float] = []
    left_only: list[float] = []
    event_by_run: dict[str, list[Mapping[str, object]]] = {}
    for row in all_event_rows:
        if bool(row["window_50ms_valid"]) and bool(row["is_target_terrain"]):
            event_by_run.setdefault(str(row["run_id"]), []).append(row)
    total = 0
    for run in manifest:
        if not run["first_target_contact_us"]:
            continue
        total += 1
        first_contact_us = int(run["first_target_contact_us"])
        candidates = event_by_run.get(run["run_id"], [])
        if candidates:
            first = min(int(row["touchdown_us"]) for row in candidates)
            bilateral.append((first + horizon_ms * 1000 - first_contact_us) / 1000.0)
        left = [row for row in candidates if row["foot"] == "left"]
        if left:
            first = min(int(row["touchdown_us"]) for row in left)
            left_only.append((first + horizon_ms * 1000 - first_contact_us) / 1000.0)
    return {
        "definition": "terrain_valid_minus_first_target_contact_any_foot",
        "ai_inference_time_included": False,
        "bilateral_shared": _delay_summary(bilateral, total),
        "left_only": _delay_summary(left_only, total),
    }


def _robustness_gate(
    robustness: Mapping[str, object],
    recall_floor: float,
    gap_max: float,
) -> dict[str, object]:
    checks: dict[str, bool] = {}
    for terrain in ("ice", "sand"):
        stable = robustness[terrain]["stable"]["recall"]
        fall = robustness[terrain]["fall"]["recall"]
        checks[f"{terrain}_subsets_present"] = stable is not None and fall is not None
        checks[f"{terrain}_recall_floor"] = bool(
            stable is not None and fall is not None and min(stable, fall) >= recall_floor
        )
        checks[f"{terrain}_gap"] = bool(
            stable is not None and fall is not None and abs(stable - fall) <= gap_max
        )
    for source in ("concrete", "marble"):
        value = robustness["source_terrain"][source]["accuracy"]
        checks[f"{source}_source_accuracy_floor"] = bool(
            value is not None and value >= recall_floor
        )
    return {"checks": checks, "passed": all(checks.values())}


def run_terrain_sensor_ablation(
    config_path: Path,
    repository_root: Path,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Select entirely on validation, then open holdout exactly once."""
    repository_root = repository_root.resolve()
    collection = load_terrain_collection_config(config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    dataset_path = collection.output_path
    artifact_path = collection.artifact_path
    if artifact_path.exists() and any(artifact_path.iterdir()):
        raise FileExistsError(f"refusing to overwrite Terrain artifacts: {artifact_path}")
    dataset_summary = validate_terrain_dataset(dataset_path)
    if not dataset_summary["coverage"]["passed"]:
        raise RuntimeError("TERRAIN_DATASET_NEEDS_REVISION: ML is prohibited")
    artifact_path.mkdir(parents=True, exist_ok=True)
    all_rows = read_event_index(dataset_path / "events.csv")
    cap = collection.max_events_per_class_per_run
    event_sets = {
        name: select_capped_events(all_rows, name, cap)
        for name in ("train", "validation", "holdout")
    }
    holdout_integrity = {
        "run_count": sum(run.split == "holdout" for run in collection.runs),
        "eligible_event_count": len(event_sets["holdout"]),
        "class_counts": {
            class_name: sum(
                row["terrain_gt"] == class_name for row in event_sets["holdout"]
            )
            for class_name in TERRAIN_CLASS_NAMES
        },
        "waveforms_opened_before_selection": False,
        "performance_evaluated_before_selection": False,
    }
    settings = config["training"]
    seeds = tuple(int(value) for value in settings["seeds"])
    macro_min = float(config["selection"]["validation_macro_f1_min"])
    recall_min = float(config["selection"]["validation_worst_class_recall_min"])

    primary_candidates: dict[str, dict[str, object]] = {}
    for profile in settings["sensor_profiles"]:
        result = _train_candidate(
            dataset_path,
            event_sets["train"],
            event_sets["validation"],
            str(profile),
            "mlp",
            50,
            seeds,
            settings,
            progress,
        )
        primary_candidates[str(profile)] = result
    sensor_summary, sensor_reason = select_minimum_sensor(
        [result["summary"] for result in primary_candidates.values()],
        macro_min,
        recall_min,
        float(config["selection"]["sensor_near_best_tolerance"]),
    )
    selected_profile = str(sensor_summary["profile"])

    family_candidates = {"mlp": primary_candidates[selected_profile]}
    family_candidates["gru"] = _train_candidate(
        dataset_path,
        event_sets["train"],
        event_sets["validation"],
        selected_profile,
        "gru",
        50,
        seeds,
        settings,
        progress,
    )
    family_summary = select_model_family(
        [result["summary"] for result in family_candidates.values()]
    )
    selected_family = str(family_summary["family"])

    horizon_candidates: dict[int, dict[str, object]] = {
        50: family_candidates[selected_family]
    }
    for horizon in (20, 30):
        horizon_train_rows = select_capped_events(
            all_rows, "train", cap, required_horizon_ms=horizon
        )
        horizon_validation_rows = select_capped_events(
            all_rows, "validation", cap, required_horizon_ms=horizon
        )
        horizon_candidates[horizon] = _train_candidate(
            dataset_path,
            horizon_train_rows,
            horizon_validation_rows,
            selected_profile,
            selected_family,
            horizon,
            seeds,
            settings,
            progress,
        )
    horizon_summary, horizon_passed = select_shortest_horizon(
        [result["summary"] for result in horizon_candidates.values()],
        macro_min,
        recall_min,
    )
    selected_horizon = int(horizon_summary["horizon_ms"])
    bilateral_candidate = horizon_candidates[selected_horizon]

    left_train_rows = select_capped_events(
        all_rows,
        "train",
        cap,
        foot="left",
        required_horizon_ms=selected_horizon,
    )
    left_validation_rows = select_capped_events(
        all_rows,
        "validation",
        cap,
        foot="left",
        required_horizon_ms=selected_horizon,
    )
    left_candidate = _train_candidate(
        dataset_path,
        left_train_rows,
        left_validation_rows,
        selected_profile,
        selected_family,
        selected_horizon,
        seeds,
        settings,
        progress,
    )
    left_supported = _qualified(left_candidate["summary"], macro_min, recall_min)
    bilateral_supported = _qualified(
        bilateral_candidate["summary"], macro_min, recall_min
    )
    if left_supported:
        selected_scheme = "left_only"
        selected_candidate = left_candidate
        selected_train_rows = left_train_rows
        selected_validation_rows = left_validation_rows
    else:
        selected_scheme = "bilateral_shared"
        selected_candidate = bilateral_candidate
        selected_train_rows = event_sets["train"]
        selected_validation_rows = event_sets["validation"]

    selection = {
        "sensor_profile": selected_profile,
        "sensor_reason": sensor_reason,
        "model_family": selected_family,
        "observation_horizon_ms": selected_horizon,
        "horizon_gate_passed": horizon_passed,
        "deployment_scheme": selected_scheme,
        "left_only_validation_supported": left_supported,
        "bilateral_validation_supported": bilateral_supported,
        "selected_before_holdout": True,
        "holdout_reselection_permitted": False,
        "selected_train_event_count": len(selected_train_rows),
        "selected_validation_event_count": len(selected_validation_rows),
    }
    _write_json(artifact_path / "selection_before_holdout.json", selection)

    guard = HoldoutGuard()
    guard.open_once()
    selected_holdout_rows = select_capped_events(
        all_rows,
        "holdout",
        cap,
        foot="left" if selected_scheme == "left_only" else None,
        required_horizon_ms=selected_horizon,
    )
    raw_holdout = build_terrain_windows(
        dataset_path,
        selected_holdout_rows,
        selected_profile,
        selected_horizon,
        holdout_guard=guard,
    )
    holdout_windows = _normalized_windows(
        raw_holdout, selected_candidate["normalizer"]
    )
    holdout_metrics, predictions = _ensemble_metrics(
        selected_candidate["models"], holdout_windows
    )
    robustness = _holdout_robustness(
        holdout_windows, predictions, selected_holdout_rows
    )
    robustness_gate = _robustness_gate(
        robustness,
        float(config["acceptance"]["robustness_catastrophic_recall_floor"]),
        float(config["acceptance"]["robustness_catastrophic_gap_max"]),
    )
    worst_holdout = min(
        float(values["recall"])
        for values in holdout_metrics["per_class"].values()
    )
    holdout_passed = bool(
        float(holdout_metrics["macro_f1"])
        >= float(config["acceptance"]["holdout_macro_f1_min"])
        and worst_holdout
        >= float(config["acceptance"]["holdout_worst_class_recall_min"])
        and robustness_gate["passed"]
    )
    clear_signal = bool(
        float(holdout_metrics["macro_f1"])
        >= float(config["acceptance"]["promising_macro_f1_min"])
        and worst_holdout
        >= float(config["acceptance"]["promising_worst_class_recall_min"])
    )
    if holdout_passed:
        verdict = config["acceptance"]["verdicts"]["supported"]
    elif clear_signal:
        verdict = config["acceptance"]["verdicts"]["promising"]
    else:
        verdict = config["acceptance"]["verdicts"]["revision"]

    delay = terrain_update_delays(dataset_path, all_rows, selected_horizon)
    hardware_channels = config["hardware_channels_including_pelvis_imu6"]
    profile_upper = {
        "fsr4": "FSR4",
        "foot_imu6": "FOOT_IMU6",
        "fusion10": "FUSION10",
    }[selected_profile]
    scheme_label = "LEFT" if selected_scheme == "left_only" else "BILATERAL"
    recommendation = (
        f"{scheme_label}_{profile_upper}_RECOMMENDED"
        if verdict == "TERRAIN_RECOGNITION_SUPPORTED"
        else "TERRAIN_SENSOR_ARCHITECTURE_UNRESOLVED"
    )
    deployment = {
        "left_only": {
            "validation": left_candidate["summary"],
            "update_delay": delay["left_only"],
            "physical_channels_including_pelvis_imu6": int(
                hardware_channels["left_only"][selected_profile]
            ),
        },
        "bilateral_shared": {
            "validation": bilateral_candidate["summary"],
            "update_delay": delay["bilateral_shared"],
            "physical_channels_including_pelvis_imu6": int(
                hardware_channels["bilateral_shared"][selected_profile]
            ),
        },
    }

    checkpoints = artifact_path / "selected_models"
    for model, result, seed in zip(
        selected_candidate["models"],
        selected_candidate["training_results"],
        seeds,
    ):
        save_checkpoint(
            checkpoints / f"seed_{seed}.pt",
            model,
            selected_family,
            selected_horizon,
            int(seed),
            result,
            input_channels=SENSOR_PROFILE_CHANNELS[selected_profile],
            class_names=TERRAIN_CLASS_NAMES,
        )
    _write_json(
        checkpoints / "normalization.json",
        selected_candidate["normalizer"].to_dict(),
    )
    metrics: dict[str, object] = {
        "experiment_id": config["experiment"]["id"],
        "dataset": dataset_summary,
        "holdout_integrity_before_selection": holdout_integrity,
        "event_cap_per_run_class": cap,
        "sensor_ablation": {
            profile: result["summary"]
            for profile, result in primary_candidates.items()
        },
        "sensor_selection": {
            "selected_profile": selected_profile,
            **sensor_reason,
        },
        "model_family_sanity": {
            family: result["summary"]
            for family, result in family_candidates.items()
        },
        "horizon_study": {
            str(horizon): result["summary"]
            for horizon, result in sorted(horizon_candidates.items())
        },
        "selection": selection,
        "holdout": {
            "guard_open_count": guard.open_count,
            "event_count": len(holdout_windows),
            "metrics": holdout_metrics,
            "worst_class_recall": worst_holdout,
            "robustness": robustness,
            "robustness_gate": robustness_gate,
        },
        "deployment": deployment,
        "update_latency": delay,
        "recommendation": recommendation,
        "final_sensor_architecture_frozen": False,
        "verdict": verdict,
    }
    _write_json(artifact_path / "metrics.json", metrics)
    return artifact_path, metrics
