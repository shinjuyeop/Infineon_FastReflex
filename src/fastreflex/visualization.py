"""Read-only visualization of frozen Hazard and Terrain decisions."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from fastreflex.dataset.hazard import (
    GROUPS,
    HazardRun,
    generate_hazard_specifications,
    i1_support_precursor_sample,
    i1_support_precursor_trace,
    load_hazard_manifest,
    load_hazard_runs,
    load_yaml,
    physical_signature,
)
from fastreflex.dataset.loader import sha256_file
from fastreflex.evaluation.hazard import (
    THRESHOLD,
    HazardReplay,
    load_hazard_normalizer,
    reflex_onset_samples,
    reflex_required_trace,
    replay_hazard_run,
    verify_supported_candidate,
)
from fastreflex.evaluation.terrain import (
    TERRAIN_STATE_NAMES,
    TerrainTrace,
    load_frozen_terrain_candidate,
    refine_hazard_cause,
    replay_terrain,
    terrain_predictions,
    verify_supported_terrain_candidate,
)
from fastreflex.simulation.g1 import (
    TESTED_POLICY_SHA256,
    SimulationConfig,
    SimulationResult,
    load_simulation_config,
    run_simulation,
)
from fastreflex.simulation.terrain import TERRAIN_CLASS_ORDER
from fastreflex.training.trainer import load_checkpoint

DEFAULT_EXPERIMENT_CONFIG = Path(
    "configs/experiment/20260829_unified_hazard_reflex_system.yaml"
)
FREEZE_PATH = Path(
    "artifacts/runs/20260829_unified_hazard_reflex_system/selection_before_holdout.json"
)
TERRAIN_MODEL_PATH = Path(
    "artifacts/runs/20260828_terrain_rebuild_sensor_ablation/selected_models"
)
SUPPORTED_SPLITS = frozenset(("train", "validation"))
SENSOR_ABSOLUTE_TOLERANCE = 0.0


@dataclass(frozen=True)
class ResolvedVisualizationRun:
    """One authorized stored run and its frozen pre-simulation specification."""

    repository_root: Path
    document: Mapping[str, object]
    manifest: Mapping[str, object]
    manifest_row: Mapping[str, object]
    specification: Mapping[str, object]
    dataset_path: Path
    run: HazardRun


@dataclass(frozen=True)
class ParityReport:
    """Fail-closed stored-versus-re-simulated trace comparison."""

    checks: Mapping[str, bool]
    sensor_absolute_tolerance: float

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


@dataclass(frozen=True)
class VisualizationTraces:
    """Frozen model outputs aligned to the authorized runtime trace."""

    hazard: HazardReplay
    hazard_probability: np.ndarray
    hazard_above_threshold: np.ndarray
    reflex_required: np.ndarray
    first_reflex_sample: int | None
    terrain: TerrainTrace
    terrain_probabilities: np.ndarray
    terrain_latest_update: np.ndarray
    terrain_touchdown_foot: np.ndarray
    i1_active: np.ndarray
    i1_sample: int | None


@dataclass(frozen=True)
class PreparedVisualization:
    """Parity-approved simulation plus display-only synchronized traces."""

    resolved: ResolvedVisualizationRun
    simulation_config: SimulationConfig
    simulation: SimulationResult
    parity: ParityReport
    traces: VisualizationTraces


def _experiment_path(root: Path) -> Path:
    return root / DEFAULT_EXPERIMENT_CONFIG


def _dataset_path(root: Path, document: Mapping[str, object]) -> Path:
    value = Path(str(document["dataset"]["path"]))
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("visualization dataset path leaves the repository") from exc
    return path


def _validate_manifest_specification(
    row: Mapping[str, object], specification: Mapping[str, object]
) -> None:
    if (
        str(row["split"]) != str(specification["split"])
        or str(row["group"]) != str(specification["group"])
        or tuple(row["physical_signature"]) != physical_signature(specification)
    ):
        raise RuntimeError(
            f"stored scenario specification changed for {row['run_id']}"
        )


def resolve_visualization_run(
    repository_root: Path, run_id: str
) -> ResolvedVisualizationRun:
    """Resolve one TRAIN/VALIDATION run without opening HOLDOUT waveforms."""
    root = repository_root.resolve()
    document = load_yaml(_experiment_path(root))
    dataset_path = _dataset_path(root, document)
    manifest = load_hazard_manifest(dataset_path)
    if str(manifest.get("dataset_id")) != str(document["dataset"]["dataset_id"]):
        raise RuntimeError("visualization dataset identity changed")

    matches = [row for row in manifest["runs"] if str(row["run_id"]) == run_id]
    if not matches:
        raise ValueError(f"unknown Unified Hazard run ID: {run_id}")
    if len(matches) != 1:
        raise RuntimeError(f"duplicate Unified Hazard run ID: {run_id}")
    row = matches[0]
    split = str(row["split"])
    if split == "holdout":
        raise ValueError(
            "HOLDOUT visualization is prohibited; representative visualization "
            "is restricted to TRAIN/VALIDATION runs"
        )
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported visualization split: {split}")
    invalid_ids = {
        str(value.get("run_id", value)) if isinstance(value, Mapping) else str(value)
        for value in manifest.get("invalid", ())
    }
    if run_id in invalid_ids:
        raise ValueError(f"run is marked invalid and cannot be visualized: {run_id}")

    specifications = {
        str(value["id"]): value
        for value in generate_hazard_specifications(document)
    }
    if run_id not in specifications:
        raise RuntimeError(f"run is absent from the frozen scenario matrix: {run_id}")
    specification = specifications[run_id]
    _validate_manifest_specification(row, specification)
    selected_manifest = {**manifest, "runs": [row]}
    run = load_hazard_runs(dataset_path, selected_manifest, (split,))[run_id]
    return ResolvedVisualizationRun(
        repository_root=root,
        document=document,
        manifest=manifest,
        manifest_row=row,
        specification=specification,
        dataset_path=dataset_path,
        run=run,
    )


def representative_validation_runs(repository_root: Path) -> dict[str, str]:
    """Select the lexicographically first valid VALIDATION run per group."""
    root = repository_root.resolve()
    document = load_yaml(_experiment_path(root))
    manifest = load_hazard_manifest(_dataset_path(root, document))
    specifications = {
        str(value["id"]): value
        for value in generate_hazard_specifications(document)
    }
    invalid_ids = {
        str(value.get("run_id", value)) if isinstance(value, Mapping) else str(value)
        for value in manifest.get("invalid", ())
    }
    selected: dict[str, str] = {}
    for group in GROUPS:
        rows = sorted(
            (
                row
                for row in manifest["runs"]
                if str(row["group"]) == group
                and str(row["split"]) == "validation"
                and str(row["run_id"]) not in invalid_ids
                and (_dataset_path(root, document) / str(row["file"])).is_file()
            ),
            key=lambda row: str(row["run_id"]),
        )
        if not rows:
            raise RuntimeError(f"no valid VALIDATION run is available for {group}")
        row = rows[0]
        run_id = str(row["run_id"])
        if run_id not in specifications:
            raise RuntimeError(
                f"representative run is absent from the matrix: {run_id}"
            )
        _validate_manifest_specification(row, specifications[run_id])
        selected[group] = run_id
    return selected


def visualization_run_ids(
    repository_root: Path,
) -> dict[str, dict[str, list[str]]]:
    """List only authorized TRAIN/VALIDATION run IDs by group and split."""
    root = repository_root.resolve()
    document = load_yaml(_experiment_path(root))
    dataset_path = _dataset_path(root, document)
    manifest = load_hazard_manifest(dataset_path)
    invalid_ids = {
        str(value.get("run_id", value)) if isinstance(value, Mapping) else str(value)
        for value in manifest.get("invalid", ())
    }
    result = {
        group: {split: [] for split in ("train", "validation")} for group in GROUPS
    }
    for row in manifest["runs"]:
        group = str(row["group"])
        split = str(row["split"])
        if (
            group in result
            and split in SUPPORTED_SPLITS
            and str(row["run_id"]) not in invalid_ids
            and (dataset_path / str(row["file"])).is_file()
        ):
            result[group][split].append(str(row["run_id"]))
    for splits in result.values():
        for values in splits.values():
            values.sort()
    return result


def reconstruct_simulation_config(
    resolved: ResolvedVisualizationRun, policy_path: Path | None = None
) -> SimulationConfig:
    """Rebuild the exact frozen scenario from config, never from outcomes."""
    root = resolved.repository_root
    source = resolved.document["source"]
    simulator_path = root / str(source["simulator_config"])
    if sha256_file(simulator_path) != str(source["simulator_config_sha256"]):
        raise RuntimeError("canonical simulator config changed")
    config = load_simulation_config(simulator_path)
    common = resolved.document["common"]
    if (
        config.physics_timestep_s != float(common["physics_timestep_s"])
        or config.sensor_rate_hz != int(common["sensor_rate_hz"])
    ):
        raise RuntimeError("simulator timing differs from the frozen dataset")

    selected_policy = (
        root / str(source["policy_path"])
        if policy_path is None
        else (policy_path if policy_path.is_absolute() else root / policy_path)
    ).resolve()
    if not selected_policy.is_file():
        raise ValueError(f"verified Unitree G1 policy not found: {selected_policy}")
    policy_hash = sha256_file(selected_policy)
    if (
        policy_hash != str(source["policy_sha256"])
        or policy_hash != TESTED_POLICY_SHA256
    ):
        raise RuntimeError("visualization policy differs from the frozen dataset")

    spec = resolved.specification
    return replace(
        config,
        duration_s=float(common["duration_s"]),
        command_speed_mps=float(spec["speed_mps"]),
        policy_path=selected_policy,
        terrain=str(spec["target_terrain"]),
        source_terrain=str(spec["source_terrain"]),
        slip_pattern=str(spec["slip_pattern"]),
        sink_pattern=str(spec["sink_pattern"]),
        sink_severity=str(spec["sink_severity"]),
        sink_support_pattern=str(spec["support_pattern"]),
        patch_start_x_m=float(spec["patch_start_x_m"]),
        patch_width_m=float(spec["patch_width_m"]),
        headless=True,
    )


def _first_true_per_foot(values: np.ndarray) -> tuple[int | None, int | None]:
    return tuple(
        None if not len(indices) else int(indices[0])
        for indices in (np.flatnonzero(values[:, 0]), np.flatnonzero(values[:, 1]))
    )  # type: ignore[return-value]


def _first_target_clocks(
    result: SimulationResult,
    target_terrain: str,
    *,
    hard_stable_control: bool = False,
) -> tuple[int | None, int | None]:
    if result.exact_terrain_contact is None:
        raise RuntimeError("re-simulation omitted exact terrain contact diagnostics")
    if hard_stable_control:
        # The frozen all-hard-ground dataset convention has no transition;
        # target material is present from the beginning of the run.
        return 0, 0
    class_id = TERRAIN_CLASS_ORDER.index(target_terrain)
    contact = result.exact_terrain_contact[:, :, class_id]
    contact_indices = np.flatnonzero(np.any(contact, axis=1))
    target_touchdown = result.diagnostics.touchdown & contact
    touchdown_indices = np.flatnonzero(np.any(target_touchdown, axis=1))
    return (
        None if not len(contact_indices) else int(contact_indices[0]),
        None if not len(touchdown_indices) else int(touchdown_indices[0]),
    )


def compare_stored_runtime(
    resolved: ResolvedVisualizationRun, result: SimulationResult
) -> ParityReport:
    """Compare all required clocks and runtime arrays using exact equality."""
    run = resolved.run
    diagnostics = result.diagnostics
    first_contact, first_touchdown = _first_target_clocks(
        result,
        run.target_terrain,
        hard_stable_control=run.hard_stable_control,
    )
    simulated_censor = result.metadata["first_fall_sample"]
    if simulated_censor is None:
        simulated_censor = len(result.runtime.timestamp_us)
    simulated_run = replace(
        run,
        support_spread_m=diagnostics.support_surface_spread_m.astype(
            np.float32, copy=False
        ),
        loaded_contact=diagnostics.loaded_contact,
    )
    simulated_i1 = i1_support_precursor_sample(simulated_run)
    row = resolved.manifest_row
    checks = {
        "timestamp_us": np.array_equal(
            run.timestamp_us, result.runtime.timestamp_us
        ),
        "pelvis_imu6": np.array_equal(
            run.features["PELVIS_IMU6"], result.runtime.pelvis_imu
        ),
        "foot_fsr8": result.runtime.foot_fsr is not None
        and np.array_equal(
            run.features["PELVIS_IMU6_FSR8"][:, 6:], result.runtime.foot_fsr
        ),
        "first_target_contact_sample": first_contact == run.first_contact_sample,
        "first_target_touchdown_sample": first_touchdown
        == run.first_touchdown_sample,
        "slip_event_samples": _first_true_per_foot(
            diagnostics.established_slip_onset
        )
        == run.slip_event_samples_per_foot,
        "support_event_samples": _first_true_per_foot(
            diagnostics.deformable_sink_onset
        )
        == run.support_event_samples_per_foot,
        "i1_precursor_sample": simulated_i1
        == (
            None
            if row["support_precursor_sample"] is None
            else int(row["support_precursor_sample"])
        ),
        "censor_sample": int(simulated_censor) == run.censor_sample,
        "tangential_drift": np.array_equal(
            run.drift_m,
            diagnostics.tangential_anchor_drift_m.astype(np.float32, copy=False),
            equal_nan=True,
        ),
        "support_spread": np.array_equal(
            run.support_spread_m,
            diagnostics.support_surface_spread_m.astype(np.float32, copy=False),
            equal_nan=True,
        ),
        "loaded_contact": np.array_equal(
            run.loaded_contact, diagnostics.loaded_contact
        ),
    }
    return ParityReport(
        checks=checks,
        sensor_absolute_tolerance=SENSOR_ABSOLUTE_TOLERANCE,
    )


def require_parity(report: ParityReport) -> None:
    if report.passed:
        return
    failed = ", ".join(name for name, passed in report.checks.items() if not passed)
    raise RuntimeError(
        "stored/re-simulated runtime parity failed; viewer will not open: " + failed
    )


def _load_frozen_hazard(
    resolved: ResolvedVisualizationRun,
) -> tuple[object, Sequence[object]]:
    root = resolved.repository_root
    verify_supported_candidate(root, resolved.document)
    freeze = json.loads((root / FREEZE_PATH).read_text(encoding="utf-8"))
    selection = freeze["selection"]
    normalizer = load_hazard_normalizer(root / str(selection["normalizer_path"]))
    models = [
        load_checkpoint(root / str(path))[0]
        for path in selection["checkpoint_sha256"]
    ]
    return normalizer, models


def _compare_stored_terrain(
    resolved: ResolvedVisualizationRun, terrain: TerrainTrace
) -> None:
    path = resolved.dataset_path / str(resolved.manifest_row["file"])
    with np.load(path, allow_pickle=False) as stored:
        checks = {
            "terrain_state": np.array_equal(stored["terrain_state"], terrain.state),
            "terrain_update_samples": np.array_equal(
                stored["terrain_update_samples"], terrain.update_samples
            ),
            "terrain_prediction_ids": np.array_equal(
                stored["terrain_prediction_ids"], terrain.prediction_ids
            ),
            "terrain_prediction_probabilities": np.array_equal(
                stored["terrain_prediction_probabilities"],
                terrain.prediction_probabilities,
            ),
            "terrain_first_target_valid_sample": int(
                stored["terrain_first_target_valid_sample"]
            )
            == (
                -1
                if terrain.first_target_valid_sample is None
                else terrain.first_target_valid_sample
            ),
        }
    failed = ", ".join(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError("frozen Terrain replay differs from stored trace: " + failed)


def _held_terrain_metadata(
    terrain: TerrainTrace, sample_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.zeros((sample_count, 4), dtype=np.float32)
    latest_update = np.full(sample_count, -1, dtype=np.int64)
    foot = np.full(sample_count, "", dtype="<U5")
    for prediction in terrain_predictions(terrain):
        start = prediction.prediction_timestamp
        probabilities[start:] = prediction.probabilities
        latest_update[start:] = start
        foot[start:] = prediction.touchdown_foot
    return probabilities, latest_update, foot


def build_visualization_traces(
    resolved: ResolvedVisualizationRun, result: SimulationResult
) -> VisualizationTraces:
    """Run only the protected inference implementations over parity-approved data."""
    normalizer, hazard_models = _load_frozen_hazard(resolved)
    hazard = replay_hazard_run(resolved.run, normalizer, hazard_models)
    sample_count = len(resolved.run.timestamp_us)
    probability = np.full(sample_count, np.nan, dtype=np.float64)
    probability[hazard.endpoints] = hazard.probabilities
    above = np.zeros(sample_count, dtype=bool)
    above[hazard.endpoints] = hazard.probabilities >= THRESHOLD
    reflex = reflex_required_trace(hazard, sample_count)
    onsets = reflex_onset_samples(hazard)

    verify_supported_terrain_candidate(resolved.repository_root)
    terrain_models, mean, std = load_frozen_terrain_candidate(
        resolved.repository_root / TERRAIN_MODEL_PATH
    )
    terrain = replay_terrain(result, resolved.run, terrain_models, mean, std)
    _compare_stored_terrain(resolved, terrain)
    terrain_probability, terrain_update, terrain_foot = _held_terrain_metadata(
        terrain, sample_count
    )
    diagnostic_run = replace(
        resolved.run,
        support_spread_m=result.diagnostics.support_surface_spread_m.astype(
            np.float32, copy=False
        ),
        loaded_contact=result.diagnostics.loaded_contact,
    )
    i1_active = i1_support_precursor_trace(diagnostic_run)
    i1_sample = i1_support_precursor_sample(diagnostic_run)
    return VisualizationTraces(
        hazard=hazard,
        hazard_probability=probability,
        hazard_above_threshold=above,
        reflex_required=reflex,
        first_reflex_sample=None if not len(onsets) else int(onsets[0]),
        terrain=terrain,
        terrain_probabilities=terrain_probability,
        terrain_latest_update=terrain_update,
        terrain_touchdown_foot=terrain_foot,
        i1_active=i1_active,
        i1_sample=i1_sample,
    )


def prepare_visualization(
    repository_root: Path, run_id: str, policy_path: Path | None = None
) -> PreparedVisualization:
    """Re-simulate, require parity, and compute frozen model traces headlessly."""
    resolved = resolve_visualization_run(repository_root, run_id)
    config = reconstruct_simulation_config(resolved, policy_path)
    result = run_simulation(config)
    parity = compare_stored_runtime(resolved, result)
    require_parity(parity)
    traces = build_visualization_traces(resolved, result)
    return PreparedVisualization(resolved, config, result, parity, traces)


def _time_text(run: HazardRun, sample: int | None) -> str:
    if sample is None:
        return "not observed"
    return f"{run.timestamp_us[sample] / 1_000_000.0:.3f} s (sample {sample})"


def _observed_time_text(
    run: HazardRun, event_sample: int | None, current_sample: int
) -> str:
    if event_sample is None or event_sample > current_sample:
        return "not reached"
    return _time_text(run, event_sample)


def _finite_max(values: np.ndarray) -> float | None:
    finite = np.asarray(values)[np.isfinite(values)]
    return None if not len(finite) else float(np.max(finite))


def _metric(value: float | None, scale: float = 1.0, suffix: str = "") -> str:
    return "N/A" if value is None else f"{value * scale:.1f}{suffix}"


def format_viewer_overlay(
    prepared: PreparedVisualization, sample: int, *, show_debug: bool = False
) -> tuple[str, str]:
    """Format explicitly separated MODEL OUTPUT and SIMULATOR GT panels."""
    run = prepared.resolved.run
    row = prepared.resolved.manifest_row
    traces = prepared.traces
    diagnostics = prepared.simulation.diagnostics
    sample = min(max(int(sample), 0), len(run.timestamp_us) - 1)
    censored = sample >= run.censor_sample
    probability = traces.hazard_probability[sample]
    probability_text = (
        "N/A"
        if censored or not np.isfinite(probability)
        else f"{probability:.6f}"
    )
    above_text = (
        "N/A"
        if censored or not np.isfinite(probability)
        else str(bool(traces.hazard_above_threshold[sample])).upper()
    )
    reflex_text = (
        "CENSORED"
        if censored
        else str(bool(traces.reflex_required[sample])).upper()
    )
    terrain_state = int(traces.terrain.state[sample])
    terrain_name = TERRAIN_STATE_NAMES[terrain_state]
    terrain_update = int(traces.terrain_latest_update[sample])
    terrain_probability = traces.terrain_probabilities[sample]
    foot = str(traces.terrain_touchdown_foot[sample]) or "N/A"
    cause = (
        "CENSORED"
        if censored
        else refine_hazard_cause(bool(traces.reflex_required[sample]), terrain_state)
    )

    model_lines = [
        "MODEL OUTPUT",
        f"Run: {run.run_id} [{run.split.upper()}]",
        f"Simulation time: {run.timestamp_us[sample] / 1_000_000.0:.3f} s",
        "",
        f"Hazard probability: {probability_text}",
        f"p >= 0.99: {above_text}",
        f"REFLEX_REQUIRED: {reflex_text}",
        "First reflex: "
        + _observed_time_text(run, traces.first_reflex_sample, sample),
        "",
        f"Terrain state: {terrain_name} (advisory only)",
        "Latest update: "
        + _time_text(run, None if terrain_update < 0 else terrain_update),
        f"Touchdown foot: {foot}",
        f"Concrete: {terrain_probability[0]:.4f}",
        f"Marble:  {terrain_probability[1]:.4f}",
        f"Ice:     {terrain_probability[2]:.4f}",
        f"Sand:    {terrain_probability[3]:.4f}",
        f"Cause refinement: {cause}",
    ]
    if show_debug:
        model_lines.extend(
            (
                "",
                "Hazard tensor: Pelvis IMU6 -> causal 80D",
                "Terrain never gates Hazard",
            )
        )

    slip_active = bool(np.any(diagnostics.established_slip[sample]))
    support_active = bool(np.any(diagnostics.deformable_sink_active[sample]))
    i1_active = bool(traces.i1_active[sample])
    slip_sample = min(
        (value for value in run.slip_event_samples_per_foot if value is not None),
        default=None,
    )
    support_sample = min(
        (value for value in run.support_event_samples_per_foot if value is not None),
        default=None,
    )
    drift = _finite_max(diagnostics.tangential_anchor_drift_m[sample])
    spread = _finite_max(diagnostics.support_surface_spread_m[sample])
    diagnostic_lines = [
        "SIMULATOR GT / DIAGNOSTIC",
        "NEVER USED AS MODEL INPUT",
        f"Physical label: {row['physical_label']}",
        "",
        f"Slip active: {str(slip_active).upper()}",
        f"Slip event: {_observed_time_text(run, slip_sample, sample)}",
        f"Tangential drift: {_metric(drift, 1000.0, ' mm')}",
        "Established Slip: 50 mm / 3 ms",
        "",
        f"I1 precursor active: {str(i1_active).upper()}",
        f"I1 event: {_observed_time_text(run, traces.i1_sample, sample)}",
        f"Support active: {str(support_active).upper()}",
        f"Support event: {_observed_time_text(run, support_sample, sample)}",
        f"Support spread: {_metric(spread, 1000.0, ' mm')}",
        "Established Support: 10 mm / 20 ms",
        "Censor: "
        + _observed_time_text(
            run,
            run.censor_sample if run.censor_sample < len(run.timestamp_us) else None,
            sample,
        ),
    ]
    if show_debug:
        diagnostic_lines.extend(
            (
                "",
                f"Group: {row['group']}",
                "Parity: exact timestamp/IMU6/FSR8/clocks",
            )
        )
    return "\n".join(model_lines), "\n".join(diagnostic_lines)


def visualize_prepared_run(
    prepared: PreparedVisualization,
    *,
    playback_speed: float = 1.0,
    show_debug: bool = False,
) -> dict[str, object]:
    """Open the viewer only after parity, then verify viewer/physics parity."""
    require_parity(prepared.parity)
    config = replace(prepared.simulation_config, headless=False)
    viewer_result = run_simulation(
        config,
        viewer_overlay=lambda sample: format_viewer_overlay(
            prepared, sample, show_debug=show_debug
        ),
        playback_speed=playback_speed,
    )
    if bool(viewer_result.metadata["terminated_by_viewer"]):
        raise RuntimeError(
            "viewer closed before replay completed; final viewer/physics parity "
            "could not be confirmed"
        )
    viewer_parity = compare_stored_runtime(prepared.resolved, viewer_result)
    require_parity(viewer_parity)
    return {
        "run_id": prepared.resolved.run.run_id,
        "split": prepared.resolved.run.split,
        "stored_resimulation_parity": prepared.parity.passed,
        "viewer_physics_parity": viewer_parity.passed,
        "sensor_absolute_tolerance": SENSOR_ABSOLUTE_TOLERANCE,
        "holdout_opened": False,
    }
