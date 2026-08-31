"""Read-only visualization of frozen Hazard and Terrain decisions."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any

import mujoco
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
    launch_passive_viewer,
    load_g1_model,
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
VISUALIZATION_MODES = ("demo", "analysis")


class PlaybackState(str, Enum):
    """User-visible snapshot playback states."""

    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    ENDED_PAUSED = "ENDED_PAUSED"


@dataclass(frozen=True)
class PlaybackEvents:
    """Read-only destinations on the 1 kHz visualization timeline."""

    first_reflex: int | None = None
    physical_hazard: int | None = None
    i1_precursor: int | None = None
    terrain_updates: tuple[int, ...] = ()


@dataclass(frozen=True)
class PlaybackView:
    """Atomic display snapshot of interactive playback state."""

    current_sample: int
    state: PlaybackState
    speed: float
    notice: str
    revision: int
    ended_reached: bool

    @property
    def status(self) -> str:
        if self.state is PlaybackState.ENDED_PAUSED:
            return "ENDED / PAUSED"
        return self.state.value


class SnapshotPlaybackControl:
    """Thread-safe keyboard state machine over immutable 1 kHz snapshots."""

    SPACE_KEY = 32
    PERIOD_KEY = 46
    A_KEY = 65
    D_KEY = 68
    G_KEY = 71
    H_KEY = 72
    I_KEY = 73
    R_KEY = 82
    T_KEY = 84
    RIGHT_ARROW_KEY = 262
    LEFT_ARROW_KEY = 263
    HOME_KEY = 268
    END_KEY = 269

    def __init__(
        self,
        total_samples: int,
        *,
        speed: float = 1.0,
        events: PlaybackEvents | None = None,
        auto_pause_sample: int | None = None,
        start_paused: bool = False,
    ) -> None:
        if total_samples <= 0:
            raise ValueError("snapshot playback requires at least one sample")
        if not np.isfinite(speed) or speed <= 0.0:
            raise ValueError("snapshot playback speed must be positive and finite")
        if auto_pause_sample is not None and not (
            0 <= auto_pause_sample < total_samples
        ):
            raise ValueError("automatic pause sample is outside the render trace")
        self._total_samples = int(total_samples)
        self._speed = float(speed)
        self._events = events or PlaybackEvents()
        self._current_sample = 0
        self._state = (
            PlaybackState.PAUSED
            if start_paused or auto_pause_sample == 0
            else PlaybackState.PLAYING
        )
        self._auto_pause_sample = auto_pause_sample
        self._auto_pause_consumed = auto_pause_sample == 0
        self._notice = (
            "Automatic pause at sample 0"
            if auto_pause_sample == 0
            else ("Single-step start" if start_paused else "")
        )
        self._revision = 0
        self._ended_reached = False
        self._lock = Lock()

    @property
    def total_samples(self) -> int:
        return self._total_samples

    def view(self) -> PlaybackView:
        with self._lock:
            return PlaybackView(
                current_sample=self._current_sample,
                state=self._state,
                speed=self._speed,
                notice=self._notice,
                revision=self._revision,
                ended_reached=self._ended_reached,
            )

    def _changed(self, notice: str = "") -> None:
        self._notice = notice
        self._revision += 1

    def _seek(self, sample: int, notice: str) -> None:
        self._current_sample = min(max(int(sample), 0), self._total_samples - 1)
        self._state = PlaybackState.PAUSED
        self._changed(notice)

    def _event_seek(self, sample: int | None, name: str) -> None:
        if sample is None:
            self._state = PlaybackState.PAUSED
            self._changed(f"{name} is not present in this run")
            return
        self._seek(sample, f"Jumped to {name}")

    def _seek_end(self) -> None:
        self._current_sample = self._total_samples - 1
        self._state = PlaybackState.ENDED_PAUSED
        self._ended_reached = True
        self._changed("Jumped to last sample; viewer remains open")

    def _next_terrain_update(self) -> None:
        target = next(
            (
                sample
                for sample in self._events.terrain_updates
                if sample > self._current_sample
            ),
            None,
        )
        self._event_seek(target, "next Terrain update")

    def _previous_terrain_update(self) -> None:
        target = next(
            (
                sample
                for sample in reversed(self._events.terrain_updates)
                if sample < self._current_sample
            ),
            None,
        )
        self._event_seek(target, "previous Terrain update")

    def key_callback(self, keycode: int) -> None:
        """Apply canonical MuJoCo/GLFW key codes without touching physics."""
        with self._lock:
            if keycode == self.SPACE_KEY:
                if self._state is PlaybackState.PLAYING:
                    self._state = PlaybackState.PAUSED
                    self._changed("Playback paused")
                elif self._state is PlaybackState.ENDED_PAUSED:
                    self._current_sample = 0
                    self._state = PlaybackState.PLAYING
                    self._changed("Restarted from first sample")
                else:
                    self._state = PlaybackState.PLAYING
                    self._changed("Playback resumed")
            elif keycode == self.LEFT_ARROW_KEY:
                self._seek(self._current_sample - 1, "Stepped -1 ms")
            elif keycode in (self.RIGHT_ARROW_KEY, self.PERIOD_KEY):
                self._seek(self._current_sample + 1, "Stepped +1 ms")
            elif keycode == self.A_KEY:
                self._seek(self._current_sample - 10, "Stepped -10 ms")
            elif keycode == self.D_KEY:
                self._seek(self._current_sample + 10, "Stepped +10 ms")
            elif keycode == self.HOME_KEY:
                self._seek(0, "Jumped to first sample")
            elif keycode == self.END_KEY:
                self._seek_end()
            elif keycode == self.R_KEY:
                self._event_seek(self._events.first_reflex, "first Reflex")
            elif keycode == self.H_KEY:
                self._event_seek(self._events.physical_hazard, "physical Hazard")
            elif keycode == self.I_KEY:
                self._event_seek(self._events.i1_precursor, "I1 precursor")
            elif keycode == self.T_KEY:
                self._next_terrain_update()
            elif keycode == self.G_KEY:
                self._previous_terrain_update()

    def advance_by(self, sample_count: int) -> PlaybackView:
        """Advance wall-clock playback, stopping exactly at pause/end boundaries."""
        if sample_count <= 0:
            return self.view()
        with self._lock:
            if self._state is not PlaybackState.PLAYING:
                return PlaybackView(
                    self._current_sample,
                    self._state,
                    self._speed,
                    self._notice,
                    self._revision,
                    self._ended_reached,
                )
            target = min(
                self._current_sample + int(sample_count),
                self._total_samples - 1,
            )
            auto_pause = self._auto_pause_sample
            if (
                not self._auto_pause_consumed
                and auto_pause is not None
                and self._current_sample < auto_pause <= target
            ):
                self._current_sample = auto_pause
                self._state = PlaybackState.PAUSED
                self._auto_pause_consumed = True
                self._changed("Automatic pause condition reached")
            elif target == self._total_samples - 1:
                self._current_sample = target
                self._state = PlaybackState.ENDED_PAUSED
                self._ended_reached = True
                self._changed("End reached; viewer remains open")
            elif target != self._current_sample:
                self._current_sample = target
                self._revision += 1
            return PlaybackView(
                self._current_sample,
                self._state,
                self._speed,
                self._notice,
                self._revision,
                self._ended_reached,
            )


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
        raise RuntimeError(f"stored scenario specification changed for {row['run_id']}")


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
        str(value["id"]): value for value in generate_hazard_specifications(document)
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
        str(value["id"]): value for value in generate_hazard_specifications(document)
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
    if config.physics_timestep_s != float(
        common["physics_timestep_s"]
    ) or config.sensor_rate_hz != int(common["sensor_rate_hz"]):
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
        "timestamp_us": np.array_equal(run.timestamp_us, result.runtime.timestamp_us),
        "pelvis_imu6": np.array_equal(
            run.features["PELVIS_IMU6"], result.runtime.pelvis_imu
        ),
        "foot_fsr8": result.runtime.foot_fsr is not None
        and np.array_equal(
            run.features["PELVIS_IMU6_FSR8"][:, 6:], result.runtime.foot_fsr
        ),
        "first_target_contact_sample": first_contact == run.first_contact_sample,
        "first_target_touchdown_sample": first_touchdown == run.first_touchdown_sample,
        "slip_event_samples": _first_true_per_foot(diagnostics.established_slip_onset)
        == run.slip_event_samples_per_foot,
        "support_event_samples": _first_true_per_foot(diagnostics.deformable_sink_onset)
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
        load_checkpoint(root / str(path))[0] for path in selection["checkpoint_sha256"]
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
    result = run_simulation(config, capture_render_trace=True)
    parity = compare_stored_runtime(resolved, result)
    require_parity(parity)
    if result.render_trace is None:
        raise RuntimeError("visualization render trace capture was not produced")
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


def _first_event_sample(values: Sequence[int | None]) -> int | None:
    return min((value for value in values if value is not None), default=None)


def _physical_reference(
    prepared: PreparedVisualization,
) -> tuple[str, str | None, int | None]:
    group = str(prepared.resolved.manifest_row["group"])
    run = prepared.resolved.run
    if group == "ICE_SLIP_HAZARD":
        return (
            "SLIP",
            "Established Slip",
            _first_event_sample(run.slip_event_samples_per_foot),
        )
    if group == "SAND_SUPPORT_HAZARD":
        return (
            "SUPPORT",
            "Established Support",
            _first_event_sample(run.support_event_samples_per_foot),
        )
    return "NO HAZARD", None, None


def playback_events(prepared: PreparedVisualization) -> PlaybackEvents:
    """Build display-only event destinations from already-approved traces."""
    _, _, physical_sample = _physical_reference(prepared)
    updates = tuple(
        sorted({int(value) for value in prepared.traces.terrain.update_samples})
    )
    return PlaybackEvents(
        first_reflex=prepared.traces.first_reflex_sample,
        physical_hazard=physical_sample,
        i1_precursor=prepared.traces.i1_sample,
        terrain_updates=updates,
    )


def _delta_text(
    run: HazardRun, detector_sample: int | None, reference_sample: int | None
) -> str:
    if detector_sample is None or reference_sample is None:
        return "N/A"
    delta_ms = (
        int(run.timestamp_us[detector_sample]) - int(run.timestamp_us[reference_sample])
    ) / 1000.0
    return f"{delta_ms:+.0f} ms"


def _event_timing_lines(prepared: PreparedVisualization) -> list[str]:
    run = prepared.resolved.run
    reflex = prepared.traces.first_reflex_sample
    _, physical_name, physical_sample = _physical_reference(prepared)
    terrain = prepared.traces.terrain.first_target_valid_sample
    lines = ["EVENT TIMING", "delta = detector - reference"]
    lines.append(f"First Reflex: {_time_text(run, reflex)}")
    if prepared.traces.i1_sample is not None or physical_name == "Established Support":
        lines.extend(
            (
                f"I1 Precursor: {_time_text(run, prepared.traces.i1_sample)}",
                "Reflex -> I1: " + _delta_text(run, reflex, prepared.traces.i1_sample),
            )
        )
    if physical_name is not None:
        lines.extend(
            (
                f"{physical_name}: {_time_text(run, physical_sample)}",
                f"Reflex -> {physical_name.removeprefix('Established ')}: "
                + _delta_text(run, reflex, physical_sample),
            )
        )
    lines.extend(
        (
            f"Terrain target: {_time_text(run, terrain)}",
            "Reflex -> Terrain: " + _delta_text(run, reflex, terrain),
        )
    )
    return lines


def _timeline_lines(
    prepared: PreparedVisualization, current_sample: int, width: int = 42
) -> list[str]:
    run = prepared.resolved.run
    sample_count = len(run.timestamp_us)
    markers: list[tuple[str, int | None]] = [
        ("R", prepared.traces.first_reflex_sample),
    ]
    _, physical_name, physical_sample = _physical_reference(prepared)
    if physical_name == "Established Slip":
        markers.append(("S", physical_sample))
    elif physical_name == "Established Support":
        markers.append(("U", physical_sample))
    markers.extend(
        (
            ("I", prepared.traces.i1_sample),
            ("T", prepared.traces.terrain.first_target_valid_sample),
            (
                "C",
                run.censor_sample if run.censor_sample < sample_count else None,
            ),
        )
    )

    def position(sample: int) -> int:
        if sample_count <= 1:
            return 0
        return int(round(sample * (width - 1) / (sample_count - 1)))

    timeline = ["-"] * width
    legend: list[str] = []
    used: dict[int, list[str]] = {}
    for marker, sample in markers:
        if sample is None:
            continue
        pos = position(sample)
        used.setdefault(pos, []).append(marker)
        timeline[pos] = marker if len(used[pos]) == 1 else "*"
        legend.append(marker)
    cursor = [" "] * width
    cursor[position(min(max(current_sample, 0), sample_count - 1))] = "^"
    end_s = run.timestamp_us[-1] / 1_000_000.0
    legend_text = " ".join(
        f"{value}={name}"
        for value, name in (
            ("R", "Reflex"),
            ("S", "Slip"),
            ("I", "I1"),
            ("U", "Support"),
            ("T", "Terrain"),
            ("C", "Censor"),
        )
        if value in legend
    )
    return [
        f"0.0s |{''.join(timeline)}| {end_s:.1f}s",
        f"      {''.join(cursor)}  NOW",
        legend_text
        + (
            " (* = co-located)"
            if any(len(values) > 1 for values in used.values())
            else ""
        ),
    ]


def format_viewer_overlay(
    prepared: PreparedVisualization,
    sample: int,
    *,
    show_debug: bool = False,
    mode: str = "analysis",
    playback: PlaybackView | None = None,
) -> tuple[str, str]:
    """Format explicitly separated MODEL OUTPUT and SIMULATOR GT panels."""
    if mode not in VISUALIZATION_MODES:
        raise ValueError(f"visualization mode must be one of {VISUALIZATION_MODES}")
    run = prepared.resolved.run
    row = prepared.resolved.manifest_row
    traces = prepared.traces
    diagnostics = prepared.simulation.diagnostics
    current_sample = min(int(sample), len(run.timestamp_us) - 1)
    initial = current_sample < 0
    sample = max(current_sample, 0)
    censored = not initial and sample >= run.censor_sample
    probability = traces.hazard_probability[sample]
    probability_text = (
        "N/A"
        if initial or censored or not np.isfinite(probability)
        else f"{probability:.6f}"
    )
    above_text = (
        "N/A"
        if initial or censored or not np.isfinite(probability)
        else str(bool(traces.hazard_above_threshold[sample])).upper()
    )
    reflex_text = (
        "CENSORED"
        if censored
        else str(bool(not initial and traces.reflex_required[sample])).upper()
    )
    reflex_occurred = bool(
        not initial
        and traces.first_reflex_sample is not None
        and traces.first_reflex_sample <= sample
    )
    terrain_state = 0 if initial else int(traces.terrain.state[sample])
    terrain_name = TERRAIN_STATE_NAMES[terrain_state]
    terrain_update = -1 if initial else int(traces.terrain_latest_update[sample])
    terrain_probability = (
        np.zeros(4, dtype=np.float32)
        if initial
        else traces.terrain_probabilities[sample]
    )
    foot = "N/A" if initial else (str(traces.terrain_touchdown_foot[sample]) or "N/A")
    current_cause = (
        "N/A (CENSORED)"
        if censored
        else refine_hazard_cause(
            bool(not initial and traces.reflex_required[sample]), terrain_state
        )
    )
    first_reflex_cause = "not reached"
    if reflex_occurred and traces.first_reflex_sample is not None:
        first_reflex_cause = refine_hazard_cause(
            True, int(traces.terrain.state[traces.first_reflex_sample])
        )
    latest_context = "N/A"
    if (
        reflex_occurred
        and traces.first_reflex_sample is not None
        and terrain_update > traces.first_reflex_sample
    ):
        latest_context = f"{terrain_name} (display only)"
    playback_status = "PLAYING" if playback is None else playback.status
    playback_notice = "" if playback is None else playback.notice
    simulation_time = (
        "0.000 s" if initial else f"{run.timestamp_us[sample] / 1_000_000.0:.3f} s"
    )
    hazard_status = (
        "CENSORED"
        if censored
        else ("REFLEX DETECTED" if reflex_text == "TRUE" else "NORMAL")
    )
    physical_reference, _, _ = _physical_reference(prepared)

    if mode == "demo":
        model_lines = [
            "MODEL OUTPUT / DEMO",
            f"Run: {run.run_id} [{run.split.upper()}]",
            f"Time: {simulation_time}",
            f"PLAYBACK: {playback_status} ({playback.speed:.1f}x)"
            if playback is not None
            else f"PLAYBACK: {playback_status}",
            *([f"Message: {playback_notice}"] if playback_notice else []),
            "",
            f"Hazard: {hazard_status}",
            f"Current reflex: {reflex_text}",
            "Reflex occurred: " + str(reflex_occurred).upper() + " (display history)",
            "First reflex: "
            + _observed_time_text(run, traces.first_reflex_sample, current_sample),
            "",
            f"Terrain: {terrain_name} (advisory only)",
            f"Current advisory cause: {current_cause}",
            f"Cause at first reflex: {first_reflex_cause}",
        ]
        diagnostic_lines = [
            "SIMULATOR GT / DIAGNOSTIC",
            "NEVER USED AS MODEL INPUT",
            f"Physical reference: {physical_reference}",
            "",
            *_event_timing_lines(prepared),
            "",
            *_timeline_lines(prepared, current_sample),
            "",
            "Space play/pause | Left/Right +/-1 ms | A/D +/-10 ms",
            "R Reflex | H Hazard | I I1 | T/G Terrain | Home/End",
        ]
        return "\n".join(model_lines), "\n".join(diagnostic_lines)

    model_lines = [
        "MODEL OUTPUT",
        f"Run: {run.run_id} [{run.split.upper()}]",
        f"Simulation time: {simulation_time}",
        f"PLAYBACK: {playback_status}"
        + ("" if playback is None else f" ({playback.speed:.1f}x)"),
        *([f"Message: {playback_notice}"] if playback_notice else []),
        "Space play/pause | Left/Right +/-1 ms | A/D +/-10 ms",
        "R Reflex | H Hazard | I I1 | T/G Terrain | Home/End",
        "",
        f"Hazard probability: {probability_text}",
        f"p >= 0.99: {above_text}",
        f"Current reflex (REFLEX_REQUIRED): {reflex_text}",
        "Reflex occurred: " + str(reflex_occurred).upper() + " (display history only)",
        "First reflex: "
        + _observed_time_text(run, traces.first_reflex_sample, current_sample),
        "",
        f"Terrain state: {terrain_name} (advisory only)",
        "Latest update: "
        + _time_text(run, None if terrain_update < 0 else terrain_update),
        f"Touchdown foot: {foot}",
        f"Concrete: {terrain_probability[0]:.4f}",
        f"Marble:  {terrain_probability[1]:.4f}",
        f"Ice:     {terrain_probability[2]:.4f}",
        f"Sand:    {terrain_probability[3]:.4f}",
        f"Current advisory cause: {current_cause}",
        f"Cause at first reflex: {first_reflex_cause}",
        f"Latest terrain context after event: {latest_context}",
    ]
    if show_debug:
        model_lines.extend(
            (
                "",
                "Hazard tensor: Pelvis IMU6 -> causal 80D",
                "Terrain never gates Hazard",
            )
        )

    slip_active = bool(not initial and np.any(diagnostics.established_slip[sample]))
    support_active = bool(
        not initial and np.any(diagnostics.deformable_sink_active[sample])
    )
    i1_active = bool(not initial and traces.i1_active[sample])
    slip_sample = _first_event_sample(run.slip_event_samples_per_foot)
    support_sample = _first_event_sample(run.support_event_samples_per_foot)
    drift = (
        None if initial else _finite_max(diagnostics.tangential_anchor_drift_m[sample])
    )
    spread = (
        None if initial else _finite_max(diagnostics.support_surface_spread_m[sample])
    )
    diagnostic_lines = [
        "SIMULATOR GT / DIAGNOSTIC",
        "NEVER USED AS MODEL INPUT",
        f"Physical reference: {physical_reference}",
        f"Physical label: {row['physical_label']}",
        "",
        f"Slip active: {str(slip_active).upper()}",
        f"Slip event: {_observed_time_text(run, slip_sample, current_sample)}",
        f"Tangential drift: {_metric(drift, 1000.0, ' mm')}",
        "Established Slip: 50 mm / 3 ms",
        "",
        f"I1 precursor active: {str(i1_active).upper()}",
        f"I1 event: {_observed_time_text(run, traces.i1_sample, current_sample)}",
        f"Support active: {str(support_active).upper()}",
        f"Support event: {_observed_time_text(run, support_sample, current_sample)}",
        f"Support spread: {_metric(spread, 1000.0, ' mm')}",
        "Established Support: 10 mm / 20 ms",
        "Censor: "
        + _observed_time_text(
            run,
            run.censor_sample if run.censor_sample < len(run.timestamp_us) else None,
            current_sample,
        ),
        "",
        *_event_timing_lines(prepared),
        "",
        *_timeline_lines(prepared, current_sample),
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


def _nearest_sample(run: HazardRun, seconds: float) -> int:
    times_s = run.timestamp_us.astype(np.float64) / 1_000_000.0
    return int(np.argmin(np.abs(times_s - seconds)))


def _validate_render_trace(
    prepared: PreparedVisualization, model: mujoco.MjModel
) -> np.ndarray:
    trace = prepared.simulation.render_trace
    if trace is None:
        raise RuntimeError("prepared visualization has no render-state snapshots")
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    expected_shape = (
        len(prepared.resolved.run.timestamp_us),
        mujoco.mj_stateSize(model, state_spec),
    )
    if trace.integration_state.shape != expected_shape:
        raise RuntimeError(
            "render-state snapshot shape differs from the reconstructed model"
        )
    if (
        trace.state_spec != state_spec
        or trace.model_nq != model.nq
        or trace.model_nv != model.nv
    ):
        raise RuntimeError("render-state snapshot model contract changed")
    return trace.integration_state


def _restore_render_snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    snapshots: np.ndarray,
    sample: int,
) -> None:
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(model, data, snapshots[sample], state_spec)
    mujoco.mj_forward(model, data)


def _set_viewer_overlay(
    viewer: Any,
    prepared: PreparedVisualization,
    view: PlaybackView,
    *,
    show_debug: bool,
    mode: str,
) -> None:
    if not hasattr(viewer, "set_texts"):
        raise RuntimeError("installed MuJoCo viewer does not support text overlays")
    model_text, diagnostic_text = format_viewer_overlay(
        prepared,
        view.current_sample,
        show_debug=show_debug,
        mode=mode,
        playback=view,
    )
    model_lines = model_text.splitlines()
    diagnostic_lines = diagnostic_text.splitlines()
    model_split = next(
        (
            index
            for index, line in enumerate(model_lines)
            if line.startswith(("Terrain state:", "Terrain:"))
        ),
        len(model_lines),
    )
    diagnostic_split = next(
        (
            index
            for index, line in enumerate(diagnostic_lines)
            if line == "EVENT TIMING"
        ),
        len(diagnostic_lines),
    )
    panels = [
        (
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "\n".join(model_lines[:model_split]),
        ),
        (
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "\n".join(diagnostic_lines[:diagnostic_split]),
        ),
    ]
    if model_split < len(model_lines):
        panels.append(
            (
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                "\n".join(
                    ("MODEL OUTPUT / TERRAIN ADVISORY (CONTINUED)",)
                    + tuple(model_lines[model_split:])
                ),
            )
        )
    if diagnostic_split < len(diagnostic_lines):
        panels.append(
            (
                mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                "\n".join(diagnostic_lines[diagnostic_split:]),
            )
        )
    viewer.set_texts(
        [
            (
                mujoco.mjtFontScale.mjFONTSCALE_100,
                position,
                text,
                "",
            )
            for position, text in panels
        ]
    )


def play_snapshot_trace(
    prepared: PreparedVisualization,
    playback: SnapshotPlaybackControl,
    *,
    show_debug: bool = False,
    mode: str = "analysis",
) -> PlaybackView:
    """Render immutable states until the user closes the passive viewer."""
    config = prepared.simulation_config
    model, _ = load_g1_model(
        terrain_name=config.terrain,
        sink_pattern=config.sink_pattern,
        sink_severity=config.sink_severity,
        slip_pattern=config.slip_pattern,
        patch_start_x_m=config.patch_start_x_m,
        patch_width_m=config.patch_width_m,
        sink_support_pattern=config.sink_support_pattern,
        source_terrain=config.source_terrain,
    )
    snapshots = _validate_render_trace(prepared, model)
    data = mujoco.MjData(model)
    initial = playback.view()
    _restore_render_snapshot(model, data, snapshots, initial.current_sample)
    viewer_context = launch_passive_viewer(
        model,
        data,
        key_callback=playback.key_callback,
    )
    with viewer_context as viewer:
        rendered_revision = -1
        last_wall_s = time.monotonic()
        next_sync_s = last_wall_s
        sample_accumulator = 0.0
        while viewer.is_running():
            now_s = time.monotonic()
            view = playback.view()
            if view.state is PlaybackState.PLAYING:
                sample_accumulator += max(now_s - last_wall_s, 0.0) * (
                    config.sensor_rate_hz * view.speed
                )
                advance = int(sample_accumulator)
                if advance:
                    view = playback.advance_by(advance)
                    sample_accumulator -= advance
                    if view.state is not PlaybackState.PLAYING:
                        sample_accumulator = 0.0
            else:
                sample_accumulator = 0.0
            last_wall_s = now_s

            paused_change = (
                view.state is not PlaybackState.PLAYING
                and view.revision != rendered_revision
            )
            if rendered_revision < 0 or paused_change or now_s >= next_sync_s:
                with viewer.lock():
                    _restore_render_snapshot(
                        model,
                        data,
                        snapshots,
                        view.current_sample,
                    )
                _set_viewer_overlay(
                    viewer,
                    prepared,
                    view,
                    show_debug=show_debug,
                    mode=mode,
                )
                viewer.sync(state_only=True)
                rendered_revision = view.revision
                next_sync_s = now_s + 1.0 / 60.0
            time.sleep(0.005)
    return playback.view()


def visualize_prepared_run(
    prepared: PreparedVisualization,
    *,
    playback_speed: float = 1.0,
    show_debug: bool = False,
    pause_at_s: float | None = None,
    pause_on_reflex: bool = False,
    single_step: bool = False,
    mode: str = "analysis",
) -> dict[str, object]:
    """Open snapshot playback only after scientific headless parity passes."""
    require_parity(prepared.parity)
    if mode not in VISUALIZATION_MODES:
        raise ValueError(f"visualization mode must be one of {VISUALIZATION_MODES}")
    if not np.isfinite(playback_speed) or playback_speed <= 0.0:
        raise ValueError("viewer playback speed must be positive and finite")
    duration_s = prepared.simulation_config.duration_s
    if pause_at_s is not None and (
        not np.isfinite(pause_at_s) or not 0.0 <= pause_at_s < duration_s
    ):
        raise ValueError(
            f"--pause-at must be in [0.0, {duration_s:.3f}) simulation seconds"
        )
    pause_samples: list[int] = []
    pause_at_sample = None
    if pause_at_s is not None:
        pause_at_sample = _nearest_sample(prepared.resolved.run, pause_at_s)
        pause_samples.append(pause_at_sample)
    reflex_pause_sample = None
    if pause_on_reflex and prepared.traces.first_reflex_sample is not None:
        reflex_pause_sample = prepared.traces.first_reflex_sample
        pause_samples.append(reflex_pause_sample)
    auto_pause_sample = min(pause_samples) if pause_samples else None
    playback = SnapshotPlaybackControl(
        len(prepared.resolved.run.timestamp_us),
        speed=playback_speed,
        events=playback_events(prepared),
        auto_pause_sample=auto_pause_sample,
        start_paused=single_step,
    )
    final_view = play_snapshot_trace(
        prepared,
        playback,
        show_debug=show_debug,
        mode=mode,
    )
    reflex_pause_s = (
        None
        if reflex_pause_sample is None
        else float(
            prepared.resolved.run.timestamp_us[reflex_pause_sample] / 1_000_000.0
        )
    )
    render_trace = prepared.simulation.render_trace
    assert render_trace is not None
    return {
        "run_id": prepared.resolved.run.run_id,
        "split": prepared.resolved.run.split,
        "stored_resimulation_parity": prepared.parity.passed,
        "viewer_physics_parity": None,
        "viewer_physics_executed": False,
        "snapshot_playback": True,
        "scientific_parity_prechecked": True,
        "viewer_closed_cleanly": True,
        "sensor_absolute_tolerance": SENSOR_ABSOLUTE_TOLERANCE,
        "holdout_opened": False,
        "mode": mode,
        "pause_at_s": pause_at_s,
        "pause_at_sample": pause_at_sample,
        "pause_on_reflex": pause_on_reflex,
        "reflex_pause_s": reflex_pause_s,
        "auto_pause_sample": auto_pause_sample,
        "single_step_started_paused": single_step,
        "playback_state_at_close": final_view.state.value,
        "current_sample_at_close": final_view.current_sample,
        "ended_paused_reached": final_view.ended_reached,
        "render_state_samples": render_trace.integration_state.shape[0],
        "render_state_size": render_trace.integration_state.shape[1],
    }
