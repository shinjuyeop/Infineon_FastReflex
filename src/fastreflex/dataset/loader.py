"""Run-disjoint loading and causal windowing for Hazard datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


CLASS_NAMES = ("NORMAL", "SLIP", "SINK")
VALID_OUTCOMES = ("BENIGN", "SLIP", "SINK")
SENSOR_PROFILE_CHANNELS = {"imu6": 6, "fsr8": 8, "fusion14": 14}


@dataclass(frozen=True)
class ManifestRecord:
    """Manifest metadata used for split validation and run lookup."""

    run_id: str
    path: Path
    observed_outcome: str
    scenario_family: str
    terrain: str
    speed_mps: float
    patch_start_x: float | None
    sink_side: str | None
    sink_severity: str | None = None
    sink_support_pattern: str | None = None
    sink_pattern: str | None = None
    split: str | None = None


@dataclass(frozen=True)
class Normalizer:
    """Per-channel z-score parameters fit on explicitly recorded runs."""

    mean: np.ndarray
    std: np.ndarray
    sample_count: int
    fit_run_ids: tuple[str, ...]
    epsilon: float

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.std).astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "per_channel_zscore",
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "sample_count": self.sample_count,
            "fit_run_ids": list(self.fit_run_ids),
            "epsilon": self.epsilon,
        }


@dataclass(frozen=True)
class WindowSet:
    """Materialized causal IMU windows and their source provenance."""

    inputs: np.ndarray
    targets: np.ndarray
    run_ids: np.ndarray
    endpoint_samples: np.ndarray
    available_by_class: tuple[int, int, int]

    @property
    def selected_by_class(self) -> tuple[int, int, int]:
        counts = np.bincount(self.targets, minlength=3)
        return tuple(int(value) for value in counts[:3])

    def __len__(self) -> int:
        return int(self.targets.shape[0])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(dataset_path: Path) -> dict[str, ManifestRecord]:
    """Load manifest metadata without touching run waveform arrays."""
    dataset_path = dataset_path.resolve()
    records: dict[str, ManifestRecord] = {}
    with (dataset_path / "manifest.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            run_id = row["run_id"]
            if run_id in records:
                raise ValueError(f"duplicate manifest run_id: {run_id}")
            relative_path = Path(row["file"])
            run_path = (dataset_path / relative_path).resolve()
            try:
                run_path.relative_to(dataset_path)
            except ValueError as exc:
                raise ValueError(f"run file escapes dataset root: {run_id}") from exc
            records[run_id] = ManifestRecord(
                run_id=run_id,
                path=run_path,
                observed_outcome=row["observed_outcome"],
                scenario_family=row["scenario_family"],
                terrain=row["terrain"],
                speed_mps=float(row["speed_mps"]),
                patch_start_x=(
                    None if not row["patch_start_x"] else float(row["patch_start_x"])
                ),
                sink_side=row["sink_side"] or None,
                sink_severity=row.get("sink_severity") or None,
                sink_support_pattern=row.get("sink_support_pattern") or None,
                sink_pattern=row.get("sink_pattern") or None,
                split=row.get("split") or None,
            )
    return records


def validate_split(
    records: Mapping[str, ManifestRecord],
    split: Mapping[str, Sequence[str]],
    expected_outcome_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, dict[str, int]]:
    """Validate run-disjoint membership and exclusion of invalid outcomes."""
    required = ("train", "validation", "holdout")
    if set(split) != set(required):
        raise ValueError("split must contain train, validation, and holdout")
    seen: set[str] = set()
    counts: dict[str, dict[str, int]] = {}
    for split_name in required:
        identifiers = list(split[split_name])
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"duplicate run inside {split_name}")
        overlap = seen.intersection(identifiers)
        if overlap:
            raise ValueError(f"run-disjoint split violation: {sorted(overlap)}")
        seen.update(identifiers)
        outcome_counts = {name: 0 for name in VALID_OUTCOMES}
        for run_id in identifiers:
            if run_id not in records:
                raise ValueError(f"split run is absent from manifest: {run_id}")
            outcome = records[run_id].observed_outcome
            if outcome not in VALID_OUTCOMES:
                raise ValueError(f"excluded outcome in split: {run_id}={outcome}")
            outcome_counts[outcome] += 1
        counts[split_name] = outcome_counts
    valid_manifest_ids = {
        run_id
        for run_id, record in records.items()
        if record.observed_outcome in VALID_OUTCOMES
    }
    if seen != valid_manifest_ids:
        missing = sorted(valid_manifest_ids - seen)
        extra = sorted(seen - valid_manifest_ids)
        raise ValueError(f"split must cover every valid run; missing={missing}, extra={extra}")
    if expected_outcome_counts is not None:
        for split_name, expected in expected_outcome_counts.items():
            if counts[split_name] != dict(expected):
                raise ValueError(
                    f"unexpected {split_name} outcome counts: {counts[split_name]}"
                )
    return counts


def _load_runtime_arrays(record: ManifestRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if record.observed_outcome not in VALID_OUTCOMES:
        raise ValueError(f"refusing excluded run: {record.run_id}")
    with np.load(record.path, allow_pickle=False) as stored:
        imu = np.asarray(stored["pelvis_imu"], dtype=np.float32)
        labels = np.asarray(stored["hazard_class_id"], dtype=np.int8)
        eligible = np.asarray(stored["training_eligible"], dtype=bool)
    if imu.ndim != 2 or imu.shape[1] != 6:
        raise ValueError(f"invalid pelvis_imu shape in {record.run_id}: {imu.shape}")
    if labels.shape != (imu.shape[0],) or eligible.shape != labels.shape:
        raise ValueError(f"annotation shape mismatch in {record.run_id}")
    used = eligible & (labels >= 0) & (labels <= 2)
    if not np.isfinite(imu[used]).all():
        raise ValueError(f"non-finite eligible pelvis_imu in {record.run_id}")
    return imu, labels, eligible


def extract_sensor_profile(
    pelvis_imu: np.ndarray,
    foot_fsr: np.ndarray,
    profile: str,
) -> np.ndarray:
    """Select one frozen raw-channel sensor profile without derived features."""
    imu = np.asarray(pelvis_imu, dtype=np.float32)
    fsr = np.asarray(foot_fsr, dtype=np.float32)
    if imu.ndim != 2 or imu.shape[1] != 6:
        raise ValueError("pelvis_imu must have shape [N,6]")
    if fsr.ndim != 2 or fsr.shape != (len(imu), 8):
        raise ValueError("foot_fsr must have shape [N,8] aligned to pelvis_imu")
    if profile == "imu6":
        return imu
    if profile == "fsr8":
        return fsr
    if profile == "fusion14":
        return np.concatenate((imu, fsr), axis=1).astype(np.float32, copy=False)
    raise ValueError(f"unsupported sensor profile: {profile}")


def _first_nonnegative(values: np.ndarray) -> int | None:
    valid = np.asarray(values, dtype=np.int64)
    valid = valid[valid >= 0]
    return None if not len(valid) else int(valid.min())


def _early_target_annotations(
    stored: Mapping[str, np.ndarray], observed_outcome: str
) -> tuple[np.ndarray, np.ndarray]:
    """Build experiment-local early targets without altering raw annotations."""
    valid = np.asarray(stored["sample_valid"], dtype=bool) & np.asarray(
        stored["pre_fall_valid"], dtype=bool
    )
    labels = np.full(len(valid), -1, dtype=np.int8)
    censor = int(stored["first_censor_sample"])
    t3 = len(valid) if censor < 0 else censor
    valid[t3:] = False
    if observed_outcome == "BENIGN":
        labels[valid] = 0
        return labels, labels >= 0

    t0 = _first_nonnegative(stored["first_patch_contact_sample_per_foot"])
    if t0 is None:
        raise ValueError("hazard-positive run has no physical patch contact")
    labels[:t0][valid[:t0]] = 0
    if observed_outcome == "SLIP":
        t1 = int(stored["first_any_slip_onset_sample"])
    elif observed_outcome == "SINK":
        t1 = _first_nonnegative(
            stored["first_sink_physical_onset_sample_per_foot"]
        )
        if not bool(stored["hazardous_sink_episode"]):
            raise ValueError("SINK run lacks retrospective hazard qualification")
    else:
        raise ValueError(f"early targets do not support outcome: {observed_outcome}")
    if t1 is None or t1 < t0 or t1 >= t3:
        raise ValueError("invalid early-target t0/t1/t3 ordering")
    class_id = 1 if observed_outcome == "SLIP" else 2
    labels[t1:t3][valid[t1:t3]] = class_id
    return labels, labels >= 0


def load_profile_arrays(
    record: ManifestRecord,
    profile: str,
    *,
    early_targets: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load aligned raw profile channels and either raw or early-target labels."""
    if record.observed_outcome not in VALID_OUTCOMES:
        raise ValueError(f"refusing excluded run: {record.run_id}")
    with np.load(record.path, allow_pickle=False) as stored_file:
        stored = {name: stored_file[name] for name in stored_file.files}
    if "foot_fsr" not in stored:
        raise ValueError(f"sensor field absent from run: {record.run_id}")
    values = extract_sensor_profile(stored["pelvis_imu"], stored["foot_fsr"], profile)
    if early_targets:
        labels, eligible = _early_target_annotations(stored, record.observed_outcome)
    else:
        labels = np.asarray(stored["hazard_class_id"], dtype=np.int8)
        eligible = np.asarray(stored["training_eligible"], dtype=bool)
    if labels.shape != (len(values),) or eligible.shape != labels.shape:
        raise ValueError(f"annotation shape mismatch in {record.run_id}")
    if not np.isfinite(values[eligible]).all():
        raise ValueError(f"non-finite eligible sensor input in {record.run_id}")
    return values, labels, eligible


def fit_profile_normalizer(
    records: Mapping[str, ManifestRecord],
    run_ids: Iterable[str],
    profile: str,
    *,
    early_targets: bool,
    epsilon: float = 1.0e-8,
) -> Normalizer:
    """Fit profile-specific per-channel moments on declared training runs only."""
    identifiers = tuple(run_ids)
    channel_count = SENSOR_PROFILE_CHANNELS.get(profile)
    if channel_count is None:
        raise ValueError(f"unsupported sensor profile: {profile}")
    total = 0
    channel_sum = np.zeros(channel_count, dtype=np.float64)
    channel_square_sum = np.zeros(channel_count, dtype=np.float64)
    for run_id in identifiers:
        values, labels, eligible = load_profile_arrays(
            records[run_id], profile, early_targets=early_targets
        )
        selected = values[eligible & (labels >= 0) & (labels <= 2)].astype(np.float64)
        total += len(selected)
        channel_sum += selected.sum(axis=0)
        channel_square_sum += np.square(selected).sum(axis=0)
    if total == 0:
        raise ValueError("normalizer has no eligible training samples")
    mean = channel_sum / total
    variance = np.maximum(channel_square_sum / total - np.square(mean), 0.0)
    raw_std = np.sqrt(variance)
    if np.any(raw_std <= epsilon):
        raise ValueError("near-constant sensor channel in training split")
    return Normalizer(
        mean=mean.astype(np.float32),
        std=np.maximum(raw_std, epsilon).astype(np.float32),
        sample_count=total,
        fit_run_ids=identifiers,
        epsilon=epsilon,
    )


def fit_normalizer(
    records: Mapping[str, ManifestRecord],
    run_ids: Iterable[str],
    epsilon: float = 1.0e-8,
) -> Normalizer:
    """Fit per-channel moments from eligible samples in the declared runs only."""
    identifiers = tuple(run_ids)
    total = 0
    channel_sum = np.zeros(6, dtype=np.float64)
    channel_square_sum = np.zeros(6, dtype=np.float64)
    for run_id in identifiers:
        imu, labels, eligible = _load_runtime_arrays(records[run_id])
        mask = eligible & (labels >= 0) & (labels <= 2)
        selected = imu[mask].astype(np.float64)
        total += int(selected.shape[0])
        channel_sum += selected.sum(axis=0)
        channel_square_sum += np.square(selected).sum(axis=0)
    if total == 0:
        raise ValueError("normalizer has no eligible training samples")
    mean = channel_sum / total
    variance = np.maximum(channel_square_sum / total - np.square(mean), 0.0)
    raw_std = np.sqrt(variance)
    if np.any(raw_std <= epsilon):
        raise ValueError("near-constant IMU channel in training split")
    return Normalizer(
        mean=mean.astype(np.float32),
        std=np.maximum(raw_std, epsilon).astype(np.float32),
        sample_count=total,
        fit_run_ids=identifiers,
        epsilon=epsilon,
    )


def _segment_endpoints(
    labels: np.ndarray,
    eligible: np.ndarray,
    window_samples: int,
    stride_samples: int,
) -> dict[int, np.ndarray]:
    valid = eligible & (labels >= 0) & (labels <= 2)
    if not valid.any():
        return {0: np.empty(0, dtype=np.int64), 1: np.empty(0, dtype=np.int64), 2: np.empty(0, dtype=np.int64)}
    changes = np.flatnonzero(
        (valid[1:] != valid[:-1])
        | (labels[1:] != labels[:-1])
    ) + 1
    boundaries = np.concatenate(([0], changes, [len(labels)]))
    result: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: []}
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if not valid[start] or stop - start < window_samples:
            continue
        class_id = int(labels[start])
        endpoints = np.arange(
            start + window_samples - 1,
            stop,
            stride_samples,
            dtype=np.int64,
        )
        result[class_id].append(endpoints)
    return {
        class_id: (
            np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        )
        for class_id, parts in result.items()
    }


def build_windows(
    records: Mapping[str, ManifestRecord],
    run_ids: Iterable[str],
    window_samples: int,
    stride_samples: int,
    normalizer: Normalizer | None,
    cap_per_run_class: int | None = None,
) -> WindowSet:
    """Materialize causal windows without crossing eligibility or class boundaries."""
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window and stride must be positive")
    if cap_per_run_class is not None and cap_per_run_class <= 0:
        raise ValueError("window cap must be positive")
    input_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    run_parts: list[np.ndarray] = []
    endpoint_parts: list[np.ndarray] = []
    available = np.zeros(3, dtype=np.int64)
    for run_id in run_ids:
        record = records[run_id]
        imu, labels, eligible = _load_runtime_arrays(record)
        endpoints_by_class = _segment_endpoints(
            labels, eligible, window_samples, stride_samples
        )
        for class_id in range(3):
            endpoints = endpoints_by_class[class_id]
            available[class_id] += len(endpoints)
            if cap_per_run_class is not None and len(endpoints) > cap_per_run_class:
                indices = np.linspace(
                    0, len(endpoints) - 1, cap_per_run_class, dtype=np.int64
                )
                endpoints = endpoints[indices]
            if not len(endpoints):
                continue
            windows = np.stack(
                [imu[end - window_samples + 1 : end + 1] for end in endpoints]
            )
            if normalizer is not None:
                windows = normalizer.transform(windows)
            input_parts.append(windows.astype(np.float32, copy=False))
            target_parts.append(np.full(len(endpoints), class_id, dtype=np.int64))
            run_parts.append(np.full(len(endpoints), run_id, dtype=object))
            endpoint_parts.append(endpoints)
    if not input_parts:
        raise ValueError("no eligible windows were produced")
    inputs = np.concatenate(input_parts, axis=0)
    targets = np.concatenate(target_parts, axis=0)
    source_run_ids = np.concatenate(run_parts, axis=0)
    endpoint_samples = np.concatenate(endpoint_parts, axis=0)
    order = np.lexsort((endpoint_samples, source_run_ids.astype(str)))
    return WindowSet(
        inputs=inputs[order],
        targets=targets[order],
        run_ids=source_run_ids[order],
        endpoint_samples=endpoint_samples[order],
        available_by_class=tuple(int(value) for value in available),
    )


def build_profile_windows(
    records: Mapping[str, ManifestRecord],
    run_ids: Iterable[str],
    profile: str,
    window_samples: int,
    stride_samples: int,
    normalizer: Normalizer | None,
    *,
    early_targets: bool,
    cap_per_run_class: int | None = None,
) -> WindowSet:
    """Materialize like-for-like early-target windows for a sensor profile."""
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window and stride must be positive")
    input_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    run_parts: list[np.ndarray] = []
    endpoint_parts: list[np.ndarray] = []
    available = np.zeros(3, dtype=np.int64)
    for run_id in run_ids:
        values, labels, eligible = load_profile_arrays(
            records[run_id], profile, early_targets=early_targets
        )
        endpoints_by_class = _segment_endpoints(
            labels, eligible, window_samples, stride_samples
        )
        for class_id in range(3):
            endpoints = endpoints_by_class[class_id]
            available[class_id] += len(endpoints)
            if cap_per_run_class is not None and len(endpoints) > cap_per_run_class:
                indices = np.linspace(
                    0, len(endpoints) - 1, cap_per_run_class, dtype=np.int64
                )
                endpoints = endpoints[indices]
            if not len(endpoints):
                continue
            windows = np.stack(
                [values[end - window_samples + 1 : end + 1] for end in endpoints]
            )
            if normalizer is not None:
                windows = normalizer.transform(windows)
            input_parts.append(windows.astype(np.float32, copy=False))
            target_parts.append(np.full(len(endpoints), class_id, dtype=np.int64))
            run_parts.append(np.full(len(endpoints), run_id, dtype=object))
            endpoint_parts.append(endpoints)
    if not input_parts:
        raise ValueError("no eligible windows were produced")
    inputs = np.concatenate(input_parts)
    targets = np.concatenate(target_parts)
    source_run_ids = np.concatenate(run_parts)
    endpoint_samples = np.concatenate(endpoint_parts)
    order = np.lexsort((endpoint_samples, source_run_ids.astype(str)))
    return WindowSet(
        inputs[order],
        targets[order],
        source_run_ids[order],
        endpoint_samples[order],
        tuple(int(value) for value in available),
    )
