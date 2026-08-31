"""Canonical dataset contract for the supported unified Hazard pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from fastreflex.dataset.loader import sha256_file

EXPERIMENT_ID = "UNIFIED_HAZARD_REFLEX_SYSTEM_VALIDATION"
PELVIS_IMU6 = "PELVIS_IMU6"
PELVIS_IMU6_FSR8 = "PELVIS_IMU6_FSR8"
EVENT_TYPE_NONE = "NONE"
EVENT_TYPE_SLIP = "SLIP"
EVENT_TYPE_SUPPORT = "SUPPORT"
EVENT_TYPE_BOTH = "SLIP_AND_SUPPORT"
LABEL_SLIP = "SLIP_HAZARD"
LABEL_SUPPORT = "SUPPORT_HAZARD"
LABEL_BOTH = "SLIP_AND_SUPPORT_HAZARD"
LABEL_NO_HAZARD = "NO_HAZARD"
LABEL_PRECURSOR_ONLY = "SUPPORT_PRECURSOR_ONLY"
GROUPS = (
    "ICE_SLIP_HAZARD",
    "SAND_SUPPORT_HAZARD",
    "SAND_BENIGN",
    "HARD_GROUND_NORMAL",
)
PHYSICAL_SIGNATURE_FIELDS = (
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


@dataclass(frozen=True)
class HazardRun:
    """One run with runtime tensors separated from privileged scoring clocks."""

    run_id: str
    split: str
    source_terrain: str
    target_terrain: str
    design_role: str
    first_contact_sample: int
    first_touchdown_sample: int
    censor_sample: int
    outcome_diagnostic: str
    fall_sample_diagnostic: int | None
    features: Mapping[str, np.ndarray]
    timestamp_us: np.ndarray
    slip_event_samples_per_foot: tuple[int | None, int | None]
    support_event_samples_per_foot: tuple[int | None, int | None]
    event_sample: int | None
    event_type: str
    hard_stable_control: bool
    drift_m: np.ndarray
    tangential_velocity_mps: np.ndarray
    support_spread_m: np.ndarray
    support_max_displacement_m: np.ndarray
    loaded_contact: np.ndarray
    sink_pattern: str
    support_pattern: str


class HoldoutGuard:
    """Fail closed unless the sealed waveform set is explicitly opened once."""

    def __init__(self) -> None:
        self._opened = False
        self._open_count = 0

    def open_once(self) -> None:
        if self._opened or self._open_count:
            raise RuntimeError("Hazard HOLDOUT may be opened exactly once")
        self._opened = True
        self._open_count = 1

    def require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("Hazard HOLDOUT waveform access is sealed")

    @property
    def open_count(self) -> int:
        return self._open_count


def load_yaml(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, Mapping):
        raise ValueError("Hazard config must be a YAML mapping")
    return document


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def physical_signature(specification: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(specification[field] for field in PHYSICAL_SIGNATURE_FIELDS)


def split_for_source_index(source: str, index: int) -> str:
    """Frozen source-balanced 38/13/13 split for every 64-run group."""
    if source not in ("concrete", "marble") or not 1 <= index <= 32:
        raise ValueError("source/index is outside the frozen unified matrix")
    if index <= 19:
        return "train"
    validation_last = 26 if source == "concrete" else 25
    return "validation" if index <= validation_last else "holdout"


def generate_hazard_specifications(
    document: Mapping[str, object],
) -> list[dict[str, object]]:
    """Expand the frozen 256-run matrix without reading any outcome."""
    groups = document["dataset"]["groups"]
    abbreviations = {
        "ICE_SLIP_HAZARD": "ice_h",
        "SAND_SUPPORT_HAZARD": "sand_h",
        "SAND_BENIGN": "sand_b",
        "HARD_GROUND_NORMAL": "hard_n",
    }
    specifications: list[dict[str, object]] = []
    for group in GROUPS:
        config = groups[group]
        for source in ("concrete", "marble"):
            for index in range(1, 33):
                common = {
                    "id": f"uhr_{abbreviations[group]}_{source[0]}{index:02d}",
                    "group": group,
                    "split": split_for_source_index(source, index),
                    "design_role": group.lower(),
                    "source_terrain": source,
                    "speed_mps": float(config.get("speed_mps", 0.25)),
                    "hard_stable_control": group == "HARD_GROUND_NORMAL",
                }
                if group == "ICE_SLIP_HAZARD":
                    width = config["width_schedule"][source]
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
                        **config["mechanics"],
                    }
                elif group == "SAND_SUPPORT_HAZARD":
                    anchor = (index - 1) % 2
                    local = (index - 1) // 2
                    width = config["width_schedule_per_anchor"][source]
                    specification = {
                        **common,
                        "target_terrain": "sand",
                        "patch_start_x_m": float(
                            config["alternating_patch_starts"][anchor]
                        ),
                        "patch_width_m": round(
                            float(width["first"]) + local * float(width["step"]), 5
                        ),
                        **config["mechanics"],
                    }
                elif group == "SAND_BENIGN":
                    anchor = (index - 1) % 2
                    local = (index - 1) // 2
                    patch = config["alternating_patch"][anchor]
                    width = config["width_schedule_per_anchor"][source]
                    specification = {
                        **common,
                        "target_terrain": "sand",
                        "patch_start_x_m": float(patch["start"]),
                        "patch_width_m": round(
                            float(width["first"]) + local * float(width["step"]), 5
                        ),
                        "sink_pattern": str(patch["sink_pattern"]),
                        **config["mechanics"],
                    }
                else:
                    speed = config["speed_schedule"]
                    specification = {
                        **common,
                        "target_terrain": source,
                        "speed_mps": round(
                            float(speed["first"]) + (index - 1) * float(speed["step"]),
                            5,
                        ),
                        "patch_start_x_m": 0.35,
                        "patch_width_m": 0.75,
                        **config["mechanics"],
                    }
                specification["intended_role"] = (
                    "fall"
                    if group in ("ICE_SLIP_HAZARD", "SAND_SUPPORT_HAZARD")
                    else "stable"
                )
                specifications.append(specification)
    return specifications


def _signature_from_csv_row(row: Mapping[str, str]) -> tuple[object, ...] | None:
    if not all(row.get(key, "") != "" for key in PHYSICAL_SIGNATURE_FIELDS):
        return None
    return tuple(
        float(row[key])
        if key in ("speed_mps", "patch_start_x_m", "patch_width_m")
        else row[key]
        for key in PHYSICAL_SIGNATURE_FIELDS
    )


def prior_physical_signatures(
    root: Path, document: Mapping[str, object]
) -> set[tuple[object, ...]]:
    result: set[tuple[object, ...]] = set()
    for record in document["source"]["prior_manifests"]:
        path = root / str(record["path"])
        if sha256_file(path) != str(record["sha256"]):
            raise RuntimeError(f"prior manifest changed: {record['path']}")
        if path.suffix == ".json":
            manifest = json.loads(path.read_text(encoding="utf-8"))
            result.update(
                tuple(row["physical_signature"])
                for row in manifest.get("runs", ())
                if "physical_signature" in row
            )
        else:
            with path.open("r", encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    signature = _signature_from_csv_row(row)
                    if signature is not None:
                        result.add(signature)
    return result


def validate_hazard_design(
    root: Path,
    document: Mapping[str, object],
    specifications: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Validate frozen physical semantics, uniqueness, and split boundaries."""
    if document["experiment"]["id"] != EXPERIMENT_ID:
        raise ValueError("unsupported unified Hazard experiment")
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
    duplicates = len(signatures) - len(set(signatures))
    overlap = len(set(signatures) & prior)
    if any(value != expected for value in counts.values()):
        raise ValueError("unified per-group split changed")
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


def slip_event_sample(run: HazardRun) -> int | None:
    values = [
        int(value) for value in run.slip_event_samples_per_foot if value is not None
    ]
    return None if not values else min(values)


def support_event_sample(run: HazardRun) -> int | None:
    values = [
        int(value) for value in run.support_event_samples_per_foot if value is not None
    ]
    return None if not values else min(values)


def i1_support_precursor_sample(
    run: HazardRun, *, threshold: float = 0.0, persistence_ms: int = 20
) -> int | None:
    """First causal loaded-foot positive spread-derivative confirmation."""
    active = i1_support_precursor_trace(
        run, threshold=threshold, persistence_ms=persistence_ms
    )
    indices = np.flatnonzero(active)
    return None if not len(indices) else int(indices[0])


def i1_support_precursor_trace(
    run: HazardRun, *, threshold: float = 0.0, persistence_ms: int = 20
) -> np.ndarray:
    """Causal I1 active trace using the frozen loaded-support definition."""
    if persistence_ms <= 0:
        raise ValueError("I1 persistence must be positive")
    spread = np.asarray(run.support_spread_m, dtype=np.float64)
    derivative = np.zeros_like(spread)
    derivative[1:] = spread[1:] - spread[:-1]
    score = np.max(
        np.where(run.loaded_contact, np.maximum(derivative, 0.0), 0.0), axis=1
    )
    active = np.zeros(len(score), dtype=bool)
    count = 0
    for sample in range(run.first_contact_sample, run.censor_sample):
        count = count + 1 if score[sample] > threshold else 0
        active[sample] = count >= persistence_ms
    return active


def physical_hazard_label(run: HazardRun, precursor: int | None) -> str:
    """Established Slip OR Support defines Hazard; other fields never do."""
    slip = slip_event_sample(run) is not None
    support = support_event_sample(run) is not None
    if slip and support:
        return LABEL_BOTH
    if slip:
        return LABEL_SLIP
    if support:
        return LABEL_SUPPORT
    if precursor is not None:
        return LABEL_PRECURSOR_ONLY
    return LABEL_NO_HAZARD


def _optional_sample(value: int) -> int | None:
    return None if value < 0 else int(value)


def _load_hazard_run(path: Path, row: Mapping[str, object]) -> HazardRun:
    with np.load(path, allow_pickle=False) as payload:
        timestamp = np.asarray(payload["timestamp_us"], dtype=np.int64)
        imu = np.asarray(payload["pelvis_imu6"], dtype=np.float32)
        fsr = np.asarray(payload["foot_fsr8"], dtype=np.float32)
        drift = np.asarray(payload["tangential_anchor_drift_m"], dtype=np.float32)
        velocity = np.asarray(payload["tangential_velocity_mps"], dtype=np.float32)
        spread = np.asarray(payload["support_surface_spread_m"], dtype=np.float32)
        deformation = np.asarray(
            payload["support_surface_max_displacement_m"], dtype=np.float32
        )
        loaded = np.asarray(payload["loaded_contact"], dtype=bool)
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
        event = _optional_sample(int(payload["first_reflex_event_sample"]))
    samples = len(timestamp)
    if (
        imu.shape != (samples, 6)
        or fsr.shape != (samples, 8)
        or drift.shape != (samples, 2)
        or velocity.shape != (samples, 2)
        or spread.shape != (samples, 2)
        or deformation.shape != (samples, 2)
        or loaded.shape != (samples, 2)
        or not np.all(np.isfinite(imu))
        or not np.all(np.isfinite(fsr))
        or np.any(fsr < 0.0)
        or not (0 <= contact < censor <= samples)
    ):
        raise ValueError(f"Hazard run {row['run_id']} contains invalid tensors")
    fusion = np.concatenate((imu, fsr), axis=1).astype(np.float32, copy=False)
    if event != row["event_sample"]:
        raise ValueError("Hazard event clock differs between manifest and payload")
    return HazardRun(
        run_id=str(row["run_id"]),
        split=str(row["split"]),
        source_terrain=str(row["source_terrain"]),
        target_terrain=str(row["target_terrain"]),
        design_role=str(row["design_role_diagnostic_only"]),
        first_contact_sample=contact,
        first_touchdown_sample=touchdown,
        censor_sample=censor,
        outcome_diagnostic=str(row["observed_outcome_diagnostic_only"]),
        fall_sample_diagnostic=(
            None
            if row["fall_sample_diagnostic_only"] is None
            else int(row["fall_sample_diagnostic_only"])
        ),
        features={PELVIS_IMU6: imu, PELVIS_IMU6_FSR8: fusion},
        timestamp_us=timestamp,
        slip_event_samples_per_foot=slip,  # type: ignore[arg-type]
        support_event_samples_per_foot=support,  # type: ignore[arg-type]
        event_sample=event,
        event_type=str(row["event_type"]),
        hard_stable_control=bool(row["hard_stable_control"]),
        drift_m=drift,
        tangential_velocity_mps=velocity,
        support_spread_m=spread,
        support_max_displacement_m=deformation,
        loaded_contact=loaded,
        sink_pattern=str(row["sink_pattern"]),
        support_pattern=str(row["support_pattern"]),
    )


def load_hazard_runs(
    dataset_path: Path,
    manifest: Mapping[str, object],
    splits: Sequence[str],
    *,
    holdout_guard: HoldoutGuard | None = None,
) -> dict[str, HazardRun]:
    """Load only requested splits, enforcing sealed HOLDOUT access."""
    selected_splits = tuple(str(value) for value in splits)
    if "holdout" in selected_splits:
        if holdout_guard is None:
            raise RuntimeError("HOLDOUT loading requires an explicit guard")
        holdout_guard.require_open()
    runs: dict[str, HazardRun] = {}
    for row in manifest["runs"]:
        if str(row["split"]) not in selected_splits:
            continue
        path = dataset_path / str(row["file"])
        if sha256_file(path) != str(row["file_sha256"]):
            raise ValueError(f"Hazard dataset integrity failed: {path.name}")
        run = _load_hazard_run(path, row)
        runs[run.run_id] = run
    return runs


def load_hazard_manifest(dataset_path: Path) -> Mapping[str, object]:
    path = dataset_path / "manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    declared = dataset_path / "manifest.sha256"
    if declared.is_file():
        expected = declared.read_text(encoding="utf-8").split()[0]
        if sha256_file(path) != expected:
            raise ValueError("Hazard manifest integrity failed")
    return document
