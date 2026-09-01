"""Inference, replay, metrics, and freeze verification for Unified Hazard."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from fastreflex.dataset.hazard import (
    LABEL_BOTH,
    LABEL_NO_HAZARD,
    LABEL_PRECURSOR_ONLY,
    LABEL_SLIP,
    LABEL_SUPPORT,
    HazardRun,
    canonical_sha256,
    physical_hazard_label,
    slip_event_sample,
    support_event_sample,
)
from fastreflex.dataset.loader import Normalizer, sha256_file
from fastreflex.dataset.generation import HazardRunAnnotations
from fastreflex.evaluation.terrain import TERRAIN_STATE_NAMES, TerrainTrace
from fastreflex.features import (
    HAZARD_FEATURE_DIMENSION,
    HAZARD_FEATURE_SCHEMA_SHA256,
    extract_hazard_features,
    feature_schema_hash,
)
from fastreflex.models.baselines import parameter_count
from fastreflex.training.trainer import load_checkpoint


HISTORY_MS = 20
THRESHOLD = 0.99
PERSISTENCE_MS = 5
PARAMETERS = 11_010
SUPPORTED_FREEZE_SHA256 = (
    "91834c88bea3012fb8d3ec049b047017a449f7fbb3b28ce5b4a3b63afcda08c2"
)
SUPPORTED_VERDICT = "UNIFIED_HAZARD_REFLEX_SUPPORTED_SINGLE_IMU"


@dataclass(frozen=True)
class HazardReplay:
    """Continuous 1 kHz probability trace over causal endpoints."""

    endpoints: np.ndarray
    probabilities: np.ndarray


def load_hazard_normalizer(path: Path) -> Normalizer:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    if (
        mean.shape != (HAZARD_FEATURE_DIMENSION,)
        or std.shape != (HAZARD_FEATURE_DIMENSION,)
        or np.any(std <= 0.0)
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(std))
    ):
        raise ValueError("frozen Hazard normalizer contract changed")
    return Normalizer(
        mean=mean,
        std=std,
        sample_count=int(payload["sample_count"]),
        fit_run_ids=tuple(str(value) for value in payload["fit_run_ids"]),
        epsilon=float(payload["epsilon"]),
    )


def predict_hazard_windows(
    models: Sequence[torch.nn.Module], windows: np.ndarray
) -> np.ndarray:
    """Average the selected three-seed binary GRU probabilities."""
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (
        HISTORY_MS,
        HAZARD_FEATURE_DIMENSION,
    ):
        raise ValueError("Hazard model input must have shape [batch,20,80]")
    tensor = torch.from_numpy(values)
    with torch.no_grad():
        probability = [
            torch.softmax(model(tensor), dim=1)[:, 1].cpu().numpy()
            for model in models
        ]
    return np.mean(np.stack(probability), axis=0).astype(np.float64)


def replay_hazard_run(
    run: HazardRun,
    normalizer: Normalizer,
    models: Sequence[torch.nn.Module],
    *,
    history_ms: int = HISTORY_MS,
    batch_size: int = 512,
) -> HazardReplay:
    """Replay at 1 ms using only current and past Pelvis IMU samples."""
    if history_ms != HISTORY_MS:
        raise ValueError("supported Hazard history is frozen at 20 ms")
    features = extract_hazard_features(run.features["PELVIS_IMU6"])
    stop = run.censor_sample
    if run.fall_sample_diagnostic is not None:
        stop = min(stop, int(run.fall_sample_diagnostic))
    endpoints = np.arange(history_ms - 1, stop, dtype=np.int64)
    offsets = np.arange(history_ms - 1, -1, -1, dtype=np.int64)
    chunks: list[np.ndarray] = []
    for first in range(0, len(endpoints), batch_size):
        selected = endpoints[first : first + batch_size]
        indices = selected[:, None] - offsets[None, :]
        windows = normalizer.transform(features[indices]).astype(
            np.float32, copy=False
        )
        chunks.append(predict_hazard_windows(models, windows))
    return HazardReplay(
        endpoints=endpoints,
        probabilities=(
            np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
        ),
    )


def replay_hazard_runs(
    runs: Mapping[str, HazardRun],
    normalizer: Normalizer,
    checkpoint_paths: Sequence[Path],
) -> dict[str, HazardReplay]:
    models = [load_checkpoint(path)[0] for path in checkpoint_paths]
    return {
        run_id: replay_hazard_run(run, normalizer, models)
        for run_id, run in sorted(runs.items())
    }


def sustained_reflex(
    probabilities: np.ndarray,
    threshold: float = THRESHOLD,
    persistence_ms: int = PERSISTENCE_MS,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen inclusive threshold and consecutive persistence."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or persistence_ms <= 0:
        raise ValueError("invalid Hazard persistence inputs")
    alert = np.zeros(len(values), dtype=bool)
    onset = np.zeros(len(values), dtype=bool)
    count = 0
    previous = False
    for index, probability in enumerate(values):
        passes = bool(probability >= threshold)
        count = count + 1 if passes else 0
        current = count >= persistence_ms
        alert[index] = current
        onset[index] = current and not previous
        previous = current
    return alert, onset


def reflex_onset_samples(
    replay: HazardReplay,
    threshold: float = THRESHOLD,
    persistence_ms: int = PERSISTENCE_MS,
) -> np.ndarray:
    _, onset = sustained_reflex(replay.probabilities, threshold, persistence_ms)
    return replay.endpoints[onset]


def reflex_required_trace(
    replay: HazardReplay,
    sample_count: int,
    threshold: float = THRESHOLD,
    persistence_ms: int = PERSISTENCE_MS,
) -> np.ndarray:
    """Materialize the control decision without consulting Terrain state."""
    alert, _ = sustained_reflex(replay.probabilities, threshold, persistence_ms)
    result = np.zeros(sample_count, dtype=bool)
    if np.any(replay.endpoints < 0) or np.any(replay.endpoints >= sample_count):
        raise ValueError("Hazard replay endpoints exceed the runtime trace")
    result[replay.endpoints] = alert
    return result


def _distribution(
    values: Sequence[int | float | None],
) -> dict[str, float | None]:
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


def _first_in_ranges(
    values: np.ndarray, ranges: Sequence[tuple[int, int]]
) -> int | None:
    selected = [
        int(value)
        for value in values
        if any(lower <= int(value) <= upper for lower, upper in ranges)
    ]
    return None if not selected else min(selected)


def evaluate_hazard_replays(
    runs: Mapping[str, HazardRun],
    replays: Mapping[str, HazardReplay],
    *,
    precursor_samples: Mapping[str, int | None],
    terrain: Mapping[str, TerrainTrace] | None = None,
    threshold: float = THRESHOLD,
    persistence_ms: int = PERSISTENCE_MS,
) -> dict[str, object]:
    """Score physical Hazard; Terrain is optional advisory annotation only."""
    rows: list[dict[str, object]] = []
    slip_latency: list[int] = []
    support_latency: list[int] = []
    support_precursor_latency: list[int] = []
    support_lead: list[int] = []
    for run_id, run in sorted(runs.items()):
        onsets = reflex_onset_samples(replays[run_id], threshold, persistence_ms)
        slip = slip_event_sample(run)
        support = support_event_sample(run)
        precursor = precursor_samples.get(run_id)
        ranges: list[tuple[int, int]] = []
        if slip is not None:
            ranges.append((slip - 30, slip + 40))
        if support is not None:
            ranges.append(
                ((support if precursor is None else int(precursor)), support + 50)
            )
        earliest = min((lower for lower, _ in ranges), default=None)
        first = None if not len(onsets) else int(onsets[0])
        valid = _first_in_ranges(onsets, ranges)
        premature = earliest is not None and first is not None and first < earliest
        if premature:
            valid = None
        slip_valid = (
            None
            if slip is None or premature
            else _first_in_ranges(onsets, ((slip - 30, slip + 40),))
        )
        support_valid = (
            None
            if support is None or premature
            else _first_in_ranges(
                onsets,
                (((support if precursor is None else int(precursor)), support + 50),),
            )
        )
        if slip is not None and slip_valid is not None:
            slip_latency.append(slip_valid - slip)
        if support is not None and support_valid is not None:
            support_latency.append(support_valid - support)
            support_lead.append(support - support_valid)
            if precursor is not None:
                support_precursor_latency.append(support_valid - int(precursor))
        label = physical_hazard_label(run, precursor)
        no_hazard = label == LABEL_NO_HAZARD
        terrain_state = None
        if terrain is not None and first is not None and first < len(terrain[run_id].state):
            terrain_state = int(terrain[run_id].state[first])
        rows.append(
            {
                "run_id": run_id,
                "split": run.split,
                "target_terrain": run.target_terrain,
                "hard_ground": run.hard_stable_control,
                "physical_label": label,
                "slip_sample": slip,
                "support_precursor_sample": precursor,
                "support_sample": support,
                "system_first_onset": first,
                "first_valid_detection": valid,
                "valid_detection": valid is not None,
                "premature": bool(premature),
                "system_false_positive": bool(no_hazard and first is not None),
                "slip_valid_detection": slip_valid,
                "support_valid_detection": support_valid,
                "terrain_at_detection": (
                    None
                    if terrain_state is None
                    else TERRAIN_STATE_NAMES[terrain_state]
                ),
            }
        )
    hazards = [
        row
        for row in rows
        if row["physical_label"] in (LABEL_SLIP, LABEL_SUPPORT, LABEL_BOTH)
    ]
    slip_rows = [row for row in rows if row["slip_sample"] is not None]
    support_rows = [row for row in rows if row["support_sample"] is not None]
    no_hazard_rows = [row for row in rows if row["physical_label"] == LABEL_NO_HAZARD]
    sand = [
        row
        for row in no_hazard_rows
        if row["target_terrain"] == "sand" and not row["hard_ground"]
    ]
    hard = [row for row in no_hazard_rows if row["hard_ground"]]

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

    premature = sum(bool(row["premature"]) for row in hazards)
    return {
        "runs": len(rows),
        "hazard_runs": len(hazards),
        "overall_hazard_recall": recall(hazards),
        "slip_hazard_runs": len(slip_rows),
        "slip_hazard_recall": recall(slip_rows),
        "support_hazard_runs": len(support_rows),
        "support_hazard_recall": recall(support_rows),
        "primary_no_hazard_runs": len(no_hazard_rows),
        "primary_no_hazard_specificity": specificity(no_hazard_rows),
        "sand_benign_runs": len(sand),
        "sand_benign_specificity": specificity(sand),
        "hard_ground_runs": len(hard),
        "hard_ground_specificity": specificity(hard),
        "system_premature_runs": premature,
        "system_premature_run_rate": (
            0.0 if not hazards else premature / len(hazards)
        ),
        "slip_latency_ms": _distribution(slip_latency),
        "support_precursor_latency_ms": _distribution(support_precursor_latency),
        "support_established_latency_ms": _distribution(support_latency),
        "support_lead_ms": _distribution(support_lead),
        "precursor_only_runs_excluded_from_specificity": sum(
            row["physical_label"] == LABEL_PRECURSOR_ONLY for row in rows
        ),
        "terrain_used_as_gate": False,
        "rows": rows,
    }


def validation_gate_results(
    metrics: Mapping[str, object], gates: Mapping[str, object]
) -> dict[str, bool]:
    slip_p95 = metrics["slip_latency_ms"]["p95"]
    support_p95 = metrics["support_established_latency_ms"]["p95"]
    support_median_lead = metrics["support_lead_ms"]["median"]
    result = {
        "overall_hazard_recall": float(metrics["overall_hazard_recall"])
        >= float(gates["overall_hazard_recall_min"]),
        "slip_hazard_recall": float(metrics["slip_hazard_recall"])
        >= float(gates["slip_hazard_recall_min"]),
        "support_hazard_recall": float(metrics["support_hazard_recall"])
        >= float(gates["support_hazard_recall_min"]),
        "primary_no_hazard_specificity": float(
            metrics["primary_no_hazard_specificity"]
        )
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
    }
    if "median_support_lead_ms_min" in gates:
        result["median_support_lead"] = support_median_lead is not None and float(
            support_median_lead
        ) >= float(gates["median_support_lead_ms_min"])
    return result


def _run_recall(rows: Sequence[Mapping[str, object]]) -> float | None:
    if not rows:
        return None
    return sum(bool(row["valid_detection"]) for row in rows) / len(rows)


def _run_specificity(rows: Sequence[Mapping[str, object]]) -> float | None:
    if not rows:
        return None
    return 1.0 - sum(bool(row["system_false_positive"]) for row in rows) / len(rows)


def _family_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    hazards = [row for row in rows if row["physical_label"] in (LABEL_SLIP, LABEL_SUPPORT, LABEL_BOTH)]
    no_hazard = [row for row in rows if row["physical_label"] == LABEL_NO_HAZARD]
    return {
        "runs": len(rows),
        "hazard_runs": len(hazards),
        "hazard_recall": _run_recall(hazards),
        "no_established_hazard_runs": len(no_hazard),
        "specificity": _run_specificity(no_hazard),
        "false_reflex": sum(bool(row["system_false_positive"]) for row in no_hazard),
        "premature_hazard_runs": sum(bool(row["premature"]) for row in hazards),
    }


def _longest_threshold_excursion(values: np.ndarray, threshold: float) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if float(value) >= threshold else 0
        longest = max(longest, current)
    return longest


def evaluate_ice_precursor_secondary(
    runs: Mapping[str, HazardRun],
    replays: Mapping[str, HazardReplay],
    annotations: Mapping[str, HazardRunAnnotations],
    manifest_rows: Mapping[str, Mapping[str, object]],
    *,
    threshold: float = THRESHOLD,
    persistence_ms: int = PERSISTENCE_MS,
) -> dict[str, object]:
    """Describe frozen 30--50 mm Ice episodes without rewriting primary scores."""
    counts: Counter[str] = Counter()
    precursor_to_reflex: list[int] = []
    reflex_to_slip: list[int] = []
    outcome: dict[str, Counter[str]] = {
        name: Counter()
        for name in (
            "SAME_EPISODE_SLIP",
            "NEXT_EPISODE_SLIP",
            "LATER_SLIP",
            "BENIGN_RELEASE",
            "CENSORED",
        )
    }
    for run_id, run in sorted(runs.items()):
        annotation = annotations[run_id]
        replay = replays[run_id]
        alert = reflex_required_trace(
            replay, len(run.timestamp_us), threshold, persistence_ms
        )
        for episode in manifest_rows[run_id]["ice_precursor_summary"]["episodes"]:
            name = str(episode["future_outcome"])
            side = int(episode["foot_id"])
            start = int(episode["start_sample"])
            end = min(int(episode["end_sample_exclusive"]), run.censor_sample)
            candidate = np.flatnonzero(
                annotation.ice_precursor_candidate[start:end, side]
            ) + start
            if not len(candidate):
                raise RuntimeError("manifest precursor episode has no candidate samples")
            crossing = int(episode["first_30mm_crossing_sample"])
            candidate_alerts = candidate[alert[candidate]]
            alert_in_candidate = bool(len(candidate_alerts))
            future_slip = episode["established_slip_sample"]
            before_slip = False
            if future_slip is not None:
                future_slip = int(future_slip)
                before_slip = bool(np.any(alert[crossing:future_slip]))
            outcome[name]["episodes"] += 1
            outcome[name]["alert_in_candidate"] += int(alert_in_candidate)
            outcome[name]["alert_before_slip"] += int(before_slip)
            counts["episodes"] += 1
            if name in (
                "SAME_EPISODE_SLIP",
                "NEXT_EPISODE_SLIP",
                "LATER_SLIP",
            ):
                counts["future_slip_episodes"] += 1
                counts["future_slip_alert_in_candidate"] += int(alert_in_candidate)
                counts["future_slip_alert_before_slip"] += int(before_slip)
                counts["precursor_states_missed"] += int(not alert_in_candidate)
            elif name == "BENIGN_RELEASE":
                release_alert = bool(np.any(alert[crossing:end]))
                counts["benign_release_episodes"] += 1
                counts["benign_release_false_alerts"] += int(release_alert)
            else:
                counts["censored_episodes"] += 1
                counts["censored_alert_in_candidate"] += int(alert_in_candidate)
            if alert_in_candidate:
                first_alert = int(candidate_alerts[0])
                precursor_to_reflex.append(first_alert - crossing)
                if future_slip is not None:
                    reflex_to_slip.append(future_slip - first_alert)
    return {
        **{key: int(value) for key, value in sorted(counts.items())},
        "by_outcome": {
            name: {key: int(value) for key, value in sorted(values.items())}
            for name, values in outcome.items()
        },
        "precursor_to_reflex_ms": _distribution(precursor_to_reflex),
        "reflex_to_established_slip_ms": _distribution(reflex_to_slip),
        "primary_scores_rewritten": False,
    }


def evaluate_model_v2_validation(
    runs: Mapping[str, HazardRun],
    replays: Mapping[str, HazardReplay],
    precursor_samples: Mapping[str, int | None],
    annotations: Mapping[str, HazardRunAnnotations],
    manifest_rows: Mapping[str, Mapping[str, object]],
    gates: Mapping[str, object],
    *,
    threshold: float = THRESHOLD,
    persistence_ms: int = PERSISTENCE_MS,
) -> dict[str, object]:
    """Apply primary V1 timing plus frozen V2 family/precursor audits."""
    primary = evaluate_hazard_replays(
        runs,
        replays,
        precursor_samples=precursor_samples,
        threshold=threshold,
        persistence_ms=persistence_ms,
    )
    rows = {str(row["run_id"]): row for row in primary["rows"]}
    confirmed = [
        row
        for run_id, row in rows.items()
        if row["physical_label"] == LABEL_NO_HAZARD
        and precursor_samples.get(run_id) is None
        and not np.any(annotations[run_id].censored_precursor)
    ]
    primary["primary_no_hazard_runs"] = len(confirmed)
    primary["primary_no_hazard_specificity"] = _run_specificity(confirmed)
    primary["false_reflex_count"] = sum(
        bool(row["system_false_positive"]) for row in confirmed
    )
    primary["missed_hazard_count"] = sum(
        not bool(row["valid_detection"])
        for row in rows.values()
        if row["physical_label"] in (LABEL_SLIP, LABEL_SUPPORT, LABEL_BOTH)
    )
    family_rows: dict[str, list[Mapping[str, object]]] = {}
    for run_id, row in rows.items():
        family_rows.setdefault(
            str(manifest_rows[run_id]["scenario_family"]), []
        ).append(row)
    families = {
        family: _family_summary(selected)
        for family, selected in sorted(family_rows.items())
    }
    ice_benign = families["ICE_BENIGN_CONTROL"]
    staged = families["STAGED_SAND_BENIGN_CONTROL"]
    speed_sand = families["SPEED_STRATIFIED_SAND_BENIGN"]
    delayed_rows = family_rows["ONE_CONTACT_DELAYED_ICE_SLIP"]
    families["ONE_CONTACT_DELAYED_ICE_SLIP"].update(
        {
            "exactly_one_runs": sum(
                manifest_rows[str(row["run_id"])]["delayed_ice_summary"]["classification"]
                == "EXACTLY_ONE_BENIGN_CONTACT_BEFORE_SLIP"
                for row in delayed_rows
            ),
            "multi_contact_runs": sum(
                manifest_rows[str(row["run_id"])]["delayed_ice_summary"]["classification"]
                == "MULTI_CONTACT_DELAYED_SLIP"
                for row in delayed_rows
            ),
        }
    )
    speed: dict[str, object] = {}
    for value in (0.20, 0.25, 0.30):
        selected_ids = {
            run_id
            for run_id, row in manifest_rows.items()
            if float(row["nominal_speed_mps"]) == value
        }
        selected = [rows[run_id] for run_id in selected_ids]
        slip = [row for row in selected if row["slip_sample"] is not None]
        support = [row for row in selected if row["support_sample"] is not None]
        no_hazard = [
            row
            for row in selected
            if row["physical_label"] == LABEL_NO_HAZARD
            and precursor_samples.get(str(row["run_id"])) is None
            and not np.any(annotations[str(row["run_id"])].censored_precursor)
        ]
        speed[f"{value:.2f}"] = {
            "slip": [sum(bool(row["valid_detection"]) for row in slip), len(slip)],
            "slip_recall": _run_recall(slip),
            "support": [sum(bool(row["valid_detection"]) for row in support), len(support)],
            "support_recall": _run_recall(support),
            "confirmed_no_hazard": [
                sum(not bool(row["system_false_positive"]) for row in no_hazard),
                len(no_hazard),
            ],
            "confirmed_no_hazard_specificity": _run_specificity(no_hazard),
        }
    side: dict[str, object] = {}
    for subtype, side_field in (
        ("slip", "slip_event_summary"),
        ("support", "support_event_summary"),
    ):
        side[subtype] = {}
        for name in ("LEFT_ONLY", "RIGHT_ONLY", "BILATERAL"):
            selected = [
                rows[run_id]
                for run_id, manifest_row in manifest_rows.items()
                if manifest_row[side_field]["side"] == name
            ]
            side[subtype][name] = {
                "detected": sum(bool(row["valid_detection"]) for row in selected),
                "runs": len(selected),
                "recall": _run_recall(selected),
            }
    ice_benign_rows = family_rows["ICE_BENIGN_CONTROL"]
    ice_benign_confirmed = [
        row
        for row in ice_benign_rows
        if row["physical_label"] == LABEL_NO_HAZARD
        and not np.any(annotations[str(row["run_id"])].censored_precursor)
    ]
    families["ICE_BENIGN_CONTROL"].update(
        {
            "confirmed_no_hazard_runs": len(ice_benign_confirmed),
            "confirmed_specificity": _run_specificity(ice_benign_confirmed),
            "threshold_excursion_ms": _distribution(
                [
                    _longest_threshold_excursion(
                        replays[str(row["run_id"])].probabilities, threshold
                    )
                    for row in ice_benign_rows
                    if row["physical_label"] == LABEL_NO_HAZARD
                ]
            ),
        }
    )
    precursor = evaluate_ice_precursor_secondary(
        runs,
        replays,
        annotations,
        manifest_rows,
        threshold=threshold,
        persistence_ms=persistence_ms,
    )
    right_support = side["support"]["RIGHT_ONLY"]["recall"]
    metric_gates = {
        "overall_hazard_recall": float(primary["overall_hazard_recall"])
        >= float(gates["overall_hazard_recall_min"]),
        "slip_hazard_recall": float(primary["slip_hazard_recall"])
        >= float(gates["slip_hazard_recall_min"]),
        "support_hazard_recall": float(primary["support_hazard_recall"])
        >= float(gates["support_hazard_recall_min"]),
        "primary_no_hazard_specificity": primary["primary_no_hazard_specificity"]
        is not None
        and float(primary["primary_no_hazard_specificity"])
        >= float(gates["primary_no_hazard_specificity_min"]),
        "ice_benign_specificity": ice_benign["specificity"] is not None
        and float(ice_benign["specificity"])
        >= float(gates["ice_benign_specificity_min"]),
        "system_premature_run_rate": float(primary["system_premature_run_rate"])
        <= float(gates["system_premature_run_rate_max"]),
        "slip_p95_latency_ms": primary["slip_latency_ms"]["p95"] is not None
        and float(primary["slip_latency_ms"]["p95"])
        <= float(gates["slip_p95_latency_ms_max"]),
        "support_p95_established_latency_ms": primary["support_established_latency_ms"]["p95"]
        is not None
        and float(primary["support_established_latency_ms"]["p95"])
        <= float(gates["support_p95_established_latency_ms_max"]),
        "staged_sand_specificity": staged["specificity"] is not None
        and float(staged["specificity"])
        >= float(gates["staged_sand_specificity_min"]),
        "speed_sand_specificity": speed_sand["specificity"] is not None
        and float(speed_sand["specificity"])
        >= float(gates["speed_sand_specificity_min"]),
        "right_support_recall": right_support is not None
        and float(right_support) >= float(gates["right_support_recall_min"]),
    }
    return {
        "primary": primary,
        "families": families,
        "speed": speed,
        "side": side,
        "ice_precursor_secondary": precursor,
        "gates": metric_gates,
        "all_gates_passed": all(metric_gates.values()),
    }


def verify_model_v2_training_result(
    root: Path, config_path: Path
) -> dict[str, object]:
    """Verify the preserved V2 training result without replaying validation."""
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if document["experiment"]["id"] != "MODEL_V2_DATA_ONLY_TRAINING":
        raise ValueError("unsupported Model V2 training config")
    artifact_path = root / str(document["artifacts"]["path"])
    result_path = artifact_path / "training_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["training_config_sha256"] != sha256_file(config_path):
        raise RuntimeError("Model V2 training config changed")
    candidate_path = artifact_path / "candidate_freeze.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if sha256_file(candidate_path) != result["candidate_freeze_sha256"]:
        raise RuntimeError("Model V2 candidate freeze changed")
    normalizer_path = artifact_path / "normalization/gru_history20.json"
    if sha256_file(normalizer_path) != candidate["normalizer_sha256"]:
        raise RuntimeError("Model V2 normalizer changed")
    for relative, expected in candidate["checkpoint_sha256"].items():
        if sha256_file(root / str(relative)) != str(expected):
            raise RuntimeError(f"Model V2 checkpoint changed: {relative}")
    if candidate["generalization_holdout_guard_count"] != 0:
        raise RuntimeError("Generalization HOLDOUT guard changed")
    return {
        "passed": True,
        "training_verdict": result["training_verdict"],
        "internal_validation_verdict": result["internal_validation_verdict"],
        "candidate_freeze_sha256": result["candidate_freeze_sha256"],
        "model_v2_freeze_created": (artifact_path / "model_v2_freeze.json").is_file(),
        "generalization_validation_v2_inference": result[
            "generalization_validation_v2_inference"
        ],
        "generalization_holdout_guard_count": result[
            "generalization_holdout_guard_count"
        ],
    }


def verify_supported_candidate(
    repository_root: Path, document: Mapping[str, object]
) -> dict[str, object]:
    """Verify protected current artifacts without opening scientific HOLDOUT."""
    root = repository_root.resolve()
    freeze_path = (
        root
        / "artifacts/runs/20260829_unified_hazard_reflex_system"
        / "selection_before_holdout.json"
    )
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    declared = str(freeze["artifact_sha256"])
    unhashed = {key: value for key, value in freeze.items() if key != "artifact_sha256"}
    if declared != canonical_sha256(unhashed) or declared != SUPPORTED_FREEZE_SHA256:
        raise RuntimeError("supported Unified Hazard freeze identity changed")
    selection = freeze["selection"]
    if (
        freeze["architecture"] != "PHASE_B_UNIFIED_HAZARD_DETECTOR"
        or int(selection["feature_dimension"]) != HAZARD_FEATURE_DIMENSION
        or str(selection["feature_schema_sha256"])
        != HAZARD_FEATURE_SCHEMA_SHA256
        or feature_schema_hash() != HAZARD_FEATURE_SCHEMA_SHA256
        or str(selection["model_family"]) != "gru"
        or int(selection["history_ms"]) != HISTORY_MS
        or float(selection["threshold"]) != THRESHOLD
        or int(selection["persistence_ms"]) != PERSISTENCE_MS
        or int(selection["parameters"]) != PARAMETERS
        or bool(freeze["terrain_used_as_gate"])
    ):
        raise RuntimeError("supported Unified Hazard contract changed")

    normalizer_path = root / str(selection["normalizer_path"])
    if sha256_file(normalizer_path) != str(selection["normalizer_sha256"]):
        raise RuntimeError("supported Hazard normalizer changed")
    load_hazard_normalizer(normalizer_path)
    checkpoint_hashes = selection["checkpoint_sha256"]
    checkpoint_paths = tuple(root / str(value) for value in checkpoint_hashes)
    for path in checkpoint_paths:
        relative = str(path.relative_to(root))
        if sha256_file(path) != str(checkpoint_hashes[relative]):
            raise RuntimeError(f"supported Hazard checkpoint changed: {relative}")
        model, metadata = load_checkpoint(path)
        if (
            metadata["family"] != "gru"
            or int(metadata["window_samples"]) != HISTORY_MS
            or int(metadata["input_channels"]) != HAZARD_FEATURE_DIMENSION
            or parameter_count(model) != PARAMETERS
        ):
            raise RuntimeError("supported Hazard GRU architecture changed")

    terrain = document["terrain_advisory"]
    protected_terrain = {
        str(terrain["normalizer"]["path"]): str(terrain["normalizer"]["sha256"]),
        **{
            str(row["path"]): str(row["sha256"])
            for row in terrain["checkpoints"]
        },
    }
    for relative, expected in protected_terrain.items():
        if sha256_file(root / relative) != expected:
            raise RuntimeError(f"protected Terrain artifact changed: {relative}")
    return {
        "passed": True,
        "verdict": SUPPORTED_VERDICT,
        "freeze_sha256": declared,
        "normalizer_sha256": selection["normalizer_sha256"],
        "checkpoint_sha256": checkpoint_hashes,
        "feature_schema_sha256": feature_schema_hash(),
        "history_ms": HISTORY_MS,
        "threshold": THRESHOLD,
        "persistence_ms": PERSISTENCE_MS,
        "parameters": PARAMETERS,
        "terrain_artifact_sha256": protected_terrain,
        "terrain_used_as_gate": False,
        "holdout_opened": False,
    }
