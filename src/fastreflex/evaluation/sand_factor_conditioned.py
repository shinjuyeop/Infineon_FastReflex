"""Fresh factor-conditioned Sand development-validation evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fastreflex.dataset.hazard import HazardRun
from fastreflex.evaluation.hazard import (
    HazardReplay,
    evaluate_hazard_replays,
    reflex_onset_samples,
)


EPSILON = 1.0e-8


def _validation_distribution(values: Sequence[float | int]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            key: None
            for key in ("minimum", "median", "p75", "p90", "p95", "maximum")
        }
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _longest_threshold_streak(probability: np.ndarray, threshold: float) -> int:
    longest = 0
    current = 0
    for above in np.asarray(probability) >= threshold:
        current = current + 1 if bool(above) else 0
        longest = max(longest, current)
    return longest


def evaluate_factor_conditioned_validation(
    runs: Mapping[str, HazardRun],
    replays: Mapping[str, HazardReplay],
    manifest_rows: Mapping[str, Mapping[str, Any]],
    precursor_samples: Mapping[str, int | None],
    *,
    threshold: float = 0.99,
    persistence_ms: int = 5,
    adverse_threshold: float = 0.95,
) -> dict[str, Any]:
    """Score one frozen model on the fresh factor-conditioned validation split."""
    primary = evaluate_hazard_replays(
        runs,
        replays,
        precursor_samples=precursor_samples,
        threshold=threshold,
        persistence_ms=persistence_ms,
    )
    primary_rows = {str(row["run_id"]): row for row in primary["rows"]}
    detailed: list[dict[str, Any]] = []
    for run_id in sorted(runs):
        row = manifest_rows[run_id]
        replay = replays[run_id]
        probability = np.asarray(replay.probabilities, dtype=np.float64)
        maximum = float(np.max(probability)) if len(probability) else 0.0
        onsets = reflex_onset_samples(replay, threshold, persistence_ms)
        physical = primary_rows[run_id]
        detailed.append(
            {
                "run_id": run_id,
                "group": row["group"],
                "source": row["source_terrain"],
                "speed_mps": float(row["speed_mps"]),
                "topology": row["sink_pattern"],
                "precontact_phase": row["target_contact_summary"]["precontact_phase"],
                "severity": row["actual_benign_severity"],
                "factor_manifold": row["factor_manifold"],
                "support_side": row["support_event_summary"]["side"],
                "physical_outcome": row["objective_physical_outcome"],
                "max_probability": maximum,
                "longest_threshold_streak": _longest_threshold_streak(
                    probability, threshold
                ),
                "adverse_margin": maximum >= adverse_threshold,
                "reflex": bool(len(onsets)),
                "first_reflex": None if not len(onsets) else int(onsets[0]),
                "valid_detection": bool(physical["valid_detection"]),
                "premature": bool(physical["premature"]),
                "support_precursor_sample": physical["support_precursor_sample"],
                "support_sample": physical["support_sample"],
            }
        )
    sand = [
        row
        for row in detailed
        if row["group"] in {"sand_benign_mild", "sand_benign_moderate"}
        and row["physical_outcome"] == "STRICT_BENIGN"
    ]
    support = [
        row
        for row in detailed
        if row["group"] in {"ordinary_support_control", "delayed_support_control"}
        and row["physical_outcome"] == "SUPPORT"
    ]
    actual_slip = [
        row
        for row in detailed
        if row["physical_outcome"] in {"SLIP", "DUAL_HAZARD"}
    ]

    def sand_summary(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        maxima = [float(row["max_probability"]) for row in selected]
        fp = sum(bool(row["reflex"]) for row in selected)
        adverse = sum(bool(row["adverse_margin"]) for row in selected)
        return {
            "runs": len(selected),
            "specific": len(selected) - fp,
            "specificity": 1.0
            if not selected
            else (len(selected) - fp) / len(selected),
            "false_reflex": fp,
            "adverse": adverse,
            "adverse_rate": 0.0 if not selected else adverse / len(selected),
            "max_probability": _validation_distribution(maxima),
        }

    bins = {
        "below_0.90": 0,
        "0.90_to_0.95": 0,
        "0.95_to_0.99": 0,
        "at_least_0.99_subpersistent": 0,
        "reflex": 0,
    }
    for row in sand:
        value = float(row["max_probability"])
        if row["reflex"]:
            bins["reflex"] += 1
        elif value >= threshold:
            bins["at_least_0.99_subpersistent"] += 1
        elif value >= adverse_threshold:
            bins["0.95_to_0.99"] += 1
        elif value >= 0.90:
            bins["0.90_to_0.95"] += 1
        else:
            bins["below_0.90"] += 1

    factor_definitions = {
        "transition_left": lambda row: row["topology"] == "transition_left",
        "transition_right": lambda row: row["topology"] == "transition_right",
        "right_single_precontact": lambda row: row["precontact_phase"]
        == "RIGHT_SINGLE_SUPPORT",
        "left_single_precontact": lambda row: row["precontact_phase"]
        == "LEFT_SINGLE_SUPPORT",
        "concrete": lambda row: row["source"] == "concrete",
        "marble": lambda row: row["source"] == "marble",
        "speed_0.20": lambda row: row["speed_mps"] == 0.20,
        "speed_0.25": lambda row: row["speed_mps"] == 0.25,
        "speed_0.30": lambda row: row["speed_mps"] == 0.30,
        "mild": lambda row: row["severity"] == "LOW",
        "moderate": lambda row: row["severity"] == "MEDIUM",
        "adverse_direction_manifold": lambda row: row["factor_manifold"]
        == "ADVERSE_DIRECTION",
        "comparison_direction_manifold": lambda row: row["factor_manifold"]
        == "COMPARISON_DIRECTION",
        "concrete_025_exception": lambda row: row["factor_manifold"]
        == "CONCRETE_025_ADVERSE_EXCEPTION",
        "transition_left_right_single": lambda row: row["topology"]
        == "transition_left"
        and row["precontact_phase"] == "RIGHT_SINGLE_SUPPORT",
        "transition_right_left_single": lambda row: row["topology"]
        == "transition_right"
        and row["precontact_phase"] == "LEFT_SINGLE_SUPPORT",
    }
    factors = {
        name: sand_summary([row for row in sand if predicate(row)])
        for name, predicate in factor_definitions.items()
    }
    source_speed = {
        f"{source}_{speed:.2f}": sand_summary(
            [
                row
                for row in sand
                if row["source"] == source and row["speed_mps"] == speed
            ]
        )
        for source in ("concrete", "marble")
        for speed in (0.20, 0.25, 0.30)
    }

    def support_summary(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        detected = sum(bool(row["valid_detection"]) for row in selected)
        i1_to_reflex = [
            int(row["first_reflex"]) - int(row["support_precursor_sample"])
            for row in selected
            if row["valid_detection"]
            and row["first_reflex"] is not None
            and row["support_precursor_sample"] is not None
        ]
        reflex_to_support = [
            int(row["support_sample"]) - int(row["first_reflex"])
            for row in selected
            if row["valid_detection"]
            and row["first_reflex"] is not None
            and row["support_sample"] is not None
        ]
        return {
            "runs": len(selected),
            "detected": detected,
            "recall": 0.0 if not selected else detected / len(selected),
            "pre_i1_false_response": sum(bool(row["premature"]) for row in selected),
            "i1_to_reflex_ms": _validation_distribution(i1_to_reflex),
            "reflex_to_support_ms": _validation_distribution(reflex_to_support),
        }

    support_groups = {
        "overall": support_summary(support),
        "ordinary": support_summary(
            [row for row in support if row["group"] == "ordinary_support_control"]
        ),
        "delayed": support_summary(
            [row for row in support if row["group"] == "delayed_support_control"]
        ),
        "concrete": support_summary(
            [row for row in support if row["source"] == "concrete"]
        ),
        "marble": support_summary(
            [row for row in support if row["source"] == "marble"]
        ),
        "left": support_summary(
            [row for row in support if row["support_side"] == "LEFT_ONLY"]
        ),
        "right": support_summary(
            [row for row in support if row["support_side"] == "RIGHT_ONLY"]
        ),
    }
    support_groups["by_speed"] = {
        f"{speed:.2f}": support_summary(
            [row for row in support if row["speed_mps"] == speed]
        )
        for speed in (0.20, 0.25, 0.30)
    }
    all_manifest_rows = list(manifest_rows.values())
    invalid = [
        row
        for row in all_manifest_rows
        if row["split"] == "FACTOR_VALIDATION" and not bool(row["valid"])
    ]
    return {
        "schema_version": 1,
        "split": "FACTOR_VALIDATION",
        "threshold": threshold,
        "persistence_ms": persistence_ms,
        "adverse_threshold": adverse_threshold,
        "sand": {
            "overall": sand_summary(sand),
            "mild": sand_summary([row for row in sand if row["severity"] == "LOW"]),
            "moderate": sand_summary(
                [row for row in sand if row["severity"] == "MEDIUM"]
            ),
            "margin_bins": bins,
            "factors": factors,
            "source_speed": source_speed,
        },
        "support": support_groups,
        "actual_slip": {
            **support_summary(actual_slip),
            "interpretation": "secondary_descriptive_tiny_denominator",
        },
        "provenance_only": {
            "invalid_or_censored_runs": len(invalid),
            "run_ids": sorted(str(row["run_id"]) for row in invalid),
            "excluded_from_primary_metrics": True,
        },
        "run_results": detailed,
        "terrain_used_as_gate": False,
    }
