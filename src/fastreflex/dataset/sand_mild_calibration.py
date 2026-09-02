"""Physical mild-Sand calibration analysis and recalibrated study design."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fastreflex.dataset.generation import _signature, canonical_sha256, sha256_file
from fastreflex.dataset.sand_calibration import (
    _historical_overlap_audit,
    _historical_signatures,
    _scenario_signatures_are_near,
)


MILD_RECALIBRATED_SPLITS = (
    "MILD_RECALIBRATED_DISCOVERY",
    "MILD_RECALIBRATED_CONFIRMATION",
)


def build_mild_physical_ledger(dataset_path: Path) -> list[dict[str, Any]]:
    """Derive model-blind contact/exposure facts for a generated mild corpus."""
    manifest_path = dataset_path / "manifest.json"
    expected_manifest_sha = (
        (dataset_path / "manifest.sha256").read_text(encoding="utf-8").split()[0]
    )
    if sha256_file(manifest_path) != expected_manifest_sha:
        raise ValueError("mild physical ledger manifest integrity failed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = [
        row
        for row in manifest["runs"]
        if row["group"] in {"broad_sand_benign", "sand_benign"}
        and row["sink_severity"] == "mild"
    ]
    ledger: list[dict[str, Any]] = []
    for row in selected:
        if row.get("model_outputs_present") or manifest.get("model_inference_runs"):
            raise ValueError("model output entered mild physical ledger")
        run_path = dataset_path / str(row["file"])
        if sha256_file(run_path) != str(row["file_sha256"]):
            raise ValueError(f"mild physical run integrity failed: {row['run_id']}")
        with np.load(run_path, allow_pickle=False) as payload:
            censor = int(payload["censor_sample"])
            fall_value = int(payload["first_fall_sample"])
            fall = None if fall_value < 0 else fall_value
            target = np.asarray(
                payload["target_terrain_contact"][:censor], dtype=bool
            )
            loaded = np.asarray(payload["loaded_contact"][:censor], dtype=bool)
            target_any = np.any(target, axis=1)
            loaded_target = target & loaded
            loaded_target_any = np.any(loaded_target, axis=1)
            target_samples = np.flatnonzero(target_any)
            loaded_target_samples = np.flatnonzero(loaded_target_any)
            first_target = (
                None if not target_samples.size else int(target_samples[0])
            )
            last_target = None if not target_samples.size else int(target_samples[-1])
            first_loaded_target = (
                None
                if not loaded_target_samples.size
                else int(loaded_target_samples[0])
            )
            last_loaded_target = (
                None
                if not loaded_target_samples.size
                else int(loaded_target_samples[-1])
            )
            episode_ids = np.asarray(payload["contact_episode_id"][:censor])
            episodes = {
                (foot, int(episode_id))
                for foot in range(2)
                for episode_id in episode_ids[:, foot][target[:, foot]]
                if int(episode_id) > 0
            }
        if fall is None:
            fall_relation = "NO_FALL"
        elif first_target is None:
            fall_relation = "PRE_TARGET"
        elif last_target is not None and fall <= last_target + 1:
            fall_relation = "DURING_TARGET_CONTACT"
        else:
            fall_relation = "AFTER_LAST_TARGET_CONTACT"
        physical = row["physical_signature"]
        target_summary = row["target_contact_summary"]
        ledger.append(
            {
                "run_id": row["run_id"],
                "split": row.get("split", "CALIBRATION_ONLY"),
                "source_terrain": row["source_terrain"],
                "speed_mps": row["speed_mps"],
                "topology": row["sink_pattern"],
                "patch_entry_x_m": row["patch_start_x_m"],
                "patch_start_x_m": row["patch_start_x_m"],
                "patch_width_m": row["patch_width_m"],
                "patch_exit_x_m": round(
                    float(row["patch_start_x_m"])
                    + float(row["patch_width_m"]),
                    3,
                ),
                "objective_physical_outcome": row["objective_physical_outcome"],
                "invalid_reason": row["invalid_reason"],
                "first_target_contact_ms": first_target,
                "last_target_contact_ms": last_target,
                "first_loaded_target_contact_ms": first_loaded_target,
                "last_loaded_target_contact_ms": last_loaded_target,
                "first_fall_ms": fall,
                "target_to_fall_ms": (
                    None
                    if fall is None or first_target is None
                    else fall - first_target
                ),
                "fall_relation": fall_relation,
                "cumulative_target_contact_ms": int(np.count_nonzero(target_any)),
                "cumulative_loaded_sand_exposure_ms": int(
                    np.count_nonzero(loaded_target_any)
                ),
                "loaded_foot_sand_exposure_ms": int(
                    np.count_nonzero(loaded_target)
                ),
                "target_contact_episode_count": len(episodes),
                "complete_target_contact_episode_count": target_summary.get(
                    "complete_episode_count"
                ),
                "incomplete_target_contact_episode_count": (
                    int(target_summary.get("episode_count", len(episodes)))
                    - int(target_summary.get("complete_episode_count", 0))
                ),
                "precontact_phase_20ms": target_summary["precontact_phase"],
                "leading_foot": target_summary["leading_foot"],
                "loaded_side_at_contact": target_summary["loaded_side_at_contact"],
                "peak_transition_displacement_m": physical[
                    "peak_transition_displacement_m"
                ],
                "peak_support_spread_m": physical["peak_support_spread_m"],
                "normalized_load_redistribution": physical[
                    "normalized_load_redistribution"
                ],
                "normalized_peak_load_derivative": physical[
                    "normalized_peak_load_derivative"
                ],
                "target_contact_duration_ms": physical[
                    "target_contact_duration_ms"
                ],
            }
        )
    return ledger


def expand_mild_recalibrated_redesign(
    document: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Expand the exact future 176-run mild-recalibrated study matrix."""
    matrix = document["scenario_matrix"]
    rows: list[dict[str, Any]] = []
    group_codes = {
        "near_hazard_sand_benign": "nh",
        "ordinary_support_control": "os",
    }
    for split in MILD_RECALIBRATED_SPLITS:
        split_code = "d" if split == MILD_RECALIBRATED_SPLITS[0] else "c"
        variants = matrix["split_variants"][split]
        profiles = matrix["broad_mild_profiles"][split]
        for cell in matrix["source_speed_cells"]:
            source = str(cell["source_terrain"])
            source_code = "c" if source == "concrete" else "m"
            speed = float(cell["speed_mps"])
            speed_code = f"{int(round(speed * 100)):03d}"
            profile_name = (
                "concrete_0.25"
                if source == "concrete" and speed == 0.25
                else "standard"
            )
            for index, profile in enumerate(profiles[profile_name], start=1):
                rows.append(
                    {
                        **dict(matrix["fixed_mechanics"]["broad_sand_benign"]),
                        **dict(profile),
                        "run_id": (
                            f"{matrix['run_id_prefix']}_{split_code}_bb_"
                            f"{source_code}_{speed_code}_{index:02d}"
                        ),
                        "scenario_family": "broad_sand_benign",
                        "group": "broad_sand_benign",
                        "split": split,
                        "source_terrain": source,
                        "speed_mps": speed,
                        "realization_id": profile["id"],
                    }
                )
            for group, anchor_key in (
                ("near_hazard_sand_benign", "moderate_anchors"),
                ("ordinary_support_control", "ordinary_support_anchors"),
            ):
                anchors = list(cell[anchor_key])
                for index, variant in enumerate(variants[group], start=1):
                    anchor = anchors[(index - 1) % len(anchors)]
                    row = {
                        **dict(matrix["fixed_mechanics"][group]),
                        "run_id": (
                            f"{matrix['run_id_prefix']}_{split_code}_"
                            f"{group_codes[group]}_{source_code}_{speed_code}_"
                            f"{index:02d}"
                        ),
                        "scenario_family": group,
                        "group": group,
                        "split": split,
                        "source_terrain": source,
                        "speed_mps": speed,
                        "patch_start_x_m": round(
                            float(anchor["patch_start_x_m"])
                            + float(variant["start_delta_m"]),
                            3,
                        ),
                        "patch_width_m": round(
                            float(anchor["patch_width_m"])
                            + float(variant["width_delta_m"]),
                            3,
                        ),
                        "sink_pattern": anchor["sink_pattern"],
                        "realization_id": variant["id"],
                    }
                    if group == "ordinary_support_control":
                        row["designed_side"] = anchor["designed_side"]
                    rows.append(row)
        for index, template in enumerate(
            matrix["delayed_support_templates"][split], start=1
        ):
            source = str(template["source_terrain"])
            source_code = "c" if source == "concrete" else "m"
            rows.append(
                {
                    **dict(matrix["fixed_mechanics"]["delayed_support_control"]),
                    **dict(template),
                    "run_id": (
                        f"{matrix['run_id_prefix']}_{split_code}_ds_"
                        f"{source_code}_{index:02d}"
                    ),
                    "scenario_family": "delayed_support_control",
                    "group": "delayed_support_control",
                    "split": split,
                    "speed_mps": 0.25,
                    "designed_side": "LEFT",
                    "realization_id": template["id"],
                }
            )
    return rows


def mild_recalibrated_component_hashes(
    document: Mapping[str, Any],
) -> dict[str, str]:
    """Return the seven deterministic component hashes."""
    keys = {
        "MILD_RECALIBRATED_PARAMETER_DOMAIN_SHA": "parameter_domain",
        "MILD_RECALIBRATED_SCENARIO_MATRIX_SHA": "scenario_matrix",
        "MILD_RECALIBRATED_SPLIT_PLAN_SHA": "split_plan",
        "MILD_RECALIBRATED_PHYSICAL_LABEL_CONTRACT_SHA": (
            "physical_label_contract"
        ),
        "MILD_RECALIBRATED_GENERATION_GATE_SHA": "generation_gates",
        "MILD_RECALIBRATED_DIVERSITY_METRIC_SHA": "diversity_metrics",
        "MILD_RECALIBRATED_CONFIRMATION_PROTOCOL_SHA": "confirmation_protocol",
    }
    return {name: canonical_sha256(document[key]) for name, key in keys.items()}


def _inside(value: float, bounds: Sequence[float]) -> bool:
    return float(bounds[0]) <= value <= float(bounds[1])


def validate_mild_recalibrated_redesign(
    root: Path, document: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed on counts, domain, contamination, split, and hash drift."""
    rows = expand_mild_recalibrated_redesign(document)
    matrix = document["scenario_matrix"]
    expected_counts = matrix["counts"]
    split_counts = Counter(str(row["split"]) for row in rows)
    group_counts = Counter(str(row["group"]) for row in rows)
    if len(rows) != int(expected_counts["total"]):
        raise ValueError("mild-recalibrated total count changed")
    for split in MILD_RECALIBRATED_SPLITS:
        if split_counts[split] != int(expected_counts[split]):
            raise ValueError(f"mild-recalibrated split count changed: {split}")
    for group in (
        "broad_sand_benign",
        "near_hazard_sand_benign",
        "ordinary_support_control",
        "delayed_support_control",
    ):
        if group_counts[group] != int(expected_counts[group]):
            raise ValueError(f"mild-recalibrated group count changed: {group}")

    ids = [str(row["run_id"]) for row in rows]
    signatures = [_signature(row) for row in rows]
    if len(set(ids)) != len(ids) or len(set(signatures)) != len(signatures):
        raise ValueError("mild-recalibrated matrix has duplicate IDs or signatures")

    domain = document["parameter_domain"]["broad_mild"]
    for row in rows:
        if row["group"] != "broad_sand_benign":
            continue
        start = float(row["patch_start_x_m"])
        width = float(row["patch_width_m"])
        patch_exit = round(start + width, 3)
        if row["sink_pattern"] == "transition_left":
            geometry = domain["common_left"]
        elif row["sink_pattern"] == "transition_right":
            if row["source_terrain"] == "concrete" and row["speed_mps"] == 0.25:
                raise ValueError("Concrete/.25 right topology entered future domain")
            geometry = domain["common_right"]
        else:
            raise ValueError(f"unknown mild topology: {row['sink_pattern']}")
        if not all(
            (
                _inside(start, geometry["patch_start_x_m"]),
                _inside(width, geometry["patch_width_m"]),
                _inside(patch_exit, geometry["patch_exit_x_m"]),
            )
        ):
            raise ValueError(f"broad mild profile left frozen domain: {row['run_id']}")

    historical, provenance = _historical_signatures(
        root, document["historical_signature_policy"]["manifests"]
    )
    historical_overlap = len(set(signatures) & historical)
    if historical_overlap:
        raise ValueError("mild-recalibrated matrix overlaps historical signatures")
    near_policy = document["historical_signature_policy"][
        "cross_split_near_duplicate"
    ]
    historical_audit = _historical_overlap_audit(
        root,
        document["historical_signature_policy"]["manifests"],
        rows,
        near_policy,
    )
    if historical_audit["exact_total"] or historical_audit["run_id_reuse_total"]:
        raise ValueError("historical scenario or run ID reuse entered redesign")

    discovery = [
        row for row in rows if row["split"] == MILD_RECALIBRATED_SPLITS[0]
    ]
    confirmation = [
        row for row in rows if row["split"] == MILD_RECALIBRATED_SPLITS[1]
    ]
    exact_overlap = len(
        {_signature(row) for row in discovery}
        & {_signature(row) for row in confirmation}
    )
    near_pairs = [
        (str(left["run_id"]), str(right["run_id"]))
        for left in discovery
        for right in confirmation
        if _scenario_signatures_are_near(
            _signature(left), _signature(right), near_policy
        )
    ]
    if exact_overlap or near_pairs:
        raise ValueError(
            "mild-recalibrated splits have exact or near overlap: "
            f"{near_pairs[:3]}"
        )

    computed = mild_recalibrated_component_hashes(document)
    expected = {
        key: value
        for key, value in document.get("design_hashes", {}).items()
        if key != "SAND_BENIGN_MILD_RECALIBRATED_STUDY_REDESIGN_SHA"
        and value != "TO_BE_FROZEN"
    }
    if expected and computed != expected:
        raise ValueError("mild-recalibrated component hashes changed")
    bundle = {
        "experiment_id": document["experiment"]["id"],
        "dataset_id": document["dataset_plan"]["dataset_id"],
        "counts": matrix["counts"],
        "component_hashes": computed,
    }
    redesign_sha = canonical_sha256(bundle)
    expected_redesign = document.get("design_hashes", {}).get(
        "SAND_BENIGN_MILD_RECALIBRATED_STUDY_REDESIGN_SHA"
    )
    if expected_redesign not in (None, "TO_BE_FROZEN"):
        if redesign_sha != expected_redesign:
            raise ValueError("complete mild-recalibrated study hash changed")
    return {
        "run_count": len(rows),
        "split_counts": dict(split_counts),
        "group_counts": dict(group_counts),
        "unique_run_ids": len(set(ids)),
        "unique_signatures": len(set(signatures)),
        "historical_signature_overlap": historical_overlap,
        "historical_run_id_reuse": historical_audit["run_id_reuse_total"],
        "cross_split_exact_overlap": exact_overlap,
        "cross_split_parameter_near_duplicates": len(near_pairs),
        "scenario_matrix_sha256": canonical_sha256(rows),
        "scenario_signature_sha256": canonical_sha256(
            [list(value) for value in signatures]
        ),
        "split_sha256": {
            split: canonical_sha256(
                [row["run_id"] for row in rows if row["split"] == split]
            )
            for split in MILD_RECALIBRATED_SPLITS
        },
        "historical_manifests": provenance,
        "historical_contamination": historical_audit,
        "component_hashes": computed,
        "redesign_sha256": redesign_sha,
    }
