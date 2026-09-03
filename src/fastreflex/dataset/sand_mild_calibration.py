"""Physical mild-Sand calibration analysis and recalibrated study design."""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fastreflex.dataset.generation import (
    _load_yaml,
    _signature,
    _write_deterministic_npz,
    _write_json,
    annotate_model_v2_result,
    canonical_sha256,
    sha256_file,
)
from fastreflex.dataset.sand_calibration import (
    _annotation_specification,
    _calibration_result_summary,
    _historical_overlap_audit,
    _historical_signatures,
    _scenario_signatures_are_near,
    audit_sand_benign_redesigned_manifest,
)
from fastreflex.simulation.g1 import SimulationConfig, run_simulation


MILD_RECALIBRATED_SPLITS = (
    "MILD_RECALIBRATED_DISCOVERY",
    "MILD_RECALIBRATED_CONFIRMATION",
)
MILD_RECALIBRATED_GENERATION_ID = (
    "SAND_BENIGN_GENERALIZATION_STUDY_MILD_RECALIBRATED_GENERATION"
)
MILD_RECALIBRATED_DATASET_ID = (
    "sand_benign_generalization_mild_recalibrated_study_20260903"
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
            target = np.asarray(payload["target_terrain_contact"][:censor], dtype=bool)
            loaded = np.asarray(payload["loaded_contact"][:censor], dtype=bool)
            target_any = np.any(target, axis=1)
            loaded_target = target & loaded
            loaded_target_any = np.any(loaded_target, axis=1)
            target_samples = np.flatnonzero(target_any)
            loaded_target_samples = np.flatnonzero(loaded_target_any)
            first_target = None if not target_samples.size else int(target_samples[0])
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
                    float(row["patch_start_x_m"]) + float(row["patch_width_m"]),
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
                "loaded_foot_sand_exposure_ms": int(np.count_nonzero(loaded_target)),
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
                "target_contact_duration_ms": physical["target_contact_duration_ms"],
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
        "MILD_RECALIBRATED_PHYSICAL_LABEL_CONTRACT_SHA": ("physical_label_contract"),
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
    near_policy = document["historical_signature_policy"]["cross_split_near_duplicate"]
    historical_audit = _historical_overlap_audit(
        root,
        document["historical_signature_policy"]["manifests"],
        rows,
        near_policy,
    )
    if historical_audit["exact_total"] or historical_audit["run_id_reuse_total"]:
        raise ValueError("historical scenario or run ID reuse entered redesign")

    discovery = [row for row in rows if row["split"] == MILD_RECALIBRATED_SPLITS[0]]
    confirmation = [row for row in rows if row["split"] == MILD_RECALIBRATED_SPLITS[1]]
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
            f"mild-recalibrated splits have exact or near overlap: {near_pairs[:3]}"
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


def _mild_run_diagnostic(
    row: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Return model-blind mild contact/exposure diagnostics for the manifest."""
    censor = int(arrays["censor_sample"])
    fall_value = int(arrays["first_fall_sample"])
    fall = None if fall_value < 0 else fall_value
    target = np.asarray(arrays["target_terrain_contact"][:censor], dtype=bool)
    loaded = np.asarray(arrays["loaded_contact"][:censor], dtype=bool)
    loaded_target = target & loaded
    loaded_any = np.any(loaded_target, axis=1)
    loaded_samples = np.flatnonzero(loaded_any)
    first_loaded = None if not loaded_samples.size else int(loaded_samples[0])
    last_loaded = None if not loaded_samples.size else int(loaded_samples[-1])
    target_summary = row["target_contact_summary"]
    first_target = target_summary["first_sample"]
    last_target = target_summary["last_sample_before_censor"]
    if fall is None:
        fall_relation = "NO_FALL"
    elif first_target is None:
        fall_relation = "PRE_TARGET"
    elif last_target is not None and fall <= int(last_target) + 1:
        fall_relation = "DURING_TARGET_CONTACT"
    else:
        fall_relation = "AFTER_LAST_TARGET_CONTACT"
    return {
        "patch_exit_x_m": round(
            float(row["patch_start_x_m"]) + float(row["patch_width_m"]), 3
        ),
        "first_loaded_target_contact_ms": first_loaded,
        "last_loaded_target_contact_ms": last_loaded,
        "cumulative_loaded_sand_exposure_ms": int(np.count_nonzero(loaded_any)),
        "loaded_foot_sand_exposure_ms": int(np.count_nonzero(loaded_target)),
        "target_contact_episode_count": int(target_summary.get("episode_count", 0)),
        "complete_target_contact_episode_count": int(
            target_summary.get("complete_episode_count", 0)
        ),
        "fall_relation": fall_relation,
        "target_to_fall_ms": (
            None if fall is None or first_target is None else fall - int(first_target)
        ),
    }


def _replace_split_keys(value: Mapping[str, Any]) -> dict[str, Any]:
    translated = {
        "REDESIGNED_DISCOVERY": MILD_RECALIBRATED_SPLITS[0],
        "REDESIGNED_CONFIRMATION": MILD_RECALIBRATED_SPLITS[1],
    }
    return {translated.get(str(key), str(key)): item for key, item in value.items()}


def audit_mild_recalibrated_manifest(
    manifest: Mapping[str, Any],
    design: Mapping[str, Any],
    matrix_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the frozen physical gates with the current split terminology."""
    to_historical = {
        MILD_RECALIBRATED_SPLITS[0]: "REDESIGNED_DISCOVERY",
        MILD_RECALIBRATED_SPLITS[1]: "REDESIGNED_CONFIRMATION",
    }
    translated_manifest = {
        **manifest,
        "runs": [
            {**row, "split": to_historical[str(row["split"])]}
            for row in manifest["runs"]
        ],
    }
    audit = audit_sand_benign_redesigned_manifest(
        translated_manifest, design, matrix_audit
    )
    split_sections = (
        "split_outcomes",
        "sand_source_speed",
        "mild",
        "moderate_boundary",
        "phase_diversity",
        "topology_contact",
        "support_controls",
    )
    for section in split_sections:
        audit[section] = _replace_split_keys(audit[section])
    audit["entry_timing"]["by_split"] = _replace_split_keys(
        audit["entry_timing"]["by_split"]
    )
    audit["generation_gates"] = {
        name.replace("REDESIGNED_DISCOVERY", MILD_RECALIBRATED_SPLITS[0]).replace(
            "REDESIGNED_CONFIRMATION", MILD_RECALIBRATED_SPLITS[1]
        ): value
        for name, value in audit["generation_gates"].items()
    }
    verdict_suffix = str(audit["generation_verdict"]).removeprefix(
        "SAND_BENIGN_GENERALIZATION_STUDY_REDESIGNED_GENERATION_"
    )
    audit["generation_verdict"] = (
        "SAND_BENIGN_GENERALIZATION_STUDY_MILD_RECALIBRATED_GENERATION_"
        f"{verdict_suffix}"
    )
    mild_rows = [row for row in manifest["runs"] if row["group"] == "broad_sand_benign"]
    exposure: dict[str, Any] = {}
    for split in MILD_RECALIBRATED_SPLITS:
        selected = [row for row in mild_rows if row["split"] == split]
        strict = [
            row
            for row in selected
            if row["objective_physical_outcome"] == "STRICT_BENIGN"
        ]
        exposure[split] = {
            "population": "BROAD_MILD",
            "strict_loaded_sand_exposure_ms": _numeric_values(
                [
                    int(
                        row["mild_physical_diagnostic"][
                            "cumulative_loaded_sand_exposure_ms"
                        ]
                    )
                    for row in strict
                ]
            ),
            "strict_contact_episode_count": _numeric_values(
                [
                    int(row["mild_physical_diagnostic"]["target_contact_episode_count"])
                    for row in strict
                ]
            ),
            "all_fall_relations": dict(
                sorted(
                    Counter(
                        str(row["mild_physical_diagnostic"]["fall_relation"])
                        for row in selected
                    ).items()
                )
            ),
            "interpretation": "DIVERSITY_DIAGNOSTIC_NOT_LABEL_THRESHOLD",
        }
    audit["mild_exposure_diagnostics"] = exposure
    return audit


def _numeric_values(values: Sequence[int | float]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def load_mild_recalibrated_manifest(dataset_path: Path) -> Mapping[str, Any]:
    """Load generated metadata without deserializing sealed waveforms."""
    manifest_path = dataset_path / "manifest.json"
    expected = (dataset_path / "manifest.sha256").read_text(encoding="utf-8").split()[0]
    if sha256_file(manifest_path) != expected:
        raise ValueError("mild-recalibrated Sand manifest integrity failed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != MILD_RECALIBRATED_DATASET_ID:
        raise ValueError("unexpected mild-recalibrated Sand dataset identity")
    return manifest


def load_mild_recalibrated_discovery_payload(
    dataset_path: Path, run_id: str
) -> dict[str, np.ndarray]:
    """Open Discovery only and reject Confirmation before NPZ access."""
    manifest = load_mild_recalibrated_manifest(dataset_path)
    row = next((item for item in manifest["runs"] if item["run_id"] == run_id), None)
    if row is None:
        raise KeyError(f"unknown mild-recalibrated Sand run: {run_id}")
    if row["split"] != MILD_RECALIBRATED_SPLITS[0]:
        raise RuntimeError("MILD_RECALIBRATED_CONFIRMATION is SEALED")
    path = dataset_path / str(row["file"])
    if sha256_file(path) != str(row["file_sha256"]):
        raise ValueError(f"mild-recalibrated Sand run integrity failed: {run_id}")
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def verify_mild_recalibrated_dataset(dataset_path: Path) -> dict[str, Any]:
    """Recompute every frozen file hash without opening waveform arrays."""
    manifest = load_mild_recalibrated_manifest(dataset_path)
    freeze_path = dataset_path / "dataset_freeze.json"
    expected_freeze_file = (
        (dataset_path / "dataset_freeze.sha256").read_text(encoding="utf-8").split()[0]
    )
    freeze_file_sha = sha256_file(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    semantic = dict(freeze)
    expected_semantic_sha = semantic.pop("MILD_RECALIBRATED_DATASET_FREEZE_SHA")
    npz_hashes = {
        str(row["file"]): sha256_file(dataset_path / str(row["file"]))
        for row in manifest["runs"]
    }
    expected_npz = {
        str(row["file"]): str(row["file_sha256"]) for row in manifest["runs"]
    }
    checks = {
        "dataset_freeze_file_sha": freeze_file_sha == expected_freeze_file,
        "dataset_freeze_semantic_sha": (
            canonical_sha256(semantic) == expected_semantic_sha
        ),
        "manifest_sha": (
            sha256_file(dataset_path / "manifest.json")
            == freeze["MILD_RECALIBRATED_STUDY_MANIFEST_SHA"]
        ),
        "physical_audit_sha": (
            sha256_file(dataset_path / "physical_audit.json")
            == freeze["MILD_RECALIBRATED_PHYSICAL_AUDIT_SHA"]
        ),
        "confirmation_seal_sha": (
            sha256_file(dataset_path / "confirmation_seal.json")
            == freeze["confirmation_seal_sha256"]
        ),
        "npz_hashes": npz_hashes == expected_npz,
        "npz_aggregate_sha": (
            canonical_sha256(npz_hashes)
            == freeze["MILD_RECALIBRATED_NPZ_AGGREGATE_SHA"]
        ),
        "run_count": len(manifest["runs"]) == 176,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "run_count": len(manifest["runs"]),
        "dataset_freeze_file_sha256": freeze_file_sha,
        "dataset_freeze_semantic_sha256": expected_semantic_sha,
    }


def collect_mild_recalibrated_study(
    root: Path,
    execution_config_path: Path,
    policy_override: Path | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Generate and freeze the exact 176-run model-blind current study."""
    execution = _load_yaml(execution_config_path)
    if execution["experiment"]["id"] != MILD_RECALIBRATED_GENERATION_ID:
        raise ValueError("unsupported mild-recalibrated Sand generation config")
    generation = execution["generation"]
    redesign_path = root / str(generation["redesign_config_path"])
    if sha256_file(redesign_path) != str(generation["redesign_config_sha256"]):
        raise RuntimeError("frozen mild-recalibrated redesign file changed")
    design = _load_yaml(redesign_path)
    matrix_audit = validate_mild_recalibrated_redesign(root, design)
    if matrix_audit["component_hashes"] != dict(generation["redesign_hashes"]):
        raise RuntimeError("frozen mild-recalibrated component hashes changed")
    if matrix_audit["redesign_sha256"] != str(generation["redesign_sha256"]):
        raise RuntimeError("complete mild-recalibrated redesign hash changed")
    if (
        matrix_audit["scenario_matrix_sha256"]
        != str(generation["expanded_scenario_matrix_sha256"])
        or matrix_audit["scenario_signature_sha256"]
        != str(generation["expected_scenario_signature_sha256"])
        or matrix_audit["split_sha256"] != dict(generation["expected_split_sha256"])
    ):
        raise RuntimeError("expanded mild-recalibrated matrix changed")
    specifications = expand_mild_recalibrated_redesign(design)
    expected_counts = {
        "total": len(specifications),
        "discovery": matrix_audit["split_counts"][MILD_RECALIBRATED_SPLITS[0]],
        "confirmation": matrix_audit["split_counts"][MILD_RECALIBRATED_SPLITS[1]],
        **matrix_audit["group_counts"],
    }
    declared_counts = {
        "total": int(generation["planned_total_runs"]),
        "discovery": int(generation["planned_discovery_runs"]),
        "confirmation": int(generation["planned_confirmation_runs"]),
        "broad_sand_benign": int(generation["planned_broad_mild_runs"]),
        "near_hazard_sand_benign": int(generation["planned_boundary_moderate_runs"]),
        "ordinary_support_control": int(generation["planned_ordinary_support_runs"]),
        "delayed_support_control": int(generation["planned_delayed_support_runs"]),
    }
    if (
        str(generation["dataset_id"]) != MILD_RECALIBRATED_DATASET_ID
        or expected_counts != declared_counts
    ):
        raise RuntimeError("mild-recalibrated execution identity/counts changed")
    configured_policy = root / str(generation["policy_path"])
    policy_path = configured_policy if policy_override is None else policy_override
    if sha256_file(policy_path) != str(generation["policy_sha256"]):
        raise RuntimeError("walking policy differs from generation freeze")
    simulator_path = root / str(generation["simulator_config_path"])
    if sha256_file(simulator_path) != str(generation["simulator_config_sha256"]):
        raise RuntimeError("simulator config differs from generation freeze")
    for category in ("implementation_artifacts", "protected_artifacts"):
        for artifact in generation[category]:
            path = root / str(artifact["path"])
            if sha256_file(path) != str(artifact["sha256"]):
                raise RuntimeError(f"frozen artifact changed: {artifact['path']}")
    guard_path = root / str(generation["consumed_holdout_guard_path"])
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    if guard.get("guard_after") != 1 or guard.get("scientific_open_count") != 1:
        raise RuntimeError("consumed Generalization HOLDOUT guard changed")
    required_true = (
        "no_model_inference",
        "no_training",
        "no_hnm",
        "no_normalizer_fit",
        "no_threshold_search",
        "no_persistence_search",
        "no_architecture_search",
        "old_holdout_access_forbidden",
        "confirmation_model_analysis_forbidden",
    )
    required_false = (
        "historical_dataset_reuse",
        "previous_sand_study_reuse",
        "calibration_pilot_reuse",
        "adaptive_backfill",
        "adaptive_replacement",
    )
    guards = execution["protocol_guards"]
    if not all(bool(guards[key]) for key in required_true) or any(
        bool(guards[key]) for key in required_false
    ):
        raise RuntimeError("mild-recalibrated execution protocol guard changed")

    output_path = root / str(generation["dataset_path"])
    partial_path = output_path.with_name(f"{output_path.name}_partial")
    if output_path.exists() or partial_path.exists():
        raise RuntimeError(f"mild-recalibrated output already exists: {output_path}")
    partial_path.mkdir(parents=True)
    config_sha = sha256_file(execution_config_path)
    pre_simulation_freeze = {
        "schema_version": 1,
        "status": "FROZEN_BEFORE_FIRST_SIMULATION",
        "dataset_id": MILD_RECALIBRATED_DATASET_ID,
        "source_commit": str(generation["source_commit"]),
        "execution_config_path": str(execution_config_path.relative_to(root)),
        "execution_config_sha256": config_sha,
        "redesign_config_sha256": str(generation["redesign_config_sha256"]),
        "redesign_sha256": matrix_audit["redesign_sha256"],
        "redesign_hashes": matrix_audit["component_hashes"],
        "expanded_scenario_matrix_sha256": matrix_audit["scenario_matrix_sha256"],
        "scenario_signature_sha256": matrix_audit["scenario_signature_sha256"],
        "split_sha256": matrix_audit["split_sha256"],
        "planned_run_count": len(specifications),
        "adaptive_backfill": False,
        "adaptive_replacement": False,
        "model_inference": False,
    }
    _write_json(partial_path / "pre_simulation_freeze.json", pre_simulation_freeze)
    rows: list[dict[str, Any]] = []
    attempted = 0
    started = time.monotonic()
    try:
        for index, specification in enumerate(specifications, start=1):
            attempted += 1
            result = run_simulation(
                SimulationConfig(
                    physics_timestep_s=float(
                        design["dataset_plan"]["physics_timestep_s"]
                    ),
                    sensor_rate_hz=int(design["dataset_plan"]["sensor_rate_hz"]),
                    duration_s=float(design["dataset_plan"]["simulation_duration_s"]),
                    command_speed_mps=float(specification["speed_mps"]),
                    policy_path=policy_path,
                    terrain=str(specification["target_terrain"]),
                    slip_pattern=str(specification["slip_pattern"]),
                    sink_pattern=str(specification["sink_pattern"]),
                    sink_severity=str(specification["sink_severity"]),
                    patch_start_x_m=float(specification["patch_start_x_m"]),
                    patch_width_m=float(specification["patch_width_m"]),
                    headless=True,
                    sink_support_pattern=str(specification["support_pattern"]),
                    source_terrain=str(specification["source_terrain"]),
                ),
                observe_fsr=True,
                observe_foot_imu=False,
            )
            row, arrays = annotate_model_v2_result(
                _annotation_specification(specification), result
            )
            row["scenario_family"] = specification["scenario_family"]
            if result.stability is None:
                raise RuntimeError("mild-recalibrated study requires exact gait phase")
            arrays["gait_phase"] = np.asarray(
                result.stability.gait_phase, dtype=np.int8
            )
            _calibration_result_summary(row, arrays, generation["label_execution"])
            if row["group"] == "broad_sand_benign":
                row["mild_physical_diagnostic"] = _mild_run_diagnostic(row, arrays)
            row["execution_status"] = "COMPLETED"
            filename = f"{specification['run_id']}.npz"
            run_path = partial_path / filename
            _write_deterministic_npz(run_path, arrays)
            row["file"] = filename
            row["file_sha256"] = sha256_file(run_path)
            row["size_bytes"] = run_path.stat().st_size
            rows.append(row)
            if progress is not None and (index == 1 or index % 5 == 0):
                progress(
                    f"generated {index}/{len(specifications)}: "
                    f"{specification['run_id']}"
                )

        manifest = {
            "schema_version": 1,
            "dataset_id": MILD_RECALIBRATED_DATASET_ID,
            "created_at": str(generation["generation_start"]),
            "generation_source_commit": str(generation["source_commit"]),
            "redesign_config_path": str(generation["redesign_config_path"]),
            "redesign_config_sha256": str(generation["redesign_config_sha256"]),
            "redesign_hashes": matrix_audit["component_hashes"],
            "redesign_sha256": matrix_audit["redesign_sha256"],
            "execution_config_path": str(execution_config_path.relative_to(root)),
            "execution_config_sha256": config_sha,
            "expanded_scenario_matrix_sha256": matrix_audit["scenario_matrix_sha256"],
            "scenario_signature_sha256": matrix_audit["scenario_signature_sha256"],
            "split_sha256": matrix_audit["split_sha256"],
            "matrix_audit": matrix_audit,
            "policy_sha256": str(generation["policy_sha256"]),
            "simulator_config_sha256": str(generation["simulator_config_sha256"]),
            "model_blind": True,
            "model_inference_runs": 0,
            "old_holdout_payload_reads": 0,
            "attempted_run_count": attempted,
            "run_count": len(rows),
            "valid_count": sum(bool(row["valid"]) for row in rows),
            "invalid_count": sum(not bool(row["valid"]) for row in rows),
            "split_counts": dict(Counter(str(row["split"]) for row in rows)),
            "adaptive_backfill_count": 0,
            "replacement_run_count": 0,
            "rerun_count": 0,
            "generation_order": [str(row["run_id"]) for row in rows],
            "model_output_fields": [],
            "runs": rows,
        }
        manifest_path = partial_path / "manifest.json"
        _write_json(manifest_path, manifest)
        manifest_sha = sha256_file(manifest_path)
        (partial_path / "manifest.sha256").write_text(
            f"{manifest_sha}  manifest.json\n", encoding="utf-8"
        )
        audit = audit_mild_recalibrated_manifest(manifest, design, matrix_audit)
        audit_path = partial_path / "physical_audit.json"
        _write_json(audit_path, audit)
        seal = {
            "schema_version": 1,
            "dataset_id": MILD_RECALIBRATED_DATASET_ID,
            "split": MILD_RECALIBRATED_SPLITS[1],
            "status": "SEALED_FOR_MILD_RECALIBRATED_CONFIRMATION",
            "generated": True,
            "objective_physical_audit_only": True,
            "model_inference": False,
            "normalized_80d_analysis": False,
            "observability_analysis": False,
            "visualization": False,
            "hypothesis_selection": False,
            "allowed_this_milestone": [
                "file_and_hash_integrity",
                "planned_signature_audit",
                "objective_physical_labels_and_generation_gates",
            ],
        }
        seal_path = partial_path / "confirmation_seal.json"
        _write_json(seal_path, seal)
        npz_hashes = {str(row["file"]): str(row["file_sha256"]) for row in rows}
        physical_outcomes = [
            {
                "run_id": row["run_id"],
                "valid": row["valid"],
                "outcome": row["objective_physical_outcome"],
                "actual_benign_severity": row["actual_benign_severity"],
                "invalid_reason": row["invalid_reason"],
            }
            for row in rows
        ]
        physical_signatures = [
            {"run_id": row["run_id"], **row["physical_signature"]} for row in rows
        ]
        freeze = {
            "schema_version": 1,
            "dataset_id": MILD_RECALIBRATED_DATASET_ID,
            "generation_source_commit": str(generation["source_commit"]),
            "redesign_config_sha256": str(generation["redesign_config_sha256"]),
            "generation_config_sha256": config_sha,
            "run_count": len(rows),
            "valid_count": manifest["valid_count"],
            "invalid_count": manifest["invalid_count"],
            "MILD_RECALIBRATED_STUDY_MANIFEST_SHA": manifest_sha,
            "MILD_RECALIBRATED_DISCOVERY_SPLIT_SHA": matrix_audit["split_sha256"][
                MILD_RECALIBRATED_SPLITS[0]
            ],
            "MILD_RECALIBRATED_CONFIRMATION_SPLIT_SHA": matrix_audit["split_sha256"][
                MILD_RECALIBRATED_SPLITS[1]
            ],
            "MILD_RECALIBRATED_SCENARIO_SIGNATURE_SHA": matrix_audit[
                "scenario_signature_sha256"
            ],
            "MILD_RECALIBRATED_PHYSICAL_SIGNATURE_SHA": canonical_sha256(
                physical_signatures
            ),
            "MILD_RECALIBRATED_NPZ_AGGREGATE_SHA": canonical_sha256(npz_hashes),
            "MILD_RECALIBRATED_PHYSICAL_OUTCOME_SHA": canonical_sha256(
                physical_outcomes
            ),
            "MILD_RECALIBRATED_GENERATION_GATE_RESULT_SHA": canonical_sha256(
                audit["generation_gates"]
            ),
            "MILD_RECALIBRATED_PHYSICAL_AUDIT_SHA": sha256_file(audit_path),
            "pre_simulation_freeze_sha256": sha256_file(
                partial_path / "pre_simulation_freeze.json"
            ),
            "confirmation_seal_sha256": sha256_file(seal_path),
            "confirmation_status": "SEALED_FOR_MILD_RECALIBRATED_CONFIRMATION",
            "generation_verdict": audit["generation_verdict"],
            "model_inference_runs": 0,
            "old_holdout_payload_reads": 0,
            "adaptive_backfill_count": 0,
            "replacement_run_count": 0,
            "rerun_count": 0,
        }
        freeze["MILD_RECALIBRATED_DATASET_FREEZE_SHA"] = canonical_sha256(freeze)
        freeze_path = partial_path / "dataset_freeze.json"
        _write_json(freeze_path, freeze)
        dataset_freeze_file_sha = sha256_file(freeze_path)
        (partial_path / "dataset_freeze.sha256").write_text(
            f"{dataset_freeze_file_sha}  dataset_freeze.json\n", encoding="utf-8"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.rename(output_path)
    except Exception:
        shutil.rmtree(partial_path, ignore_errors=True)
        raise

    summary = {
        "dataset_id": MILD_RECALIBRATED_DATASET_ID,
        "output_path": str(output_path),
        "planned_runs": len(specifications),
        "attempted_runs": attempted,
        "completed_runs": len(rows),
        "valid_runs": sum(bool(row["valid"]) for row in rows),
        "invalid_runs": sum(not bool(row["valid"]) for row in rows),
        "discovery_runs": matrix_audit["split_counts"][MILD_RECALIBRATED_SPLITS[0]],
        "confirmation_runs": matrix_audit["split_counts"][MILD_RECALIBRATED_SPLITS[1]],
        "adaptive_backfill_count": 0,
        "replacement_run_count": 0,
        "rerun_count": 0,
        "npz_bytes": sum(int(row["size_bytes"]) for row in rows),
        "file_count": len(list(output_path.iterdir())) + 1,
        "generation_seconds": round(time.monotonic() - started, 3),
        "manifest_sha256": manifest_sha,
        "dataset_freeze_file_sha256": dataset_freeze_file_sha,
        "dataset_freeze_semantic_sha256": freeze[
            "MILD_RECALIBRATED_DATASET_FREEZE_SHA"
        ],
        "generation_verdict": audit["generation_verdict"],
        "confirmation_status": "SEALED_FOR_MILD_RECALIBRATED_CONFIRMATION",
        "model_inference_runs": 0,
        "old_holdout_payload_reads": 0,
    }
    _write_json(output_path / "generation_summary.json", summary)
    return output_path, summary
