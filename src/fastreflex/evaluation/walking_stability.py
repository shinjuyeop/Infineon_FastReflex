"""Calibrate and freshly validate the causal simulator-only stability clock."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from datetime import date, datetime
import hashlib
import inspect
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import yaml

from fastreflex.evaluation.transition_scenarios import (
    VALID_FALL,
    VALID_STABLE,
    classify_scenario_outcome,
    fusion_regression,
    scenario_timing_row,
    target_contact_mask,
    transition_simulation_config,
)
from fastreflex.simulation.g1 import (
    PHYSICS_TIMESTEP_S,
    SENSOR_RATE_HZ,
    RuntimeTrace,
    SimulationConfig,
    SimulationResult,
    load_simulation_config,
    run_simulation,
)
from fastreflex.simulation.stability import (
    PHASE_NAMES,
    InstabilityTrace,
    PhaseEnvelope,
    StableCalibrationRun,
    detect_instability,
    fit_phase_envelope,
)


SIGNATURE_FIELDS = (
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
class OracleRun:
    """One observed scenario and its ungated/transition-gated oracle traces."""

    specification: Mapping[str, object]
    result: SimulationResult
    outcome: str
    first_contact_sample: int
    ungated: InstabilityTrace
    primary: InstabilityTrace

    @property
    def run_id(self) -> str:
        return str(self.specification["id"])


def _json_default(value: object) -> object:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=_json_default)
        stream.write("\n")


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=_json_default
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protected_hashes(repository_root: Path, paths: Sequence[str]) -> dict[str, str]:
    hashes = {}
    for relative in paths:
        path = (repository_root / relative).resolve()
        path.relative_to(repository_root)
        if not path.is_file():
            raise FileNotFoundError(f"protected Terrain path is missing: {path}")
        hashes[relative] = _file_sha256(path)
    return hashes


def _first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(np.asarray(values, dtype=bool))
    return None if not indices.size else int(indices[0])


def _time_ms(result: SimulationResult, sample: int | None) -> float | None:
    if sample is None or not 0 <= sample < len(result.runtime.timestamp_us):
        return None
    return float(result.runtime.timestamp_us[sample]) / 1000.0


def _condition_signature(specification: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(specification[field] for field in SIGNATURE_FIELDS)


def validate_experiment_design(document: Mapping[str, object]) -> dict[str, object]:
    """Fail before simulation if the fixed cohort or fresh matrix is invalid."""
    oracle = document["stability_oracle"]
    expected_fixed = {
        "phase_lower_quantile": 0.005,
        "fixed_margin_m": 0.010,
        "persistence_ms": 20,
    }
    for field, expected in expected_fixed.items():
        if not np.isclose(float(oracle[field]), expected, atol=0.0, rtol=0.0):
            raise ValueError(f"stability oracle {field} is not the frozen value")
    if oracle["runtime_inputs"] or bool(oracle["future_fall_dependency"]):
        raise ValueError("privileged oracle cannot use runtime input or future fall")

    hard = list(document["calibration"]["hard_stable_runs"])
    calibration = list(document["calibration"]["transition_runs"])
    validation = list(document["fresh_validation"]["runs"])
    all_runs = [*hard, *calibration, *validation]
    run_ids = [str(item["id"]) for item in all_runs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("walking-stability run IDs must be unique")
    if not 32 <= len(validation) <= 48:
        raise ValueError("fresh oracle validation must contain 32-48 runs")

    calibration_signatures = {_condition_signature(item) for item in calibration}
    validation_signatures = {_condition_signature(item) for item in validation}
    if calibration_signatures & validation_signatures:
        raise ValueError("calibration and fresh physical conditions overlap")
    if len(validation_signatures) != len(validation):
        raise ValueError("fresh physical-condition signatures must be unique")

    domains = document["frozen_operating_domains"]
    group_counts: dict[str, int] = {}
    for item in validation:
        group = str(item["frozen_group"])
        domain = domains[group]
        group_key = f"{item['source_terrain']}_{group}"
        group_counts[group_key] = group_counts.get(group_key, 0) + 1
        for field in (
            "speed_mps",
            "patch_start_x_m",
            "slip_pattern",
            "sink_pattern",
            "sink_severity",
            "support_pattern",
        ):
            allowed = domain[field]
            allowed_values = allowed if isinstance(allowed, list) else [allowed]
            if item[field] not in allowed_values:
                raise ValueError(
                    f"fresh run {item['id']} escapes frozen {group} {field}"
                )
        minimum, maximum = domain["patch_width_m"]
        if not float(minimum) <= float(item["patch_width_m"]) <= float(maximum):
            raise ValueError(f"fresh run {item['id']} escapes frozen width")
    required_groups = {
        f"{source}_{group}"
        for source in ("concrete", "marble")
        for group in ("ice_stable", "ice_fall", "sand_stable", "sand_fall")
    }
    if set(group_counts) != required_groups or any(
        group_counts[group] < 4 for group in required_groups
    ):
        raise ValueError("fresh matrix must contain at least four runs per group")
    return {
        "passed": True,
        "calibration_runs": len(hard) + len(calibration),
        "fresh_runs": len(validation),
        "fresh_group_counts": group_counts,
        "calibration_validation_disjoint": True,
        "fresh_signatures_unique": True,
    }


def _hard_outcome(result: SimulationResult) -> str:
    finite = bool(
        np.all(np.isfinite(result.runtime.pelvis_imu))
        and result.stability is not None
        and np.all(np.isfinite(result.stability.com_xyz_m))
        and np.all(np.isfinite(result.stability.com_velocity_xyz_m_s))
        and result.metadata["actual_samples"] == result.metadata["expected_samples"]
        and not result.metadata["terminated_by_viewer"]
    )
    if not finite:
        return "INVALID_OTHER"
    return VALID_STABLE if result.metadata["first_fall_sample"] is None else VALID_FALL


def _prepared_specification(
    raw: Mapping[str, object], common: Mapping[str, object]
) -> dict[str, object]:
    specification = dict(raw)
    specification["minimum_normal_prefix_ms"] = int(common["minimum_normal_prefix_ms"])
    specification["minimum_post_contact_ms"] = int(common["minimum_post_contact_ms"])
    return specification


def _simulate_cohort(
    base: SimulationConfig,
    specifications: Sequence[Mapping[str, object]],
    policy_path: Path,
    common: Mapping[str, object],
    progress: Callable[[str], None],
    label: str,
) -> dict[str, tuple[dict[str, object], SimulationResult, str]]:
    simulations = {}
    duration_s = float(common["duration_s"])
    for index, raw in enumerate(specifications, start=1):
        specification = _prepared_specification(raw, common)
        result = run_simulation(
            transition_simulation_config(base, specification, policy_path, duration_s)
        )
        target = str(specification["target_terrain"])
        outcome = (
            _hard_outcome(result)
            if target in {"concrete", "marble"}
            else classify_scenario_outcome(result, specification)
        )
        simulations[str(specification["id"])] = (
            specification,
            result,
            outcome,
        )
        progress(
            f"{label} {index}/{len(specifications)} {specification['id']}: {outcome}"
        )
    return simulations


def _first_contact(result: SimulationResult, target: str) -> int:
    if target in {"concrete", "marble"}:
        return 0
    contact = _first_true(np.any(target_contact_mask(result, target), axis=1))
    if contact is None:
        raise ValueError("valid transition has no target contact")
    return contact


def _apply_oracle(
    simulations: Mapping[str, tuple[Mapping[str, object], SimulationResult, str]],
    envelope: PhaseEnvelope,
    fixed_margin_m: float,
    persistence_samples: int,
) -> dict[str, OracleRun]:
    runs = {}
    for run_id, (specification, result, outcome) in simulations.items():
        if outcome not in {VALID_STABLE, VALID_FALL}:
            continue
        if result.stability is None:
            raise RuntimeError("simulation did not capture exact stability state")
        contact = _first_contact(result, str(specification["target_terrain"]))
        ungated = detect_instability(
            result.stability, envelope, fixed_margin_m, persistence_samples
        )
        primary = detect_instability(
            result.stability,
            envelope,
            fixed_margin_m,
            persistence_samples,
            eligible_from_sample=contact,
        )
        runs[run_id] = OracleRun(
            specification=specification,
            result=result,
            outcome=outcome,
            first_contact_sample=contact,
            ungated=ungated,
            primary=primary,
        )
    return runs


def _finite_min(values: np.ndarray) -> float | None:
    finite = np.asarray(values)[np.isfinite(values)]
    return None if not finite.size else float(np.min(finite))


def _finite_max(values: np.ndarray) -> float | None:
    finite = np.asarray(values)[np.isfinite(values)]
    return None if not finite.size else float(np.max(finite))


def valid_prefall_instability(onset: int | None, fall: int | None) -> bool:
    """Count only a causal instability confirmation strictly before fall."""
    return bool(onset is not None and fall is not None and onset < fall)


def _run_row(run: OracleRun) -> dict[str, object]:
    result = run.result
    target = str(run.specification["target_terrain"])
    fall_raw = result.metadata["first_fall_sample"]
    fall = None if fall_raw is None else int(fall_raw)
    onset = _first_true(run.primary.onset)
    ungated_onset = _first_true(run.ungated.onset)
    pretransition = bool(
        ungated_onset is not None and ungated_onset < run.first_contact_sample
    )
    stop = (
        len(result.runtime.sequence)
        if fall is None
        else max(run.first_contact_sample + 1, fall)
    )
    episode = slice(run.first_contact_sample, stop)
    slip = _first_true(result.diagnostics.any_established_slip_after_patch_onset)
    sink = _first_true(np.any(result.diagnostics.deformable_sink_onset, axis=1))
    xcom = result.stability.xcom_xy_m
    xcom_delta = np.diff(xcom, axis=0) * SENSOR_RATE_HZ
    xcom_speed = np.linalg.norm(xcom_delta, axis=1)
    com_speed = np.linalg.norm(result.stability.com_velocity_xyz_m_s, axis=1)
    angular_speed = np.linalg.norm(
        result.diagnostics.pelvis_angular_velocity_rad_s, axis=1
    )
    valid_detection = valid_prefall_instability(onset, fall)
    late_detection = bool(fall is not None and onset is not None and onset >= fall)
    status = (
        "VALID_PREFALL"
        if valid_detection
        else "LATE_POSTFALL"
        if late_detection
        else "NO_EVENT"
    )
    return {
        "run_id": run.run_id,
        "source_terrain": str(run.specification["source_terrain"]),
        "target_terrain": target,
        "transition": f"{run.specification['source_terrain']}->{target}",
        "speed_mps": float(run.specification["speed_mps"]),
        "design_role": run.specification.get("design_role"),
        "observed_outcome": "stable" if run.outcome == VALID_STABLE else "fall",
        "first_target_contact_ms": _time_ms(result, run.first_contact_sample),
        "physical_slip_onset_ms": _time_ms(result, slip),
        "physical_sink_onset_ms": _time_ms(result, sink),
        "max_support_deformation_m": float(
            np.max(result.diagnostics.support_surface_max_displacement_m)
        ),
        "minimum_raw_mos_m": _finite_min(
            result.stability.raw_margin_of_stability_m[episode]
        ),
        "minimum_stability_residual_m": _finite_min(run.primary.residual_m[episode]),
        "t_instability_ms": _time_ms(result, onset),
        "t_fall_ms": _time_ms(result, fall),
        "fall_lead_ms": (
            None
            if not valid_detection
            else float(
                result.runtime.timestamp_us[fall] - result.runtime.timestamp_us[onset]
            )
            / 1000.0
        ),
        "false_instability": bool(run.outcome == VALID_STABLE and onset is not None),
        "valid_prefall_detection": valid_detection,
        "detection_status": status,
        "pretransition_ungated_onset_ms": _time_ms(result, ungated_onset)
        if pretransition
        else None,
        "pretransition_false_instability": pretransition,
        "support_phase_at_instability": (
            None
            if onset is None
            else PHASE_NAMES[int(result.stability.gait_phase[onset])]
        ),
        "raw_mos_at_instability_m": (
            None
            if onset is None
            else float(result.stability.raw_margin_of_stability_m[onset])
        ),
        "residual_at_instability_m": (
            None if onset is None else float(run.primary.residual_m[onset])
        ),
        "no_support_samples": int(np.count_nonzero(result.stability.gait_phase == 0)),
        "diagnostics": {
            "max_abs_pelvis_roll_deg": float(
                np.degrees(np.max(np.abs(result.diagnostics.pelvis_roll_rad)))
            ),
            "max_abs_pelvis_pitch_deg": float(
                np.degrees(np.max(np.abs(result.diagnostics.pelvis_pitch_rad)))
            ),
            "peak_pelvis_angular_speed_rad_s": _finite_max(angular_speed),
            "minimum_pelvis_height_m": _finite_min(result.diagnostics.pelvis_world_z_m),
            "peak_com_speed_m_s": _finite_max(com_speed),
            "peak_xcom_speed_m_s": _finite_max(xcom_speed),
        },
    }


def _outcome_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for row in rows:
        key = (
            f"{row['source_terrain']}_{row['target_terrain']}_"
            f"{row['observed_outcome']}"
        )
        counts[key] = counts.get(key, 0) + 1
    return {"total": len(rows), "groups": counts}


def _coverage_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    stable = [row for row in rows if row["observed_outcome"] == "stable"]
    falling = [row for row in rows if row["observed_outcome"] == "fall"]
    stable_fp = [row for row in stable if row["false_instability"]]
    detected = [row for row in falling if row["valid_prefall_detection"]]

    def grouped(field: str, values: Sequence[str]) -> dict[str, object]:
        result = {}
        for value in values:
            selected_stable = [row for row in stable if row[field] == value]
            selected_fall = [row for row in falling if row[field] == value]
            fp = [row for row in selected_stable if row["false_instability"]]
            true = [row for row in selected_fall if row["valid_prefall_detection"]]
            result[value] = {
                "stable_runs": len(selected_stable),
                "stable_false_instability_runs": len(fp),
                "stable_false_instability_run_rate": (
                    len(fp) / len(selected_stable) if selected_stable else None
                ),
                "fall_runs": len(selected_fall),
                "detected_fall_runs": len(true),
                "fall_coverage": len(true) / len(selected_fall)
                if selected_fall
                else 0.0,
            }
        return result

    leads = [float(row["fall_lead_ms"]) for row in detected]
    pretransition = [
        str(row["run_id"]) for row in rows if row["pretransition_false_instability"]
    ]
    return {
        "stable_runs": len(stable),
        "stable_false_instability_runs": [str(row["run_id"]) for row in stable_fp],
        "stable_false_instability_run_rate": len(stable_fp) / len(stable)
        if stable
        else 1.0,
        "fall_runs": len(falling),
        "detected_fall_runs": [str(row["run_id"]) for row in detected],
        "fall_coverage": len(detected) / len(falling) if falling else 0.0,
        "by_terrain": grouped("target_terrain", ("ice", "sand")),
        "by_source": grouped("source_terrain", ("concrete", "marble")),
        "pretransition_false_instability_runs": pretransition,
        "pretransition_false_instability_run_count": len(pretransition),
        "fall_lead_ms": {
            "minimum": float(np.min(leads)) if leads else None,
            "p10": float(np.percentile(leads, 10)) if leads else None,
            "p50": float(np.percentile(leads, 50)) if leads else None,
            "p95": float(np.percentile(leads, 95)) if leads else None,
            "maximum": float(np.max(leads)) if leads else None,
        },
    }


def _paired_analysis(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    paired = {}
    for terrain in ("ice", "sand"):
        paired[terrain] = {}
        for outcome in ("stable", "fall"):
            selected = [
                row
                for row in rows
                if row["target_terrain"] == terrain
                and row["observed_outcome"] == outcome
            ]
            raw = [
                float(row["minimum_raw_mos_m"])
                for row in selected
                if row["minimum_raw_mos_m"] is not None
            ]
            residual = [
                float(row["minimum_stability_residual_m"])
                for row in selected
                if row["minimum_stability_residual_m"] is not None
            ]
            paired[terrain][outcome] = {
                "runs": len(selected),
                "minimum_raw_mos_m_median": float(np.median(raw)) if raw else None,
                "minimum_residual_m_median": float(np.median(residual))
                if residual
                else None,
                "slip_or_sink_diagnostic_runs": sum(
                    row[
                        "physical_slip_onset_ms"
                        if terrain == "ice"
                        else "physical_sink_onset_ms"
                    ]
                    is not None
                    for row in selected
                ),
                "instability_runs": sum(
                    row["t_instability_ms"] is not None for row in selected
                ),
            }
    return paired


def future_suffix_independence(
    run: OracleRun,
    envelope: PhaseEnvelope,
    fixed_margin_m: float,
    persistence_samples: int,
) -> dict[str, object]:
    """Change only the future residual suffix and compare the fixed prefix."""
    onset = _first_true(run.primary.onset)
    boundary = (
        onset
        if onset is not None
        else max(run.first_contact_sample, len(run.primary.onset) // 2)
    )
    changed_margin = run.result.stability.raw_margin_of_stability_m.copy()
    changed_margin[boundary + 1 :] = 1.0
    changed_diagnostics = replace(
        run.result.stability, raw_margin_of_stability_m=changed_margin
    )
    changed = detect_instability(
        changed_diagnostics,
        envelope,
        fixed_margin_m,
        persistence_samples,
        eligible_from_sample=run.first_contact_sample,
    )
    end = boundary + 1
    passed = bool(
        np.array_equal(run.primary.candidate[:end], changed.candidate[:end])
        and np.array_equal(run.primary.active[:end], changed.active[:end])
        and np.array_equal(run.primary.onset[:end], changed.onset[:end])
        and np.array_equal(
            run.primary.residual_m[:end], changed.residual_m[:end], equal_nan=True
        )
    )
    return {
        "passed": passed,
        "run_id": run.run_id,
        "comparison_through_sample": boundary,
        "comparison_through_ms": _time_ms(run.result, boundary),
        "future_margin_suffix_replaced": True,
        "future_fall_and_postfall_are_not_oracle_inputs": True,
    }


def _terrain_regression(
    document: Mapping[str, object],
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> dict[str, object]:
    runtime_fields = {field.name for field in fields(RuntimeTrace)}
    forbidden = {
        "com",
        "xcom",
        "support_polygon",
        "raw_mos",
        "stability_residual",
        "fall",
    }
    oracle_parameters = set(inspect.signature(detect_instability).parameters)
    independent = "terrain" not in oracle_parameters and not (
        runtime_fields & forbidden
    )
    untouched = dict(before) == dict(after)
    return {
        "passed": untouched and independent,
        "protected_sha256_before": dict(before),
        "protected_sha256_after": dict(after),
        "dataset_model_report_untouched": untouched,
        "terrain_retraining_performed": False,
        "candidate_unchanged": document["terrain_regression"]["candidate"],
        "producer_independence": independent,
    }


def _status_block(row: Mapping[str, object]) -> str:
    has_onset = row["t_instability_ms"] is not None
    support_phase = (
        row["support_phase_at_instability"] if has_onset else "NO_PRIMARY_ONSET"
    )
    raw_mos = row["raw_mos_at_instability_m"] if has_onset else row["minimum_raw_mos_m"]
    residual = (
        row["residual_at_instability_m"]
        if has_onset
        else row["minimum_stability_residual_m"]
    )
    return "\n".join(
        (
            f"RUN {row['run_id']}",
            f"TERRAIN={str(row['target_terrain']).upper()} OUTCOME={str(row['observed_outcome']).upper()}",
            f"CONTACT_MS={row['first_target_contact_ms']}",
            f"SAMPLE_KIND={'PRIMARY_ONSET' if has_onset else 'EPISODE_MINIMUM_SUMMARY'}",
            f"SUPPORT_PHASE={support_phase}",
            f"RAW_MOS_M={raw_mos}",
            f"STABILITY_RESIDUAL_M={residual}",
            f"T_INSTABILITY_MS={row['t_instability_ms']}",
            f"T_FALL_MS={row['t_fall_ms']}",
        )
    )


def _viewer_replay(
    rows: Sequence[Mapping[str, object]]
) -> tuple[dict[str, object], str]:
    selected = []
    for terrain in ("ice", "sand"):
        for outcome in ("stable", "fall"):
            candidates = [
                row
                for row in rows
                if row["target_terrain"] == terrain
                and row["observed_outcome"] == outcome
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda row: (
                    row["source_terrain"] != "concrete",
                    str(row["run_id"]),
                )
            )
            selected.append(candidates[0])
    blocks = [_status_block(row) for row in selected]
    parity = len(selected) == 4 and all(
        f"T_INSTABILITY_MS={row['t_instability_ms']}" in block
        and f"T_FALL_MS={row['t_fall_ms']}" in block
        for row, block in zip(selected, blocks)
    )
    return (
        {
            "representative_run_ids": [str(row["run_id"]) for row in selected],
            "physics_mutation": False,
            "status_matches_evaluation": parity,
            "passed": parity,
        },
        "\n\n".join(blocks) + "\n",
    )


def _acceptance(
    summary: Mapping[str, object],
    causality: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[dict[str, bool], str]:
    by_terrain = summary["by_terrain"]
    by_source = summary["by_source"]
    lead = summary["fall_lead_ms"]["p50"]
    gates = {
        "stable_specificity": float(summary["stable_false_instability_run_rate"])
        <= float(config["stable_false_instability_run_rate_max"]),
        "fall_coverage": float(summary["fall_coverage"])
        >= float(config["fall_prefall_instability_coverage_min"]),
        "ice_coverage": float(by_terrain["ice"]["fall_coverage"])
        >= float(config["ice_fall_coverage_min"]),
        "sand_coverage": float(by_terrain["sand"]["fall_coverage"])
        >= float(config["sand_fall_coverage_min"]),
        "source_robustness": all(
            int(by_source[source]["detected_fall_runs"]) > 0
            for source in config["source_meaningful_detection_required"]
        ),
        "lead_time": lead is not None
        and float(lead) >= float(config["median_fall_lead_ms_min"]),
        "transition_cleanliness": int(
            summary["pretransition_false_instability_run_count"]
        )
        <= int(config["pretransition_false_instability_runs_max"]),
        "causality": bool(causality["passed"]),
    }
    verdicts = config["verdicts"]
    if all(gates.values()):
        verdict = str(verdicts["pass"])
    else:
        failures = [name for name, passed in gates.items() if not passed]
        mild = bool(
            len(failures) == 1
            and gates["causality"]
            and gates["transition_cleanliness"]
            and float(summary["stable_false_instability_run_rate"]) <= 0.15
            and float(summary["fall_coverage"]) >= 0.75
            and float(by_terrain["ice"]["fall_coverage"]) >= 0.70
            and float(by_terrain["sand"]["fall_coverage"]) >= 0.70
            and lead is not None
            and float(lead) >= 150.0
        )
        verdict = str(verdicts["promising"] if mild else verdicts["fail"])
    return gates, verdict


def run_walking_stability_ground_truth_sanity(
    config_path: Path,
    repository_root: Path,
    progress: Callable[[str], None] = print,
) -> tuple[Path, dict[str, object]]:
    """Fit on observed-stable calibration runs, freeze, then validate once."""
    repository_root = repository_root.resolve()
    config_path = config_path.resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if document["experiment"]["id"] != "WALKING_STABILITY_GROUND_TRUTH_SANITY":
        raise ValueError("unsupported walking stability experiment")
    design = validate_experiment_design(document)
    if PHYSICS_TIMESTEP_S != float(document["common"]["physics_timestep_s"]):
        raise ValueError("experiment physics timestep differs from canonical value")
    if SENSOR_RATE_HZ != int(document["common"]["sensor_rate_hz"]):
        raise ValueError("experiment sensor rate differs from canonical value")

    artifact_path = (repository_root / document["artifacts"]["path"]).resolve()
    artifact_path.relative_to(repository_root)
    if artifact_path.exists() and any(artifact_path.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite experiment artifacts: {artifact_path}"
        )
    artifact_path.mkdir(parents=True, exist_ok=True)
    base = load_simulation_config(
        (repository_root / document["source"]["simulator_config"]).resolve()
    )
    policy_path = (repository_root / document["source"]["policy_path"]).resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"verified G1 policy is unavailable: {policy_path}")
    if _file_sha256(policy_path) != str(document["source"]["policy_sha256"]):
        raise ValueError("G1 policy SHA-256 differs from frozen experiment")
    protected_paths = [
        str(path) for path in document["terrain_regression"]["protected_paths"]
    ]
    terrain_hashes_before = _protected_hashes(repository_root, protected_paths)

    calibration_specs = [
        *document["calibration"]["hard_stable_runs"],
        *document["calibration"]["transition_runs"],
    ]
    calibration_simulations = _simulate_cohort(
        base,
        calibration_specs,
        policy_path,
        document["common"],
        progress,
        "ORACLE CALIBRATION",
    )
    calibration_outcomes = {
        run_id: outcome for run_id, (_, _, outcome) in calibration_simulations.items()
    }
    prior_matches = all(
        calibration_outcomes[str(item["id"])]
        == (VALID_STABLE if item["prior_observed_outcome"] == "stable" else VALID_FALL)
        for item in document["calibration"]["transition_runs"]
    )
    hard_ids = {str(item["id"]) for item in document["calibration"]["hard_stable_runs"]}
    hard_stable = all(
        calibration_outcomes[run_id] == VALID_STABLE for run_id in hard_ids
    )
    valid_calibration = all(
        outcome in {VALID_STABLE, VALID_FALL}
        for outcome in calibration_outcomes.values()
    )

    stable_calibration: list[StableCalibrationRun] = []
    for run_id, (specification, result, outcome) in calibration_simulations.items():
        if outcome != VALID_STABLE:
            continue
        if result.stability is None:
            raise RuntimeError("calibration did not capture exact stability state")
        stable_calibration.append(
            StableCalibrationRun(
                run_id=run_id,
                diagnostics=result.stability,
                observed_stable=True,
                observed_fall=False,
                source_terrain=str(specification["source_terrain"]),
                target_terrain=str(specification["target_terrain"]),
            )
        )
    composition = {
        f"{source}_{target}": sum(
            run.source_terrain == source and run.target_terrain == target
            for run in stable_calibration
        )
        for source, target in (
            ("concrete", "concrete"),
            ("marble", "marble"),
            ("concrete", "ice"),
            ("marble", "ice"),
            ("concrete", "sand"),
            ("marble", "sand"),
        )
    }
    calibration_gate = bool(
        prior_matches
        and hard_stable
        and valid_calibration
        and all(value > 0 for value in composition.values())
    )
    if not calibration_gate:
        metrics = {
            "experiment": document["experiment"],
            "design": design,
            "oracle_calibration": {
                "performed": True,
                "passed": False,
                "prior_outcomes_reproduced": prior_matches,
                "hard_controls_stable": hard_stable,
                "all_scenarios_valid": valid_calibration,
                "stable_composition": composition,
            },
            "fresh_oracle_validation": {"performed": False},
            "verdict": document["acceptance"]["verdicts"]["fail"],
        }
        _write_json(artifact_path / "results.json", metrics)
        return artifact_path, metrics

    oracle = document["stability_oracle"]
    envelope = fit_phase_envelope(
        stable_calibration, float(oracle["phase_lower_quantile"])
    )
    persistence_samples = int(
        round(int(oracle["persistence_ms"]) * SENSOR_RATE_HZ / 1000)
    )
    fixed_margin_m = float(oracle["fixed_margin_m"])
    calibration_runs = _apply_oracle(
        calibration_simulations,
        envelope,
        fixed_margin_m,
        persistence_samples,
    )
    calibration_rows = [
        _run_row(calibration_runs[run_id]) for run_id in calibration_runs
    ]
    calibration_contract = {
        "quantile": envelope.quantile,
        "quantile_method": oracle["quantile_method"],
        "phase_lower_bound_m": {
            PHASE_NAMES[int(phase)]: value
            for phase, value in envelope.lower_bound_m.items()
        },
        "fixed_margin_m": fixed_margin_m,
        "persistence_ms": int(oracle["persistence_ms"]),
        "persistence_samples": persistence_samples,
        "transition_gate": oracle["transition_gate"],
        "terrain_agnostic": True,
        "stable_calibration_run_ids": list(envelope.calibration_run_ids),
    }
    frozen_contract_hash = _canonical_hash(calibration_contract)
    _write_json(
        artifact_path / "calibration.json",
        {
            "label": document["calibration"]["label"],
            "contract": calibration_contract,
            "sha256": frozen_contract_hash,
            "stable_composition": composition,
            "fall_runs_excluded_from_fit": all(
                run_id not in envelope.calibration_run_ids
                for run_id, (_, _, outcome) in calibration_simulations.items()
                if outcome == VALID_FALL
            ),
        },
    )
    progress(
        "ORACLE CALIBRATION frozen: "
        + ", ".join(
            f"{PHASE_NAMES[int(phase)]}={bound:.6f} m"
            for phase, bound in envelope.lower_bound_m.items()
        )
    )

    validation_simulations = _simulate_cohort(
        base,
        document["fresh_validation"]["runs"],
        policy_path,
        document["common"],
        progress,
        "FRESH ORACLE VALIDATION",
    )
    valid_validation = {
        run_id: value
        for run_id, value in validation_simulations.items()
        if value[2] in {VALID_STABLE, VALID_FALL}
    }
    invalid_rows = [
        scenario_timing_row(result, specification)
        for specification, result, outcome in validation_simulations.values()
        if outcome not in {VALID_STABLE, VALID_FALL}
    ]
    validation_runs = _apply_oracle(
        valid_validation,
        envelope,
        fixed_margin_m,
        persistence_samples,
    )
    validation_rows = [_run_row(validation_runs[run_id]) for run_id in validation_runs]
    summary = _coverage_summary(validation_rows)
    causality_source = next(
        (
            run
            for run in validation_runs.values()
            if run.outcome == VALID_FALL and _first_true(run.primary.onset) is not None
        ),
        next(iter(validation_runs.values())),
    )
    causality = future_suffix_independence(
        causality_source, envelope, fixed_margin_m, persistence_samples
    )
    contract_hash_after_validation = _canonical_hash(calibration_contract)
    contract_immutable = contract_hash_after_validation == frozen_contract_hash
    viewer, status_text = _viewer_replay(validation_rows)
    with (artifact_path / "viewer_status.txt").open("w", encoding="utf-8") as stream:
        stream.write(status_text)
    progress(status_text)

    terrain_hashes_after = _protected_hashes(repository_root, protected_paths)
    terrain = _terrain_regression(document, terrain_hashes_before, terrain_hashes_after)
    fusion = fusion_regression()
    gates, verdict = _acceptance(summary, causality, document["acceptance"])
    if (
        not contract_immutable
        or not terrain["passed"]
        or not fusion["passed"]
        or not viewer["passed"]
    ):
        verdict = str(document["acceptance"]["verdicts"]["fail"])
    metrics = {
        "experiment": document["experiment"],
        "design": design,
        "oracle_calibration": {
            "label": document["calibration"]["label"],
            "performed": True,
            "passed": calibration_gate,
            "runs": calibration_rows,
            "outcome_counts": _outcome_counts(calibration_rows),
            "stable_runs_used": len(stable_calibration),
            "stable_composition": composition,
            "phase_envelope": calibration_contract,
            "fall_runs_excluded_from_normal_fit": True,
            "paired_analysis": _paired_analysis(calibration_rows),
        },
        "fresh_oracle_validation": {
            "label": document["fresh_validation"]["label"],
            "performed": True,
            "scenario_gate": {
                "configured_runs": len(validation_simulations),
                "valid_runs": len(valid_validation),
                "invalid_runs": len(invalid_rows),
                "invalid": invalid_rows,
                "pretransition_fall_runs": [
                    row["run_id"] for row in invalid_rows if row["pretransition_fall"]
                ],
            },
            "outcome_counts": _outcome_counts(validation_rows),
            "stable_table": [
                row for row in validation_rows if row["observed_outcome"] == "stable"
            ],
            "fall_table": [
                row for row in validation_rows if row["observed_outcome"] == "fall"
            ],
            "summary": summary,
            "acceptance_gates": gates,
            "contract_sha256_before_validation": frozen_contract_hash,
            "contract_sha256_after_validation": contract_hash_after_validation,
            "contract_immutable": contract_immutable,
            "threshold_retuning_performed": False,
        },
        "causality": causality,
        "terrain_regression": terrain,
        "fusion_regression": fusion,
        "viewer": viewer,
        "privileged_only_contract": {
            "runtime_inputs": oracle["runtime_inputs"],
            "forbidden_runtime_inputs": oracle["forbidden_runtime_inputs"],
            "pelvis_orientation_is_diagnostic_only": True,
            "future_fall_is_validation_outcome_only": True,
        },
        "verdict": verdict,
    }
    _write_json(artifact_path / "results.json", metrics)
    return artifact_path, metrics
