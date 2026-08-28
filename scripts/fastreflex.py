#!/usr/bin/env python3
"""Canonical command-line entry point for FastReflex research workflows."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

PLACEHOLDER_COMMANDS = ("export",)
DEFAULT_SIMULATOR_CONFIG = REPOSITORY_ROOT / "configs" / "simulator" / "g1.yaml"
DEFAULT_COLLECTION_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "experiment"
    / "20260827_hazard_pilot_dataset.yaml"
)
DEFAULT_TRAINING_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "experiment"
    / "20260827_first_classification_poc.yaml"
)
DEFAULT_EVALUATION_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "experiment"
    / "20260827_time_to_separation.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FastReflex research CLI."
    )
    subparsers = parser.add_subparsers(dest="command")
    simulate = subparsers.add_parser(
        "simulate", help="run an in-memory Unitree G1 MuJoCo smoke simulation"
    )
    simulate.add_argument("--config", type=Path, default=DEFAULT_SIMULATOR_CONFIG)
    simulate.add_argument(
        "--terrain", choices=("concrete", "marble", "ice", "sand")
    )
    simulate.add_argument(
        "--source-terrain",
        choices=("concrete", "marble"),
        help="select the hard A-side terrain for a finite Ice/Sand transition",
    )
    simulate.add_argument("--speed", type=float)
    simulate.add_argument("--duration", type=float)
    simulate.add_argument("--patch-start-x", type=float)
    simulate.add_argument("--patch-width", type=float)
    simulate.add_argument(
        "--slip-pattern",
        choices=("uniform", "transition"),
        help="select uniform terrain or a finite full-width low-friction patch",
    )
    simulate.add_argument(
        "--sink-pattern",
        choices=(
            "uniform",
            "asymmetric_left",
            "asymmetric_right",
            "transition_left",
            "transition_right",
        ),
        help="select uniform ground, a full lane, or a finite compliance patch",
    )
    simulate.add_argument(
        "--sink-severity",
        choices=("mild", "moderate", "severe"),
        help="select the synthetic compliance of the asymmetric lane",
    )
    simulate.add_argument(
        "--sink-support-pattern",
        choices=(
            "balanced_soft",
            "medial_soft",
            "lateral_soft",
            "localized_soft",
            "balanced_deformable",
            "medial_deformable",
            "lateral_deformable",
            "localized_deformable",
        ),
        help="select balanced or spatially heterogeneous finite Sink support",
    )
    simulate.add_argument(
        "--policy",
        type=Path,
        help=(
            "user-supplied verified Unitree G1 ONNX policy; alternatively set "
            "FASTREFLEX_G1_POLICY"
        ),
    )
    simulate.add_argument(
        "--status-calibration",
        type=Path,
        help=(
            "render a timestamp-synchronized terrain/stability status replay "
            "using a calibration.json produced by the integrated sanity"
        ),
    )
    mode = simulate.add_mutually_exclusive_group()
    mode.add_argument(
        "--headless",
        action="store_true",
        help="run at maximum speed without opening a window (default)",
    )
    mode.add_argument(
        "--viewer",
        action="store_true",
        help="open the official MuJoCo viewer and pace simulation near real time",
    )
    collect = subparsers.add_parser(
        "collect", help="materialize and validate a raw Hazard pilot dataset"
    )
    collect.add_argument("--config", type=Path, default=DEFAULT_COLLECTION_CONFIG)
    collect.add_argument(
        "--policy",
        type=Path,
        help=(
            "user-supplied verified Unitree G1 ONNX policy; alternatively set "
            "FASTREFLEX_G1_POLICY"
        ),
    )
    train = subparsers.add_parser(
        "train", help="run the bounded first pelvis IMU classification PoC"
    )
    train.add_argument("--config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    evaluate = subparsers.add_parser(
        "evaluate", help="replay a frozen classifier around physical hazard events"
    )
    evaluate.add_argument("--config", type=Path, default=DEFAULT_EVALUATION_CONFIG)
    for command in PLACEHOLDER_COMMANDS:
        subparsers.add_parser(command, help="reserved for a later milestone")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "simulate":
        from fastreflex.simulation.g1 import (
            load_simulation_config,
            run_simulation,
            summarize_result,
        )

        config = load_simulation_config(args.config)
        environment_policy = os.environ.get("FASTREFLEX_G1_POLICY")
        policy_path = args.policy
        if policy_path is None and environment_policy:
            policy_path = Path(environment_policy)
        headless = config.headless
        if args.viewer:
            headless = False
        elif args.headless:
            headless = True
        config = replace(
            config,
            terrain=config.terrain if args.terrain is None else args.terrain,
            source_terrain=(
                config.source_terrain
                if args.source_terrain is None
                else args.source_terrain
            ),
            command_speed_mps=(
                config.command_speed_mps if args.speed is None else args.speed
            ),
            duration_s=config.duration_s if args.duration is None else args.duration,
            policy_path=config.policy_path if policy_path is None else policy_path,
            slip_pattern=(
                config.slip_pattern
                if args.slip_pattern is None
                else args.slip_pattern
            ),
            sink_pattern=(
                config.sink_pattern
                if args.sink_pattern is None
                else args.sink_pattern
            ),
            sink_severity=(
                config.sink_severity
                if args.sink_severity is None
                else args.sink_severity
            ),
            sink_support_pattern=(
                config.sink_support_pattern
                if args.sink_support_pattern is None
                else args.sink_support_pattern
            ),
            patch_start_x_m=(
                config.patch_start_x_m
                if args.patch_start_x is None
                else args.patch_start_x
            ),
            patch_width_m=(
                config.patch_width_m
                if args.patch_width is None
                else args.patch_width
            ),
            headless=headless,
        )
        result = run_simulation(config)
        if args.status_calibration is not None:
            from fastreflex.evaluation.integrated_stability import (
                render_simulation_status,
            )

            print(
                render_simulation_status(result, args.status_calibration),
                file=sys.stderr,
            )
        print(json.dumps(summarize_result(result), indent=2, sort_keys=True))
        return 0

    if args.command == "collect":
        import yaml

        environment_policy = os.environ.get("FASTREFLEX_G1_POLICY")
        policy_path = args.policy
        if policy_path is None and environment_policy:
            policy_path = Path(environment_policy)
        if policy_path is None:
            parser.error(
                "collect requires --policy or the FASTREFLEX_G1_POLICY environment variable"
            )
        with args.config.open("r", encoding="utf-8") as stream:
            experiment_id = yaml.safe_load(stream)["experiment"]["id"]
        if experiment_id == "TERRAIN_REBUILD_AND_SENSOR_ABLATION":
            from fastreflex.dataset.terrain import collect_terrain_dataset

            output_path, summary = collect_terrain_dataset(args.config, policy_path)
        else:
            from fastreflex.dataset.collector import collect_dataset

            output_path, summary = collect_dataset(args.config, policy_path)
        print(
            json.dumps(
                {"output_path": str(output_path), **summary},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "train":
        import yaml

        with args.config.open("r", encoding="utf-8") as stream:
            experiment_id = yaml.safe_load(stream)["experiment"]["id"]
        if experiment_id in {
            "FSR_OBSERVABILITY_PILOT",
            "SINK_SENSOR_OBSERVABILITY_STUDY",
        }:
            from fastreflex.training.sensor_ablation import (
                run_fsr_observability_pilot,
                run_sink_sensor_observability_study,
            )

            runner = (
                run_fsr_observability_pilot
                if experiment_id == "FSR_OBSERVABILITY_PILOT"
                else run_sink_sensor_observability_study
            )
            output_path, metrics = runner(args.config, REPOSITORY_ROOT)
            if experiment_id == "SINK_SENSOR_OBSERVABILITY_STUDY":
                print(
                    json.dumps(
                        {
                            "output_path": str(output_path),
                            "selected": metrics["selection"]["candidate_id"],
                            "verdict": metrics["verdict"],
                            "holdout_macro_f1": metrics["holdout"]["metrics"][
                                "macro_f1"
                            ],
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            print(
                json.dumps(
                    {
                        "output_path": str(output_path),
                        "profiles": list(metrics["classification"]),
                        "window_ms": 100,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if experiment_id == "TERRAIN_REBUILD_AND_SENSOR_ABLATION":
            from fastreflex.training.terrain import run_terrain_sensor_ablation

            output_path, metrics = run_terrain_sensor_ablation(
                args.config, REPOSITORY_ROOT
            )
            print(
                json.dumps(
                    {
                        "output_path": str(output_path),
                        "selected_profile": metrics["selection"]["sensor_profile"],
                        "selected_family": metrics["selection"]["model_family"],
                        "selected_horizon_ms": metrics["selection"]["observation_horizon_ms"],
                        "selected_scheme": metrics["selection"]["deployment_scheme"],
                        "holdout_macro_f1": metrics["holdout"]["metrics"]["macro_f1"],
                        "verdict": metrics["verdict"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        from fastreflex.training.trainer import run_first_classification_poc

        output_path, metrics = run_first_classification_poc(
            args.config, REPOSITORY_ROOT
        )
        print(
            json.dumps(
                {
                    "output_path": str(output_path),
                    "selected_candidate": metrics["selection"]["candidate_id"],
                    "holdout_macro_f1": metrics["holdout"]["metrics"]["macro_f1"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "evaluate":
        import yaml

        with args.config.open("r", encoding="utf-8") as stream:
            experiment_id = yaml.safe_load(stream)["experiment"]["id"]
        if experiment_id == "CONTINUOUS_SLIP_REFLEX_DETECTOR_DEVELOPMENT":
            from fastreflex.evaluation.continuous_slip_reflex import (
                run_continuous_slip_reflex_detector,
            )

            output_path, metrics = run_continuous_slip_reflex_detector(
                args.config, REPOSITORY_ROOT
            )
            print(
                json.dumps(
                    {
                        "output_path": str(output_path),
                        "phase_a_selection": metrics["phase_a"]["selection"],
                        "phase_b_activated": metrics["phase_b"]["activated"],
                        "holdout_performed": metrics["holdout"]["performed"],
                        "verdict": metrics["verdict"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if experiment_id == "TERRAIN_CONDITIONED_REFLEX_DETECTOR_DEVELOPMENT":
            from fastreflex.evaluation.terrain_conditioned_reflex import (
                run_terrain_conditioned_reflex_detector,
            )

            output_path, metrics = run_terrain_conditioned_reflex_detector(
                args.config, REPOSITORY_ROOT
            )
            print(
                json.dumps(
                    {
                        "output_path": str(output_path),
                        "terrain_timing": metrics["terrain_timing"],
                        "final_selection": metrics["final_selection"],
                        "holdout_performed": metrics["holdout"]["performed"],
                        "verdict": metrics["verdict"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if experiment_id == "EVENT_CENTRIC_REFLEX_TRIGGER_DEVELOPMENT":
            from fastreflex.evaluation.reflex_event import (
                run_event_centric_reflex_trigger,
            )

            output_path, metrics = run_event_centric_reflex_trigger(
                args.config, REPOSITORY_ROOT
            )
            print(
                json.dumps(
                    {
                        "output_path": str(output_path),
                        "dataset": metrics["dataset"],
                        "readiness": metrics["readiness"],
                        "selection": metrics.get("selection"),
                        "verdict": metrics["verdict"],
                        "architecture_recommendation": metrics[
                            "architecture_recommendation"
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if experiment_id == "DENSE_FALL_RISK_DATASET_AND_DETECTOR_POC":
            from fastreflex.evaluation.stability_dense import (
                run_dense_fall_risk_detector_poc,
            )

            output_path, metrics = run_dense_fall_risk_detector_poc(
                args.config, REPOSITORY_ROOT
            )
            print(
                json.dumps(
                    {
                        "output_path": str(output_path),
                        "dataset": metrics["dataset"],
                        "readiness": metrics["readiness"],
                        "holdout_performed": metrics["holdout"]["performed"],
                        "verdict": metrics["verdict"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if experiment_id == "TEMPORAL_STABILITY_SEPARABILITY_AUDIT":
            from fastreflex.evaluation.stability_temporal import (
                run_temporal_stability_separability_audit,
            )

            output_path, metrics = run_temporal_stability_separability_audit(
                args.config, REPOSITORY_ROOT
            )
            print(
                json.dumps(
                    {
                        "output_path": str(output_path),
                        "cohort": metrics["cohort"],
                        "holdout_performed": metrics["holdout"]["performed"],
                        "verdict": metrics["verdict"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if experiment_id == "FULL_STATE_STABILITY_GROUND_TRUTH_SANITY":
            from fastreflex.evaluation.stability_ground_truth import (
                run_full_state_stability_ground_truth_sanity,
            )

            output_path, metrics = run_full_state_stability_ground_truth_sanity(
                args.config, REPOSITORY_ROOT
            )
            print(
                json.dumps(
                    {
                        "output_path": str(output_path),
                        "selected_candidate": metrics.get("selected_candidate"),
                        "fresh_validation_performed": metrics[
                            "fresh_validation"
                        ]["performed"],
                        "verdict": metrics["verdict"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if experiment_id == "WALKING_STABILITY_GROUND_TRUTH_SANITY":
            from fastreflex.evaluation.walking_stability import (
                run_walking_stability_ground_truth_sanity,
            )

            output_path, metrics = run_walking_stability_ground_truth_sanity(
                args.config, REPOSITORY_ROOT
            )
            validation = metrics["fresh_oracle_validation"]
            print(
                json.dumps(
                    {
                        "output_path": str(output_path),
                        "calibration_passed": metrics["oracle_calibration"][
                            "passed"
                        ],
                        "fresh_validation_performed": validation["performed"],
                        "acceptance_gates": validation.get(
                            "acceptance_gates"
                        ),
                        "verdict": metrics["verdict"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if experiment_id == "TRANSITION_SCENARIO_CALIBRATION":
            from fastreflex.evaluation.transition_scenarios import (
                run_transition_scenario_calibration,
            )

            output_path, metrics = run_transition_scenario_calibration(
                args.config, REPOSITORY_ROOT
            )
            print(
                json.dumps(
                    {
                        "output_path": str(output_path),
                        "prefix_parity": metrics["prefix_parity"]["verdict"],
                        "fresh_concrete_validation": metrics[
                            "fresh_validation"
                        ]["performed"],
                        "marble_robustness": metrics["marble_robustness"][
                            "performed"
                        ],
                        "verdict": metrics["verdict"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if experiment_id == "TERRAIN_STABILITY_INTEGRATED_SANITY":
            from fastreflex.evaluation.integrated_stability import (
                run_integrated_stability_sanity,
            )

            output_path, metrics = run_integrated_stability_sanity(
                args.config, REPOSITORY_ROOT
            )
            print(
                json.dumps(
                    {
                        "output_path": str(output_path),
                        "terrain_runtime_status": metrics["terrain_runtime"][
                            "status"
                        ],
                        "scenario_gate": metrics["scenario_gate"]["passed"],
                        "stability_ground_truth_gate": metrics["oracle_gate"][
                            "passed"
                        ],
                        "ai_performed": metrics["ai"]["performed"],
                        "verdict": metrics["verdict"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if experiment_id == "FSR_TEMPORAL_REDISTRIBUTION_ANALYSIS":
            from fastreflex.evaluation.fsr_temporal import (
                run_fsr_temporal_redistribution_analysis,
            )

            output_path, summary = run_fsr_temporal_redistribution_analysis(
                args.config, REPOSITORY_ROOT
            )
            print(
                json.dumps(
                    {
                        "output_path": str(output_path),
                        "dataset_id": summary["dataset_id"],
                        "analysis_only": summary["analysis_only"],
                        "simulation_executed": summary["simulation_executed"],
                        "training_executed": summary["training_executed"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if experiment_id == "FSR_LOAD_DISTRIBUTION_ANALYSIS":
            from fastreflex.evaluation.fsr_distribution import (
                run_fsr_load_distribution_analysis,
            )

            output_path, summary = run_fsr_load_distribution_analysis(
                args.config, REPOSITORY_ROOT
            )
            print(
                json.dumps(
                    {
                        "output_path": str(output_path),
                        "dataset_id": summary["dataset_id"],
                        "analysis_only": summary["analysis_only"],
                        "simulation_executed": summary["simulation_executed"],
                        "training_executed": summary["training_executed"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        from fastreflex.evaluation.time_to_separation import run_time_to_separation

        output_path, metrics = run_time_to_separation(
            args.config, REPOSITORY_ROOT
        )
        print(
            json.dumps(
                {
                    "output_path": str(output_path),
                    "model": metrics["replay"]["model"],
                    "seeds": metrics["replay"]["seeds"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(
        f"'{args.command}' is not implemented: "
        "this workflow has not been implemented yet."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
