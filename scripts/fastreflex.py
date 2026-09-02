#!/usr/bin/env python3
"""Canonical command-line entry point for supported FastReflex workflows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) in sys.path:
    sys.path.remove(str(SOURCE_ROOT))
sys.path.insert(0, str(SOURCE_ROOT))

DEFAULT_SIMULATOR_CONFIG = REPOSITORY_ROOT / "configs/simulator/g1.yaml"
HAZARD_EXPERIMENT_ID = "UNIFIED_HAZARD_REFLEX_SYSTEM_VALIDATION"
TERRAIN_EXPERIMENT_ID = "TERRAIN_REBUILD_AND_SENSOR_ABLATION"
MODEL_V2_GENERATION_ID = "MODEL_V2_DATASET_GENERATION"
MODEL_V2_TRAINING_ID = "MODEL_V2_DATA_ONLY_TRAINING"
MODEL_V2_REBALANCED_TRAINING_ID = "MODEL_V2_EXTRACTION_REBALANCED_TRAINING"
MODEL_V2_ANCHOR_REFINED_TRAINING_ID = "MODEL_V2_ANCHOR_REFINED_TRAINING"
MODEL_V2_GENERALIZATION_DEVELOPMENT_EVALUATION_ID = (
    "MODEL_V2_GENERALIZATION_DEVELOPMENT_EVALUATION"
)
MODEL_V2_FINAL_CANDIDATE_HOLDOUT_READINESS_REVIEW_ID = (
    "MODEL_V2_FINAL_CANDIDATE_FREEZE_AND_HOLDOUT_READINESS_REVIEW"
)
MODEL_V2_GENERALIZATION_HOLDOUT_ONE_SHOT_EVALUATION_ID = (
    "MODEL_V2_GENERALIZATION_HOLDOUT_ONE_SHOT_EVALUATION"
)
SUPPORTED_EXPERIMENT_IDS = frozenset(
    (
        HAZARD_EXPERIMENT_ID,
        TERRAIN_EXPERIMENT_ID,
        MODEL_V2_GENERATION_ID,
        MODEL_V2_TRAINING_ID,
        MODEL_V2_REBALANCED_TRAINING_ID,
        MODEL_V2_ANCHOR_REFINED_TRAINING_ID,
        MODEL_V2_GENERALIZATION_DEVELOPMENT_EVALUATION_ID,
        MODEL_V2_FINAL_CANDIDATE_HOLDOUT_READINESS_REVIEW_ID,
        MODEL_V2_GENERALIZATION_HOLDOUT_ONE_SHOT_EVALUATION_ID,
    )
)
HISTORICAL_MESSAGE = (
    "This experiment is historical and is not runnable from the current "
    "consolidated source tree. Use the source commit recorded in its "
    "report/config provenance."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FastReflex supported research pipeline CLI."
    )
    subparsers = parser.add_subparsers(dest="command")

    simulate = subparsers.add_parser(
        "simulate", help="run the canonical Unitree G1 MuJoCo simulation"
    )
    simulate.add_argument("--config", type=Path, default=DEFAULT_SIMULATOR_CONFIG)
    simulate.add_argument("--terrain", choices=("concrete", "marble", "ice", "sand"))
    simulate.add_argument("--source-terrain", choices=("concrete", "marble"))
    simulate.add_argument("--speed", type=float)
    simulate.add_argument("--duration", type=float)
    simulate.add_argument("--patch-start-x", type=float)
    simulate.add_argument("--patch-width", type=float)
    simulate.add_argument("--slip-pattern", choices=("uniform", "transition"))
    simulate.add_argument(
        "--sink-pattern",
        choices=(
            "uniform",
            "asymmetric_left",
            "asymmetric_right",
            "transition_left",
            "transition_right",
        ),
    )
    simulate.add_argument("--sink-severity", choices=("mild", "moderate", "severe"))
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
            "staged_lateral_deformable",
        ),
    )
    simulate.add_argument(
        "--policy",
        type=Path,
        help="verified Unitree G1 ONNX policy; or set FASTREFLEX_G1_POLICY",
    )
    mode = simulate.add_mutually_exclusive_group()
    mode.add_argument("--headless", action="store_true")
    mode.add_argument("--viewer", action="store_true")

    collect = subparsers.add_parser(
        "collect", help="collect a supported dataset from an explicit config"
    )
    collect.add_argument("--config", type=Path, required=True)
    collect.add_argument("--policy", type=Path)

    train = subparsers.add_parser(
        "train", help="train from an explicit supported experiment config"
    )
    train.add_argument("--config", type=Path, required=True)
    train.add_argument(
        "--dry-run",
        action="store_true",
        help="freeze and report extraction without fitting or optimizer steps",
    )

    evaluate = subparsers.add_parser(
        "evaluate", help="verify a frozen supported candidate without HOLDOUT access"
    )
    evaluate.add_argument("--config", type=Path, required=True)

    visualize = subparsers.add_parser(
        "visualize",
        help="re-simulate a TRAIN/VALIDATION run with frozen decision overlays",
    )
    selection = visualize.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-id")
    selection.add_argument("--list-runs", action="store_true")
    visualize.add_argument(
        "--speed",
        type=float,
        choices=(0.5, 1.0, 2.0),
        default=1.0,
        help="viewer playback speed (default: 1.0)",
    )
    visualize.add_argument(
        "--pause-at",
        type=float,
        metavar="SECONDS",
        help="pause once at the requested simulation time",
    )
    visualize.add_argument(
        "--pause-on-reflex",
        action="store_true",
        help="pause at the first frozen REFLEX_REQUIRED onset",
    )
    visualize.add_argument(
        "--single-step",
        action="store_true",
        help="start paused for keyboard frame stepping",
    )
    visualize.add_argument(
        "--mode",
        choices=("demo", "analysis"),
        default="analysis",
        help="compact demo or detailed analysis overlay (default: analysis)",
    )
    visualize.add_argument("--show-debug", action="store_true")
    visualize.add_argument(
        "--policy",
        type=Path,
        help="verified policy override; canonical frozen path is used by default",
    )

    subparsers.add_parser(
        "export", help="reserved for the later reviewed Research-to-Deployment release"
    )
    return parser


def _experiment_id(path: Path) -> str:
    with path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    try:
        return str(document["experiment"]["id"])
    except (KeyError, TypeError) as exc:
        raise ValueError("config does not declare experiment.id") from exc


def _require_supported(path: Path) -> str:
    experiment_id = _experiment_id(path)
    if experiment_id not in SUPPORTED_EXPERIMENT_IDS:
        raise ValueError(HISTORICAL_MESSAGE)
    return experiment_id


def _policy_path(argument: Path | None) -> Path | None:
    environment = os.environ.get("FASTREFLEX_G1_POLICY")
    return (
        argument
        if argument is not None
        else (Path(environment) if environment else None)
    )


def _simulate(args: argparse.Namespace) -> int:
    from fastreflex.simulation.g1 import (
        load_simulation_config,
        run_simulation,
        summarize_result,
    )

    config = load_simulation_config(args.config)
    policy = _policy_path(args.policy)
    headless = False if args.viewer else (True if args.headless else config.headless)
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
        policy_path=config.policy_path if policy is None else policy,
        slip_pattern=(
            config.slip_pattern if args.slip_pattern is None else args.slip_pattern
        ),
        sink_pattern=(
            config.sink_pattern if args.sink_pattern is None else args.sink_pattern
        ),
        sink_severity=(
            config.sink_severity if args.sink_severity is None else args.sink_severity
        ),
        sink_support_pattern=(
            config.sink_support_pattern
            if args.sink_support_pattern is None
            else args.sink_support_pattern
        ),
        patch_start_x_m=(
            config.patch_start_x_m if args.patch_start_x is None else args.patch_start_x
        ),
        patch_width_m=(
            config.patch_width_m if args.patch_width is None else args.patch_width
        ),
        headless=headless,
    )
    print(
        json.dumps(summarize_result(run_simulation(config)), indent=2, sort_keys=True)
    )
    return 0


def _collect(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    experiment_id = _require_supported(args.config)
    policy = _policy_path(args.policy)
    if policy is None:
        parser.error("collect requires --policy or FASTREFLEX_G1_POLICY")
    if experiment_id == HAZARD_EXPERIMENT_ID:
        parser.error(
            "The supported Unified Hazard corpus is frozen; this consolidated "
            "milestone does not regenerate it."
        )
    if experiment_id == MODEL_V2_GENERATION_ID:
        from fastreflex.dataset.generation import collect_model_v2_dataset

        output_path, summary = collect_model_v2_dataset(
            REPOSITORY_ROOT,
            args.config.resolve(),
            policy.resolve(),
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        print(
            json.dumps(
                {"output_path": str(output_path), **summary},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    from fastreflex.dataset.terrain import collect_terrain_dataset

    output_path, summary = collect_terrain_dataset(args.config, policy)
    print(
        json.dumps(
            {"output_path": str(output_path), **summary}, indent=2, sort_keys=True
        )
    )
    return 0


def _train(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    experiment_id = _require_supported(args.config)
    if experiment_id == MODEL_V2_ANCHOR_REFINED_TRAINING_ID:
        from fastreflex.training.hazard import run_model_v2_anchor_refined_training

        result = run_model_v2_anchor_refined_training(
            REPOSITORY_ROOT,
            args.config.resolve(),
            dry_run=bool(args.dry_run),
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if experiment_id == MODEL_V2_REBALANCED_TRAINING_ID:
        from fastreflex.training.hazard import (
            run_model_v2_extraction_rebalanced_training,
        )

        result = run_model_v2_extraction_rebalanced_training(
            REPOSITORY_ROOT,
            args.config.resolve(),
            dry_run=bool(args.dry_run),
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if experiment_id == MODEL_V2_TRAINING_ID:
        from fastreflex.training.hazard import run_model_v2_data_only_training

        result = run_model_v2_data_only_training(
            REPOSITORY_ROOT,
            args.config.resolve(),
            dry_run=bool(args.dry_run),
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    name = "Unified Hazard" if experiment_id == HAZARD_EXPERIMENT_ID else "Terrain"
    parser.error(
        f"{name} is a frozen supported candidate. Training requires a separately "
        "reviewed experiment/output identity and is intentionally not implicit."
    )


def _evaluate(args: argparse.Namespace) -> int:
    experiment_id = _require_supported(args.config)
    if experiment_id == HAZARD_EXPERIMENT_ID:
        from fastreflex.dataset.hazard import load_yaml
        from fastreflex.evaluation.hazard import verify_supported_candidate

        result = verify_supported_candidate(REPOSITORY_ROOT, load_yaml(args.config))
    elif experiment_id == MODEL_V2_GENERALIZATION_DEVELOPMENT_EVALUATION_ID:
        from fastreflex.evaluation.generalization import (
            run_generalization_development_evaluation,
        )

        result = run_generalization_development_evaluation(
            REPOSITORY_ROOT, args.config.resolve()
        )
    elif experiment_id == MODEL_V2_FINAL_CANDIDATE_HOLDOUT_READINESS_REVIEW_ID:
        from fastreflex.evaluation.readiness import (
            run_final_candidate_holdout_readiness_review,
        )

        result = run_final_candidate_holdout_readiness_review(
            REPOSITORY_ROOT, args.config.resolve()
        )
    elif experiment_id == MODEL_V2_GENERALIZATION_HOLDOUT_ONE_SHOT_EVALUATION_ID:
        from fastreflex.evaluation.holdout import (
            run_generalization_holdout_one_shot_evaluation,
        )

        result = run_generalization_holdout_one_shot_evaluation(
            REPOSITORY_ROOT, args.config.resolve()
        )
    elif experiment_id == MODEL_V2_ANCHOR_REFINED_TRAINING_ID:
        from fastreflex.evaluation.hazard import (
            verify_model_v2_anchor_refined_training_result,
        )

        result = verify_model_v2_anchor_refined_training_result(
            REPOSITORY_ROOT, args.config.resolve()
        )
    elif experiment_id == MODEL_V2_REBALANCED_TRAINING_ID:
        from fastreflex.evaluation.hazard import (
            verify_model_v2_extraction_rebalanced_training_result,
        )

        result = verify_model_v2_extraction_rebalanced_training_result(
            REPOSITORY_ROOT, args.config.resolve()
        )
    elif experiment_id == MODEL_V2_TRAINING_ID:
        from fastreflex.evaluation.hazard import verify_model_v2_training_result

        result = verify_model_v2_training_result(
            REPOSITORY_ROOT, args.config.resolve()
        )
    else:
        from fastreflex.evaluation.terrain import verify_supported_terrain_candidate

        result = verify_supported_terrain_candidate(REPOSITORY_ROOT)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _visualize(args: argparse.Namespace) -> int:
    from fastreflex.visualization import (
        prepare_visualization,
        representative_validation_runs,
        visualization_run_ids,
        visualize_prepared_run,
    )

    if args.list_runs:
        print(
            json.dumps(
                {
                    "representative_validation": representative_validation_runs(
                        REPOSITORY_ROOT
                    ),
                    "runs": visualization_run_ids(REPOSITORY_ROOT),
                    "holdout_included": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    policy = _policy_path(args.policy)
    prepared = prepare_visualization(REPOSITORY_ROOT, args.run_id, policy)
    first_reflex_s = (
        None
        if prepared.traces.first_reflex_sample is None
        else float(
            prepared.resolved.run.timestamp_us[prepared.traces.first_reflex_sample]
            / 1_000_000.0
        )
    )
    print(
        json.dumps(
            {
                "run_id": prepared.resolved.run.run_id,
                "split": prepared.resolved.run.split,
                "parity": prepared.parity.checks,
                "sensor_absolute_tolerance": (
                    prepared.parity.sensor_absolute_tolerance
                ),
                "status": "PARITY_PASSED_OPENING_VIEWER",
                "pause_at_s": args.pause_at,
                "pause_on_reflex": args.pause_on_reflex,
                "first_reflex_s": first_reflex_s,
                "single_step": args.single_step,
                "mode": args.mode,
                "controls": {
                    "pause_play": "Space",
                    "step_1ms": "Left/Right Arrow (period also steps forward)",
                    "step_10ms": "A/D",
                    "first_last": "Home/End",
                    "event_jumps": "R Reflex, H Hazard, I I1, T/G Terrain",
                },
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    result = visualize_prepared_run(
        prepared,
        playback_speed=args.speed,
        show_debug=args.show_debug,
        pause_at_s=args.pause_at,
        pause_on_reflex=args.pause_on_reflex,
        single_step=args.single_step,
        mode=args.mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "simulate":
            return _simulate(args)
        if args.command == "collect":
            return _collect(args, parser)
        if args.command == "train":
            return _train(args, parser)
        if args.command == "evaluate":
            return _evaluate(args)
        if args.command == "visualize":
            return _visualize(args)
        if args.command == "export":
            parser.error(
                "export is reserved for the later reviewed E84 deployment milestone"
            )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    raise AssertionError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())
