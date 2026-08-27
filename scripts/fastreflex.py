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

PLACEHOLDER_COMMANDS = ("evaluate", "export")
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
        "--policy",
        type=Path,
        help=(
            "user-supplied verified Unitree G1 ONNX policy; alternatively set "
            "FASTREFLEX_G1_POLICY"
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
        print(json.dumps(summarize_result(result), indent=2, sort_keys=True))
        return 0

    if args.command == "collect":
        from fastreflex.dataset.collector import collect_dataset

        environment_policy = os.environ.get("FASTREFLEX_G1_POLICY")
        policy_path = args.policy
        if policy_path is None and environment_policy:
            policy_path = Path(environment_policy)
        if policy_path is None:
            parser.error(
                "collect requires --policy or the FASTREFLEX_G1_POLICY environment variable"
            )
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

    print(
        f"'{args.command}' is not implemented: "
        "this workflow has not been implemented yet."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
