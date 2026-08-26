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

PLACEHOLDER_COMMANDS = ("collect", "train", "evaluate", "export")
DEFAULT_SIMULATOR_CONFIG = REPOSITORY_ROOT / "configs" / "simulator" / "g1.yaml"


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
            headless=headless,
        )
        result = run_simulation(config)
        print(json.dumps(summarize_result(result), indent=2, sort_keys=True))
        return 0

    print(
        f"'{args.command}' is not implemented: "
        "this milestone only provides the MuJoCo simulation baseline."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
