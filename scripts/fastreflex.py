#!/usr/bin/env python3
"""Canonical command-line entry point for FastReflex research workflows."""

from __future__ import annotations

import argparse


COMMANDS = ("collect", "train", "evaluate", "export")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FastReflex research CLI (project scaffold only)."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=COMMANDS,
        help="Workflow command reserved for future implementation.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command is None:
        build_parser().print_help()
        return 0

    print(
        f"'{args.command}' is not implemented: "
        "the repository status is PROJECT_SCAFFOLD_ONLY."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
