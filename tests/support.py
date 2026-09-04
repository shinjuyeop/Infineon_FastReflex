"""Small test doubles and repository paths shared by contract tests."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    """Load a UTF-8 JSON fixture."""

    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> Any:
    """Load a UTF-8 YAML fixture."""

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def assert_dataclass_equal(left: object, right: object) -> None:
    """Compare every field, including NumPy arrays, without repeated loops."""

    assert type(left) is type(right)
    for field in fields(left):
        np.testing.assert_equal(getattr(left, field.name), getattr(right, field.name))


def assert_contains(text: str, *fragments: str) -> None:
    """Report the first missing expected fragment clearly."""

    for fragment in fragments:
        assert fragment in text


def assert_false_fields(mapping: Mapping[str, Any], *names: str) -> None:
    """Check a group of explicit fail-closed counters or flags."""

    for name in names:
        assert not mapping[name], name


def assert_mapping_values(
    mapping: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    """Compare an explicit subset of a result mapping."""

    for name, value in expected.items():
        assert mapping[name] == value, name


def assert_attributes(value: object, expected: dict[str, Any]) -> None:
    """Compare a compact set of public contract attributes."""

    for name, expected_value in expected.items():
        assert getattr(value, name) == expected_value, name


def imported_modules(*paths: Path) -> set[str]:
    """Return direct imports from source files used in dependency-boundary tests."""

    modules: set[str] = set()
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules.update(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )
        modules.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
    return modules


class ViewerStub:
    """Minimal passive-viewer double used by simulation and playback tests."""

    def __init__(self, running_checks: int | None = 3) -> None:
        self.sync_count = 0
        self.texts: list[object] = []
        self.running_checks = running_checks
        self.is_running_count = 0

    def __enter__(self) -> "ViewerStub":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def is_running(self) -> bool:
        self.is_running_count += 1
        return (
            self.running_checks is None or self.is_running_count <= self.running_checks
        )

    def sync(self, state_only: bool = False) -> None:
        self.sync_count += 1

    def lock(self) -> nullcontext[None]:
        return nullcontext()

    def set_texts(self, texts: object) -> None:
        self.texts.append(texts)
