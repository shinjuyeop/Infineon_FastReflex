"""Frozen contracts for the Sand-benign generalization study design."""

from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import yaml

from fastreflex.evaluation.holdout import verify_generalization_holdout_evaluation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/experiment/20260902_sand_benign_generalization_study_design.yaml"
)
HOLDOUT_CONFIG = (
    ROOT
    / "configs/experiment/20260902_model_v2_generalization_holdout_one_shot_evaluation.yaml"
)
COMPONENTS = {
    "STUDY_PARAMETER_DOMAIN_SHA": "parameter_domain",
    "STUDY_SCENARIO_MATRIX_SHA": "scenario_matrix",
    "STUDY_SPLIT_PLAN_SHA": "split_plan",
    "STUDY_PHYSICAL_LABEL_CONTRACT_SHA": "physical_label_contract",
    "STUDY_DIVERSITY_METRICS_SHA": "diversity_metrics",
    "STUDY_OBSERVABILITY_METRICS_SHA": "observability_metrics",
    "STUDY_DECISION_RULE_SHA": "decision_rules",
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expand_matrix(document: dict[str, object]) -> list[dict[str, object]]:
    matrix = document["scenario_matrix"]
    assert isinstance(matrix, dict)
    groups = matrix["groups"]
    assert isinstance(groups, dict)
    sand_mechanics = matrix["fixed_sand_mechanics"]
    assert isinstance(sand_mechanics, dict)
    rows: list[dict[str, object]] = []
    for group_name, raw_group in groups.items():
        assert isinstance(raw_group, dict)
        fixed = (
            sand_mechanics
            if "sand_benign" in group_name
            else raw_group["fixed_mechanics"]
        )
        assert isinstance(fixed, dict)
        templates = raw_group["templates"]
        assert isinstance(templates, dict)
        for split, raw_templates in templates.items():
            assert isinstance(raw_templates, list)
            for template in raw_templates:
                assert isinstance(template, dict)
                for source in raw_group["sources"]:
                    for speed in raw_group["speeds_mps"]:
                        rows.append(
                            {
                                **fixed,
                                **template,
                                "group": group_name,
                                "split": split,
                                "source_terrain": source,
                                "speed_mps": speed,
                            }
                        )
    return rows


class SandBenignStudyDesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        cls.rows = _expand_matrix(cls.document)
        cls.signature_fields = cls.document["scenario_matrix"]["signature_fields"]

    def _signature(self, row: dict[str, object]) -> tuple[object, ...]:
        return tuple(row[field] for field in self.signature_fields)

    def test_design_hashes_are_deterministic(self) -> None:
        hashes = self.document["design_hashes"]
        computed = {
            name: _canonical_sha256(self.document[key])
            for name, key in COMPONENTS.items()
        }
        self.assertEqual({name: hashes[name] for name in COMPONENTS}, computed)
        bundle = {
            "experiment_id": self.document["experiment"]["id"],
            "dataset_id": self.document["dataset_plan"]["dataset_id"],
            "counts": self.document["scenario_matrix"]["counts"],
            "component_hashes": computed,
        }
        self.assertEqual(
            hashes["SAND_BENIGN_GENERALIZATION_STUDY_DESIGN_SHA"],
            _canonical_sha256(bundle),
        )

    def test_matrix_counts_and_balance_are_exact(self) -> None:
        self.assertEqual(len(self.rows), 176)
        self.assertEqual(
            Counter(row["split"] for row in self.rows),
            {"STUDY_DISCOVERY": 88, "STUDY_CONFIRMATION": 88},
        )
        self.assertEqual(
            Counter(row["group"] for row in self.rows),
            {
                "broad_sand_benign": 96,
                "near_hazard_sand_benign": 48,
                "ordinary_support_control": 24,
                "delayed_support_control": 8,
            },
        )
        sand = [row for row in self.rows if "sand_benign" in row["group"]]
        self.assertEqual(len(sand), 144)
        self.assertEqual(
            set(
                Counter(
                    (row["source_terrain"], row["speed_mps"]) for row in sand
                ).values()
            ),
            {24},
        )
        for split in ("STUDY_DISCOVERY", "STUDY_CONFIRMATION"):
            selected = [row for row in sand if row["split"] == split]
            self.assertEqual(
                Counter(row["severity_intent"] for row in selected),
                {"LOW": 24, "MEDIUM": 24, "NEAR_HAZARD": 24},
            )
            self.assertEqual(
                Counter(row["start_stratum"] for row in selected),
                {"EARLY": 24, "MID": 24, "LATE": 24},
            )
            self.assertEqual(
                Counter(row["width_stratum"] for row in selected),
                {"NARROW": 24, "MEDIUM": 24, "WIDE": 24},
            )
            self.assertEqual(
                Counter(row["sink_pattern"] for row in selected),
                {"transition_left": 36, "transition_right": 36},
            )

    def test_planned_signatures_are_unique_and_cross_split_not_near(self) -> None:
        signatures = [self._signature(row) for row in self.rows]
        self.assertEqual(len(signatures), len(set(signatures)))
        comparable = (
            "source_terrain",
            "target_terrain",
            "speed_mps",
            "slip_pattern",
            "sink_pattern",
            "sink_severity",
            "support_pattern",
        )
        for index, left in enumerate(self.rows):
            for right in self.rows[index + 1 :]:
                if left["split"] == right["split"]:
                    continue
                same_domain = all(left[key] == right[key] for key in comparable)
                near_geometry = (
                    abs(left["patch_start_x_m"] - right["patch_start_x_m"])
                    < 0.002
                    and abs(left["patch_width_m"] - right["patch_width_m"])
                    < 0.004
                )
                self.assertFalse(same_domain and near_geometry)

    def test_no_exact_overlap_with_protected_historical_metadata(self) -> None:
        historical: set[tuple[object, ...]] = set()
        for record in self.document["protected_objects"]["datasets"]:
            path = ROOT / record["path"]
            self.assertEqual(_file_sha256(path), record["sha256"])
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["runs"]), record["count"])
            for row in manifest["runs"]:
                signature = row.get("physical_signature")
                if isinstance(signature, list) and len(signature) == 9:
                    historical.add(tuple(signature))
                elif all(field in row for field in self.signature_fields):
                    historical.add(self._signature(row))
        self.assertFalse(set(map(self._signature, self.rows)) & historical)

    def test_consumed_holdout_stays_closed_and_is_not_targeted(self) -> None:
        boundary = self.document["consumed_holdout_boundary"]
        counters = self.document["counters"]
        anti = self.document["anti_overfitting"]
        self.assertEqual(boundary["guard"], 1)
        self.assertEqual(counters["holdout_payload_reads"], 0)
        self.assertEqual(counters["holdout_model_inference"], 0)
        self.assertEqual(counters["holdout_feature_reconstructions"], 0)
        self.assertEqual(counters["holdout_visualizations"], 0)
        self.assertFalse(anti["exact_patch_start_0.362_used"])
        self.assertFalse(anti["exact_patch_width_0.735_used"])
        self.assertNotIn(0.362, [row["patch_start_x_m"] for row in self.rows])
        self.assertNotIn(0.735, [row["patch_width_m"] for row in self.rows])
        with patch(
            "fastreflex.evaluation.generalization.np.load",
            side_effect=AssertionError("consumed HOLDOUT payload access"),
        ):
            verification = verify_generalization_holdout_evaluation(
                ROOT,
                HOLDOUT_CONFIG,
            )
        self.assertTrue(verification["passed"])
        self.assertFalse(verification["holdout_payload_deserialized"])
        self.assertEqual(verification["guard_after"], 1)

    def test_design_is_non_executing_and_keeps_exact_final_candidate(self) -> None:
        counters = self.document["counters"]
        for key in (
            "optimizer_steps",
            "checkpoint_writes",
            "normalizer_fits",
            "hnm_rounds",
            "threshold_searches",
            "persistence_searches",
            "architecture_searches",
            "seed_searches",
            "new_simulation_runs",
        ):
            self.assertEqual(counters[key], 0)
        final = self.document["protected_objects"]["models"]["final_v2"]
        self.assertEqual(
            _file_sha256(ROOT / final["record"]["path"]),
            final["record"]["sha256"],
        )
        self.assertTrue(self.document["no_training"])
        self.assertTrue(self.document["no_generation"])
        self.assertEqual(
            self.document["verdict"],
            "SAND_BENIGN_GENERALIZATION_STUDY_DESIGN_READY",
        )
        self.assertEqual(
            self.document["recommended_next_milestone"],
            "SAND_BENIGN_GENERALIZATION_STUDY_GENERATION",
        )


if __name__ == "__main__":
    unittest.main()
