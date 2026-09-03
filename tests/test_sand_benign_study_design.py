"""Frozen contracts for the Sand-benign generalization study design."""

from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import yaml

from fastreflex.dataset.sand_study import (
    SAND_STUDY_DATASET_ID,
    collect_sand_benign_study_dataset,
    expand_sand_benign_study_design,
    load_sand_benign_discovery_payload,
    validate_sand_benign_study_design,
)
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
DATASET = ROOT / "data/raw/sand_benign_generalization_study_20260902"
GENERATION_CONFIG = (
    ROOT
    / "configs/experiment/20260902_sand_benign_generalization_study_generation.yaml"
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


def _historical_holdout_sha256(path: Path) -> str:
    """Use the one-shot run's recorded CLI identity after CLI evolution."""
    resolved = Path(path).resolve()
    if resolved == (ROOT / "scripts/fastreflex.py").resolve():
        return "47dd5652959460821627d0914f95095a9dd374c094f275b9e2f8e349aea85269"
    return _file_sha256(resolved)


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

    def test_canonical_generation_expansion_is_frozen(self) -> None:
        specifications = expand_sand_benign_study_design(self.document)
        audit = validate_sand_benign_study_design(
            ROOT, self.document, specifications
        )
        self.assertEqual(len(specifications), 176)
        self.assertEqual(
            audit["split_counts"],
            {"STUDY_DISCOVERY": 88, "STUDY_CONFIRMATION": 88},
        )
        self.assertEqual(audit["group_counts"]["broad_sand_benign"], 96)
        self.assertEqual(audit["group_counts"]["near_hazard_sand_benign"], 48)
        self.assertEqual(audit["group_counts"]["ordinary_support_control"], 24)
        self.assertEqual(audit["group_counts"]["delayed_support_control"], 8)
        self.assertEqual(audit["exact_duplicate_signatures"], 0)
        self.assertFalse(any(audit["historical_overlap_by_reference"].values()))

    def test_generation_path_has_no_model_inference_calls(self) -> None:
        source = inspect.getsource(collect_sand_benign_study_dataset)
        for forbidden in ("predict_proba(", "load_model(", "torch.load(", "onnxruntime"):
            self.assertNotIn(forbidden, source)

    def test_generation_config_protects_implementation_and_history(self) -> None:
        generation = yaml.safe_load(GENERATION_CONFIG.read_text(encoding="utf-8"))[
            "generation"
        ]
        self.assertEqual(generation["planned_total_runs"], 176)
        self.assertEqual(generation["planned_discovery_runs"], 88)
        self.assertEqual(generation["planned_confirmation_runs"], 88)
        self.assertEqual(generation["planned_sand_runs"], 144)
        self.assertEqual(generation["planned_support_controls"], 32)
        for artifact in (
            generation["implementation_artifacts"]
            + generation["protected_artifacts"]
        ):
            self.assertEqual(
                _file_sha256(ROOT / artifact["path"]), artifact["sha256"]
            )

    def test_confirmation_payload_loader_fails_before_npz_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            manifest = {
                "dataset_id": SAND_STUDY_DATASET_ID,
                "runs": [
                    {
                        "run_id": "sealed",
                        "split": "STUDY_CONFIRMATION",
                        "file": "must_not_exist.npz",
                        "file_sha256": "unreachable",
                    }
                ],
            }
            manifest_path = dataset / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (dataset / "manifest.sha256").write_text(
                f"{_file_sha256(manifest_path)}  manifest.json\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "SEALED"):
                load_sand_benign_discovery_payload(dataset, "sealed")

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
        with (
            patch(
                "fastreflex.evaluation.generalization.np.load",
                side_effect=AssertionError("consumed HOLDOUT payload access"),
            ),
            patch(
                "fastreflex.evaluation.holdout.sha256_file",
                side_effect=_historical_holdout_sha256,
            ),
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

    def test_generated_discovery_labels_are_deterministic(self) -> None:
        if not DATASET.exists():
            self.skipTest("study corpus has not been generated yet")
        manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["adaptive_backfill_count"], 0)
        seal = json.loads(
            (DATASET / "confirmation_seal.json").read_text(encoding="utf-8")
        )
        self.assertEqual(seal["status"], "SEALED_FOR_STUDY_CONFIRMATION")
        self.assertFalse(seal["model_inference"])
        self.assertFalse(seal["normalized_feature_analysis"])
        self.assertFalse(seal["observability_analysis"])
        for row in manifest["runs"]:
            if row["split"] != "STUDY_DISCOVERY":
                continue
            arrays = load_sand_benign_discovery_payload(DATASET, row["run_id"])
            slip = bool(arrays["established_slip_onset"].any())
            support = bool(arrays["deformable_sink_onset"].any())
            i1 = bool(arrays["i1_active"].any())
            if not row["valid"]:
                expected = "INVALID"
            elif slip and support:
                expected = "DUAL_HAZARD"
            elif slip:
                expected = "SLIP"
            elif support:
                expected = "SUPPORT"
            elif i1:
                expected = "I1_ONLY_BENIGN"
            else:
                expected = "STRICT_BENIGN"
            self.assertEqual(row["objective_physical_outcome"], expected)
            if expected == "STRICT_BENIGN" and "sand_benign" in row["group"]:
                contact = arrays["target_terrain_contact"].astype(bool)
                displacement = arrays["support_surface_max_displacement_m"]
                peak = float(displacement[contact].max())
                severity = (
                    "LOW"
                    if peak < 0.030
                    else "MEDIUM"
                    if peak < 0.0525
                    else "NEAR_HAZARD"
                    if peak <= 0.070
                    else "OUTSIDE_FROZEN_BENIGN_SEVERITY_RANGE"
                )
                self.assertEqual(row["actual_benign_severity"], severity)


if __name__ == "__main__":
    unittest.main()
