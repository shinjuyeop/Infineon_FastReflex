"""Regression tests for mild-Sand physical calibration and redesign."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from fastreflex.dataset.generation import _load_yaml
from fastreflex.dataset.loader import sha256_file
from fastreflex.dataset.sand_mild_calibration import (
    MILD_RECALIBRATED_SPLITS,
    build_mild_physical_ledger,
    collect_mild_recalibrated_study,
    expand_mild_recalibrated_redesign,
    load_mild_recalibrated_discovery_payload,
    validate_mild_recalibrated_redesign,
    verify_mild_recalibrated_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/experiment"
MASTER = CONFIG_DIR / "20260903_sand_benign_mild_domain_calibration_redesign.yaml"
REDESIGN = (
    CONFIG_DIR
    / "20260903_sand_benign_generalization_study_mild_recalibrated_redesign.yaml"
)
GENERATION_CONFIG = (
    CONFIG_DIR
    / "20260903_sand_benign_generalization_study_mild_recalibrated_generation.yaml"
)
CURRENT = ROOT / "data/raw/sand_benign_generalization_redesigned_study_20260902"
FUTURE = ROOT / "data/raw/sand_benign_generalization_mild_recalibrated_study_20260903"
HOLDOUT_GUARD = (
    ROOT
    / "artifacts/runs/20260902_model_v2_generalization_holdout_one_shot_evaluation"
    / "holdout_access_guard.json"
)
PILOTS = {
    "a": {
        "config_sha256": "04a42b67dc964b2d49e4f341735dd807e1bbf2191356cd3cc4c4203710e27b66",
        "manifest_sha256": "1f5f7fe0e83a55a7e4be8240c3ff4ab2282e00194d6151c3ac909ac1a697c14f",
        "dataset_freeze_sha256": "7c55cd63ba7084c7265c2515d71124cb5ee8d8435659d7815e1ae9d2e427e97b",
        "runs": 36,
        "strict": 27,
    },
    "b": {
        "config_sha256": "b2e296739708fd00e502e108afaf986ea333b09bfb65bdad9ed951f74a7bd183",
        "manifest_sha256": "b57603925da065f01657d65bacaf93d9bebd0859e319c32f9973ce7c22c6335a",
        "dataset_freeze_sha256": "3ba8f2661e6a6110194c5e94ccf3dfb66926757690e065c61e057f923237ad03",
        "runs": 12,
        "strict": 10,
    },
    "c": {
        "config_sha256": "b51e74871235be578ef62686f7ad440450a392693a7365b37502da33011be1bd",
        "manifest_sha256": "90bb0ce2cd9fb2e52f57bc45ad247a64e1217d477e4b098d016fe3393d1dcb50",
        "dataset_freeze_sha256": "5b4582a43b7c6177c9961d68dcce39e1f53d5aadaef073fc4ae37f5e53a22f0e",
        "runs": 24,
        "strict": 24,
    },
}


class SandMildCalibrationTest(unittest.TestCase):
    def test_master_contract_is_frozen_and_model_blind(self) -> None:
        document = _load_yaml(MASTER)
        self.assertEqual(
            sha256_file(MASTER),
            "0ac6a6017b38fc412c6311a8ac89c513bd0381e15317de5e7c6dc623bb11c7bd",
        )
        protocol = document["pilot_protocol"]
        self.assertEqual(protocol["maximum_new_simulations"], 72)
        self.assertEqual(protocol["maximum_batches"], 3)
        self.assertTrue(protocol["model_blind"])
        self.assertEqual(protocol["within_batch_adaptation"], "forbidden")
        self.assertEqual(protocol["adaptive_replacement_or_backfill"], "forbidden")
        self.assertEqual(document["scientific_boundary"]["required_guard_after"], 1)
        for name, value in document["counters"].items():
            self.assertEqual(value, 0, name)

    def test_pilot_configs_and_generated_batches_are_frozen(self) -> None:
        total = 0
        for name, expected in PILOTS.items():
            config = CONFIG_DIR / (
                f"20260903_sand_benign_mild_domain_calibration_pilot_{name}.yaml"
            )
            self.assertEqual(sha256_file(config), expected["config_sha256"])
            dataset = (
                ROOT
                / f"data/raw/sand_benign_mild_domain_calibration_pilot_{name}_20260903"
            )
            if not dataset.exists():
                continue
            manifest_path = dataset / "manifest.json"
            self.assertEqual(sha256_file(manifest_path), expected["manifest_sha256"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_count"], expected["runs"])
            self.assertFalse(manifest["adaptive_within_batch"])
            self.assertEqual(manifest["replacement_run_count"], 0)
            self.assertTrue(manifest["model_blind"])
            self.assertEqual(manifest["model_inference_runs"], 0)
            self.assertTrue(
                all(not row["model_outputs_present"] for row in manifest["runs"])
            )
            outcomes = Counter(
                row["objective_physical_outcome"] for row in manifest["runs"]
            )
            self.assertEqual(outcomes["STRICT_BENIGN"], expected["strict"])
            self.assertEqual(outcomes["SLIP"], 0)
            self.assertEqual(outcomes["SUPPORT"], 0)
            freeze = json.loads(
                (dataset / "dataset_freeze.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                freeze["dataset_freeze_sha256"], expected["dataset_freeze_sha256"]
            )
            prefreeze = json.loads(
                (dataset / "pre_simulation_freeze.json").read_text(encoding="utf-8")
            )
            self.assertEqual(prefreeze["config_sha256"], expected["config_sha256"])
            self.assertEqual(prefreeze["run_count"], expected["runs"])
            self.assertEqual(prefreeze["status"], "FROZEN_BEFORE_FIRST_SIMULATION")
            total += manifest["run_count"]

        if total:
            self.assertEqual(total, 72)

    def test_final_pilot_is_balanced_and_all_strict(self) -> None:
        dataset = ROOT / "data/raw/sand_benign_mild_domain_calibration_pilot_c_20260903"
        if not dataset.exists():
            self.skipTest("mild calibration pilots are local Gitignored artifacts")
        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        cells = Counter(
            (row["source_terrain"], float(row["speed_mps"])) for row in manifest["runs"]
        )
        self.assertEqual(set(cells.values()), {4})
        self.assertEqual(len(cells), 6)
        self.assertTrue(
            all(
                row["objective_physical_outcome"] == "STRICT_BENIGN"
                for row in manifest["runs"]
            )
        )

    def test_physical_ledgers_reproduce_current_and_pilot_results(self) -> None:
        pilot_paths = [
            ROOT / f"data/raw/sand_benign_mild_domain_calibration_pilot_{name}_20260903"
            for name in PILOTS
        ]
        if not CURRENT.exists() or not all(path.exists() for path in pilot_paths):
            self.skipTest("physical corpora are local Gitignored artifacts")
        current = build_mild_physical_ledger(CURRENT)
        pilots = [
            row for path in pilot_paths for row in build_mild_physical_ledger(path)
        ]
        current_outcomes = Counter(row["objective_physical_outcome"] for row in current)
        pilot_outcomes = Counter(row["objective_physical_outcome"] for row in pilots)
        self.assertEqual((len(current), current_outcomes["STRICT_BENIGN"]), (96, 80))
        self.assertEqual((len(pilots), pilot_outcomes["STRICT_BENIGN"]), (72, 61))
        current_invalid = [
            row for row in current if row["objective_physical_outcome"] == "INVALID"
        ]
        pilot_invalid = [
            row for row in pilots if row["objective_physical_outcome"] == "INVALID"
        ]
        self.assertEqual(len(current_invalid), 16)
        self.assertEqual(len(pilot_invalid), 11)
        self.assertEqual(
            Counter(row["fall_relation"] for row in current_invalid),
            {
                "PRE_TARGET": 2,
                "DURING_TARGET_CONTACT": 10,
                "AFTER_LAST_TARGET_CONTACT": 4,
            },
        )
        self.assertEqual(
            Counter(row["fall_relation"] for row in pilot_invalid),
            {"DURING_TARGET_CONTACT": 9, "AFTER_LAST_TARGET_CONTACT": 2},
        )

    def test_future_matrix_is_deterministic_fresh_and_conditioned(self) -> None:
        document = _load_yaml(REDESIGN)
        self.assertEqual(
            sha256_file(REDESIGN),
            "1301d64391b423eb50b2ac4188058e1ac0cd988dce477323f87891b946aeaeb5",
        )
        rows = expand_mild_recalibrated_redesign(document)
        self.assertEqual(len(rows), 176)
        self.assertEqual(len({row["run_id"] for row in rows}), 176)
        broad = [row for row in rows if row["group"] == "broad_sand_benign"]
        for split in MILD_RECALIBRATED_SPLITS:
            split_rows = [row for row in broad if row["split"] == split]
            self.assertEqual(len(split_rows), 48)
            for source in ("concrete", "marble"):
                for speed in (0.20, 0.25, 0.30):
                    cell = [
                        row
                        for row in split_rows
                        if row["source_terrain"] == source
                        and float(row["speed_mps"]) == speed
                    ]
                    patterns = Counter(row["sink_pattern"] for row in cell)
                    expected = (
                        {"transition_left": 8}
                        if source == "concrete" and speed == 0.25
                        else {"transition_left": 6, "transition_right": 2}
                    )
                    self.assertEqual(patterns, expected)

        historical_paths = [
            ROOT / item["path"]
            for item in document["historical_signature_policy"]["manifests"]
        ]
        if all(path.exists() for path in historical_paths):
            audit = validate_mild_recalibrated_redesign(ROOT, document)
        else:
            empty_audit = {
                "exact_by_reference": {},
                "near_by_reference": {},
                "run_id_reuse_by_reference": {},
                "exact_total": 0,
                "near_total": 0,
                "run_id_reuse_total": 0,
            }
            with (
                patch(
                    "fastreflex.dataset.sand_mild_calibration._historical_signatures",
                    return_value=(set(), []),
                ),
                patch(
                    "fastreflex.dataset.sand_mild_calibration._historical_overlap_audit",
                    return_value=empty_audit,
                ),
            ):
                audit = validate_mild_recalibrated_redesign(ROOT, document)
        self.assertEqual(audit["historical_signature_overlap"], 0)
        self.assertEqual(audit["historical_run_id_reuse"], 0)
        self.assertEqual(audit["cross_split_exact_overlap"], 0)
        self.assertEqual(audit["cross_split_parameter_near_duplicates"], 0)
        self.assertEqual(
            audit["redesign_sha256"],
            "09c2e1a22d47ba115dc2ef3db0251a7dd836096ffe2b9e370fbe9d1677416356",
        )

    def test_model_boundary_and_consumed_holdout_guard(self) -> None:
        source = inspect.getsource(build_mild_physical_ledger) + inspect.getsource(
            collect_mild_recalibrated_study
        )
        for forbidden in (
            "predict_proba(",
            "load_model(",
            "torch.load(",
            "onnxruntime.InferenceSession",
        ):
            self.assertNotIn(forbidden, source)
        guard = json.loads(HOLDOUT_GUARD.read_text(encoding="utf-8"))
        self.assertEqual(guard["guard_after"], 1)
        self.assertEqual(guard["scientific_open_count"], 1)
        self.assertTrue(guard["second_scientific_open_forbidden"])

    def test_generation_execution_config_freezes_all_inputs(self) -> None:
        execution = _load_yaml(GENERATION_CONFIG)
        generation = execution["generation"]
        self.assertEqual(
            sha256_file(GENERATION_CONFIG),
            "151c89523a27fb92dd46cbc3dc3f60193c1916fb16503bce903e44fe6006add3",
        )
        self.assertEqual(generation["planned_total_runs"], 176)
        self.assertEqual(generation["planned_discovery_runs"], 88)
        self.assertEqual(generation["planned_confirmation_runs"], 88)
        self.assertEqual(generation["planned_broad_mild_runs"], 96)
        self.assertEqual(generation["planned_boundary_moderate_runs"], 48)
        self.assertEqual(generation["planned_ordinary_support_runs"], 24)
        self.assertEqual(generation["planned_delayed_support_runs"], 8)
        self.assertEqual(generation["redesign_config_sha256"], sha256_file(REDESIGN))
        for category in ("implementation_artifacts", "protected_artifacts"):
            for artifact in generation[category]:
                self.assertEqual(
                    artifact["sha256"], sha256_file(ROOT / artifact["path"])
                )
        guards = execution["protocol_guards"]
        for key in (
            "historical_dataset_reuse",
            "previous_sand_study_reuse",
            "calibration_pilot_reuse",
            "adaptive_backfill",
            "adaptive_replacement",
        ):
            self.assertFalse(guards[key])

    def test_confirmation_loader_refuses_before_npz_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            manifest = {
                "dataset_id": (
                    "sand_benign_generalization_mild_recalibrated_study_20260903"
                ),
                "runs": [
                    {
                        "run_id": "sealed",
                        "split": "MILD_RECALIBRATED_CONFIRMATION",
                        "file": "must_not_exist.npz",
                        "file_sha256": "unreachable",
                    }
                ],
            }
            path = dataset / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            (dataset / "manifest.sha256").write_text(
                f"{sha256_file(path)}  manifest.json\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "SEALED"):
                load_mild_recalibrated_discovery_payload(dataset, "sealed")

    def test_generated_dataset_freeze_when_present(self) -> None:
        if not FUTURE.exists():
            self.skipTest("mild-recalibrated corpus has not been generated yet")
        verification = verify_mild_recalibrated_dataset(FUTURE)
        self.assertTrue(verification["passed"])
        self.assertEqual(verification["run_count"], 176)

        manifest = json.loads((FUTURE / "manifest.json").read_text(encoding="utf-8"))
        audit = json.loads((FUTURE / "physical_audit.json").read_text(encoding="utf-8"))
        freeze = json.loads(
            (FUTURE / "dataset_freeze.json").read_text(encoding="utf-8")
        )
        seal = json.loads(
            (FUTURE / "confirmation_seal.json").read_text(encoding="utf-8")
        )
        prefreeze = json.loads(
            (FUTURE / "pre_simulation_freeze.json").read_text(encoding="utf-8")
        )

        self.assertEqual(
            manifest["split_counts"], dict.fromkeys(MILD_RECALIBRATED_SPLITS, 88)
        )
        self.assertEqual(manifest["valid_count"], 169)
        self.assertEqual(manifest["invalid_count"], 7)
        self.assertEqual(manifest["adaptive_backfill_count"], 0)
        self.assertEqual(manifest["replacement_run_count"], 0)
        self.assertEqual(manifest["rerun_count"], 0)
        self.assertEqual(manifest["model_inference_runs"], 0)
        self.assertEqual(manifest["old_holdout_payload_reads"], 0)
        self.assertEqual(manifest["model_output_fields"], [])
        self.assertTrue(
            all(not row["model_outputs_present"] for row in manifest["runs"])
        )

        self.assertEqual(
            audit["overall_outcomes"],
            {
                "STRICT_BENIGN": 134,
                "SUPPORT": 32,
                "SLIP": 3,
                "DUAL_HAZARD": 0,
                "INVALID": 7,
            },
        )
        self.assertEqual(audit["objective_valid"], 169)
        self.assertEqual(audit["objective_invalid"], 7)
        for split in MILD_RECALIBRATED_SPLITS:
            self.assertEqual(
                sum(
                    cell["strict_benign_mild"] for cell in audit["mild"][split].values()
                ),
                48,
            )
            self.assertTrue(
                all(
                    cell["strict_benign_mild"] == 8
                    for cell in audit["mild"][split].values()
                )
            )
            moderate = audit["moderate_boundary"][split]
            expected_moderate = 21 if split.endswith("DISCOVERY") else 17
            self.assertEqual(
                sum(cell["strict_boundary_moderate"] for cell in moderate.values()),
                expected_moderate,
            )
            self.assertTrue(
                all(cell["strict_boundary_moderate"] >= 1 for cell in moderate.values())
            )
            self.assertEqual(
                audit["support_controls"][split]["ordinary"]["qualified_support"], 12
            )
            self.assertEqual(
                audit["support_controls"][split]["delayed"]["qualified_support"], 4
            )
            self.assertEqual(
                audit["phase_diversity"][split]["cells_with_both_principal_phases"], 5
            )
            self.assertEqual(
                audit["phase_diversity"][split]["cells_with_usable_phase"], 6
            )
            self.assertTrue(
                audit["phase_diversity"][split][
                    "concrete_0.25_predeclared_exception_used"
                ]
            )
            self.assertEqual(audit["entry_timing"]["by_split"][split]["span"], 590)

        self.assertEqual(audit["invalidity"]["pretarget_fall"], 1)
        self.assertEqual(audit["invalidity"]["posttarget_or_censor"], 6)
        self.assertEqual(audit["invalidity"]["other"], 0)
        signatures = audit["physical_signatures"]
        self.assertEqual(
            (signatures["unique_count"], signatures["valid_count"]), (167, 169)
        )
        self.assertEqual(signatures["exact_duplicates"], 2)
        self.assertEqual(signatures["near_duplicate_count"], 64)
        self.assertEqual(signatures["cross_split_near_duplicate_count"], 37)
        self.assertEqual(audit["historical_contamination"]["exact_total"], 0)
        self.assertEqual(audit["historical_contamination"]["run_id_reuse_total"], 0)
        self.assertEqual(manifest["matrix_audit"]["cross_split_exact_overlap"], 0)
        self.assertEqual(
            manifest["matrix_audit"]["cross_split_parameter_near_duplicates"], 0
        )
        self.assertEqual(len(audit["generation_gates"]), 70)
        self.assertTrue(
            all(gate["passed"] for gate in audit["generation_gates"].values())
        )
        self.assertEqual(
            audit["generation_verdict"],
            "SAND_BENIGN_GENERALIZATION_STUDY_MILD_RECALIBRATED_GENERATION_READY",
        )

        self.assertEqual(
            prefreeze["execution_config_sha256"],
            "151c89523a27fb92dd46cbc3dc3f60193c1916fb16503bce903e44fe6006add3",
        )
        self.assertEqual(prefreeze["status"], "FROZEN_BEFORE_FIRST_SIMULATION")
        self.assertEqual(
            freeze["MILD_RECALIBRATED_STUDY_MANIFEST_SHA"],
            "f19ec527cb9faac0d8f3a385a1a63e8a951ced7f275c7cfc3dd459cc42f375d1",
        )
        self.assertEqual(
            freeze["MILD_RECALIBRATED_DATASET_FREEZE_SHA"],
            "706d939c03bf31df0fb39d1043e99dbbb05922664e207425c8c96ab7c93ee675",
        )
        self.assertEqual(
            verification["dataset_freeze_file_sha256"],
            "dafcf8f42b2cc701358d6b55f40727ec234ce975ae992e7c3811b8640c5805b6",
        )
        self.assertEqual(seal["status"], "SEALED_FOR_MILD_RECALIBRATED_CONFIRMATION")
        self.assertTrue(seal["objective_physical_audit_only"])
        for forbidden in (
            "model_inference",
            "normalized_80d_analysis",
            "observability_analysis",
            "hypothesis_selection",
            "visualization",
        ):
            self.assertFalse(seal[forbidden])


if __name__ == "__main__":
    unittest.main()
