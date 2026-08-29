import inspect
import unittest
from pathlib import Path

import numpy as np

from fastreflex.evaluation.reflex_event import EventHoldoutGuard, _load_yaml
from fastreflex.evaluation.stability_temporal import _file_sha256
from fastreflex.evaluation.support_terrain_fusion import (
    POLICY_PF1,
    POLICY_PF2,
    fsr_loaded_feet,
    per_foot_context,
    per_foot_terrain_memory,
    raw_support_alert,
    select_per_foot_policy,
)
from fastreflex.evaluation.terrain_conditioned_reflex import (
    CONCRETE,
    MARBLE,
    SAND,
    UNKNOWN,
    TerrainGateTrace,
    TerrainPrediction,
    terrain_predictions,
)
from fastreflex.evaluation.transition_scenarios import fusion_regression


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiment/20260829_per_foot_terrain_memory_support_fusion.yaml"


def _trace() -> TerrainGateTrace:
    samples = 20
    return TerrainGateTrace(
        state=np.full(samples, MARBLE, dtype=np.int8),
        update_samples=np.asarray((2, 5, 9), dtype=np.int64),
        prediction_ids=np.asarray((3, 1, 0), dtype=np.int8),
        prediction_probabilities=np.asarray(
            ((0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
            dtype=np.float32,
        ),
        first_target_valid_sample=2,
        clean_event_count=3,
        prediction_feet=np.asarray(("LEFT", "RIGHT", "LEFT")),
    )


class TerrainPredictionProvenanceTest(unittest.TestCase):
    def test_prediction_schema_preserves_scheduler_foot_end_to_end(self) -> None:
        records = terrain_predictions(_trace())
        self.assertTrue(all(isinstance(record, TerrainPrediction) for record in records))
        self.assertEqual(
            [record.touchdown_foot for record in records],
            ["LEFT", "RIGHT", "LEFT"],
        )
        self.assertEqual([record.prediction_timestamp for record in records], [2, 5, 9])
        self.assertFalse(any("terrain_truth" in field for field in TerrainPrediction.__dataclass_fields__))

    def test_per_foot_updates_are_independent_and_never_expire(self) -> None:
        memory = per_foot_terrain_memory(_trace())
        self.assertEqual(memory.state[0].tolist(), [UNKNOWN, UNKNOWN])
        self.assertEqual(memory.state[2].tolist(), [SAND, UNKNOWN])
        self.assertEqual(memory.state[5].tolist(), [SAND, MARBLE])
        self.assertEqual(memory.state[9].tolist(), [CONCRETE, MARBLE])
        self.assertEqual(memory.state[-1].tolist(), [CONCRETE, MARBLE])
        self.assertEqual(memory.last_update_sample[-1].tolist(), [9, 5])

    def test_invalid_or_absent_prediction_cannot_overwrite_and_reset_is_unknown(self) -> None:
        one = TerrainGateTrace(
            state=np.zeros(8, dtype=np.int8),
            update_samples=np.asarray((1,), dtype=np.int64),
            prediction_ids=np.asarray((3,), dtype=np.int8),
            prediction_probabilities=np.ones((1, 4), dtype=np.float32),
            first_target_valid_sample=1,
            clean_event_count=1,
            prediction_feet=np.asarray(("LEFT",)),
        )
        memory = per_foot_terrain_memory(one)
        self.assertEqual(memory.state[-1].tolist(), [SAND, UNKNOWN])
        reset = TerrainGateTrace(
            state=np.zeros(8, dtype=np.int8),
            update_samples=np.empty(0, dtype=np.int64),
            prediction_ids=np.empty(0, dtype=np.int8),
            prediction_probabilities=np.empty((0, 4), dtype=np.float32),
            first_target_valid_sample=None,
            clean_event_count=0,
            prediction_feet=np.empty(0, dtype="<U5"),
        )
        self.assertTrue(np.all(per_foot_terrain_memory(reset).state == UNKNOWN))


class LoadedFootAndFusionTest(unittest.TestCase):
    def test_loaded_rule_is_current_fsr_only_and_causal(self) -> None:
        fsr = np.zeros((4, 8), dtype=np.float32)
        fsr[1, 0] = 1.0e-6
        fsr[2, 0] = 1.1e-6
        fsr[3, 4] = 2.0
        loaded, totals = fsr_loaded_feet(fsr)
        self.assertEqual(
            loaded.tolist(),
            [[False, False], [False, False], [True, False], [False, True]],
        )
        changed = fsr.copy()
        changed[3] = 999.0
        changed_loaded, _ = fsr_loaded_feet(changed)
        np.testing.assert_array_equal(loaded[:3], changed_loaded[:3])
        self.assertEqual(totals.shape, (4, 2))

    def test_pf1_rejects_unloaded_sand_memory(self) -> None:
        memory = np.asarray(((SAND, MARBLE), (SAND, MARBLE)), dtype=np.int8)
        loaded = np.asarray(((False, True), (True, True)), dtype=bool)
        totals = np.asarray(((0.0, 10.0), (5.0, 10.0)))
        context, dominant = per_foot_context(POLICY_PF1, memory, loaded, totals)
        self.assertEqual(context.tolist(), [False, True])
        self.assertEqual(dominant.tolist(), [1, 1])

    def test_pf2_uses_dominant_loaded_foot_and_left_tie_break(self) -> None:
        memory = np.asarray(((SAND, MARBLE), (SAND, MARBLE), (MARBLE, SAND)), dtype=np.int8)
        loaded = np.asarray(((True, True), (True, True), (False, False)), dtype=bool)
        totals = np.asarray(((5.0, 10.0), (5.0, 5.0), (0.0, 0.0)))
        context, dominant = per_foot_context(POLICY_PF2, memory, loaded, totals)
        self.assertEqual(dominant.tolist(), [1, 0, 0])
        self.assertEqual(context.tolist(), [False, True, False])

    def test_raw_support_trace_is_memory_independent(self) -> None:
        probability = np.asarray((0.95,) * 10)
        before = raw_support_alert(probability)
        memory_a = np.full((10, 2), SAND, dtype=np.int8)
        memory_b = np.full((10, 2), MARBLE, dtype=np.int8)
        loaded = np.ones((10, 2), dtype=bool)
        totals = np.ones((10, 2))
        per_foot_context(POLICY_PF1, memory_a, loaded, totals)
        per_foot_context(POLICY_PF2, memory_b, loaded, totals)
        after = raw_support_alert(probability)
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])


class SelectionAndRegressionTest(unittest.TestCase):
    def test_selection_pool_is_exact_and_near_tie_prefers_pf1(self) -> None:
        metrics = {
            "support_recall": 1.0,
            "sand_benign_specificity": 1.0,
            "premature_event_run_rate": 0.0,
            "context_suppression_rate": 0.0,
            "hard_ground_specificity": 1.0,
            "fusion_latency_ms": {"median": 0.0, "p95": 0.0},
        }
        gates = {
            "support_recall_min": 0.95,
            "sand_benign_specificity_min": 0.95,
            "premature_event_run_rate_max": 0.05,
            "context_suppression_rate_max": 0.05,
            "hard_ground_specificity_min": 0.95,
            "median_fusion_latency_ms_max": 20,
            "p95_fusion_latency_ms_max": 50,
        }
        selected = select_per_foot_policy(
            {POLICY_PF1: metrics, POLICY_PF2: metrics}, gates
        )
        self.assertEqual(selected["selected"], POLICY_PF1)
        self.assertEqual(
            {row["policy"] for row in selected["candidates"]},
            {POLICY_PF1, POLICY_PF2},
        )

    def test_config_freezes_model_support_memory_and_holdout_contracts(self) -> None:
        document = _load_yaml(CONFIG)
        terrain = document["bilateral_terrain"]
        self.assertEqual((terrain["profile"], terrain["family"], terrain["observation_ms"]), ("fsr4", "mlp", 50))
        self.assertFalse(terrain["foot_id_in_model_tensor"])
        self.assertEqual(terrain["seeds"], [17, 29, 43])
        self.assertEqual(document["loaded_foot"]["epsilon_n"], 1.0e-6)
        self.assertEqual(document["policies"]["allowed_selection_pool"], ["PF1", "PF2"])
        self.assertEqual(document["frozen_support"]["probability_threshold"], 0.94)
        self.assertEqual(document["frozen_support"]["persistence_ms"], 5)
        self.assertEqual(document["holdout"]["open_count_before_freeze"], 0)

    def test_protected_hashes_and_no_privileged_runtime_policy(self) -> None:
        document = _load_yaml(CONFIG)
        support = document["frozen_support"]
        for row in (support["normalizer"], *support["checkpoints"]):
            self.assertEqual(_file_sha256(ROOT / row["path"]), row["sha256"])
        source = inspect.getsource(per_foot_context)
        self.assertNotIn("terrain_truth", source)
        self.assertNotIn("loaded_contact", source)
        self.assertTrue(fusion_regression()["passed"])
        cli = (ROOT / "scripts/fastreflex.py").read_text(encoding="utf-8")
        self.assertIn("PER_FOOT_TERRAIN_MEMORY_SUPPORT_FUSION", cli)
        guard = EventHoldoutGuard()
        with self.assertRaises(RuntimeError):
            guard.require_open()
        guard.open_once()
        self.assertEqual(guard.open_count, 1)


if __name__ == "__main__":
    unittest.main()
