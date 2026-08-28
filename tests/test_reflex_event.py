from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import yaml

from fastreflex.dataset.loader import Normalizer
from fastreflex.evaluation.reflex_event import (
    EVENT_REPRESENTATIONS,
    EVENT_CLASS_NAMES,
    EVENT_TYPE_BOTH,
    EVENT_TYPE_NONE,
    EVENT_TYPE_SLIP,
    EVENT_TYPE_SUPPORT,
    PELVIS_IMU6,
    PELVIS_IMU6_FSR8,
    EventHoldoutGuard,
    EventRun,
    ReplayTrace,
    _protected_hashes,
    _selection_recommendation,
    build_event_windows,
    classify_event_detection,
    event_early_negative_endpoints,
    event_positive_endpoints,
    evaluate_event_runs,
    generate_event_specifications,
    persistent_threshold_events,
    physical_signature,
    select_event_threshold,
    sustained_confirmation_sample,
    union_event_clock,
    validate_event_design,
)
from fastreflex.evaluation.transition_scenarios import (
    VALID_FALL,
    VALID_STABLE,
    fusion_regression,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiment/20260828_event_centric_reflex_trigger.yaml"
DENSE_CONFIG = ROOT / "configs/experiment/20260828_dense_fall_risk_detector_poc.yaml"


def synthetic_run(
    run_id: str,
    *,
    event_sample: int | None,
    event_type: str,
    target: str = "sand",
    outcome: str = VALID_STABLE,
    control: bool = False,
) -> EventRun:
    samples = 500
    imu = np.zeros((samples, 6), dtype=np.float32)
    fsr = np.ones((samples, 8), dtype=np.float32)
    slip = (event_sample, None) if event_type in (EVENT_TYPE_SLIP, EVENT_TYPE_BOTH) else (None, None)
    support = (None, event_sample) if event_type in (EVENT_TYPE_SUPPORT, EVENT_TYPE_BOTH) else (None, None)
    return EventRun(
        run_id=run_id,
        split="train",
        source_terrain="concrete",
        target_terrain=target,
        design_role="diagnostic_only",
        first_contact_sample=100,
        first_touchdown_sample=100,
        censor_sample=samples,
        outcome_diagnostic=outcome,
        fall_sample_diagnostic=450 if outcome == VALID_FALL else None,
        features={
            PELVIS_IMU6: imu,
            PELVIS_IMU6_FSR8: np.concatenate((imu, fsr), axis=1),
        },
        timestamp_us=(np.arange(samples) + 1).astype(np.int64) * 1000,
        slip_event_samples_per_foot=slip,
        support_event_samples_per_foot=support,
        event_sample=event_sample,
        event_type=event_type,
        hard_stable_control=control,
        drift_m=np.zeros((samples, 2), dtype=np.float32),
        tangential_velocity_mps=np.zeros((samples, 2), dtype=np.float32),
        support_spread_m=np.zeros((samples, 2), dtype=np.float32),
        support_max_displacement_m=np.zeros((samples, 2), dtype=np.float32),
        loaded_contact=np.ones((samples, 2), dtype=bool),
        sink_pattern="transition_left" if target == "sand" else "uniform",
        support_pattern="balanced_deformable" if target == "sand" else "balanced_soft",
    )


class PhysicalOracleTests(unittest.TestCase):
    def test_frozen_slip_threshold_and_three_ms_persistence(self) -> None:
        drift = np.zeros((12, 2), dtype=np.float64)
        drift[5:8, 0] = 0.050
        valid = np.ones((12, 2), dtype=bool)
        episodes = np.zeros((12, 2), dtype=np.int64)
        active, onset = persistent_threshold_events(drift, valid, episodes, 0.050, 3)
        self.assertFalse(active[6, 0])
        self.assertTrue(active[7, 0])
        self.assertTrue(onset[7, 0])

    def test_any_slip_left_right_and_bilateral(self) -> None:
        support = np.zeros((20, 2), dtype=bool)
        for samples, expected in (
            ((4, None), EVENT_TYPE_SLIP),
            ((None, 6), EVENT_TYPE_SLIP),
            ((4, 6), EVENT_TYPE_SLIP),
        ):
            slip = np.zeros((20, 2), dtype=bool)
            for side, sample in enumerate(samples):
                if sample is not None:
                    slip[sample, side] = True
            event, event_type = union_event_clock(slip, support)
            self.assertEqual(event, min(sample for sample in samples if sample is not None))
            self.assertEqual(event_type, expected)

    def test_affected_foot_not_required_and_union_clock_is_deterministic(self) -> None:
        slip = np.zeros((20, 2), dtype=bool)
        support = np.zeros((20, 2), dtype=bool)
        slip[8, 1] = True
        support[10, 0] = True
        self.assertEqual(union_event_clock(slip, support), (8, EVENT_TYPE_BOTH))
        self.assertEqual(union_event_clock(slip.copy(), support.copy()), (8, EVENT_TYPE_BOTH))

    def test_support_ten_mm_twenty_ms_and_balanced_negative(self) -> None:
        spread = np.zeros((40, 2), dtype=np.float64)
        spread[5:25, 1] = 0.010
        valid = np.ones((40, 2), dtype=bool)
        episodes = np.zeros((40, 2), dtype=np.int64)
        active, onset = persistent_threshold_events(spread, valid, episodes, 0.010, 20)
        self.assertFalse(active[23, 1])
        self.assertTrue(active[24, 1])
        self.assertTrue(onset[24, 1])
        balanced = np.full((40, 2), 0.0099)
        active_balanced, _ = persistent_threshold_events(balanced, valid, episodes, 0.010, 20)
        self.assertFalse(active_balanced.any())

    def test_episode_change_resets_persistence(self) -> None:
        spread = np.full((30, 2), 0.012)
        valid = np.ones((30, 2), dtype=bool)
        episodes = np.zeros((30, 2), dtype=np.int64)
        episodes[10:, 0] = 1
        active, onset = persistent_threshold_events(spread, valid, episodes, 0.010, 20)
        self.assertFalse(active[:29, 0].any())
        self.assertTrue(onset[29, 0])


class EventDatasetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        cls.dense = yaml.safe_load(DENSE_CONFIG.read_text(encoding="utf-8"))

    def test_design_is_run_disjoint_and_holdout_is_fresh(self) -> None:
        primary, controls = generate_event_specifications(self.document, self.dense)
        design = validate_event_design(self.document, self.dense, primary, controls)
        self.assertEqual(design["split_counts"], {"train": 144, "validation": 48, "holdout": 48})
        self.assertEqual(design["duplicate_signatures"], 0)
        self.assertEqual(design["holdout_historical_dense_signature_overlap"], 0)
        self.assertEqual(len({physical_signature(row) for row in primary}), 240)

    def test_exact_sensor_candidates_exclude_terrain_fall_and_oracle_fields(self) -> None:
        self.assertEqual(EVENT_REPRESENTATIONS, (PELVIS_IMU6, PELVIS_IMU6_FSR8))
        self.assertEqual(EVENT_CLASS_NAMES, ("NORMAL", "REFLEX_EVENT"))
        forbidden = ("terrain", "fall", "time_to", "slip", "support", "event")
        for names in self.document["representations"].values():
            order = tuple(names["feature_order"])
            self.assertFalse(any(token in name for token in forbidden for name in order))

    def test_fall_outcome_does_not_change_labels_or_tensors(self) -> None:
        stable = synthetic_run("stable", event_sample=250, event_type=EVENT_TYPE_SUPPORT)
        fall = replace(stable, run_id="fall", outcome_diagnostic=VALID_FALL, fall_sample_diagnostic=450)
        normalizer = Normalizer(np.zeros(6, np.float32), np.ones(6, np.float32), 1, ("train",), 1e-8)
        batches = [
            build_event_windows({run.run_id: run, "negative": synthetic_run("negative", event_sample=None, event_type=EVENT_TYPE_NONE)}, [run.run_id, "negative"], PELVIS_IMU6, 20, normalizer)
            for run in (stable, fall)
        ]
        self.assertTrue(np.array_equal(batches[0].windows.targets, batches[1].windows.targets))
        self.assertTrue(np.array_equal(batches[0].windows.inputs, batches[1].windows.inputs))

    def test_window_bounds_stride_cap_and_pre_event_negatives(self) -> None:
        event = synthetic_run("event", event_sample=250, event_type=EVENT_TYPE_SUPPORT)
        positive = event_positive_endpoints(event, 20)
        negative = event_early_negative_endpoints(event, 20, len(positive))
        self.assertTrue(np.array_equal(np.diff(positive), np.full(len(positive) - 1, 5)))
        self.assertEqual((positive[0] - 250, positive[-1] - 250), (-10, 50))
        self.assertLessEqual(int(negative[-1]), 220)
        self.assertLessEqual(len(positive), 13)

    def test_holdout_guard(self) -> None:
        guard = EventHoldoutGuard()
        with self.assertRaises(RuntimeError):
            guard.require_open()
        guard.open_once()
        guard.require_open()
        with self.assertRaises(RuntimeError):
            guard.open_once()


class EventRuntimeEvaluationTests(unittest.TestCase):
    def test_five_ms_runtime_persistence_reports_confirmation_sample(self) -> None:
        endpoints = np.arange(100, 112, dtype=np.int64)
        probabilities = np.zeros(len(endpoints), dtype=np.float64)
        probabilities[3:8] = 0.8
        self.assertEqual(
            sustained_confirmation_sample(endpoints, probabilities, 0.5, 5),
            107,
        )

    def test_valid_premature_late_and_no_event_classification(self) -> None:
        self.assertEqual(classify_event_detection(100, 80), "EVENT_VALID_DETECTION")
        self.assertEqual(classify_event_detection(100, 79), "EVENT_PREMATURE_FP")
        self.assertEqual(classify_event_detection(100, 150), "EVENT_VALID_DETECTION")
        self.assertEqual(classify_event_detection(100, 151), "EVENT_LATE")
        self.assertEqual(classify_event_detection(None, 100), "NO_EVENT_FP")
        self.assertEqual(classify_event_detection(None, None), "NO_EVENT_TN")

    def test_threshold_selection_uses_frozen_false_alarm_feasibility(self) -> None:
        def metrics(recall: float, fp: float, hard: float, premature: float) -> dict[str, object]:
            return {
                "overall_event_recall": recall,
                "slip_event_recall": recall,
                "support_event_recall": recall,
                "no_event_transition_fp_rate": fp,
                "hard_ground_fp_rate": hard,
                "premature_event_run_fp_rate": premature,
                "latency_ms": {"p95": 20.0},
            }
        selection = select_event_threshold(
            [
                {"threshold": 0.2, "metrics": metrics(1.0, 0.2, 0.0, 0.0)},
                {"threshold": 0.4, "metrics": metrics(0.9, 0.1, 0.0, 0.1)},
            ],
            {"no_event_transition_fp_rate_max": 0.1, "hard_ground_fp_rate_max": 0.05, "premature_event_run_fp_rate_max": 0.1},
        )
        self.assertEqual(selection["selected"]["threshold"], 0.4)

    def test_sensor_near_tie_prefers_imu6(self) -> None:
        holdout = {
            PELVIS_IMU6: {"passed": True, "metrics": {"overall_event_recall": 0.90, "latency_ms": {"p95": 20.0}}},
            PELVIS_IMU6_FSR8: {"passed": True, "metrics": {"overall_event_recall": 0.92, "latency_ms": {"p95": 25.0}}},
        }
        result = _selection_recommendation({}, holdout, {"recall_difference_max": 0.03, "p95_latency_difference_ms_max": 10})
        self.assertEqual(result["representation"], PELVIS_IMU6)
        self.assertTrue(result["near_tie"])

    def test_run_level_event_no_event_and_hard_false_positive_metrics(self) -> None:
        event = synthetic_run("event", event_sample=200, event_type=EVENT_TYPE_SUPPORT)
        no_event = synthetic_run("no_event", event_sample=None, event_type=EVENT_TYPE_NONE)
        hard = replace(
            synthetic_run("hard", event_sample=None, event_type=EVENT_TYPE_NONE, target="concrete", control=True),
            hard_stable_control=True,
        )
        endpoints = np.arange(100, 300, dtype=np.int64)
        event_probability = np.zeros(len(endpoints))
        event_probability[84:89] = 1.0  # confirmation 188, latency -12 ms
        false_probability = np.zeros(len(endpoints))
        false_probability[100:105] = 1.0
        metrics = evaluate_event_runs(
            {run.run_id: run for run in (event, no_event, hard)},
            {
                "event": ReplayTrace(endpoints, event_probability),
                "no_event": ReplayTrace(endpoints, false_probability),
                "hard": ReplayTrace(endpoints, false_probability),
            },
            0.5,
            5,
        )
        self.assertEqual(metrics["overall_event_recall"], 1.0)
        self.assertEqual(metrics["no_event_transition_specificity"], 0.0)
        self.assertEqual(metrics["hard_ground_specificity"], 0.0)
        self.assertEqual(metrics["premature_event_run_fp_rate"], 0.0)

    def test_terrain_hash_and_fusion_regressions(self) -> None:
        document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        paths = document["terrain_regression"]["protected_paths"]
        before = _protected_hashes(ROOT, paths)
        after = _protected_hashes(ROOT, paths)
        self.assertEqual(before, after)
        self.assertTrue(fusion_regression()["passed"])

    def test_canonical_cli_dispatch_and_viewer_contract_remain_present(self) -> None:
        script = (ROOT / "scripts/fastreflex.py").read_text(encoding="utf-8")
        simulation_tests = (ROOT / "tests/test_simulation.py").read_text(encoding="utf-8")
        self.assertIn("EVENT_CENTRIC_REFLEX_TRIGGER_DEVELOPMENT", script)
        self.assertIn("test_deformable_support_runtime_and_viewer_physics_parity", simulation_tests)


if __name__ == "__main__":
    unittest.main()
