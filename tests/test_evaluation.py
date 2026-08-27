from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from fastreflex.dataset.loader import Normalizer, sha256_file
from fastreflex.evaluation.time_to_separation import (
    audit_false_positives,
    causal_window_indices,
    extract_event_samples,
    first_sustained_endpoint,
    horizon_detected,
    load_and_verify_replay_contract,
    pre_degradation_detected,
    ReplayTrace,
    replay_causal,
)
from scripts.fastreflex import DEFAULT_EVALUATION_CONFIG, build_parser


class EndpointModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        endpoint = inputs[:, -1, 0]
        return torch.stack((endpoint, -endpoint, torch.zeros_like(endpoint)), dim=1)


class TimeToSeparationTest(unittest.TestCase):
    def test_evaluate_cli_uses_canonical_config(self) -> None:
        args = build_parser().parse_args(["evaluate"])
        self.assertEqual(args.command, "evaluate")
        self.assertEqual(args.config.resolve(), DEFAULT_EVALUATION_CONFIG)

    def test_causal_windows_have_no_future_and_one_ms_endpoints(self) -> None:
        endpoints, indices = causal_window_indices(8, 4, 1)
        np.testing.assert_array_equal(endpoints, [3, 4, 5, 6, 7])
        np.testing.assert_array_equal(indices[0], [0, 1, 2, 3])
        np.testing.assert_array_equal(indices[-1], [4, 5, 6, 7])
        self.assertTrue(np.all(indices[:, -1] == endpoints))
        self.assertTrue(np.all(indices <= endpoints[:, None]))

    def test_replay_endpoint_alignment_is_one_ms(self) -> None:
        imu = np.zeros((8, 6), dtype=np.float32)
        imu[:, 0] = np.arange(8)
        normalizer = Normalizer(
            mean=np.zeros(6, dtype=np.float32),
            std=np.ones(6, dtype=np.float32),
            sample_count=8,
            fit_run_ids=("train",),
            epsilon=1.0e-8,
        )
        trace = replay_causal(EndpointModel(), imu, normalizer, 4, batch_size=2)
        np.testing.assert_array_equal(trace.endpoint_samples, [3, 4, 5, 6, 7])
        np.testing.assert_array_equal(trace.logits[:, 0], [3, 4, 5, 6, 7])

    def test_event_alignment_and_zero_margin_sink(self) -> None:
        arrays = {
            "pelvis_imu": np.zeros((200, 6), dtype=np.float32),
            "first_patch_contact_sample_per_foot": np.asarray([50, -1]),
            "first_any_slip_onset_sample": np.asarray(80),
            "first_sink_physical_onset_sample_per_foot": np.asarray([90, -1]),
            "first_sink_degradation_onset_sample": np.asarray(90),
            "first_censor_sample": np.asarray(150),
        }
        slip = extract_event_samples(arrays, "SLIP")
        self.assertEqual((slip.t0, slip.t1, slip.t2, slip.t3), (50, 80, None, 150))
        sink = extract_event_samples(arrays, "SINK")
        self.assertEqual((sink.t0, sink.t1, sink.t2, sink.t3), (50, 90, 90, 150))
        self.assertIsNone(pre_degradation_detected(95, 50, 90, 90, 20))

    def test_sustained_correct_uses_tenth_confirmation_endpoint(self) -> None:
        endpoints = np.arange(100, 130)
        predictions = np.zeros(30, dtype=np.int8)
        predictions[5:15] = 1
        confirmed = first_sustained_endpoint(
            predictions, endpoints, 1, 100, 130, persistence_samples=10
        )
        self.assertEqual(confirmed, 114)

    def test_pre_event_false_positive_audit(self) -> None:
        endpoints = np.arange(100, 130)
        predictions = np.zeros(30, dtype=np.int8)
        predictions[2:12] = 2
        logits = np.zeros((30, 3), dtype=np.float32)
        probabilities = np.full((30, 3), 1 / 3, dtype=np.float32)
        trace = ReplayTrace(endpoints, logits, probabilities, predictions)
        audit = audit_false_positives(trace, 100, 120, 10)
        self.assertTrue(audit["sustained_any_hazard"])
        self.assertFalse(audit["sustained_slip"])
        self.assertTrue(audit["sustained_sink"])
        self.assertEqual(audit["sink_window_count"], 10)

    def test_horizon_recall_boundary(self) -> None:
        self.assertTrue(horizon_detected(120, 100, 200, 20))
        self.assertFalse(horizon_detected(121, 100, 200, 20))
        self.assertTrue(pre_degradation_detected(120, 50, 100, 140, 20))
        self.assertFalse(pre_degradation_detected(141, 50, 100, 140, 100))

    def test_frozen_contract_hashes_are_verified_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for name, content in (
                ("split.json", b"split"),
                ("normalization.json", b"normalization"),
                ("metrics.json", b"metrics"),
                ("checkpoint.pt", b"checkpoint"),
            ):
                path = root / name
                path.write_bytes(content)
                files[name] = path
            config = {
                "experiment": {"id": "TIME_TO_SEPARATION"},
                "dataset": {"path": "dataset"},
                "first_poc": {
                    "split": {
                        "path": "split.json",
                        "sha256": sha256_file(files["split.json"]),
                    },
                    "normalization": {
                        "path": "normalization.json",
                        "sha256": sha256_file(files["normalization.json"]),
                    },
                    "selection_metrics": {
                        "path": "metrics.json",
                        "sha256": sha256_file(files["metrics.json"]),
                    },
                },
                "primary_model": {
                    "checkpoints": [
                        {
                            "seed": 1,
                            "path": "checkpoint.pt",
                            "sha256": sha256_file(files["checkpoint.pt"]),
                        }
                    ]
                },
                "artifacts": {"path": "artifacts"},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            before = {name: path.read_bytes() for name, path in files.items()}
            _, resolved = load_and_verify_replay_contract(config_path, root)
            self.assertEqual(resolved["checkpoint_1"], files["checkpoint.pt"])
            self.assertEqual(
                before, {name: path.read_bytes() for name, path in files.items()}
            )
            files["checkpoint.pt"].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint.*SHA-256"):
                load_and_verify_replay_contract(config_path, root)


if __name__ == "__main__":
    unittest.main()
