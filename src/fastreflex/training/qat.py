"""Deployment-aware QAT for the frozen GRU20 engineering derivative."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from fastreflex.dataset.hazard import canonical_sha256
from fastreflex.dataset.loader import WindowSet, sha256_file
from fastreflex.evaluation.hazard import load_hazard_normalizer
from fastreflex.features import feature_schema_hash
from fastreflex.models.baselines import parameter_count
from fastreflex.models.checkpoint import load_checkpoint
from fastreflex.training.hazard import (
    EVENT_CLASS_NAMES,
    _merge_endpoint_maps,
    _mine_training_round,
    _train_monitor_partition,
    _training_recipe,
    build_hazard_windows,
    model_v2_anchor_refined_positive_plan,
    prepare_model_v2_training_data,
)
from fastreflex.training.trainer import set_deterministic


QAT_EXPERIMENT_ID = "DEPLOYMENT_AWARE_QAT"
QAT_ROLE = "DEPLOYMENT_QAT_ENGINEERING_CANDIDATE"
SOURCE_CANDIDATE_ID = "model_v2_anchor_refined_gru20_20260902"
IMMUTABLE_SCIENTIFIC_VERDICT = "MODEL_V2_GENERALIZATION_HOLDOUT_NOT_SUPPORTED"
PROTECTED_SOURCES = frozenset(
    {
        "Generalization_HOLDOUT",
        "Generalization_VALIDATION",
        "Unified_HOLDOUT",
        "consumed_FACTOR_VALIDATION",
        "protected_Generalization_payload",
        "BOUNDARY_RESOLUTION_VALIDATION",
        "failed_boundary_resolution_payload",
        "sand_scientific_intervention_corpora",
    }
)
STATE_KEYS = (
    "gru.weight_ih_l0",
    "gru.weight_hh_l0",
    "gru.bias_ih_l0",
    "gru.bias_hh_l0",
    "classifier.weight",
    "classifier.bias",
)


@dataclass(frozen=True)
class QuantizationSpec:
    """One fixed affine fake-quant tensor contract."""

    scale: float
    zero_point: int
    qmin: int = -128
    qmax: int = 127

    def to_dict(self) -> dict[str, float | int]:
        return {
            "scale": self.scale,
            "zero_point": self.zero_point,
            "qmin": self.qmin,
            "qmax": self.qmax,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> QuantizationSpec:
        return cls(
            scale=float(value["scale"]),
            zero_point=int(value["zero_point"]),
            qmin=int(value["qmin"]),
            qmax=int(value["qmax"]),
        )


class _MinMaxObservers:
    def __init__(self) -> None:
        self.ranges: dict[str, list[float]] = {}

    def add(self, name: str, value: torch.Tensor) -> None:
        minimum = float(torch.amin(value.detach()).cpu())
        maximum = float(torch.amax(value.detach()).cpu())
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise RuntimeError(f"nonfinite QAT observer value: {name}")
        row = self.ranges.setdefault(name, [0.0, 0.0])
        row[0] = min(row[0], minimum)
        row[1] = max(row[1], maximum)

    def affine_specs(self) -> dict[str, QuantizationSpec]:
        return {
            name: _affine_spec(minimum, maximum)
            for name, (minimum, maximum) in sorted(self.ranges.items())
        }


def _affine_spec(minimum: float, maximum: float) -> QuantizationSpec:
    minimum = min(float(minimum), 0.0)
    maximum = max(float(maximum), 0.0)
    if maximum <= minimum:
        return QuantizationSpec(scale=1.0, zero_point=0)
    scale = (maximum - minimum) / 255.0
    zero_point = int(round(-128.0 - minimum / scale))
    zero_point = min(127, max(-128, zero_point))
    return QuantizationSpec(scale=scale, zero_point=zero_point)


def _fake_quant(value: torch.Tensor, spec: QuantizationSpec) -> torch.Tensor:
    return torch.fake_quantize_per_tensor_affine(
        value,
        scale=spec.scale,
        zero_point=spec.zero_point,
        quant_min=spec.qmin,
        quant_max=spec.qmax,
    )


def _fake_quant_weight(
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scales = torch.amax(torch.abs(value.detach()), dim=1).clamp_min(1.0e-12) / 127.0
    zero_points = torch.zeros(len(value), dtype=torch.int32, device=value.device)
    quantized = torch.fake_quantize_per_channel_affine(
        value,
        scales,
        zero_points,
        axis=0,
        quant_min=-127,
        quant_max=127,
    )
    return quantized, scales


def _fake_quant_bias(value: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    safe = scales.detach().clamp_min(1.0e-20)
    dequantized = torch.round(value / safe) * safe
    return value + (dequantized - value).detach()


def _json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _git_output(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_qat_protocol(document: Mapping[str, object]) -> None:
    """Reject scope expansion or protected evidence before any model operation."""
    if document.get("experiment", {}).get("id") != QAT_EXPERIMENT_ID:
        raise ValueError("unsupported deployment-aware QAT config")
    candidate = document.get("candidate", {})
    if (
        candidate.get("role") != QAT_ROLE
        or candidate.get("source_candidate_id") != SOURCE_CANDIDATE_ID
        or candidate.get("scientific_candidate") is not False
        or candidate.get("scientific_release") is not False
        or candidate.get("generalization_supported") is not False
    ):
        raise ValueError("QAT engineering/scientific role boundary changed")
    prohibited = set(document.get("scientific_boundary", {}).get("prohibited_data_sources", ()))
    if prohibited != PROTECTED_SOURCES:
        raise ValueError("QAT protected-data rejection list changed")
    operations = document.get("scientific_boundary", {}).get(
        "historical_holdout_operations", {}
    )
    if not operations or any(int(value) != 0 for value in operations.values()):
        raise ValueError("QAT config authorizes historical HOLDOUT access")
    allowed = set(document.get("data", {}).get("allowed_sources", ()))
    if allowed & PROTECTED_SOURCES or allowed != {
        "Unified_TRAIN",
        "valid_V2_TRAIN",
        "frozen_TRAIN_monitor",
        "frozen_TRAIN_derived_int8_calibration_windows",
    }:
        raise ValueError("QAT allowed data sources changed")
    source = document.get("source_contract", {})
    architecture = source.get("architecture", {})
    if (
        architecture.get("model_family") != "gru"
        or int(architecture.get("input_size", 0)) != 80
        or int(architecture.get("hidden_size", 0)) != 32
        or int(architecture.get("layers", 0)) != 1
        or architecture.get("bidirectional") is not False
        or float(architecture.get("dropout", -1.0)) != 0.0
        or int(architecture.get("parameters_per_member", 0)) != 11_010
    ):
        raise ValueError("QAT architecture contract changed")
    runtime = source.get("runtime", {})
    if (
        runtime.get("ensemble_seeds") != [20260828, 20260829, 20260830]
        or float(runtime.get("threshold", -1.0)) != 0.99
        or runtime.get("comparison") != "greater_than_or_equal"
        or int(runtime.get("persistence_ms", 0)) != 5
    ):
        raise ValueError("QAT runtime decision contract changed")
    fake = document.get("fake_quantization", {})
    if (
        fake.get("implementation")
        != "explicit_unrolled_pytorch_reset_after_gru_with_ste"
        or fake.get("equation") != "n_plus_z_times_h_minus_n"
        or int(fake.get("timesteps", 0)) != 20
        or int(fake.get("projection_partition", {}).get("block_width", 0)) != 16
        or fake.get("recurrent_hidden", {}).get("feedback_uses_fake_quantized_hidden")
        is not True
    ):
        raise ValueError("QAT recurrent fake-quant contract changed")
    if document.get("objective", {}).get("scientific_loss_redesign") is not False:
        raise ValueError("scientific loss redesign is prohibited in QAT")
    if document.get("objective", {}).get("within_class_domain_balanced_loss_mass") is not False:
        raise ValueError("Sand-boundary loss is prohibited in QAT")


def _state(model: nn.Module) -> dict[str, torch.Tensor]:
    state = dict(model.named_parameters())
    if tuple(state) != STATE_KEYS:
        raise ValueError("frozen GRU state layout changed")
    return state


def explicit_float_forward(
    model: nn.Module, inputs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute the exact reset-after equation and expose projection/hidden trace."""
    state = _state(model)
    projected = F.linear(
        inputs, state["gru.weight_ih_l0"], state["gru.bias_ih_l0"]
    )
    input_reset, input_update, input_new = projected.chunk(3, dim=2)
    hidden = torch.zeros((len(inputs), 32), dtype=inputs.dtype, device=inputs.device)
    trace: list[torch.Tensor] = []
    for timestep in range(20):
        hidden_reset, hidden_update, hidden_new = F.linear(
            hidden, state["gru.weight_hh_l0"], state["gru.bias_hh_l0"]
        ).chunk(3, dim=1)
        reset = torch.sigmoid(input_reset[:, timestep] + hidden_reset)
        update = torch.sigmoid(input_update[:, timestep] + hidden_update)
        new = torch.tanh(input_new[:, timestep] + reset * hidden_new)
        hidden = new + update * (hidden - new)
        trace.append(hidden)
    logits = F.linear(
        hidden, state["classifier.weight"], state["classifier.bias"]
    )
    return logits, torch.stack(trace, dim=1), projected


def collect_activation_specs(
    model: nn.Module,
    calibration_windows: np.ndarray,
    *,
    batch_size: int = 128,
    input_bound: float = 4.132843623161313,
) -> tuple[dict[str, QuantizationSpec], dict[str, object]]:
    """Calibrate fixed per-op affine ranges with one TRAIN-only Float pass."""
    if calibration_windows.shape != (2597, 20, 80):
        raise ValueError("QAT calibration artifact shape changed")
    state = _state(model)
    observer = _MinMaxObservers()
    with torch.no_grad():
        for first in range(0, len(calibration_windows), batch_size):
            inputs = torch.from_numpy(
                np.clip(
                    calibration_windows[first : first + batch_size],
                    -input_bound,
                    input_bound,
                ).astype(np.float32, copy=False)
            )
            input_blocks: list[torch.Tensor] = []
            for block in range(6):
                start = block * 16
                stop = start + 16
                value = F.linear(
                    inputs,
                    state["gru.weight_ih_l0"][start:stop],
                    state["gru.bias_ih_l0"][start:stop],
                )
                observer.add(f"input_projection_b{block}", value)
                input_blocks.append(value)
            hidden = torch.zeros((len(inputs), 32), dtype=torch.float32)
            for timestep in range(20):
                hidden_blocks: list[torch.Tensor] = []
                for block in range(6):
                    start = block * 16
                    stop = start + 16
                    value = F.linear(
                        hidden,
                        state["gru.weight_hh_l0"][start:stop],
                        state["gru.bias_hh_l0"][start:stop],
                    )
                    observer.add(f"hidden_projection_t{timestep}_b{block}", value)
                    hidden_blocks.append(value)
                next_hidden: list[torch.Tensor] = []
                for block in range(2):
                    reset_preact = (
                        input_blocks[block][:, timestep]
                        + hidden_blocks[block]
                    )
                    update_preact = (
                        input_blocks[2 + block][:, timestep]
                        + hidden_blocks[2 + block]
                    )
                    observer.add(f"reset_preact_t{timestep}_b{block}", reset_preact)
                    observer.add(f"update_preact_t{timestep}_b{block}", update_preact)
                    reset = torch.sigmoid(reset_preact)
                    update = torch.sigmoid(update_preact)
                    observer.add(f"reset_gate_t{timestep}_b{block}", reset)
                    observer.add(f"update_gate_t{timestep}_b{block}", update)
                    reset_hidden = reset * hidden_blocks[4 + block]
                    observer.add(
                        f"reset_hidden_new_t{timestep}_b{block}", reset_hidden
                    )
                    candidate_preact = (
                        input_blocks[4 + block][:, timestep] + reset_hidden
                    )
                    observer.add(
                        f"candidate_preact_t{timestep}_b{block}", candidate_preact
                    )
                    candidate = torch.tanh(candidate_preact)
                    observer.add(f"candidate_gate_t{timestep}_b{block}", candidate)
                    start = block * 16
                    previous = hidden[:, start : start + 16]
                    difference = previous - candidate
                    observer.add(
                        f"hidden_difference_t{timestep}_b{block}", difference
                    )
                    delta = update * difference
                    observer.add(f"update_delta_t{timestep}_b{block}", delta)
                    value = candidate + delta
                    observer.add(f"hidden_block_t{timestep}_b{block}", value)
                    next_hidden.append(value)
                hidden = torch.cat(next_hidden, dim=1)
                observer.add(f"hidden_state_t{timestep}", hidden)
            logits = F.linear(
                hidden, state["classifier.weight"], state["classifier.bias"]
            )
            observer.add("classifier_logits", logits)
            observer.add("softmax_probabilities", torch.softmax(logits, dim=1))
    specs = observer.affine_specs()
    expected_count = 6 + 20 * (6 + 2 * 10 + 1) + 2
    if len(specs) != expected_count:
        raise RuntimeError(
            f"QAT observer topology changed: {len(specs)} != {expected_count}"
        )
    ranges = {
        name: {"minimum": values[0], "maximum": values[1], **specs[name].to_dict()}
        for name, values in sorted(observer.ranges.items())
    }
    return specs, {
        "schema_version": 1,
        "observer": "model_blind_minmax_one_pass",
        "source": "frozen_train_derived_int8_calibration_windows",
        "window_count": len(calibration_windows),
        "input_clip_bound": input_bound,
        "observer_update_during_training": False,
        "tensor_count": len(specs),
        "ranges": ranges,
    }


class DeploymentAwareQATGRU(nn.Module):
    """State-compatible GRU20 with an explicit E84-aligned fake-quant forward."""

    def __init__(
        self,
        source: nn.Module,
        activation_specs: Mapping[str, QuantizationSpec],
        *,
        input_scale: float = 0.03241445869207382,
        input_zero_point: int = 0,
    ) -> None:
        super().__init__()
        if parameter_count(source) != 11_010:
            raise ValueError("QAT source architecture changed")
        copied = deepcopy(source)
        self.gru = copied.gru
        self.classifier = copied.classifier
        self.activation_specs = dict(activation_specs)
        self.input_spec = QuantizationSpec(input_scale, input_zero_point)
        self.sigmoid_spec = QuantizationSpec(0.00390625, -128)
        self.tanh_spec = QuantizationSpec(0.0078125, 0)
        self.softmax_spec = QuantizationSpec(0.00390625, -128)

    def _q(self, name: str, value: torch.Tensor) -> torch.Tensor:
        try:
            spec = self.activation_specs[name]
        except KeyError as exc:
            raise RuntimeError(f"missing frozen QAT activation spec: {name}") from exc
        return _fake_quant(value, spec)

    def forward_with_hidden(
        self, inputs: torch.Tensor, *, fake_quant: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not fake_quant:
            return explicit_float_forward(self, inputs)
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (20, 80):
            raise ValueError("QAT input must have shape [batch,20,80]")
        state = _state(self)
        values = _fake_quant(inputs, self.input_spec)
        weight_ih, scale_ih = _fake_quant_weight(state["gru.weight_ih_l0"])
        weight_hh, scale_hh = _fake_quant_weight(state["gru.weight_hh_l0"])
        classifier_weight, classifier_scale = _fake_quant_weight(
            state["classifier.weight"]
        )
        input_blocks: list[torch.Tensor] = []
        for block in range(6):
            start = block * 16
            stop = start + 16
            bias = _fake_quant_bias(
                state["gru.bias_ih_l0"][start:stop],
                self.input_spec.scale * scale_ih[start:stop],
            )
            projected = F.linear(values, weight_ih[start:stop], bias)
            input_blocks.append(self._q(f"input_projection_b{block}", projected))
        hidden = torch.zeros((len(inputs), 32), dtype=inputs.dtype, device=inputs.device)
        trace: list[torch.Tensor] = []
        for timestep in range(20):
            hidden_input_scale = (
                1.0 / 127.0
                if timestep == 0
                else self.activation_specs[f"hidden_state_t{timestep - 1}"].scale
            )
            hidden_blocks: list[torch.Tensor] = []
            for block in range(6):
                start = block * 16
                stop = start + 16
                bias = _fake_quant_bias(
                    state["gru.bias_hh_l0"][start:stop],
                    hidden_input_scale * scale_hh[start:stop],
                )
                projected = F.linear(hidden, weight_hh[start:stop], bias)
                hidden_blocks.append(
                    self._q(f"hidden_projection_t{timestep}_b{block}", projected)
                )
            next_hidden: list[torch.Tensor] = []
            for block in range(2):
                reset_preact = self._q(
                    f"reset_preact_t{timestep}_b{block}",
                    input_blocks[block][:, timestep] + hidden_blocks[block],
                )
                update_preact = self._q(
                    f"update_preact_t{timestep}_b{block}",
                    input_blocks[2 + block][:, timestep]
                    + hidden_blocks[2 + block],
                )
                reset = _fake_quant(torch.sigmoid(reset_preact), self.sigmoid_spec)
                update = _fake_quant(torch.sigmoid(update_preact), self.sigmoid_spec)
                reset_hidden = self._q(
                    f"reset_hidden_new_t{timestep}_b{block}",
                    reset * hidden_blocks[4 + block],
                )
                candidate_preact = self._q(
                    f"candidate_preact_t{timestep}_b{block}",
                    input_blocks[4 + block][:, timestep] + reset_hidden,
                )
                candidate = _fake_quant(torch.tanh(candidate_preact), self.tanh_spec)
                start = block * 16
                difference = self._q(
                    f"hidden_difference_t{timestep}_b{block}",
                    hidden[:, start : start + 16] - candidate,
                )
                delta = self._q(
                    f"update_delta_t{timestep}_b{block}", update * difference
                )
                value = self._q(
                    f"hidden_block_t{timestep}_b{block}", candidate + delta
                )
                next_hidden.append(value)
            hidden = self._q(
                f"hidden_state_t{timestep}", torch.cat(next_hidden, dim=1)
            )
            trace.append(hidden)
        classifier_bias = _fake_quant_bias(
            state["classifier.bias"],
            self.activation_specs["hidden_state_t19"].scale * classifier_scale,
        )
        logits = self._q(
            "classifier_logits",
            F.linear(hidden, classifier_weight, classifier_bias),
        )
        return logits, torch.stack(trace, dim=1), torch.cat(input_blocks, dim=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_hidden(inputs, fake_quant=True)[0]

    def probabilities(self, inputs: torch.Tensor, *, fake_quant: bool) -> torch.Tensor:
        logits = self.forward_with_hidden(inputs, fake_quant=fake_quant)[0]
        probability = torch.softmax(logits, dim=1)
        return _fake_quant(probability, self.softmax_spec) if fake_quant else probability


def _verified_file(root: Path, record: Mapping[str, object]) -> Path:
    path = root / str(record["path"])
    if not path.is_file() or sha256_file(path) != str(record["sha256"]):
        raise RuntimeError(f"frozen QAT source changed: {record['path']}")
    return path


def _load_calibration(root: Path, document: Mapping[str, object]) -> np.ndarray:
    path = _verified_file(root, document["data"]["calibration"])
    with np.load(path, allow_pickle=False) as payload:
        windows = payload["model_windows"].copy()
    if windows.shape != (2597, 20, 80) or windows.dtype != np.float32:
        raise RuntimeError("QAT calibration tensor contract changed")
    return windows


def _source_training_document(
    root: Path, document: Mapping[str, object]
) -> tuple[Path, dict[str, object]]:
    path = _verified_file(root, document["data"]["source_training_config"])
    return path, yaml.safe_load(path.read_text(encoding="utf-8"))


def _source_models(
    root: Path, document: Mapping[str, object]
) -> list[tuple[int, nn.Module, Path]]:
    result: list[tuple[int, nn.Module, Path]] = []
    for record in document["candidate"]["source_checkpoints"]:
        path = _verified_file(root, record)
        model, metadata = load_checkpoint(path)
        seed = int(record["seed"])
        if (
            int(metadata["seed"]) != seed
            or metadata["family"] != "gru"
            or int(metadata["window_samples"]) != 20
            or int(metadata["input_channels"]) != 80
            or metadata["class_names"] != list(EVENT_CLASS_NAMES)
            or parameter_count(model) != 11_010
        ):
            raise RuntimeError(f"QAT source checkpoint contract changed: {seed}")
        result.append((seed, model, path))
    if [row[0] for row in result] != [20260828, 20260829, 20260830]:
        raise RuntimeError("QAT source ensemble membership changed")
    return result


def _load_training_windows(
    root: Path, document: Mapping[str, object]
) -> tuple[WindowSet, WindowSet, dict[str, object]]:
    _, source_document = _source_training_document(root, document)
    data = prepare_model_v2_training_data(root, source_document)
    normalizer_path = _verified_file(
        root, document["source_contract"]["preprocessing"]["normalizer"]
    )
    normalizer = load_hazard_normalizer(normalizer_path)
    fit_ids, monitor_ids = _train_monitor_partition(
        data.runs, sorted(data.runs), data.precursor_samples
    )
    fit_positive, monitor_positive = model_v2_anchor_refined_positive_plan(
        data.runs, data.precursor_samples, data.annotations
    )
    source_manifest = json.loads(
        (root / "artifacts/releases/model_v2_anchor_refined_gru20_20260902/model_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    freeze_record = source_manifest["provenance"]["candidate_freeze"]
    freeze_path = _verified_file(root, freeze_record)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    training_path = root / "artifacts/runs/20260902_model_v2_anchor_refined_training/training_record.json"
    if sha256_file(training_path) != str(freeze["training_record_sha256"]):
        raise RuntimeError("frozen source training ledger changed")
    training_record = json.loads(training_path.read_text(encoding="utf-8"))
    accumulated: dict[str, tuple[int, ...]] = {}
    reconstructed_hnm: list[dict[str, object]] = []
    for row in training_record["candidate"]["rounds"][:3]:
        round_id = int(row["round"])
        checkpoint_records = row["checkpoint_sha256"]
        checkpoint_paths: list[Path] = []
        for seed in (20260828, 20260829, 20260830):
            relative = (
                "artifacts/runs/20260902_model_v2_anchor_refined_training/"
                f"checkpoints/model_v2_anchor_refined_gru_history20_round{round_id}_"
                f"seed{seed}.pt"
            )
            path = root / relative
            if sha256_file(path) != str(checkpoint_records[relative]):
                raise RuntimeError("frozen source HNM checkpoint changed")
            checkpoint_paths.append(path)
        selected, reconstruction = _mine_training_round(
            data.runs,
            data.precursor_samples,
            normalizer,
            checkpoint_paths,
            accumulated,
            data.annotations,
        )
        expected = row["hard_negative_mining"]
        identity_fields = (
            "selected_endpoint_sha256",
            "mined_windows",
            "runs_contributing",
            "duplicate_mined_windows",
            "spacing_violations",
            "forbidden_mask_violations",
        )
        if any(reconstruction[name] != expected[name] for name in identity_fields):
            raise RuntimeError(
                f"frozen source HNM round {round_id} identity did not reproduce: "
                f"expected {expected['selected_endpoint_sha256']}, observed "
                f"{reconstruction['selected_endpoint_sha256']}"
            )
        accumulated = _merge_endpoint_maps(accumulated, selected)
        reconstructed_hnm.append(
            {
                "round": round_id + 1,
                "selected_endpoint_sha256": reconstruction[
                    "selected_endpoint_sha256"
                ],
                "mined_windows": reconstruction["mined_windows"],
            }
        )
    recipe = _training_recipe(source_document)
    fit = build_hazard_windows(
        data.runs,
        sorted(set(fit_ids) | set(fit_positive)),
        data.precursor_samples,
        normalizer,
        extra_negative_endpoints=accumulated,
        annotations=data.annotations,
        per_category=int(recipe["initial_negative_per_gait_category"]),
        positive_cap=int(recipe["positive_cap_per_run"]),
        target_contact_cap=int(recipe["target_contact_cap_per_run"]),
        benign_precursor_cap=int(recipe["benign_precursor_cap_per_run"]),
        positive_endpoints=fit_positive,
        negative_run_ids=fit_ids,
    )
    monitor = build_hazard_windows(
        data.runs,
        sorted(set(monitor_ids) | set(monitor_positive)),
        data.precursor_samples,
        normalizer,
        annotations=data.annotations,
        per_category=int(recipe["initial_negative_per_gait_category"]),
        positive_cap=int(recipe["positive_cap_per_run"]),
        target_contact_cap=int(recipe["target_contact_cap_per_run"]),
        benign_precursor_cap=int(recipe["benign_precursor_cap_per_run"]),
        positive_endpoints=monitor_positive,
        negative_run_ids=monitor_ids,
    )
    source_round = training_record["candidate"]["rounds"][3]
    if (
        len(fit) != int(source_round["fit_windows"])
        or list(fit.selected_by_class) != source_round["fit_class_counts"]
        or len(monitor) != int(source_round["monitor_windows"])
        or list(monitor.selected_by_class) != source_round["monitor_class_counts"]
    ):
        raise RuntimeError("QAT windows differ from frozen source Round-3 endpoints")

    def identity(windows: WindowSet) -> str:
        return canonical_sha256(
            [
                f"{run_id}:{int(endpoint)}:{int(target)}"
                for run_id, endpoint, target in zip(
                    windows.run_ids, windows.endpoint_samples, windows.targets
                )
            ]
        )

    return fit, monitor, {
        "effective_train": data.composition,
        "input_audit": data.input_audit,
        "fit_windows": len(fit),
        "fit_class_counts": list(fit.selected_by_class),
        "fit_window_identity_sha256": identity(fit),
        "monitor_windows": len(monitor),
        "monitor_class_counts": list(monitor.selected_by_class),
        "monitor_window_identity_sha256": identity(monitor),
        "source_training_record": {
            "path": str(training_path.relative_to(root)),
            "sha256": sha256_file(training_path),
        },
        "source_candidate_freeze": {
            "path": str(freeze_path.relative_to(root)),
            "sha256": sha256_file(freeze_path),
        },
        "normalizer_fits": 0,
        "new_hard_negative_mining_rounds": 0,
        "frozen_source_hnm_reconstruction": reconstructed_hnm,
    }


def _load_activation_specs(path: Path) -> dict[str, QuantizationSpec]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: QuantizationSpec.from_dict(record)
        for name, record in document["ranges"].items()
    }


def _loss_terms(
    candidate: DeploymentAwareQATGRU,
    teacher: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    coefficients: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    with torch.no_grad():
        teacher_logits, teacher_hidden, _ = explicit_float_forward(teacher, inputs)
    float_logits, _, _ = candidate.forward_with_hidden(inputs, fake_quant=False)
    fake_logits, fake_hidden, _ = candidate.forward_with_hidden(inputs, fake_quant=True)
    terms = {
        "supervised_cross_entropy": F.cross_entropy(
            fake_logits, targets, weight=class_weights
        ),
        "teacher_logit_fake_smooth_l1": F.smooth_l1_loss(
            fake_logits, teacher_logits
        ),
        "teacher_logit_float_smooth_l1": F.smooth_l1_loss(
            float_logits, teacher_logits
        ),
        "teacher_hidden_fake_mse": F.mse_loss(fake_hidden, teacher_hidden),
    }
    loss = (
        coefficients["supervised_cross_entropy"]
        * terms["supervised_cross_entropy"]
        + coefficients["teacher_logit_fake_smooth_l1"]
        * terms["teacher_logit_fake_smooth_l1"]
        + coefficients["teacher_logit_float_smooth_l1"]
        * terms["teacher_logit_float_smooth_l1"]
        + coefficients["teacher_hidden_fake_mse"]
        * terms["teacher_hidden_fake_mse"]
    )
    return loss, terms


def _coefficients(document: Mapping[str, object]) -> dict[str, float]:
    terms = document["objective"]["terms"]
    return {
        "supervised_cross_entropy": float(
            terms["supervised_inverse_frequency_cross_entropy"]["coefficient"]
        ),
        "teacher_logit_fake_smooth_l1": float(
            terms["teacher_logit_smooth_l1_fake_quant"]["coefficient"]
        ),
        "teacher_logit_float_smooth_l1": float(
            terms["teacher_logit_smooth_l1_float_bypass"]["coefficient"]
        ),
        "teacher_hidden_fake_mse": float(
            terms["teacher_hidden_mean_squared_error_fake_quant"]["coefficient"]
        ),
    }


def _loader(windows: WindowSet, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(windows.inputs), torch.from_numpy(windows.targets)
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def _evaluate_objective(
    candidate: DeploymentAwareQATGRU,
    teacher: nn.Module,
    windows: WindowSet,
    class_weights: torch.Tensor,
    coefficients: Mapping[str, float],
    batch_size: int,
) -> dict[str, float]:
    candidate.eval()
    totals = {
        "composite": 0.0,
        "supervised_cross_entropy": 0.0,
        "teacher_logit_fake_smooth_l1": 0.0,
        "teacher_logit_float_smooth_l1": 0.0,
        "teacher_hidden_fake_mse": 0.0,
    }
    with torch.no_grad():
        for inputs, targets in _loader(windows, batch_size, 0, False):
            loss, terms = _loss_terms(
                candidate, teacher, inputs, targets, class_weights, coefficients
            )
            count = len(targets)
            totals["composite"] += float(loss) * count
            for name, value in terms.items():
                totals[name] += float(value) * count
    return {name: value / len(windows) for name, value in totals.items()}


def _train_member(
    source: nn.Module,
    specs: Mapping[str, QuantizationSpec],
    fit: WindowSet,
    monitor: WindowSet,
    document: Mapping[str, object],
    seed: int,
) -> tuple[DeploymentAwareQATGRU, dict[str, object]]:
    settings = document["training"]
    set_deterministic(seed)
    torch.set_num_threads(1)
    teacher = deepcopy(source).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    candidate = DeploymentAwareQATGRU(source, specs)
    counts = np.bincount(fit.targets, minlength=2).astype(np.float64)
    weights = counts.sum() / (2.0 * counts)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    optimizer = torch.optim.Adam(
        candidate.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    coefficients = _coefficients(document)
    data = _loader(fit, int(settings["batch_size"]), seed, True)
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    no_improvement = 0
    optimizer_steps = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, int(settings["maximum_epochs"]) + 1):
        candidate.train()
        totals = {
            "composite": 0.0,
            "supervised_cross_entropy": 0.0,
            "teacher_logit_fake_smooth_l1": 0.0,
            "teacher_logit_float_smooth_l1": 0.0,
            "teacher_hidden_fake_mse": 0.0,
        }
        for inputs, targets in data:
            optimizer.zero_grad(set_to_none=True)
            loss, terms = _loss_terms(
                candidate, teacher, inputs, targets, class_weights, coefficients
            )
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            count = len(targets)
            totals["composite"] += float(loss.detach()) * count
            for name, value in terms.items():
                totals[name] += float(value.detach()) * count
        train_losses = {name: value / len(fit) for name, value in totals.items()}
        monitor_losses = _evaluate_objective(
            candidate,
            teacher,
            monitor,
            class_weights,
            coefficients,
            int(settings["batch_size"]),
        )
        history.append(
            {"epoch": epoch, "fit": train_losses, "train_monitor": monitor_losses}
        )
        if monitor_losses["composite"] < best_loss:
            best_loss = monitor_losses["composite"]
            best_epoch = epoch
            best_state = deepcopy(candidate.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= int(settings["patience"]):
                break
    if best_state is None:
        raise RuntimeError("QAT produced no selectable TRAIN-monitor checkpoint")
    candidate.load_state_dict(best_state)
    candidate.eval()
    return candidate, {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "optimizer_steps": optimizer_steps,
        "selection": "minimum_composite_objective_on_frozen_train_monitor",
        "class_weights": {
            "NORMAL": float(weights[0]),
            "HAZARD_REFLEX_REQUIRED": float(weights[1]),
        },
        "loss_coefficients": coefficients,
        "best_train_monitor_composite": best_loss,
        "history": history,
    }


def _save_qat_checkpoint(
    path: Path,
    candidate: DeploymentAwareQATGRU,
    seed: int,
    best_epoch: int,
    protocol_sha256: str,
    activation_spec_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "fastreflex_raw_imu_baseline",
            "family": "gru",
            "window_samples": 20,
            "input_channels": 80,
            "class_names": list(EVENT_CLASS_NAMES),
            "seed": seed,
            "best_epoch": best_epoch,
            "state_dict": candidate.state_dict(),
            "engineering_role": QAT_ROLE,
            "source_candidate_id": SOURCE_CANDIDATE_ID,
            "qat_protocol_sha256": protocol_sha256,
            "activation_spec_sha256": activation_spec_sha256,
        },
        path,
    )


def _error_distribution(actual: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    signed = actual.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(signed)
    index = int(np.argmax(absolute))
    return {
        "maximum_absolute_error": float(absolute[index]),
        "maximum_error_index": index,
        "p95_absolute_error": float(np.percentile(absolute, 95)),
        "p50_absolute_error": float(np.percentile(absolute, 50)),
        "signed_bias": float(np.mean(signed)),
        "reference_at_maximum": float(reference.reshape(-1)[index]),
        "actual_at_maximum": float(actual.reshape(-1)[index]),
    }


def _predict(
    model: nn.Module | DeploymentAwareQATGRU,
    windows: WindowSet,
    *,
    fake_quant: bool | None,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    logits: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    inputs = torch.from_numpy(windows.inputs)
    with torch.no_grad():
        for first in range(0, len(windows), batch_size):
            batch = inputs[first : first + batch_size]
            if isinstance(model, DeploymentAwareQATGRU):
                output = model.forward_with_hidden(batch, fake_quant=bool(fake_quant))[0]
                probability = torch.softmax(output, dim=1)
                if fake_quant:
                    probability = _fake_quant(probability, model.softmax_spec)
            else:
                output = model(batch)
                probability = torch.softmax(output, dim=1)
            logits.append(output.cpu().numpy())
            probabilities.append(probability[:, 1].cpu().numpy())
    return (
        np.concatenate(logits).astype(np.float32, copy=False),
        np.concatenate(probabilities).astype(np.float64, copy=False),
    )


def _sequence_parity(
    actual: np.ndarray,
    reference: np.ndarray,
    windows: WindowSet,
    threshold: float,
    persistence: int,
) -> dict[str, object]:
    order = np.lexsort((windows.endpoint_samples, windows.run_ids))

    def states(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        crossing = probabilities >= threshold
        counts = np.zeros(len(probabilities), dtype=np.int64)
        onset = np.zeros(len(probabilities), dtype=bool)
        prior_run: str | None = None
        prior_endpoint: int | None = None
        count = 0
        for index in order:
            run_id = str(windows.run_ids[index])
            endpoint = int(windows.endpoint_samples[index])
            contiguous = run_id == prior_run and endpoint == (prior_endpoint or -2) + 1
            if not contiguous:
                count = 0
            count = count + 1 if bool(crossing[index]) else 0
            counts[index] = count
            onset[index] = count == persistence
            prior_run = run_id
            prior_endpoint = endpoint
        return crossing, counts, onset

    expected = states(reference)
    observed = states(actual)
    names = ("threshold_crossing", "persistence_count", "reflex_onset")
    return {
        name: {
            "exact": bool(np.array_equal(left, right)),
            "mismatch_count": int(np.count_nonzero(left != right)),
        }
        for name, left, right in zip(names, observed, expected)
    }


def _recurrent_error(
    teacher: nn.Module,
    candidate: DeploymentAwareQATGRU,
    window: np.ndarray,
) -> dict[str, object]:
    inputs = torch.from_numpy(window[None, ...])
    with torch.no_grad():
        teacher_logits, teacher_hidden, teacher_projection = explicit_float_forward(
            teacher, inputs
        )
        fake_logits, fake_hidden, fake_projection = candidate.forward_with_hidden(
            inputs, fake_quant=True
        )
    hidden_error = torch.amax(torch.abs(fake_hidden - teacher_hidden), dim=2)[0]
    return {
        "first_projection_maximum_absolute_error": float(
            torch.max(torch.abs(fake_projection - teacher_projection))
        ),
        "hidden_maximum_absolute_error_by_timestep": [
            float(value) for value in hidden_error
        ],
        "timestep_19_hidden_maximum_absolute_error": float(hidden_error[19]),
        "classifier_logit_maximum_absolute_error": float(
            torch.max(torch.abs(fake_logits - teacher_logits))
        ),
    }


def _train_only_audit(
    sources: Sequence[tuple[int, nn.Module, Path]],
    candidates: Sequence[DeploymentAwareQATGRU],
    baseline_wrappers: Sequence[DeploymentAwareQATGRU],
    fit: WindowSet,
    document: Mapping[str, object],
) -> dict[str, object]:
    member_reference: list[np.ndarray] = []
    baseline_fake: list[np.ndarray] = []
    candidate_float: list[np.ndarray] = []
    candidate_fake: list[np.ndarray] = []
    by_seed: dict[str, object] = {}
    for (seed, teacher, _), candidate, baseline in zip(
        sources, candidates, baseline_wrappers
    ):
        _, reference = _predict(teacher, fit, fake_quant=None, batch_size=128)
        _, original_fake = _predict(baseline, fit, fake_quant=True, batch_size=128)
        _, qat_float = _predict(candidate, fit, fake_quant=False, batch_size=128)
        _, qat_fake = _predict(candidate, fit, fake_quant=True, batch_size=128)
        member_reference.append(reference)
        baseline_fake.append(original_fake)
        candidate_float.append(qat_float)
        candidate_fake.append(qat_fake)
        maximum_index = int(np.argmax(np.abs(qat_fake - reference)))
        by_seed[str(seed)] = {
            "original_ptq_style_fake_quant_vs_reference": _error_distribution(
                original_fake, reference
            ),
            "qat_float_vs_reference": _error_distribution(qat_float, reference),
            "qat_fake_quant_vs_reference": _error_distribution(qat_fake, reference),
            "decision_parity": _sequence_parity(
                qat_fake,
                reference,
                fit,
                float(document["source_contract"]["runtime"]["threshold"]),
                int(document["source_contract"]["runtime"]["persistence_ms"]),
            ),
            "worst_window": {
                "index": maximum_index,
                "run_id": str(fit.run_ids[maximum_index]),
                "endpoint_sample": int(fit.endpoint_samples[maximum_index]),
                "recurrent_error": _recurrent_error(
                    teacher, candidate, fit.inputs[maximum_index]
                ),
            },
        }
    reference_array = np.stack(member_reference)
    baseline_array = np.stack(baseline_fake)
    float_array = np.stack(candidate_float)
    fake_array = np.stack(candidate_fake)
    reference_ensemble = np.mean(reference_array, axis=0, dtype=np.float64)
    baseline_ensemble = np.mean(baseline_array, axis=0, dtype=np.float64)
    float_ensemble = np.mean(float_array, axis=0, dtype=np.float64)
    fake_ensemble = np.mean(fake_array, axis=0, dtype=np.float64)
    baseline_ensemble_error = _error_distribution(
        baseline_ensemble, reference_ensemble
    )
    fake_ensemble_error = _error_distribution(fake_ensemble, reference_ensemble)
    contract = document["train_only_acceptance"]
    probability = contract["probability_contract"]
    improvement = contract["material_improvement"]
    member_fake_records = [
        by_seed[str(seed)]["qat_fake_quant_vs_reference"]
        for seed, _, _ in sources
    ]
    baseline_maximum = max(
        float(by_seed[str(seed)]["original_ptq_style_fake_quant_vs_reference"]["maximum_absolute_error"])
        for seed, _, _ in sources
    )
    candidate_maximum = max(
        float(record["maximum_absolute_error"]) for record in member_fake_records
    )
    checks = {
        "every_member_maximum_absolute_error": candidate_maximum
        <= float(probability["every_member_maximum_absolute_error_maximum"]),
        "every_member_p95_absolute_error": max(
            float(record["p95_absolute_error"]) for record in member_fake_records
        )
        <= float(probability["every_member_p95_absolute_error_maximum"]),
        "ensemble_maximum_absolute_error": float(
            fake_ensemble_error["maximum_absolute_error"]
        )
        <= float(probability["ensemble_maximum_absolute_error_maximum"]),
        "ensemble_p95_absolute_error": float(fake_ensemble_error["p95_absolute_error"])
        <= float(probability["ensemble_p95_absolute_error_maximum"]),
        "ensemble_absolute_bias": abs(float(fake_ensemble_error["signed_bias"]))
        <= float(probability["ensemble_absolute_bias_maximum"]),
        "worst_member_material_improvement": candidate_maximum
        <= baseline_maximum
        * float(improvement["worst_member_maximum_error_ratio_to_original_ptq_maximum"]),
        "ensemble_p95_material_improvement": float(
            fake_ensemble_error["p95_absolute_error"]
        )
        <= float(baseline_ensemble_error["p95_absolute_error"])
        * float(improvement["ensemble_p95_error_ratio_to_original_ptq_p95"]),
        "threshold_crossing_parity": all(
            record["decision_parity"]["threshold_crossing"]["exact"]
            for record in by_seed.values()
        ),
        "persistence_parity": all(
            record["decision_parity"]["persistence_count"]["exact"]
            for record in by_seed.values()
        ),
        "reflex_onset_parity": all(
            record["decision_parity"]["reflex_onset"]["exact"]
            for record in by_seed.values()
        ),
    }
    return {
        "schema_version": 1,
        "scope": "exact_effective_train_round3_fit_windows_only",
        "scientific_evidence": False,
        "actual_tflite_acceptance": False,
        "window_count": len(fit),
        "by_seed": by_seed,
        "ensemble": {
            "original_ptq_style_fake_quant_vs_reference": baseline_ensemble_error,
            "qat_float_vs_reference": _error_distribution(
                float_ensemble, reference_ensemble
            ),
            "qat_fake_quant_vs_reference": fake_ensemble_error,
            "decision_parity": _sequence_parity(
                fake_ensemble,
                reference_ensemble,
                fit,
                float(document["source_contract"]["runtime"]["threshold"]),
                int(document["source_contract"]["runtime"]["persistence_ms"]),
            ),
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "protected_data_access": False,
    }


def _protocol_commit(root: Path, config_path: Path) -> str:
    relative = str(config_path.resolve().relative_to(root.resolve()))
    if _git_output(root, "status", "--porcelain=v1", "--", relative):
        raise RuntimeError("QAT protocol config changed after its commit")
    commit = _git_output(root, "log", "-1", "--format=%H", "--", relative)
    if not commit:
        raise RuntimeError("QAT protocol config is not committed")
    return commit


def _preflight(
    root: Path,
    config_path: Path,
    document: Mapping[str, object],
    artifact_path: Path,
) -> dict[str, object]:
    validate_qat_protocol(document)
    protocol_sha = sha256_file(config_path)
    protocol_commit = _protocol_commit(root, config_path)
    if feature_schema_hash() != str(
        document["source_contract"]["preprocessing"]["feature_schema_sha256"]
    ):
        raise RuntimeError("QAT feature schema changed")
    _verified_file(root, document["candidate"]["source_release_manifest"])
    calibration = _load_calibration(root, document)
    fit, monitor, data_audit = _load_training_windows(root, document)
    sources = _source_models(root, document)
    activation_records: dict[str, object] = {}
    parity: dict[str, object] = {}
    input_bound = float(document["fake_quantization"]["input"]["source_bound"])
    for seed, source, _ in sources:
        specs, ranges = collect_activation_specs(
            source, calibration, input_bound=input_bound
        )
        range_path = artifact_path / "activation_ranges" / f"member_seed{seed}.json"
        _json_write(range_path, ranges)
        activation_records[str(seed)] = {
            "path": str(range_path.relative_to(root)),
            "sha256": sha256_file(range_path),
            "tensor_count": len(specs),
        }
        wrapper = DeploymentAwareQATGRU(source, specs)
        maximum = 0.0
        with torch.no_grad():
            for first in range(0, len(calibration), 128):
                inputs = torch.from_numpy(calibration[first : first + 128])
                expected = source(inputs)
                actual = wrapper.forward_with_hidden(inputs, fake_quant=False)[0]
                maximum = max(maximum, float(torch.max(torch.abs(actual - expected))))
        tolerance = document["fake_quantization"]["float_bypass"]
        passed = bool(
            maximum
            <= float(tolerance["parity_absolute_tolerance"])
            + float(tolerance["parity_relative_tolerance"])
            * max(1.0, float(torch.max(torch.abs(expected))))
        )
        parity[str(seed)] = {
            "maximum_absolute_logit_error": maximum,
            "passed": passed,
        }
        if not passed:
            raise RuntimeError(f"explicit Float GRU parity failed for seed {seed}")
    return {
        "schema_version": 1,
        "status": "DEPLOYMENT_AWARE_QAT_PREFLIGHT_PASS",
        "protocol": {
            "path": str(config_path.relative_to(root)),
            "sha256": protocol_sha,
            "commit": protocol_commit,
            "committed_before_optimizer_step_1": True,
        },
        "source_candidate_id": SOURCE_CANDIDATE_ID,
        "source_checkpoints": {
            str(seed): {
                "path": str(path.relative_to(root)),
                "sha256": sha256_file(path),
            }
            for seed, _, path in sources
        },
        "architecture_sha256": document["source_contract"]["architecture"][
            "architecture_sha256"
        ],
        "feature_schema_sha256": feature_schema_hash(),
        "normalizer_sha256": document["source_contract"]["preprocessing"][
            "normalizer"
        ]["sha256"],
        "calibration_sha256": document["data"]["calibration"]["sha256"],
        "data_audit": data_audit,
        "activation_ranges": activation_records,
        "float_bypass_parity": parity,
        "optimizer_steps": 0,
        "normalizer_fits": 0,
        "new_hard_negative_mining_rounds": 0,
        "protected_data_access": False,
        "generalization_holdout_operations": 0,
        "factor_validation_operations": 0,
        "boundary_resolution_payload_operations": 0,
    }


def run_deployment_aware_qat(
    root: Path,
    config_path: Path,
    *,
    dry_run: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Preflight or train the one frozen three-seed QAT candidate family."""
    root = root.resolve()
    config_path = config_path.resolve()
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifact_path = root / str(document["artifacts"]["run_path"])
    preflight_path = artifact_path / "pretraining_audit.json"
    current = _preflight(root, config_path, document, artifact_path)
    if dry_run:
        if any((artifact_path / "checkpoints").glob("*.pt")):
            raise RuntimeError("QAT optimizer artifacts already exist")
        _json_write(preflight_path, current)
        return {
            "status": current["status"],
            "pretraining_audit_sha256": sha256_file(preflight_path),
            "optimizer_steps": 0,
            "protected_data_access": False,
        }
    if not preflight_path.is_file() or json.loads(
        preflight_path.read_text(encoding="utf-8")
    ) != current:
        raise RuntimeError("run deterministic QAT dry-run before optimizer step 1")
    forbidden = (
        artifact_path / "training_ledger.json",
        artifact_path / "train_only_quantization_audit.json",
        artifact_path / "candidate_freeze.json",
        artifact_path / "behavior_preservation.json",
    )
    if any(path.exists() for path in forbidden) or any(
        (artifact_path / "checkpoints").glob("*.pt")
    ):
        raise RuntimeError("QAT training artifacts already exist")
    fit, monitor, data_audit = _load_training_windows(root, document)
    sources = _source_models(root, document)
    candidates: list[DeploymentAwareQATGRU] = []
    baseline_wrappers: list[DeploymentAwareQATGRU] = []
    records: list[dict[str, object]] = []
    checkpoint_records: list[dict[str, object]] = []
    for seed, source, _ in sources:
        range_record = current["activation_ranges"][str(seed)]
        range_path = root / str(range_record["path"])
        if sha256_file(range_path) != str(range_record["sha256"]):
            raise RuntimeError("QAT activation observer freeze changed")
        specs = _load_activation_specs(range_path)
        baseline_wrappers.append(DeploymentAwareQATGRU(source, specs).eval())
        candidate, record = _train_member(
            source, specs, fit, monitor, document, seed
        )
        checkpoint_path = (
            artifact_path / "checkpoints" / f"member_seed{seed}.pt"
        )
        _save_qat_checkpoint(
            checkpoint_path,
            candidate,
            seed,
            int(record["best_epoch"]),
            str(current["protocol"]["sha256"]),
            str(range_record["sha256"]),
        )
        candidates.append(candidate)
        records.append(record)
        checkpoint_records.append(
            {
                "seed": seed,
                "path": str(checkpoint_path.relative_to(root)),
                "sha256": sha256_file(checkpoint_path),
            }
        )
        progress(
            f"QAT seed={seed} best_epoch={record['best_epoch']} "
            f"steps={record['optimizer_steps']}"
        )
    ledger = {
        "schema_version": 1,
        "candidate_id": document["candidate"]["id"],
        "candidate_role": QAT_ROLE,
        "protocol_sha256": current["protocol"]["sha256"],
        "source_candidate_id": SOURCE_CANDIDATE_ID,
        "data_audit": data_audit,
        "members": records,
        "optimizer_steps": sum(int(row["optimizer_steps"]) for row in records),
        "epochs_completed": [int(row["epochs_completed"]) for row in records],
        "best_epochs": [int(row["best_epoch"]) for row in records],
        "checkpoints": checkpoint_records,
        "normalizer_fits": 0,
        "new_hard_negative_mining_rounds": 0,
        "protected_data_access": False,
    }
    ledger_path = artifact_path / "training_ledger.json"
    _json_write(ledger_path, ledger)
    audit = _train_only_audit(
        sources, candidates, baseline_wrappers, fit, document
    )
    audit_path = artifact_path / "train_only_quantization_audit.json"
    _json_write(audit_path, audit)
    status = (
        "DEPLOYMENT_AWARE_QAT_TRAIN_ACCEPTANCE_PASS"
        if audit["all_checks_passed"]
        else "DEPLOYMENT_AWARE_QAT_TRAIN_ACCEPTANCE_FAIL"
    )
    result = {
        "status": status,
        "candidate_id": document["candidate"]["id"],
        "candidate_role": QAT_ROLE,
        "training_ledger_sha256": sha256_file(ledger_path),
        "train_only_quantization_audit_sha256": sha256_file(audit_path),
        "checkpoints": checkpoint_records,
        "optimizer_steps": ledger["optimizer_steps"],
        "three_seeds_preserved": True,
        "architecture_changed": False,
        "normalizer_changed": False,
        "scientific_verdict_changed": False,
        "protected_data_access": False,
    }
    if audit["all_checks_passed"]:
        freeze = {
            "schema_version": 1,
            "candidate_id": document["candidate"]["id"],
            "candidate_role": QAT_ROLE,
            "source_candidate_id": SOURCE_CANDIDATE_ID,
            "source_release_manifest_sha256": document["candidate"][
                "source_release_manifest"
            ]["sha256"],
            "source_checkpoint_sha256": {
                str(row["seed"]): row["sha256"]
                for row in document["candidate"]["source_checkpoints"]
            },
            "checkpoint_sha256": {
                str(row["seed"]): row["sha256"] for row in checkpoint_records
            },
            "architecture": document["source_contract"]["architecture"],
            "feature_schema_sha256": document["source_contract"]["preprocessing"][
                "feature_schema_sha256"
            ],
            "normalizer": document["source_contract"]["preprocessing"][
                "normalizer"
            ],
            "qat_protocol": current["protocol"],
            "training_ledger": {
                "path": str(ledger_path.relative_to(root)),
                "sha256": sha256_file(ledger_path),
            },
            "train_only_quantization_audit": {
                "path": str(audit_path.relative_to(root)),
                "sha256": sha256_file(audit_path),
            },
            "quantization_specification": document["fake_quantization"],
            "activation_ranges": current["activation_ranges"],
            "calibration": document["data"]["calibration"],
            "decision_contract": document["source_contract"]["runtime"],
            "scientific_release": False,
            "real_robot_supported": False,
            "safety_certified": False,
            "generalization_supported": False,
            "deployment_qat_candidate": True,
            "requires_e84_int8_revalidation": True,
            "candidate_frozen_before_post_training_development_evaluation": True,
            "post_training_development_evaluation_performed": False,
            "protected_data_access": False,
        }
        freeze_path = artifact_path / "candidate_freeze.json"
        _json_write(freeze_path, freeze)
        result["candidate_freeze_sha256"] = sha256_file(freeze_path)
    result_path = artifact_path / "training_result.json"
    _json_write(result_path, result)
    return result
