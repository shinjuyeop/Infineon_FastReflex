"""Deployment-aware QAT contract and equation tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastreflex.models.baselines import build_model, parameter_count
from fastreflex.training.qat import (
    DeploymentAwareQATGRU,
    PROTECTED_SOURCES,
    QuantizationSpec,
    explicit_float_forward,
    validate_qat_protocol,
)
from tests.support import REPOSITORY_ROOT as ROOT, load_yaml

CONFIG = ROOT / "configs/experiment/20260904_deployment_aware_qat.yaml"


def _specs() -> dict[str, QuantizationSpec]:
    names = {f"input_projection_b{block}" for block in range(6)}
    for timestep in range(20):
        names.add(f"hidden_state_t{timestep}")
        for block in range(6):
            names.add(f"hidden_projection_t{timestep}_b{block}")
        for block in range(2):
            for stage in (
                "reset_preact",
                "update_preact",
                "reset_hidden_new",
                "candidate_preact",
                "hidden_difference",
                "update_delta",
                "hidden_block",
            ):
                names.add(f"{stage}_t{timestep}_b{block}")
    names.add("classifier_logits")
    return {name: QuantizationSpec(0.02, 0) for name in names}


def test_qat_protocol_is_one_family_and_rejects_protected_data() -> None:
    document = load_yaml(CONFIG)
    validate_qat_protocol(document)
    assert set(document["scientific_boundary"]["prohibited_data_sources"]) == set(
        PROTECTED_SOURCES
    )
    assert document["experiment"]["candidate_family_count"] == 1
    assert document["training"]["seeds"] == [20260828, 20260829, 20260830]
    assert document["objective"]["within_class_domain_balanced_loss_mass"] is False

    modified = load_yaml(CONFIG)
    modified["data"]["allowed_sources"].append("Generalization_HOLDOUT")
    with pytest.raises(ValueError, match="allowed data"):
        validate_qat_protocol(modified)


def test_explicit_float_bypass_preserves_frozen_gru_equation() -> None:
    torch.manual_seed(41)
    model = build_model("gru", 20, input_channels=80, class_count=2).eval()
    inputs = torch.from_numpy(
        np.random.default_rng(41).normal(size=(7, 20, 80)).astype(np.float32)
    )
    wrapper = DeploymentAwareQATGRU(model, _specs()).eval()
    with torch.no_grad():
        expected = model(inputs)
        direct, hidden, projected = explicit_float_forward(model, inputs)
        bypass, bypass_hidden, bypass_projected = wrapper.forward_with_hidden(
            inputs, fake_quant=False
        )
    torch.testing.assert_close(direct, expected, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(bypass, expected, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(bypass_hidden, hidden, atol=0.0, rtol=0.0)
    torch.testing.assert_close(bypass_projected, projected, atol=0.0, rtol=0.0)
    assert parameter_count(wrapper) == 11_010
    assert tuple(wrapper.state_dict()) == tuple(model.state_dict())


def test_fake_quant_forward_has_finite_contract_and_hidden_feedback_gradient() -> None:
    torch.manual_seed(19)
    model = build_model("gru", 20, input_channels=80, class_count=2).eval()
    wrapper = DeploymentAwareQATGRU(model, _specs())
    inputs = torch.randn(3, 20, 80)
    logits, hidden, projected = wrapper.forward_with_hidden(inputs, fake_quant=True)
    assert logits.shape == (3, 2)
    assert hidden.shape == (3, 20, 32)
    assert projected.shape == (3, 20, 96)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert wrapper.gru.weight_hh_l0.grad is not None
    assert torch.isfinite(wrapper.gru.weight_hh_l0.grad).all()
