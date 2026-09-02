"""Read-only loading for frozen FastReflex model checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from fastreflex.dataset.loader import CLASS_NAMES
from fastreflex.models.baselines import build_model


def load_checkpoint(path: Path) -> tuple[nn.Module, dict[str, object]]:
    """Load one frozen checkpoint for inference without a training dependency."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "fastreflex_raw_imu_baseline":
        raise ValueError("unsupported checkpoint format")
    model = build_model(
        checkpoint["family"],
        int(checkpoint["window_samples"]),
        int(checkpoint.get("input_channels", 6)),
        class_count=len(checkpoint.get("class_names", CLASS_NAMES)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    metadata = {
        key: checkpoint[key]
        for key in ("family", "window_samples", "seed", "best_epoch")
    }
    metadata["input_channels"] = int(checkpoint.get("input_channels", 6))
    metadata["class_names"] = list(checkpoint.get("class_names", CLASS_NAMES))
    return model, metadata
