from __future__ import annotations

import torch


def total_variation_2d(probs: torch.Tensor) -> torch.Tensor:
    """
    Total variation on spatial map. probs: [B,1,H,W]
    """
    dh = (probs[:, :, 1:, :] - probs[:, :, :-1, :]).abs().mean()
    dw = (probs[:, :, :, 1:] - probs[:, :, :, :-1]).abs().mean()
    return dh + dw
