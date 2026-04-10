from __future__ import annotations

import torch
import torch.nn.functional as F


def build_patch_targets(
    pred: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    patch: int,
    quantile: float,
) -> torch.Tensor:
    """
    pred, y: [B,2,H,W], mask: [B,H,W]
    Returns binary unsafe target [B,1,H//patch,W//patch] (pooled error > per-sample quantile).
    """
    err = (pred - y).abs().mean(dim=1, keepdim=True) * mask.unsqueeze(1)
    pooled = F.avg_pool2d(err, kernel_size=patch, stride=patch)
    mask_p = F.avg_pool2d(mask.unsqueeze(1), kernel_size=patch, stride=patch)
    valid = mask_p > 1e-3
    B, _, hp, wp = pooled.shape
    tgt = torch.zeros_like(pooled)
    for b in range(B):
        pv = pooled[b, 0][valid[b, 0]]
        if pv.numel() < 4:
            continue
        thr = torch.quantile(pv, quantile)
        tgt[b, 0] = (pooled[b, 0] > thr).float() * valid[b, 0].float()
    return tgt
