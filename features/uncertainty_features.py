from __future__ import annotations

import numpy as np
import torch


def perturbation_ensemble_variance(
    model: torch.nn.Module,
    x: torch.Tensor,
    num_samples: int = 6,
    noise_std: float = 0.01,
) -> np.ndarray:
    """
    Approximate predictive uncertainty with test-time input perturbation ensemble.
    Returns variance map over channels averaged to [H,W].
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(num_samples):
            xn = x + noise_std * torch.randn_like(x)
            y = model(xn).cpu().numpy()[0]  # [C,H,W]
            preds.append(y)
    arr = np.stack(preds, axis=0)  # [S,C,H,W]
    var_map = np.var(arr, axis=0).mean(axis=0)  # [H,W]
    return var_map.astype(np.float32)
