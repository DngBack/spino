from __future__ import annotations

import numpy as np
import torch


def compute_rollout_drift_map(
    model: torch.nn.Module,
    gt: np.ndarray,
    horizon: int,
    device: str,
) -> tuple[np.ndarray, float]:
    """
    Compare recursive rollout vs one-step teacher-forced predictions.
    gt shape: [T,2,H,W]
    Returns:
      - spatial drift map [H,W]
      - temporal growth slope scalar
    """
    model.eval()
    t_steps = min(horizon, gt.shape[0] - 1)
    if t_steps <= 1:
        h, w = gt.shape[-2], gt.shape[-1]
        return np.zeros((h, w), dtype=np.float32), 0.0

    per_t_err = []
    drift_maps = []
    cur = gt[0:1].copy()
    with torch.no_grad():
        for t in range(t_steps):
            x_teacher = torch.from_numpy(gt[t : t + 1]).to(device)
            one_step = model(x_teacher).cpu().numpy()[0]  # [2,H,W]
            x_rec = torch.from_numpy(cur).to(device)
            rec_step = model(x_rec).cpu().numpy()[0]  # [2,H,W]
            cur = rec_step[None, ...]

            diff = np.abs(one_step - rec_step).mean(axis=0)  # [H,W]
            drift_maps.append(diff)
            per_t_err.append(float(np.mean(diff)))

    drift_map = np.mean(np.stack(drift_maps, axis=0), axis=0).astype(np.float32)
    x = np.arange(len(per_t_err), dtype=np.float32)
    y = np.array(per_t_err, dtype=np.float32)
    if len(x) >= 2:
        slope = float(np.polyfit(x, y, deg=1)[0])
    else:
        slope = 0.0
    return drift_map, slope
