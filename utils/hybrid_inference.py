from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

from losses.physics_loss import pde_residual_magnitude_map

# Match `simulate_case` in scripts/generate_synthetic_ep2d_dataset.py (stabilizes rollouts).
_EP_V_CLIP: tuple[float, float] = (-1.0, 1.5)
_EP_R_CLIP: tuple[float, float] = (-1.0, 2.0)


def _clamp_ep_state(x: torch.Tensor) -> torch.Tensor:
    """Clamp (V, R) to generator bounds; x is [B, 2, H, W]."""
    v_lo, v_hi = _EP_V_CLIP
    r_lo, r_hi = _EP_R_CLIP
    v = x[:, 0:1].clamp(v_lo, v_hi)
    r = x[:, 1:2].clamp(r_lo, r_hi)
    return torch.cat([v, r], dim=1)


@dataclass
class HybridStats:
    n_fno_forwards: int = 0
    n_full_case_defers: int = 0
    n_pixel_repairs: int = 0
    n_tissue_pixels: int = 0
    n_spacetime_pairs: int = 0
    wall_s_forward: float = 0.0
    wall_s_merge: float = 0.0

    def compute_units(self, oracle_equiv_per_case: float, pixels_per_forward: float) -> float:
        """Scalar proxy: FNO steps + oracle-equivalent full deferrals + pixel repairs scaled to forwards."""
        return (
            float(self.n_fno_forwards)
            + oracle_equiv_per_case * float(self.n_full_case_defers)
            + float(self.n_pixel_repairs) / max(pixels_per_forward, 1.0)
        )

    def deferred_fraction(self) -> float:
        if self.n_full_case_defers > 0:
            return 1.0
        if self.n_spacetime_pairs <= 0 or self.n_tissue_pixels <= 0:
            return 0.0
        return float(self.n_pixel_repairs) / float(self.n_spacetime_pairs * self.n_tissue_pixels)


def masked_rollout_rmse(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """pred, gt: [T, 2, H, W]; mask: [H, W] tissue indicator."""
    # float64 accumulation: float32 diff**2 can overflow on long rollouts / large domains.
    w = mask.astype(np.float64, copy=False)[None, None, ...]
    pd = pred.astype(np.float64, copy=False)
    gd = gt.astype(np.float64, copy=False)
    diff = (pd - gd) * w
    den = float(np.sum(w * w, dtype=np.float64)) * float(pred.shape[0] * pred.shape[1])
    if den <= 0:
        return float("nan")
    mse = float(np.sum(diff * diff, dtype=np.float64) / den)
    return float(np.sqrt(mse))


def stack_rollout(v: np.ndarray, r: np.ndarray) -> np.ndarray:
    return np.stack([v, r], axis=1)


@torch.no_grad()
def rollout_fno_only(
    fno: torch.nn.Module,
    v: np.ndarray,
    r: np.ndarray,
    device: str,
    sync_cuda: bool = False,
) -> tuple[np.ndarray, HybridStats]:
    stats = HybridStats()
    fno.eval()
    T = v.shape[0]
    cur = np.stack([v[0], r[0]], axis=0)[None, ...]
    preds = [cur[0].copy()]
    t0 = time.perf_counter()
    for t in range(T - 1):
        x = torch.from_numpy(cur).to(device)
        if sync_cuda and device.startswith("cuda"):
            torch.cuda.synchronize()
        y = fno(x)
        if sync_cuda and device.startswith("cuda"):
            torch.cuda.synchronize()
        cur = y.cpu().numpy()
        stats.n_fno_forwards += 1
        preds.append(cur[0].copy())
    stats.wall_s_forward = time.perf_counter() - t0
    pred = np.stack(preds, axis=0).astype(np.float32)
    return pred, stats


@torch.no_grad()
def rollout_hybrid_global(
    fno: torch.nn.Module,
    v: np.ndarray,
    r: np.ndarray,
    mask: np.ndarray,
    global_score: float,
    tau: float,
    device: str,
    sync_cuda: bool = False,
) -> tuple[np.ndarray, HybridStats]:
    stats = HybridStats()
    T, H, W = v.shape
    stats.n_tissue_pixels = int((mask > 0).sum())
    stats.n_spacetime_pairs = T - 1

    if global_score > tau:  # defer entire trajectory (trusted oracle / simulator)
        stats.n_full_case_defers = 1
        pred = stack_rollout(v, r)
        return pred.astype(np.float32), stats

    pred, s2 = rollout_fno_only(fno, v, r, device, sync_cuda=sync_cuda)
    stats.n_fno_forwards = s2.n_fno_forwards
    stats.wall_s_forward = s2.wall_s_forward
    return pred, stats


@torch.no_grad()
def rollout_hybrid_local(
    fno: torch.nn.Module,
    local_head: torch.nn.Module,
    v: torch.Tensor,
    r: torch.Tensor,
    mask_t: torch.Tensor,
    params_t: torch.Tensor,
    dt_t: torch.Tensor,
    tau_prob: float,
    device: str,
    sync_cuda: bool = False,
) -> tuple[np.ndarray, HybridStats]:
    """Autoregressive hybrid: at each step merge FNO prediction with GT on high-risk patches."""
    stats = HybridStats()
    fno.eval()
    local_head.eval()
    T, H, W = v.shape
    tissue = mask_t > 0
    stats.n_tissue_pixels = int(tissue.sum().item())
    stats.n_spacetime_pairs = T - 1

    preds_list: list[np.ndarray] = []
    cur = torch.stack([v[0], r[0]], dim=0).unsqueeze(0)
    preds_list.append(cur[0].cpu().numpy().copy())

    t_fwd = 0.0
    t_merge = 0.0

    for t in range(T - 1):
        y_t = torch.stack([v[t], r[t]], dim=0).unsqueeze(0)
        x = torch.where(torch.isfinite(cur), cur, y_t)
        x = _clamp_ep_state(x)
        t0 = time.perf_counter()
        if sync_cuda and device.startswith("cuda"):
            torch.cuda.synchronize()
        pred = fno(x)
        if sync_cuda and device.startswith("cuda"):
            torch.cuda.synchronize()
        t_fwd += time.perf_counter() - t0
        stats.n_fno_forwards += 1

        y = torch.stack([v[t + 1], r[t + 1]], dim=0).unsqueeze(0)

        t0 = time.perf_counter()
        res_mag = pde_residual_magnitude_map(x, pred, params_t, dt_t, tissue.unsqueeze(0).float())
        inp = torch.cat([x, pred, res_mag], dim=1)
        logits = local_head(inp)
        probs = torch.sigmoid(logits)
        defer_grid = (probs > tau_prob).float()
        defer_hw = F.interpolate(defer_grid, size=(H, W), mode="bilinear", align_corners=False)
        defer_hw = (defer_hw * tissue.unsqueeze(0).unsqueeze(0).float()).clamp(0.0, 1.0)

        merged = pred * (1.0 - defer_hw) + y * defer_hw
        merged = torch.where(torch.isfinite(merged), merged, y)
        merged = _clamp_ep_state(merged)
        t_merge += time.perf_counter() - t0

        stats.n_pixel_repairs += int(defer_hw.sum().item())
        cur = merged
        preds_list.append(merged[0].cpu().numpy().copy())

    stats.wall_s_forward = t_fwd
    stats.wall_s_merge = t_merge
    pred_np = np.stack(preds_list, axis=0).astype(np.float32)
    return pred_np, stats
