from __future__ import annotations

import numpy as np


def laplacian2d(z: np.ndarray) -> np.ndarray:
    return (
        np.roll(z, 1, axis=-2)
        + np.roll(z, -1, axis=-2)
        + np.roll(z, 1, axis=-1)
        + np.roll(z, -1, axis=-1)
        - 4.0 * z
    )


def pde_residual_map(
    v_t: np.ndarray,
    r_t: np.ndarray,
    v_t1_pred: np.ndarray,
    r_t1_pred: np.ndarray,
    diffusion: float,
    excitability: float,
    restitution: float,
    dt: float,
    mask: np.ndarray,
) -> np.ndarray:
    lap = laplacian2d(v_t)
    reaction = excitability * v_t * (1.0 - v_t) * (v_t - 0.08) - r_t
    recov = restitution * (v_t - 0.15 * r_t)
    res_v = (v_t1_pred - v_t) / dt - (diffusion * lap + reaction)
    res_r = (r_t1_pred - r_t) / dt - recov
    res = np.sqrt(res_v**2 + res_r**2) * mask
    return res.astype(np.float32)
