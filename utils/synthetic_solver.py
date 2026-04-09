from __future__ import annotations

import numpy as np


def _laplacian(z: np.ndarray) -> np.ndarray:
    return (
        np.roll(z, 1, axis=0)
        + np.roll(z, -1, axis=0)
        + np.roll(z, 1, axis=1)
        + np.roll(z, -1, axis=1)
        - 4.0 * z
    )


def simulate_ep2d_from_initial(
    v0: np.ndarray,
    r0: np.ndarray,
    mask: np.ndarray,
    diffusion: float,
    excitability: float,
    restitution: float,
    dt: float,
    num_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lightweight deterministic reference solver for runtime comparison.
    Returns full trajectories with num_steps frames including step 0.
    """
    v = v0.astype(np.float32).copy()
    r = r0.astype(np.float32).copy()
    traj_v = np.zeros((num_steps, *v.shape), dtype=np.float32)
    traj_r = np.zeros((num_steps, *r.shape), dtype=np.float32)
    traj_v[0] = v
    traj_r[0] = r

    for t in range(1, num_steps):
        lap = _laplacian(v)
        reaction = excitability * v * (1.0 - v) * (v - 0.08) - r
        recov = restitution * (v - 0.15 * r)
        v = v + dt * (diffusion * lap + reaction)
        r = r + dt * recov
        v = np.clip(v * mask, -1.0, 1.5)
        r = np.clip(r * mask, -1.0, 2.0)
        traj_v[t] = v
        traj_r[t] = r

    return traj_v, traj_r
