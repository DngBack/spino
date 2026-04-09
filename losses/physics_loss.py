from __future__ import annotations

import torch


def laplacian2d(z: torch.Tensor) -> torch.Tensor:
    return (
        torch.roll(z, shifts=1, dims=-2)
        + torch.roll(z, shifts=-1, dims=-2)
        + torch.roll(z, shifts=1, dims=-1)
        + torch.roll(z, shifts=-1, dims=-1)
        - 4.0 * z
    )


def pde_residual_loss(
    x_t: torch.Tensor,
    pred_t1: torch.Tensor,
    params: torch.Tensor,
    dt: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Residual for simple reaction-diffusion update used in synthetic generator.
    x_t and pred_t1 shape: [B,2,H,W], params: [B,3] -> [diffusion, excitability, restitution].
    """
    v_t = x_t[:, 0]
    r_t = x_t[:, 1]
    v_n = pred_t1[:, 0]
    r_n = pred_t1[:, 1]
    m = mask
    if m.ndim == 3:
        pass
    else:
        m = m[:, 0]

    diffusion = params[:, 0].view(-1, 1, 1)
    excitability = params[:, 1].view(-1, 1, 1)
    restitution = params[:, 2].view(-1, 1, 1)
    dt_ = dt.view(-1, 1, 1)

    lap = laplacian2d(v_t)
    reaction = excitability * v_t * (1.0 - v_t) * (v_t - 0.08) - r_t
    recov = restitution * (v_t - 0.15 * r_t)

    res_v = (v_n - v_t) / dt_ - (diffusion * lap + reaction)
    res_r = (r_n - r_t) / dt_ - recov

    # Evaluate physics only inside active domain.
    res_v = res_v * m
    res_r = res_r * m
    return torch.mean(res_v**2) + torch.mean(res_r**2)


def bc_loss(pred_t1: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Penalize non-zero border values inside domain as a simple boundary stabilizer.
    """
    v = pred_t1[:, 0]
    m = mask if mask.ndim == 3 else mask[:, 0]
    top = v[:, 0, :] * m[:, 0, :]
    bottom = v[:, -1, :] * m[:, -1, :]
    left = v[:, :, 0] * m[:, :, 0]
    right = v[:, :, -1] * m[:, :, -1]
    return (top.pow(2).mean() + bottom.pow(2).mean() + left.pow(2).mean() + right.pow(2).mean()) / 4.0


def ic_loss(pred_t1: torch.Tensor, y_t1: torch.Tensor, t_index: torch.Tensor) -> torch.Tensor:
    """
    Additional penalty on first transition samples (t=0), complementing data loss.
    """
    mask = (t_index == 0).float().view(-1, 1, 1, 1)
    if torch.sum(mask) < 1:
        return torch.tensor(0.0, device=pred_t1.device)
    return torch.mean(((pred_t1 - y_t1) * mask) ** 2)
