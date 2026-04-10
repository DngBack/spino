from __future__ import annotations

import torch


def coverage_regularization(scores: torch.Tensor, tau: torch.Tensor, target_coverage: float) -> torch.Tensor:
    """
    Accept if score <= tau.
    Smooth surrogate coverage with sigmoid temperature.
    """
    temp = 10.0
    accept_prob = torch.sigmoid((tau - scores) * temp)
    cov = torch.mean(accept_prob)
    return (cov - target_coverage) ** 2
