from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_risk_loss(scores: torch.Tensor, unsafe_labels: torch.Tensor) -> torch.Tensor:
    """
    scores: higher => riskier
    unsafe_labels: 1 for unsafe, 0 for safe
    """
    return F.binary_cross_entropy_with_logits(scores, unsafe_labels.float())


def pairwise_ranking_loss(scores: torch.Tensor, target_risk: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """
    Encourage higher score for samples with higher target risk.
    """
    n = scores.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=scores.device)
    diff_score = scores.view(-1, 1) - scores.view(1, -1)
    diff_risk = target_risk.view(-1, 1) - target_risk.view(1, -1)
    mask = diff_risk > 0
    if not torch.any(mask):
        return torch.tensor(0.0, device=scores.device)
    loss = torch.relu(margin - diff_score[mask])
    return loss.mean()
