from __future__ import annotations

import torch
import torch.nn as nn


class GlobalRejectorMLP(nn.Module):
    """
    Outputs a scalar risk score per case (higher = riskier).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 32, depth: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(d, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
