from __future__ import annotations

import torch
import torch.nn as nn


class LocalRejectorCNN(nn.Module):
    """
    Patchwise risk head: full-resolution multi-channel input -> logits on H/ps x W/ps grid.
    Higher logit => higher local risk (defer region).
    """

    def __init__(self, in_channels: int = 5, hidden: int = 64, patch_stride: int = 4) -> None:
        super().__init__()
        self.patch_stride = patch_stride
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        ]
        if patch_stride == 8:
            layers.extend(
                [
                    nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden),
                    nn.ReLU(inplace=True),
                ]
            )
        layers.append(nn.Conv2d(hidden, 1, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
