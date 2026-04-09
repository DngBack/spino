from __future__ import annotations

import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes_h: int, modes_w: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_h = modes_h
        self.modes_w = modes_w
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale
            * torch.randn(
                in_channels, out_channels, modes_h, modes_w, dtype=torch.cfloat
            )
        )

    def compl_mul2d(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            bsz,
            self.out_channels,
            h,
            w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        mh = min(self.modes_h, h)
        mw = min(self.modes_w, (w // 2 + 1))
        out_ft[:, :, :mh, :mw] = self.compl_mul2d(
            x_ft[:, :, :mh, :mw],
            self.weights[:, :, :mh, :mw],
        )
        x = torch.fft.irfft2(out_ft, s=(h, w))
        return x


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes_h: int, modes_w: int) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes_h, modes_w)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.pointwise(x)
        return self.act(y)


class FNO2d(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        width: int = 32,
        depth: int = 4,
        modes_h: int = 16,
        modes_w: int = 16,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [FNOBlock(width, modes_h=modes_h, modes_w=modes_w) for _ in range(depth)]
        )
        self.output_proj = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        for blk in self.blocks:
            z = blk(z)
        return self.output_proj(z)
