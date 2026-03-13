from __future__ import annotations

import torch
import torch.nn as nn

from .threshold import SoftThresholdDenoise


class ATM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 2,
        padding: int = 0,
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        s = int(stride)
        p = int(padding)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=True
        )
        self.threshold = SoftThresholdDenoise(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                out = self.conv(x.float())
                out = self.threshold(out)
            return out
        out = self.conv(x)
        out = self.threshold(out)
        return out
