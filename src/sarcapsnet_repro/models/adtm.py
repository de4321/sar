from __future__ import annotations

import torch
import torch.nn as nn

from .threshold import SoftThresholdDenoise

try:
    from torchvision.ops import DeformConv2d
except Exception:  # pragma: no cover
    DeformConv2d = None


class ADTM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        if DeformConv2d is None:
            raise ImportError(
                "torchvision.ops.DeformConv2d is required for ADTM. "
                "Install torchvision matching your PyTorch build."
            )

        k = int(kernel_size)
        s = int(stride)
        p = int(padding)

        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * k * k,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=True,
        )
        # Common DCN practice: start from regular grid (zero offsets) for stability.
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)
        self.deform = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=True,
        )
        self.threshold = SoftThresholdDenoise(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DeformConv2d + offsets can overflow under AMP; force float32 math.
        if x.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                x32 = x.float()
                offset = self.offset_conv(x32)
                out = self.deform(x32, offset)
        else:
            offset = self.offset_conv(x)
            out = self.deform(x, offset)
        return self.threshold(out)
