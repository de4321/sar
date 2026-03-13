from __future__ import annotations

import torch
import torch.nn as nn


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    b, c, h, w = x.shape
    if c % groups != 0:
        raise ValueError(f"channels({c}) must be divisible by groups({groups})")
    x = x.view(b, groups, c // groups, h, w)
    x = x.transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class ShuffleAttention(nn.Module):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        channels = int(channels)
        groups = int(groups)
        if channels % groups != 0:
            raise ValueError(f"channels({channels}) must be divisible by groups({groups})")
        if (channels // groups) % 2 != 0:
            raise ValueError(
                f"(channels/groups) must be even, got channels={channels}, groups={groups}"
            )

        self.channels = channels
        self.groups = groups
        self.pool = nn.AdaptiveAvgPool2d(1)

        group_ch = channels // groups
        branch_ch = group_ch // 2

        self.channel_fc = nn.Conv2d(branch_ch, branch_ch, kernel_size=1, bias=True)
        self.spatial_norm = nn.GroupNorm(num_groups=1, num_channels=branch_ch)
        self.spatial_fc = nn.Conv2d(branch_ch, branch_ch, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"expected channels={self.channels}, got {c}")

        g = self.groups
        x = x.view(b * g, c // g, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        w_ch = self.sigmoid(self.channel_fc(self.pool(x_0)))
        x_0 = x_0 * w_ch

        w_sp = self.sigmoid(self.spatial_fc(self.spatial_norm(x_1)))
        x_1 = x_1 * w_sp

        x = torch.cat([x_0, x_1], dim=1).view(b, c, h, w)
        return channel_shuffle(x, g)

