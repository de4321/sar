from __future__ import annotations

import torch
import torch.nn as nn


class SoftThresholdDenoise(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        channels = int(channels)
        hidden = max(1, channels // int(reduction))
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        abs_x = x.abs()
        alpha = abs_x.mean(dim=(2, 3))  # [B,C]
        beta = self.fc(alpha)  # [B,C] in (0,1)
        gamma = (alpha * beta).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        return torch.sign(x) * torch.relu(abs_x - gamma)

