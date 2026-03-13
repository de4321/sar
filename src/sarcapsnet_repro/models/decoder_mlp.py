from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructionMLP(nn.Module):
    def __init__(
        self,
        num_classes: int,
        caps_dim: int = 16,
        out_hw: int = 32,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.caps_dim = int(caps_dim)
        self.out_hw = int(out_hw)

        in_dim = self.num_classes * self.caps_dim
        out_dim = self.out_hw * self.out_hw
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_dim, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, v: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # v: [B, num_classes, caps_dim]
        # y: [B]
        b, c, d = v.shape
        if c != self.num_classes or d != self.caps_dim:
            raise ValueError(
                f"expected v shape [B,{self.num_classes},{self.caps_dim}], got {tuple(v.shape)}"
            )
        if y.ndim != 1 or y.shape[0] != b:
            raise ValueError(f"expected y shape [B], got {tuple(y.shape)}")

        mask = torch.zeros((b, self.num_classes), device=v.device, dtype=v.dtype)
        mask.scatter_(1, y.view(-1, 1), 1.0)
        masked = (v * mask.unsqueeze(-1)).reshape(b, self.num_classes * self.caps_dim)

        out = self.mlp(masked)
        return out.view(b, 1, self.out_hw, self.out_hw)

