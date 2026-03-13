from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionDeconv(nn.Module):
    """Paper-style reconstruction decoder: FC + transposed convolutions.

    The paper specifies:
      - FC output = 640
      - TrConv kernels: 7x7, 7x7, 2x2
      - Strides: 2, 1, 1
      - ReLU after hidden layers, final Sigmoid

    With start_hw=8 and valid transposed convolutions (padding=0), the spatial
    sizes follow 8 -> 21 -> 27 -> 28, matching Fig.1.
    """

    def __init__(
        self,
        num_classes: int,
        caps_dim: int = 16,
        out_hw: int = 28,
        fc_out: int = 640,
        start_hw: int = 8,
        mid_channels1: int = 64,
        mid_channels2: int = 32,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.caps_dim = int(caps_dim)
        self.out_hw = int(out_hw)
        self.fc_out = int(fc_out)
        self.start_hw = int(start_hw)

        in_dim = self.num_classes * self.caps_dim
        if self.fc_out % (self.start_hw * self.start_hw) != 0:
            raise ValueError(
                f"fc_out({self.fc_out}) must be divisible by start_hw^2({self.start_hw*self.start_hw})"
            )
        self.start_channels = self.fc_out // (self.start_hw * self.start_hw)

        self.fc = nn.Linear(in_dim, self.fc_out, bias=True)

        # Paper Table III / Fig.1: TrConv1 7x7(s=2), TrConv2 7x7(s=1), TrConv3 2x2(s=1).
        self.deconv1 = nn.ConvTranspose2d(
            self.start_channels,
            int(mid_channels1),
            kernel_size=7,
            stride=2,
            padding=0,
            output_padding=0,
            bias=True,
        )
        self.deconv2 = nn.ConvTranspose2d(
            int(mid_channels1),
            int(mid_channels2),
            kernel_size=7,
            stride=1,
            padding=0,
            output_padding=0,
            bias=True,
        )
        self.deconv3 = nn.ConvTranspose2d(
            int(mid_channels2),
            1,
            kernel_size=2,
            stride=1,
            padding=0,
            output_padding=0,
            bias=True,
        )

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _center_pad_or_crop(x: torch.Tensor, out_hw: int) -> torch.Tensor:
        _, _, h, w = x.shape
        target = int(out_hw)

        # Center-crop if larger.
        if h > target:
            top = (h - target) // 2
            x = x[..., top : top + target, :]
        if w > target:
            left = (w - target) // 2
            x = x[..., :, left : left + target]

        # Center-pad if smaller.
        _, _, h2, w2 = x.shape
        pad_h = max(0, target - h2)
        pad_w = max(0, target - w2)
        if pad_h or pad_w:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return x

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

        z = self.fc(masked)  # [B, fc_out]
        z = z.view(b, self.start_channels, self.start_hw, self.start_hw)

        z = self.relu(self.deconv1(z))
        z = self.relu(self.deconv2(z))
        z = self.deconv3(z)
        if z.shape[-1] != self.out_hw or z.shape[-2] != self.out_hw:
            z = self._center_pad_or_crop(z, out_hw=self.out_hw)
        return self.sigmoid(z)
