from __future__ import annotations

import torch
import torch.nn as nn

from .adtm import ADTM
from .atm import ATM
from .capsule import CapsClassifier, PrimaryCapsReshape
from .decoder_deconv import ReconstructionDeconv
from .shuffle_attention import ShuffleAttention


class SARCapsNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 6,
        channels: int = 256,
        sa_groups: int = 8,
        routing_iters: int = 3,
        input_size: int = 28,
        adtm_kernel: int = 5,
        atm_kernel: int = 5,
        primary_caps_dim: int = 8,
        primary_caps_types: int = 32,
        sar_caps_dim: int = 16,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.channels = int(channels)
        self.input_size = int(input_size)
        self.adtm_kernel = int(adtm_kernel)
        self.atm_kernel = int(atm_kernel)
        self.primary_caps_dim = int(primary_caps_dim)
        self.primary_caps_types = int(primary_caps_types)
        self.sar_caps_dim = int(sar_caps_dim)

        # Paper Fig.1 uses valid convolutions (no padding):
        # 28->24 (ADTM1), 24->20 (ADTM2), 20->8 (ATM, stride 2).
        self.adtm1 = ADTM(
            1,
            self.channels,
            kernel_size=self.adtm_kernel,
            stride=1,
            padding=0,
        )
        self.adtm2 = ADTM(
            self.channels,
            self.channels,
            kernel_size=self.adtm_kernel,
            stride=1,
            padding=0,
        )
        self.atm = ATM(
            self.channels,
            self.channels,
            kernel_size=self.atm_kernel,
            stride=2,
            padding=0,
        )
        self.sa = ShuffleAttention(self.channels, groups=int(sa_groups))

        def _conv_out_hw(
            in_hw: int,
            kernel: int,
            stride: int,
            padding: int,
            dilation: int = 1,
        ) -> int:
            in_hw = int(in_hw)
            kernel = int(kernel)
            stride = int(stride)
            padding = int(padding)
            dilation = int(dilation)
            return (in_hw + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1

        target_primary_hw = 8
        adtm1_hw = _conv_out_hw(
            self.input_size,
            kernel=self.adtm_kernel,
            stride=1,
            padding=0,
        )
        adtm2_hw = _conv_out_hw(
            adtm1_hw,
            kernel=self.adtm_kernel,
            stride=1,
            padding=0,
        )
        atm_hw = _conv_out_hw(
            adtm2_hw,
            kernel=self.atm_kernel,
            stride=2,
            padding=0,
        )
        if atm_hw != target_primary_hw:
            raise ValueError(
                f"paper-aligned encoder expects ATM output {target_primary_hw}x{target_primary_hw}, "
                f"got {atm_hw}x{atm_hw} from input_size={self.input_size} "
                f"(ADTM/ATM kernels={self.adtm_kernel}/{self.atm_kernel}, valid conv)."
            )

        expected_channels = self.primary_caps_types * self.primary_caps_dim
        if self.channels != expected_channels:
            raise ValueError(
                f"channels must equal primary_caps_types*caps_dim for partition mode; "
                f"got channels={self.channels}, expected {expected_channels}"
            )

        # Paper text: extracted features are partitioned into primary capsules.
        self.primary_caps = PrimaryCapsReshape(
            capsule_types=self.primary_caps_types,
            capsule_dim=self.primary_caps_dim,
            expected_hw=target_primary_hw,
        )
        self.class_caps = CapsClassifier(
            num_primary=self.primary_caps.num_primary,
            num_classes=self.num_classes,
            in_dim=self.primary_caps_dim,
            out_dim=self.sar_caps_dim,
            routing_iters=int(routing_iters),
        )
        self.decoder = ReconstructionDeconv(
            num_classes=self.num_classes,
            caps_dim=self.sar_caps_dim,
            out_hw=self.input_size,
            start_hw=target_primary_hw,
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        debug: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        def _check(t: torch.Tensor, name: str) -> None:
            if debug and not torch.isfinite(t).all():
                raise FloatingPointError(f"Non-finite tensor at {name}")

        # x: [B,1,H,W]
        if x.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                x = x.float()
                x = self.adtm1(x)
                _check(x, "adtm1")
                x = self.adtm2(x)
                _check(x, "adtm2")
                x = self.atm(x)
                _check(x, "atm")
                x = self.sa(x)
                _check(x, "shuffle_attention")

                u = self.primary_caps(x)
                _check(u, "primary_caps")
                v, logits = self.class_caps(u)
                _check(v, "sar_caps")
                _check(logits, "logits")

                recon = None
                if y is not None:
                    recon = self.decoder(v, y)
                    _check(recon, "recon")

                return logits, v, recon

        x = self.adtm1(x)
        _check(x, "adtm1")
        x = self.adtm2(x)
        _check(x, "adtm2")
        x = self.atm(x)
        _check(x, "atm")
        x = self.sa(x)
        _check(x, "shuffle_attention")

        u = self.primary_caps(x)
        _check(u, "primary_caps")
        v, logits = self.class_caps(u)
        _check(v, "sar_caps")
        _check(logits, "logits")

        recon = None
        if y is not None:
            recon = self.decoder(v, y)
            _check(recon, "recon")

        return logits, v, recon
