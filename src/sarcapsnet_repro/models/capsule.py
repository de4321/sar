from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(x: torch.Tensor, dim: int = -1, eps: float = 1e-7) -> torch.Tensor:
    # v = (||s||^2 / (1+||s||^2)) * (s / ||s||)
    # Do routing math in float32 for numerical stability (AMP can overflow).
    x32 = x.float()
    norm = torch.linalg.vector_norm(x32, dim=dim, keepdim=True).clamp_min(eps)
    norm_sq = norm * norm
    # Stable form for large norm_sq: avoids inf/inf -> NaN.
    scale = 1.0 - 1.0 / (1.0 + norm_sq)
    return scale * (x32 / norm)


class PrimaryCapsReshape(nn.Module):
    """Paper-style primary capsule partition: reshape feature map by channels.

    Input feature map is expected to be [B, capsule_types*capsule_dim, H, W].
    Capsules are formed by grouping channels into `capsule_types` with
    `capsule_dim` dimensions at each spatial location.
    """

    def __init__(
        self,
        capsule_types: int = 32,
        capsule_dim: int = 8,
        expected_hw: int | None = 8,
    ) -> None:
        super().__init__()
        self.capsule_types = int(capsule_types)
        self.capsule_dim = int(capsule_dim)
        self.expected_hw = None if expected_hw is None else int(expected_hw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        expected_c = self.capsule_types * self.capsule_dim
        if c != expected_c:
            raise ValueError(
                f"expected channels={expected_c} for primary partition, got {c}"
            )
        if self.expected_hw is not None and (h != self.expected_hw or w != self.expected_hw):
            raise ValueError(
                f"expected primary capsule grid {self.expected_hw}x{self.expected_hw}, got {h}x{w}"
            )

        x = x.view(b, self.capsule_types, self.capsule_dim, h, w)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # [B,types,h,w,dim]
        x = x.view(b, self.capsule_types * h * w, self.capsule_dim)  # [B,N,dim]
        return squash(x, dim=-1)

    @property
    def num_primary(self) -> int:
        if self.expected_hw is None:
            raise RuntimeError("num_primary is unknown; pass expected_hw to PrimaryCapsReshape")
        return self.capsule_types * self.expected_hw * self.expected_hw


class PrimaryCapsConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        capsule_types: int = 32,
        capsule_dim: int = 8,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        expected_hw: int | None = None,
    ) -> None:
        super().__init__()
        self.capsule_types = int(capsule_types)
        self.capsule_dim = int(capsule_dim)
        self.expected_hw = None if expected_hw is None else int(expected_hw)
        out_channels = self.capsule_types * self.capsule_dim
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=int(padding),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] -> primary capsules [B,N,dim]
        x = self.conv(x)
        b, c, h, w = x.shape
        expected_c = self.capsule_types * self.capsule_dim
        if c != expected_c:
            raise ValueError(f"expected channels={expected_c}, got {c}")
        if self.expected_hw is not None and (h != self.expected_hw or w != self.expected_hw):
            raise ValueError(
                f"expected primary capsule grid {self.expected_hw}x{self.expected_hw}, got {h}x{w}"
            )

        x = x.view(b, self.capsule_types, self.capsule_dim, h, w)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # [B,types,h,w,dim]
        x = x.view(b, self.capsule_types * h * w, self.capsule_dim)  # [B,N,dim]
        return squash(x, dim=-1)

    @property
    def num_primary(self) -> int:
        if self.expected_hw is None:
            raise RuntimeError("num_primary is unknown; pass expected_hw to PrimaryCapsConv")
        return self.capsule_types * self.expected_hw * self.expected_hw


class CapsClassifier(nn.Module):
    def __init__(
        self,
        num_primary: int,
        num_classes: int,
        in_dim: int = 8,
        out_dim: int = 16,
        routing_iters: int = 3,
        detach_routing: bool = False,
    ) -> None:
        super().__init__()
        self.num_primary = int(num_primary)
        self.num_classes = int(num_classes)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.routing_iters = int(routing_iters)
        self.detach_routing = bool(detach_routing)
        if self.routing_iters < 1:
            raise ValueError("routing_iters must be >= 1")

        w = torch.empty(self.num_primary, self.num_classes, self.out_dim, self.in_dim)
        nn.init.normal_(w, mean=0.0, std=0.01)
        self.W = nn.Parameter(w)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # u: [B, num_primary, in_dim]
        # Routing is sensitive to AMP; force float32 math regardless of autocast.
        b, n, d = u.shape
        if n != self.num_primary or d != self.in_dim:
            raise ValueError(
                f"expected u shape [B,{self.num_primary},{self.in_dim}], got {tuple(u.shape)}"
            )

        if u.is_cuda:
            with torch.amp.autocast("cuda", enabled=False):
                u_hat = torch.einsum("bin,ijon->bijo", u.float(), self.W.float())
                u_hat_iter = u_hat.detach() if self.detach_routing else u_hat
                b_ij = torch.zeros(
                    (b, self.num_primary, self.num_classes),
                    device=u.device,
                    dtype=torch.float32,
                )
                for it in range(self.routing_iters):
                    c_ij = F.softmax(b_ij, dim=-1)
                    if it == self.routing_iters - 1:
                        s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=1)
                        v_j = squash(s_j, dim=-1)
                    else:
                        s_j = (c_ij.unsqueeze(-1) * u_hat_iter).sum(dim=1)
                        v_j = squash(s_j, dim=-1)
                        agreement = (u_hat_iter * v_j.unsqueeze(1)).sum(dim=-1)
                        b_ij = b_ij + agreement
        else:
            u_hat = torch.einsum("bin,ijon->bijo", u.float(), self.W.float())
            u_hat_iter = u_hat.detach() if self.detach_routing else u_hat
            b_ij = torch.zeros(
                (b, self.num_primary, self.num_classes),
                device=u.device,
                dtype=torch.float32,
            )
            for it in range(self.routing_iters):
                c_ij = F.softmax(b_ij, dim=-1)
                if it == self.routing_iters - 1:
                    s_j = (c_ij.unsqueeze(-1) * u_hat).sum(dim=1)
                    v_j = squash(s_j, dim=-1)
                else:
                    s_j = (c_ij.unsqueeze(-1) * u_hat_iter).sum(dim=1)
                    v_j = squash(s_j, dim=-1)
                    agreement = (u_hat_iter * v_j.unsqueeze(1)).sum(dim=-1)
                    b_ij = b_ij + agreement

        logits = torch.linalg.vector_norm(v_j, dim=-1)  # float32
        return v_j, logits
