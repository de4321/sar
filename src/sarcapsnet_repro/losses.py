from __future__ import annotations

import torch
import torch.nn.functional as F


def margin_loss(
    v: torch.Tensor,
    y: torch.Tensor,
    m_plus: float = 0.9,
    m_minus: float = 0.1,
    lambda_: float = 0.5,
) -> torch.Tensor:
    # v: [B,C,16]
    b, c, _ = v.shape
    if y.ndim != 1 or y.shape[0] != b:
        raise ValueError(f"expected y shape [B], got {tuple(y.shape)}")

    lengths = torch.linalg.vector_norm(v, dim=-1)  # [B,C]
    t = F.one_hot(y, num_classes=c).to(dtype=v.dtype)

    pos = t * F.relu(m_plus - lengths).pow(2)
    neg = (1.0 - t) * F.relu(lengths - m_minus).pow(2)
    loss = pos + float(lambda_) * neg
    return loss.sum(dim=1).mean()

