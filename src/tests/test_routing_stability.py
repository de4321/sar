from __future__ import annotations

import torch

from sarcapsnet_repro.models.capsule import CapsClassifier


def test_routing_no_nan():
    b = 2
    num_primary = 32
    u = torch.randn(b, num_primary, 8)
    caps = CapsClassifier(num_primary=num_primary, num_classes=6, routing_iters=3)
    v, logits = caps(u)
    assert torch.isfinite(v).all()
    assert torch.isfinite(logits).all()

