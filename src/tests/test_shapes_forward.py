from __future__ import annotations

import pytest
import torch

from sarcapsnet_repro.models.sarcapsnet import SARCapsNet


@pytest.mark.parametrize("batch", [2])
def test_forward_shapes(batch: int):
    model = SARCapsNet(num_classes=6, input_size=28)
    x = torch.randn(batch, 1, 28, 28)
    y = torch.zeros(batch, dtype=torch.long)

    logits, v, recon = model(x, y=y)
    assert logits.shape == (batch, 6)
    assert v.shape == (batch, 6, 16)
    assert recon is not None
    assert recon.shape == (batch, 1, 28, 28)
