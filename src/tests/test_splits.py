from __future__ import annotations

from pathlib import Path

import pytest

from sarcapsnet_repro.data.splits import SAR_ACD_CLASSES, make_split_sar_acd_dict


def test_sar_acd_split_counts_match_80_20():
    data_root = Path("dataset/SAR-ACD")
    if not data_root.exists():
        pytest.skip("dataset/SAR-ACD not found")
    payload = make_split_sar_acd_dict(data_root, seed=0, train_ratio=0.8)

    total = 0
    for cls in SAR_ACD_CLASSES:
        n_train = len(payload["train"][cls])
        n_test = len(payload["test"][cls])
        n_all = n_train + n_test
        assert n_train == int(n_all * 0.8)
        assert n_test == n_all - n_train
        total += n_all

    assert total == 3032
