from __future__ import annotations

import numpy as np
import torch


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    correct = (pred == y).sum().item()
    return float(correct) / float(y.shape[0])


def confusion_matrix(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    num_classes: int,
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    k = int(num_classes)
    cm = np.zeros((k, k), dtype=np.int64)
    idx = k * y_true + y_pred
    bins = np.bincount(idx, minlength=k * k)
    cm[:] = bins.reshape(k, k)
    return cm

