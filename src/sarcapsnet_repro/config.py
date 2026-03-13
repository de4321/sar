from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ExperimentConfig:
    # Data
    data_root: Path
    split_json: Path
    limited_rate: float = 1.0
    seed: int = 0
    input_size: int = 28
    resize_mode: Literal["stretch", "letterbox"] = "letterbox"
    num_classes: int = 6

    # Training
    batch_size: int = 16
    epochs: int = 200
    lr: float = 1e-3
    lr_gamma: float = 0.98
    weight_decay: float = 0.0
    num_workers: int = 0

    # Model (Table III)
    channels: int = 256
    adtm_kernel: int = 5
    atm_kernel: int = 5
    sa_groups: int = 8

    primary_caps_dim: int = 8
    primary_caps_types: int = 32
    sar_caps_dim: int = 16
    routing_iters: int = 3

    # Loss (Table III)
    m_plus: float = 0.9
    m_minus: float = 0.1
    lambda_: float = 0.5
    kappa: float = 0.392

    # Runtime
    device: Literal["auto", "cpu", "cuda"] = "auto"
    amp: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_root"] = str(self.data_root)
        d["split_json"] = str(self.split_json)
        return d
