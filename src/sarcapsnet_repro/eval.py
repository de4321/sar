from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .data.sar_acd_dataset import SarAcdDataset
    from .models.sarcapsnet import SARCapsNet
    from .utils.metrics import confusion_matrix
except ImportError as e:  # pragma: no cover
    if "attempted relative import" not in str(e):
        raise
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # add `src/`

    from sarcapsnet_repro.data.sar_acd_dataset import SarAcdDataset
    from sarcapsnet_repro.models.sarcapsnet import SARCapsNet
    from sarcapsnet_repro.utils.metrics import confusion_matrix


def _resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--split", type=Path, default=None)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--input-size", type=int, default=None)
    p.add_argument("--resize-mode", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out-cm", type=Path, default=None)
    args = p.parse_args(argv)

    device = _resolve_device(args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_cfg = ckpt.get("config", {})
    if not isinstance(ckpt_cfg, dict):
        ckpt_cfg = {}

    input_size = (
        int(args.input_size)
        if args.input_size is not None
        else int(ckpt_cfg.get("input_size", 28))
    )
    resize_mode = args.resize_mode or str(ckpt_cfg.get("resize_mode", "letterbox"))
    if resize_mode not in ("stretch", "letterbox"):
        raise ValueError(
            f"invalid resize_mode={resize_mode!r}; expected 'stretch' or 'letterbox'"
        )

    split_path = args.split
    if split_path is None:
        candidate = args.ckpt.parent / "split_used.json"
        if candidate.exists():
            split_path = candidate
        else:
            raise ValueError(
                "--split is required when checkpoint directory has no split_used.json"
            )

    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    num_classes = len(split_payload["classes"])
    ckpt_classes = ckpt.get("classes")
    if ckpt_classes is not None and list(ckpt_classes) != list(split_payload["classes"]):
        raise ValueError(
            "class order mismatch between checkpoint and split file; "
            "use the same split/classes used during training."
        )

    data_root = (
        args.data_root
        if args.data_root is not None
        else Path(str(ckpt_cfg.get("data_root", "dataset/SAR-ACD")))
    )

    test_ds = SarAcdDataset(
        data_root,
        split_path,
        split="test",
        input_size=input_size,
        resize_mode=resize_mode,
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = SARCapsNet(
        num_classes=num_classes,
        channels=int(ckpt_cfg.get("channels", 256)),
        sa_groups=int(ckpt_cfg.get("sa_groups", 8)),
        routing_iters=int(ckpt_cfg.get("routing_iters", 3)),
        input_size=input_size,
        adtm_kernel=int(ckpt_cfg.get("adtm_kernel", 5)),
        atm_kernel=int(ckpt_cfg.get("atm_kernel", 5)),
        primary_caps_dim=int(ckpt_cfg.get("primary_caps_dim", 8)),
        primary_caps_types=int(ckpt_cfg.get("primary_caps_types", 32)),
        sar_caps_dim=int(ckpt_cfg.get("sar_caps_dim", 16)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ys: list[int] = []
    ps: list[int] = []
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits, _, _ = model(x, y=None)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
            ys.extend(y.cpu().numpy().tolist())
            ps.extend(pred.cpu().numpy().tolist())

    acc = float(correct) / float(total)
    cm = confusion_matrix(ys, ps, num_classes=num_classes)
    out_cm = args.out_cm if args.out_cm is not None else args.ckpt.parent / "confusion_matrix_eval.csv"
    out_cm.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_cm, cm, fmt="%d", delimiter=",")
    print(f"test_acc={acc:.4f}")
    print(f"confusion_matrix={out_cm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
