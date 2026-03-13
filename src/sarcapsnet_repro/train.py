from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .config import ExperimentConfig
    from .data.sar_acd_dataset import SarAcdDataset
    from .data.splits import make_limited_train_subset_dict
    from .losses import margin_loss
    from .models.sarcapsnet import SARCapsNet
    from .utils.io import atomic_write_json, write_csv_rows
    from .utils.metrics import confusion_matrix
    from .utils.seed import set_seed
except ImportError as e:  # pragma: no cover
    # Allow running as a script: `python src/sarcapsnet_repro/train.py ...`
    if "attempted relative import" not in str(e):
        raise
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # add `src/`

    from sarcapsnet_repro.config import ExperimentConfig
    from sarcapsnet_repro.data.sar_acd_dataset import SarAcdDataset
    from sarcapsnet_repro.data.splits import make_limited_train_subset_dict
    from sarcapsnet_repro.losses import margin_loss
    from sarcapsnet_repro.models.sarcapsnet import SARCapsNet
    from sarcapsnet_repro.utils.io import atomic_write_json, write_csv_rows
    from sarcapsnet_repro.utils.metrics import confusion_matrix
    from sarcapsnet_repro.utils.seed import set_seed


def _resolve_device(device: str) -> torch.device:
    """根据字符串配置解析实际训练设备。

    - 当指定为 "cpu" 或 "cuda" 时尽量使用对应设备；
    - 当为 "auto" 或其他值时，优先选择可用的 GPU，否则回落到 CPU。
    """
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(
    model: SARCapsNet,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict:
    """在验证/测试集上评估模型分类精度与混淆矩阵。

    仅前向推理，不进行梯度计算。
    """
    model.eval()
    ys: list[int] = []
    ps: list[int] = []
    total = 0
    correct = 0
    for x, y, _ in loader:
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
    return {"acc": acc, "cm": cm}


def train_one_epoch(
    model: SARCapsNet,
    loader: DataLoader,
    device: torch.device,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    cfg: ExperimentConfig,
    grad_clip: float,
    debug_finite: bool,
) -> dict:
    """在给定数据加载器上训练模型一个 epoch。

    使用胶囊网络的 margin loss 与重建损失联合优化，并可选启用 AMP 与梯度裁剪。
    返回该 epoch 的平均 loss 与训练集分类精度。
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    # 是否启用混合精度训练（AMP）
    use_amp = scaler is not None and scaler.is_enabled()

    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)
        bs = int(y.shape[0])

        optim.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, v, recon = model(x, y=y, debug=debug_finite)

        # Always compute losses in float32 for stability.
        l_margin = margin_loss(
            v.float(),
            y,
            m_plus=cfg.m_plus,
            m_minus=cfg.m_minus,
            lambda_=cfg.lambda_,
        )
        l_recon = F.mse_loss(recon.float(), x.float())
        loss = l_margin + cfg.kappa * l_recon

        # 数值稳定性检查：一旦出现 NaN/Inf，打印详细诊断信息并中止训练
        if not torch.isfinite(loss):
            try:
                with torch.no_grad():
                    model(x, y=y, debug=True)
            except Exception as e:  # noqa: BLE001
                print("Failed to locate first non-finite stage:", repr(e))

            diag = {
                "loss": float(loss.detach().cpu().item()),
                "logits_finite": bool(torch.isfinite(logits).all().item()),
                "v_finite": bool(torch.isfinite(v).all().item()),
                "recon_finite": bool(torch.isfinite(recon).all().item()),
                "logits_abs_max": float(logits.detach().abs().max().cpu().item()),
                "v_abs_max": float(v.detach().abs().max().cpu().item()),
                "recon_abs_max": float(recon.detach().abs().max().cpu().item()),
            }
            print("Non-finite loss detected. Diagnostics:", diag)
            for name, param in model.named_parameters():
                if not torch.isfinite(param).all():
                    print(f"Parameter has NaN/Inf: {name}")
                    break
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    print(f"Gradient has NaN/Inf: {name}")
                    break
            raise FloatingPointError("Non-finite loss detected (NaN/Inf).")

        # 根据是否使用 AMP 选择对应的反向与更新逻辑，并在需要时做梯度裁剪
        if not use_amp:
            loss.backward()
            if grad_clip and grad_clip > 0:
                clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            optim.step()
        else:
            assert scaler is not None
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optim)
                clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            scaler.step(optim)
            scaler.update()

        total_loss += float(loss.detach().cpu().item()) * bs
        total_correct += int((logits.detach().argmax(dim=1) == y).sum().item())
        total_seen += bs

    return {"loss": total_loss / total_seen, "acc": float(total_correct) / total_seen}


def main(argv: list[str] | None = None) -> int:
    """训练 SARCapsNet 在 SAR-ACD 数据集上的命令行入口。

    整体流程：
    1. 解析命令行参数、设置随机种子和 PyTorch 选项；
    2. 读取数据划分文件，并按 limited_rate 生成子集（可做小数据实验）；
    3. 构建实验配置、数据集/DataLoader、模型、优化器与学习率调度；
    4. 迭代多轮训练 + 测试评估，记录指标到 CSV，并在精度提升时保存最佳模型与混淆矩阵。
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("dataset/SAR-ACD"))
    p.add_argument("--split", type=Path, default=Path("splits/sar_acd_seed0.json"))
    p.add_argument("--limited-rate", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--input-size", type=int, default=28)
    p.add_argument(
        "--resize-mode",
        type=str,
        default="letterbox",
        choices=["stretch", "letterbox"],
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-gamma", type=float, default=0.98)
    p.add_argument("--num-workers", type=int, default=0)
    # 默认关闭 AMP，需要时显式添加 --amp 开启
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--detect-anomaly", action="store_true")
    p.add_argument("--debug-finite", action="store_true")
    p.add_argument("--out-dir", type=Path, default=Path("runs"))
    p.add_argument("--run-name", type=str, default="")
    args = p.parse_args(argv)

    # 固定随机种子，保证结果可复现；可根据需要选择确定性模式
    set_seed(args.seed, deterministic=True)
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    base_split = json.loads(args.split.read_text(encoding="utf-8"))
    # 读取基础划分文件；如设置 limited_rate<1.0，则对子集采样做有限训练
    if args.limited_rate < 1.0:
        split_payload = make_limited_train_subset_dict(
            base_split, rate=args.limited_rate, seed=args.seed
        )
    else:
        split_payload = base_split

    cfg = ExperimentConfig(
        data_root=args.data_root,
        split_json=args.split,
        limited_rate=float(args.limited_rate),
        seed=int(args.seed),
        input_size=int(args.input_size),
        resize_mode=str(args.resize_mode),
        num_classes=len(split_payload["classes"]),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        lr_gamma=float(args.lr_gamma),
        num_workers=int(args.num_workers),
        device=args.device,
        amp=bool(args.amp),
    )

    # 根据配置解析真实训练设备，并决定是否启用 AMP 以及 GradScaler
    device = _resolve_device(cfg.device)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = (
        args.run_name.strip()
        or f"sarcapsnet_sar-acd_r{cfg.limited_rate}_s{cfg.seed}_{ts}"
    )
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 将本次运行的配置与实际使用的数据划分保存到输出目录，便于复现实验
    atomic_write_json(run_dir / "config.json", cfg.to_dict())
    atomic_write_json(run_dir / "split_used.json", split_payload)

    split_used_path = run_dir / "split_used.json"
    # 构建训练与测试数据集；`split_used.json` 明确记录了各划分的样本
    train_ds = SarAcdDataset(
        cfg.data_root,
        split_used_path,
        split="train",
        input_size=cfg.input_size,
        resize_mode=cfg.resize_mode,
    )
    test_ds = SarAcdDataset(
        cfg.data_root,
        split_used_path,
        split="test",
        input_size=cfg.input_size,
        resize_mode=cfg.resize_mode,
    )

    # DataLoader 负责按 batch 迭代数据，并在 GPU 上启用 pin_memory 提升拷贝效率
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # 构建胶囊网络模型，并按照实验配置设置通道数、注意力分组、路由迭代次数等
    model = SARCapsNet(
        num_classes=cfg.num_classes,
        channels=cfg.channels,
        sa_groups=cfg.sa_groups,
        routing_iters=cfg.routing_iters,
        input_size=cfg.input_size,
        adtm_kernel=cfg.adtm_kernel,
        atm_kernel=cfg.atm_kernel,
        primary_caps_dim=cfg.primary_caps_dim,
        primary_caps_types=cfg.primary_caps_types,
        sar_caps_dim=cfg.sar_caps_dim,
    ).to(device)

    # 使用 Adam 优化器与指数衰减学习率调度
    optim = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = ExponentialLR(optim, gamma=cfg.lr_gamma)

    best_acc = -1.0
    metrics_rows: list[dict] = []

    # 主训练循环：每个 epoch 先训练，再在测试集评估，并记录与保存最佳结果
    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            device,
            optim,
            scaler,
            cfg,
            grad_clip=float(args.grad_clip),
            debug_finite=bool(args.debug_finite),
        )
        eval_stats = evaluate(model, test_loader, device, num_classes=cfg.num_classes)
        sched.step()

        row = {
            "epoch": epoch,
            "lr": sched.get_last_lr()[0],
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "test_acc": eval_stats["acc"],
        }
        metrics_rows.append(row)
        write_csv_rows(
            run_dir / "metrics.csv",
            fieldnames=list(row.keys()),
            rows=metrics_rows,
        )

        if eval_stats["acc"] > best_acc:
            best_acc = float(eval_stats["acc"])
            ckpt = {
                "epoch": epoch,
                "best_acc": best_acc,
                "classes": split_payload["classes"],
                "config": cfg.to_dict(),
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
            }
            torch.save(ckpt, run_dir / "best.pt")
            # 记录在“最佳精度”对应的混淆矩阵，便于后续做错误分析
            np.savetxt(
                run_dir / "confusion_matrix_best.csv",
                eval_stats["cm"],
                fmt="%d",
                delimiter=",",
            )

        print(
            f"[epoch {epoch:03d}] loss={train_stats['loss']:.4f} "
            f"train_acc={train_stats['acc']:.4f} "
            f"test_acc={eval_stats['acc']:.4f} best={best_acc:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
