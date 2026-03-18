from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data.sar_acd_dataset import _resize_square
from .models.sarcapsnet import SARCapsNet


def resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_latest_checkpoint(runs_dir: Path = Path("runs")) -> Path | None:
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return None

    candidates = sorted(
        runs_dir.glob("**/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


@dataclass(frozen=True)
class PredictionResult:
    image_path: Path
    predicted_index: int
    predicted_class: str
    confidence: float
    probabilities: list[dict[str, float | str]]
    device: str
    model_input_image: np.ndarray
    attention_map: np.ndarray
    attention_peak: tuple[int, int]


class SARCapsPredictor:
    def __init__(self, ckpt_path: Path, device: str = "auto") -> None:
        self.ckpt_path = Path(ckpt_path)
        if not self.ckpt_path.exists():
            raise FileNotFoundError(str(self.ckpt_path))

        self.device = resolve_device(device)
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        ckpt_cfg = ckpt.get("config", {})
        if not isinstance(ckpt_cfg, dict):
            ckpt_cfg = {}

        self.input_size = int(ckpt_cfg.get("input_size", 28))
        self.resize_mode = str(ckpt_cfg.get("resize_mode", "letterbox"))
        if self.resize_mode not in ("stretch", "letterbox"):
            raise ValueError(
                f"invalid resize_mode={self.resize_mode!r}; expected 'stretch' or 'letterbox'"
            )

        split_path = self.ckpt_path.parent / "split_used.json"
        if split_path.exists():
            split_payload = json.loads(split_path.read_text(encoding="utf-8"))
            self.classes = list(split_payload["classes"])
        else:
            self.classes = list(ckpt.get("classes") or [])

        if not self.classes:
            raise ValueError("No class labels found in checkpoint or split_used.json")

        self.model = SARCapsNet(
            num_classes=len(self.classes),
            channels=int(ckpt_cfg.get("channels", 256)),
            sa_groups=int(ckpt_cfg.get("sa_groups", 8)),
            routing_iters=int(ckpt_cfg.get("routing_iters", 3)),
            input_size=self.input_size,
            adtm_kernel=int(ckpt_cfg.get("adtm_kernel", 5)),
            atm_kernel=int(ckpt_cfg.get("atm_kernel", 5)),
            primary_caps_dim=int(ckpt_cfg.get("primary_caps_dim", 8)),
            primary_caps_types=int(ckpt_cfg.get("primary_caps_types", 32)),
            sar_caps_dim=int(ckpt_cfg.get("sar_caps_dim", 16)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def preprocess_image(self, image_path: Path) -> torch.Tensor:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(str(image_path))

        with Image.open(image_path) as raw_img:
            img = raw_img.convert("L")
            img = _resize_square(img, self.input_size, mode=self.resize_mode)

        tensor = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor

    @staticmethod
    def _attention_from_feature(
        feature: torch.Tensor | None,
        target_hw: int,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        if feature is None or feature.ndim != 4 or feature.shape[0] == 0:
            fallback = np.zeros((target_hw, target_hw), dtype=np.float32)
            center = (target_hw // 2, target_hw // 2)
            return fallback, center

        with torch.no_grad():
            heat = feature[0].abs().mean(dim=0, keepdim=True).unsqueeze(0)
            heat = F.interpolate(
                heat,
                size=(target_hw, target_hw),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            min_v = torch.min(heat)
            max_v = torch.max(heat)
            denom = max_v - min_v
            if float(denom.item()) > 1e-8:
                heat = (heat - min_v) / denom
            else:
                heat = torch.zeros_like(heat)

            flat_idx = int(torch.argmax(heat).item())
            width = int(heat.shape[1])
            peak = (flat_idx // width, flat_idx % width)
            return heat.cpu().numpy().astype(np.float32), peak

    @torch.no_grad()
    def predict_image(self, image_path: Path, top_k: int = 3) -> PredictionResult:
        x = self.preprocess_image(image_path).to(self.device)
        sa_feature: torch.Tensor | None = None

        def _capture_sa(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            nonlocal sa_feature
            sa_feature = output.detach()

        handle = self.model.sa.register_forward_hook(_capture_sa)
        try:
            logits, _, _ = self.model(x, y=None)
        finally:
            handle.remove()

        probs = torch.softmax(logits[0], dim=0)
        top_probs, top_indices = torch.topk(probs, k=min(int(top_k), len(self.classes)))

        pred_index = int(torch.argmax(probs).item())
        pred_class = self.classes[pred_index]
        confidence = float(probs[pred_index].item())

        rows: list[dict[str, float | str]] = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist(), strict=True):
            rows.append(
                {
                    "class_name": self.classes[int(idx)],
                    "index": int(idx),
                    "probability": float(prob),
                }
            )

        model_input = x[0, 0].detach().cpu().numpy().astype(np.float32)
        attention_map, attention_peak = self._attention_from_feature(
            sa_feature,
            target_hw=self.input_size,
        )

        return PredictionResult(
            image_path=Path(image_path),
            predicted_index=pred_index,
            predicted_class=pred_class,
            confidence=confidence,
            probabilities=rows,
            device=str(self.device),
            model_input_image=model_input,
            attention_map=attention_map,
            attention_peak=attention_peak,
        )
