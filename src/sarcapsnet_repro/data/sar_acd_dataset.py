from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _resize_square(
    img: Image.Image,
    size: int,
    mode: Literal["stretch", "letterbox"] = "letterbox",
) -> Image.Image:
    size = int(size)
    if mode == "stretch":
        return img.resize((size, size), resample=Image.BILINEAR)
    if mode != "letterbox":
        raise ValueError(f"unknown resize mode: {mode}")

    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid image size: {img.size}")

    scale = min(size / float(w), size / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = img.resize((new_w, new_h), resample=Image.BILINEAR)

    canvas = Image.new("L", (size, size), color=0)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


class SarAcdDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split_json: Path,
        split: Literal["train", "test"],
        input_size: int = 28,
        resize_mode: Literal["stretch", "letterbox"] = "letterbox",
    ) -> None:
        self.data_root = Path(data_root)
        self.images_root = self.data_root / "images"
        self.input_size = int(input_size)
        self.resize_mode = resize_mode

        split_path = Path(split_json)
        payload = json.loads(split_path.read_text(encoding="utf-8"))

        self.classes: list[str] = payload["classes"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        if split not in ("train", "test"):
            raise ValueError(f"Unknown split: {split}")

        split_map: dict[str, list[str]] = payload[split]
        samples: list[tuple[str, int]] = []
        for class_name in self.classes:
            relpaths = split_map.get(class_name, [])
            label = self.class_to_idx[class_name]
            for relpath in relpaths:
                samples.append((relpath, label))

        if not samples:
            raise ValueError(f"No samples found for split={split} in {split_path}")

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        relpath, label = self.samples[index]
        path = self.images_root / relpath
        if not path.exists():
            raise FileNotFoundError(str(path))

        with Image.open(path) as raw_img:
            img = raw_img.convert("L")
            img = _resize_square(img, self.input_size, mode=self.resize_mode)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]

        return x, int(label), relpath
