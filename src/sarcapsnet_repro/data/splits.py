from __future__ import annotations

import json
import random
from pathlib import Path

SAR_ACD_CLASSES = ["A220", "A320321", "A330", "ARJ21", "Boeing737", "Boeing787"]


def make_split_sar_acd_dict(
    data_root: Path,
    seed: int = 0,
    train_ratio: float = 0.8,
    classes: list[str] | None = None,
) -> dict:
    data_root = Path(data_root)
    images_root = data_root / "images"
    if not images_root.exists():
        raise FileNotFoundError(str(images_root))

    if classes is None:
        classes = SAR_ACD_CLASSES

    rng = random.Random(int(seed))

    train: dict[str, list[str]] = {}
    test: dict[str, list[str]] = {}

    for class_name in classes:
        class_dir = images_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(str(class_dir))

        files = sorted([p.name for p in class_dir.glob("*.jpg")])
        if not files:
            raise ValueError(f"No .jpg files found in {class_dir}")

        rng.shuffle(files)
        n_train = int(len(files) * float(train_ratio))
        n_train = max(1, min(n_train, len(files) - 1))

        train[class_name] = [f"{class_name}/{name}" for name in files[:n_train]]
        test[class_name] = [f"{class_name}/{name}" for name in files[n_train:]]

    payload = {
        "dataset": "SAR-ACD",
        "seed": int(seed),
        "train_ratio": float(train_ratio),
        "classes": list(classes),
        "train": train,
        "test": test,
    }
    return payload


def make_limited_train_subset_dict(
    base_split: dict,
    rate: float,
    seed: int = 0,
) -> dict:
    rate = float(rate)
    if not (0.0 < rate <= 1.0):
        raise ValueError(f"rate must be in (0,1], got {rate}")

    rng = random.Random(int(seed))
    classes = list(base_split["classes"])

    train_base: dict[str, list[str]] = base_split["train"]
    train_sub: dict[str, list[str]] = {}

    for class_name in classes:
        items = list(train_base[class_name])
        rng.shuffle(items)
        n = int(len(items) * rate)
        n = max(1, n)
        train_sub[class_name] = items[:n]

    payload = {
        "dataset": base_split.get("dataset", "SAR-ACD"),
        "base_seed": base_split.get("seed", None),
        "subset_seed": int(seed),
        "train_ratio": base_split.get("train_ratio", None),
        "limited_rate": rate,
        "classes": classes,
        "train": train_sub,
        "test": base_split["test"],
    }
    return payload


def save_split_json(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

