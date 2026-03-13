from __future__ import annotations

import argparse
from pathlib import Path

from sarcapsnet_repro.data.splits import make_split_sar_acd_dict, save_split_json


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("dataset/SAR-ACD"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("splits/sar_acd_seed0.json"))
    args = p.parse_args()

    payload = make_split_sar_acd_dict(args.data_root, seed=args.seed, train_ratio=0.8)
    save_split_json(payload, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

