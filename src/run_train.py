from __future__ import annotations

import sys

from sarcapsnet_repro.train import main as train_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--cli":
        return train_main(args[1:])
    from sarcapsnet_repro.train_gui import main as train_gui_main

    return train_gui_main(args)

if __name__ == "__main__":
    raise SystemExit(main())
