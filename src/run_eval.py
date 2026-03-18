from __future__ import annotations

import sys

from sarcapsnet_repro.eval import main as eval_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--cli":
        return eval_main(args[1:])

    from sarcapsnet_repro.eval_gui import main as eval_gui_main

    return eval_gui_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
