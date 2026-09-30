"""`nba-fantasy` console entry point: thin dispatcher over the existing modules.

    nba-fantasy bootstrap [args]   rebuild the dataset from public sources (src.ops.bootstrap)
    nba-fantasy app [args]         launch the Streamlit draft board (synthetic demo needs no data)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent / "app" / "draft_board.py"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cmd, rest = (args[0], args[1:]) if args else ("", [])
    if cmd == "bootstrap":
        from src.ops.bootstrap import main as bootstrap_main

        return bootstrap_main(rest)
    if cmd == "app":
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(APP), *rest])
    print(__doc__, file=sys.stderr)
    return 0 if cmd in ("-h", "--help", "help") else 2


if __name__ == "__main__":
    raise SystemExit(main())
