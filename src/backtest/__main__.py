"""``python -m src.backtest`` entry point."""
from __future__ import annotations

import sys

from src.backtest.runner import main

if __name__ == "__main__":
    sys.exit(main())
