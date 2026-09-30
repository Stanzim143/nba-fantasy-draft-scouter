"""Console setup shared by the in-season CLIs."""
from __future__ import annotations

import sys


def use_utf8_console() -> None:
    """Print player names with accents (Jokić, Marković) instead of ``Joki?`` on a cp1252 Windows console.

    Reconfigures stdout and stderr to UTF-8 with ``errors='replace'``; a stream that cannot be reconfigured
    (already replaced by a test harness, pythonw with no console) is left alone.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")   # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass
