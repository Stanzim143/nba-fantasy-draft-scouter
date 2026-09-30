"""One-command dataset rebuild: `python -m src.ops.bootstrap` (or `./dev bootstrap`).

The repository never ships data (see DATA.md); this runs the ingest steps in order to reconstruct it
under `NBA_DATA_DIR` (default `~/dev-data/nba-fantasy-2026`) and then builds a draft board:

1. NBA game logs        `src.ingest.nba_stats`        (stats.nba.com via nba_api; slow, throttled, cached)
2. Offseason + ADP      `src.ingest.preseason_refresh` (rosters, rookies, Summer League/preseason, ESPN ADP)
3. Draft board          `src.value.board`

Each step is skipped when its output already exists (use `--force` to redo), so re-running resumes.
Yahoo / FantasyPros rankings are manual exports and optional; see DATA.md.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.contracts import data_dir

DEFAULT_SEASONS = "2015-16:2025-26"
DEFAULT_TARGET = "2026-27"


def build_steps(seasons: str, target: str, base: Path, board_out: Path) -> list[tuple[str, list[str], Path]]:
    """(label, argv, sentinel path whose existence means the step is done)."""
    py = sys.executable
    return [
        ("NBA game logs", [py, "-m", "src.ingest.nba_stats", "--seasons", seasons],
         base / "processed" / "game_logs.parquet"),
        ("Offseason layer + ESPN ADP", [py, "-m", "src.ingest.preseason_refresh", "--season", target],
         base / "processed" / "adp.parquet"),
        ("Draft board", [py, "-m", "src.value.board", "--season", target, "--model", "baseline",
                         "--adp", str(base / "processed" / "adp.parquet"), "--out", str(board_out)],
         board_out),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", default=DEFAULT_SEASONS, help=f"history to pull (default {DEFAULT_SEASONS}); "
                    "use a short window such as 2023-24:2025-26 for a quick trial")
    ap.add_argument("--target", default=DEFAULT_TARGET, help="season to project (default %(default)s)")
    ap.add_argument("--out", type=Path, default=Path("board.csv"), help="board CSV to write")
    ap.add_argument("--force", action="store_true", help="re-run steps whose output already exists")
    ap.add_argument("--dry-run", action="store_true", help="print the steps without running them")
    args = ap.parse_args(argv)

    base = data_dir()
    steps = build_steps(args.seasons, args.target, base, args.out)
    print(f"data dir: {base}\n(no data is committed to git; see DATA.md for sources and terms)\n")
    for i, (label, cmd, sentinel) in enumerate(steps, 1):
        if sentinel.exists() and not args.force:
            print(f"[{i}/{len(steps)}] {label}: already done ({sentinel.name}), skipping (--force to redo)")
            continue
        print(f"[{i}/{len(steps)}] {label}: {' '.join(cmd[1:])}")
        if args.dry_run:
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"\nstep failed: {label} (exit {rc}). Fix the cause and re-run; finished steps are kept.",
                  file=sys.stderr)
            return rc
    print("\nDone. Explore: ./dev app   (or: streamlit run src/app/draft_board.py)")
    print("Optional manual exports (Yahoo / FantasyPros rankings): see DATA.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
