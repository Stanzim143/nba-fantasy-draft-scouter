"""Parse a manually-downloaded Yahoo Fantasy Basketball "Draft Analysis" workbook (.xlsx) into the
shared external-rankings shape -- see ``docs/adr/0028-external-rankings-compare.md``.

Unlike ``espn_adp.py``, this is not a live HTTP pull: the user exports the workbook from Yahoo's own
UI by hand and drops it somewhere on disk, and this module just reads that file. The workbook's
"Players" sheet is not a clean table starting at row 0 -- it carries a few banner/caption rows above
the real header (title, scoring-format note, capture-date note, a blank row), so :func:`parse_yahoo_xlsx`
locates the header row defensively by scanning for the literal cell value ``"Yahoo Display Order"``
rather than assuming a fixed number of rows to skip.

Pipeline:

    parse_yahoo_xlsx(path)                        -> DataFrame (the "parsed" shape resolve_players wants)
    load_yahoo_rankings(path, players)            -> ResolveResult (parsed, then resolved onto player_id)

``read_source_notes`` is a small best-effort helper for the workbook's "Source & Notes" sheet, kept
separate since it's informational only (shown by the app/CLI as context), not part of the resolved
contract shape.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from src.contracts import data_dir
from src.ingest.external_rankings import ResolveResult, resolve_players
from src.store import read_table

#: The literal first-cell value of the real header row in the "Players" sheet. Everything above this
#: row is banner/caption text; everything below it (until the sheet ends) is data.
HEADER_ANCHOR = "Yahoo Display Order"

PLAYERS_SHEET = "Players"
SOURCE_NOTES_SHEET = "Source & Notes"

#: Env var / default location the CLI looks for the workbook when --path is not given.
DEFAULT_PATH_ENV = "YAHOO_RANKINGS_PATH"


class YahooRankingsError(RuntimeError):
    """The workbook is missing a sheet or the expected header row -- stop rather than guess."""


def _find_header_row(raw: pd.DataFrame) -> int:
    """Scan column 0 of a header=None-read sheet for :data:`HEADER_ANCHOR`."""
    first_col = raw.iloc[:, 0]
    for i, val in enumerate(first_col):
        if isinstance(val, str) and val.strip() == HEADER_ANCHOR:
            return i
    raise YahooRankingsError(
        f"could not find header row (first cell == {HEADER_ANCHOR!r}) in the {PLAYERS_SHEET!r} sheet"
    )


def _clean_cell(val):
    """openpyxl/pandas gives us NaN for a blank cell; normalize that (and empty strings) to None."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if isinstance(val, str) and not val.strip():
        return None
    return val


def parse_yahoo_xlsx(path: Path) -> pd.DataFrame:
    """Parse the "Players" sheet of a Yahoo draft-analysis workbook into the ``resolve_players``
    "parsed" shape: ``source_id, source_name_raw, name_parsed, team, positions, status_tag, ext_rank,
    adp, ecr_vs_adp``. Row order follows Yahoo's own Rank/display order. Raises
    :class:`YahooRankingsError` if the sheet or its header row cannot be found.
    """
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None, engine="openpyxl")
    except (ValueError, KeyError) as exc:
        raise YahooRankingsError(f"could not open {path}: {exc}") from exc

    if PLAYERS_SHEET not in sheets:
        raise YahooRankingsError(
            f"{path}: no {PLAYERS_SHEET!r} sheet found (sheets present: {sorted(sheets)})"
        )
    raw = sheets[PLAYERS_SHEET]

    header_row = _find_header_row(raw)
    columns = [str(c).strip() for c in raw.iloc[header_row].tolist()]
    data = raw.iloc[header_row + 1:].copy()
    data.columns = columns
    # Drop fully-blank trailing rows (e.g. sheet padding beyond the last real player).
    data = data.dropna(how="all")

    required_cols = {"Yahoo Display Order", "Player", "Team", "Eligible Positions", "Status",
                     "Rank", "Preseason ADP", "All Drafts ADP"}
    missing = required_cols - set(data.columns)
    if missing:
        raise YahooRankingsError(f"{path}: header row missing expected column(s) {sorted(missing)}")

    out_rows = []
    for row in data.itertuples(index=False):
        d = dict(zip(data.columns, row))
        display_order = _clean_cell(d.get("Yahoo Display Order"))
        if display_order is None:
            continue  # blank/padding row that survived dropna(how="all") via other stray cells
        player = _clean_cell(d.get("Player"))
        status = _clean_cell(d.get("Status"))
        all_drafts_adp = _clean_cell(d.get("All Drafts ADP"))
        preseason_adp = _clean_cell(d.get("Preseason ADP"))
        adp = all_drafts_adp if all_drafts_adp is not None else preseason_adp
        out_rows.append({
            "source_id": str(int(display_order)),
            "source_name_raw": player,
            "name_parsed": player,
            "team": _clean_cell(d.get("Team")),
            "positions": _clean_cell(d.get("Eligible Positions")),
            "status_tag": status,
            "ext_rank": int(_clean_cell(d.get("Rank"))),
            "adp": float(adp) if adp is not None else None,
            "ecr_vs_adp": None,
        })

    cols = ["source_id", "source_name_raw", "name_parsed", "team", "positions", "status_tag",
           "ext_rank", "adp", "ecr_vs_adp"]
    return pd.DataFrame(out_rows, columns=cols)


def read_source_notes(path: Path) -> pd.DataFrame:
    """Best-effort read of the workbook's "Source & Notes" sheet as a plain two-column (Item, Details)
    frame, for the app/CLI to show as context/caveats. Returns an empty (but correctly-shaped) frame
    if the sheet is missing or doesn't parse cleanly -- this is informational only, never load-bearing.
    """
    empty = pd.DataFrame(columns=["Item", "Details"])
    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None, engine="openpyxl")
    except (ValueError, KeyError, OSError):
        return empty
    if SOURCE_NOTES_SHEET not in sheets:
        return empty
    raw = sheets[SOURCE_NOTES_SHEET]
    try:
        rows = []
        for row in raw.itertuples(index=False):
            vals = [_clean_cell(v) for v in row]
            if not any(v is not None for v in vals):
                continue
            item = vals[0] if len(vals) > 0 else None
            details = vals[1] if len(vals) > 1 else None
            rows.append({"Item": item, "Details": details})
        return pd.DataFrame(rows, columns=["Item", "Details"]) if rows else empty
    except Exception:
        return empty


def load_yahoo_rankings(path: Path, players: pd.DataFrame) -> ResolveResult:
    """Parse the workbook and resolve every row onto the canonical NBA ``player_id``."""
    return resolve_players(parse_yahoo_xlsx(path), players, source="yahoo")


# --------------------------------------------------------------------------- CLI

def default_path() -> Path:
    env = os.environ.get(DEFAULT_PATH_ENV)
    if env:
        return Path(env)
    return data_dir() / "external_rankings" / "yahoo_latest.xlsx"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.ingest.yahoo_rankings",
        description="Parse a Yahoo Fantasy Basketball draft-analysis workbook and match it onto NBA player_id.",
    )
    p.add_argument("--path", type=Path, default=None,
                  help=f"path to the .xlsx workbook (default: ${DEFAULT_PATH_ENV} or "
                       "<data-dir>/external_rankings/yahoo_latest.xlsx)")
    p.add_argument("--data-dir", type=Path, default=None,
                  help="override the data root (default: NBA_DATA_DIR or ~/dev-data/nba-fantasy-2026)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    base = args.data_dir or data_dir()
    path = args.path or default_path()
    try:
        players = read_table("players", base)
        result = load_yahoo_rankings(path, players)
    except (YahooRankingsError, FileNotFoundError, OSError) as exc:
        print(f"yahoo rankings ingest failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    unresolved = int((~result.frame["matched"]).sum())
    print(result.report.summary())
    print(f"{unresolved} row(s) unmatched/ambiguous out of {len(result.frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
