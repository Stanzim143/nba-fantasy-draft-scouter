"""Data prep for the app's Transactions tab (ADR 0017). No Streamlit import: plain functions, tested offline.

The tab reads the local ledger (``python -m src.ingest.espn_transactions``, also run by the daily refresh); it never makes a
network request. It joins the *already loaded* board, so ranks match what the rest of the app shows.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.features import txn_impact as ti

DISPLAY_COLUMNS = ["txn_date", "kind", "person", "position", "rank", "adp", "from_abbr", "to_abbr", "detail", "context"]
KIND_CHOICES = ["trade", "signed", "resigned", "extended", "converted", "claimed", "waived"]


class LedgerUnavailable(Exception):
    """No ledger yet; the message is safe to show verbatim."""


def load_ledger(data_dir: Path | None = None) -> pd.DataFrame:
    from src.ingest.espn_transactions import read_ledger

    try:
        return read_ledger(data_dir)
    except FileNotFoundError as exc:
        raise LedgerUnavailable(f"{exc}") from exc


def load_roster(data_dir: Path | None = None) -> pd.DataFrame | None:
    try:
        from src.ingest.nba_incoming import latest_snapshot, read_roster_snapshots

        return latest_snapshot(read_roster_snapshots(data_dir))
    except (FileNotFoundError, OSError):
        return None


def moves(ledger: pd.DataFrame, board: pd.DataFrame | None, roster: pd.DataFrame | None, *, days: int = 14,
          teams: list[str] | None = None, kinds: list[str] | None = None, player: str | None = None,
          relevant_only: bool = True, today: date | None = None) -> pd.DataFrame:
    """The player moves in the last ``days`` days, annotated with the board, filtered, best-ranked first within each day."""
    since = pd.Timestamp((today or date.today()) - timedelta(days=days))
    ann = ti.annotate(ti.events(ledger[ledger["txn_date"] >= since]), board, roster)
    ann = ti.filter_events(ann, teams=teams, kinds=kinds, player=player)
    if relevant_only and board is not None and not player:
        ann = ann[ann["relevant"]]
    return ann.sort_values(["txn_date", "rank"], ascending=[False, True], na_position="last", kind="mergesort").reset_index(drop=True)


def staff(ledger: pd.DataFrame, *, days: int = 120, teams: list[str] | None = None, today: date | None = None) -> pd.DataFrame:
    since = pd.Timestamp((today or date.today()) - timedelta(days=days))
    out = ti.staff_events(ledger[ledger["txn_date"] >= since])
    if teams:
        out = out[out["team_abbr"].isin({t.upper() for t in teams})]
    return out[["txn_date", "team_abbr", "kind", "person", "role"]].reset_index(drop=True)


def freshness(ledger: pd.DataFrame, today: date | None = None) -> str:
    last_seen = ledger["last_seen"].max()
    through = ledger["txn_date"].max()
    age = (pd.Timestamp(today or date.today()) - pd.Timestamp(last_seen).normalize()).days
    return (f"Feed through {through:%Y-%m-%d}; the ledger was last refreshed {last_seen:%Y-%m-%d %H:%M} UTC"
            + (f" ({age} days ago: run python -m src.ops.txn_watch --refresh)" if age >= 2 else ""))
