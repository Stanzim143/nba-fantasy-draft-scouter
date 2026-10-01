"""The transaction tracker CLI: what changed in the league, and does it matter for the draft board (ADR 0017).

    python -m src.ops.txn_watch                      # new since you last acknowledged (first run: the last 7 days)
    python -m src.ops.txn_watch --refresh --ack      # pull the feed first, then mark what you have just seen as read
    python -m src.ops.txn_watch --days 30 --team MIA --all
    python -m src.ops.txn_watch --player Giannis
    python -m src.ops.txn_watch --staff              # coach / front-office moves

The ledger is filled by ``python -m src.ingest.espn_transactions`` (the daily refresh runs it too). "New" means first
seen by the ledger after the watermark in ``<data>/ops/txn_watch.json``; ``--ack`` moves the watermark, so a plain run
is read-only. By default only players the board cares about are listed (top ``RELEVANT_RANK``); ``--all`` lists
everyone, including the camp signings and cuts of fringe players.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.contracts import data_dir
from src.features import txn_impact as ti

WATERMARK_FILE = "txn_watch.json"
DEFAULT_MODEL = "baseline_offseason_debut"


def watermark_path(base: Path) -> Path:
    return base / "ops" / WATERMARK_FILE


def read_watermark(base: Path) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(json.loads(watermark_path(base).read_text("utf-8"))["acked_first_seen"])
    except (OSError, ValueError, KeyError):
        return None


def write_watermark(base: Path, ts: pd.Timestamp) -> None:
    from src.ops.daily_refresh import atomic_write_text

    atomic_write_text(watermark_path(base), json.dumps({"acked_first_seen": pd.Timestamp(ts).isoformat()}))


def select_window(ledger: pd.DataFrame, *, days: int | None, since: date | None, watermark: pd.Timestamp | None,
                  today: date) -> tuple[pd.DataFrame, str]:
    """The ledger rows to show and a label for how they were chosen. Explicit ``--days``/``--since`` win; otherwise rows
    first seen after the watermark; with no watermark yet, the last 7 days by transaction date."""
    if since is not None:
        return ledger[ledger["txn_date"] >= pd.Timestamp(since)], f"since {since}"
    if days is not None:
        return ledger[ledger["txn_date"] >= pd.Timestamp(today - timedelta(days=days))], f"last {days} days"
    if watermark is not None:
        return ledger[ledger["first_seen"] > watermark], f"new since {watermark:%Y-%m-%d %H:%M} UTC"
    return ledger[ledger["txn_date"] >= pd.Timestamp(today - timedelta(days=7))], "last 7 days (nothing acknowledged yet)"


def load_board_and_roster(season: str, model: str, board_csv: Path | None, base: Path):
    """Best-effort: a board (CSV or built live) and the newest roster snapshot. Either can be None; the tracker still runs."""
    board = roster = None
    notes: list[str] = []
    if board_csv is not None:
        try:
            board = pd.read_csv(board_csv)
        except (OSError, ValueError) as exc:
            notes.append(f"could not read --board-csv ({exc}); moves are shown without ranks")
    else:
        try:
            from src.app.loader import load_board

            board = load_board(season, model, data_dir=base)
        except Exception as exc:                              # noqa: BLE001 - the board is an extra, never fatal
            notes.append(f"no board ({type(exc).__name__}: {exc}); moves are shown without ranks")
    try:
        from src.ingest.nba_incoming import latest_snapshot, read_roster_snapshots

        roster = latest_snapshot(read_roster_snapshots(base))
    except (FileNotFoundError, ImportError, OSError) as exc:
        notes.append(f"no roster snapshot ({exc}); teammate context omitted")
    return board, roster, notes


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.ops.txn_watch", description=__doc__.split("\n\n")[0])
    p.add_argument("--season", default="2026-27")
    p.add_argument("--model", default=DEFAULT_MODEL, help="board model used for ranks (default %(default)s)")
    p.add_argument("--board-csv", type=Path, default=None, help="use this exported board instead of building one")
    p.add_argument("--days", type=int, default=None, help="show the last N days by transaction date")
    p.add_argument("--since", default=None, help="show moves on or after YYYY-MM-DD")
    p.add_argument("--team", action="append", default=None, help="only moves involving this team abbreviation (repeatable)")
    p.add_argument("--kind", action="append", default=None,
                   help="only this kind: trade, signed, resigned, extended, converted, claimed, waived")
    p.add_argument("--player", default=None, help="only players whose name contains this text")
    p.add_argument("--all", action="store_true", help="include players outside the board's top ranks")
    p.add_argument("--staff", action="store_true", help="show coach / front-office moves instead of player moves")
    p.add_argument("--refresh", action="store_true", help="pull the last 14 days from ESPN first")
    p.add_argument("--ack", action="store_true", help="mark everything currently in the ledger as seen")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--offline", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    base = args.data_dir or data_dir()
    if args.refresh:
        from src.ingest import espn_transactions as et
        from src.ingest.http_cache import CachedHttpClient

        client = CachedHttpClient(base / "raw" / "espn", offline=True if args.offline else None, min_interval=2.0)
        rc = et.main(["--days", "14", "--data-dir", str(base)] + (["--offline"] if args.offline else []), client=client)
        if rc != 0:
            print("refresh failed; showing what the ledger already has", file=sys.stderr)
    from src.ingest.espn_transactions import read_ledger

    try:
        ledger = read_ledger(base)
    except FileNotFoundError as exc:
        print(f"{exc}", file=sys.stderr)
        return 2
    since = date.fromisoformat(args.since) if args.since else None
    window, label = select_window(ledger, days=args.days, since=since, watermark=read_watermark(base), today=date.today())
    print(f"Transactions: {label}  (ledger through {ledger['txn_date'].max():%Y-%m-%d}, {len(ledger)} rows)")

    if args.staff:
        st = ti.staff_events(window)
        if args.team:
            st = st[st["team_abbr"].isin({t.upper() for t in args.team})]
        if st.empty:
            print("no coach or front-office moves in this window")
        for r in st.itertuples(index=False):
            print(f"{r.txn_date:%Y-%m-%d}  {r.team_abbr:<4} {r.kind.upper():<15} {r.person} ({r.role.replace('_', ' ')})")
    else:
        board, roster, notes = load_board_and_roster(args.season, args.model, args.board_csv, base)
        for n in notes:
            print(f"note: {n}")
        ann = ti.annotate(ti.events(window), board, roster)
        ann = ti.filter_events(ann, teams=args.team, kinds=args.kind, player=args.player)
        relevant_only = not args.all and board is not None and not args.player
        lines = ti.digest(ann, relevant_only=relevant_only, limit=None)
        print("\n".join(lines) if lines else "no moves in this window")
    if args.ack:
        write_watermark(base, pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)))
        print("\nacknowledged: the next plain run shows only newer moves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
