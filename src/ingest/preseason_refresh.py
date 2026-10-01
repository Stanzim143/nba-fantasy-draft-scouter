"""One command to bring the data current before a draft: roster snapshot, Summer League + preseason, ADP (ADR 0012).

    python -m src.ingest.preseason_refresh [--season 2026-27] [--only roster,offseason,adp] [--offline]

Run it as often as you like in the days before the draft (it is idempotent, cached and polite), and once more right
before it. The three steps depend on each other in this order, which is why they are one command:

1. **roster**  (``nba_incoming``): today's roster snapshot, and any newly drafted rookies added to ``players``.
   It runs first because step 3 maps ESPN's ADP rows onto ``players``: a rookie missing from ``players`` is an
   unmapped ADP row and never reaches the board.
2. **offseason**  (``nba_offseason``): the live season's Summer League and preseason box scores, always re-downloaded
   (a cached "0 preseason games" from September must not mask the games played in October).
3. **adp**  (``espn_adp``): a fresh pull of the live season's ESPN player universe, then the ADP table and id map are
   rebuilt from the cache. Only the live season is re-downloaded; ESPN has already wiped 2025-26, so history is
   never refreshed.

A failing step never stops the others; the exit code is 1 if any failed. Afterwards::

    python -m src.value.breakouts --season 2026-27 --out reports/watchlist.csv
    streamlit run src/app/draft_board.py
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from src.contracts import ContractError, data_dir, season_start, season_str
from src.ingest import espn_adp, espn_status, espn_transactions, nba_incoming, nba_offseason, nba_profiles, wiki_coaches
from src.ingest import nba_transform as tf
from src.ingest.http_cache import CachedHttpClient, HttpCacheError
from src.ingest.nba_client import NBAClient, NBAClientError

STEPS = ("roster", "offseason", "adp")
# Optional steps (ADR 0016), opt-in with ``--with-extras`` or ``--only``; they always run after the three above.
EXTRA_STEPS = ("profiles", "status", "transactions", "coaches")
ALL_STEPS = STEPS + EXTRA_STEPS
HANDLED = (NBAClientError, nba_incoming.IncomingError, nba_offseason.OffseasonError, nba_profiles.ProfilesError, HttpCacheError,
           espn_adp.EspnAdpError, espn_transactions.EspnTransactionsError, wiki_coaches.WikiCoachesError, ContractError, FileNotFoundError, tf.TransformError, ValueError, OSError)


@dataclass
class StepResult:
    name: str
    ok: bool
    seconds: float
    summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


# --------------------------------------------------------------------------- steps

def step_roster(season: str, base: Path, nba: NBAClient, log: Callable[[str], None]) -> dict[str, Any]:
    r = nba_incoming.run_ingest(season, nba, base, log=log)
    return {"rostered": r.rostered, "rookies_added": r.added_players, "moves_since_previous_snapshot": r.moves_since_previous,
            "debuts_not_projected": r.earlier_draft_debut_names, "missing_birthdates": r.missing_birthdates}


def step_offseason(season: str, base: Path, nba: NBAClient, log: Callable[[str], None]) -> dict[str, Any]:
    r = nba_offseason.run_ingest([season], nba, base, log=log, live_start=season_start(season))
    out: dict[str, Any] = {}
    for ctx in nba_offseason.CONTEXTS:
        e = r.events[f"{season}/{ctx}"]
        out[ctx] = {"games": e["n_games"], "players": e["n_players"], "last_date": e["last_date"]}
    return out


def step_adp(season: str, base: Path, espn: CachedHttpClient, fp: CachedHttpClient,
             log: Callable[[str], None]) -> dict[str, Any]:
    """Refresh only the live season's ESPN payload, then rebuild the ADP table and id map from the cache."""
    espn_adp.fetch_espn_players(espn, espn_adp.espn_season_id(season), refresh=not espn.offline)
    first = season_str(nba_offseason.MIN_SEASON_START)
    seasons = espn_adp.parse_seasons(f"{first}:{season}")
    r = espn_adp.run_ingest(seasons, espn, base, fp_client=fp, log=log)
    adp = pd.read_parquet(espn_adp.adp_path(base))
    from src.store import read_table

    ids = set(read_table("player_id_map", base)["source_id"].astype(str))
    live = adp[(adp["season"] == season) & (adp["source"] == espn_adp.SOURCE)]
    unmapped = live[~live["source_id"].astype(str).isin(ids)]
    return {"adp_rows_for_season": len(live), "unmapped_names": sorted(unmapped["name"].astype(str)),
            "id_matching": r.match_report.summary() if r.match_report is not None else None}


def step_profiles(season: str, base: Path, nba: NBAClient, log: Callable[[str], None]) -> dict[str, Any]:
    """Country / previous-organisation profiles (no new download beyond the roster step's) and draft-combine rows."""
    r = nba_profiles.run_ingest(season, nba, base, log=log)
    return {"profiles": r.profiles, "with_birthdate": r.with_birthdate, "combine_rows": r.combine_rows,
            "combine_failed_years": r.combine_failed_years}


def step_status(season: str, base: Path, espn: CachedHttpClient, log: Callable[[str], None]) -> dict[str, Any]:
    """Archive the ESPN injury status the adp step just refreshed (a dated snapshot; ESPN keeps no history)."""
    return espn_status.run_ingest(season, espn, base, log=log)


TRANSACTIONS_LOOKBACK_DAYS = 45


def step_transactions(season: str, base: Path, espn: CachedHttpClient, log: Callable[[str], None]) -> dict[str, Any]:
    """Pull ESPN's league transactions (trades, signings, waivers, coach moves) into the append-only ledger (ADR 0017).
    A 45-day lookback covers any gap between runs; rows already in the ledger keep their ``first_seen``."""
    from datetime import date, timedelta

    season_start(season)
    today = date.today()
    r = espn_transactions.run_ingest(today - timedelta(days=TRANSACTIONS_LOOKBACK_DAYS), today, espn, base, log=log)
    return {"feed_rows": r.feed_rows, "records": r.records, "new_rows": r.new_rows, "ledger_rows": r.ledger_rows,
            "by_kind": r.by_kind, "parse": r.stats, "matched": r.match.get("summary"), "unresolved": len(r.unresolved)}


def step_coaches(season: str, base: Path, offline: bool, log: Callable[[str], None]) -> dict[str, Any]:
    """Refresh the live season's head coaches from Wikipedia's list of current head coaches (one request; ADR 0020)."""
    season_start(season)
    wiki = CachedHttpClient(base / "raw" / wiki_coaches.CACHE_DIR_NAME, offline=True if offline else None, min_interval=1.0,
                            headers={"User-Agent": wiki_coaches.UA})
    r = wiki_coaches.run_current(season, wiki, base, log=log)
    table = wiki_coaches.read_team_coaches(base)
    cur = table[(table["season"] == season) & table["is_opening"]]
    return {"teams": r["teams"], "coaches": dict(zip(cur["team_id"].astype(int).astype(str), cur["coach_name"]))}


# --------------------------------------------------------------------------- orchestration

def run_refresh(season: str, base: Path, *, nba: NBAClient, espn: CachedHttpClient, fp: CachedHttpClient,
                steps: Sequence[str] = STEPS, log: Callable[[str], None] = print) -> list[StepResult]:
    season_start(season)
    unknown = [s for s in steps if s not in ALL_STEPS]
    if unknown:
        raise ValueError(f"unknown steps {unknown}; expected some of {list(ALL_STEPS)}")
    runners: dict[str, Callable[[], dict[str, Any]]] = {
        "roster": lambda: step_roster(season, base, nba, log),
        "offseason": lambda: step_offseason(season, base, nba, log),
        "adp": lambda: step_adp(season, base, espn, fp, log),
        "profiles": lambda: step_profiles(season, base, nba, log),
        "status": lambda: step_status(season, base, espn, log),
        "transactions": lambda: step_transactions(season, base, espn, log),
        "coaches": lambda: step_coaches(season, base, espn.offline, log),
    }
    results: list[StepResult] = []
    for name in ALL_STEPS:                                 # always in dependency order, whatever order --only listed
        if name not in steps:
            continue
        log(f"== {name}")
        t0 = time.monotonic()
        try:
            summary = runners[name]()
        except HANDLED as exc:
            results.append(StepResult(name, False, time.monotonic() - t0, error=f"{type(exc).__name__}: {exc}"))
            log(f"   FAILED: {type(exc).__name__}: {exc}")
        else:
            results.append(StepResult(name, True, time.monotonic() - t0, summary))
    return results


def render_summary(season: str, results: Sequence[StepResult]) -> str:
    lines = [f"Preseason refresh for {season}"]
    for r in results:
        lines.append(f"  [{'ok' if r.ok else 'FAILED'}] {r.name} ({r.seconds:.0f}s)" + (f": {r.error}" if r.error else ""))
        s = r.summary
        if r.name == "roster" and s:
            lines.append(f"      {s['rostered']} players on rosters; +{s['rookies_added']} rookies added to the player table")
            moves = s["moves_since_previous_snapshot"]
            lines.append(f"      roster moves since the previous snapshot: {moves or 'none (or first snapshot)'}")
            if s["debuts_not_projected"]:
                lines.append("      making their NBA debut but NOT projected by baseline / baseline_offseason (drafted in an earlier "
                             f"year; baseline_offseason_debut projects them, low confidence): {', '.join(s['debuts_not_projected'])}")
        if r.name == "offseason" and s:
            for ctx, e in s.items():
                lines.append(f"      {ctx}: {e['games']} games, {e['players']} players, through {e['last_date'] or 'n/a'}")
            if not s.get("preseason", {}).get("games"):
                lines.append("      no preseason games yet: the watchlist can only use Summer League until they exist")
        if r.name == "profiles" and s:
            lines.append(f"      {s['profiles']} player profiles ({s['with_birthdate']} with a birthdate), {s['combine_rows']} draft-combine rows"
                         + (f"; combine years failed: {s['combine_failed_years']}" if s["combine_failed_years"] else ""))
        if r.name == "status" and s:
            lines.append(f"      ESPN injury status archived for {s['players']} players: {s['by_status']}")
        if r.name == "coaches" and s:
            lines.append(f"      {s['teams']} head coaches for the season (Wikipedia list of current head coaches)")
        if r.name == "transactions" and s:
            lines.append(f"      {s['new_rows']} new ledger rows ({s['records']} in the window; ledger {s['ledger_rows']}): {s['by_kind']}; players {s['matched']}")
        if r.name == "adp" and s:
            lines.append(f"      {s['adp_rows_for_season']} ADP rows for the season; unmapped: {s['unmapped_names'] or 'none'}")
    ok = all(r.ok for r in results)
    lines.append("Next: python -m src.value.breakouts --season " + season + "   |   streamlit run src/app/draft_board.py"
                 if ok else "Some steps failed; the data from the others is still current. Re-run once the cause is fixed.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.ingest.preseason_refresh", description=__doc__.split("\n\n")[0])
    p.add_argument("--season", default=season_str(nba_offseason.live_season_start()), help="the season about to start")
    p.add_argument("--only", default=",".join(STEPS), help=f"comma-separated subset of {list(ALL_STEPS)}")
    p.add_argument("--with-extras", action="store_true",
                   help=f"also run {list(EXTRA_STEPS)}: player profiles / draft combine, the ESPN injury-status archive (ADR 0016) and the league transactions ledger (ADR 0017)")
    p.add_argument("--offline", action="store_true", help="never touch the network (also NBA_OFFLINE=1)")
    p.add_argument("--data-dir", type=Path, default=None)
    return p


def main(argv: Sequence[str] | None = None, *, nba: NBAClient | None = None, espn: CachedHttpClient | None = None,
         fp: CachedHttpClient | None = None) -> int:
    args = build_parser().parse_args(argv)
    steps = [s.strip() for s in args.only.split(",") if s.strip()]
    if args.with_extras:
        steps += [s for s in EXTRA_STEPS if s not in steps]
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")   # a player name like Markovic must never crash a Windows console run
    base = args.data_dir or data_dir()
    offline = True if args.offline else None
    nba = nba or NBAClient(base / "raw" / "nba_api", offline=offline, min_interval=1.0)
    espn = espn or CachedHttpClient(base / "raw" / espn_adp.CACHE_DIR_NAME, offline=offline, min_interval=2.0)
    fp = fp or CachedHttpClient(base / "raw" / espn_adp.FP_CACHE_DIR_NAME, offline=offline, min_interval=5.0)
    try:
        results = run_refresh(args.season, base, nba=nba, espn=espn, fp=fp, steps=steps)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("\n" + render_summary(args.season, results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
