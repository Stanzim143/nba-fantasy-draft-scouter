"""Schedule awareness: the ``schedule_games`` table and what the league's H2H weeks make of it.

    python -m src.inseason.schedule --season 2026-27 --ingest            # pull + cache the ESPN pro schedule
    python -m src.inseason.schedule --season 2026-27 --as-of 2026-11-02  # weekly games per team, flags
    python -m src.inseason.schedule --season 2025-26 --from-team-games   # a finished season, no network

**The table.** ``schedule_games`` (``data_dir()/processed/schedule_games.parquet``) is a standalone
table under the ADR 0011 D1 pattern (like ``adp`` and ``team_transactions``): its own validation, not in
``src/contracts.py``' ``TABLES`` and not threaded through ``History``. One row per game::

    season, game_id (str), scoring_period (int, nullable), game_date (date), home_team_id, away_team_id
    (canonical NBA team ids), home_abbr, away_abbr (NBA abbreviations), source ("espn" | "team_games"),
    time_tbd (bool)

A schedule is *known in advance*, so a future game in this table is not leakage. What it must never
carry is a result, and it does not: ``statsOfficial`` and scores are ignored. The one leakage rule is on
the consumers: "games remaining after ``as_of``" is computed from dates only.

**Sources.** ESPN's ``proTeamSchedules_wl`` (``ingest``; one cached GET through ``CachedHttpClient``,
ADR 0005 D1) for the live season, and the realised ``team_games`` table for finished seasons (used by the
backtest and the real-data smoke run). ESPN pro team ids are mapped to NBA team ids through the table
below (checked against real ``team_games`` in ``tests/inseason/test_schedule_real_data.py``). Until NBA
Cup group play resolves ESPN lists 80 of 82 games per team; the two missing games are simply absent, so
weeks near Dec 7 read low, and :func:`weekly_team_table` reports each week's league-wide mean so that is
visible rather than hidden.

**Weekly view** (:func:`weekly_team_table`): per (week, team) the games, games still to play after
``as_of``, back-to-back second nights, games on "off nights" (league-wide days with few games, where an
extra player fills an otherwise empty lineup slot), and heavy/light flags on a games-per-7-days basis
(``heavy`` >= 4 in 7 days, ``light`` <= 2).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts import ContractError, data_dir, raw_dir, season_start
from src.inseason.weeks import MatchupCalendar, as_date, calendar_from_settings
from src.value.league import load_league

TABLE = "schedule_games"
SOURCE_ESPN = "espn"
SOURCE_TEAM_GAMES = "team_games"

SCHEDULE_COLUMNS: dict[str, str] = {
    "season": "str", "game_id": "str", "scoring_period": "int?", "game_date": "date",
    "home_team_id": "int", "away_team_id": "int", "home_abbr": "str", "away_abbr": "str",
    "source": "str", "time_tbd": "bool",
}
SCHEDULE_KEY = ("game_id",)

PRO_SCHEDULE_URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season_id}"
USER_AGENT = "nba-fantasy-2026-research/0.1 (personal, non-commercial; contact via GitHub repo)"
CACHE_SOURCE = "espn"

# ESPN pro team id -> (NBA abbreviation, NBA team id). ESPN ids and abbreviations verified in
# docs/research/espn-api-findings.md section 2; NBA ids are the stable stats.nba.com team ids.
ESPN_TEAMS: dict[int, tuple[str, int]] = {
    1: ("ATL", 1610612737), 2: ("BOS", 1610612738), 3: ("NOP", 1610612740), 4: ("CHI", 1610612741),
    5: ("CLE", 1610612739), 6: ("DAL", 1610612742), 7: ("DEN", 1610612743), 8: ("DET", 1610612765),
    9: ("GSW", 1610612744), 10: ("HOU", 1610612745), 11: ("IND", 1610612754), 12: ("LAC", 1610612746),
    13: ("LAL", 1610612747), 14: ("MIA", 1610612748), 15: ("MIL", 1610612749), 16: ("MIN", 1610612750),
    17: ("BKN", 1610612751), 18: ("NYK", 1610612752), 19: ("ORL", 1610612753), 20: ("PHI", 1610612755),
    21: ("PHX", 1610612756), 22: ("POR", 1610612757), 23: ("SAC", 1610612758), 24: ("SAS", 1610612759),
    25: ("OKC", 1610612760), 26: ("UTA", 1610612762), 27: ("WAS", 1610612764), 28: ("TOR", 1610612761),
    29: ("MEM", 1610612763), 30: ("CHA", 1610612766),
}
ABBR_OF_TEAM_ID = {tid: abbr for abbr, tid in ESPN_TEAMS.values()}
TEAM_ID_OF_ESPN = {eid: tid for eid, (_a, tid) in ESPN_TEAMS.items()}

HEAVY_PER_7 = 3.75   # >= 4 games in 7 days (a 6-day week 1 with 4 games is heavy too)
LIGHT_PER_7 = 2.5    # <= 2 games in 7 days
OFF_NIGHT_QUANTILE = 0.35


class ScheduleError(RuntimeError):
    """The schedule data is unusable."""


# --------------------------------------------------------------------------- table IO

def table_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / "processed" / f"{TABLE}.parquet"


def validate_schedule(df: pd.DataFrame) -> pd.DataFrame:
    problems: list[str] = []
    missing = [c for c in SCHEDULE_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    else:
        for col, kind in SCHEDULE_COLUMNS.items():
            if not kind.endswith("?") and df[col].isna().any():
                problems.append(f"{col}: nulls in a non-nullable column")
        if df.duplicated(list(SCHEDULE_KEY)).any():
            problems.append(f"duplicate rows on key {SCHEDULE_KEY}")
        if (df["home_team_id"] == df["away_team_id"]).any():
            problems.append("a team plays itself")
        known = set(ABBR_OF_TEAM_ID)
        bad = (set(df["home_team_id"]) | set(df["away_team_id"])) - known
        if bad:
            problems.append(f"unknown NBA team ids {sorted(bad)[:5]}")
    if problems:
        raise ContractError(f"{TABLE}: " + "; ".join(problems))
    return df


def read_schedule(base: Path | None = None, season: str | None = None) -> pd.DataFrame:
    path = table_path(base)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.inseason.schedule --season S --ingest`")
    df = pd.read_parquet(path)
    validate_schedule(df)
    return df if season is None else df[df["season"] == season].reset_index(drop=True)


def write_schedule(df: pd.DataFrame, base: Path | None = None) -> Path:
    """Replace the rows of every season present in ``df``; keep other seasons. Atomic."""
    import os
    import tempfile

    validate_schedule(df)
    path = table_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df
    if path.exists():
        old = pd.read_parquet(path)
        out = pd.concat([old[~old["season"].isin(set(df["season"]))], df], ignore_index=True)
    out = out.sort_values(["season", "game_date", "game_id"], kind="mergesort").reset_index(drop=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        out.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


# --------------------------------------------------------------------------- builders

def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(SCHEDULE_COLUMNS))
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["scoring_period"] = df["scoring_period"].astype("Int64")
    for c in ("home_team_id", "away_team_id"):
        df[c] = df[c].astype("int64")
    df["time_tbd"] = df["time_tbd"].astype(bool)
    return df.sort_values(["game_date", "game_id"], kind="mergesort").reset_index(drop=True)


def parse_pro_schedule(payload: dict, season: str) -> pd.DataFrame:
    """ESPN ``proTeamSchedules_wl`` payload -> ``schedule_games`` rows (one per unique game id).

    A game's calendar date is its scoring period's day, not the UTC timestamp (an 8 pm ET tip is the
    next day in UTC). Day zero is inferred from the data: the (timestamp - 6 h) date of every game minus
    its scoring period must agree, and the most common value wins.
    """
    teams = ((payload.get("settings") or {}).get("proTeams")) or []
    games: dict[int, dict] = {}
    for t in teams:
        for _sp, glist in (t.get("proGamesByScoringPeriod") or {}).items():
            for g in glist:
                games.setdefault(int(g["id"]), g)
    if not games:
        raise ScheduleError("the ESPN payload holds no games (settings.proTeams[*].proGamesByScoringPeriod)")
    offsets = []
    for g in games.values():
        utc = pd.Timestamp(int(g["date"]), unit="ms")
        offsets.append((utc - pd.Timedelta(hours=6)).normalize() - pd.Timedelta(days=int(g["scoringPeriodId"]) - 1))
    day_one = pd.Series(offsets).mode().iloc[0]
    rows, unknown = [], set()
    for gid, g in games.items():
        h, a = int(g["homeProTeamId"]), int(g["awayProTeamId"])
        if h not in ESPN_TEAMS or a not in ESPN_TEAMS:
            unknown |= {x for x in (h, a) if x not in ESPN_TEAMS}
            continue
        sp = int(g["scoringPeriodId"])
        rows.append({
            "season": season, "game_id": str(gid), "scoring_period": sp,
            "game_date": day_one + pd.Timedelta(days=sp - 1),
            "home_team_id": ESPN_TEAMS[h][1], "away_team_id": ESPN_TEAMS[a][1],
            "home_abbr": ESPN_TEAMS[h][0], "away_abbr": ESPN_TEAMS[a][0],
            "source": SOURCE_ESPN, "time_tbd": bool(g.get("startTimeTBD", False)),
        })
    if unknown:
        raise ScheduleError(f"unknown ESPN pro team ids in the schedule: {sorted(unknown)}")
    return validate_schedule(_frame(rows))


def from_team_games(team_games: pd.DataFrame, season: str) -> pd.DataFrame:
    """The realised schedule of a finished season from the ``team_games`` contract table."""
    tg = team_games[team_games["season"] == season]
    if tg.empty:
        raise ScheduleError(f"team_games has no rows for {season}")
    rows = []
    for gid, g in tg.groupby("game_id", sort=False):
        if len(g) != 2:
            continue                       # a game with one team row is unusable as a fixture
        flagged = g[g["is_home"]]
        # Real data has a few games (5 in 2025-26) where neither row is flagged home, e.g. neutral-site games.
        # Home/away does not matter to any tool here, so the lower team id is called home, deterministically.
        h = flagged.iloc[0] if len(flagged) == 1 else g.sort_values("team_id").iloc[0]
        a = g[g["team_id"] != h["team_id"]].iloc[0]
        rows.append({
            "season": season, "game_id": str(gid), "scoring_period": None, "game_date": h["game_date"],
            "home_team_id": int(h["team_id"]), "away_team_id": int(a["team_id"]),
            "home_abbr": str(h["team_abbr"]), "away_abbr": str(a["team_abbr"]),
            "source": SOURCE_TEAM_GAMES, "time_tbd": False})
    df = _frame(rows)
    # synthetic and real team ids need not be the 30 real ones; validate only what the table promises
    return df


def fetch_pro_schedule(client, season: str, *, refresh: bool = False) -> dict:
    """One cached GET of ESPN's pro-team schedule (``client`` is a ``CachedHttpClient``)."""
    sid = season_start(season) + 1
    return client.get_json(f"proschedule/{sid}", PRO_SCHEDULE_URL.format(season_id=sid),
                           {"view": "proTeamSchedules_wl"}, headers={"User-Agent": USER_AGENT}, refresh=refresh)


# --------------------------------------------------------------------------- analytics

def team_game_days(sched: pd.DataFrame) -> pd.DataFrame:
    """Long form, one row per (team, game): ``team_id, opp_team_id, is_home, game_date, game_id``."""
    home = pd.DataFrame({"game_id": sched["game_id"], "game_date": pd.to_datetime(sched["game_date"]),
                         "team_id": sched["home_team_id"], "opp_team_id": sched["away_team_id"], "is_home": True})
    away = pd.DataFrame({"game_id": sched["game_id"], "game_date": pd.to_datetime(sched["game_date"]),
                         "team_id": sched["away_team_id"], "opp_team_id": sched["home_team_id"], "is_home": False})
    return pd.concat([home, away], ignore_index=True).sort_values(
        ["team_id", "game_date", "game_id"], kind="mergesort").reset_index(drop=True)


def games_per_day(sched: pd.DataFrame) -> pd.Series:
    return sched.groupby(pd.to_datetime(sched["game_date"]).dt.normalize())["game_id"].nunique().sort_index()


def off_night_max(day_sizes: pd.Series, quantile: float = OFF_NIGHT_QUANTILE) -> int:
    """Largest slate that still counts as an off night: the ``quantile`` of game-day sizes (data-derived)."""
    return int(np.floor(day_sizes.quantile(quantile))) if len(day_sizes) else 0


def flag_back_to_backs(tgd: pd.DataFrame) -> pd.DataFrame:
    """Adds ``b2b`` (second night of consecutive days) to a :func:`team_game_days` frame."""
    out = tgd.sort_values(["team_id", "game_date"], kind="mergesort").copy()
    gap = out.groupby("team_id")["game_date"].diff().dt.days
    out["b2b"] = gap == 1
    return out


def weekly_team_table(sched: pd.DataFrame, cal: MatchupCalendar, as_of=None, *, off_max: int | None = None) -> pd.DataFrame:
    """One row per (week, team): games, games left after ``as_of``, b2b, off-night games and flags.

    Columns: ``week, kind, start, end, n_days, team_id, abbr, games, games_left, b2b, off_night_games,
    per7, heavy, light, week_mean`` (``week_mean`` = the league-wide mean games per team that week, so a
    week the schedule is still missing games for is visible).
    """
    tgd = flag_back_to_backs(team_game_days(sched))
    sizes = games_per_day(sched)
    limit = off_night_max(sizes) if off_max is None else off_max
    small = set(sizes[sizes <= limit].index)
    tgd["off_night"] = tgd["game_date"].dt.normalize().isin(small)
    cutoff = None if as_of is None else pd.Timestamp(as_of_date(as_of))
    team_ids = sorted(set(tgd["team_id"]))
    rows = []
    for w in cal.weeks:
        a, b = pd.Timestamp(w.start), pd.Timestamp(w.end)
        inw = tgd[(tgd["game_date"] >= a) & (tgd["game_date"] <= b)]
        g = inw.groupby("team_id")
        games = g.size().reindex(team_ids, fill_value=0)
        b2b = g["b2b"].sum().reindex(team_ids, fill_value=0)
        off = g["off_night"].sum().reindex(team_ids, fill_value=0)
        if cutoff is None:
            left = games
        else:
            left = inw[inw["game_date"] > cutoff].groupby("team_id").size().reindex(team_ids, fill_value=0)
        per7 = games * 7.0 / w.n_days
        rows.append(pd.DataFrame({
            "week": w.week, "kind": w.kind, "start": w.start, "end": w.end, "n_days": w.n_days,
            "team_id": team_ids, "abbr": [ABBR_OF_TEAM_ID.get(t, str(t)) for t in team_ids],
            "games": games.to_numpy(), "games_left": left.to_numpy(), "b2b": b2b.to_numpy(),
            "off_night_games": off.to_numpy(), "per7": per7.to_numpy(),
            "heavy": (per7 >= HEAVY_PER_7).to_numpy(), "light": (per7 <= LIGHT_PER_7).to_numpy(),
            "week_mean": float(games.mean()),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def as_of_date(x) -> date:
    return as_date(x)


def games_remaining(sched: pd.DataFrame, as_of) -> pd.Series:
    """Games each team still has to play strictly after ``as_of`` (team_id -> count)."""
    tgd = team_game_days(sched)
    left = tgd[tgd["game_date"] > pd.Timestamp(as_of_date(as_of))]
    return left.groupby("team_id").size()


def playoff_weeks_table(weekly: pd.DataFrame) -> pd.DataFrame:
    """Just the playoff weeks, one row per team with games per playoff week as columns."""
    if weekly.empty:
        return weekly
    p = weekly[weekly["kind"] == "playoff"]
    if p.empty:
        return p
    wide = p.pivot(index=["team_id", "abbr"], columns="week", values="games")
    wide.columns = [f"wk{c}" for c in wide.columns]
    wide["total"] = wide.sum(axis=1)
    return wide.reset_index().sort_values(["total", "abbr"], ascending=[False, True]).reset_index(drop=True)


def expected_week_games(team_ids, weekly: pd.DataFrame, week: int, *, remaining_only: bool = True,
                        availability=None) -> np.ndarray:
    """Expected games in matchup ``week`` per player: the team's games (left, by default) x availability.

    ``availability`` is a per-player probability of playing a given team game (scalar or array aligned to
    ``team_ids``); ``None`` means 1.
    """
    w = weekly[weekly["week"] == week].set_index("team_id")
    col = "games_left" if remaining_only else "games"
    games = pd.Series(team_ids).map(w[col]).fillna(0).to_numpy(dtype=float)
    if availability is None:
        return games
    return games * np.asarray(availability, dtype=float)


# --------------------------------------------------------------------------- league calendar loading

def league_settings_payload(league_id: int | None, season: str, *, data_root: Path | None = None) -> dict | None:
    """The cached raw league settings payload (ADR 0008) or ``None``. Never touches the network."""
    if league_id is None:
        return None
    from src.ingest.espn_league import ESPNLeagueClient, ESPNLeagueError

    client = ESPNLeagueClient(cache_dir=(data_root / "raw" / "espn_league") if data_root else raw_dir("espn_league"),
                              offline=True)
    sid = season_start(season) + 1
    try:
        return client.peek(league_id, sid, ["mSettings", "mTeam", "mRoster", "mStandings"])
    except ESPNLeagueError:
        return None


def build_calendar(sched: pd.DataFrame, season: str, *, league_id: int | None = None, cfg: dict | None = None,
                   data_root: Path | None = None) -> MatchupCalendar:
    """The league's matchup calendar for ``sched``'s season (ESPN periods when real, else derived)."""
    cfg = cfg or load_league()
    day_one = pd.to_datetime(sched["game_date"]).min()
    payload = league_settings_payload(league_id, season, data_root=data_root)
    return calendar_from_settings(payload, day_one, cfg=cfg, games_per_day=games_per_day(sched))


# --------------------------------------------------------------------------- CLI

def _safe_print(text: str, *, file=None) -> None:
    stream = file or sys.stdout
    try:
        print(text, file=stream)
    except UnicodeEncodeError:
        enc = getattr(stream, "encoding", None) or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc), file=stream)


def _int_env(name: str) -> int | None:
    import os

    v = os.environ.get(name, "").strip()
    return int(v) if v else None


def main(argv: list[str] | None = None) -> int:
    from src.inseason._console import use_utf8_console

    use_utf8_console()
    ap = argparse.ArgumentParser(prog="python -m src.inseason.schedule", description=__doc__.split("\n\n")[0])
    ap.add_argument("--season", default=None, help="e.g. 2026-27 (default: config/league.yaml)")
    ap.add_argument("--ingest", action="store_true", help="fetch the ESPN pro-team schedule (one cached GET) and store it")
    ap.add_argument("--refresh", action="store_true", help="with --ingest: re-fetch even if cached")
    ap.add_argument("--from-team-games", action="store_true",
                    help="build the schedule of a finished season from the team_games table (no network)")
    ap.add_argument("--offline", action="store_true", help="never touch the network; cache only")
    ap.add_argument("--as-of", default=None, help="date for 'games left' (default: today)")
    ap.add_argument("--week", type=int, default=None, help="show only this matchup week")
    ap.add_argument("--league-id", type=int, default=None, help="ESPN league id for the real matchup periods ($ESPN_LEAGUE_ID)")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args(argv)

    cfg = load_league()
    season = args.season or cfg["league"]["season"]
    base = args.data_dir
    league_id = args.league_id if args.league_id is not None else _int_env("ESPN_LEAGUE_ID")
    try:
        if args.ingest:
            from src.ingest.http_cache import CachedHttpClient, HttpCacheError

            client = CachedHttpClient((base / "raw" / CACHE_SOURCE) if base else raw_dir(CACHE_SOURCE),
                                      offline=args.offline or None)
            try:
                payload = fetch_pro_schedule(client, season, refresh=args.refresh)
            except HttpCacheError as exc:
                _safe_print(f"error: {exc}", file=sys.stderr)
                return 5
            sched = parse_pro_schedule(payload, season)
            path = write_schedule(sched, base)
            per_team = pd.concat([sched["home_team_id"], sched["away_team_id"]]).value_counts()
            _safe_print(f"wrote {len(sched)} games for {season} -> {path} (network requests this run: "
                        f"{client.stats.network_requests}; games per team {per_team.min()}-{per_team.max()})")
        elif args.from_team_games:
            from src.store import read_table

            sched = from_team_games(read_table("team_games", base), season)
            path = write_schedule(sched, base)
            _safe_print(f"wrote {len(sched)} games for {season} from team_games -> {path}")
        else:
            sched = read_schedule(base, season)
    except (FileNotFoundError, ScheduleError, ContractError) as exc:
        _safe_print(f"error: {exc}", file=sys.stderr)
        return 2
    if sched.empty:
        _safe_print(f"error: no schedule rows for {season}", file=sys.stderr)
        return 2

    as_of = as_of_date(args.as_of) if args.as_of else date.today()
    cal = build_calendar(sched, season, league_id=league_id, cfg=cfg, data_root=base)
    weekly = weekly_team_table(sched, cal, as_of)
    cur = cal.current_week(as_of)
    _safe_print(f"\n{season}: {len(sched)} games, {sched['game_date'].min().date()} .. {sched['game_date'].max().date()}; "
                f"matchup calendar source: {cal.source} ({len(cal.weeks)} periods, "
                f"{cal.first_day} .. {cal.last_day}); as of {as_of}, current week: {cur.week if cur else 'none (season over)'}")
    if cal.source == "derived":
        _safe_print("note: not read from ESPN (unpublished, or league not synced); using the derived "
                    "Monday-Sunday calendar with the All-Star week merged (see src/inseason/weeks.py).")
    summary = weekly.groupby(["week", "kind", "start", "end", "n_days"]).agg(
        mean_games=("games", "mean"), min_games=("games", "min"), max_games=("games", "max"),
        heavy_teams=("heavy", "sum"), light_teams=("light", "sum"), b2b_total=("b2b", "sum")).reset_index()
    _safe_print("\nLeague-wide by matchup week:")
    _safe_print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if args.week is not None or cur is not None:
        wk = args.week if args.week is not None else cur.week
        one = weekly[weekly["week"] == wk].sort_values(["games", "off_night_games"], ascending=False)
        _safe_print(f"\nWeek {wk}, by team (games_left counts games after {as_of}):")
        _safe_print(one[["abbr", "games", "games_left", "b2b", "off_night_games", "heavy", "light"]]
                    .head(args.top).to_string(index=False))
    playoff = playoff_weeks_table(weekly)
    if len(playoff):
        _safe_print("\nPlayoff weeks (games per team):")
        _safe_print(playoff.head(args.top).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
