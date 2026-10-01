"""NBA official injury-report PDFs -> ``injury_reports`` (ADR 0033; source verdict in docs/research/data-sources.md 4.3).

    python -m src.ingest.nba_injury_reports [--seasons 2018-19:2025-26] [--since 2026-10-20] [--max-requests N] [--offline]

The league publishes one PDF per report time at ``ak-static.cms.nba.com/referee/injury/Injury-Report_<date>_<slot>.pdf``: for
every game that day, each player listed with a game status (Out / Doubtful / Questionable / Probable / Available) and a
reason (``Injury/Illness - Right Knee; Sprain``, ``G League - Two-Way``, ``Personal Reasons``, ``Not With Team`` ...). That is
the labelled cause the games-missed proxy lacks: it tells injury from rest from G League assignment from suspension. Reports
exist from 2018-12-19 (earlier dates answer 403); it gives no duration or return date, which the feature layer derives from
consecutive listings.

What this does (and only this):

* for each NBA game date in the requested seasons it fetches **one** report, the first slot that exists of ``05PM`` (the
  5:30 PM ET report, present on every sampled date), then ``06PM``, ``01PM``, ``03PM``, ``08AM``; absent dates are remembered
  so a re-run does not probe them again. One request per 1.5 s at most, a descriptive User-Agent without an e-mail address
  (the host resets the connection otherwise), robots.txt allows ``/referee/injury/`` with ``Crawl-Delay: 1``;
* raw PDFs stay under ``<data dir>/raw/nba_injury`` (private, never committed: NBA.com's terms forbid redistribution; same
  accepted personal, non-commercial, local-only stance as ADR 0005);
* ``pdftotext -table`` (poppler/xpdf, must be on PATH; nothing is pip-installed) turns each PDF into one row per player;
  ``parse_report_text`` recognises fields by shape rather than position because column offsets drift between pages and
  layouts (2018-21: Category | Reason | Current Status | Previous Status; 2021+: Current Status | Reason);
* ``injury_reports.parquet`` is rebuilt from every cached PDF, so an interrupted run resumes and a parser fix applies to
  everything already downloaded. ``--offline`` parses only what is cached and never touches the network.

Each row carries ``season`` (from ``team_games`` when available) so the table can ride in ``History.extras`` and be sliced
to seasons strictly before the target like every other dated input.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.contracts import data_dir, season_start, season_str
from src.ingest.id_map import alias_key

TABLE = "injury_reports"
REPORT_NAME = "nba_injury_report.json"
URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{slot}.pdf"
USER_AGENT = "Mozilla/5.0 (compatible; nba-fantasy-2026/0.1; personal non-commercial research)"
FIRST_REPORT_DATE = date(2018, 12, 19)
SLOTS = ("05PM", "06PM", "01PM", "03PM", "08AM")
MIN_INTERVAL = 1.5
RETRY_DAYS = 5          # a date this recent with no report yet is probed again on the next run (published late, or a blip)
STATUSES = ("Out", "Doubtful", "Questionable", "Probable", "Available")
COLUMNS = ["season", "game_date", "report_ts", "slot", "matchup", "team_abbr", "player_name", "player_id", "status",
           "category", "detail"]

TEAM_NICKNAMES = {
    "ATL": "Hawks", "BOS": "Celtics", "BKN": "Nets", "CHA": "Hornets", "CHI": "Bulls", "CLE": "Cavaliers",
    "DAL": "Mavericks", "DEN": "Nuggets", "DET": "Pistons", "GSW": "Warriors", "HOU": "Rockets", "IND": "Pacers",
    "LAC": "Clippers", "LAL": "Lakers", "MEM": "Grizzlies", "MIA": "Heat", "MIL": "Bucks", "MIN": "Timberwolves",
    "NOP": "Pelicans", "NYK": "Knicks", "OKC": "Thunder", "ORL": "Magic", "PHI": "76ers", "PHX": "Suns",
    "POR": "Trail Blazers", "SAC": "Kings", "SAS": "Spurs", "TOR": "Raptors", "UTA": "Jazz", "WAS": "Wizards",
}


TEAM_CITIES = {
    "ATL": "Atlanta", "BOS": "Boston", "BKN": "Brooklyn", "CHA": "Charlotte", "CHI": "Chicago", "CLE": "Cleveland",
    "DAL": "Dallas", "DEN": "Denver", "DET": "Detroit", "GSW": "Golden State", "HOU": "Houston", "IND": "Indiana",
    "MEM": "Memphis", "MIA": "Miami", "MIL": "Milwaukee", "MIN": "Minnesota", "NOP": "New Orleans", "NYK": "New York",
    "OKC": "Oklahoma City", "ORL": "Orlando", "PHI": "Philadelphia", "PHX": "Phoenix", "POR": "Portland",
    "SAC": "Sacramento", "SAS": "San Antonio", "TOR": "Toronto", "UTA": "Utah", "WAS": "Washington",
}


class InjuryReportError(RuntimeError):
    pass


def raw_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / "raw" / "nba_injury"


def table_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / "processed" / f"{TABLE}.parquet"


def report_url(day: date, slot: str) -> str:
    return URL.format(date=day.isoformat(), slot=slot)


# --------------------------------------------------------------------------- parsing

_STAMP = re.compile(r"Injury\s+Report:\s+(\d\d)/(\d\d)/(\d\d)\s+(\d\d:\d\d)\s*([AP]M)")
_DATE = re.compile(r"^\d\d/\d\d/\d{4}$")
_TIME = re.compile(r"^\d\d:\d\d \(ET\)$")
_MATCHUP = re.compile(r"^[A-Z]{2,3}@[A-Z]{2,3}$")
_STATUS = set(STATUSES) | {"NOT YET SUBMITTED"}


_WRAPPED_TEAM_TAILS = set(TEAM_NICKNAMES.values())


def split_reason(reason: str) -> tuple[str, str]:
    """``'Injury/Illness - Right Knee; Sprain'`` -> ``('Injury/Illness', 'Right Knee; Sprain')``; ``'-'`` -> ``('', '')``."""
    r = reason.strip()
    if not r or r == "-":
        return "", ""
    head, sep, tail = r.partition(" - ")
    return (head.strip(), tail.strip()) if sep else (head.strip(), "")


def parse_report_text(text: str) -> list[dict]:
    """One dict per player-status row from ``pdftotext -table`` output.

    Content-based: after splitting on runs of 2+ spaces a token is a date (mm/dd/yyyy), a time (``07:00 (ET)``), a matchup
    (``AAA@BBB``), a team (no comma, before a player), a player (``Last, First``) or a status (closed vocabulary). In the
    2018-21 layout (header has ``Category``) the tokens between player and status are ``[category, reason]`` and the token
    after the status is the previous status; afterwards the token after the status is ``reason`` (``Category - detail``).
    A line with no status is the wrapped tail of the previous reason. ``NOT YET SUBMITTED`` rows are team-level and skipped.
    """
    legacy = "Category" in text
    rows: list[dict] = []
    report_ts = game_date = game_time = matchup = team = None
    for raw in text.splitlines():
        m = _STAMP.search(raw)
        if m:
            report_ts = f"20{m.group(3)}-{m.group(1)}-{m.group(2)} {m.group(4)} {m.group(5)}"
            continue
        line = raw.strip()
        if not line or line.startswith("Page ") or line.startswith("Game Date") or "Injury Report" in line:
            continue
        toks = re.split(r"\s{2,}", line)
        i_status = next((i for i, t in enumerate(toks) if t in _STATUS), None)
        j = 0
        while j < len(toks) and (i_status is None or j < i_status):
            t = toks[j]
            if _DATE.match(t):
                game_date = t
            elif _TIME.match(t):
                game_time = t
            elif _MATCHUP.match(t):
                matchup = t
            elif i_status is not None and "," not in t and j + 1 <= i_status - 1:
                team = t                                   # a team name: no comma, followed by a player before the status
            else:
                break
            j += 1
        rest = toks[j:]
        if i_status is None:
            if rest and " ".join(rest) in _WRAPPED_TEAM_TAILS:
                continue                                   # the second line of a wrapped team name, not a reason
            if rows and rest:                              # wrapped continuation of the previous row's reason
                rows[-1]["detail"] = (rows[-1]["detail"] + " " + " ".join(rest)).strip()
            continue
        k = i_status - j
        before, status, after = rest[:k], rest[k], rest[k + 1:]
        if status == "NOT YET SUBMITTED":
            team = before[-1] if before else team
            continue
        player = before[0] if before else ""
        if not player:
            continue
        if legacy:
            mid = before[1:]
            category = mid[0] if mid else ""
            detail = " ".join(mid[1:]).strip()
            category, detail = ("", "") if category == "-" else (category, "" if detail == "-" else detail)
        else:
            category, detail = split_reason(" ".join(after))
        rows.append({"report_ts": report_ts, "game_date": game_date, "game_time": game_time, "matchup": matchup,
                     "team": team, "player": player, "status": status, "category": category, "detail": detail})
    return rows


def display_name(last_first: str) -> str:
    """``'Moore Jr., Wendell'`` -> ``'Wendell Moore Jr.'``; a name without a comma is returned unchanged."""
    last, sep, first = last_first.partition(",")
    return f"{first.strip()} {last.strip()}".strip() if sep else last_first.strip()


def team_abbr_of(team_name: str | None) -> str | None:
    """Abbreviation of a report's team column. A long name wraps onto two lines (``Minnesota`` / ``Timberwolves``), so the
    first line alone (a city) must resolve too."""
    if not team_name:
        return None
    name = team_name.strip()
    for abbr, nick in TEAM_NICKNAMES.items():
        if name.endswith(nick):
            return abbr
    for abbr, city in TEAM_CITIES.items():
        if name == city or name.startswith(city + " "):
            return abbr
    return None


def pdf_to_text(pdf: Path) -> str:
    try:
        out = subprocess.run(["pdftotext", "-table", str(pdf), "-"], capture_output=True, text=True, check=True, timeout=60)
    except FileNotFoundError as exc:
        raise InjuryReportError("pdftotext (poppler/xpdf) is not on PATH; it is needed to read the injury-report PDFs") from exc
    except subprocess.CalledProcessError as exc:
        raise InjuryReportError(f"pdftotext failed on {pdf.name}: {exc.stderr.strip()[:200]}") from exc
    return out.stdout


# --------------------------------------------------------------------------- id matching and the table

@dataclass
class PlayerResolver:
    """Report name + team + season -> ``player_id``: unique names resolve directly, ties go to who played for that team."""

    by_name: dict
    teams: dict                      # (player_id, season_start) -> set(team_abbr)

    @classmethod
    def build(cls, players: pd.DataFrame, game_logs: pd.DataFrame | None) -> "PlayerResolver":
        by_name: dict[str, set] = {}
        for pid, name in zip(players["player_id"].to_numpy(), players["player_name"].to_numpy()):
            by_name.setdefault(alias_key(name), set()).add(int(pid))
        teams: dict = {}
        if game_logs is not None and len(game_logs):
            g = game_logs[["player_id", "season", "team_abbr"]].drop_duplicates()
            for pid, s, ab in zip(g["player_id"].to_numpy(), g["season"].map(season_start).to_numpy(), g["team_abbr"].to_numpy()):
                teams.setdefault((int(pid), int(s)), set()).add(ab)
        return cls(by_name, teams)

    def resolve(self, name: str, team: str | None, season: str | None) -> int | None:
        cands = self.by_name.get(alias_key(name), set())
        if len(cands) == 1:
            return next(iter(cands))
        if not cands or season is None or team is None:
            return None
        s = season_start(season)
        on_team = [c for c in cands if team in self.teams.get((c, s), set()) | self.teams.get((c, s - 1), set())]
        return on_team[0] if len(on_team) == 1 else None


def season_of(day: date, date_to_season: dict | None) -> str:
    if date_to_season and day in date_to_season:
        return date_to_season[day]
    y = day.year if day.month >= 8 else day.year - 1
    if date(2020, 7, 30) <= day <= date(2020, 10, 12):       # the 2019-20 restart in the bubble
        y = 2019
    return season_str(y)


def rows_to_frame(rows: list[dict], slot: str, resolver: PlayerResolver | None, date_to_season: dict | None) -> pd.DataFrame:
    out = []
    for r in rows:
        if not r["game_date"] or not r["report_ts"]:
            continue
        gd = datetime.strptime(r["game_date"], "%m/%d/%Y").date()
        season = season_of(gd, date_to_season)
        abbr = team_abbr_of(r["team"])
        name = display_name(r["player"])
        pid = resolver.resolve(name, abbr, season) if resolver else None
        out.append({"season": season, "game_date": pd.Timestamp(gd), "report_ts": pd.to_datetime(r["report_ts"], format="%Y-%m-%d %I:%M %p"),
                    "slot": slot, "matchup": r["matchup"], "team_abbr": abbr, "player_name": name, "player_id": pid,
                    "status": r["status"], "category": r["category"], "detail": r["detail"]})
    df = pd.DataFrame(out, columns=COLUMNS)
    df["player_id"] = pd.array(df["player_id"], dtype="Int64")
    df["game_date"] = df["game_date"].astype("datetime64[ms]")
    df["report_ts"] = df["report_ts"].astype("datetime64[ms]")
    return df


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_injury_reports(base: Path | None = None) -> pd.DataFrame:
    path = table_path(base)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; has `python -m src.ingest.nba_injury_reports` run?")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- fetching

class Fetcher:
    """Polite one-file-at-a-time downloader with a disk cache and a memory of absent keys (S3 answers 403)."""

    def __init__(self, base: Path | None = None, *, interval: float = MIN_INTERVAL, offline: bool = False,
                 opener=None, sleep=time.sleep, clock=time.monotonic, max_requests: int | None = None):
        self.dir = raw_path(base)
        self.interval, self.offline = interval, offline
        self._open = opener or self._default_open
        self._sleep, self._clock = sleep, clock
        self.max_requests = max_requests
        self.requests = 0
        self._last = None
        self.index = self._load_index()

    # -- index: which (date, slot) exist, which are known absent
    @property
    def index_file(self) -> Path:
        return self.dir / "index.json"

    def _load_index(self) -> dict:
        try:
            idx = json.loads(self.index_file.read_text(encoding="utf-8"))
            return {"fetched": dict(idx.get("fetched", {})), "absent": {k: list(v) for k, v in idx.get("absent", {}).items()}}
        except (FileNotFoundError, ValueError):
            return {"fetched": {}, "absent": {}}

    def save_index(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_file.write_text(json.dumps(self.index, indent=1, sort_keys=True), encoding="utf-8")

    def pdf_file(self, day: date, slot: str) -> Path:
        return self.dir / f"{day.isoformat()}_{slot}.pdf"

    @staticmethod
    def _default_open(url: str, timeout: float = 30.0) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    def budget_left(self) -> bool:
        return self.max_requests is None or self.requests < self.max_requests

    def _get(self, day: date, slot: str) -> bytes | None:
        if self._last is not None:
            wait = self.interval - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()
        self.requests += 1
        try:
            return self._open(report_url(day, slot))
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            raise InjuryReportError(f"HTTP {e.code} for {report_url(day, slot)}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise InjuryReportError(f"network error for {report_url(day, slot)}: {e}") from e

    def fetch_day(self, day: date) -> str | None:
        """Slot of the report cached or downloaded for ``day`` (``None`` if none exists or the budget ran out first)."""
        key = day.isoformat()
        slot = self.index["fetched"].get(key)
        if slot and self.pdf_file(day, slot).exists():
            return slot
        if self.offline:
            return None
        tried = set() if self._recent(day) else set(self.index["absent"].get(key, []))
        for slot in SLOTS:
            if slot in tried:
                continue
            if not self.budget_left():
                return None
            data = self._get(day, slot)
            if data is None:
                tried.add(slot)
                self.index["absent"][key] = sorted(tried)
                continue
            self.dir.mkdir(parents=True, exist_ok=True)
            self.pdf_file(day, slot).write_bytes(data)
            self.index["fetched"][key] = slot
            self.index["absent"].pop(key, None)
            return slot
        return None

    @staticmethod
    def _recent(day: date) -> bool:
        return day >= date.today() - timedelta(days=RETRY_DAYS)

    def done(self, day: date) -> bool:
        """Fetched, or every slot known absent for a date old enough that a late publication is no longer expected."""
        key = day.isoformat()
        return key in self.index["fetched"] or (set(SLOTS) <= set(self.index["absent"].get(key, [])) and not self._recent(day))


# --------------------------------------------------------------------------- ingest


@dataclass
class IngestResult:
    n_dates: int = 0
    n_fetched: int = 0
    n_absent: int = 0
    n_pending: int = 0
    requests: int = 0
    n_rows: int = 0
    n_players_matched: int = 0
    unmatched_names: list = field(default_factory=list)
    parse_failures: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("n_dates", "n_fetched", "n_absent", "n_pending", "requests", "n_rows",
                                               "n_players_matched")} | {"unmatched_names": self.unmatched_names[:50],
                                                                        "parse_failures": self.parse_failures[:20]}


def game_dates(team_games: pd.DataFrame, seasons: list[str] | None, since: date | None = None,
               until: date | None = None) -> tuple[list[date], dict]:
    """Sorted game dates on or after the first report date (and ``since``) and the date -> season map."""
    tg = team_games[["season", "game_date"]].drop_duplicates()
    if seasons:
        tg = tg[tg["season"].isin(seasons)]
    d2s = {pd.Timestamp(d).date(): s for d, s in zip(tg["game_date"], tg["season"])}
    lo = max(FIRST_REPORT_DATE, since) if since else FIRST_REPORT_DATE
    hi = until or date.today()
    return sorted(d for d in d2s if lo <= d <= hi), d2s


def rebuild_table(fetcher: Fetcher, resolver: PlayerResolver | None, date_to_season: dict | None,
                  result: IngestResult | None = None, only: set[str] | None = None) -> pd.DataFrame:
    """Parse cached PDFs into one frame (``only``: just those ISO dates; default every cached PDF).

    Parsing every PDF is a pure function of the cache, so the full rebuild is the reference; the incremental path in
    ``ingest`` parses only the days fetched this run and merges them into the stored table.
    """
    frames = []
    for key, slot in sorted(fetcher.index["fetched"].items()):
        if only is not None and key not in only:
            continue
        day = date.fromisoformat(key)
        pdf = fetcher.pdf_file(day, slot)
        if not pdf.exists():
            continue
        try:
            rows = parse_report_text(pdf_to_text(pdf))
        except InjuryReportError as exc:
            if result is not None:
                result.parse_failures.append(f"{key} {slot}: {exc}")
            continue
        if not rows and result is not None:
            result.parse_failures.append(f"{key} {slot}: no rows parsed")
        frames.append(rows_to_frame(rows, slot, resolver, date_to_season))
    if not frames:
        return pd.DataFrame(columns=COLUMNS).astype({"player_id": "Int64"})
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["game_date", "matchup", "team_abbr", "player_name"], kind="mergesort").reset_index(drop=True)


def ingest(base: Path | None = None, *, seasons: list[str] | None = None, since: date | None = None,
           until: date | None = None, max_requests: int | None = None, interval: float = MIN_INTERVAL,
           offline: bool = False, rebuild: bool = False, opener=None, sleep=time.sleep, clock=time.monotonic,
           progress=None) -> IngestResult:
    from src.store import read_table

    base = base or data_dir()
    team_games = read_table("team_games", base)
    days, d2s = game_dates(team_games, seasons, since, until)
    players = read_table("players", base)
    try:
        logs = read_table("game_logs", base)
    except FileNotFoundError:
        logs = None
    resolver = PlayerResolver.build(players, logs)
    fetcher = Fetcher(base, interval=interval, offline=offline, opener=opener, sleep=sleep, clock=clock,
                      max_requests=max_requests)
    before = set(fetcher.index["fetched"])
    res = IngestResult(n_dates=len(days))
    try:
        for i, day in enumerate(days):
            if fetcher.done(day) or (offline and day.isoformat() not in fetcher.index["fetched"]):
                continue
            if not fetcher.budget_left():
                break
            fetcher.fetch_day(day)
            if progress and (i % 50 == 0):
                progress(f"{day}: {fetcher.requests} requests, {len(fetcher.index['fetched'])} reports cached")
            if fetcher.requests and fetcher.requests % 100 == 0:
                fetcher.save_index()
    finally:
        fetcher.save_index()
    res.requests = fetcher.requests
    res.n_fetched = sum(1 for d in days if d.isoformat() in fetcher.index["fetched"])
    res.n_absent = sum(1 for d in days if d.isoformat() not in fetcher.index["fetched"] and fetcher.done(d))
    res.n_pending = res.n_dates - res.n_fetched - res.n_absent
    new_days = set(fetcher.index["fetched"]) - before
    path = table_path(base)
    if path.exists() and not rebuild and not offline:
        # incremental: parse only what this run downloaded and merge it into the stored table (a nightly run is seconds, not minutes)
        old = pd.read_parquet(path)
        if new_days:
            fresh = rebuild_table(fetcher, resolver, d2s, res, only=new_days)
            drop = old["game_date"].dt.date.astype(str).isin(new_days) if len(old) else []
            table = pd.concat([old[~drop] if len(old) else old, fresh], ignore_index=True).sort_values(
                ["game_date", "matchup", "team_abbr", "player_name"], kind="mergesort").reset_index(drop=True)
        else:
            table = old
    else:
        table = rebuild_table(fetcher, resolver, d2s, res)
    atomic_write_parquet(table, path)
    res.n_rows = len(table)
    res.n_players_matched = int(table["player_id"].notna().sum())
    un = table[table["player_id"].isna()][["player_name", "team_abbr"]].drop_duplicates()
    res.unmatched_names = [f"{n} ({t})" for n, t in zip(un["player_name"], un["team_abbr"])]
    (base / "processed").mkdir(parents=True, exist_ok=True)
    (base / "processed" / REPORT_NAME).write_text(json.dumps(res.as_dict(), indent=2), encoding="utf-8")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.ingest.nba_injury_reports", description=__doc__.split("\n\n")[0])
    ap.add_argument("--seasons", default=None, help="e.g. 2018-19:2025-26 (default: every season in team_games)")
    ap.add_argument("--since", default=None, help="only game dates on or after YYYY-MM-DD (incremental in-season runs)")
    ap.add_argument("--until", default=None, help="only game dates on or before YYYY-MM-DD (default today)")
    ap.add_argument("--max-requests", type=int, default=None, help="stop after this many HTTP requests (resumable)")
    ap.add_argument("--interval", type=float, default=MIN_INTERVAL, help="seconds between requests (>= 1, robots Crawl-Delay)")
    ap.add_argument("--offline", action="store_true", help="parse the cached PDFs only; no network (always a full rebuild)")
    ap.add_argument("--rebuild", action="store_true", help="re-parse every cached PDF instead of merging only this run's new days")
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.interval < 1.0 and not args.offline:
        print("error: --interval must be at least 1 second (the host's robots.txt Crawl-Delay)", file=sys.stderr)
        return 2
    seasons = None
    if args.seasons:
        from src.backtest.harness import expand_seasons

        seasons = expand_seasons(args.seasons)
    try:
        res = ingest(args.data_dir, seasons=seasons, since=date.fromisoformat(args.since) if args.since else None,
                     until=date.fromisoformat(args.until) if args.until else None, max_requests=args.max_requests,
                     interval=args.interval, offline=args.offline, rebuild=args.rebuild,
                     progress=lambda s: print(s, flush=True))
    except (InjuryReportError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"injury reports: {res.n_fetched}/{res.n_dates} game dates fetched, {res.n_absent} absent, {res.n_pending} pending; "
          f"{res.requests} requests; {res.n_rows} rows, {res.n_players_matched} matched to a player_id; "
          f"{len(res.parse_failures)} parse problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
