"""A fake stats.nba.com for the nightly-job tests: a tiny league that plays one game per chosen date.

``World`` builds ``playergamelogs`` / ``leaguegamelog`` / ``playerindex`` / bio payloads (the real response shapes, from
``tests/ingest/ingest_fakes``) for the games played so far, honours ``DateFrom`` like the real endpoint, counts requests,
and can be told to have an outage, return a truncated window, or correct an old box score.
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ingest"))

from ingest_fakes import (  # noqa: E402
    BOS, NYK, bio_stats_payload, common_player_info_payload, player_index_payload, player_payload, player_row, team_payload, team_row,
)

from src.contracts import season_start  # noqa: E402
from src.ingest.nba_client import NBAClientError  # noqa: E402

SEASON = "2026-27"
PLAYERS = {11: ("Alpha One", BOS, "BOS"), 12: ("Bravo Two", BOS, "BOS"), 21: ("Charlie Three", NYK, "NYK"), 22: ("Delta Four", NYK, "NYK")}
ROOKIE = (23, "Echo Five", NYK, "NYK")


class World:
    def __init__(self, season: str = SEASON):
        self.season = season
        self.games: list[tuple[str, date]] = []          # (game_id, date) in play order
        self.version: dict[str, int] = {}                # game_id -> box-score version (bumped by ``correct``)
        self.rookie_from: date | None = None
        self.outage = False
        self.truncate_window = False
        self.unfinished: dict[str, str] = {}             # game_id -> "no_result" (WL blank) | "partial_box" (a player row missing) | "no_team_rows"
        self.vanished: set[str] = set()                  # game ids the source no longer returns at all
        self.calls: list[tuple[str, dict]] = []
        self.stats = SimpleNamespace(network_requests=0, cache_hits=0)
        self.offline = False

    # ------------------------------------------------------------- the league
    def play(self, *days: str) -> None:
        yy = season_start(self.season) % 100
        for d in days:
            gid = f"002{yy:02d}{len(self.games) + 1:05d}"
            self.games.append((gid, date.fromisoformat(d)))
            self.version[gid] = 0

    def correct(self, game_id: str) -> None:
        self.version[game_id] += 1

    def players_of(self, day: date) -> dict[int, tuple]:
        out = dict(PLAYERS)
        if self.rookie_from and day >= self.rookie_from:
            out[ROOKIE[0]] = (ROOKIE[1], ROOKIE[2], ROOKIE[3])
        return out

    def _rows(self, date_from: date | None, date_to: date | None = None):
        p_rows, t_rows = [], []
        for gid, day in self.games:
            if gid in self.vanished:
                continue
            if date_from and day < date_from:
                continue
            if date_to and day > date_to:
                continue
            v = self.version[gid]
            totals = {BOS: 0, NYK: 0}
            n = int(gid[-5:])
            plyrs = self.players_of(day)
            for pid, (name, tid, abbr) in plyrs.items():
                if self.unfinished.get(gid) == "partial_box" and pid in (12, 22):
                    continue                                     # the box score is still being built
                home = tid == BOS
                row = player_row(player_id=pid, name=name, team_id=tid, abbr=abbr, game_id=gid, date=f"{day.isoformat()}T00:00:00",
                                 matchup=f"{abbr} vs. {'NYK' if home else 'BOS'}" if home else f"{abbr} @ BOS", minutes=30.0 + pid % 7,
                                 fgm=3 + n % 4 + v, fga=9, fg3m=1, fg3a=3, ftm=2, fta=2, oreb=1, dreb=3, season=self.season)
                p_rows.append(row)
                totals[tid] += row[29]
            for tid, abbr in ((BOS, "BOS"), (NYK, "NYK")):
                if self.unfinished.get(gid) == "no_team_rows":
                    continue                                     # player rows are out, the team endpoint has not caught up
                t_rows.append(team_row(team_id=tid, abbr=abbr, game_id=gid, date=day.isoformat(),
                                       matchup="BOS vs. NYK" if tid == BOS else "NYK @ BOS", pts=totals[tid],
                                       season_id=f"2{season_start(self.season)}"))
                if self.unfinished.get(gid) == "no_result":
                    t_rows[-1][7] = None                         # WL is empty until the game is final
                if self.unfinished.get(gid) == "partial_box":
                    t_rows[-1][26] += 10                         # the team total already includes points the players do not

        return p_rows, t_rows

    # ------------------------------------------------------------- the client interface
    def get(self, endpoint, params, *, refresh=False):
        self.calls.append((endpoint, dict(params)))
        if self.offline and refresh:
            raise NBAClientError("offline")
        if self.outage:
            raise NBAClientError("stats.nba.com is down (fake outage)")
        self.stats.network_requests += 1
        df = params.get("DateFrom") or ""
        dt = params.get("DateTo") or ""
        date_from = datetime.strptime(df, "%m/%d/%Y").date() if df else None      # the real API's format; anything else is a 400
        date_to = datetime.strptime(dt, "%m/%d/%Y").date() if dt else None
        if endpoint == "playergamelogs":
            p, _ = self._rows(date_from, date_to)
            if self.truncate_window:
                p = p[: len(p) // 4]
            return player_payload(p)
        if endpoint == "leaguegamelog":
            _, t = self._rows(date_from, date_to)
            if self.truncate_window:
                t = t[: len(t) // 4]
            return team_payload(t)
        if endpoint == "playerindex":
            rows = [{"id": pid, "first": n.split()[0], "last": n.split()[1], "pos": "G", "height": "6-4", "weight": "200",
                     "dy": 2015, "dr": 1, "dn": pid, "fy": 2015, "ty": 2026} for pid, (n, *_r) in {**PLAYERS, ROOKIE[0]: ROOKIE[1:]}.items()]
            return player_index_payload(rows)
        if endpoint == "leaguedashplayerbiostats":
            return bio_stats_payload([{"id": pid, "name": n, "age": 27} for pid, (n, *_r) in {**PLAYERS, ROOKIE[0]: ROOKIE[1:]}.items()])
        if endpoint == "commonplayerinfo":
            return common_player_info_payload(int(params["PlayerID"]), birthdate="1999-05-17T00:00:00")
        raise AssertionError(f"unexpected endpoint {endpoint}")

    def calls_to(self, endpoint: str) -> int:
        return sum(1 for e, _ in self.calls if e == endpoint)
