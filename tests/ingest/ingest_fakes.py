"""Test doubles and tiny fixture builders for the NBA ingest tests. No network, no real data.

The payload builders mirror the *real* stats.nba.com response shapes observed while building the
ingest (header lists trimmed to what the code reads plus a few realistic extras).
"""
from __future__ import annotations

import json
from typing import Any

# --------------------------------------------------------------------------- HTTP / clock fakes


class FakeResponse:
    def __init__(self, status: int = 200, body: bytes | str | dict | None = None, headers: dict | None = None):
        if isinstance(body, dict):
            body = json.dumps(body)
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.status_code = status
        self.content = body if body is not None else b""
        self.headers = headers or {}


class FakeSession:
    """Plays back a script of responses/exceptions, one per ``get`` call, then repeats the last."""

    def __init__(self, script: list[Any]):
        self.script = list(script)
        self.calls: list[dict] = []
        self._i = 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {}), "timeout": timeout})
        item = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClock:
    """Injectable clock+sleep: sleeping advances time, so rate limiting is testable instantly."""

    def __init__(self, start: float = 1000.0):
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def tick(self, seconds: float) -> None:
        self.now += seconds


# --------------------------------------------------------------------------- payload builders

def wrap(name: str, headers: list[str], rows: list[list[Any]], resource: str = "x") -> dict:
    return {"resource": resource, "parameters": {}, "resultSets": [{"name": name, "headers": list(headers), "rowSet": [list(r) for r in rows]}]}


PLAYER_LOG_HEADERS = [
    "SEASON_YEAR", "PLAYER_ID", "PLAYER_NAME", "NICKNAME", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME",
    "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PF", "PTS", "PLUS_MINUS",
    "MIN_SEC", "AVAILABLE_FLAG",
]

LEAGUE_P_HEADERS = [
    "SEASON_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID",
    "GAME_DATE", "MATCHUP", "WL", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA",
    "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS", "FANTASY_PTS",
    "VIDEO_AVAILABLE",
]

TEAM_LOG_HEADERS = [
    "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN",
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST",
    "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS", "VIDEO_AVAILABLE",
]

# team ids used in fixtures
BOS, NYK, GSW, BKN = 1610612738, 1610612752, 1610612744, 1610612751


def player_row(
    player_id=201939, name="Stephen Curry", team_id=GSW, abbr="GSW", game_id="0021800002",
    date="2018-10-16T00:00:00", matchup="GSW vs. OKC", minutes: Any = 33.5, fgm=8, fga=17, fg3m=4, fg3a=9,
    ftm=3, fta=3, oreb=1, dreb=4, ast=6, stl=1, blk=0, tov=2, pf=2, plus_minus=8, season="2018-19",
    reb=None, pts=None,
) -> list[Any]:
    """One ``playergamelogs``-shaped row; pts/reb default to the box-score identities."""
    pts = 2 * fgm + fg3m + ftm if pts is None else pts
    reb = oreb + dreb if reb is None else reb
    ok = isinstance(minutes, (int, float)) and minutes == minutes  # excludes None/str/NaN
    mmss = f"{int(minutes)}:{int(round((minutes % 1) * 60)):02d}" if ok else ""
    return [
        season, player_id, name, name.split()[0], team_id, abbr, "Team", game_id, date, matchup, "W", minutes,
        fgm, fga, round(fgm / fga, 3) if fga else None, fg3m, fg3a, round(fg3m / fg3a, 3) if fg3a else None,
        ftm, fta, round(ftm / fta, 3) if fta else None, oreb, dreb, reb, ast, tov, stl, blk, pf, pts,
        plus_minus, mmss, 1,
    ]


def player_payload(rows: list[list[Any]]) -> dict:
    return wrap("PlayerGameLogs", PLAYER_LOG_HEADERS, rows, "gamelogs")


def league_p_payload(rows: list[list[Any]]) -> dict:
    """Convert playergamelogs-shaped rows into the ``leaguegamelog`` P shape (integer minutes)."""
    idx = {h: i for i, h in enumerate(PLAYER_LOG_HEADERS)}
    out = []
    for r in rows:
        m = r[idx["MIN"]]
        out.append([
            "22018", r[idx["PLAYER_ID"]], r[idx["PLAYER_NAME"]], r[idx["TEAM_ID"]], r[idx["TEAM_ABBREVIATION"]],
            "Team", r[idx["GAME_ID"]], r[idx["GAME_DATE"]][:10], r[idx["MATCHUP"]], "W",
            None if m is None else int(round(m)) if isinstance(m, (int, float)) else m,
            r[idx["FGM"]], r[idx["FGA"]], None, r[idx["FG3M"]], r[idx["FG3A"]], None, r[idx["FTM"]], r[idx["FTA"]],
            None, r[idx["OREB"]], r[idx["DREB"]], r[idx["REB"]], r[idx["AST"]], r[idx["STL"]], r[idx["BLK"]],
            r[idx["TOV"]], r[idx["PF"]], r[idx["PTS"]], r[idx["PLUS_MINUS"]], 0.0, 1,
        ])
    return wrap("LeagueGameLog", LEAGUE_P_HEADERS, out, "leaguegamelog")


def team_row(team_id=GSW, abbr="GSW", game_id="0021800002", date="2018-10-16", matchup="GSW vs. OKC",
             pts=108, minutes=240, season_id="22018") -> list[Any]:
    return [season_id, team_id, abbr, "Team", game_id, date, matchup, "W", minutes, 42, 95, 0.442, 7, 26, 0.269,
            17, 18, 0.944, 17, 41, 58, 28, 7, 7, 21, 29, pts, 8, 1]


def team_payload(rows: list[list[Any]]) -> dict:
    return wrap("LeagueGameLog", TEAM_LOG_HEADERS, rows, "leaguegamelog")


def common_player_info_payload(player_id: int, birthdate: str | None = "1988-03-14T00:00:00", position="Guard",
                               height="6-2", weight="185", draft=("2009", "1", "7"), from_year="2009",
                               to_year="2025") -> dict:
    headers = ["PERSON_ID", "DISPLAY_FIRST_LAST", "BIRTHDATE", "HEIGHT", "WEIGHT", "POSITION",
               "FROM_YEAR", "TO_YEAR", "DRAFT_YEAR", "DRAFT_ROUND", "DRAFT_NUMBER"]
    row = [player_id, "Some Player", birthdate, height, weight, position, from_year, to_year, *draft]
    return {"resource": "commonplayerinfo", "parameters": {}, "resultSets": [
        {"name": "CommonPlayerInfo", "headers": headers, "rowSet": [row]},
        {"name": "AvailableSeasons", "headers": ["SEASON_ID"], "rowSet": []},
    ]}


def player_index_payload(rows: list[dict]) -> dict:
    headers = ["PERSON_ID", "PLAYER_LAST_NAME", "PLAYER_FIRST_NAME", "POSITION", "HEIGHT", "WEIGHT",
               "DRAFT_YEAR", "DRAFT_ROUND", "DRAFT_NUMBER", "FROM_YEAR", "TO_YEAR"]
    out = [[r["id"], r.get("last", "Last"), r.get("first", "First"), r.get("pos"), r.get("height"),
            r.get("weight"), r.get("dy"), r.get("dr"), r.get("dn"), r.get("fy"), r.get("ty")] for r in rows]
    return wrap("PlayerIndex", headers, out, "playerindex")


def bio_stats_payload(rows: list[dict]) -> dict:
    headers = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE", "PLAYER_HEIGHT",
               "PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT", "COLLEGE", "COUNTRY", "DRAFT_YEAR", "DRAFT_ROUND",
               "DRAFT_NUMBER", "GP"]
    out = [[r["id"], r.get("name", "N"), r.get("team", GSW), "GSW", r.get("age"), "6-2", r.get("hin", 74),
            str(r.get("w", 190)), "X", "USA", r.get("dy", "2010"), r.get("dr", "1"), r.get("dn", "5"), 50]
           for r in rows]
    return wrap("LeagueDashPlayerBioStats", headers, out, "leaguedashplayerbiostats")


# --------------------------------------------------------------------------- tiny consistent league

from src.contracts import season_start  # noqa: E402  (kept next to its only user)

# player_id -> (name, birthdate, team_id, abbr): p2 is traded BOS -> NYK in the second half of any season
LEAGUE_PLAYERS = {
    11: ("Alpha One", "1995-03-01", BOS, "BOS"),
    12: ("Bravo Two", "1990-10-01", BOS, "BOS"),
    21: ("Charlie Three", "1999-12-25", NYK, "NYK"),
    22: ("Delta Four", None, NYK, "NYK"),  # no birthdate anywhere
}


def league_payloads(season: str, n_games: int = 4) -> dict:
    """A small but fully consistent season: 2 teams, ``n_games`` games, 2 players per team.

    Player points sum to team points and player minutes sum to 240 per team, so the payloads pass
    every cross-table check. Player 12 is traded BOS -> NYK for the second half of the season.
    Returns the payloads keyed the way ``RoutedSession`` serves them.
    """
    yy = season_start(season) % 100
    y = season_start(season)
    p_rows, t_rows = [], []
    for g in range(1, n_games + 1):
        gid = f"002{yy:02d}{g:05d}"
        date = f"{y}-11-{g:02d}T00:00:00"
        traded = g > n_games // 2
        team_of = {11: BOS, 12: NYK if traded else BOS, 21: NYK, 22: NYK}
        abbr_of = {BOS: "BOS", NYK: "NYK"}
        totals = {BOS: 0, NYK: 0}
        for pid, (name, _bd, _t, _a) in LEAGUE_PLAYERS.items():
            tid = team_of[pid]
            home = tid == BOS
            row = player_row(player_id=pid, name=name, team_id=tid, abbr=abbr_of[tid], game_id=gid, date=date,
                             matchup=f"{abbr_of[tid]} vs. X" if home else f"{abbr_of[tid]} @ X", minutes=60.0 + pid % 5,
                             fgm=3 + g, fga=8 + g, fg3m=1, fg3a=3, ftm=2, fta=2, oreb=1, dreb=3 + pid % 3,
                             season=season)
            p_rows.append(row)
            totals[tid] += row[29]  # PTS
        # team minutes: players on the team must sum to 240 -> patch below
        for tid in (BOS, NYK):
            t_rows.append(team_row(team_id=tid, abbr=abbr_of[tid], game_id=gid, date=date[:10],
                                   matchup=f"{abbr_of[tid]} vs. {abbr_of[NYK if tid == BOS else BOS]}" if tid == BOS
                                   else f"{abbr_of[tid]} @ {abbr_of[BOS]}", pts=totals[tid], season_id=f"2{y}"))
    # normalise minutes so each (game, team) sums to exactly 240
    idx = {h: i for i, h in enumerate(PLAYER_LOG_HEADERS)}
    by_key: dict = {}
    for r in p_rows:
        by_key.setdefault((r[idx["GAME_ID"]], r[idx["TEAM_ID"]]), []).append(r)
    for rows in by_key.values():
        total = sum(r[idx["MIN"]] for r in rows)
        for r in rows:
            r[idx["MIN"]] = r[idx["MIN"]] * 240.0 / total
            r[idx["MIN_SEC"]] = ""
    return {
        "playergamelogs": player_payload(p_rows),
        "team": team_payload(t_rows),
        "league_p": league_p_payload(p_rows),
        "bio": bio_stats_payload([
            {"id": pid, "name": n, "age": None if bd is None else _age_jun30(bd, y + 1)}
            for pid, (n, bd, _t, _a) in LEAGUE_PLAYERS.items()]),
    }


def _age_jun30(birth: str, year: int) -> int:
    b = [int(x) for x in birth.split("-")]
    return year - b[0] - ((6, 30) < (b[1], b[2]))


def league_index_payload() -> dict:
    return player_index_payload([
        {"id": pid, "first": n.split()[0], "last": n.split()[1], "pos": "G", "height": "6-4", "weight": "200",
         "dy": 2015, "dr": 1, "dn": pid, "fy": 2015, "ty": 2025} for pid, (n, *_r) in LEAGUE_PLAYERS.items()])


class RoutedSession:
    """Fake requests.Session that answers each stats.nba.com endpoint from prepared payloads."""

    def __init__(self, seasons: list[str], n_games: int = 4, fail_player_info_after: int | None = None,
                 extra: dict | None = None):
        self.by_season = {s: league_payloads(s, n_games) for s in seasons}
        self.index = league_index_payload()
        self.calls: list[tuple[str, dict]] = []
        self.fail_player_info_after = fail_player_info_after
        self._cpi_served = 0
        self.overrides = extra or {}

    def calls_to(self, endpoint: str) -> int:
        return sum(1 for e, _ in self.calls if e == endpoint)

    def get(self, url, params=None, headers=None, timeout=None):
        endpoint = url.rsplit("/", 1)[1]
        params = dict(params or {})
        self.calls.append((endpoint, params))
        key = (endpoint, params.get("Season"), params.get("PlayerOrTeam"))
        if key in self.overrides:
            return FakeResponse(200, self.overrides[key])
        if endpoint == "playerindex":
            return FakeResponse(200, self.index)
        if endpoint == "commonplayerinfo":
            if self.fail_player_info_after is not None and self._cpi_served >= self.fail_player_info_after:
                return FakeResponse(500)
            self._cpi_served += 1
            pid = int(params["PlayerID"])
            bd = LEAGUE_PLAYERS[pid][1]
            return FakeResponse(200, common_player_info_payload(pid, None if bd is None else bd + "T00:00:00"))
        season = params.get("Season")
        if season not in self.by_season:  # a season the source has no games for yet: valid but empty
            p = {"playergamelogs": player_payload([]), "team": team_payload([]), "league_p": league_p_payload([]),
                 "bio": bio_stats_payload([])}
        else:
            p = self.by_season[season]
        if endpoint == "playergamelogs":
            return FakeResponse(200, p["playergamelogs"])
        if endpoint == "leaguegamelog":
            return FakeResponse(200, p["team"] if params["PlayerOrTeam"] == "T" else p["league_p"])
        if endpoint == "leaguedashplayerbiostats":
            return FakeResponse(200, p["bio"])
        raise AssertionError(f"unexpected endpoint {endpoint}")
