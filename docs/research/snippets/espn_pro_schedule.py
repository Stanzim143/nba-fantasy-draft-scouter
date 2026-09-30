"""NBA schedule from ESPN (no auth) -> games per team per Monday-Sunday week.

RESEARCH PROTOTYPE - not imported by the codebase, not run by pytest.

Verified 2026-09-22:
  * GET https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/2027?view=proTeamSchedules_wl
    -> HTTP 200, 423 KB JSON, no cookies.  ``settings.proTeams[*].proGamesByScoringPeriod`` maps
    scoringPeriodId (str) -> [ {id (ESPN event id), date (epoch ms UTC), homeProTeamId, awayProTeamId,
    scoringPeriodId, startTimeTBD, statsOfficial, validForLocking} ].
  * scoringPeriodId is a DAY counter: 1 == 2026-10-20 (opening night), 174 == 2027-04-11 (last regular-season
    day) ==> date = 2026-10-20 + (scoringPeriodId - 1) days (ET calendar day of the tip-off).
  * It contains 1,200 unique games, exactly 80 per team, NOT 82: the NBA published dates/opponents for
    80 of 82 games; the other 2 per team are decided by NBA Cup group play and played Dec 4-10, 2026
    (https://www.nba.com/news/2026-27-nba-regular-season-schedule).  So weeks around Dec 7 are
    INCOMPLETE until ESPN adds those games (this script shows 0.93 games/team for the week of 2026-12-07).
  * Week grid (Mon-Sun) games per team, e.g. 2026-10-26: mean 3.27; 2026-11-30: 1.93; 2027-02-15 (All-Star
    week, break Feb 19-21): 2.0; range per team within a normal week: 2-4 (occasionally 5 in Jan-Feb).
NOT verified: which scoring periods our LEAGUE groups into matchup periods.  ESPN league settings expose it as
``settings.scheduleSettings.matchupPeriods`` ({matchupPeriodId: [scoringPeriodIds]}, per espn-api source
``espn_api/base_settings.py``) - needs the league id (view=mSettings).  The Mon-Sun grouping below is the
usual ESPN convention (week 1 = opening night through the first Sunday) and MUST be replaced by the league's
own matchupPeriods once the league id is available.
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import urllib.request

UA = "Mozilla/5.0 (compatible; nba-fantasy-2026-research/0.1; personal research)"
URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season_id}?view=proTeamSchedules_wl"


def fetch(season_id: int = 2027) -> dict:
    req = urllib.request.Request(URL.format(season_id=season_id), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def games(payload: dict) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for team in payload["settings"]["proTeams"]:
        if team["id"] == 0:
            continue
        for gl in team["proGamesByScoringPeriod"].values():
            for g in gl:
                out[g["id"]] = g
    return out


def games_per_team_week(payload: dict) -> dict[dt.date, collections.Counter]:
    gs = games(payload)
    first = min(gs.values(), key=lambda g: g["date"])
    # anchor: scoring period of the first game -> ET date of its tip-off (UTC-4 in October)
    anchor_dt = dt.datetime.fromtimestamp(first["date"] / 1000, dt.timezone.utc) - dt.timedelta(hours=4)
    anchor = anchor_dt.date() - dt.timedelta(days=first["scoringPeriodId"] - 1)
    weeks: dict[dt.date, collections.Counter] = collections.defaultdict(collections.Counter)
    for g in gs.values():
        d = anchor + dt.timedelta(days=g["scoringPeriodId"] - 1)
        monday = d - dt.timedelta(days=d.weekday())
        weeks[monday][g["homeProTeamId"]] += 1
        weeks[monday][g["awayProTeamId"]] += 1
    return weeks


if __name__ == "__main__":
    p = fetch(2027)
    gs = games(p)
    print(len(gs), "games;", collections.Counter(
        sum(1 for g in gs.values() if t in (g["homeProTeamId"], g["awayProTeamId"]))
        for t in {g["homeProTeamId"] for g in gs.values()}).most_common(3), "(games per team)")
    for monday, c in sorted(games_per_team_week(p).items()):
        v = list(c.values())
        print(monday, f"teams={len(v)} min={min(v)} max={max(v)} mean={sum(v) / 30:.2f}")
