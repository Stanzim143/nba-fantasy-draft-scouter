"""A simulated week of nights (fake clock, fake stats.nba.com, canned ESPN steps) through the real orchestrator.

The games step is the REAL incremental ingest (``src.ops.live_ingest``) against ``nightly_fakes.World``; the ESPN/league/
schedule/analysis steps are canned per night. What this proves: the season-window gate, opening-night handling, incremental
and idempotent ingest (final tables equal a from-scratch pull), an outage handled by failure isolation, the alert
raised after two bad nights and cleared by recovery, catch-up after the outage, a no-game day, and the diffed report.
It does NOT prove anything about live services; see docs/adr/0018-nightly-inseason-job.md for what has and has not run live.
"""
import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from nightly_fakes import SEASON, World

from src.inseason import schedule as sch
from src.ops import daily_refresh as dr
from src.ops import live_ingest as li
from src.ops import nightly as nt
from src.ops import nightly_analysis as na
from src.ops import refresh_diff as rd
from src.store import read_table

NZDT = timezone(timedelta(hours=13))
GAME_DAYS = [date(2026, 10, d) for d in (20, 21, 22, 23, 24, 26, 27, 28)]          # no game on Oct 25


def morning(day: int) -> datetime:
    return datetime(2026, 10, day, 9, 30, tzinfo=NZDT)


class Sim:
    """Step runner for one simulated night; ``night`` says what the outside world did."""

    def __init__(self, base, world):
        self.base, self.world, self.now, self.script, self.calls, self.snap_n = base, world, None, {}, [], 0

    def __call__(self, step, timeout):
        self.calls.append((self.now.day, step))
        through = na.completed_through(self.now)
        script = self.script.get(self.now.day, {})
        if step == "games":
            client = self.world
            r = li.run_live_ingest(SEASON, client, self.base, completed_through=through, opening_night=date(2026, 10, 20), log=lambda *_: None)
            summary = {"skipped": r.skipped, "date_from": r.date_from, "last_game_date": r.last_game_date, "new_games": r.new_games,
                       "new_player_rows": r.new_player_rows, "new_players": r.new_players, "network_requests": r.network_requests}
            return dr.StepOutcome("ok", summary, {"games": summary})
        if step == "snapshot":
            status = script.get("status", "ACTIVE")
            raw = [{"id": 1, "fullName": "Alpha One", "proTeamId": 2, "injuryStatus": status, "injured": False, "ownership": {"percentOwned": 95.0}},
                   {"id": 2, "fullName": "Charlie Three", "proTeamId": 18, "injuryStatus": "ACTIVE", "injured": False, "ownership": {"percentOwned": 40.0}}]
            path, _ = rd.write_snapshot(self.base, SEASON, raw, self.now.astimezone(timezone.utc) - timedelta(minutes=5))
            return dr.StepOutcome("ok", {}, {"snapshot": str(path), "snapshot_taken_at": (self.now.astimezone(timezone.utc) - timedelta(minutes=5)).isoformat()})
        if step == "league":
            mine = ["1", "2"] + (["9"] if script.get("pickup") else [])
            st = {"rosters": {"3": {"name": "My Team", "players": mine}, "4": {"name": "Rival", "players": ["5"]}},
                  "names": {"1": "Alpha One", "2": "Charlie Three", "9": "Waiver Guy", "5": "Rival Guy"}, "transactions": [],
                  "fetched_at": self.now.astimezone(timezone.utc).isoformat()}
            return dr.StepOutcome("ok", {"name": "Example League"}, {"league_state": st})
        if step == "schedule":
            s = {"games": len(GAME_DAYS), "changed": False, "moved": 0, "added": 0, "removed": 0}
            return dr.StepOutcome("ok", s, {"schedule": s})
        if step == "roster":
            return dr.StepOutcome("ok", {}, {"roster_map": {"11": ["Alpha One", "BOS"]}})
        if step == "analysis":
            rising = {"21": {"name": "Charlie Three", "min_delta": 5.0}} if script.get("rising") else {}
            report = {"meta": {"team_label": "My Team (ESPN team 3)", "week": {"slate_games": 1, "calendar_source": "derived"}}, "tables": {}, "notes": [], "errors": {}, "files": {}}
            return dr.StepOutcome("ok", {}, {"analysis": {"rising_minutes": rising, "calendar_source": "derived"}, "_report": report})
        return dr.StepOutcome("ok")


def test_a_simulated_week_of_nights(tmp_path):
    base, rep = tmp_path / "data", tmp_path / "rep"
    (base / "processed").mkdir(parents=True)
    sch.write_schedule(sch._frame([{"season": SEASON, "game_id": f"s{i}", "scoring_period": None, "game_date": pd.Timestamp(d), "home_team_id": 1610612738,
                                    "away_team_id": 1610612752, "home_abbr": "BOS", "away_abbr": "NYK", "source": "espn", "time_tbd": False}
                                   for i, d in enumerate(GAME_DAYS)]), base)
    world = World()
    sim = Sim(base, world)
    toasts = []
    outcomes = {}

    def night(day, **script):
        sim.now = morning(day)
        sim.script[day] = script
        through = na.completed_through(sim.now)
        for i, d in enumerate(GAME_DAYS):                           # the real world plays the schedule; the fake API lists finished games
            if d <= through and i >= len(world.games):
                world.play(d.isoformat())
        rc = nt.main(["--data-dir", str(base), "--reports-dir", str(rep), "--repo-root", str(tmp_path), "--season", SEASON, "--league-id", "1",
                      "--team-id", "3", "--quiet"], runner=sim, now=sim.now, env={}, notifier=lambda t, m: toasts.append((day, m)))
        st = json.loads((base / "nightly" / "status.json").read_text()) if (base / "nightly" / "status.json").exists() else None
        outcomes[day] = (rc, st)
        return rc

    # ---- Oct 19: the day before the window opens: the gate keeps everything quiet
    assert night(19) == 0 and sim.calls == [] and not (base / "nightly" / "status.json").exists() and not rep.exists()

    # ---- Oct 20 and 21: window open, no regular-season game has been played: games step skips without any request
    assert night(20) == 0 and night(21) == 0
    assert world.calls == [] and not (base / "processed" / "game_logs.parquet").exists()
    assert "opens 2026-10-20" in (rep / "2026-10-21.md").read_text(encoding="utf-8")

    # ---- Oct 22: the first real night (US Oct 20): everything is new
    assert night(22) == 0
    tg = read_table("team_games", base)
    assert tg["game_id"].nunique() == 1 and world.stats.network_requests == 8       # 4 pulls + a CommonPlayerInfo per new player
    md = (rep / "latest.md").read_text(encoding="utf-8")
    assert "1 new game(s)" in md and "games stored through 2026-10-20" in md and "First archived nightly run" not in md      # earlier nights exist

    # ---- Oct 23: incremental (window from Oct 17), a rookie debuts (players refreshed), an injury and a waiver pickup show up in the diff
    world.rookie_from = date(2026, 10, 21)
    world.calls.clear()
    assert night(23, status="OUT", pickup=True, rising=True) == 0
    sent = [p for e, p in world.calls if e == "playergamelogs"]
    assert len(sent) == 1 and sent[0]["DateFrom"] == "10/17/2026"
    assert world.calls_to("playerindex") == 1 and 23 in set(read_table("players", base)["player_id"])
    md = (rep / "2026-10-23.md").read_text(encoding="utf-8")
    assert "Alpha One: ACTIVE -> OUT" in md and "added Waiver Guy" in md and "now rising in minutes/usage: Charlie Three" in md

    # ---- Oct 24 and 25: stats.nba.com is down. Isolation: every other step still runs; alert only after the second bad night
    world.outage = True
    world.calls.clear()
    assert night(24) == 1 and [s for d, s in sim.calls if d == 24] == list(nt.STEP_ORDER)
    assert outcomes[24][1]["outcome"] == "partial" and outcomes[24][1]["steps"]["games"]["status"] == "failed"
    assert not (rep / "ALERT.txt").exists() and toasts == []
    assert night(25) == 1
    assert (rep / "ALERT.txt").exists() and len(toasts) == 1 and toasts[0][0] == 25
    alert = (rep / "ALERT.txt").read_text()
    assert "consecutive runs had failing steps" in alert and "game data is stale" in alert
    assert outcomes[25][1]["consecutive_failures"] == 2 and outcomes[25][1]["alert"] is True
    assert "**games** (failed)" in (rep / "latest.md").read_text(encoding="utf-8")

    # ---- Oct 26: recovered; catches up three game days in one window; alert cleared
    world.outage = False
    world.calls.clear()
    assert night(26) == 0
    assert not (rep / "ALERT.txt").exists() and outcomes[26][1]["consecutive_failures"] == 0 and outcomes[26][1]["alert"] is False
    assert outcomes[26][1]["steps"]["games"]["summary"]["new_games"] == 3          # Oct 22, 23, 24 (the outage nights' games)
    assert read_table("team_games", base)["game_id"].nunique() == 5 and len(toasts) == 1

    # ---- Oct 27: US Oct 25 had no games (an off day in the fake calendar): nothing new, the report says so, still healthy
    assert night(27) == 0
    assert outcomes[27][1]["steps"]["games"]["summary"]["new_games"] == 0 and outcomes[27][1]["steps"]["games"]["summary"]["last_game_date"] == "2026-10-24"

    # ---- a second run on the same morning changes nothing at all (idempotent), and the lock is free afterwards
    before = {n: (base / "processed" / f"{n}.parquet").stat().st_mtime_ns for n in ("game_logs", "team_games", "players", "player_season_bio")}
    assert night(27) == 0
    assert outcomes[27][1]["steps"]["games"]["summary"]["new_games"] == 0
    assert before == {n: (base / "processed" / f"{n}.parquet").stat().st_mtime_ns for n in before}
    assert not (base / "daily_refresh" / "lock.json").exists()

    # ---- the stored tables equal a from-scratch pull of everything the fake API knows: incremental == full
    fresh_base = tmp_path / "fresh"
    li.run_live_ingest(SEASON, world, fresh_base, completed_through=date(2026, 10, 27), log=lambda *_: None)
    for name in ("game_logs", "team_games"):
        pd.testing.assert_frame_equal(read_table(name, base), read_table(name, fresh_base))

    # ---- bookkeeping: one dated report per executed night, history, rotating log, one toast in the whole week
    reports = sorted(p.name for p in rep.glob("2026-*.md"))
    assert reports == [f"2026-10-{d}.md" for d in (20, 21, 22, 23, 24, 25, 26, 27)]
    history = [json.loads(line) for line in (base / "nightly" / "history.jsonl").read_text().splitlines()]
    assert [h["ok"] for h in history] == [True, True, True, True, False, False, True, True, True]
    assert (base / "nightly" / "logs" / "nightly.log").exists() and len(toasts) == 1
