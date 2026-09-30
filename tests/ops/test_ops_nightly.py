"""Nightly job orchestration with a fake step runner: window gate, time zones, settings, isolation, alerts, diffs, report."""
import json
import sys
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from src.inseason import schedule as sch
from src.ops import daily_refresh as dr
from src.ops import nightly as nt
from src.ops import nightly_analysis as na
from src.ops import refresh_diff as rd
from src.ops.nightly_report import render_report
from src.ops.runlock import RunLock

NZDT = timezone(timedelta(hours=13))
NOW = datetime(2026, 10, 22, 9, 30, tzinfo=NZDT)          # the morning after the first full night of games


def parse(tmp_path, *argv):
    return nt.build_parser().parse_args(["--data-dir", str(tmp_path / "data"), "--reports-dir", str(tmp_path / "rep"),
                                         "--repo-root", str(tmp_path), "--league-id", "1", "--team-id", "3", *argv])


def settings(tmp_path, now=NOW, **kw):
    s = nt.resolve_settings(parse(tmp_path, *kw.pop("argv", [])), {}, now)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


class Fake:
    """Step runner with canned outcomes; ``fail``/``degrade`` name steps; every call is recorded."""

    def __init__(self, tmp, fail=(), degrade=(), state_for=None):
        self.tmp, self.fail, self.degrade, self.calls = tmp, set(fail), set(degrade), []
        self.state_for = state_for or self.fresh_state

    @staticmethod
    def fresh_state(step):
        """What a healthy night's steps would report: an ESPN snapshot and a league sync taken this very morning."""
        stamp = NOW.astimezone(timezone.utc).isoformat()
        return {"snapshot_taken_at": stamp} if step == "snapshot" else {"league_state": {"fetched_at": stamp}} if step == "league" else {}

    def __call__(self, step, timeout):
        self.calls.append(step)
        if step in self.fail:
            return dr.StepOutcome("failed", error=f"{step} exploded")
        if step in self.degrade:
            return dr.StepOutcome("degraded", {}, self.state_for(step), error=f"{step}: one artifact failed")
        return dr.StepOutcome("ok", {}, self.state_for(step))


# --------------------------------------------------------------------------- time zones and the "last completed game day"

def test_eastern_offset_switches_on_the_us_dst_dates():
    off = lambda y, m, d, h: na.eastern_offset_hours(datetime(y, m, d, h, tzinfo=timezone.utc))  # noqa: E731
    assert off(2026, 10, 20, 12) == -4 and off(2026, 11, 1, 5) == -4 and off(2026, 11, 1, 7) == -5      # ends 06:00 UTC Nov 1 2026
    assert off(2027, 3, 14, 6) == -5 and off(2027, 3, 14, 8) == -4 and off(2027, 4, 1, 0) == -4          # starts 07:00 UTC Mar 14 2027


@pytest.mark.parametrize("utc, expected", [
    ("2026-10-21T07:00", "2026-10-19"),   # 03:00 ET: the Oct 20 late games (10:30 pm tips) may still be running
    ("2026-10-21T08:00", "2026-10-20"),   # 04:00 ET: everything from Oct 20 is final
    ("2026-10-21T20:30", "2026-10-20"),   # 16:30 ET on Oct 21, before that night's tips: still Oct 20
    ("2026-10-22T03:00", "2026-10-20"),   # Oct 21's games are in progress
])
def test_completed_through_uses_us_eastern_game_dates(utc, expected):
    assert na.completed_through(datetime.fromisoformat(utc).replace(tzinfo=timezone.utc)) == date.fromisoformat(expected)


def test_the_default_09_30_new_zealand_run_sees_the_previous_us_night_as_final():
    # 09:30 NZDT (UTC+13) is 16:30 EDT the previous US day: that day's games have not started, the day before's are final.
    assert na.completed_through(NOW) == date(2026, 10, 20)
    # An evening run would only help once the Pacific-time games are over: 19:30 NZDT is 02:30 EDT (still running), 20:30 is 03:30 EDT (final).
    assert na.completed_through(NOW.replace(hour=19, minute=30)) == date(2026, 10, 20)
    assert na.completed_through(NOW.replace(hour=20, minute=30)) == date(2026, 10, 21)


# --------------------------------------------------------------------------- the season window

def write_schedule(base, first="2026-10-20", last="2027-04-12"):
    days = pd.date_range(first, last, freq="7D")
    rows = [{"season": "2026-27", "game_id": f"g{i}", "scoring_period": None, "game_date": d, "home_team_id": 1610612738,
             "away_team_id": 1610612752, "home_abbr": "BOS", "away_abbr": "NYK", "source": "espn", "time_tbd": False}
            for i, d in enumerate(days)]
    sch.write_schedule(sch._frame(rows), base)


def test_window_falls_back_to_the_opening_night_and_167_days_without_a_schedule(tmp_path):
    start, end, how = nt.season_window("2026-27", tmp_path, None)
    assert start == date(2026, 10, 20) and end == date(2026, 10, 20) + timedelta(days=166 + nt.GRACE_DAYS)
    assert "constant" in how and "no schedule" in how
    assert nt.season_window("2027-28", tmp_path, None)[0] == date(2027, 10, 20)         # any other season: October 20


def test_window_comes_from_the_schedule_table_and_the_leagues_final_period(tmp_path):
    write_schedule(tmp_path, first="2026-10-21")
    start, end, how = nt.season_window("2026-27", tmp_path, None)
    assert start == date(2026, 10, 21) and "first game in schedule_games" in how
    assert "league final matchup period" in how and end > start + timedelta(days=100)
    assert nt.season_window("2026-27", tmp_path, None, start_override=date(2026, 10, 25), end_override=date(2027, 1, 1))[:2] == (date(2026, 10, 25), date(2027, 1, 1))


def test_missing_league_config_still_gives_a_window(tmp_path, monkeypatch):
    write_schedule(tmp_path)
    import src.value.league as league

    monkeypatch.setattr(league, "load_league", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no league.yaml")))
    start, end, how = nt.season_window("2026-27", tmp_path, 1)
    assert start == date(2026, 10, 20) and "last scheduled game" in how and end == date(2027, 4, 6) + timedelta(days=nt.GRACE_DAYS)   # the last weekly test game


def test_settings_precedence_and_bad_input(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "daily_refresh.json").write_text(json.dumps({"league_id": 7, "team_id": 2}))
    ns = nt.build_parser().parse_args(["--data-dir", str(data), "--repo-root", str(tmp_path)])
    s = nt.resolve_settings(ns, {}, NOW)
    assert (s.league_id, s.team_id) == (7, 2)                                   # the pre-draft file supplies them
    (data / "nightly.json").write_text(json.dumps({"team_id": 5}))
    assert nt.resolve_settings(ns, {}, NOW).team_id == 5                        # nightly.json beats it
    assert nt.resolve_settings(ns, {"ESPN_TEAM_ID": "9"}, NOW).team_id == 9    # env beats the file
    ns2 = nt.build_parser().parse_args(["--data-dir", str(data), "--repo-root", str(tmp_path), "--team-id", "11"])
    assert nt.resolve_settings(ns2, {"ESPN_TEAM_ID": "9"}, NOW).team_id == 11  # flag beats env
    empty = nt.resolve_settings(nt.build_parser().parse_args(["--data-dir", str(tmp_path / "none"), "--repo-root", str(tmp_path)]), {}, NOW)
    assert empty.league_id is None and empty.team_id is None
    for bad in (["--season", "nonsense"], ["--window-end", "soon"], ["--only", "nope"], ["--as-of", "yesterday"]):
        with pytest.raises(nt.ConfigError):
            nt.resolve_settings(nt.build_parser().parse_args(["--data-dir", str(tmp_path / "x"), "--repo-root", str(tmp_path), *bad]), {}, NOW)
    (data / "nightly.json").write_text("[1]")
    with pytest.raises(nt.ConfigError):
        nt.resolve_settings(ns, {}, NOW)


# --------------------------------------------------------------------------- the gate

def run_main(tmp_path, now, fake, *extra):
    return nt.main(["--data-dir", str(tmp_path / "data"), "--reports-dir", str(tmp_path / "rep"), "--repo-root", str(tmp_path),
                    "--league-id", "1", "--quiet", *extra], runner=fake, now=now, env={}, notifier=lambda *a: None)


@pytest.mark.parametrize("day", [datetime(2026, 10, 19, 9, 30, tzinfo=NZDT), datetime(2027, 4, 7, 9, 30, tzinfo=NZDT),
                                 datetime(2026, 7, 1, 9, 30, tzinfo=NZDT)])
def test_outside_the_window_does_nothing_and_exits_zero(tmp_path, day):
    fake = Fake(tmp_path)
    assert run_main(tmp_path, day, fake) == 0 and fake.calls == [] and not (tmp_path / "rep").exists()


@pytest.mark.parametrize("day", [datetime(2026, 10, 20, 9, 30, tzinfo=NZDT), datetime(2027, 4, 6, 9, 30, tzinfo=NZDT)])
def test_first_and_last_day_of_the_window_run(tmp_path, day):
    fake = Fake(tmp_path)
    assert run_main(tmp_path, day, fake) == 0 and fake.calls == list(nt.STEP_ORDER)


def test_force_and_as_of_override_the_gate(tmp_path):
    fake = Fake(tmp_path)
    late = datetime(2027, 6, 1, 9, 30, tzinfo=NZDT)
    assert run_main(tmp_path, late, fake, "--force") == 0 and fake.calls
    fake2 = Fake(tmp_path)
    assert run_main(tmp_path, late, fake2, "--as-of", "2026-12-10", "--season", "2026-27") == 0 and fake2.calls           # the slate day 12-11 is in season
    fake3 = Fake(tmp_path)
    assert run_main(tmp_path, NOW, fake3, "--as-of", "2026-05-01", "--season", "2026-27") == 0 and fake3.calls == []      # replaying an off-season day: nothing to do
    rep = (tmp_path / "rep" / "latest.md").read_text(encoding="utf-8")
    assert "REPLAY" in rep and "2026-12-11" in "".join(p.name for p in (tmp_path / "rep").glob("*.md"))     # dated by the slate, not by today


def test_dry_run_writes_nothing_and_says_where_it_stands(tmp_path, capsys):
    assert run_main(tmp_path, NOW, None, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "INSIDE the window 2026-10-20" in out and "completed through 2026-10-20" in out and "team id: not configured" in out
    assert not (tmp_path / "data").exists() and not (tmp_path / "rep").exists()
    assert run_main(tmp_path, NOW, None, "--dry-run", "--only", "nope") == nt.EXIT_CONFIG


# --------------------------------------------------------------------------- isolation, alerts, lock

def test_a_failing_or_degraded_step_never_stops_the_others_and_alerts_after_two_nights(tmp_path):
    s = settings(tmp_path, stale_hours=1000.0)          # the canned ESPN state is timestamped once; only the failure count is under test
    toasts = []
    bad = Fake(tmp_path, fail=("games", "adp"), degrade=("analysis",))
    assert nt.execute(s, bad, now=NOW, through=date(2026, 10, 20), slate=date(2026, 10, 21), notifier=lambda t, m: toasts.append(m)) == 1
    assert bad.calls == list(nt.STEP_ORDER)
    st = json.loads((tmp_path / "data" / "nightly" / "status.json").read_text())
    assert st["outcome"] == "partial" and st["steps"]["analysis"]["status"] == "degraded" and st["exit_code"] == 1
    assert not (tmp_path / "rep" / "ALERT.txt").exists() and toasts == []
    assert nt.execute(s, bad, now=NOW + timedelta(days=1), through=date(2026, 10, 21), slate=date(2026, 10, 22), notifier=lambda t, m: toasts.append(m)) == 1
    alert = (tmp_path / "rep" / "ALERT.txt").read_text()
    assert "consecutive runs had failing steps" in alert and "so far: 2" in alert and len(toasts) == 1
    assert nt.execute(s, bad, now=NOW + timedelta(days=2), through=date(2026, 10, 22), slate=date(2026, 10, 23), notifier=lambda t, m: toasts.append(m)) == 1
    assert len(toasts) == 1                                        # the same reasons do not toast again every night
    good = Fake(tmp_path)
    assert nt.execute(s, good, now=NOW + timedelta(days=3), through=date(2026, 10, 23), slate=date(2026, 10, 24), notifier=lambda *a: None) == 0
    assert not (tmp_path / "rep" / "ALERT.txt").exists()


def test_a_crashing_runner_and_a_busy_shared_lock(tmp_path):
    s = settings(tmp_path, argv=["--lock-wait-minutes", "0"])

    def crashing(step, timeout):
        if step == "roster":
            raise RuntimeError("kaboom")
        return dr.StepOutcome("ok")
    assert nt.execute(s, crashing, now=NOW, through=date(2026, 10, 20), slate=date(2026, 10, 21), notifier=lambda *a: None) == 1
    assert "kaboom" in (tmp_path / "rep" / "latest.md").read_text(encoding="utf-8")
    held = RunLock(tmp_path / "data" / "daily_refresh" / "lock.json").acquire()          # the pre-draft refresh is running
    f = Fake(tmp_path)
    assert nt.execute(s, f, now=NOW, through=date(2026, 10, 20), slate=date(2026, 10, 21), notifier=lambda *a: None) == nt.EXIT_BUSY and f.calls == []
    held.release()
    assert not (tmp_path / "data" / "daily_refresh" / "lock.json").exists()


def test_league_is_skipped_cleanly_without_a_league_id(tmp_path):
    s = settings(tmp_path, league_id=None)
    f = Fake(tmp_path)
    assert nt.execute(s, f, now=NOW, through=date(2026, 10, 20), slate=date(2026, 10, 21), notifier=lambda *a: None) == 0
    assert "league" not in f.calls and "analysis" in f.calls
    assert json.loads((tmp_path / "data" / "nightly" / "status.json").read_text())["steps"]["league"]["status"] == "skipped"


def test_budget_exhaustion_is_reported_as_a_failure(tmp_path):
    ticks = iter(range(0, 100000, 1500))
    s = settings(tmp_path, budget_s=3000.0)
    assert nt.execute(s, Fake(tmp_path), now=NOW, through=date(2026, 10, 20), slate=date(2026, 10, 21), notifier=lambda *a: None,
                      clock=lambda: next(ticks)) == 1
    st = json.loads((tmp_path / "data" / "nightly" / "status.json").read_text())
    assert any(v["status"] == "skipped" and "budget" in (v["error"] or "") for v in st["steps"].values())


def test_stale_espn_data_and_stale_games_raise_alerts_without_any_step_failing(tmp_path):
    write_schedule(tmp_path / "data", first="2026-10-20")
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"season": ["2026-27"], "game_date": [pd.Timestamp("2026-10-13")]}).to_parquet(proc / "team_games.parquet")
    s = settings(tmp_path)

    def state(step):
        return {"snapshot_taken_at": "2026-10-19T00:00:00+00:00"} if step == "snapshot" else {}
    assert nt.execute(s, Fake(tmp_path, state_for=state), now=NOW, through=date(2026, 10, 27), slate=date(2026, 10, 28),
                      notifier=lambda *a: None) == 0
    alert = (tmp_path / "rep" / "ALERT.txt").read_text()
    assert "ESPN data is" in alert and "game data is stale" in alert


def test_replay_never_alerts(tmp_path):
    s = settings(tmp_path, as_of=date(2026, 12, 10))
    bad = Fake(tmp_path, fail=("games",))
    for i in range(3):
        nt.execute(s, bad, now=NOW + timedelta(days=i), through=date(2026, 12, 10), slate=date(2026, 12, 11), notifier=lambda *a: pytest.fail("toast"))
    assert not (tmp_path / "rep" / "ALERT.txt").exists()


# --------------------------------------------------------------------------- diffs and the report

def test_build_diffs_league_injury_minutes_and_calendar(tmp_path):
    raw = lambda a, b: [{"id": 1, "fullName": "Star", "proTeamId": 1, "injuryStatus": a, "injured": False, "ownership": {"percentOwned": 99.0}},  # noqa: E731
                        {"id": 2, "fullName": "Other", "proTeamId": 1, "injuryStatus": b, "injured": False, "ownership": {"percentOwned": 10.0}}]
    p1, _ = rd.write_snapshot(tmp_path, "2026-27", raw("ACTIVE", "ACTIVE"), datetime(2026, 10, 22, tzinfo=timezone.utc))
    p2, _ = rd.write_snapshot(tmp_path, "2026-27", raw("ACTIVE", "OUT"), datetime(2026, 10, 23, tzinfo=timezone.utc))
    league1 = {"rosters": {"3": {"name": "Mine", "players": ["1"]}, "4": {"name": "Rival", "players": ["2"]}}, "names": {"1": "Star", "2": "Other", "9": "Waiver Guy"}, "transactions": []}
    league2 = {"rosters": {"3": {"name": "Mine", "players": ["1", "9"]}, "4": {"name": "Rival", "players": []}}, "names": league1["names"],
               "transactions": [{"id": 5, "type": "WAIVER", "status": "EXECUTED", "team_id": 3, "items": [{"type": "ADD", "player": "9", "team": 3}]}],
               "my_team_id": "3", "my_espn_ids": ["1", "9"]}
    prev = {"snapshot": str(p1), "league_state": league1, "roster_map": {"1": ["Star", "BOS"]},
            "analysis": {"rising_minutes": {"5": {"name": "Riser", "min_delta": 4.0}}, "adds_top": {"7": {"name": "A", "gain": 10.0, "drop": "X"}},
                         "ros_rank": {"1": {"rank": 5, "name": "Star"}, "2": {"rank": 50, "name": "Other"}}, "calendar_source": "derived"}}
    cur = {"snapshot": str(p2), "league_state": league2, "roster_map": {"1": ["Star", "MIA"]},
           "analysis": {"rising_minutes": {"6": {"name": "NewRiser", "min_delta": 5.0}}, "adds_top": {"8": {"name": "B", "gain": 12.0, "drop": "Y"}},
                        "ros_rank": {"1": {"rank": 30, "name": "Star"}, "2": {"rank": 45, "name": "Other"}}, "calendar_source": "espn"}}
    d = nt.build_diffs([prev], cur)
    assert d["injuries"][0]["name"] == "Other" and d["injuries"][0]["to"] == "OUT"
    assert d["league"]["moves"][0]["mine"] and d["league"]["moves"][0]["added"] == ["Waiver Guy"]
    assert d["league"]["new_transactions"][0]["items"] == ["ADD Waiver Guy"]
    assert [x["name"] for x in d["rising"]["new"]] == ["NewRiser"] and [x["name"] for x in d["rising"]["gone"]] == ["Riser"]
    assert d["adds"]["new"][0]["name"] == "B" and d["ros"]["fallers"][0]["name"] == "Star"
    assert d["calendar"] == {"from": "derived", "to": "espn"} and d["roster"]["moved"][0]["to"] == "MIA"
    assert nt.build_diffs([], cur) == {}                                    # first ever run: nothing to diff


def record(**over):
    run = {"run_id": "20261022T000000Z", "date": "2026-10-22", "season": "2026-27", "outcome": "ok", "seconds": 90.0, "completed_through": "2026-10-20",
           "slate": "2026-10-21", "window": ["2026-10-20", "2027-04-06"], "days_left": 167, "prev_run_id": None, "steps": [{"name": "games", "status": "ok", "seconds": 3}],
           "diffs": {}, "games": {"new_games": 7, "new_player_rows": 210, "last_game_date": "2026-10-20", "date_from": "2026-10-17", "network_requests": 2},
           "schedule": {"games": 1230, "changed": True, "moved": 1, "added": 0, "removed": 0}, "analysis": None, "freshness": {}, "alert": {"reasons": []}, "files": {}}
    run.update(over)
    return run


def test_report_renders_every_section_and_survives_missing_pieces():
    md = render_report(record())
    assert "# Nightly 2026-10-22: OK" in md and "7 new game(s)" in md and "CHANGED: 1 moved" in md and "First archived nightly run" in md
    full = record(prev_run_id="p", analysis={
        "meta": {"team_label": "Mine (ESPN team 3)", "week": {"slate_games": 0, "next_game_day": "2027-02-19", "calendar_source": "espn",
                                                               "this_week": {"week": 4, "start": "2026-11-09", "end": "2026-11-15", "kind": "regular", "mean_games": 3.4}},
                 "lineup": {"week": 4, "week_expected_fp": 812.0, "empty_slots": ["C"], "today_players": 6, "today_expected_fp": 210.0, "slate": "2026-11-10",
                            "swaps": [{"add": "Stream Guy", "drop": "Bench Guy", "week_gain_fp": 22.0, "games": 3.0}]}},
        "tables": {"week_this_week": [{"abbr": "BOS", "games": 4, "games_left": 3, "b2b": 1, "off_night_games": 1, "heavy": True, "light": False, "mine": True}],
                   "waivers_adds": [{"name": "Add Guy", "position": "F", "team": "BOS", "gain_ros_fp": 120.5, "drop": "Drop Guy", "ros_fppg": 30.0, "ros_games": 60.0, "injury": ""}],
                   "alerts_rising": [{"name": "Riser", "team": "NYK", "status": "free agent", "base_mpg": 15.0, "recent_mpg": 24.0, "min_delta": 9.0}],
                   "lineup": [{"name": "Me Guy", "position": "G", "team": "BOS", "slot_this_week": "PG", "games_left_week": 3.0, "plays_today": True, "week_exp_fp": 120.0, "hint": "check"}]},
        "notes": ["a note"], "errors": {"alerts": "boom"}},
        alert={"reasons": ["something stale"]}, diffs={"calendar": {"from": "derived", "to": "espn"}, "injuries": [{"name": "Star", "from": "ACTIVE", "to": "OUT", "mine": True, "pct_owned": 99.0}]})
    md = render_report(full)
    for needle in ("## ALERT", "something stale", "no games on the next slate", "next game day 2027-02-19", "Add Guy", "120.5", "Riser", "Stream Guy",
                   "MATCHUP CALENDAR SOURCE CHANGED", "**MINE** Star: ACTIVE -> OUT", "artifact `alerts` failed", "Empty starting slots: C", "Note: a note"):
        assert needle in md, needle


# --------------------------------------------------------------------------- subprocess boundary and workers

def test_subprocess_runner_maps_worker_results(tmp_path):
    s = settings(tmp_path)

    def proc(payload, code=0):
        def run(cmd, timeout, cwd, env):
            if payload is not None:
                open(cmd[cmd.index("--result-file") + 1], "w", encoding="utf-8").write(json.dumps(payload))
            return code, ""
        return run
    r = lambda p, c=0: nt.SubprocessRunner(s, date(2026, 10, 20), date(2026, 10, 21), process_runner=proc(p, c))("analysis", 5)  # noqa: E731
    assert r({"ok": True, "summary": {"a": 1}, "state": {"k": 1}}).status == "ok"
    o = r({"ok": True, "summary": {}, "state": {"k": 1}, "degraded": ["lineup: boom"]})
    assert o.status == "degraded" and "lineup: boom" in o.error and o.state == {"k": 1}
    assert r({"ok": False, "error": "X"}, 1).status == "failed"
    assert r(None, 1).status == "failed" and "without a result" in r(None, 1).error
    assert nt.SubprocessRunner(s, date(2026, 10, 20), date(2026, 10, 21), process_runner=lambda *a, **k: (None, ""))("games", 1).status == "timeout"


def test_worker_result_file_carries_degraded_and_keeps_report_state_out_of_the_summary(tmp_path, monkeypatch):
    monkeypatch.setitem(nt.WORKERS, "analysis", lambda a: ({"x": float("nan")}, {"analysis": {"k": 1}, "_report": {"meta": {}}, "_degraded": ["a: b"]}))
    res = tmp_path / "r.json"
    a = nt.build_parser().parse_args(["--worker", "analysis", "--result-file", str(res), "--data-dir", str(tmp_path), "--reports-dir", str(tmp_path), "--through", "2026-10-20", "--slate", "2026-10-21"])
    a.league_id = None
    assert nt.run_worker(a) == 0
    got = json.loads(res.read_text())
    assert got["degraded"] == ["a: b"] and got["summary"] == {"x": None} and "_degraded" not in got["state"]
    assert nt.main(["--worker", "analysis"]) == nt.EXIT_CONFIG


def test_process_timeout_really_kills(tmp_path):
    code, _ = dr.run_worker_process([sys.executable, "-c", "import time; time.sleep(30)"], 2, cwd=tmp_path, env={})
    assert code is None


def espn_payload(games):
    """games: list of (game_id, scoring_period, home_espn_id, away_espn_id); day one is 2026-10-20 (period 1)."""
    by = {}
    for gid, sp, h, a in games:
        ms = int((datetime(2026, 10, 20, 23, 0, tzinfo=timezone.utc) + timedelta(days=sp - 1)).timestamp() * 1000)
        g = {"id": gid, "date": ms, "scoringPeriodId": sp, "homeProTeamId": h, "awayProTeamId": a}
        for t in (h, a):
            by.setdefault(t, {}).setdefault(str(sp), []).append(g)
    return {"settings": {"proTeams": [{"id": t, "proGamesByScoringPeriod": v} for t, v in by.items()]}}


def test_schedule_worker_reports_moves_backs_up_and_refuses_a_shrunken_schedule(tmp_path, monkeypatch):
    base = [(i, 1 + i % 5, 2, 18) for i in range(1, 30)]
    payloads = iter([espn_payload(base), espn_payload(base), espn_payload([(g, sp + 1 if g == 1 else sp, h, a) for g, sp, h, a in base] + [(99, 3, 2, 18)]),
                     espn_payload(base[:5])])
    monkeypatch.setattr(sch, "fetch_pro_schedule", lambda client, season, refresh=False: next(payloads))
    a = nt.build_parser().parse_args(["--worker", "schedule", "--result-file", "x", "--data-dir", str(tmp_path), "--through", "2026-10-20", "--slate", "2026-10-21", "--season", "2026-27"])
    s1, _ = nt.worker_schedule(a)
    assert s1["changed"] and s1["games"] == 29 and s1["added"] == 29                       # first time: everything is new
    s2, _ = nt.worker_schedule(a)
    assert not s2["changed"] and s2["moved"] == 0                                           # identical: nothing rewritten
    s3, st3 = nt.worker_schedule(a)
    assert s3["moved"] == 1 and s3["added"] == 1 and st3["schedule"]["moved_games"][0]["game_id"] == "1"
    assert (tmp_path / "processed" / "schedule_games.parquet.prev_nightly").exists()
    with pytest.raises(nt.StepError, match="refusing"):
        nt.worker_schedule(a)


def test_games_worker_uses_the_incremental_ingest(tmp_path, monkeypatch):
    from nightly_fakes import World

    w = World()
    w.play("2026-10-20")
    import src.ingest.nba_client as nc

    monkeypatch.setattr(nc, "NBAClient", lambda *a, **k: w)
    a = nt.build_parser().parse_args(["--worker", "games", "--result-file", "x", "--data-dir", str(tmp_path), "--through", "2026-10-20", "--slate", "2026-10-21", "--season", "2026-27"])
    summary, state = nt.worker_games(a)
    assert summary["new_games"] == 1 and state["games"]["last_game_date"] == "2026-10-20" and summary["tables_changed"]
