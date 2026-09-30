"""What the real-week replay of 2026-01 / 2025-12 / 2026-02 against stats.nba.com pinned (ADR 0018, 2026-09-26).

Offline: a fake stats.nba.com that has the real API's parameter format, honours DateFrom and DateTo, and can hold a game
back as not final. Also: the run-time arithmetic of the 09:30 New Zealand trigger and the ``--set-team`` helper.
"""
import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest
from nightly_fakes import SEASON, World
from test_ops_nightly import Fake

from src.ops import live_ingest as li
from src.ops import nightly as nt
from src.ops import nightly_analysis as na
from src.ops.nightly_report import render_report
from src.store import read_table

NZDT, NZST = timezone(timedelta(hours=13)), timezone(timedelta(hours=12))
NOW = datetime(2026, 10, 22, 9, 30, tzinfo=NZDT)


def run(world, base, through, **kw):
    return li.run_live_ingest(SEASON, world, base, completed_through=date.fromisoformat(through), log=lambda *_: None, **kw)


# --------------------------------------------------------------------------- the API's date format and window

def test_the_window_uses_the_apis_us_date_format_on_both_ends():
    p = li.windowed({"Season": "2025-26", "DateFrom": "", "DateTo": ""}, date(2026, 1, 8), date(2026, 1, 12))
    assert p["DateFrom"] == "01/08/2026" and p["DateTo"] == "01/12/2026"        # MM/DD/YYYY: accepted by the live endpoint 2026-09-26
    assert li.windowed({"DateFrom": "", "DateTo": ""}, None)["DateTo"] == ""


def test_date_to_bounds_a_replayed_night_and_leaves_later_rows_alone(tmp_path):
    w = World()
    w.play("2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23")
    run(w, tmp_path, "2026-10-23")                                   # a full table, as if the season were further along
    full = read_table("game_logs", tmp_path)
    w.calls.clear()
    r = run(w, tmp_path, "2026-10-21", date_to=date(2026, 10, 21))   # replaying night 2 must not delete nights 3 and 4
    assert w.calls[0][1]["DateTo"] == "10/21/2026"
    pd.testing.assert_frame_equal(full, read_table("game_logs", tmp_path))
    assert r.new_games == 0 and not any(v["changed"] for v in r.tables.values())


# --------------------------------------------------------------------------- games that are not final are never stored

@pytest.mark.parametrize("how", ["no_result", "partial_box"])
def test_a_game_that_is_not_final_is_left_out_and_repaired_next_night(tmp_path, how):
    w = World()
    w.play("2026-10-20", "2026-10-21", "2026-10-22")
    live = w.games[-1][0]
    w.unfinished[live] = how                                         # still being played (or half published) at run time
    r = run(w, tmp_path, "2026-10-22")
    assert r.pending_games == [live] and r.new_games == 2
    assert live not in set(read_table("team_games", tmp_path)["game_id"]) and live not in set(read_table("game_logs", tmp_path)["game_id"])
    assert r.last_game_date == "2026-10-21"
    w.unfinished.clear()                                             # it ends
    r = run(w, tmp_path, "2026-10-22")
    assert r.pending_games == [] and r.new_games == 1
    gl, tg = read_table("game_logs", tmp_path), read_table("team_games", tmp_path)
    assert live in set(tg["game_id"]) and len(gl[gl["game_id"] == live]) == 4
    sums = gl.groupby(["game_id", "team_id"])["pts"].sum().reset_index().merge(tg, on=["game_id", "team_id"])
    assert (sums["pts"] == sums["pts_for"]).all()                    # the invariant that defines "final"


def test_a_stored_game_is_never_removed_when_it_shows_up_unfinished_again(tmp_path):
    w = World()
    w.play("2026-10-20", "2026-10-21")
    run(w, tmp_path, "2026-10-21")
    before = read_table("game_logs", tmp_path)
    w.unfinished[w.games[-1][0]] = "partial_box"                     # a bad upstream refresh of a game we already hold
    r = run(w, tmp_path, "2026-10-21")
    assert r.pending_games == [w.games[-1][0]]
    pd.testing.assert_frame_equal(before, read_table("game_logs", tmp_path))
    assert len(read_table("team_games", tmp_path)) == 4


def test_the_games_worker_reports_pending_games_and_bounds_a_replay(tmp_path, monkeypatch):
    w = World()
    w.play("2026-10-20", "2026-10-21")
    w.unfinished[w.games[-1][0]] = "no_result"
    import src.ingest.nba_client as nc

    monkeypatch.setattr(nc, "NBAClient", lambda *a, **k: w)
    a = nt.build_parser().parse_args(["--worker", "games", "--replay", "--result-file", "x", "--data-dir", str(tmp_path), "--through", "2026-10-21",
                                      "--slate", "2026-10-22", "--season", SEASON])
    summary, _ = nt.worker_games(a)
    assert summary["pending_games"] == [w.games[-1][0]] and summary["new_games"] == 1
    assert w.calls[0][1]["DateTo"] == "10/21/2026"


# --------------------------------------------------------------------------- the trigger time, worked out

@pytest.mark.parametrize("run_at, through, slate, note", [
    # NZ morning D, 09:30 NZDT = 16:30 EDT on D-1: D-1's games have not started, D-2's ended about 13 h ago.
    (datetime(2026, 10, 20, 9, 30, tzinfo=NZDT), date(2026, 10, 18), date(2026, 10, 19), "first scheduled run: no games yet"),
    (datetime(2026, 10, 21, 9, 30, tzinfo=NZDT), date(2026, 10, 19), date(2026, 10, 20), "slate = opening night, tips 12:30 NZDT"),
    (datetime(2026, 10, 22, 9, 30, tzinfo=NZDT), date(2026, 10, 20), date(2026, 10, 21), "first run that can ingest opening night"),
    (datetime(2026, 11, 2, 9, 30, tzinfo=NZDT), date(2026, 10, 31), date(2026, 11, 1), "US clocks went back: 15:30 EST the day before"),
    (datetime(2027, 3, 15, 9, 30, tzinfo=NZDT), date(2027, 3, 13), date(2027, 3, 14), "US clocks went forward Mar 14"),
    (datetime(2027, 4, 5, 9, 30, tzinfo=NZST), date(2027, 4, 3), date(2027, 4, 4), "NZ back on standard time (UTC+12)"),
])
def test_the_09_30_nz_run_sees_the_last_completely_final_us_day_and_the_next_slate(run_at, through, slate, note):
    assert na.completed_through(run_at) == through, note
    assert na.completed_through(run_at) + timedelta(days=1) == slate


def test_opening_night_ends_more_than_12_hours_before_the_first_run_that_ingests_it():
    # opening night Oct 20 (EDT): the latest realistic finish is 03:30 ET on Oct 21 = 07:30 UTC; the run is 2026-10-22 09:30 NZDT = Oct 21 20:30 UTC
    end = datetime(2026, 10, 21, 7, 30, tzinfo=timezone.utc)
    first_run = datetime(2026, 10, 22, 9, 30, tzinfo=NZDT)
    assert (first_run - end).total_seconds() / 3600 == 13
    # and the opening-night tip-off (7:30 pm EDT = 23:30 UTC) comes 3 h after the run that reports that slate
    tip = datetime(2026, 10, 20, 23, 30, tzinfo=timezone.utc)
    assert (tip - datetime(2026, 10, 21, 9, 30, tzinfo=NZDT)).total_seconds() / 3600 == 3


# --------------------------------------------------------------------------- your team

def league_file(tmp_path, *, drafted=True):
    teams = [{"team_id": 1, "name": "Alpha", "abbrev": "ALP", "owner_ids": ["{A}"], "roster": [{"espn_player_id": 1}] if drafted else []},
             {"team_id": 3, "name": "Gamma", "abbrev": "GAM", "owner_ids": ["{G}"], "roster": [{"espn_player_id": 2}] if drafted else []},
             {"team_id": 15, "name": "Team 15", "abbrev": "TM15", "owner_ids": [], "roster": []}]
    path = tmp_path / "data" / "processed" / "espn_league" / "1_2027.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fetched_at": "2026-10-19T00:00:00+00:00", "settings": {"name": "L"}, "draft_completed": drafted, "teams": teams,
                                "members": [{"id": "{A}", "display_name": "ann"}, {"id": "{G}", "display_name": "gus"}]}), encoding="utf-8")


def team_settings(tmp_path, *extra):
    args = nt.build_parser().parse_args(["--data-dir", str(tmp_path / "data"), "--reports-dir", str(tmp_path / "rep"), "--repo-root", str(tmp_path),
                                         "--league-id", "1", *extra])
    return nt.resolve_settings(args, {}, NOW)


def test_set_team_lists_teams_with_owners_and_saves_a_validated_id(tmp_path, capsys):
    league_file(tmp_path)
    s = team_settings(tmp_path)
    assert nt.set_team_command(s, "list") == 0
    listing = capsys.readouterr().out
    assert "Gamma" in listing and "gus" in listing and "(unowned)" in listing and "currently configured: nothing" in listing
    (tmp_path / "data" / "nightly.json").write_text(json.dumps({"window_end": "2027-04-06"}), encoding="utf-8")
    assert nt.set_team_command(s, "3") == 0
    saved = json.loads((tmp_path / "data" / "nightly.json").read_text(encoding="utf-8"))
    assert saved == {"window_end": "2027-04-06", "team_id": 3}                  # other settings survive
    assert team_settings(tmp_path).team_id == 3
    assert "Gamma" in capsys.readouterr().out


def test_set_team_rejects_unknown_ids_and_missing_league_data(tmp_path, capsys):
    s = team_settings(tmp_path)
    assert nt.set_team_command(s, "3") == nt.EXIT_CONFIG and "no synced league data" in capsys.readouterr().out
    league_file(tmp_path)
    assert nt.set_team_command(s, "99") == nt.EXIT_CONFIG and "not in league 1" in capsys.readouterr().out
    assert nt.set_team_command(s, "abc") == nt.EXIT_CONFIG
    assert not (tmp_path / "data" / "nightly.json").exists()
    assert nt.main(["--data-dir", str(tmp_path / "data"), "--league-id", "1", "--repo-root", str(tmp_path), "--set-team"]) == 0     # the CLI flag lists


def _minimal_run(**kw):
    return {"date": "2026-10-22", "outcome": "ok", "run_id": "r", "season": SEASON, "completed_through": "2026-10-20", "slate": "2026-10-21",
            "window": ["2026-10-20", "2027-04-06"], "days_left": 100, "steps": [], "diffs": {}, "analysis": {}, "freshness": {}, **kw}


def test_team_setup_problem_is_named_and_loud(tmp_path):
    league_file(tmp_path)
    problem = nt.team_setup_problem(team_settings(tmp_path))
    assert "NOT SET" in problem and "--set-team" in problem
    assert "not a team of league 1" in nt.team_setup_problem(team_settings(tmp_path, "--team-id", "7"))
    assert nt.team_setup_problem(team_settings(tmp_path, "--team-id", "3")) is None
    assert nt.team_setup_problem(team_settings(tmp_path, "--mock-team", "2")) is None
    lines = render_report(_minimal_run(team_setup=problem)).splitlines()
    assert lines[2].startswith("**ACTION NEEDED: YOUR TEAM IS NOT SET")                   # right under the title, not buried
    assert "ACTION NEEDED" not in render_report(_minimal_run(team_setup=None))


def test_an_unset_team_after_the_draft_alerts_once_but_not_before_it(tmp_path):
    toasts = []
    league_file(tmp_path, drafted=False)
    s = team_settings(tmp_path, "--as-of", "")                                           # a real (non-replay) run
    s.as_of = None
    s.stale_hours = 10_000.0
    assert nt.execute(s, Fake(tmp_path), now=NOW, through=date(2026, 10, 20), slate=date(2026, 10, 21), notifier=lambda t, m: toasts.append(m)) == 0
    assert not (tmp_path / "rep" / "ALERT.txt").exists() and toasts == []               # nobody has a roster to be missing yet
    league_file(tmp_path, drafted=True)
    assert nt.execute(s, Fake(tmp_path), now=NOW + timedelta(days=1), through=date(2026, 10, 21), slate=date(2026, 10, 22),
                      notifier=lambda t, m: toasts.append(m)) == 0
    assert "YOUR TEAM IS NOT SET" in (tmp_path / "rep" / "ALERT.txt").read_text()
    assert "ACTION NEEDED" in (tmp_path / "rep" / "latest.md").read_text() and len(toasts) == 1
    assert nt.execute(s, Fake(tmp_path), now=NOW + timedelta(days=2), through=date(2026, 10, 22), slate=date(2026, 10, 23),
                      notifier=lambda t, m: toasts.append(m)) == 0
    assert len(toasts) == 1                                                             # the same reason does not toast every morning
    s.team_id = 3
    assert nt.execute(s, Fake(tmp_path), now=NOW + timedelta(days=3), through=date(2026, 10, 23), slate=date(2026, 10, 24), notifier=lambda *a: None) == 0
    assert not (tmp_path / "rep" / "ALERT.txt").exists()


# --------------------------------------------------------------------------- lock contention at the handoff, catch-up, cache pruning

def test_a_busy_lock_is_waited_for_and_the_night_still_runs(tmp_path):
    from src.ops.runlock import RunLock

    s = team_settings(tmp_path, "--team-id", "3", "--lock-wait-minutes", "20")
    assert s.lock_wait_s == 1200
    s.stale_hours = 10_000.0
    held = RunLock(tmp_path / "data" / "daily_refresh" / "lock.json").acquire()        # the pre-draft refresh is finishing
    naps = []
    clock = {"t": 0.0}

    def sleep(sec):
        naps.append(sec)
        clock["t"] += sec
        if len(naps) == 3:
            held.release()                                                             # the daily job ends after ~90 s

    f = Fake(tmp_path)
    assert nt.execute(s, f, now=NOW, through=date(2026, 10, 20), slate=date(2026, 10, 21), notifier=lambda *a: None,
                      clock=lambda: clock["t"], sleep=sleep) == 0
    assert len(naps) == 3 and f.calls == list(nt.STEP_ORDER)
    # ... and gives up (exit 3, nothing run) only after the whole wait
    held = RunLock(tmp_path / "data" / "daily_refresh" / "lock.json").acquire()
    clock["t"], naps[:] = 0.0, []
    f2 = Fake(tmp_path)
    assert nt.execute(s, f2, now=NOW, through=date(2026, 10, 20), slate=date(2026, 10, 21), notifier=lambda *a: None,
                      clock=lambda: clock["t"], sleep=lambda sec: clock.__setitem__("t", clock["t"] + sec)) == nt.EXIT_BUSY
    assert f2.calls == [] and clock["t"] >= 1200
    held.release()


def test_two_missed_nights_are_recovered_by_one_catch_up_run(tmp_path):
    w = World()
    w.play("2026-10-20", "2026-10-21", "2026-10-22", "2026-10-23")
    run(w, tmp_path, "2026-10-21", date_to=date(2026, 10, 21))       # the last night that ran covered through Oct 21
    r = run(w, tmp_path, "2026-10-23")                               # the machine slept; one StartWhenAvailable run two nights later
    assert r.new_games == 2 and r.last_game_date == "2026-10-23" and r.date_from == "2026-10-18"
    assert read_table("team_games", tmp_path)["game_id"].nunique() == 4


def test_old_date_window_cache_files_are_pruned_but_full_season_pulls_are_kept(tmp_path):
    import os
    import time

    from types import SimpleNamespace

    d = tmp_path / "playergamelogs"
    d.mkdir()
    old = d / "01-08-2026_01-12-2026_00_Base__aaaa.json"
    new = d / "01-09-2026_01-13-2026_00_Base__bbbb.json"
    full = d / "00_Base_0_Totals_2025-26_Regular-Season__cccc.json"
    for f in (old, new, full):
        f.write_text("{}")
    long_ago = time.time() - 40 * 86400
    for f in (old, full):
        os.utime(f, (long_ago, long_ago))
    assert li.prune_window_cache(SimpleNamespace(cache_dir=tmp_path)) == 1
    assert not old.exists() and new.exists() and full.exists()
