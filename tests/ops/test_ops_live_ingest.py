"""Incremental live ingest: window pulls, idempotency, healing, refusal to shrink, polite request counts. Offline, fake NBA."""
from datetime import date

import pandas as pd
import pytest
from nightly_fakes import SEASON, World

from src.contracts import HISTORY_TABLES, validate_table
from src.ops import live_ingest as li
from src.store import read_table


def run(world, base, through, **kw):
    return li.run_live_ingest(SEASON, world, base, completed_through=date.fromisoformat(through), log=lambda *_: None, **kw)


def snapshot_tables(base):
    return {n: read_table(n, base) for n in HISTORY_TABLES}


def test_first_night_pulls_the_season_and_builds_all_tables(tmp_path):
    w = World()
    w.play("2026-10-20", "2026-10-21")
    r = run(w, tmp_path, "2026-10-21")
    assert r.new_games == 2 and r.date_from is None and r.last_game_date == "2026-10-21" and r.new_players == 4
    assert r.network_requests == 8 and {c[0] for c in w.calls} == {"playergamelogs", "leaguegamelog", "playerindex", "leaguedashplayerbiostats", "commonplayerinfo"}
    assert w.calls_to("commonplayerinfo") == 4                   # one per new player: exact birthdates, as the full ingest has them
    t = snapshot_tables(tmp_path)
    assert len(t["game_logs"]) == 8 and len(t["team_games"]) == 4 and len(t["players"]) == 4 and len(t["player_season_bio"]) == 4
    for name, df in t.items():
        validate_table(df, name)


def test_second_identical_run_is_a_no_op(tmp_path):
    w = World()
    w.play("2026-10-20", "2026-10-21")
    run(w, tmp_path, "2026-10-21")
    before = snapshot_tables(tmp_path)
    mtimes = {n: (tmp_path / "processed" / f"{n}.parquet").stat().st_mtime_ns for n in HISTORY_TABLES}
    w.calls.clear()
    r = run(w, tmp_path, "2026-10-21")
    assert r.new_games == 0 and r.new_player_rows == 0 and r.new_players == 0
    assert not any(v["changed"] for v in r.tables.values())
    assert r.date_from == "2026-10-18"                           # newest stored game minus three days of overlap
    assert w.calls_to("playerindex") == 0 and w.calls_to("leaguedashplayerbiostats") == 0    # nobody new: no bio pulls
    after = snapshot_tables(tmp_path)
    for n in HISTORY_TABLES:
        pd.testing.assert_frame_equal(before[n], after[n])
        assert (tmp_path / "processed" / f"{n}.parquet").stat().st_mtime_ns == mtimes[n]        # not even rewritten


def test_next_night_is_incremental_and_leaves_older_rows_untouched(tmp_path):
    w = World()
    w.play("2026-10-20", "2026-10-21", "2026-10-22")
    run(w, tmp_path, "2026-10-22")
    old = snapshot_tables(tmp_path)
    w.play("2026-10-23")
    w.calls.clear()
    r = run(w, tmp_path, "2026-10-23")
    assert r.new_games == 1 and r.new_player_rows == 4 and r.network_requests == 2
    sent = [p for e, p in w.calls if e == "playergamelogs"][0]
    assert sent["DateFrom"] == "10/19/2026"
    new = snapshot_tables(tmp_path)
    assert len(new["game_logs"]) == 16 and new["team_games"]["game_id"].nunique() == 4
    cut = pd.Timestamp("2026-10-23")
    pd.testing.assert_frame_equal(old["game_logs"], new["game_logs"][new["game_logs"]["game_date"] < cut].reset_index(drop=True))


def test_overlap_repairs_a_corrected_box_score_and_a_rolled_back_run(tmp_path):
    w = World()
    w.play("2026-10-20", "2026-10-21", "2026-10-22")
    run(w, tmp_path, "2026-10-22")
    gid = w.games[-1][0]
    w.correct(gid)                                               # upstream fixes the newest box score
    run(w, tmp_path, "2026-10-22")
    gl = read_table("game_logs", tmp_path)
    assert gl[gl["game_id"] == gid]["fgm"].tolist() == [4, 4, 4, 4] or gl[gl["game_id"] == gid]["fgm"].nunique() == 1
    truth = w._rows(None)[0]
    assert int(gl["pts"].sum()) == sum(r[29] for r in truth)      # tables equal the source after the correction


def test_a_new_player_triggers_the_player_and_bio_refresh_once(tmp_path):
    w = World()
    w.play("2026-10-20")
    run(w, tmp_path, "2026-10-20")
    w.play("2026-10-21")
    w.rookie_from = date(2026, 10, 21)
    w.calls.clear()
    r = run(w, tmp_path, "2026-10-21")
    assert r.new_players == 1 and w.calls_to("playerindex") == 1 and w.calls_to("leaguedashplayerbiostats") == 1
    assert 23 in set(read_table("players", tmp_path)["player_id"]) and 23 in set(read_table("player_season_bio", tmp_path)["player_id"])
    assert w.calls_to("commonplayerinfo") == 1                   # only the rookie, not the four known players
    assert read_table("players", tmp_path).set_index("player_id").loc[23, "birthdate"] == pd.Timestamp("1999-05-17")
    w.calls.clear()
    assert run(w, tmp_path, "2026-10-21").new_players == 0 and w.calls_to("playerindex") == 0


def test_before_opening_night_makes_no_request_and_an_empty_source_is_not_an_error(tmp_path):
    w = World()
    r = run(w, tmp_path, "2026-10-19", opening_night=date(2026, 10, 20))
    assert r.skipped and "opens 2026-10-20" in r.skipped and w.calls == [] and r.network_requests == 0
    r = run(w, tmp_path, "2026-10-21", opening_night=date(2026, 10, 20))     # the day has passed but the source is still empty
    assert r.skipped and "no regular-season games" in r.skipped and not (tmp_path / "processed" / "game_logs.parquet").exists()


def test_refuses_to_shrink_tables_and_keeps_a_rolling_backup(tmp_path):
    w = World()
    w.play(*[f"2026-10-{d}" for d in range(20, 29)])
    run(w, tmp_path, "2026-10-28")
    before = read_table("game_logs", tmp_path)
    w.truncate_window = True                                      # a partial upstream response
    with pytest.raises(li.LiveIngestError, match="refusing to shrink"):
        run(w, tmp_path, "2026-10-28")
    pd.testing.assert_frame_equal(before, read_table("game_logs", tmp_path))
    w.truncate_window = False
    w.play("2026-10-29")
    run(w, tmp_path, "2026-10-29")
    assert (tmp_path / "processed" / f"game_logs.parquet{li.BACKUP_SUFFIX}").exists()
    prev = pd.read_parquet(tmp_path / "processed" / f"game_logs.parquet{li.BACKUP_SUFFIX}")
    assert len(prev) == len(before) and len(read_table("game_logs", tmp_path)) == len(before) + 4


def test_outage_raises_and_leaves_tables_alone(tmp_path):
    w = World()
    w.play("2026-10-20")
    run(w, tmp_path, "2026-10-20")
    before = snapshot_tables(tmp_path)
    w.play("2026-10-21")
    w.outage = True
    with pytest.raises(Exception, match="down"):
        run(w, tmp_path, "2026-10-21")
    for n, df in before.items():
        pd.testing.assert_frame_equal(df, read_table(n, tmp_path))


def test_other_seasons_are_never_touched(tmp_path):
    w = World("2025-26")
    w.play("2026-03-01", "2026-03-02")
    li.run_live_ingest("2025-26", w, tmp_path, completed_through=date(2026, 3, 2), log=lambda *_: None)
    old = snapshot_tables(tmp_path)
    w2 = World(SEASON)
    w2.play("2026-10-20")
    run(w2, tmp_path, "2026-10-20")
    now = snapshot_tables(tmp_path)
    keep = now["game_logs"][now["game_logs"]["season"] == "2025-26"].reset_index(drop=True)
    pd.testing.assert_frame_equal(old["game_logs"], keep)
    assert set(now["team_games"]["season"]) == {"2025-26", SEASON}


def test_orphan_game_is_held_pending_while_the_final_games_are_stored(tmp_path):
    w = World()
    w.play("2026-10-20", "2026-10-21", "2026-10-21")
    orphan = w.games[-1][0]
    w.unfinished[orphan] = "no_team_rows"                        # player rows are out, the team endpoint has not caught up
    r = run(w, tmp_path, "2026-10-21")
    assert r.pending_games == [orphan] and r.new_games == 2
    assert orphan not in set(read_table("team_games", tmp_path)["game_id"]) | set(read_table("game_logs", tmp_path)["game_id"])
    w.unfinished.clear()
    r = run(w, tmp_path, "2026-10-21")
    assert r.pending_games == [] and r.new_games == 1 and len(read_table("team_games", tmp_path)) == 6


def test_a_stored_game_is_kept_when_its_team_rows_go_missing_again(tmp_path):
    w = World()
    w.play("2026-10-20", "2026-10-21")
    run(w, tmp_path, "2026-10-21")
    before = snapshot_tables(tmp_path)
    w.unfinished[w.games[-1][0]] = "no_team_rows"
    r = run(w, tmp_path, "2026-10-21")
    assert r.pending_games == [w.games[-1][0]]
    for n in ("game_logs", "team_games"):
        pd.testing.assert_frame_equal(before[n], read_table(n, tmp_path))


def test_all_games_orphaned_means_a_broken_team_endpoint_and_raises(tmp_path):
    w = World()
    w.play("2026-10-20", "2026-10-21")
    run(w, tmp_path, "2026-10-20")
    before = snapshot_tables(tmp_path)
    for gid, _ in w.games:
        w.unfinished[gid] = "no_team_rows"
    with pytest.raises(li.LiveIngestError, match="no team rows"):
        run(w, tmp_path, "2026-10-21")
    for n, df in before.items():
        pd.testing.assert_frame_equal(df, read_table(n, tmp_path))


def test_a_vanished_stored_game_is_never_deleted_and_the_guard_message_names_it(tmp_path):
    w = World()
    w.play(*[f"2026-10-{d}" for d in range(20, 29)])
    run(w, tmp_path, "2026-10-28")
    before = snapshot_tables(tmp_path)
    gone = w.games[-1][0]
    w.vanished.add(gone)                                          # small window: one game is >20% of it, the guard trips
    with pytest.raises(li.LiveIngestError, match="refusing to shrink") as exc:
        run(w, tmp_path, "2026-10-28")
    assert gone in str(exc.value) and "never deleted" in str(exc.value)
    for n, df in before.items():
        pd.testing.assert_frame_equal(df, read_table(n, tmp_path))


def test_a_vanished_stored_game_in_a_busy_window_is_kept_as_stored(tmp_path):
    w = World()
    w.play("2026-10-20", *["2026-10-28"] * 20)                    # a 20-game window: one vanishing game is within tolerance
    run(w, tmp_path, "2026-10-28")
    before = snapshot_tables(tmp_path)
    gone = w.games[-1][0]
    w.vanished.add(gone)
    r = run(w, tmp_path, "2026-10-28")
    assert r.vanished_games == [gone]
    for n, df in before.items():
        pd.testing.assert_frame_equal(df, read_table(n, tmp_path))
