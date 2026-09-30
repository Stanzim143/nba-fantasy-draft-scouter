"""nba_offseason: Summer League / preseason transforms, data-quality rules, storage, refresh policy, CLI.

No network: a routed fake client serves payloads built with the repo's real-shape builders.
"""
import json
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest
from ingest_fakes import BOS, NYK, league_p_payload, player_payload, player_row, team_payload, team_row, wrap

from src.ingest import nba_offseason as no
from src.ingest.nba_transform import TransformError

SL, PRE = no.SUMMER_LEAGUE, no.PRESEASON
EV = "2018-19"          # July 2018 Summer League and October 2018 preseason
SL_ID, PRE_ID = "15218", "00118"


# --------------------------------------------------------------------------- builders

def prow(pid=11, name="Alpha One", team=BOS, abbr="BOS", gid=SL_ID + "00001", d="2018-07-06", minutes=30.4,
         matchup="BOS vs. NYK", **kw):
    return player_row(player_id=pid, name=name, team_id=team, abbr=abbr, game_id=gid, date=d + "T00:00:00",
                      matchup=matchup, minutes=minutes, season=kw.pop("season", "2018"), **kw)


def payloads(rows, teams=None):
    """(playergamelogs, leaguegamelog P, leaguegamelog T) payloads for ``rows``; teams derived if not given."""
    if teams is None:
        seen, teams = set(), []
        for r in rows:
            key = (r[7], r[4])
            if key not in seen:
                seen.add(key)
                teams.append(team_row(team_id=r[4], abbr=r[5], game_id=r[7], date=r[8][:10], matchup=r[9], pts=90))
    return player_payload(rows), league_p_payload(rows), team_payload(teams)


class FakeClient:
    """Serves payloads by (endpoint, context, player-or-team); records every call including ``refresh``."""

    def __init__(self, table, offline=False):
        self.table = table
        self.offline = offline
        self.calls: list[tuple] = []
        self.stats = SimpleNamespace(network_requests=0, cache_hits=0)
        self.min_interval = 0.0

    @staticmethod
    def kind(endpoint, params):
        ctx = SL if params.get("LeagueID") == no.SUMMER_LEAGUE_ID else PRE
        if endpoint == "playergamelogs":
            return (params["Season"], ctx, "fractional")
        return (params["Season"], ctx, params["PlayerOrTeam"])

    def get(self, endpoint, params, *, refresh=False):
        self.calls.append((endpoint, self.kind(endpoint, params), refresh))
        self.stats.network_requests += 1
        key = self.kind(endpoint, params)
        if key not in self.table:
            raise AssertionError(f"unexpected request {key}")
        return self.table[key]


def served(event=EV, ctx=SL, rows=None, teams=None):
    pgl, llg, tm = payloads(rows if rows is not None else [prow()], teams)
    return {(event, ctx, "fractional"): pgl, (event, ctx, "P"): llg, (event, ctx, "T"): tm}


def empty_event(event, ctx):
    return {(event, ctx, "fractional"): player_payload([]), (event, ctx, "P"): league_p_payload([]),
            (event, ctx, "T"): team_payload([])}


# --------------------------------------------------------------------------- labels and parameters

def test_previous_season_tags_the_season_an_event_follows():
    assert no.previous_season("2026-27") == "2025-26"
    assert no.previous_season("2015-16") == "2014-15"


def test_event_windows_are_calendar_bounded():
    lo, hi = no.event_window(SL, "2026-27")
    assert (lo, hi) == (pd.Timestamp(2026, 6, 1), pd.Timestamp(2026, 9, 30))
    lo, hi = no.event_window(PRE, "2019-20")
    assert (lo, hi) == (pd.Timestamp(2019, 9, 1), pd.Timestamp(2019, 12, 31))
    with pytest.raises(ValueError):
        no.event_window("bogus", "2019-20")


def test_live_season_start_flips_in_june():
    assert no.live_season_start(date(2026, 9, 24)) == 2026
    assert no.live_season_start(date(2027, 1, 5)) == 2026
    assert no.live_season_start(date(2027, 5, 31)) == 2026
    assert no.live_season_start(date(2027, 6, 1)) == 2027


def test_endpoint_parameters_select_the_context():
    p = no.player_logs_params("2025-26", SL)
    assert p["LeagueID"] == "15" and p["PlayerOrTeam"] == "P" and p["SeasonType"] == "Regular Season"
    p = no.player_logs_params("2025-26", PRE)
    assert p["LeagueID"] == "00" and p["SeasonType"] == "Pre Season"
    assert no.team_logs_params("2025-26", SL)["PlayerOrTeam"] == "T"
    frac = no.fractional_minutes_params("2025-26", PRE)
    assert "PlayerOrTeam" not in frac and frac["SeasonType"] == "Pre Season" and frac["Season"] == "2025-26"
    with pytest.raises(ValueError):
        no.player_logs_params("2025-26", "bogus")


# --------------------------------------------------------------------------- player-log transform

def test_transform_tags_seasons_and_returns_a_validated_table():
    rows = [prow(pid=11, gid=SL_ID + "00001"), prow(pid=12, name="Bravo Two", gid=SL_ID + "00001", minutes=22.0)]
    _, llg, _ = payloads(rows)
    out, rep = no.transform_offseason_logs(llg, EV, SL)
    assert len(out) == 2 and rep.rows_in == 2 and rep.rows_out == 2 and not rep.dropped
    assert set(out["season"]) == {"2017-18"} and set(out["event_season"]) == {EV} and set(out["context"]) == {SL}
    assert out["player_id"].dtype == "int64" and out["min"].dtype == "float64"
    no.validate_offseason_logs(out)


def test_rows_outside_the_event_window_are_dropped_and_counted():
    """The 2019-20 preseason pull contained July 2020 bubble games: they are after that season began."""
    ev = "2019-20"
    rows = [prow(gid="0011900001", d="2019-10-05", season="2019-20"),
            prow(gid="0011900099", d="2020-07-28", season="2019-20", matchup="BOS @ NYK")]
    _, llg, _ = payloads(rows)
    out, rep = no.transform_offseason_logs(llg, ev, PRE)
    assert len(out) == 1 and rep.dropped == {"outside_event_window": 1}
    assert out["game_date"].max() < pd.Timestamp(2019, 12, 31)


def test_unexpected_game_ids_are_dropped():
    rows = [prow(gid=SL_ID + "00001"), prow(gid="0021800001", d="2018-07-07")]   # a regular-season id
    _, llg, _ = payloads(rows)
    out, rep = no.transform_offseason_logs(llg, EV, SL)
    assert len(out) == 1 and rep.dropped == {"unexpected_game_id": 1}


def test_zero_minute_and_incomplete_rows_are_dropped():
    rows = [prow(pid=11), prow(pid=12, minutes=0, gid=SL_ID + "00002"), prow(pid=13, gid=SL_ID + "00003")]
    _, llg, _ = payloads(rows)
    llg["resultSets"][0]["rowSet"][2][llg["resultSets"][0]["headers"].index("FGM")] = None
    out, rep = no.transform_offseason_logs(llg, EV, SL)
    assert list(out["player_id"]) == [11]
    assert rep.dropped == {"no_minutes_played": 1, "missing_box_score_fields": 1}


def test_impossible_box_scores_are_dropped():
    good = prow(pid=11)
    made_gt_att = prow(pid=12, gid=SL_ID + "00002", fgm=9, fga=8, fg3m=0, fg3a=0, ftm=0, fta=0, pts=18)
    below = prow(pid=13, gid=SL_ID + "00003", pts=5)        # fewer points than the made shots explain
    _, llg, _ = payloads([good, made_gt_att, below])
    out, rep = no.transform_offseason_logs(llg, EV, SL)
    assert list(out["player_id"]) == [11]
    assert rep.dropped == {"made_exceeds_attempted": 1, "points_below_made_shots": 1}


def test_a_row_breaking_two_rules_is_counted_once():
    both = prow(pid=12, gid=SL_ID + "00002", fgm=9, fga=8, fg3m=0, fg3a=0, ftm=0, fta=0, pts=5)   # made > attempted AND pts too low
    _, llg, _ = payloads([prow(pid=11), both])
    out, rep = no.transform_offseason_logs(llg, EV, SL)
    assert list(out["player_id"]) == [11]
    assert sum(rep.dropped.values()) == 1 and rep.rows_in - rep.rows_out == 1


def test_points_beyond_recorded_made_shots_are_kept_and_measured():
    """July 2026: official points exceed 2*fgm + fg3m + ftm (free throws missing from the FT counters).
    Player sums still match team totals, so the rows are kept and the excess is reported, not hidden."""
    normal = prow(pid=11)
    excess = prow(pid=12, gid=SL_ID + "00002", fgm=5, fga=10, fg3m=1, fg3a=3, ftm=1, fta=1, pts=2 * 5 + 1 + 1 + 3)
    _, llg, _ = payloads([normal, excess])
    out, rep = no.transform_offseason_logs(llg, EV, SL)
    assert len(out) == 2 and not rep.dropped
    assert rep.info["rows_with_unrecorded_points"] == 1 and rep.info["unrecorded_points_total"] == 3
    assert int(no.unrecorded_points(out).sum()) == 3


def test_a_blank_name_becomes_a_placeholder_and_is_counted():
    _, llg, _ = payloads([prow(pid=77)])
    llg["resultSets"][0]["rowSet"][0][llg["resultSets"][0]["headers"].index("PLAYER_NAME")] = None
    out, rep = no.transform_offseason_logs(llg, EV, SL)
    assert out["player_name"].iloc[0] == "player 77" and rep.info["names_unresolved"] == 1


def test_duplicate_game_player_rows_keep_the_one_with_most_minutes():
    a, b = prow(minutes=10.0), prow(minutes=30.0)
    _, llg, _ = payloads([a, b])
    out, rep = no.transform_offseason_logs(llg, EV, SL)
    assert len(out) == 1 and out["min"].iloc[0] == 30.0 and rep.dropped == {"duplicate_row_conflicting": 1}


def test_fractional_minutes_replace_integers_only_for_exact_matches():
    rows = [prow(pid=11, minutes=30.4), prow(pid=12, minutes=25.6, gid=SL_ID + "00002"),
            prow(pid=13, minutes=20.0, gid=SL_ID + "00003")]
    pgl, llg, _ = payloads(rows)
    # pid 12: placeholder-id world -- fractional payload knows a *different* id, so nothing matches
    pgl["resultSets"][0]["rowSet"][1][pgl["resultSets"][0]["headers"].index("PLAYER_ID")] = 99999
    # pid 13: fractional minutes wildly off the integer ones -> out of tolerance, keep the integer
    pgl["resultSets"][0]["rowSet"][2][pgl["resultSets"][0]["headers"].index("MIN")] = 35.0
    out, rep = no.transform_offseason_logs(llg, EV, SL, fractional_payload=pgl)
    by = out.set_index("player_id")["min"]
    assert by[11] == pytest.approx(30.4, abs=0.01)      # refined to the fractional value
    assert by[12] == 26.0                                # integer (25.6 rounded) kept: no id match
    assert by[13] == 20.0                                # integer kept: out of tolerance
    info = rep.info
    assert info["fractional_minutes_used"] == 1 and info["integer_minutes_only"] == 2
    assert info["fractional_rows_without_real_id"] == 1 and info["fractional_minutes_out_of_tolerance"] == 1


def test_empty_events_return_typed_empty_tables():
    out, rep = no.transform_offseason_logs(league_p_payload([]), EV, SL)
    assert out.empty and list(out.columns) == list(no.OFFSEASON_LOGS_COLUMNS) and rep.rows_in == 0
    no.validate_offseason_logs(out)
    tm, _ = no.transform_offseason_team_games(team_payload([]), EV, SL)
    assert tm.empty and list(tm.columns) == list(no.OFFSEASON_TEAM_GAMES_COLUMNS)
    no.validate_offseason_team_games(tm)


def test_malformed_payloads_and_seasons_raise():
    with pytest.raises(TransformError):
        no.transform_offseason_logs(wrap("X", ["A"], [[1]]), EV, SL)
    with pytest.raises(ValueError):
        no.transform_offseason_logs(league_p_payload([]), "2018", SL)


# --------------------------------------------------------------------------- team games

def test_team_games_carry_home_flag_and_opponent_points():
    teams = [team_row(BOS, "BOS", SL_ID + "00001", "2018-07-06", "BOS vs. NYK", pts=100),
             team_row(NYK, "NYK", SL_ID + "00001", "2018-07-06", "NYK @ BOS", pts=90),
             team_row(BOS, "BOS", SL_ID + "00002", "2018-07-07", "BOS @ NYK", pts=80)]   # opponent row missing
    out, rep = no.transform_offseason_team_games(team_payload(teams), EV, SL)
    g1 = out[out["game_id"] == SL_ID + "00001"].set_index("team_id")
    assert bool(g1.at[BOS, "is_home"]) and not bool(g1.at[NYK, "is_home"])
    assert g1.at[BOS, "pts_for"] == 100 and g1.at[BOS, "pts_against"] == 90
    lone = out[out["game_id"] == SL_ID + "00002"].iloc[0]
    assert lone["pts_for"] == 80 and pd.isna(lone["pts_against"])
    assert set(out["season"]) == {"2017-18"} and set(out["event_season"]) == {EV}


# --------------------------------------------------------------------------- validation

def test_validators_reject_duplicates_nulls_and_bad_dtypes():
    _, llg, _ = payloads([prow()])
    out, _ = no.transform_offseason_logs(llg, EV, SL)
    with pytest.raises(no.OffseasonError, match="duplicate"):
        no.validate_offseason_logs(pd.concat([out, out], ignore_index=True))
    bad = out.copy()
    bad["team_abbr"] = None
    with pytest.raises(no.OffseasonError, match="nulls"):
        no.validate_offseason_logs(bad)
    bad = out.copy()
    bad["min"] = bad["min"].astype(str)
    with pytest.raises(no.OffseasonError, match="dtype"):
        no.validate_offseason_logs(bad)
    bad = out.copy()
    bad["season"] = "2018-19"           # must be the season BEFORE event_season
    with pytest.raises(no.OffseasonError, match="season before"):
        no.validate_offseason_logs(bad)
    bad = out.copy()
    bad["context"] = "regular_season"
    with pytest.raises(no.OffseasonError, match="context"):
        no.validate_offseason_logs(bad)


# --------------------------------------------------------------------------- pipeline

def _two_event_table():
    t = {}
    t.update(served("2018-19", SL, [prow(pid=11), prow(pid=12, name="Bravo Two", gid=SL_ID + "00002")]))
    t.update(served("2018-19", PRE, [prow(pid=11, gid=PRE_ID + "00001", d="2018-10-01", minutes=18.0)]))
    return t


def run(base, table, seasons=("2018-19",), **kw):
    client = FakeClient(table, offline=kw.pop("offline", False))
    result = no.run_ingest(list(seasons), client, base, log=lambda *_: None, live_start=kw.pop("live_start", 9999), **kw)
    return result, client


def test_pipeline_writes_tables_and_a_report(tmp_path):
    result, _ = run(tmp_path, _two_event_table())
    logs, teams = no.read_offseason_logs(tmp_path), no.read_offseason_team_games(tmp_path)
    assert len(logs) == 3 and set(logs["context"]) == {SL, PRE}
    assert len(teams) == 3 and set(teams["context"]) == {SL, PRE}   # two Summer League games, one preseason game
    assert result.tables["offseason_logs"]["changed"] is True
    report = json.loads(no.report_path(tmp_path).read_text(encoding="utf-8"))
    assert set(report["events"]) == {"2018-19/summer_league", "2018-19/preseason"}
    assert report["events"]["2018-19/summer_league"]["n_players"] == 2


def test_rerunning_the_same_ingest_changes_nothing(tmp_path):
    run(tmp_path, _two_event_table())
    before = {p.name: p.read_bytes() for p in (tmp_path / "processed").iterdir()}
    result, _ = run(tmp_path, _two_event_table())
    assert result.tables["offseason_logs"]["changed"] is False and result.tables["offseason_team_games"]["changed"] is False
    assert {p.name: p.read_bytes() for p in (tmp_path / "processed").iterdir()} == before


def test_ingesting_one_event_keeps_the_others(tmp_path):
    table = _two_event_table()
    table.update(served("2019-20", SL, [prow(pid=21, gid="1521900001", d="2019-07-05", season="2019")]))
    table.update(empty_event("2019-20", PRE))
    run(tmp_path, table, seasons=("2018-19", "2019-20"))
    run(tmp_path, table, seasons=("2019-20",), contexts=(SL,))
    logs = no.read_offseason_logs(tmp_path)
    assert set(zip(logs["event_season"], logs["context"])) == {("2018-19", SL), ("2018-19", PRE), ("2019-20", SL)}


def test_replacing_an_event_replaces_only_that_event(tmp_path):
    table = _two_event_table()
    run(tmp_path, table)
    table.update(served("2018-19", PRE, [prow(pid=11, gid=PRE_ID + "00001", d="2018-10-01", minutes=18.0),
                                          prow(pid=14, name="Delta", gid=PRE_ID + "00001", d="2018-10-01", minutes=9.0)]))
    run(tmp_path, table)
    logs = no.read_offseason_logs(tmp_path)
    assert (logs["context"] == PRE).sum() == 2 and (logs["context"] == SL).sum() == 2


def test_only_the_live_season_is_refreshed(tmp_path):
    table = _two_event_table()
    table.update(served("2019-20", SL, [prow(pid=21, gid="1521900001", d="2019-07-05", season="2019")]))
    table.update(empty_event("2019-20", PRE))
    _, client = run(tmp_path, table, seasons=("2018-19", "2019-20"), live_start=2019)
    flags = {(k[0], k[1]): r for _, k, r in client.calls}
    assert flags[("2018-19", SL)] is False and flags[("2018-19", PRE)] is False
    assert flags[("2019-20", SL)] is True and flags[("2019-20", PRE)] is True
    _, client = run(tmp_path, table, seasons=("2018-19",), refresh=True)
    assert all(r for _, _, r in client.calls)


def test_offline_mode_never_asks_for_a_refresh(tmp_path):
    _, client = run(tmp_path, _two_event_table(), live_start=2018, offline=True)
    assert not any(r for _, _, r in client.calls)


def test_player_rows_for_teams_without_a_team_game_are_dropped(tmp_path):
    """An exhibition against a non-NBA club has player rows but no team row for the foreign side."""
    rows = [prow(pid=11), prow(pid=12, team=1234, abbr="XXX", name="Foreign", matchup="XXX @ BOS")]
    pgl, llg, _ = payloads(rows)
    team = team_payload([team_row(BOS, "BOS", SL_ID + "00001", "2018-07-06", "BOS vs. XXX", pts=90)])
    table = {("2018-19", SL, "fractional"): pgl, ("2018-19", SL, "P"): llg, ("2018-19", SL, "T"): team,
             **empty_event("2018-19", PRE)}
    result, _ = run(tmp_path, table)
    assert list(no.read_offseason_logs(tmp_path)["player_id"]) == [11]
    assert result.events["2018-19/summer_league"]["player_logs"]["dropped"] == {"no_matching_team_game": 1}


def test_a_season_before_the_project_window_is_refused(tmp_path):
    with pytest.raises(no.OffseasonError, match="earlier"):
        no.run_ingest(["2009-10"], FakeClient({}), tmp_path, log=lambda *_: None)
    with pytest.raises(no.OffseasonError):
        no.run_ingest([], FakeClient({}), tmp_path, log=lambda *_: None)


def test_readers_explain_a_missing_ingest(tmp_path):
    with pytest.raises(FileNotFoundError, match="nba_offseason"):
        no.read_offseason_logs(tmp_path)
    with pytest.raises(FileNotFoundError, match="nba_offseason"):
        no.read_offseason_team_games(tmp_path)


def test_write_refuses_an_invalid_table_and_an_unknown_name(tmp_path):
    with pytest.raises(no.OffseasonError):
        no.write_offseason_table(pd.DataFrame({"season": ["x"]}), "offseason_logs", tmp_path)
    with pytest.raises(KeyError):
        no.write_offseason_table(pd.DataFrame(), "nope", tmp_path)


# --------------------------------------------------------------------------- CLI

def test_cli_runs_with_an_injected_client(tmp_path, capsys):
    code = no.main(["--seasons", "2018-19", "--data-dir", str(tmp_path)], client=FakeClient(_two_event_table()))
    assert code == 0
    assert "done: 2 events" in capsys.readouterr().out
    assert len(no.read_offseason_logs(tmp_path)) == 3


def test_cli_context_flag_limits_the_pull(tmp_path):
    client = FakeClient(_two_event_table())
    assert no.main(["--seasons", "2018-19", "--context", "preseason", "--data-dir", str(tmp_path)], client=client) == 0
    assert {k[1] for _, k, _ in client.calls} == {PRE}


def test_cli_reports_a_failure_as_exit_code_one(tmp_path, capsys):
    assert no.main(["--seasons", "2009-10", "--data-dir", str(tmp_path)], client=FakeClient({})) == 1
    assert "ingest failed" in capsys.readouterr().err


def test_cli_rejects_a_malformed_season_spec(tmp_path):
    with pytest.raises(SystemExit) as e:
        no.main(["--seasons", "nonsense", "--data-dir", str(tmp_path)], client=FakeClient({}))
    assert e.value.code == 2
