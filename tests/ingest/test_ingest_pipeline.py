"""nba_stats: season parsing, endpoint parameters, end-to-end pipeline (fake HTTP), idempotent merging, CLI."""
import hashlib

import pandas as pd
import pytest

from ingest_fakes import (
    BOS, NYK, FakeClock, LEAGUE_PLAYERS, RoutedSession, league_payloads, wrap, PLAYER_LOG_HEADERS,
)
from src.contracts import HISTORY_TABLES, ContractError, History, validate_table
from src.ingest import nba_stats as ns
from src.ingest.nba_client import NBAClient
from src.store import read_table

S1, S2 = "2018-19", "2019-20"


def make_client(base, session, **kw):
    clock = FakeClock()
    kw.setdefault("min_interval", 0)
    kw.setdefault("jitter", 0)
    return NBAClient(base / "raw" / "nba_api", session=session, clock=clock, sleep=clock.sleep, offline=False, **kw)


def run(base, seasons, session, **kw):
    client = make_client(base, session)
    return ns.run_ingest(seasons, client, base, log=lambda *_: None, **kw), client


def tables(base):
    return {n: read_table(n, base) for n in HISTORY_TABLES}


def digests(base):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((base / "processed").iterdir())}


# ============================================================ season parsing

@pytest.mark.parametrize("spec, expected", [
    ("2015-16:2017-18", ["2015-16", "2016-17", "2017-18"]),
    ("2023-24", ["2023-24"]),
    ("2015-16,2018-19", ["2015-16", "2018-19"]),
    ("2015-16:2016-17,2020-21", ["2015-16", "2016-17", "2020-21"]),
    ("2018-19,2015-16:2016-17,2016-17", ["2015-16", "2016-17", "2018-19"]),   # sorted + unique
    ("  2015-16 : 2016-17 , 2019-20 ", ["2015-16", "2016-17", "2019-20"]),
    ("2015-16:2015-16", ["2015-16"]),
    ("1999-00:2000-01", ["1999-00", "2000-01"]),                              # century rollover
])
def test_parse_seasons_valid(spec, expected):
    assert ns.parse_seasons(spec) == expected


def test_parse_seasons_full_project_range_has_eleven_seasons():
    s = ns.parse_seasons("2015-16:2025-26")
    assert len(s) == 11 and s[0] == "2015-16" and s[-1] == "2025-26"


@pytest.mark.parametrize("spec, msg", [
    ("", "no seasons"), ("   ", "no seasons"),
    ("2015", "malformed"), ("2015-17", "malformed"), ("15-16", "malformed"), ("2015-16:", "malformed"),
    (":2016-17", "malformed"), ("2015-16:x", "malformed"), ("2015/16", "malformed"),
    ("2016-17:2015-16", "reversed"),
    ("1990-91", "before 1996-97"),
    ("2015-16,,2016-17", "empty item"),
])
def test_parse_seasons_errors_are_readable(spec, msg):
    with pytest.raises(ValueError, match=msg):
        ns.parse_seasons(spec)


# ============================================================ endpoint parameters vs nba_api

def test_leaguegamelog_params_match_nba_api():
    from nba_api.stats.endpoints import LeagueGameLog
    for pt in ("P", "T"):
        theirs = LeagueGameLog(player_or_team_abbreviation=pt, season=S1, get_request=False).parameters
        assert {k: str(v) for k, v in ns.gamelog_params(S1, pt).items()} == {k: str(v) for k, v in theirs.items()}
    with pytest.raises(ValueError):
        ns.gamelog_params(S1, "X")


def test_player_gamelogs_params_match_nba_api_names():
    from nba_api.stats.endpoints import PlayerGameLogs
    theirs = PlayerGameLogs(season_nullable=S1, get_request=False).parameters
    mine = ns.player_gamelogs_params(S1)
    assert set(mine) == set(theirs)
    assert mine["Season"] == S1 and mine["SeasonType"] == "Regular Season" and mine["MeasureType"] == "Base"
    assert mine["PerMode"] == "Totals" and mine["PlayerID"] == "" and mine["TeamID"] == ""


def test_bio_stats_and_index_and_cpi_params_match_nba_api():
    from nba_api.stats.endpoints import CommonPlayerInfo, LeagueDashPlayerBioStats, PlayerIndex
    def norm(d):
        return {k: str(v) for k, v in d.items()}
    assert norm(ns.bio_stats_params(S1)) == norm(LeagueDashPlayerBioStats(season=S1, get_request=False).parameters)
    assert norm(ns.player_index_params(S1)) == norm(PlayerIndex(historical_nullable=1, season=S1, get_request=False).parameters)
    assert norm(ns.common_player_info_params(2544)) == norm(CommonPlayerInfo(player_id=2544, get_request=False).parameters)


def test_params_reject_malformed_season():
    for fn in (ns.player_gamelogs_params, ns.bio_stats_params, ns.player_index_params):
        with pytest.raises(ValueError):
            fn("2018")


# ============================================================ merge helpers

def _gl(season, ids):
    return pd.DataFrame({"season": season, "game_date": pd.Timestamp("2018-11-01"), "game_id": "g", "team_id": 1,
                         "player_id": ids})


def test_merge_seasons_replaces_only_requested_and_sorts_by_season_then_keys():
    old = pd.concat([_gl("2019-20", [1, 2]), _gl("2018-19", [1, 2])], ignore_index=True)
    new = _gl("2019-20", [9])
    out = ns.merge_seasons(old, new, ["2019-20"], ["player_id"])
    assert out[["season", "player_id"]].values.tolist() == [["2018-19", 1], ["2018-19", 2], ["2019-20", 9]]
    assert out.index.tolist() == [0, 1, 2]


def test_merge_seasons_with_no_existing_table():
    out = ns.merge_seasons(None, _gl("2019-20", [2, 1]), ["2019-20"], ["player_id"])
    assert out["player_id"].tolist() == [1, 2]


def test_merge_seasons_sorts_seasons_chronologically_across_century():
    a, b = _gl("1999-00", [1]), _gl("2000-01", [1])
    out = ns.merge_seasons(b, a, ["1999-00"], ["player_id"])
    assert out["season"].tolist() == ["1999-00", "2000-01"]


def _players(rows):
    df = pd.DataFrame(rows)
    for c in ("height_in", "weight_lb", "birthdate", "position", "draft_year", "draft_round", "draft_number", "from_year", "to_year"):
        if c not in df:
            df[c] = None
    return ns.conform_players(df)


def test_merge_players_null_never_erases_stored_value_but_value_replaces():
    old = _players([{"player_id": 1, "player_name": "A", "birthdate": "1990-01-01", "position": "G", "height_in": 70.0},
                    {"player_id": 2, "player_name": "B", "position": "F"}])
    new = _players([{"player_id": 1, "player_name": "A2", "position": None, "height_in": 71.0},
                    {"player_id": 3, "player_name": "C"}])
    out = ns.merge_players(old, new).set_index("player_id")
    assert out.loc[1, "birthdate"] == pd.Timestamp("1990-01-01")   # kept although new has none
    assert out.loc[1, "position"] == "G"                          # kept
    assert out.loc[1, "height_in"] == 71.0 and out.loc[1, "player_name"] == "A2"
    assert set(out.index) == {1, 2, 3}
    validate_table(out.reset_index(), "players")


def test_frames_equal_ignores_dtype_width_but_not_content():
    a = pd.DataFrame({"x": [1, 2], "d": pd.to_datetime(["2018-01-01", "2018-01-02"]).astype("datetime64[us]")})
    b = pd.DataFrame({"x": [1, 2], "d": pd.to_datetime(["2018-01-01", "2018-01-02"]).astype("datetime64[ns]")})
    assert ns.frames_equal(a, b)
    assert not ns.frames_equal(a, b.assign(x=[1, 3]))
    assert not ns.frames_equal(None, b)
    assert not ns.frames_equal(a, b.iloc[:1])


# ============================================================ end-to-end pipeline

@pytest.fixture
def base(tmp_path):
    return tmp_path / "data"


def test_full_pipeline_writes_four_valid_contract_tables(base):
    session = RoutedSession([S1, S2])
    result, _ = run(base, [S1, S2], session)
    t = tables(base)
    for name, df in t.items():
        validate_table(df, name)
    assert set(result.tables) == set(HISTORY_TABLES) and all(v["changed"] for v in result.tables.values())
    # 4 games x 4 players x 2 seasons; 4 games x 2 teams x 2 seasons
    assert len(t["game_logs"]) == 32 and len(t["team_games"]) == 16
    assert t["game_logs"]["season"].tolist() == [S1] * 16 + [S2] * 16
    assert set(t["players"]["player_id"]) == set(LEAGUE_PLAYERS)
    # 3 players x 2 seasons; player 22 has no birthdate and no source AGE, so no bio row (counted in the report)
    assert len(t["player_season_bio"]) == 6


def test_traded_player_bio_team_is_end_of_season_team(base):
    run(base, [S1], RoutedSession([S1]))
    bio = read_table("player_season_bio", base).set_index("player_id")
    assert bio.loc[12, "team_id"] == NYK and bio.loc[11, "team_id"] == BOS
    logs = read_table("game_logs", base)
    assert set(logs[logs.player_id == 12]["team_id"]) == {BOS, NYK}   # per-game team preserved


def test_without_birthdates_flag_ages_use_documented_fallback_and_no_player_info_is_fetched(base):
    session = RoutedSession([S1])
    run(base, [S1], session)
    assert session.calls_to("commonplayerinfo") == 0
    assert read_table("players", base)["birthdate"].isna().all()
    bio = read_table("player_season_bio", base).set_index("player_id")
    exact = (pd.Timestamp("2018-10-01") - pd.Timestamp("1995-03-01")).days / 365.25
    assert abs(bio.loc[11, "age_at_season_start"] - exact) <= 0.5 + 0.01
    assert 22 not in bio.index   # no birthdate and no AGE anywhere: dropped, counted in the report
    assert ns.read_report(base)["seasons"][S1]["player_season_bio"]["dropped"] == {"no_age_source": 1}


def test_birthdates_flag_fetches_each_player_once_and_gives_exact_ages(base):
    session = RoutedSession([S1, S2])
    result, client = run(base, [S1, S2], session, birthdates=True)
    assert session.calls_to("commonplayerinfo") == 4
    players = read_table("players", base).set_index("player_id")
    assert players.loc[11, "birthdate"] == pd.Timestamp("1995-03-01") and pd.isna(players.loc[22, "birthdate"])
    bio = read_table("player_season_bio", base).set_index(["season", "player_id"])
    assert bio.loc[(S1, 12), "age_at_season_start"] == pytest.approx(28.0, abs=0.002)   # born 1990-10-01
    assert bio.loc[(S2, 12), "age_at_season_start"] == pytest.approx(29.0, abs=0.002)
    assert result.birthdates["downloaded"] == 4 and result.birthdates["no_birthdate"] == 1


def test_birthdates_are_resumable_after_a_failure(base):
    session = RoutedSession([S1], fail_player_info_after=2)
    client = make_client(base, session, max_retries=1)
    with pytest.raises(Exception, match="gave up"):
        ns.run_ingest([S1], client, base, birthdates=True, log=lambda *_: None)
    assert not (base / "processed" / "game_logs.parquet").exists()      # nothing half-written
    assert session.calls_to("commonplayerinfo") == 2 + 2                # 2 served, then the failing attempts
    cached_before = len(list((base / "raw" / "nba_api" / "commonplayerinfo").glob("*.json")))
    assert cached_before == 2
    healthy = RoutedSession([S1])
    ns.run_ingest([S1], make_client(base, healthy), base, birthdates=True, log=lambda *_: None)
    assert healthy.calls_to("commonplayerinfo") == 2                    # only the 2 missing players
    assert healthy.calls_to("playergamelogs") == 0                      # season responses cached earlier
    assert read_table("players", base)["birthdate"].notna().sum() == 3


def test_rerun_is_idempotent_bytes_identical_and_offline(base):
    session = RoutedSession([S1, S2])
    run(base, [S1, S2], session, birthdates=True)
    before = digests(base)
    session2 = RoutedSession([S1, S2])
    result, client = run(base, [S1, S2], session2, birthdates=True)
    assert session2.calls == []                       # everything came from the raw cache
    assert not any(v["changed"] for v in result.tables.values())
    assert digests(base) == before                    # tables AND the ingest report byte-identical
    off = NBAClient(base / "raw" / "nba_api", offline=True)
    r3 = ns.run_ingest([S1, S2], off, base, birthdates=True, log=lambda *_: None)
    assert not any(v["changed"] for v in r3.tables.values()) and digests(base) == before


def test_seasons_can_be_added_incrementally_and_match_a_single_full_run(base, tmp_path):
    session = RoutedSession([S1, S2])
    run(base, [S1], session)
    run(base, [S2], session)
    run(base, [S1], session)                           # re-touching an old season changes nothing
    other = tmp_path / "other"
    run(other, [S1, S2], RoutedSession([S1, S2]))
    a, b = tables(base), tables(other)
    for name in HISTORY_TABLES:
        pd.testing.assert_frame_equal(a[name], b[name], check_dtype=False, obj=name)


def test_later_run_without_birthdates_keeps_birthdates_and_ages_stable(base):
    run(base, [S1], RoutedSession([S1]), birthdates=True)
    ages = read_table("player_season_bio", base)
    players = read_table("players", base)
    run(base, [S1], RoutedSession([S1]), birthdates=False)   # cache still has the birthdates
    pd.testing.assert_frame_equal(read_table("player_season_bio", base), ages)
    pd.testing.assert_frame_equal(read_table("players", base), players)


def test_history_guard_works_on_ingested_tables(base):
    run(base, [S1, S2], RoutedSession([S1, S2]), birthdates=True)
    hist = History.until(tables(base), S2)
    hist.assert_no_future()
    assert set(hist.game_logs["season"]) == {S1}


def test_short_season_game_counts_are_data_driven(base):
    run(base, [S1], RoutedSession([S1], n_games=2))
    tg = read_table("team_games", base)
    assert tg.groupby("team_id").size().to_dict() == {BOS: 2, NYK: 2}   # nothing assumes 82


def test_report_records_drops_crosscheck_and_consistency(base):
    run(base, [S1], RoutedSession([S1]))
    rep = ns.read_report(base)["seasons"][S1]
    assert rep["game_logs"]["rows_out"] == 16 and rep["n_games"] == 4
    assert rep["consistency"]["orphan_player_rows"] == 0 and rep["consistency"]["team_points_mismatch"] == 0
    assert rep["crosscheck"]["stat_mismatches"] == {} and rep["crosscheck"]["n_only_in_primary_over_half_minute"] == 0


def test_no_crosscheck_skips_the_extra_pull_but_keeps_the_previous_result(base):
    s = RoutedSession([S1])
    run(base, [S1], s)
    n_before = s.calls_to("leaguegamelog")
    fresh = RoutedSession([S1])
    run(base, [S1], fresh, crosscheck=False, refresh=True)
    assert fresh.calls_to("leaguegamelog") == 1        # only the team-game pull
    assert "crosscheck" in ns.read_report(base)["seasons"][S1] and n_before == 2


def test_refresh_redownloads_season_level_responses(base):
    s1 = RoutedSession([S1])
    run(base, [S1], s1)
    s2 = RoutedSession([S1])
    run(base, [S1], s2, refresh=True)
    assert s2.calls_to("playergamelogs") == 1 and s2.calls_to("leaguedashplayerbiostats") == 1


# ============================================================ failures surface clearly, nothing half-written

def test_empty_season_is_an_error_and_writes_nothing(base):
    session = RoutedSession([S1])
    with pytest.raises(ns.IngestError, match="no regular-season games"):
        run(base, ["2026-27"], session)
    assert not (base / "processed").exists()


def test_orphan_player_rows_are_refused(base):
    p = league_payloads(S1)
    hdr = p["playergamelogs"]["resultSets"][0]["headers"]
    for row in p["playergamelogs"]["resultSets"][0]["rowSet"]:
        if row[hdr.index("PLAYER_ID")] == 21:
            row[hdr.index("TEAM_ID")] = 1610612999   # a team that has no team_games row
    session = RoutedSession([S1], extra={("playergamelogs", S1, None): p["playergamelogs"]})
    with pytest.raises(ns.IngestError, match="missing from team_games"):
        run(base, [S1], session)
    assert not (base / "processed").exists()


def test_box_score_identity_violation_stops_the_ingest_before_writing(base):
    p = league_payloads(S1)
    hdr = p["playergamelogs"]["resultSets"][0]["headers"]
    p["playergamelogs"]["resultSets"][0]["rowSet"][0][hdr.index("PTS")] += 5
    session = RoutedSession([S1], extra={("playergamelogs", S1, None): p["playergamelogs"]})
    with pytest.raises(ContractError, match="pts != 2\\*fgm"):
        run(base, [S1], session)
    assert not (base / "processed").exists()


def test_existing_tables_survive_a_failed_run(base):
    run(base, [S1], RoutedSession([S1]))
    before = digests(base)
    bad = RoutedSession([S1, S2], extra={("playergamelogs", S2, None): wrap("PlayerGameLogs", PLAYER_LOG_HEADERS, [], "gamelogs")})
    with pytest.raises(ns.IngestError):
        run(base, [S2], bad)
    assert digests(base) == before


# ============================================================ CLI

def test_cli_happy_path_and_summary(base, capsys):
    session = RoutedSession([S1, S2])
    rc = ns.main(["--seasons", f"{S1}:{S2}", "--data-dir", str(base), "--no-crosscheck"],
                 client=make_client(base, session))
    out = capsys.readouterr().out
    assert rc == 0 and "done: 2 seasons" in out and "game_logs: 32 rows (written)" in out
    assert read_table("game_logs", base).shape[0] == 32


def test_cli_second_run_reports_unchanged(base, capsys):
    session = RoutedSession([S1])
    ns.main(["--seasons", S1, "--data-dir", str(base)], client=make_client(base, session))
    capsys.readouterr()
    ns.main(["--seasons", S1, "--data-dir", str(base)], client=make_client(base, RoutedSession([S1])))
    out = capsys.readouterr().out
    assert "(unchanged)" in out and "(written)" not in out


@pytest.mark.parametrize("argv", [[], ["--seasons"], ["--seasons", "2015"], ["--seasons", "2016-17:2015-16"], ["--bogus"]])
def test_cli_bad_arguments_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as ei:
        ns.main(argv)
    assert ei.value.code == 2
    assert capsys.readouterr().err   # a readable message on stderr


def test_cli_bad_season_message_names_the_problem(capsys):
    with pytest.raises(SystemExit):
        ns.main(["--seasons", "2015-17"])
    assert "malformed season '2015-17'" in capsys.readouterr().err


def test_cli_flags_are_parsed():
    a = ns.build_parser().parse_args(["--seasons", "2015-16:2025-26", "--offline", "--birthdates", "--refresh",
                                      "--min-interval", "2.5", "--data-dir", "x"])
    assert a.offline and a.birthdates and a.refresh and a.min_interval == 2.5 and str(a.data_dir) == "x"
    d = ns.build_parser().parse_args(["--seasons", "2015-16"])
    assert not d.offline and not d.birthdates and not d.refresh and d.min_interval == 1.0 and d.max_retries == 6


def test_cli_offline_with_empty_cache_fails_with_clear_message_and_exit_1(base, capsys):
    rc = ns.main(["--seasons", S1, "--offline", "--data-dir", str(base)])
    err = capsys.readouterr().err
    assert rc == 1 and "NBAOfflineCacheMiss" in err and "offline mode" in err and "NBA_OFFLINE" in err
    assert not (base / "processed").exists()


def test_cli_offline_env_var_is_honoured(base, monkeypatch, capsys):
    monkeypatch.setenv("NBA_OFFLINE", "1")
    rc = ns.main(["--seasons", S1, "--data-dir", str(base)])
    assert rc == 1 and "NBAOfflineCacheMiss" in capsys.readouterr().err


def test_cli_offline_works_from_a_populated_cache(base, capsys):
    ns.main(["--seasons", S1, "--data-dir", str(base)], client=make_client(base, RoutedSession([S1])))
    capsys.readouterr()
    rc = ns.main(["--seasons", S1, "--offline", "--data-dir", str(base)])
    out = capsys.readouterr().out
    assert rc == 0 and "0 network requests" in out


def test_cli_reports_contract_errors_as_exit_1_not_a_traceback(base, capsys):
    p = league_payloads(S1)
    hdr = p["playergamelogs"]["resultSets"][0]["headers"]
    p["playergamelogs"]["resultSets"][0]["rowSet"][0][hdr.index("REB")] += 7
    session = RoutedSession([S1], extra={("playergamelogs", S1, None): p["playergamelogs"]})
    rc = ns.main(["--seasons", S1, "--data-dir", str(base)], client=make_client(base, session))
    assert rc == 1 and "ContractError" in capsys.readouterr().err


def test_cli_enforces_politeness_floor(base, capsys, monkeypatch):
    captured = {}

    class Spy(NBAClient):
        def __init__(self, *a, **k):
            captured.update(k)
            super().__init__(*a, **{**k, "session": RoutedSession([S1]), "sleep": lambda s: None})

    monkeypatch.setattr(ns, "NBAClient", Spy)
    assert ns.main(["--seasons", S1, "--min-interval", "0.1", "--data-dir", str(base)]) == 0
    assert "below the 0.8 s politeness floor" in capsys.readouterr().err
    assert captured["min_interval"] == 0.8 and captured["offline"] is None


def test_cli_offline_flag_reaches_the_client(base, monkeypatch):
    captured = {}

    class Spy(NBAClient):
        def __init__(self, *a, **k):
            captured.update(k)
            super().__init__(*a, **k)

    monkeypatch.setattr(ns, "NBAClient", Spy)
    ns.main(["--seasons", S1, "--offline", "--data-dir", str(base)])
    assert captured["offline"] is True


def test_default_data_dir_follows_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path / "envdata"))
    rc = ns.main(["--seasons", S1, "--offline"])
    assert rc == 1
    assert str(tmp_path / "envdata") in capsys.readouterr().err   # the cache path in the error message
