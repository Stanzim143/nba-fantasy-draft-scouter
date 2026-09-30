"""Raw stats.nba.com JSON -> contract tables. Every edge case listed in the ingest spec has a test."""
import math
import random

import numpy as np
import pandas as pd
import pytest

from ingest_fakes import (
    BKN, BOS, GSW, NYK, bio_stats_payload, common_player_info_payload, league_p_payload, player_index_payload,
    player_payload, player_row, team_payload, team_row,
)
from src.contracts import ContractError, TABLES, validate_table
from src.ingest import nba_transform as tf

S = "2018-19"


def gl(rows, season=S):
    return tf.transform_game_logs(player_payload(rows), season)


# ============================================================ generic parsers

@pytest.mark.parametrize("raw, expected", [
    (34.5, 34.5), (34, 34.0), ("34:30", 34.5), ("0:30", 0.5), ("12:00", 12.0), ("7:05", 7 + 5 / 60),
    ("34:30.60", 34 + 30.6 / 60), ("36.45", 36.45), (" 10:15 ", 10.25),
])
def test_parse_minutes_valid(raw, expected):
    assert tf.parse_minutes(pd.Series([raw])).iloc[0] == pytest.approx(expected)


@pytest.mark.parametrize("raw", [None, np.nan, "", "-", "None", "  "])
def test_parse_minutes_missing_is_nan(raw):
    assert math.isnan(tf.parse_minutes(pd.Series([raw])).iloc[0])


def test_parse_minutes_garbage_raises():
    with pytest.raises(tf.TransformError, match="unparseable minutes"):
        tf.parse_minutes(pd.Series(["DNP - Coach's Decision"]))


def test_parse_game_date_formats():
    s = pd.Series(["2019-04-10T00:00:00", "2019-04-10", "APR 10, 2019", " 2019-04-10 "])
    out = tf.parse_game_date(s)
    assert (out == pd.Timestamp("2019-04-10")).all()
    assert pd.api.types.is_datetime64_any_dtype(out)


def test_parse_game_date_bad_value_raises_with_examples():
    with pytest.raises(tf.TransformError, match="unparseable game dates.*not a date"):
        tf.parse_game_date(pd.Series(["2019-04-10", "not a date"]))


@pytest.mark.parametrize("raw, expected", [("6-9", 81), ("7-0", 84), ("5-10", 70), (" 6 - 2 ", 74)])
def test_parse_height(raw, expected):
    assert tf.parse_height_inches(pd.Series([raw])).iloc[0] == expected


@pytest.mark.parametrize("raw", [None, "", "6-", "tall", np.nan, "-"])
def test_parse_height_junk_is_nan(raw):
    assert math.isnan(tf.parse_height_inches(pd.Series([raw])).iloc[0])


@pytest.mark.parametrize("raw, expected", [
    ("Guard-Forward", "G-F"), ("Center", "C"), ("G-F", "G-F"), ("F-C", "F-C"), ("", None), (None, None), (np.nan, None),
    ("Forward-Center", "F-C"),
])
def test_normalize_position(raw, expected):
    assert tf.normalize_position(raw) == expected


@pytest.mark.parametrize("gid, ok", [
    ("0021800001", True), ("0021801230", True),
    ("0011800001", False),  # preseason
    ("0031800001", False),  # All-Star
    ("0041800101", False),  # playoffs
    ("0051800001", False),  # play-in
    ("0061800001", False),  # NBA Cup knockout
    ("1021800001", False),  # WNBA / other league
    ("2021800001", False),
    ("0021900001", False),  # regular season, but a different season
    ("002180001", False),   # malformed length
    ("", False),
])
def test_regular_season_game_id(gid, ok):
    assert tf.is_regular_season_game_id(gid, "2018-19") is ok


def test_game_id_prefix_century_boundary():
    assert tf.game_id_prefix("1999-00") == "00299"
    assert tf.game_id_prefix("2000-01") == "00200"


def test_result_set_picks_by_name_and_reports_missing():
    p = {"resultSets": [{"name": "A", "headers": ["x"], "rowSet": [[1]]}, {"name": "B", "headers": ["y"], "rowSet": [[2]]}]}
    assert tf.result_set(p, "B")["y"].tolist() == [2]
    assert tf.result_set(p)["x"].tolist() == [1]
    with pytest.raises(tf.TransformError, match="no result set named 'C'.*\\['A', 'B'\\]"):
        tf.result_set(p, "C")
    with pytest.raises(tf.TransformError):
        tf.result_set({"resultSets": []})
    with pytest.raises(tf.TransformError):
        tf.result_set([1, 2])
    assert tf.result_set({"resultSet": {"name": "A", "headers": ["x"], "rowSet": [[1]]}})["x"].tolist() == [1]


# ============================================================ game_logs

def test_basic_game_log_matches_contract_exactly():
    df, rep = gl([player_row(), player_row(player_id=203110, name="Draymond Green", minutes=30.25)])
    assert list(df.columns) == list(TABLES["game_logs"].columns)
    assert rep.rows_in == 2 and rep.rows_out == 2 and rep.dropped == {}
    validate_table(df, "game_logs", allow_extra=False)
    r = df[df.player_id == 201939].iloc[0]
    assert r["season"] == S and r["game_id"] == "0021800002"
    assert r["game_date"] == pd.Timestamp("2018-10-16")
    assert r["min"] == pytest.approx(33.5) and r["pts"] == 2 * 8 + 4 + 3
    assert r["team_abbr"] == "GSW" and r["matchup"] == "GSW vs. OKC"
    for c in ("fgm", "pts", "player_id", "team_id", "reb"):
        assert df[c].dtype == np.int64


def test_output_is_sorted_and_index_is_clean():
    rows = [player_row(player_id=2, game_id="0021800005", date="2018-10-20"),
            player_row(player_id=1, game_id="0021800002", date="2018-10-16"),
            player_row(player_id=3, game_id="0021800002", date="2018-10-16")]
    df, _ = gl(rows)
    assert df["game_id"].tolist() == ["0021800002", "0021800002", "0021800005"]
    assert df["player_id"].tolist() == [1, 3, 2]
    assert df.index.tolist() == [0, 1, 2]


def test_fractional_minutes_are_kept_not_rounded():
    df, _ = gl([player_row(minutes=36.451666666666664)])
    assert df["min"].iloc[0] == pytest.approx(36.4516666)


def test_minutes_as_mmss_strings():
    df, _ = gl([player_row(player_id=1, minutes="34:30"), player_row(player_id=2, minutes="0:45")])
    assert df.set_index("player_id")["min"].to_dict() == {1: 34.5, 2: 0.75}


def test_minutes_fall_back_to_min_sec_column_when_min_is_all_null():
    rows = [player_row(minutes=None)]
    payload = player_payload(rows)
    idx = payload["resultSets"][0]["headers"].index("MIN_SEC")
    payload["resultSets"][0]["rowSet"][0][idx] = "21:30"
    df, rep = tf.transform_game_logs(payload, S)
    assert df["min"].iloc[0] == 21.5 and rep.dropped == {}


@pytest.mark.parametrize("bad_minutes", [0, 0.0, None, np.nan, "", "0:00", "-"])
def test_rows_without_minutes_are_dropped_as_did_not_play(bad_minutes):
    rows = [player_row(player_id=1), player_row(player_id=2, minutes=bad_minutes, fgm=0, fga=0, fg3m=0, fg3a=0,
                                                ftm=0, fta=0, oreb=0, dreb=0, ast=0, stl=0, blk=0, tov=0, pf=0)]
    df, rep = gl(rows)
    assert df["player_id"].tolist() == [1]
    assert rep.dropped == {"no_minutes_played": 1}


def test_tiny_but_positive_minutes_are_kept():
    df, rep = gl([player_row(minutes=0.05)])
    assert len(df) == 1 and rep.dropped == {}


def test_zero_minute_row_with_points_is_still_dropped_and_counted():
    # e.g. a bench player credited with a free throw at 0:00. Contract says min > 0; we drop and report.
    rows = [player_row(player_id=1), player_row(player_id=2, minutes="0:00", fgm=0, fga=0, fg3m=0, fg3a=0, ftm=1, fta=1,
                                                oreb=0, dreb=0)]
    df, rep = gl(rows)
    assert 2 not in df["player_id"].tolist() and rep.dropped["no_minutes_played"] == 1


def test_identical_duplicate_rows_are_dropped_once():
    r = player_row()
    df, rep = gl([r, list(r), list(r)])
    assert len(df) == 1 and rep.dropped == {"duplicate_row_identical": 2}


def test_conflicting_duplicates_keep_the_row_with_most_minutes():
    df, rep = gl([player_row(minutes=10.0, pts=None), player_row(minutes=33.5), player_row(minutes=20.0)])
    assert len(df) == 1 and df["min"].iloc[0] == 33.5
    assert rep.dropped == {"duplicate_row_conflicting": 2}


def test_conflicting_duplicates_result_is_order_independent():
    rows = [player_row(minutes=10.0), player_row(minutes=33.5), player_row(minutes=20.0)]
    a, _ = gl(rows)
    b, _ = gl(rows[::-1])
    pd.testing.assert_frame_equal(a, b)


def test_all_star_preseason_playoff_and_other_league_ids_are_dropped():
    rows = [player_row(player_id=1, game_id="0021800002"),
            player_row(player_id=2, game_id="0031800001"),   # All-Star
            player_row(player_id=3, game_id="0011800001"),   # preseason
            player_row(player_id=4, game_id="0041800101"),   # playoffs
            player_row(player_id=5, game_id="1021800001"),   # WNBA-style id
            player_row(player_id=6, game_id="0021700002")]   # other season
    df, rep = gl(rows)
    assert df["player_id"].tolist() == [1]
    assert rep.dropped == {"not_regular_season_game_id": 5}


def test_traded_player_keeps_the_team_of_each_game():
    rows = [player_row(team_id=BOS, abbr="BOS", game_id="0021800010", date="2018-11-01", matchup="BOS vs. NYK"),
            player_row(team_id=NYK, abbr="NYK", game_id="0021800900", date="2019-02-20", matchup="NYK @ BOS")]
    df, _ = gl(rows)
    assert df["team_id"].tolist() == [BOS, NYK] and df["team_abbr"].tolist() == ["BOS", "NYK"]
    assert df["player_id"].nunique() == 1  # one person, two teams


def test_team_abbreviation_changes_do_not_change_team_id_identity():
    old = gl([player_row(team_id=BKN, abbr="NJN", matchup="NJN vs. BOS", game_id="0021500001", date="2015-10-28",
                         season="2015-16")], "2015-16")[0]
    new = gl([player_row(team_id=BKN, abbr="BKN", matchup="BKN vs. BOS", game_id="0021600001", date="2016-10-28",
                         season="2016-17")], "2016-17")[0]
    both = pd.concat([old, new], ignore_index=True)
    assert both["team_id"].nunique() == 1  # team_id is the truth ...
    assert both["team_abbr"].tolist() == ["NJN", "BKN"]  # ... abbreviations stay as the source reported them


def test_missing_box_score_fields_drop_the_row_and_count_it():
    payload = player_payload([player_row(player_id=1), player_row(player_id=2)])
    hdr = payload["resultSets"][0]["headers"]
    payload["resultSets"][0]["rowSet"][1][hdr.index("FGA")] = None
    df, rep = tf.transform_game_logs(payload, S)
    assert df["player_id"].tolist() == [1] and rep.dropped == {"missing_box_score_fields": 1}


def test_null_plus_minus_is_allowed():
    df, _ = gl([player_row(plus_minus=None)])
    assert math.isnan(df["plus_minus"].iloc[0])


def test_names_and_abbreviations_are_stripped():
    df, _ = gl([player_row(name="  Nikola Jokić ", abbr="GSW ")])
    assert df["player_name"].iloc[0] == "Nikola Jokić" and df["team_abbr"].iloc[0] == "GSW"


def test_wrong_season_rows_are_rejected():
    with pytest.raises(tf.TransformError, match="other seasons"):
        gl([player_row(season="2017-18")])


def test_missing_columns_named_in_error():
    payload = player_payload([player_row()])
    payload["resultSets"][0]["headers"][payload["resultSets"][0]["headers"].index("TOV")] = "TURNOVERS"
    with pytest.raises(tf.TransformError, match="missing columns.*TOV"):
        tf.transform_game_logs(payload, S)


def test_malformed_season_argument():
    with pytest.raises(ValueError, match="malformed season"):
        tf.transform_game_logs(player_payload([player_row()]), "2018")


def test_box_score_identity_violation_surfaces_as_contract_error():
    with pytest.raises(ContractError, match=r"pts != 2\*fgm \+ fg3m \+ ftm"):
        gl([player_row(pts=99)])
    with pytest.raises(ContractError, match=r"reb != oreb \+ dreb"):
        gl([player_row(reb=99)])
    with pytest.raises(ContractError, match="fgm > fga"):
        gl([player_row(fgm=9, fga=8, fg3m=0, fg3a=0)])


def test_integer_minute_league_log_shape_is_accepted():
    rows = [player_row(minutes=33.6), player_row(player_id=2, minutes=0.2)]
    df, rep = tf.transform_game_logs(league_p_payload(rows), S)
    assert df["min"].tolist() == [34.0]  # rounded to whole minutes by the source; 0.2 -> 0 -> dropped
    assert rep.dropped == {"no_minutes_played": 1}
    assert df["game_date"].iloc[0] == pd.Timestamp("2018-10-16")


def test_shortened_season_needs_no_82_game_assumption():
    # 3 games for one team is a perfectly valid "season" as far as the transform is concerned
    rows = [player_row(game_id=f"002190{i:04d}", date=f"2020-03-0{i}", season="2019-20") for i in range(1, 4)]
    df, _ = gl(rows, "2019-20")
    assert len(df) == 3


def test_empty_payload_gives_valid_empty_table():
    df, rep = gl([])
    assert df.empty and list(df.columns) == list(TABLES["game_logs"].columns) and rep.rows_out == 0


# ============================================================ crosscheck

def test_crosscheck_identical_sources_agree():
    rows = [player_row(player_id=1, minutes=33.6), player_row(player_id=2, minutes=20.2)]
    a, _ = gl(rows)
    b, _ = tf.transform_game_logs(league_p_payload(rows), S)
    rep = tf.crosscheck_game_logs(a, b)
    assert rep["stat_mismatches"] == {} and rep["n_only_in_primary"] == 0
    assert rep["minutes_out_of_tolerance"] == 0 and rep["minutes_max_abs_diff"] <= 0.5 + 1e-9


def test_crosscheck_flags_stat_mismatch_and_missing_rows():
    rows = [player_row(player_id=1, minutes=33.6), player_row(player_id=2, minutes=20.2),
            player_row(player_id=3, minutes=0.3, fgm=0, fga=0, fg3m=0, fg3a=0, ftm=0, fta=0, oreb=0, dreb=0)]
    a, _ = gl(rows)
    b, _ = tf.transform_game_logs(league_p_payload(rows), S)  # player 3 rounds to 0 minutes: absent
    b.loc[b.player_id == 1, "ast"] += 1
    rep = tf.crosscheck_game_logs(a, b)
    assert rep["stat_mismatches"] == {"ast": 1}
    assert rep["n_only_in_primary"] == 1 and rep["n_only_in_primary_over_half_minute"] == 0  # explained by rounding
    b2 = b[b.player_id != 2]
    rep2 = tf.crosscheck_game_logs(a, b2)
    assert rep2["n_only_in_primary_over_half_minute"] == 1  # a real, unexplained gap


# ============================================================ team_games

def tg(rows, season=S):
    return tf.transform_team_games(team_payload(rows), season)


def test_team_games_basic():
    rows = [team_row(team_id=GSW, abbr="GSW", matchup="GSW vs. OKC", pts=108),
            team_row(team_id=1610612760, abbr="OKC", matchup="OKC @ GSW", pts=100)]
    df, rep = tg(rows)
    assert list(df.columns) == list(TABLES["team_games"].columns)
    validate_table(df, "team_games", allow_extra=False)
    g = df.set_index("team_abbr")
    assert g.loc["GSW", "is_home"] and not g.loc["OKC", "is_home"]
    assert g.loc["GSW", "pts_for"] == 108 and g.loc["GSW", "pts_against"] == 100
    assert g.loc["OKC", "pts_for"] == 100 and g.loc["OKC", "pts_against"] == 108
    assert df["pts_for"].dtype == np.int64 and df["is_home"].dtype == bool
    assert rep.rows_out == 2 and rep.dropped == {}


def test_team_games_single_row_game_has_null_pts_against():
    df, _ = tg([team_row()])
    assert pd.isna(df["pts_against"].iloc[0]) and df["pts_for"].iloc[0] == 108
    validate_table(df, "team_games")


def test_team_games_drops_duplicates_and_non_regular_games():
    rows = [team_row(), team_row(), team_row(game_id="0031800001", matchup="GSW vs. OKC"),
            team_row(team_id=1610612760, abbr="OKC", matchup="OKC @ GSW", pts=100)]
    df, rep = tg(rows)
    assert len(df) == 2
    assert rep.dropped == {"not_regular_season_game_id": 1, "duplicate_row": 1}


def test_team_games_unrecognised_matchup_is_an_error():
    with pytest.raises(tf.TransformError, match="unrecognised MATCHUP"):
        tg([team_row(matchup="GSW v OKC")])


def test_team_games_missing_columns():
    payload = team_payload([team_row()])
    payload["resultSets"][0]["headers"][payload["resultSets"][0]["headers"].index("PTS")] = "POINTS"
    with pytest.raises(tf.TransformError, match="missing columns.*PTS"):
        tf.transform_team_games(payload, S)


def test_team_games_short_season_is_fine_and_counts_are_data_driven():
    rows = []
    for i in range(1, 4):  # 3 games between two teams
        gid = f"002190{i:04d}"
        rows += [team_row(game_id=gid, matchup="BOS vs. NYK", team_id=BOS, abbr="BOS", date=f"2020-03-0{i}", pts=100 + i),
                 team_row(game_id=gid, matchup="NYK @ BOS", team_id=NYK, abbr="NYK", date=f"2020-03-0{i}", pts=90 + i)]
    df, _ = tf.transform_team_games(team_payload(rows), "2019-20")
    assert df.groupby("team_id").size().to_dict() == {BOS: 3, NYK: 3}


def test_team_games_overtime_minutes_do_not_matter():
    rows = [team_row(minutes=265, matchup="GSW vs. OKC"), team_row(team_id=1610612760, abbr="OKC", matchup="OKC @ GSW", minutes=265)]
    df, _ = tg(rows)
    assert len(df) == 2


def test_team_games_missing_pts_nulls_pts_against_for_both_teams_not_zero():
    # A malformed row with no PTS must not silently make the OTHER team's pts_against compute as
    # sum-of-valid-minus-own (which lands on 0, a wrong non-null value) instead of unknown/null.
    rows = [team_row(pts=None, matchup="GSW vs. OKC"),
            team_row(team_id=1610612760, abbr="OKC", matchup="OKC @ GSW", pts=100)]
    df, _ = tg(rows)
    g = df.set_index("team_abbr")
    assert pd.isna(g.loc["GSW", "pts_for"])
    assert pd.isna(g.loc["GSW", "pts_against"])
    assert g.loc["OKC", "pts_for"] == 100
    assert pd.isna(g.loc["OKC", "pts_against"])  # must NOT be 0


# ============================================================ birthdates / players

@pytest.mark.parametrize("raw, ok", [
    ("1988-03-14T00:00:00", True), ("1988-03-14", True), (None, False), ("", False), (np.nan, False),
    ("0001-01-01T00:00:00", False), ("not a date", False), ("2999-01-01", False),
])
def test_parse_birthdate(raw, ok):
    got = tf.parse_birthdate(raw)
    assert (got is not None) is ok
    if ok:
        assert got == pd.Timestamp("1988-03-14")


def test_common_player_info_extracts_birthdate_and_attrs():
    p = common_player_info_payload(201939, birthdate="1988-03-14T00:00:00", position="Guard-Forward",
                                   height="6-2", weight="185", draft=("2009", "1", "7"))
    assert tf.birthdates_from_common_player_info(p) == (201939, pd.Timestamp("1988-03-14"))
    a = tf.player_attributes_from_common_player_info(p)
    assert a["position"] == "G-F" and a["height_in"] == 74 and a["weight_lb"] == 185
    assert (a["draft_year"], a["draft_round"], a["draft_number"]) == (2009, 1, 7)


def test_common_player_info_no_birthdate_and_undrafted():
    p = common_player_info_payload(5, birthdate=None, draft=("Undrafted", "Undrafted", "Undrafted"))
    assert tf.birthdates_from_common_player_info(p) == (5, None)
    a = tf.player_attributes_from_common_player_info(p)
    assert a["draft_year"] is None and a["draft_round"] is None


def test_common_player_info_empty_response_is_a_transform_error():
    empty = {"resultSets": [{"name": "CommonPlayerInfo", "headers": ["PERSON_ID", "BIRTHDATE"], "rowSet": []}]}
    with pytest.raises(tf.TransformError, match="no CommonPlayerInfo row"):
        tf.birthdates_from_common_player_info(empty)


def _logs_two_players():
    rows = [player_row(player_id=1, name="Old Name", game_id="0021800002", date="2018-10-16"),
            player_row(player_id=1, name="New Name", game_id="0021800900", date="2019-02-01"),
            player_row(player_id=2, name="Rookie Guy", game_id="0021800002", date="2018-10-16")]
    return gl(rows)[0]


def test_player_index_frame_types_and_junk():
    idx = tf.player_index_frame(player_index_payload([
        {"id": 1, "first": "Kobe", "last": "Bryant", "pos": "G-F", "height": "6-6", "weight": "212",
         "dy": 1996.0, "dr": 1.0, "dn": 13.0, "fy": "1996", "ty": "2015"},
        {"id": 2, "first": "Undrafted", "last": "Guy", "pos": None, "height": None, "weight": None,
         "dy": None, "dr": None, "dn": None, "fy": "2018", "ty": "2019"},
    ]))
    a = idx.set_index("player_id")
    assert a.loc[1, "player_name"] == "Kobe Bryant" and a.loc[1, "height_in"] == 78 and a.loc[1, "from_year"] == 1996
    assert pd.isna(a.loc[2, "draft_year"]) and pd.isna(a.loc[2, "height_in"]) and a.loc[2, "position"] is None
    assert str(a["draft_year"].dtype) == "Int64"


def test_build_players_contract_and_name_is_most_recent():
    logs = _logs_two_players()
    idx = tf.player_index_frame(player_index_payload([
        {"id": 1, "pos": "G", "height": "6-2", "weight": "185", "dy": 2009, "dr": 1, "dn": 7, "fy": 2009, "ty": 2025},
        {"id": 2, "pos": "F-C", "height": "6-11", "weight": "250", "dy": None, "dr": None, "dn": None, "fy": 2018, "ty": 2018}]))
    out = tf.build_players(logs, idx, None, {1: pd.Timestamp("1988-03-14")})
    assert list(out.columns) == list(TABLES["players"].columns)
    p = out.set_index("player_id")
    assert p.loc[1, "player_name"] == "New Name"
    assert p.loc[1, "birthdate"] == pd.Timestamp("1988-03-14") and pd.isna(p.loc[2, "birthdate"])  # no birthdate: null, never guessed
    assert p.loc[2, "position"] == "F-C" and pd.isna(p.loc[2, "draft_year"])
    assert set(out["player_id"]) == {1, 2}  # exactly the players who appear in game_logs


def test_build_players_restricts_to_players_in_game_logs():
    logs = _logs_two_players()
    idx = tf.player_index_frame(player_index_payload([{"id": i, "pos": "G"} for i in (1, 2, 3, 4, 5)]))
    assert set(tf.build_players(logs, idx)["player_id"]) == {1, 2}


def test_build_players_falls_back_from_index_to_common_info_to_bio_stats():
    logs = _logs_two_players()
    idx = tf.player_index_frame(player_index_payload([{"id": 1, "pos": "G", "height": "6-2"}]))  # nothing about player 2
    extra = pd.DataFrame([{"player_id": 2, "position": "C", "height_in": 84.0, "weight_lb": 260.0,
                           "draft_year": None, "draft_round": None, "draft_number": None, "from_year": 2018, "to_year": 2019}])
    bio = {"2017-18": tf.bio_stats_frame(bio_stats_payload([{"id": 1, "age": 20, "hin": 70, "w": 150, "dy": "2009", "dr": "1", "dn": "3"}])),
           "2018-19": tf.bio_stats_frame(bio_stats_payload([{"id": 1, "age": 21, "hin": 75, "w": 199, "dy": "2009", "dr": "1", "dn": "7"}]))}
    p = tf.build_players(logs, idx, bio, None, extra).set_index("player_id")
    assert p.loc[1, "height_in"] == 74            # index wins over bio stats
    assert p.loc[1, "weight_lb"] == 199           # index has none -> most recent bio-stats season wins
    assert p.loc[1, "draft_number"] == 7
    assert p.loc[2, "position"] == "C" and p.loc[2, "height_in"] == 84


def test_build_players_with_nothing_but_game_logs_still_valid():
    p = tf.build_players(_logs_two_players())
    assert len(p) == 2 and p["birthdate"].isna().all() and p["position"].isna().all()
    validate_table(p, "players")


# ============================================================ player_season_bio

def _bio_logs():
    rows = [
        # player 1 traded mid-season BOS -> NYK: bio team = NYK (last game)
        player_row(player_id=1, team_id=BOS, abbr="BOS", game_id="0021800010", date="2018-11-01"),
        player_row(player_id=1, team_id=NYK, abbr="NYK", game_id="0021800900", date="2019-02-20"),
        player_row(player_id=1, team_id=NYK, abbr="NYK", game_id="0021801200", date="2019-04-05"),
        # player 2: two games on the same date; the later game_id is "last"
        player_row(player_id=2, team_id=GSW, abbr="GSW", game_id="0021800200", date="2018-12-25"),
        player_row(player_id=2, team_id=BOS, abbr="BOS", game_id="0021800201", date="2018-12-25"),
        # player 3: a different season entirely
        player_row(player_id=3, team_id=GSW, abbr="GSW", game_id="0021700100", date="2017-11-25", season="2017-18"),
    ]
    frames = [gl([r], r[0])[0] for r in rows]
    return pd.concat(frames, ignore_index=True)


def test_bio_team_is_the_team_at_end_of_season_for_traded_players():
    bio, rep = tf.build_player_season_bio(_bio_logs(), {1: pd.Timestamp("1990-01-01"), 2: pd.Timestamp("1990-01-01"),
                                                        3: pd.Timestamp("1990-01-01")})
    b = bio.set_index(["season", "player_id"])
    assert b.loc[(S, 1), "team_id"] == NYK
    assert b.loc[(S, 2), "team_id"] == BOS       # tie on date broken by game_id
    assert b.loc[("2017-18", 3), "team_id"] == GSW
    assert list(bio.columns) == list(TABLES["player_season_bio"].columns)
    assert len(bio) == 3 and rep.dropped == {}


def test_bio_one_row_per_player_season_sorted():
    logs = _bio_logs()
    bd = {i: pd.Timestamp("1990-01-01") for i in (1, 2, 3)}
    bio, _ = tf.build_player_season_bio(logs, bd)
    assert bio["season"].tolist() == ["2017-18", S, S] and bio["player_id"].tolist() == [3, 1, 2]


@pytest.mark.parametrize("birth, season, expected", [
    ("2000-10-01", "2020-21", 20.0),                 # birthday exactly on Oct 1 -> exactly 20
    ("1990-10-01", "2015-16", 25.0),
    ("2000-10-02", "2020-21", 20 - 1 / 365.25),      # one day short of 20
    ("1995-09-16", "2018-19", (pd.Timestamp("2018-10-01") - pd.Timestamp("1995-09-16")).days / 365.25),
])
def test_exact_age_is_as_of_october_first(birth, season, expected):
    got = tf.age_at_season_start(pd.Series([pd.Timestamp(birth)]), season).iloc[0]
    assert got == pytest.approx(expected, abs=0.002)  # 365.25-day year: leap days cost < 0.002 y


def test_exact_age_matches_calendar_age_within_a_hundredth_of_a_year():
    rng = random.Random(3)
    for _ in range(500):
        birth = pd.Timestamp("1975-01-01") + pd.Timedelta(days=rng.randrange(0, 365 * 30))
        season_year = rng.randrange(2015, 2026)
        oct1 = pd.Timestamp(year=season_year, month=10, day=1)
        got = tf.age_at_season_start(pd.Series([birth]), f"{season_year}-{(season_year + 1) % 100:02d}").iloc[0]
        whole = oct1.year - birth.year - ((oct1.month, oct1.day) < (birth.month, birth.day))
        assert whole <= got < whole + 1.0 + 0.01  # never off by a whole year


def test_bio_age_uses_birthdate_when_available_and_never_touches_bio_stats_then():
    logs = _bio_logs()
    stats = {S: tf.bio_stats_frame(bio_stats_payload([{"id": 1, "age": 99}]))}
    bio, _ = tf.build_player_season_bio(logs, {1: pd.Timestamp("1990-06-15"), 2: pd.Timestamp("1990-06-15"),
                                               3: pd.Timestamp("1990-06-15")}, stats)
    age = bio.set_index(["season", "player_id"]).loc[(S, 1), "age_at_season_start"]
    assert age == pytest.approx((pd.Timestamp("2018-10-01") - pd.Timestamp("1990-06-15")).days / 365.25)


def test_bio_age_falls_back_to_integer_season_age_with_documented_offset():
    logs = _bio_logs()
    stats = {S: tf.bio_stats_frame(bio_stats_payload([{"id": 1, "age": 27}, {"id": 2, "age": 30}])),
             "2017-18": tf.bio_stats_frame(bio_stats_payload([{"id": 3, "age": 24}]))}
    bio, rep = tf.build_player_season_bio(logs, {}, stats)
    b = bio.set_index(["season", "player_id"])["age_at_season_start"]
    assert b[(S, 1)] == pytest.approx(27 - tf.AGE_OCT1_OFFSET)
    assert b[(S, 2)] == pytest.approx(30 - tf.AGE_OCT1_OFFSET)
    assert b[("2017-18", 3)] == pytest.approx(24 - tf.AGE_OCT1_OFFSET)
    assert 0.24 < tf.AGE_OCT1_OFFSET < 0.25 and rep.dropped == {}


def test_fallback_error_bound_is_half_a_year_against_simulated_source():
    """The source's AGE is floor(age on June 30 of the season's END year); verify our estimate's bound."""
    rng = random.Random(11)
    worst = 0.0
    for _ in range(2000):
        birth = pd.Timestamp("1975-01-01") + pd.Timedelta(days=rng.randrange(0, 365 * 30))
        start = rng.randrange(2015, 2026)
        jun30 = pd.Timestamp(year=start + 1, month=6, day=30)
        source_age = (jun30.year - birth.year) - ((jun30.month, jun30.day) < (birth.month, birth.day))
        season = f"{start}-{(start + 1) % 100:02d}"
        exact = tf.age_at_season_start(pd.Series([birth]), season).iloc[0]
        estimate = source_age - tf.AGE_OCT1_OFFSET
        worst = max(worst, abs(estimate - exact))
    assert worst <= 0.5 + 0.01


def test_bio_age_borrows_from_neighbouring_season_when_this_season_has_no_bio_row():
    logs = _bio_logs()  # player 1 plays in 2018-19
    stats = {"2017-18": tf.bio_stats_frame(bio_stats_payload([{"id": 1, "age": 26}]))}
    bio, _ = tf.build_player_season_bio(logs, {2: pd.Timestamp("1990-01-01"), 3: pd.Timestamp("1990-01-01")}, stats)
    age = bio.set_index(["season", "player_id"]).loc[(S, 1), "age_at_season_start"]
    assert age == pytest.approx(26 + 1 - tf.AGE_OCT1_OFFSET)  # one season later = one year older


def test_bio_row_dropped_and_counted_when_no_age_source_exists():
    logs = _bio_logs()
    bio, rep = tf.build_player_season_bio(logs, {1: pd.Timestamp("1990-01-01")}, {})
    assert set(bio["player_id"]) == {1}
    assert rep.dropped == {"no_age_source": 2} and rep.rows_in == 3 and rep.rows_out == 1


def test_bio_with_empty_logs_is_valid_and_empty():
    bio, _ = tf.build_player_season_bio(pd.DataFrame(columns=list(TABLES["game_logs"].columns)))
    assert bio.empty
    validate_table(bio, "player_season_bio")


def test_bio_stats_frame_handles_missing_age_and_junk_draft():
    f = tf.bio_stats_frame(bio_stats_payload([{"id": 1, "age": None, "dy": "Undrafted", "dr": "Undrafted", "dn": "Undrafted"}]))
    assert pd.isna(f["bio_age"].iloc[0]) and pd.isna(f["draft_year"].iloc[0])


# ============================================================ consistency

def _consistent_pair():
    logs, _ = gl([player_row(player_id=1, minutes=48.0, pts=None), player_row(player_id=2, minutes=48.0)])
    logs["min"] = [120.0, 120.0]  # two players carrying a full 240 team minutes
    tgs, _ = tg([team_row(pts=int(logs["pts"].sum()), matchup="GSW vs. OKC"),
                 team_row(team_id=1610612760, abbr="OKC", matchup="OKC @ GSW", pts=100)])
    return logs, tgs


def test_consistency_clean_case():
    logs, tgs = _consistent_pair()
    rep = tf.check_consistency(logs, tgs)
    assert rep["orphan_player_rows"] == 0 and rep["team_points_mismatch"] == 0
    assert rep["team_minutes_off_grid"] == 0
    assert rep["team_games_without_players"] == 1  # OKC has no player rows in this tiny fixture


def test_consistency_detects_orphans_points_and_minutes():
    logs, tgs = _consistent_pair()
    orphan = logs.copy()
    orphan["team_id"] = 999
    assert tf.check_consistency(orphan, tgs)["orphan_player_rows"] == 2
    bad_pts = tgs.copy()
    bad_pts.loc[bad_pts.team_id == GSW, "pts_for"] += 3
    assert tf.check_consistency(logs, bad_pts)["team_points_mismatch"] == 1
    short = logs.copy()
    short["min"] = [100.0, 100.0]
    assert tf.check_consistency(short, tgs)["team_minutes_off_grid"] == 1


def test_consistency_overtime_minutes_are_on_grid():
    logs, tgs = _consistent_pair()
    logs["min"] = [132.5, 132.5]  # 265 = one overtime
    assert tf.check_consistency(logs, tgs)["team_minutes_off_grid"] == 0


# ============================================================ DropReport

def test_drop_report_merge_and_dict():
    a = tf.DropReport(10, 8, {"x": 2})
    b = tf.DropReport(5, 4, {"x": 1, "y": 1})
    m = a.merge(b)
    assert m.as_dict() == {"rows_in": 15, "rows_out": 12, "dropped": {"x": 3, "y": 1}}
    assert a.dropped == {"x": 2}  # merge does not mutate
    a.add("z", 0)
    assert "z" not in a.dropped
