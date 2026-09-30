"""src.ingest.quality: pure computations on hand-built frames, including deliberately broken ones."""
import pandas as pd
import pytest

from src.ingest import quality as q

S1, S2 = "2018-19", "2019-20"
BOS, NYK = 1610612738, 1610612752


def gl_row(season=S1, game_id="0021800001", date="2018-10-16", player_id=1, name="Al Pha", team_id=BOS,
          abbr="BOS", matchup="BOS vs. NYK", minutes=30.0, fgm=5, fga=10, fg3m=1, fg3a=3, ftm=2, fta=2,
          oreb=1, dreb=4, ast=3, stl=1, blk=0, tov=2, pf=2, plus_minus=5.0, reb=None, pts=None):
    reb = oreb + dreb if reb is None else reb
    pts = 2 * fgm + fg3m + ftm if pts is None else pts
    return {"season": season, "game_id": game_id, "game_date": pd.Timestamp(date), "player_id": player_id,
            "player_name": name, "team_id": team_id, "team_abbr": abbr, "matchup": matchup, "min": minutes,
            "fgm": fgm, "fga": fga, "fg3m": fg3m, "fg3a": fg3a, "ftm": ftm, "fta": fta, "oreb": oreb, "dreb": dreb,
            "reb": reb, "ast": ast, "stl": stl, "blk": blk, "tov": tov, "pf": pf, "pts": pts, "plus_minus": plus_minus}


def tg_row(season=S1, game_id="0021800001", date="2018-10-16", team_id=BOS, abbr="BOS", is_home=True,
          pts_for=100, pts_against=95):
    return {"season": season, "game_id": game_id, "game_date": pd.Timestamp(date), "team_id": team_id,
            "team_abbr": abbr, "is_home": is_home, "pts_for": pts_for, "pts_against": pts_against}


def small_league():
    """One game, two teams, one player each: internally consistent (points and minutes add up)."""
    gl = pd.DataFrame([
        gl_row(player_id=1, name="Al Pha", team_id=BOS, abbr="BOS", matchup="BOS vs. NYK", minutes=240.0, pts=100, fgm=40, fg3m=10, ftm=10, oreb=5, dreb=35, ast=20, stl=5, blk=5, tov=10, pf=15, fga=90, fg3a=25, fta=12),
        gl_row(player_id=2, name="Bee Ta", team_id=NYK, abbr="NYK", matchup="NYK @ BOS", minutes=240.0, pts=95, fgm=38, fg3m=9, ftm=10, oreb=5, dreb=33, ast=18, stl=4, blk=4, tov=9, pf=14, fga=88, fg3a=22, fta=12),
    ])
    tg = pd.DataFrame([tg_row(team_id=BOS, abbr="BOS", is_home=True, pts_for=100, pts_against=95),
                       tg_row(team_id=NYK, abbr="NYK", is_home=False, pts_for=95, pts_against=100)])
    pl = pd.DataFrame([{"player_id": 1, "player_name": "Al Pha", "birthdate": pd.Timestamp("1990-01-01"),
                        "position": "G", "height_in": 74.0, "weight_lb": 200.0, "draft_year": 2012,
                        "draft_round": 1, "draft_number": 5, "from_year": 2012, "to_year": 2025},
                       {"player_id": 2, "player_name": "Bee Ta", "birthdate": pd.NaT, "position": None,
                        "height_in": 80.0, "weight_lb": 220.0, "draft_year": pd.NA, "draft_round": pd.NA,
                        "draft_number": pd.NA, "from_year": 2015, "to_year": 2025}])
    for c in ("draft_year", "draft_round", "draft_number", "from_year", "to_year"):
        pl[c] = pl[c].astype("Int64")
    bio = pd.DataFrame([{"season": S1, "player_id": 1, "age_at_season_start": 28.75, "team_id": BOS},
                        {"season": S1, "player_id": 2, "age_at_season_start": 23.5, "team_id": NYK}])
    return {"game_logs": gl, "team_games": tg, "players": pl, "player_season_bio": bio}


# ============================================================ inventory / coverage

def test_inventory_reports_row_and_season_counts():
    t = small_league()
    inv = q.inventory(t).set_index("table")
    assert inv.loc["game_logs", "rows"] == 2 and inv.loc["players", "seasons"] == "n/a (static)"
    assert inv.loc["game_logs", "seasons"] == f"{S1} .. {S1} (1)"


def test_season_coverage_matches_expected_82_games():
    t = small_league()
    cov = q.season_coverage(t["game_logs"], t["team_games"])
    r = cov.iloc[0]
    assert r["games"] == 1 and r["teams"] == 2 and r["games/team min"] == r["games/team max"] == 1
    assert r["check"] == "MISMATCH: 2 teams"   # 1 game != 82 expected: correctly flagged


def test_season_coverage_covid_season_is_not_flagged_mismatch():
    tg = pd.DataFrame([tg_row(season=S2, team_id=BOS, is_home=True), tg_row(season=S2, team_id=NYK, is_home=False)])
    gl = pd.DataFrame([gl_row(season=S2, team_id=BOS), gl_row(season=S2, team_id=NYK, player_id=2)])
    cov = q.season_coverage(gl, tg)
    assert cov.iloc[0]["check"] == "n/a (COVID, see below)"


def test_season_coverage_shortened_2020_21_uses_72():
    s = "2020-21"
    rows_tg, rows_gl = [], []
    for i in range(72):
        gid = f"0022000{i:03d}"
        rows_tg += [tg_row(season=s, game_id=gid, team_id=BOS), tg_row(season=s, game_id=gid, team_id=NYK, is_home=False)]
        rows_gl += [gl_row(season=s, game_id=gid, team_id=BOS), gl_row(season=s, game_id=gid, team_id=NYK, player_id=2)]
    cov = q.season_coverage(pd.DataFrame(rows_gl), pd.DataFrame(rows_tg))
    assert cov.iloc[0]["check"] == "ok" and cov.iloc[0]["expected"] == 72


def test_covid_summary_empty_when_no_2019_20_data():
    assert q.covid_summary(small_league()["team_games"]) == {}


def test_covid_summary_computes_from_real_shaped_data():
    rows = []
    for i, date in enumerate(["2020-03-01", "2020-03-05", "2020-08-01"]):
        gid = f"0021900{i:03d}"
        rows += [tg_row(season=S2, game_id=gid, date=date, team_id=BOS), tg_row(season=S2, game_id=gid, date=date, team_id=NYK, is_home=False)]
    c = q.covid_summary(pd.DataFrame(rows))
    assert c["games"] == 3 and c["pre_suspension_min"] == c["pre_suspension_max"] == 2
    assert c["teams_playing_after_restart"] == 2 and c["games_after_restart"] == 1


# ============================================================ duplicates / ids

def test_duplicate_keys_zero_on_clean_data():
    t = small_league()
    assert q.duplicate_keys(t) == {"game_logs": 0, "team_games": 0, "players": 0, "player_season_bio": 0}


def test_duplicate_keys_detects_a_repeated_key():
    t = small_league()
    t["game_logs"] = pd.concat([t["game_logs"], t["game_logs"].iloc[[0]]], ignore_index=True)
    assert q.duplicate_keys(t)["game_logs"] == 1


def test_id_checks_clean_case():
    t = small_league()
    ids = q.id_checks(t)
    assert ids["players_unique_in_players"] is True
    assert ids["ids_with_multiple_names"] == 0 and ids["names_with_multiple_ids"] == {}
    assert ids["team_ids_with_multiple_abbrs"] == {} and ids["abbrs_with_multiple_team_ids"] == {}
    assert ids["games_without_exactly_two_teams"] == 0 and ids["neutral_site_games"] == 0
    assert ids["game_log_players_missing_from_players"] == 0 and ids["players_without_game_logs"] == 0


def test_id_checks_finds_name_collision_and_id_with_two_names():
    t = small_league()
    gl2 = t["game_logs"].copy()
    gl2.loc[gl2.player_id == 2, "player_name"] = "Al Pha"      # name collides with player 1
    row = gl_row(player_id=1, name="Al Pha II", game_id="0021800099")  # id 1 now has two names
    gl2 = pd.concat([gl2, pd.DataFrame([row])], ignore_index=True)
    ids = q.id_checks({**t, "game_logs": gl2})
    assert ids["ids_with_multiple_names"] == 1
    assert ids["names_with_multiple_ids"] == {"Al Pha": 2}


def test_id_checks_detects_team_abbreviation_change():
    t = small_league()
    extra = gl_row(player_id=1, team_id=BOS, abbr="XYZ", game_id="0021800050", season=S2)
    gl2 = pd.concat([t["game_logs"], pd.DataFrame([extra])], ignore_index=True)
    ids = q.id_checks({**t, "game_logs": gl2})
    assert ids["team_ids_with_multiple_abbrs"] == {BOS: ["BOS", "XYZ"]}


def test_id_checks_finds_neutral_site_game():
    t = small_league()
    tg2 = t["team_games"].copy()
    tg2["is_home"] = False   # both teams "away" -> neutral site
    ids = q.id_checks({**t, "team_games": tg2})
    assert ids["neutral_site_games"] == 1 and ids["games_with_two_home_teams"] == 0


def test_id_checks_bio_gaps():
    t = small_league()
    bio2 = t["player_season_bio"].iloc[:1]  # drop player 2's bio row
    ids = q.id_checks({**t, "player_season_bio": bio2})
    assert ids["player_seasons_without_bio"] == 1


# ============================================================ box score

def test_box_score_checks_clean_case_has_no_violations():
    t = small_league()
    b = q.box_score_checks(t["game_logs"], t["team_games"])
    assert sum(b["row_identities"].values()) == 0
    assert b["orphan_player_team_games"] == 0 and b["team_points_mismatch"] == 0 and b["team_minutes_off_grid"] == 0


def test_box_score_checks_detects_identity_violation_without_raising():
    t = small_league()
    gl2 = t["game_logs"].copy()
    gl2.loc[0, "pts"] += 3   # break pts = 2fgm+3m+ftm
    b = q.box_score_checks(gl2, t["team_games"])
    assert b["row_identities"]["pts != 2*fgm + fg3m + ftm"] == 1


def test_box_score_checks_detects_points_mismatch_and_minutes_off_grid():
    t = small_league()
    tg2 = t["team_games"].copy()
    tg2.loc[tg2.team_abbr == "BOS", "pts_for"] = 999
    b = q.box_score_checks(t["game_logs"], tg2)
    assert b["team_points_mismatch"] == 1
    assert len(b["team_points_mismatch_examples"]) == 1

    gl2 = t["game_logs"].copy()
    gl2.loc[0, "min"] = 100.0
    b2 = q.box_score_checks(gl2, t["team_games"])
    assert b2["team_minutes_off_grid"] == 1


def test_box_score_checks_orphan_and_missing_player_rows():
    t = small_league()
    gl2 = t["game_logs"].copy()
    gl2.loc[0, "team_id"] = 999999
    b = q.box_score_checks(gl2, t["team_games"])
    assert b["orphan_player_team_games"] >= 1
    b2 = q.box_score_checks(t["game_logs"].iloc[:0], t["team_games"])
    assert b2["team_games_without_player_rows"] == 2


def test_box_score_checks_overtime_is_on_grid():
    t = small_league()
    gl2 = t["game_logs"].copy()
    gl2["min"] = 265.0  # each team has one player logging all 265 (240 + one 25-minute overtime)
    b = q.box_score_checks(gl2, t["team_games"])
    assert b["team_minutes_off_grid"] == 0 and b["overtime_team_games"] == 2  # both team-games show 265


# ============================================================ minutes / outliers

def _minutes_league(bos_minutes, nyk_minutes):
    """Two teams, several players each, so team totals can be set exactly without one giant stint."""
    rows = [gl_row(player_id=1, name="Al Pha", team_id=BOS, minutes=bos_minutes[0], pts=10, fgm=4, fg3m=0, ftm=2, oreb=1, dreb=1),
            gl_row(player_id=2, name="Bee Ta", team_id=NYK, abbr="NYK", minutes=nyk_minutes[0], pts=10, fgm=4, fg3m=0, ftm=2, oreb=1, dreb=1)]
    for i, m in enumerate(bos_minutes[1:], start=3):
        rows.append(gl_row(player_id=i, name=f"P{i}", team_id=BOS, minutes=m, pts=2, fgm=1, fg3m=0, ftm=0, oreb=0, dreb=1))
    for i, m in enumerate(nyk_minutes[1:], start=10):
        rows.append(gl_row(player_id=i, name=f"P{i}", team_id=NYK, abbr="NYK", minutes=m, pts=2, fgm=1, fg3m=0, ftm=0, oreb=0, dreb=1))
    return pd.DataFrame(rows)


def test_minutes_checks_basic_stats_and_impossible_detection():
    # BOS: one player at 55 (impossible; team total 55+185=240, i.e. no overtime) + filler; NYK: normal 240
    gl = _minutes_league(bos_minutes=[55.0] + [37.0] * 5, nyk_minutes=[48.0] * 5)
    m = q.minutes_checks(gl)
    assert m["rows_over_48"] == 1
    assert m["impossible_for_game_length"] == 1   # BOS team minutes total 240 -> no overtime -> ceiling 48
    assert m["max_minutes"] == 55.0


def test_minutes_checks_overtime_raises_the_legal_ceiling():
    # BOS team total 265 (one overtime) with one player at 53: legal once overtime is accounted for
    gl = _minutes_league(bos_minutes=[53.0] + [53.0] * 3, nyk_minutes=[53.0] * 5)  # NYK totals 265 too
    m = q.minutes_checks(gl)
    assert m["rows_over_48"] >= 1                     # over the flat 48 threshold ...
    assert m["impossible_for_game_length"] == 0        # ... but legal once overtime is accounted for


def test_plausibility_outliers_flags_and_counts():
    gl = pd.DataFrame([gl_row(player_id=1, name="Al Pha", pts=65, fgm=26, fga=40, fg3m=5, ftm=8, fta=8, oreb=1, dreb=1),
                       gl_row(player_id=2, name="Bee Ta", team_id=NYK, abbr="NYK", pts=10, fgm=4, fga=9, fg3m=0, ftm=2, fta=2)])
    tg = pd.DataFrame([tg_row(team_id=BOS, is_home=True, pts_for=65, pts_against=10),
                       tg_row(team_id=NYK, abbr="NYK", is_home=False, pts_for=10, pts_against=65)])
    out = q.plausibility_outliers(gl, tg)
    table = out["table"].set_index("stat")
    assert table.loc["points", "rows"] == 1 and table.loc["points", "max"] == 65
    assert "team score" in out["top_points_games"].columns


def test_plausibility_outliers_works_without_team_games():
    t = small_league()
    out = q.plausibility_outliers(t["game_logs"], None)
    assert "team score" not in out["top_points_games"].columns


# ============================================================ missing attributes

def test_missing_attributes_reports_gaps_and_undrafted():
    t = small_league()
    m = q.missing_attributes(t)
    tbl = m["table"].set_index("column")
    assert tbl.loc["birthdate", "missing players"] == 1     # player 2
    assert tbl.loc["draft_year", "missing players"] == 1
    assert m["undrafted_players"] == 1
    assert len(m["birthdate_missing_examples"]) == 1


def test_age_fallback_calibration_matches_documented_offset():
    from src.ingest import nba_transform as tf
    frame = pd.DataFrame({"player_id": [1], "bio_age": [30], "height_in": [74.0], "weight_lb": [200.0],
                          "draft_year": pd.array([2012], dtype="Int64"), "draft_round": pd.array([1], dtype="Int64"),
                          "draft_number": pd.array([5], dtype="Int64")})
    birth = pd.Timestamp("1990-01-01")
    cal = tf.age_fallback_calibration(frame, "2019-20", {1: birth})
    exact = tf.age_at_season_start(pd.Series([birth]), "2019-20").iloc[0]
    assert cal["pairs"] == 1
    assert cal["max_abs_error"] == pytest.approx(abs((30 - tf.AGE_OCT1_OFFSET) - exact), abs=1e-3)


def test_age_calibration_summary_aggregates_across_seasons():
    report = {"seasons": {
        S1: {"age_fallback_calibration": {"pairs": 10, "floor_age_on_jun30_matches": 9, "within_half_year": 10, "mean_error": 0.1, "max_abs_error": 0.3}},
        S2: {"age_fallback_calibration": {"pairs": 5, "floor_age_on_jun30_matches": 5, "within_half_year": 5, "mean_error": -0.2, "max_abs_error": 0.4}},
        "_bio": {},
    }}
    agg = q.age_calibration_summary(report)
    assert agg["pairs"] == 15 and agg["floor_age_on_jun30_matches"] == 14 and agg["max_abs_error"] == 0.4
    assert agg["mean_error"] == pytest.approx((0.1 * 10 + -0.2 * 5) / 15)


def test_age_calibration_summary_empty_when_nothing_recorded():
    assert q.age_calibration_summary({"seasons": {S1: {}}}) == {}
    assert q.age_calibration_summary(None) == {}


# ============================================================ leaders / facts

def test_season_leaders_and_qualification_threshold():
    rows = []
    for gid in range(50):
        rows.append(gl_row(game_id=f"g{gid}", player_id=1, name="Star", pts=40, fgm=15, fg3m=2, ftm=8, oreb=2, dreb=8, ast=5, team_id=BOS))
    for gid in range(5):  # too few games to qualify despite a huge ppg
        rows.append(gl_row(game_id=f"h{gid}", player_id=2, name="SmallSample", pts=60, fgm=20, fg3m=4, ftm=16, oreb=1, dreb=9, ast=1, team_id=NYK))
    tg = pd.DataFrame([tg_row(team_id=BOS, game_id=f"g{i}") for i in range(50)]
                      + [tg_row(team_id=BOS, game_id=f"g{i}", is_home=False) for i in range(50)])
    leaders = q.season_leaders(pd.DataFrame(rows), tg, scoring={"PTS": 1, "REB": 1, "AST": 2, "STL": 4, "BLK": 4, "TO": -2,
                                                                "FGM": 2, "FGA": -1, "FTM": 1, "FTA": -1, "3PM": 1})
    assert leaders.iloc[0]["ppg leader"] == "Star"   # SmallSample didn't play half the max team games


def _season_of_82_game_record(season, abbr, opp_abbr, wins, losses):
    rows = []
    for i in range(wins + losses):
        gid = f"g{i}"
        win = i < wins
        rows.append(tg_row(season=season, game_id=gid, team_id=BOS, abbr=abbr, is_home=True,
                           pts_for=100 if win else 90, pts_against=90 if win else 100))
        rows.append(tg_row(season=season, game_id=gid, team_id=NYK, abbr=opp_abbr, is_home=False,
                           pts_for=90 if win else 100, pts_against=100 if win else 90))
    return pd.DataFrame(rows)


def test_known_facts_ok_and_fail_and_season_not_ingested():
    gl = pd.DataFrame([gl_row(season="2015-16", date="2016-04-13", name="Kobe Bryant", team_id=BOS, abbr="LAL",
                              pts=60, fga=50, fgm=22, fg3m=6, ftm=10, oreb=0, dreb=4, minutes=42.15)])
    tg = _season_of_82_game_record("2015-16", "GSW", "OPP", 73, 9)
    facts = q.known_facts(gl, tg, leaders=pd.DataFrame(columns=["season", "ppg leader", "gp", "ppg", "fppg leader", "fppg"]))
    by_fact = facts.set_index("fact")
    assert by_fact.loc["Kobe Bryant's 60-point farewell", "ok"] == "ok"
    assert by_fact.loc["2015-16 GSW record 73-9", "ok"] == "ok"
    assert by_fact.loc["2018-19 scoring title: James Harden 36.1 ppg", "ok"] == "n/a"   # season not ingested
    assert by_fact.loc["Devin Booker's 70 points", "ok"] == "n/a"                        # 2016-17 not ingested


def test_known_facts_flags_a_real_mismatch():
    gl = pd.DataFrame([gl_row(season="2015-16", date="2016-04-13", name="Kobe Bryant", abbr="LAL", pts=61, fga=50,
                              fgm=22, fg3m=6, ftm=11, oreb=0, dreb=4)])  # points off the known 60
    facts = q.known_facts(gl, pd.DataFrame(columns=["season", "team_abbr", "pts_for", "pts_against"]),
                          leaders=pd.DataFrame(columns=["season", "ppg leader", "gp", "ppg", "fppg leader", "fppg"]))
    assert facts.set_index("fact").loc["Kobe Bryant's 60-point farewell", "ok"] == "FAIL"


def test_known_facts_accent_insensitive_name_match():
    gl = pd.DataFrame([gl_row(season="2023-24", date="2024-01-26", name="Luka Doncic", abbr="DAL", pts=73, fga=24,
                              fgm=25, fg3m=6, ftm=17, oreb=0, dreb=8, minutes=44.7)])
    facts = q.known_facts(gl, pd.DataFrame(columns=["season", "team_abbr", "pts_for", "pts_against"]),
                          leaders=pd.DataFrame(columns=["season", "ppg leader", "gp", "ppg", "fppg leader", "fppg"]))
    assert facts.set_index("fact").loc["Luka Doncic's 73 points", "ok"] == "ok"  # "Doncic" matches "Dončić"


# ============================================================ end-to-end / rendering

def test_compute_quality_end_to_end_and_render_produces_all_sections():
    t = small_league()
    result = q.compute_quality(t, report={"seasons": {}})
    md = q.render_markdown(result)
    for heading in ["## 1. Inventory", "## 2. Games per team-season", "## 3. Duplicate keys", "## 4. Box-score",
                    "## 5. Minutes", "## 6. Plausibility", "## 7. Missing", "## 8. Sanity checks",
                    "## 9. Data dropped", "## 10. Coverage gaps"]:
        assert heading in md
    assert "|" in md and "game_logs" in md


def test_render_markdown_handles_empty_optional_sections_gracefully():
    t = small_league()
    t["player_season_bio"] = t["player_season_bio"].iloc[:0]
    result = q.compute_quality(t, report={})
    md = q.render_markdown(result)
    assert "Data-quality report" in md


def test_md_table_escapes_pipe_and_formats_types():
    df = pd.DataFrame([{"a": "x|y", "b": 3, "c": 1.5, "d": None, "e": pd.Timestamp("2020-01-01")}])
    out = q._md_table(df)
    assert "x\\|y" in out and "1.50" in out and "2020-01-01" in out
    assert q._md_table(pd.DataFrame()) == "_(none)_\n"
