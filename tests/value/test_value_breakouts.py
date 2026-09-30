"""Breakout watchlist: calibration, model choice, watchlist assembly and the CLI."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from src.backtest.breakouts import BreakoutConfig  # noqa: E402
from src.value import breakouts as vb  # noqa: E402

CFG = BreakoutConfig()


# --------------------------------------------------------------------------- logistic fit

def test_logistic_fit_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 30_000
    x, r = rng.normal(size=n), (rng.random(n) < 0.3).astype(float)
    y = rng.random(n) < vb._sigmoid(-1.0 + 0.8 * x + 0.5 * r)
    b = vb.logistic_fit(np.column_stack([np.ones(n), x, r]), y)
    np.testing.assert_allclose(b, [-1.0, 0.8, 0.5], atol=0.08)


def test_logistic_fit_stays_finite_on_perfectly_separable_data():
    x = np.r_[np.linspace(-3, -1, 30), np.linspace(1, 3, 30)]
    y = (x > 0).astype(float)
    b = vb.logistic_fit(np.column_stack([np.ones(60), x]), y)
    assert np.isfinite(b).all() and b[1] > 0


def test_sigmoid_is_bounded_and_monotone():
    z = np.array([-1e6, -5.0, 0.0, 5.0, 1e6])
    p = vb._sigmoid(z)
    assert (p >= 0).all() and (p <= 1).all() and np.all(np.diff(p) >= 0) and p[2] == 0.5


# --------------------------------------------------------------------------- calibration

def calibration_frame(n=2000, seed=0, signal=True):
    rng = np.random.default_rng(seed)
    score = rng.normal(0, 3, n)
    rookie = rng.random(n) < 0.25
    p = vb._sigmoid(-1.7 + (0.25 * score if signal else 0.0) + 0.6 * rookie)
    breakout = rng.random(n) < p
    useful = breakout & (rng.random(n) < 0.6)
    return pd.DataFrame({"season": rng.choice(["2020-21", "2021-22", "2022-23"], n), "played": True, "young": True,
                         "score": score, "is_rookie": rookie, "breakout": breakout, "useful": useful})


def test_calibration_is_monotone_in_the_score_and_higher_for_rookies():
    c = vb.fit_calibration(calibration_frame(), "baseline_offseason")
    s = np.array([-4.0, 0.0, 4.0, 8.0])
    for target in vb.TARGETS:
        p = c.probability(s, False, target)
        assert np.all(np.diff(p) > 0) and (p > 0).all() and (p < 1).all()
    assert (c.probability(s, True, "breakout") > c.probability(s, False, "breakout")).all()
    assert c.breakout.coef_score == pytest.approx(0.25, abs=0.06)
    assert (c.useful.base_rate < c.breakout.base_rate) and c.breakout.n == 2000
    assert c.seasons == ["2020-21", "2021-22", "2022-23"]


def test_a_calibration_without_signal_has_a_flat_slope():
    c = vb.fit_calibration(calibration_frame(signal=False), "m")
    assert abs(c.breakout.coef_score) < 0.05
    assert abs(float(c.probability(6.0, False)) - float(c.probability(-6.0, False))) < 0.15


def test_probability_rejects_an_unknown_target():
    c = vb.fit_calibration(calibration_frame(), "m")
    with pytest.raises(ValueError, match="target"):
        c.probability(1.0, False, "nope")


def test_calibration_needs_enough_rows_and_both_outcomes():
    with pytest.raises(ValueError, match="too few"):
        vb.fit_calibration(calibration_frame(n=20), "m")
    f = calibration_frame()
    f["useful"] = False
    with pytest.raises(ValueError, match="one outcome class"):
        vb.fit_calibration(f, "m")


def test_only_young_players_who_played_enter_the_calibration():
    f = calibration_frame()
    f.loc[:999, "young"] = False
    assert vb.fit_calibration(f, "m").breakout.n == 1000
    f2 = calibration_frame()
    f2.loc[:499, "played"] = False
    assert vb.fit_calibration(f2, "m").breakout.n == 1500


def test_calibration_round_trips_through_disk_and_ignores_old_or_missing_files(tmp_path):
    c = vb.fit_calibration(calibration_frame(), "baseline_offseason")
    path = c.save(tmp_path)
    assert path == vb.calibration_path("baseline_offseason", tmp_path)
    back = vb.load_calibration("baseline_offseason", tmp_path)
    assert back == c
    np.testing.assert_allclose(back.probability(2.0, True, "useful"), c.probability(2.0, True, "useful"))
    assert vb.load_calibration("baseline_summer_league", tmp_path) is None
    path.write_text(json.dumps({"model": "baseline_offseason", "intercept": -1.0, "coef_score": 0.1}), encoding="utf-8")
    assert vb.load_calibration("baseline_offseason", tmp_path) is None            # an older shape: absent, never guessed


# --------------------------------------------------------------------------- model choice

def test_the_model_follows_the_evidence_that_exists():
    ev = "2026-27"
    sl = pd.DataFrame({"event_season": [ev], "context": ["summer_league"]})
    both = pd.DataFrame({"event_season": [ev, ev], "context": ["summer_league", "preseason"]})
    last_year_pre = pd.DataFrame({"event_season": [ev, "2025-26"], "context": ["summer_league", "preseason"]})
    assert vb.choose_model(sl, ev) == vb.SUMMER_ONLY_MODEL
    assert vb.choose_model(both, ev) == vb.FULL_MODEL
    assert vb.choose_model(last_year_pre, ev) == vb.SUMMER_ONLY_MODEL            # last year's preseason is not this year's
    assert vb.choose_model(None, ev) == vb.SUMMER_ONLY_MODEL and vb.choose_model(sl.iloc[0:0], ev) == vb.SUMMER_ONLY_MODEL


# --------------------------------------------------------------------------- watchlist assembly

def parts():
    proj = pd.DataFrame({
        "player_id": [1, 2, 3, 4, 5, 6],
        "age": [20.0, 21.0, 22.5, 19.5, 27.0, 23.0],
        "is_rookie": [True, False, False, True, False, False],
        "proj_fppg": [24.0, 20.0, 18.0, 15.0, 30.0, 12.0],
        "proj_gp": [70.0, 65.0, 60.0, 50.0, 75.0, 40.0],
        "proj_total_fp": [1680.0, 1300.0, 1080.0, 750.0, 2250.0, 480.0],
        "offseason_adj": [4.0, 2.0, 0.0, 6.0, 5.0, -1.5],
        "n_hist_seasons": [0, 2, 3, 0, 6, 1],
        "sl_gp": [5, np.nan, 4, 5, np.nan, np.nan], "sl_mpg": [28.0, np.nan, 20.0, 31.0, np.nan, np.nan],
        "sl_fp36": [40.0, np.nan, 30.0, 33.0, np.nan, np.nan], "sl_z": [1.5, np.nan, 0.2, 0.9, np.nan, np.nan],
        "pre_gp": [3, 4, np.nan, 3, 4, 2], "pre_mpg": [24.0, 22.0, np.nan, 12.0, 30.0, 8.0],
        "pre_fp36": [35.0, 28.0, np.nan, 20.0, 40.0, 10.0], "pre_z": [0.8, 0.3, np.nan, -0.2, 1.1, -0.9],
    })
    board = pd.DataFrame({
        "player_id": [1, 2, 3, 4, 5, 6], "rank": [40, 90, 130, 220, 12, 400],
        "name": ["Rook A", "Second B", "Third C", "Rook D", "Vet E", "Fringe F"],
        "position": ["G", "F", "C", "G", "F", "C"], "vorp": [900.0, 500.0, 200.0, -100.0, 1800.0, -800.0],
        "tier": [3, 4, 5, 6, 1, 8], "adp": [150.0, 60.0, np.nan, np.nan, 10.0, np.nan],
        "adp_gap": [110.0, -30.0, np.nan, np.nan, -2.0, np.nan]})
    players = pd.DataFrame({"player_id": [1, 2, 3, 4, 5, 6], "draft_number": [5, 22, 41, 58, 3, np.nan]})
    return proj, board, players


def calib():
    return vb.fit_calibration(calibration_frame(), "baseline_offseason")


# Only three players have an ADP in this fixture, so "worse than rank 2" plays the role of production's "worse than 100".
RADAR = BreakoutConfig(radar_rank=2)


def test_watchlist_lists_young_underpriced_players_with_positive_evidence():
    proj, board, players = parts()
    w = vb.build_watchlist(proj, board, players, cfg=RADAR, calibration=calib())
    # excluded: Vet E (27), Second B (ADP rank 2: priced), Third C (uplift exactly 0), Fringe F (uplift < 0)
    assert set(w["name"]) == {"Rook A", "Rook D"}
    assert list(w["watch_rank"]) == [1, 2] and w["breakout_prob"].notna().all() and w["useful_prob"].notna().all()


def test_the_radar_threshold_is_an_adp_rank_not_an_adp_value():
    proj, board, players = parts()
    strict = vb.build_watchlist(proj, board, players, cfg=BreakoutConfig(radar_rank=100), calibration=calib())
    assert set(strict["name"]) == {"Rook D"}                # with the production threshold every listed ADP is "priced"
    w = vb.build_watchlist(proj, board, players, cfg=RADAR, calibration=calib())
    assert "Rook A" in set(w["name"])                       # ADP 150 is rank 3 of 3 listed: worse than rank 2 -> under the radar
    assert "Second B" not in set(w["name"])                 # ADP 60 is rank 2: priced


def test_filters_can_be_switched_off():
    proj, board, players = parts()
    everyone = vb.build_watchlist(proj, board, players, young_only=False, under_radar_only=False, min_uplift=-100.0)
    assert set(everyone["name"]) == set(board["name"])
    old_ok = vb.build_watchlist(proj, board, players, young_only=False, under_radar_only=False)
    assert "Vet E" in set(old_ok["name"]) and "Fringe F" not in set(old_ok["name"])
    high = vb.build_watchlist(proj, board, players, young_only=False, under_radar_only=False, min_uplift=4.5)
    assert set(high["name"]) == {"Rook D", "Vet E"}


def test_default_sort_is_the_calibrated_probability_and_falls_back_to_uplift():
    proj, board, players = parts()
    w = vb.build_watchlist(proj, board, players, calibration=calib(), under_radar_only=False, young_only=True)
    assert w["useful_prob"].is_monotonic_decreasing
    no_cal = vb.build_watchlist(proj, board, players, under_radar_only=False)
    assert no_cal["useful_prob"].isna().all() and no_cal["breakout_prob"].isna().all()
    assert no_cal["uplift"].is_monotonic_decreasing


def test_raw_performance_sorts_ignore_the_model():
    proj, board, players = parts()
    by_sl = vb.build_watchlist(proj, board, players, under_radar_only=False, young_only=False, sort_by="sl_z")
    listed = by_sl.dropna(subset=["sl_z"])
    assert list(listed["name"])[:2] == ["Rook A", "Rook D"] and by_sl["sl_z"].dropna().is_monotonic_decreasing
    by_pre = vb.build_watchlist(proj, board, players, under_radar_only=False, young_only=False, sort_by="pre_z")
    assert by_pre["name"].iloc[0] == "Vet E"
    by_uplift = vb.build_watchlist(proj, board, players, under_radar_only=False, young_only=False, sort_by="uplift")
    assert by_uplift["uplift"].is_monotonic_decreasing and by_uplift["name"].iloc[0] == "Rook D"
    with pytest.raises(ValueError, match="sort_by"):
        vb.build_watchlist(proj, board, players, sort_by="nope")


def test_watchlist_row_contents():
    proj, board, players = parts()
    w = vb.build_watchlist(proj, board, players, cfg=RADAR, calibration=calib()).set_index("name")
    a = w.loc["Rook A"]
    assert a["base_fppg"] == pytest.approx(20.0) and a["layer_fppg"] == 24.0 and a["uplift"] == 4.0
    assert a["draft_pick"] == 5 and a["board_rank"] == 40 and a["adp"] == 150.0
    assert a["evidence"] == "SL 5g 28mpg z+1.5 | pre 3g 24mpg z+0.8"
    assert w.at["Rook D", "evidence"] == "SL 5g 31mpg z+0.9 | pre 3g 12mpg z-0.2"
    assert bool(w.at["Rook D", "is_rookie"]) and pd.isna(w.at["Rook D", "adp"])


def test_a_player_with_no_offseason_line_says_so():
    proj, board, players = parts()
    proj.loc[proj["player_id"] == 4, ["sl_gp", "pre_gp"]] = np.nan
    w = vb.build_watchlist(proj, board, players, cfg=RADAR).set_index("name")
    assert w.at["Rook D", "evidence"] == "no offseason line"


def test_team_and_team_change_come_from_the_roster_snapshot():
    proj, board, players = parts()
    roster = pd.DataFrame({"player_id": [1, 4], "team_abbr": ["MIL", "CLE"], "team_id": [10, 20]})
    last_team = pd.Series({1: 10, 4: 99})
    w = vb.build_watchlist(proj, board, players, cfg=RADAR, roster=roster, last_team=last_team).set_index("name")
    assert w.at["Rook A", "team"] == "MIL" and not w.at["Rook A", "changed_team"]
    assert w.at["Rook D", "team"] == "CLE" and w.at["Rook D", "changed_team"]
    none = vb.build_watchlist(proj, board, players, cfg=RADAR).set_index("name")
    assert none["team"].isna().all() and not none["changed_team"].any()


def test_missing_offseason_columns_are_tolerated():
    proj, board, players = parts()
    slim = proj[["player_id", "age", "is_rookie", "proj_fppg", "proj_gp", "proj_total_fp"]]
    w = vb.build_watchlist(slim, board, players)
    assert w.empty                                           # no offseason_adj -> no positive evidence -> no flags


def test_board_without_adp_columns_treats_everyone_as_unpriced():
    proj, board, players = parts()
    w = vb.build_watchlist(proj, board.drop(columns=["adp", "adp_gap"]), players, young_only=True)
    assert {"Rook A", "Second B", "Rook D"} <= set(w["name"])


def test_last_season_team_takes_each_players_most_recent_team():
    bio = pd.DataFrame({"season": ["2023-24", "2024-25", "2024-25", "2023-24"], "player_id": [1, 1, 2, 3],
                        "age_at_season_start": [20.0, 21.0, 22.0, 23.0], "team_id": [10.0, 11.0, np.nan, 12.0]})
    t = vb.last_season_team(bio)
    assert t.at[1] == 11 and t.at[3] == 12 and 2 not in t.index


# --------------------------------------------------------------------------- CLI

@pytest.fixture(scope="module")
def store(tmp_path_factory):
    from model_testkit import make_league
    from offseason_testkit import make_offseason

    from src.contracts import HISTORY_TABLES
    from src.ingest import nba_incoming as ni
    from src.ingest import nba_offseason as no
    from src.store import write_table

    base = tmp_path_factory.mktemp("data")
    tables = make_league()
    for name in HISTORY_TABLES:
        write_table(tables[name], name, base)
    logs, tg = make_offseason(tables, "signal", seed=2)
    no.write_offseason_table(logs, "offseason_logs", base)
    no.write_offseason_table(tg, "offseason_team_games", base)
    gl = tables["game_logs"]
    pids = gl[gl["season"] == "2018-19"]["player_id"].unique()[:20]              # players the 2019-20 projection includes
    snap = pd.DataFrame({"snapshot_date": pd.Timestamp("2019-09-20"), "season": "2019-20", "player_id": pids,
                         "player_name": "x", "team_id": 1610612738, "team_abbr": "BOS",
                         "position": "G", "draft_year": pd.array([None] * 20, dtype="Int64"),
                         "draft_number": pd.array([None] * 20, dtype="Int64")})
    ni.write_roster_snapshot(snap, base)
    return base


def test_cli_prints_and_writes_the_watchlist(store, tmp_path, capsys):
    out = tmp_path / "wl.csv"
    code = vb.main(["--season", "2019-20", "--data-dir", str(store), "--out", str(out), "--top", "5",
                    "--include-priced", "--all-ages", "--model", "baseline_offseason", "--min-uplift", "-100"])
    assert code == 0
    text = capsys.readouterr().out
    assert "breakout watchlist for 2019-20 (baseline_offseason)" in text and "no calibration" in text.lower()
    wl = pd.read_csv(out)
    assert {"watch_rank", "name", "uplift", "evidence", "player_id", "useful_prob"} <= set(wl.columns)
    assert len(wl) > 50 and wl["team"].notna().sum() == 20 and (wl["team"].dropna() == "BOS").all()


def test_cli_uses_a_stored_calibration_and_picks_the_model_from_the_data(store, tmp_path, capsys):
    vb.fit_calibration(calibration_frame(), "baseline_offseason").save(store)
    out = tmp_path / "wl.csv"
    assert vb.main(["--season", "2019-20", "--data-dir", str(store), "--out", str(out), "--all-ages", "--include-priced",
                    "--min-uplift", "-100"]) == 0
    text = capsys.readouterr().out
    assert "(baseline_offseason)" in text                              # the synthetic data has a 2019-20 preseason
    assert pd.read_csv(out)["useful_prob"].notna().all()


def test_cli_accepts_a_plain_adp_file(store, tmp_path, capsys):
    tables = pd.read_parquet(store / "processed" / "game_logs.parquet")
    pid = int(tables["player_id"].iloc[0])
    adp = tmp_path / "adp.csv"
    pd.DataFrame({"player_id": [pid], "adp": [12.0]}).to_csv(adp, index=False)
    out = tmp_path / "wl.csv"
    assert vb.main(["--season", "2019-20", "--data-dir", str(store), "--adp", str(adp), "--out", str(out), "--all-ages",
                    "--include-priced", "--min-uplift", "-100"]) == 0
    wl = pd.read_csv(out)
    assert wl["adp"].notna().sum() <= 1


def test_cli_without_history_reports_an_error(tmp_path, capsys):
    assert vb.main(["--season", "2019-20", "--data-dir", str(tmp_path / "empty")]) == 2
    assert "error" in capsys.readouterr().out.lower()


def test_cli_without_offseason_tables_warns_and_returns_an_empty_list(tmp_path, capsys):
    from model_testkit import make_league

    from src.contracts import HISTORY_TABLES
    from src.store import write_table

    base = tmp_path / "d"
    tables = make_league()
    for name in HISTORY_TABLES:
        write_table(tables[name], name, base)
    assert vb.main(["--season", "2019-20", "--data-dir", str(base)]) == 0
    text = capsys.readouterr().out
    assert "no offseason tables" in text.lower() and "0 players" in text


# --------------------------------------------------------------------------- select / assemble

def test_select_matches_build_with_selection():
    proj, board, players = parts()
    full = vb.build_watchlist(proj, board, players, cfg=RADAR, calibration=calib(), select=False)
    assert len(full) == len(board) and {"young", "under_radar"} <= set(full.columns) and "watch_rank" not in full.columns
    for kw in ({}, {"young_only": False}, {"under_radar_only": False}, {"min_uplift": 1.0, "sort_by": "uplift"}):
        built = vb.build_watchlist(proj, board, players, cfg=RADAR, calibration=calib(), **kw)
        selected = vb.select_watchlist(full, **kw)
        pd.testing.assert_frame_equal(built.reset_index(drop=True), selected.reset_index(drop=True))


def test_select_needs_no_projection_and_validates_the_sort():
    proj, board, players = parts()
    full = vb.build_watchlist(proj, board, players, cfg=RADAR, select=False)
    assert vb.select_watchlist(full, young_only=False, under_radar_only=False)["uplift"].is_monotonic_decreasing
    with pytest.raises(ValueError, match="sort_by"):
        vb.select_watchlist(full, sort_by="nope")


def test_assemble_returns_notes_for_every_missing_piece(store, tmp_path):
    import shutil

    base = tmp_path / "bare"
    shutil.copytree(store, base)
    for junk in (base / "processed").glob("roster_snapshots.parquet"):
        junk.unlink()
    for junk in (base / "processed").glob("breakout_calibration_*.json"):
        junk.unlink()
    r = vb.assemble_watchlist("2019-20", data_dir=base, young_only=False, under_radar_only=False, min_uplift=-100.0)
    text = " ".join(r.notes)
    assert "No ADP file" in text and "No roster snapshot" in text and "No calibration" in text
    assert r.model == vb.FULL_MODEL and not r.calibrated and len(r.watchlist) > 50
    assert "preseason games exist" in text


def test_assemble_with_select_false_keeps_everyone_flagged(store):
    r = vb.assemble_watchlist("2019-20", data_dir=store, select=False)
    assert {"young", "under_radar", "useful_prob"} <= set(r.watchlist.columns) and len(r.watchlist) > 100


def test_assemble_raises_when_history_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        vb.assemble_watchlist("2019-20", data_dir=tmp_path / "nothing")
