"""Return-from-injury study (ADR 0021): profile features, cohort rules, matching, no look-ahead, report smoke test."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from model_testkit import box_line, make_league  # noqa: E402
from src.backtest import return_report as R  # noqa: E402
from src.backtest.leakage import scrambled_future  # noqa: E402
from src.contracts import HISTORY_TABLES, History  # noqa: E402

TEAM = 1610612737
L = 40


def _tables(patterns: dict[tuple[int, str], np.ndarray], *, seasons=("2018-19", "2019-20"), team_for=None) -> dict[str, pd.DataFrame]:
    """One team, ``L`` games per season; ``patterns[(pid, season)]`` is a bool array of games the player appeared in."""
    tg, gl = [], []
    for season in seasons:
        y = int(season[:4])
        for g in range(L):
            tg.append(dict(season=season, game_id=f"{season}-{g:03d}", game_date=pd.Timestamp(y, 10, 20) + pd.Timedelta(days=2 * g),
                           team_id=TEAM, team_abbr="AAA", is_home=True, pts_for=100, pts_against=99))
    tg = pd.DataFrame(tg)
    for (pid, season), present in patterns.items():
        assert len(present) == L
        team = (team_for or {}).get((pid, season), TEAM)
        for g in np.flatnonzero(present):
            row = tg[(tg["season"] == season)].iloc[g]
            gl.append(dict(season=season, game_id=row["game_id"], game_date=row["game_date"], player_id=pid, player_name=f"P{pid}",
                           team_id=team, team_abbr="AAA", matchup="AAA vs. X", plus_minus=0.0, **box_line(28.0)))
    pids = sorted({p for p, _ in patterns})
    players = pd.DataFrame({"player_id": pids, "player_name": [f"P{p}" for p in pids], "birthdate": pd.Timestamp("1995-01-01"),
                            "position": "F", "height_in": 78.0, "weight_lb": 210.0, "draft_year": 2015, "draft_round": 1,
                            "draft_number": 5, "from_year": 2015, "to_year": 2030})
    bio = pd.DataFrame([dict(season=s, player_id=p, age_at_season_start=24.0, team_id=TEAM) for (p, s) in patterns])
    return {"game_logs": pd.DataFrame(gl), "team_games": tg, "players": players, "player_season_bio": bio}


def _pat(*spans: tuple[int, int, bool]) -> np.ndarray:
    """Presence array of length L from (start, stop, present) spans, absent elsewhere."""
    a = np.zeros(L, bool)
    for s, e, v in spans:
        a[s:e] = v
    return a


def test_runs_finds_every_missed_block():
    starts, lens = R._runs(np.array([1, 1, 0, 0, 1, 0, 1, 1, 1], bool))
    assert starts.tolist() == [0, 4, 6] and lens.tolist() == [2, 1, 3]
    assert R._runs(np.zeros(5, bool))[0].size == 0 and R._runs(np.ones(0, bool))[0].size == 0


def test_profile_lead_block_and_tail_health_use_team_game_counts():
    # Misses games 0..19 (a 20-game block = half the 40-game season), plays 18 of the remaining 20.
    present = _pat((20, 40, True))
    present[[25, 33]] = False
    h = History.until(_tables({(1, "2018-19"): present, (1, "2019-20"): _pat((0, 40, True))}), "2019-20")
    row = R.prior_features(h).iloc[0]
    assert (row["L"], row["gp"], row["first_idx"], row["missed"]) == (40, 18, 20, 22)
    assert row["longest_run"] == 20 and row["n_runs"] == 3 and row["veteran"]
    assert row["f"] == pytest.approx(18 / 40)
    df = pd.DataFrame([row]).assign(season="2019-20", age=25.0)
    tail_health = row["gp"] / (row["L"] - row["first_idx"])
    assert tail_health == pytest.approx(0.9)
    assert R.in_cohort(df, R.PRIMARY)[0]                                 # 50% block, 20-game tail, 90% healthy
    assert not R.in_cohort(df, R.CohortConfig(tail_x=0.95))[0]           # tail health threshold bites
    assert not R.in_cohort(df, R.CohortConfig(min_tail=25))[0]           # tail-length threshold bites
    assert not R.in_cohort(df, R.CohortConfig(t1=0.60))[0]               # block threshold bites


def test_two_blocks_or_a_shaky_tail_is_not_a_single_contiguous_absence():
    two = _pat((10, 20, True), (30, 40, True))             # 10-game lead miss, 10-game miss in the middle
    shaky = _pat((12, 40, True))
    shaky[13:40:2] = False                                  # back after 12 games but then absent every other game
    h = History.until(_tables({(1, "2018-19"): two, (2, "2018-19"): shaky, (1, "2019-20"): two, (2, "2019-20"): shaky}), "2019-20")
    d = R.prior_features(h).assign(season="2019-20", age=25.0)
    lead = R.CohortConfig(t1=0.25, tail_x=0.75, min_tail=15)
    assert not R.in_cohort(d, lead).any()
    # The two-block player has a 10-game (25%) lead block and an 20-game tail played 10/30: fails health; and as a "mid" cohort the
    # interior block (10 games, 25%) is followed by only 10 games, below the tail minimum.
    assert not R.in_cohort(d, R.CohortConfig(kind="mid", min_tail=15)).any()


def test_mid_season_return_after_playing_is_a_mid_cohort_member_not_a_lead_one():
    # Plays games 0..9, out 10..24 (15 games = 37.5%), back for the last 15 (all played).
    present = _pat((0, 10, True), (25, 40, True))
    h = History.until(_tables({(1, "2018-19"): present, (1, "2019-20"): _pat((0, 40, True))}), "2019-20")
    d = R.prior_features(h).assign(season="2019-20", age=25.0)
    r = d.iloc[0]
    assert (r["block_start"], r["block_len"], r["block_tail"], r["first_idx"]) == (10, 15, 15, 0)
    assert R.in_cohort(d, R.CohortConfig(kind="mid", t1=0.25))[0]
    assert not R.in_cohort(d, R.CohortConfig(kind="lead", t1=0.25))[0]
    assert not R.in_cohort(d, R.CohortConfig(kind="mid", t1=0.60))[0]


def test_universe_filters_debutants_traded_players_and_low_minutes():
    df = pd.DataFrame({"season": "2019-20", "veteran": [True, False, True, True], "mpg": [25.0, 25.0, 25.0, 10.0],
                       "n_teams": [1, 1, 2, 1], "L": 40})
    assert R.in_universe(df, R.PRIMARY).tolist() == [True, False, False, False]
    assert R.in_universe(df, R.CohortConfig(single_team=False, min_mpg=0.0)).tolist() == [True, False, True, True]
    assert not R.in_universe(df, R.CohortConfig(exclude_targets=("2019-20",))).any()


def test_a_traded_player_is_flagged_and_attributed_to_the_team_he_played_most_for():
    tables = _tables({(1, "2018-19"): _pat((0, 40, True)), (1, "2019-20"): _pat((0, 40, True))})
    other = 1610612738
    first = tables["game_logs"]
    mask = (first["season"] == "2018-19") & (first["game_id"].str[-3:].astype(int) < 10)
    tables["game_logs"].loc[mask, "team_id"] = other        # 10 games for another team, 30 for the primary one
    row = R.prior_features(History.until(tables, "2019-20")).iloc[0]
    assert row["n_teams"] == 2 and row["team_id"] == TEAM and row["gp"] == 30


def test_features_ignore_the_future_and_the_target_season():
    """The point-in-time guard: prior-season features are identical whether or not the future exists or is scrambled."""
    rng = np.random.default_rng(3)
    pats = {}
    for pid in range(1, 9):
        for season in ("2018-19", "2019-20"):
            pats[(pid, season)] = rng.random(L) < 0.6
    pats[(1, "2018-19")] = _pat((20, 40, True))
    tables = _tables(pats)
    base = R.prior_features(History.until(tables, "2019-20"))
    # season 2019-20 (the target) is in the tables but must not be read
    with scrambled_future(tables, "2019-20"):
        assert R.prior_features(History.until(tables, "2019-20")).equals(base)
    # extending the tables with an extra future season changes nothing either
    longer = _tables({**pats, **{(p, "2020-21"): rng.random(L) < 0.3 for p in range(1, 9)}}, seasons=("2018-19", "2019-20", "2020-21"))
    assert R.prior_features(History.until(longer, "2019-20")).equals(base)


def test_match_controls_stays_in_the_season_and_within_calipers():
    df = pd.DataFrame({
        "season": ["a", "a", "a", "a", "a", "b"], "f": [0.30, 0.32, 0.31, 0.60, 0.33, 0.30],
        "age": [25.0, 25.5, 30.0, 25.0, 26.0, 25.0], "mpg": [25.0, 24.0, 25.0, 25.0, 27.0, 25.0]})
    cohort = np.array([True, False, False, False, False, False])
    pool = ~cohort
    (m,) = R.match_controls(df, cohort, pool, k=3)
    assert set(m) == {1, 4}            # 2: age gap; 3: fraction gap; 5: other season
    assert len(R.match_controls(df, cohort, np.zeros(6, bool))[0]) == 0


def test_boot_mean_is_deterministic_stratified_and_brackets_the_mean():
    v = np.r_[np.arange(10.0), np.arange(10.0) + 100]
    st = np.r_[np.zeros(10), np.ones(10)]
    a, b = R.boot_mean(v, st, n_boot=500, seed=1), R.boot_mean(v, st, n_boot=500, seed=1)
    assert a == b and a[0] == pytest.approx(v.mean()) and a[1] < a[0] < a[2]
    # stratified: every replicate keeps 10 rows per season, so the CI is far narrower than an unstratified resample would give
    assert a[2] - a[1] < 10
    assert all(np.isnan(x) for x in R.boot_mean([np.nan], [0], n_boot=10))


def test_signed_error_convention_positive_means_under_projected():
    f = R.add_outcomes(pd.DataFrame({"actual_gp": [60.0], "proj_gp": [50.0], "actual_total_fp": [2000.0], "proj_total_fp": [1500.0],
                                     "actual_fppg": [33.0], "proj_fppg": [30.0]}))
    assert f["err_gp"][0] == 10 and f["err_fp"][0] == 500 and f["above"][0] == 1 and f["err_fppg"][0] == 3


# ------------------------------------------------------------------ end to end on a synthetic league

@pytest.fixture(scope="module")
def league_with_returners():
    """A synthetic league plus a set of contract-valid veterans who miss the first half of each season then play on."""
    tables = make_league(first_start=2012, last_start=2019, n_teams=12, games_per_team=60, seed=5)
    tg = tables["team_games"]
    rows, bios, plrs = [], [], []
    rng = np.random.default_rng(0)
    for k in range(40):
        pid, team = 9_000_000 + k, 1610612737 + (k % 12)
        for season in ("2015-16", "2016-17", "2017-18", "2018-19"):
            sched = tg[(tg["season"] == season) & (tg["team_id"] == team)].sort_values("game_date")
            if k % 2 == 0 and season == "2017-18":
                keep = np.arange(len(sched)) >= 24                       # out for 24 of 60 games, then healthy
                keep &= rng.random(len(sched)) < 0.95
            else:
                keep = rng.random(len(sched)) < 0.8
            for _, g in sched[keep].iterrows():
                rows.append(dict(season=season, game_id=g["game_id"], game_date=g["game_date"], player_id=pid, player_name=f"R{k}",
                                 team_id=team, team_abbr=g["team_abbr"], matchup=f"{g['team_abbr']} vs. X", plus_minus=0.0,
                                 **box_line(26.0)))
            bios.append(dict(season=season, player_id=pid, age_at_season_start=26.0, team_id=team))
        plrs.append(dict(player_id=pid, player_name=f"R{k}", birthdate=pd.Timestamp("1991-01-01"), position="F", height_in=78.0,
                         weight_lb=210.0, draft_year=2012, draft_round=1, draft_number=9, from_year=2012, to_year=2019))
    out = {k: v.copy() for k, v in tables.items()}
    gl = pd.concat([out["game_logs"], pd.DataFrame(rows)[out["game_logs"].columns]], ignore_index=True)
    for c in out["game_logs"].columns:
        gl[c] = gl[c].astype(out["game_logs"][c].dtype)
    out["game_logs"] = gl
    out["player_season_bio"] = pd.concat([out["player_season_bio"], pd.DataFrame(bios)[out["player_season_bio"].columns]], ignore_index=True)
    new = pd.DataFrame(plrs)
    for c in out["players"].columns:
        new[c] = new[c].astype(out["players"][c].dtype)
    out["players"] = pd.concat([out["players"], new], ignore_index=True)
    return out


def test_study_frame_and_report_end_to_end(league_with_returners, tmp_path, monkeypatch, capsys):
    from src.store import write_table

    for name in HISTORY_TABLES:
        write_table(league_with_returners[name], name, tmp_path)
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path))
    out = tmp_path / "rep"
    rc = R.main(["--seasons", "2017-18:2018-19", "--compare", "", "--out", str(out), "--n-boot", "100"])
    assert rc == 0
    text = (out / "return_report.md").read_text(encoding="utf-8")
    for h in ("# Return-from-injury study", "## 1. Primary cohort vs matched controls", "## 3. Sensitivity grid", "## 4. Calibration",
              "## 6. Players in the primary cohort now"):
        assert h in text
    frame = pd.read_parquet(out / "study_frame.parquet")
    assert set(frame["season"]) == {"2017-18", "2018-19"}
    assert (frame["proj_total_fp"] - frame["proj_gp"] * frame["proj_fppg"]).abs().max() < 1e-3
    grid = pd.read_csv(out / "sensitivity.csv")
    assert grid["primary"].sum() == 1 and len(grid) >= 15
    # the 2018-19 target sees the injected 2017-18 returners as a cohort; the study found some
    assert R.in_cohort(frame.reset_index(drop=True), R.PRIMARY).sum() >= 1
    for name in ("primary.csv", "calibration.csv", "current_cohort.csv"):
        assert (out / name).exists()


def test_empty_cohort_is_reported_not_crashed():
    f = R.add_outcomes(pd.DataFrame({
        "season": ["a"] * 4, "player_id": range(4), "veteran": True, "mpg": 25.0, "n_teams": 1, "L": 40, "gp": 30, "first_idx": 0,
        "missed": 10, "longest_run": 3, "block_start": 0, "block_len": 0, "block_tail": 0, "f": 0.75, "age": 25.0,
        "proj_gp": 50.0, "proj_fppg": 20.0, "proj_total_fp": 1000.0, "actual_gp": 40.0, "actual_total_fp": 800.0, "actual_fppg": 20.0}))
    res = R.evaluate(f, n_boot=50)
    assert res["n"] == 0 and R.verdict(res) == "no controls"
    assert len(R.sensitivity_grid(f, n_boot=20)) >= 15
