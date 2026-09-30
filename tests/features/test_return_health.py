"""Return-health features (ADR 0022): lead block, tail health, shrink, traded/debut players, NaN handling, widths, no look-ahead."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from model_testkit import box_line  # noqa: E402
from src.backtest import return_report as R  # noqa: E402
from src.backtest.leakage import scrambled_future  # noqa: E402
from src.contracts import History  # noqa: E402
from src.features import return_health as RH  # noqa: E402

TEAM, OTHER = 1610612737, 1610612738
L = 40
NAMES = RH.FEATURE_NAMES


def _tables(patterns, *, seasons=("2017-18", "2018-19", "2019-20"), from_year=None) -> dict[str, pd.DataFrame]:
    """One team, ``L`` games a season; ``patterns[(pid, season)]`` = bool array of games the player appeared in."""
    tg, gl = [], []
    for season in seasons:
        y = int(season[:4])
        for g in range(L):
            tg.append(dict(season=season, game_id=f"{season}-{g:03d}", game_date=pd.Timestamp(y, 10, 20) + pd.Timedelta(days=2 * g),
                           team_id=TEAM, team_abbr="AAA", is_home=True, pts_for=100, pts_against=99))
    tg = pd.DataFrame(tg)
    for (pid, season), present in patterns.items():
        assert len(present) == L
        for g in np.flatnonzero(present):
            row = tg[tg["season"] == season].iloc[g]
            gl.append(dict(season=season, game_id=row["game_id"], game_date=row["game_date"], player_id=pid, player_name=f"P{pid}",
                           team_id=TEAM, team_abbr="AAA", matchup="AAA vs. X", plus_minus=0.0, **box_line(28.0)))
    pids = sorted({p for p, _ in patterns})
    frm = from_year or {}
    players = pd.DataFrame({"player_id": pids, "player_name": [f"P{p}" for p in pids], "birthdate": pd.Timestamp("1995-01-01"),
                            "position": "F", "height_in": 78.0, "weight_lb": 210.0, "draft_year": 2015, "draft_round": 1,
                            "draft_number": 5, "from_year": [frm.get(p, 2015) for p in pids], "to_year": 2030})
    bio = pd.DataFrame([dict(season=s, player_id=p, age_at_season_start=24.0, team_id=TEAM) for (p, s) in patterns])
    return {"game_logs": pd.DataFrame(gl), "team_games": tg, "players": players, "player_season_bio": bio}


def _pat(*spans):
    a = np.zeros(L, bool)
    for s, e, v in spans:
        a[s:e] = v
    return a


def _feat(tables, target, pids, *, n_lags=3, decay=0.6):
    h = History.until(tables, target)
    f = RH.ReturnFeatures.fit(h, n_lags=n_lags, decay=decay)
    s = int(target[:4])
    return f, f.build(np.asarray(pids, "int64"), np.full(len(pids), s, "int64"), np.full(len(pids), 25.0))


def col(x, name):
    return x[:, NAMES.index(name)]


# ------------------------------------------------------------------ the panel

def test_panel_lead_block_tail_and_shrunk_health_on_team_games():
    # misses games 0..19 (a 20-game lead block = 50% of a 40-game season), then plays 18 of the remaining 20
    present = _pat((20, 40, True))
    present[25] = present[30] = False
    t = _tables({(1, "2017-18"): np.ones(L, bool), (1, "2018-19"): present})
    p = RH.build_return_panel(History.until(t, "2019-20"))
    r = p[(p.player_id == 1) & (p.s == 2018)].iloc[0]
    assert (r["L"], r["gp"], r["first_idx"]) == (L, 18, 20)
    assert r["lead_frac"] == pytest.approx(0.5) and r["tail_len_frac"] == pytest.approx(0.5)
    assert r["tail_health_s"] == pytest.approx((18 + RH.SHRINK_K * RH.NEUTRAL_HEALTH) / (20 + RH.SHRINK_K))
    assert r["lead_season"] == 1.0 and r["return_flag"] == 1.0            # 50% block, 20-game tail, 90% raw health


def test_flag_follows_the_adr_0021_definitions():
    cases = {
        2: _pat((10, 40, True)),                              # block 10/40 = 25%, tail 30, health 100%  -> flag (boundary)
        3: _pat((9, 40, True)),                               # block 9/40 = 22.5% < 25%                  -> no flag
        4: _pat((30, 40, True)),                              # block 75% but tail of 10 < 15 games       -> no flag
        5: np.r_[np.zeros(15, bool), np.tile([True, False], 13)[:25]],   # tail 25 games at 52% health   -> no flag
    }
    t = _tables({**{(p, "2018-19"): a for p, a in cases.items()}, **{(p, "2017-18"): np.ones(L, bool) for p in cases}})
    p = RH.build_return_panel(History.until(t, "2019-20")).query("s == 2018").set_index("player_id")
    assert p["return_flag"].to_dict() == {2: 1.0, 3: 0.0, 4: 0.0, 5: 0.0}
    assert p.loc[4, "lead_season"] == 1.0 and p.loc[4, "lead_frac"] == pytest.approx(0.75)


def test_short_tail_is_shrunk_toward_neutral_health_not_read_as_perfect():
    t = _tables({(1, "2017-18"): np.ones(L, bool), (1, "2018-19"): _pat((36, 40, True))})   # 4 games back, all played
    p = RH.build_return_panel(History.until(t, "2019-20"))
    h = p[(p.player_id == 1) & (p.s == 2018)]["tail_health_s"].iloc[0]
    assert h == pytest.approx((4 + 7.5) / 14) and RH.NEUTRAL_HEALTH < h < 1.0


def test_no_lead_block_has_neutral_tail_health_and_zero_flag_and_lead():
    t = _tables({(1, "2017-18"): np.ones(L, bool), (1, "2018-19"): np.ones(L, bool)})
    _, x = _feat(t, "2019-20", [1])
    assert col(x, "lead_bar")[0] == 0 and col(x, "return_bar")[0] == 0 and col(x, "tail_len_bar")[0] == 0
    assert col(x, "tail_health_bar")[0] == RH.NEUTRAL_HEALTH and col(x, "tail_health_last")[0] == RH.NEUTRAL_HEALTH
    assert col(x, "lead_last")[0] == 0 and col(x, "return_last")[0] == 0


def test_a_small_lead_block_is_not_a_lead_season():
    t = _tables({(1, "2017-18"): np.ones(L, bool), (1, "2018-19"): _pat((3, 40, True))})   # 3 of 40 = 7.5% < LEAD_MIN
    p = RH.build_return_panel(History.until(t, "2019-20"))
    assert p[p.s == 2018]["lead_season"].iloc[0] == 0.0
    _, x = _feat(t, "2019-20", [1])
    assert col(x, "lead_last")[0] == pytest.approx(3 / 40) and col(x, "tail_health_last")[0] == RH.NEUTRAL_HEALTH


# ------------------------------------------------------------------ who counts

def test_debut_season_is_not_a_lead_block():
    """A player whose first NBA season this was (a mid-season call-up) shows 'games missed' that are not an absence."""
    t = _tables({(1, "2017-18"): np.ones(L, bool), (1, "2018-19"): _pat((20, 40, True)),
                 (2, "2018-19"): _pat((20, 40, True))}, from_year={2: 2018})      # player 2 debuts in 2018-19
    p = RH.build_return_panel(History.until(t, "2019-20")).query("s == 2018").set_index("player_id")
    assert bool(p.loc[1, "veteran"]) and not bool(p.loc[2, "veteran"])
    assert p.loc[1, "return_flag"] == 1.0 and p.loc[2, "return_flag"] == 0.0 and p.loc[2, "lead_frac"] == 0.0
    _, x = _feat(t, "2019-20", [2])
    assert x[0].tolist() == _feat(t, "2019-20", [999])[1][0].tolist()          # same as "no information"


def test_traded_player_is_treated_as_no_lead_block():
    t = _tables({(1, "2017-18"): np.ones(L, bool), (1, "2018-19"): _pat((20, 40, True))})
    gl = t["game_logs"]
    gl.loc[(gl["season"] == "2018-19") & (gl["game_id"].str[-3:].astype(int) >= 36), "team_id"] = OTHER   # 4 games elsewhere
    p = RH.build_return_panel(History.until(t, "2019-20")).query("s == 2018").iloc[0]
    assert p["n_teams"] == 2 and p["L"] == L and p["gp"] == 16            # primary team's schedule and games, as ADR 0006 / 0021
    assert p["lead_frac"] == 0.0 and p["return_flag"] == 0.0


def test_denominator_is_the_primary_teams_games_not_82():
    t = _tables({(1, "2017-18"): np.ones(L, bool), (1, "2018-19"): _pat((10, 40, True))})
    p = RH.build_return_panel(History.until(t, "2019-20")).query("s == 2018").iloc[0]
    assert p["lead_frac"] == pytest.approx(10 / L)


def test_agrees_with_the_return_report_profile():
    """Same first-appearance index, schedule length and games played as Task A's season profile (ADR 0021)."""
    rng = np.random.default_rng(4)
    pats = {(p, s): rng.random(L) < 0.6 for p in range(1, 12) for s in ("2017-18", "2018-19")}
    pats[(1, "2018-19")] = _pat((20, 40, True))
    t = _tables(pats)
    h = History.until(t, "2019-20")
    a = RH.build_return_panel(h).set_index(["player_id", "s"]).sort_index()
    b = R.season_profiles(h).set_index(["player_id", "s"]).sort_index()
    assert a.index.equals(b.index)
    for mine, theirs in (("L", "L"), ("gp", "gp"), ("first_idx", "first_idx")):
        assert (a[mine].to_numpy() == b[theirs].to_numpy()).all()


# ------------------------------------------------------------------ the feature matrix

def test_columns_recency_weighting_and_last_season_versions():
    # two seasons back: healthy all year; last season: lead block of 20 then a healthy tail
    t = _tables({(1, "2017-18"): np.ones(L, bool), (1, "2018-19"): _pat((20, 40, True))})
    _, x = _feat(t, "2019-20", [1], decay=0.5)
    w = np.array([1.0, 0.5])                      # last season weight 1, the one before 0.5
    assert col(x, "lead_bar")[0] == pytest.approx((1.0 * 0.5 + 0.5 * 0.0) / 1.5)
    assert col(x, "return_bar")[0] == pytest.approx(1.0 / 1.5)
    h = (20 + 7.5) / 30
    assert col(x, "tail_health_bar")[0] == pytest.approx(h)               # lead seasons only: just last season
    assert col(x, "tail_len_bar")[0] == pytest.approx(0.5 / 1.5)
    assert col(x, "lead_last")[0] == pytest.approx(0.5) and col(x, "tail_health_last")[0] == pytest.approx(h)
    assert col(x, "return_last")[0] == 1.0
    assert w.sum() == 1.5


def test_player_who_sat_out_last_season_has_zero_last_columns_but_keeps_older_history():
    t = _tables({(1, "2017-18"): _pat((20, 40, True)), (1, "2018-19"): np.zeros(L, bool), (2, "2018-19"): np.ones(L, bool)})
    _, x = _feat(t, "2019-20", [1], decay=0.6)
    assert col(x, "lead_last")[0] == 0 and col(x, "return_last")[0] == 0 and col(x, "tail_health_last")[0] == RH.NEUTRAL_HEALTH
    assert col(x, "return_bar")[0] == pytest.approx(1.0)                 # only 2017-18 is present, so it carries the full weight


def test_player_with_no_prior_season_and_an_empty_history_are_neutral_and_not_nan():
    t = _tables({(1, "2018-19"): np.ones(L, bool)})
    f, x = _feat(t, "2019-20", [1, 424242])
    neutral = np.zeros(RH.N_COLS)
    neutral[[1, 5]] = RH.NEUTRAL_HEALTH
    assert np.array_equal(x[1], neutral) and np.isfinite(x).all()
    empty = RH.ReturnFeatures.fit(History.until(t, "2018-19"), n_lags=3, decay=0.6)      # a history with no games at all
    e = empty.build(np.array([1, 2], "int64"), np.array([2018, 2018], "int64"), np.array([25.0, 25.0]))
    assert e.shape == (2, RH.N_COLS) and np.isfinite(e).all() and np.array_equal(e[0], neutral)


def test_never_nan_and_width_is_constant_on_a_random_league():
    rng = np.random.default_rng(0)
    pats = {(p, s): rng.random(L) < rng.uniform(0.2, 1.0) for p in range(1, 30) for s in ("2017-18", "2018-19")}
    pats[(3, "2018-19")] = np.zeros(L, bool)
    t = _tables(pats)
    pids = np.arange(1, 40)
    f, x = _feat(t, "2019-20", pids)
    assert x.shape == (len(pids), RH.N_COLS) == (len(pids), len(NAMES)) and f.n_cols == RH.N_COLS
    assert np.isfinite(x).all()
    assert ((x[:, [0, 2, 3, 4, 6]] >= 0) & (x[:, [0, 2, 3, 4, 6]] <= 1)).all() and ((x[:, [1, 5]] > 0) & (x[:, [1, 5]] <= 1)).all()
    assert f.build(np.array([], "int64"), np.array([], "int64"), np.array([])).shape == (0, RH.N_COLS)


def test_concat_features_stacks_columns_in_order():
    t = _tables({(1, "2017-18"): np.ones(L, bool), (1, "2018-19"): _pat((20, 40, True))})
    f, x = _feat(t, "2019-20", [1])
    c = RH.ConcatFeatures([f, f])
    y = c.build(np.array([1], "int64"), np.array([2019], "int64"), np.array([25.0]))
    assert c.n_cols == 2 * RH.N_COLS and y.shape == (1, 2 * RH.N_COLS)
    assert np.array_equal(y[0, :RH.N_COLS], x[0]) and np.array_equal(y[0, RH.N_COLS:], x[0])


# ------------------------------------------------------------------ point in time

def test_features_ignore_the_target_season_and_the_future():
    rng = np.random.default_rng(3)
    pats = {(p, s): rng.random(L) < 0.6 for p in range(1, 9) for s in ("2017-18", "2018-19", "2019-20")}
    pats[(1, "2018-19")] = _pat((20, 40, True))
    t = _tables(pats)
    pids = np.arange(1, 9)
    _, base = _feat(t, "2019-20", pids)
    panel = RH.build_return_panel(History.until(t, "2019-20"))
    with scrambled_future(t, "2019-20"):
        _, again = _feat(t, "2019-20", pids)
        assert RH.build_return_panel(History.until(t, "2019-20")).equals(panel)
    assert np.array_equal(base, again)
    longer = _tables({**pats, **{(p, "2020-21"): rng.random(L) < 0.3 for p in range(1, 9)}}, seasons=("2017-18", "2018-19", "2019-20", "2020-21"))
    _, ext = _feat(longer, "2019-20", pids)
    assert np.array_equal(base, ext)
    assert RH.build_return_panel(History.until(longer, "2019-20"))["s"].max() == 2018
