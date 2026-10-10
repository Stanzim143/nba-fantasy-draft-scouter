"""ADR 0039 part B study: calendars per season, the share-gap label, point-in-time features, walk-forward training, the rule."""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from model_testkit import make_league  # noqa: E402

from src.backtest import playoff_availability as P  # noqa: E402
from src.backtest.leakage import scrambled_future  # noqa: E402
from src.contracts import History  # noqa: E402
from src.value.league import load_league  # noqa: E402

TARGET = "2018-19"


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(load_league())      # the synthetic seasons are 40 games (~11 weeks): an 8 + 3 week league fits them
    c["league"]["schedule"].update(regular_season_matchups=8, playoff_rounds_weeks=[1, 1, 1])
    return c


@pytest.fixture(scope="module")
def tables():
    t = make_league(n_teams=14, games_per_team=40, seed=8)
    for k in ("game_logs", "team_games"):    # the testkit's league has no minimum-minutes cut: lower the universe bar below
        t[k] = t[k].copy()
    return t


@pytest.fixture(scope="module", autouse=True)
def _small_universe():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(P, "MIN_GP", 20)
        mp.setattr(P, "MIN_MPG", 10.0)
        yield


SEASONS = ["2017-18", "2018-19", "2019-20"]


@pytest.fixture(scope="module")
def built(tables, cfg):
    gl = tables["game_logs"]
    proj = (gl[gl["season"].isin(SEASONS)].groupby(["season", "player_id"]).size().rename("proj_gp").reset_index()
            .assign(proj_fppg=20.0))
    return P.build_frame(tables, SEASONS, proj, cfg=cfg, log=lambda *_: None)


def test_season_windows_use_the_league_period_counts_and_flag_short_seasons(tables, cfg):
    w = P.season_windows(tables["team_games"], TARGET, cfg)
    assert w["po_weeks"] == (9, 11) and w["usable"] and w["po_start"] > w["reg_end"]
    assert (w["team_po"] > 0).all()
    # the real league's 19 + 3 weeks run past a 40-game synthetic season: its playoff weeks are empty, so it is not scored
    assert not P.season_windows(tables["team_games"], TARGET, load_league())["usable"]


def test_late_season_table_gap_is_playoff_share_minus_regular_share(cfg):
    days = pd.date_range("2026-10-19", periods=77, freq="D")    # 11 Monday-Sunday weeks
    tg_rows, gl_rows = [], []
    for i, d in enumerate(days[::2]):                            # one game every other day, team 1 v team 2
        for tid, home in ((1, True), (2, False)):
            tg_rows.append({"season": "2026-27", "game_id": f"g{i}", "game_date": d, "team_id": tid, "team_abbr": str(tid),
                            "is_home": home, "pts_for": 100 + tid, "pts_against": 100})
    tg = pd.DataFrame(tg_rows)
    w = P.season_windows(tg, "2026-27", cfg)
    for i, d in enumerate(days[::2]):
        late = d >= w["po_start"]
        if late and i % 2:                                       # player 7 plays every regular game, half the playoff-week games
            continue
        gl_rows.append({"season": "2026-27", "game_id": f"g{i}", "game_date": d, "player_id": 7, "team_id": 1,
                        "min": 30.0, "pts": 10, "reb": 0, "ast": 0, "stl": 0, "blk": 0, "tov": 0, "fgm": 4, "fga": 8,
                        "fg3m": 0, "ftm": 2, "fta": 2})
    t = P.late_season_table(pd.DataFrame(gl_rows), w, {"PTS": 1})
    r = t.set_index("player_id").loc[7]
    assert r["gp_reg"] == w["team_reg"][1]
    assert r["gap"] == pytest.approx(r["gp_po"] / w["team_po"][1] - 1.0) and r["gap"] < 0
    assert r["fp_po"] == pytest.approx(10 * r["gp_po"])


def test_features_never_read_the_target_season(tables, cfg):
    scoring = load_league()["scoring"]
    clean = P.prior_features(History.until(tables, TARGET), scoring, cfg)
    with scrambled_future(tables, TARGET):
        dirty = P.prior_features(History.until(tables, TARGET), scoring, cfg)
    pd.testing.assert_frame_equal(clean, dirty)
    assert len(clean) > 20 and set(P.FEATURES) <= set(clean.columns)
    # and through build_frame: dropping the target season's games changes the labels, never a feature
    proj = pd.DataFrame(columns=["season", "player_id", "proj_gp", "proj_fppg"])
    full, _ = P.build_frame(tables, [TARGET], proj, cfg=cfg, log=lambda *_: None)
    cut = {k: (v[v["season"] != TARGET] if k == "game_logs" else v) for k, v in tables.items()}
    trunc, _ = P.build_frame(cut, [TARGET], proj, cfg=cfg, log=lambda *_: None)
    a, b = full[full["season"] == TARGET], trunc[trunc["season"] == TARGET]
    pd.testing.assert_frame_equal(a[["player_id", *P.FEATURES]].reset_index(drop=True),
                                  b[["player_id", *P.FEATURES]].reset_index(drop=True))
    assert (b["gp_po"] == 0).all() and (a["gp_po"] > 0).any()


def test_training_rows_are_strictly_earlier_seasons(built):
    frame, _ = built
    tr = P.training_rows(frame, "2018-19")
    assert len(tr) and set(tr["season"]) <= {"2013-14", "2014-15", "2015-16", "2016-17", "2017-18"}
    assert set(P.training_rows(frame, "2019-20")["season"]) - set(tr["season"]) == {"2018-19"}


def test_predict_evaluate_and_decide_end_to_end(built):
    frame, meta = built
    pred = P.predict(frame, SEASONS)
    assert {"g_mean", "g_ols", "e1_mean", "e1_ols"} <= set(pred.columns)
    e = pred["e1_ols"].dropna()
    assert len(e) and ((e >= 0) & (e <= pred.loc[e.index, "team_po"] + 1e-9)).all()
    res = P.evaluate(pred)
    assert res["decision"] in {"adopt_ols", "adopt_mean", "reject"}
    assert res["mean"]["games_mse"]["n_seasons"] == len(res["seasons"]) and "| season |" in P.render(res, meta)


def _cand(lo, wins, n, mae_hi):
    return {"games_mse": {"lo": lo, "wins": wins, "n_seasons": n}, "games_mae": {"hi": mae_hi}}


def test_decision_rule_matches_the_preregistration():
    good, weak = _cand(0.01, 7, 9, 0.1), _cand(-0.01, 9, 9, 0.1)
    assert P.passes(good) and not P.passes(weak)
    assert not P.passes(_cand(0.01, 6, 9, 0.1))                     # 6/9 < 70%
    assert P.passes(_cand(0.01, 6, 8, 0.1))                         # 6/8 = 75%
    assert not P.passes(_cand(0.01, 9, 9, -0.01))                   # MAE significantly worse
    assert P.decide({"ols": good, "mean": weak, "ols_over_mean_games_mse": {"lo": -1}}) == "adopt_ols"
    assert P.decide({"ols": good, "mean": good, "ols_over_mean_games_mse": {"lo": -1}}) == "adopt_mean"
    assert P.decide({"ols": good, "mean": good, "ols_over_mean_games_mse": {"lo": 0.01}}) == "adopt_ols"
    assert P.decide({"ols": weak, "mean": weak, "ols_over_mean_games_mse": {"lo": 1}}) == "reject"
    assert np.isclose(P.MIN_WIN_SHARE, 0.7)
