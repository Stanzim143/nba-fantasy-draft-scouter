"""Rookie appearance stage (ADR 0038): fit, partial-preseason handling, fallbacks and the projector wiring."""
import numpy as np
import pandas as pd
import pytest
from debutant_testkit import TARGET, build
from model_testkit import history_for

from src.backtest.leakage import assert_projector_ignores_future
from src.contracts import season_start, validate_table
from src.models import rookie_appearance as ra
from src.models.debutants import DebutantBaselineProjector
from src.models.registry import available_projectors, get_projector

SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2010, 2018)]


def _world(n_per: int = 40, seed: int = 0, target: str = "2018-19", target_dates: int = 10):
    """Drafted classes where a preseason game nearly guarantees a rookie game and its absence makes one a coin flip."""
    rng = np.random.default_rng(seed)
    players, gl, logs = [], [], []
    pid = 1
    for s in SEASONS + [target]:
        sy = season_start(s)
        n_dates = target_dates if s == target else 10
        for i in range(n_per):
            pick = 1 + (i * 60) // n_per
            pre = rng.random() < (0.95 if pick < 30 else 0.7)
            players.append(dict(player_id=pid, draft_year=sy, draft_round=1 if pick <= 30 else 2, draft_number=pick))
            if pre:
                for d in range(min(3, n_dates)):
                    logs.append(dict(event_season=s, season=f"{sy - 1}-{str(sy)[-2:]}", context="preseason", player_id=pid,
                                     game_id=f"{s}-{pid}-{d}", game_date=pd.Timestamp(sy, 10, 1) + pd.Timedelta(days=d), min=15.0))
            if s != target and rng.random() < (0.99 if pre else 0.35):
                gl.append(dict(season=s, player_id=pid, game_id=f"g{s}{pid}"))
            pid += 1
        # pad every preseason to its full number of game dates
        for d in range(n_dates):
            logs.append(dict(event_season=s, season=f"{sy - 1}-{str(sy)[-2:]}", context="preseason", player_id=10**6,
                             game_id=f"{s}-pad-{d}", game_date=pd.Timestamp(sy, 10, 1) + pd.Timedelta(days=d), min=10.0))
    return pd.DataFrame(players), pd.DataFrame(gl), pd.DataFrame(logs)


def test_fit_learns_that_a_preseason_game_predicts_appearance():
    players, gl, logs = _world()
    st = ra.fit(players, gl, logs, "2018-19")
    assert st.enabled and st.fraction == 1.0 and st.columns == ra.COLUMNS
    assert st.n_train == len(SEASONS) * 40
    ev = pd.DataFrame({"pre_gp": [3.0, np.nan], "pre_mpg": [15.0, np.nan], "sl_gp": [np.nan, np.nan], "sl_mpg": [np.nan, np.nan]})
    p_with, p_without = st.predict(np.array([45.0, 45.0]), ev)
    assert p_with > 0.9 and p_without < 0.6
    assert np.all((st.predict(np.array([1.0, 60.0]), ev) >= ra.P_CLIP[0]) & (st.predict(np.array([1.0, 60.0]), ev) <= ra.P_CLIP[1]))


def test_live_a_partial_target_preseason_cuts_every_training_preseason_to_the_same_share():
    players, gl, logs = _world(target_dates=5)
    st = ra.fit(players, gl, logs, "2018-19", live=True)
    assert st.fraction == pytest.approx(0.5)
    cut = ra.cut_logs(logs, st.fraction)
    pre = cut[(cut["context"] == "preseason") & (cut["event_season"] == "2012-13")]
    assert pre["game_date"].nunique() == 5


def test_a_short_finished_preseason_is_never_cut_in_a_backtest():
    players, gl, logs = _world(target_dates=5)          # e.g. 2020-21: 9 game dates against a 15-22 norm, but complete
    a, b = ra.fit(players, gl, logs, "2018-19"), ra.fit(*_world()[:2], _world()[2], "2018-19")
    assert a.fraction == 1.0 and a.columns == ra.COLUMNS
    np.testing.assert_allclose(a.beta, b.beta)


def test_live_reference_length_is_the_recent_preseasons_median():
    players, gl, logs = _world(target_dates=10)
    old = logs[(logs["context"] == "preseason") & logs["event_season"].isin(["2010-11", "2011-12"])]
    longer = old.assign(game_date=old["game_date"] + pd.Timedelta(days=20))      # two old, longer preseasons
    logs = pd.concat([logs, longer], ignore_index=True)
    assert ra.target_fraction(logs, "2018-19", SEASONS, live=True) == 1.0
    assert ra.target_fraction(logs, "2018-19", SEASONS, live=False) == 1.0
    assert ra.target_fraction(logs[logs["event_season"] != "2018-19"], "2018-19", SEASONS, live=True) == 0.0


def test_no_target_preseason_drops_the_preseason_columns():
    players, gl, logs = _world()
    logs = logs[logs["event_season"] != "2018-19"]
    st = ra.fit(players, gl, logs, "2018-19")
    assert st.fraction == 0.0 and st.enabled
    assert "has_pre" not in st.columns and "pre_mpg" not in st.columns
    assert any("dropped" in n for n in st.notes)
    assert ra.cut_logs(logs, 0.0)["context"].ne("preseason").all()


def test_too_few_rookies_or_no_misses_switches_the_stage_off():
    players, gl, logs = _world(n_per=5)
    st = ra.fit(players, gl, logs, "2018-19")
    assert not st.enabled and np.all(st.predict(np.array([10.0, 50.0]), pd.DataFrame(index=range(2))) == 1.0)
    players, gl, logs = _world()
    every = pd.DataFrame({"season": [s for s in SEASONS for _ in range(40)], "player_id": players["player_id"].iloc[:len(SEASONS) * 40],
                          "game_id": "x"})
    assert not ra.fit(players, every, logs, "2018-19").enabled


def test_training_rows_never_read_the_target_season():
    players, gl, logs = _world()
    gl_future = pd.concat([gl, pd.DataFrame({"season": ["2018-19"] * 40, "player_id": range(1, 41), "game_id": "f"})])
    a, b = ra.fit(players, gl, logs, "2018-19"), ra.fit(players, gl_future, logs, "2018-19")
    np.testing.assert_array_equal(a.beta, b.beta)


# --------------------------------------------------------------------------- projector wiring (synthetic league)

T = build()


@pytest.fixture(scope="module")
def pair():
    off = DebutantBaselineProjector(min_analog_seasons=2).project(history_for(T))
    on_p = DebutantBaselineProjector(min_analog_seasons=2, rookie_appearance=True)
    return off, on_p, on_p.project(history_for(T))


def test_the_stage_is_registered_on_the_board_stack():
    name = "baseline_hurdle_adp_depth_rookie_offseason_debut"
    assert name in available_projectors()
    assert get_projector(name).base.rookie_appearance is True


def test_only_rookie_games_change(pair):
    off, proj, on = pair
    validate_table(on, "projections")
    a, b = off.set_index("player_id").sort_index(), on.set_index("player_id").sort_index()
    assert list(a.index) == list(b.index)
    vet = a["projection_class"] != "rookie"
    for c in ("proj_fppg", "proj_gp", "proj_total_fp"):
        np.testing.assert_allclose(b.loc[vet, c], a.loc[vet, c], rtol=1e-12)
    np.testing.assert_allclose(b["proj_fppg"], a["proj_fppg"], rtol=1e-12)


def test_an_enabled_stage_scales_rookie_games_by_its_probability(monkeypatch):
    hist = history_for(T)
    fixed = ra.RookieAppearance(columns=("log_pick",), beta=np.array([0.0, 0.0]), mean=np.zeros(1), scale=np.ones(1),
                                enabled=True)
    monkeypatch.setattr(ra, "fit", lambda *a, **k: fixed)
    off = DebutantBaselineProjector(min_analog_seasons=2).project(hist).set_index("player_id").sort_index()
    on = DebutantBaselineProjector(min_analog_seasons=2, rookie_appearance=True).project(hist).set_index("player_id").sort_index()
    rk = off["projection_class"] == "rookie"
    assert rk.any()
    np.testing.assert_allclose(on.loc[rk, "proj_gp"], 0.5 * off.loc[rk, "proj_gp"], rtol=1e-9)
    np.testing.assert_allclose(on.loc[~rk, "proj_gp"], off.loc[~rk, "proj_gp"], rtol=1e-12)
    # plain (non-hurdle) config: the games band becomes the zero-inflated mixture on rookie rows only
    assert (on.loc[rk, "proj_gp_p10"] == 0.0).all() and (off.loc[rk, "proj_gp_p10"] > 0.0).all()
    for c in ("proj_gp_sd", "proj_gp_p10", "proj_gp_p90"):
        np.testing.assert_array_equal(on.loc[~rk, c].to_numpy(), off.loc[~rk, c].to_numpy())


def test_the_projector_with_the_stage_ignores_the_future():
    assert_projector_ignores_future(DebutantBaselineProjector(min_analog_seasons=2, rookie_appearance=True), T, TARGET)
