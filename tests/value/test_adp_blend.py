"""ADP + model blend for the board (ADR 0032): the fit, the application, the board columns, and the loader."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from model_testkit import history_for, make_league  # noqa: E402

from src.models.baseline import BaselineProjector  # noqa: E402
from src.value import adp_blend as AB  # noqa: E402
from src.value.board import build_board, load_blend  # noqa: E402


def _frames(n_seasons=4, n=200, seed=0, true=(300.0, -900.0, 60.0, 0.4, 0.0)):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n_seasons):
        adp = np.sort(rng.uniform(1, 200, n))
        model = np.maximum(3500 - 14 * adp + rng.normal(0, 300, n), 50)
        la = np.log(adp)
        y = true[0] + true[1] * la + true[2] * la ** 2 + true[3] * model + rng.normal(0, 30, n)
        out[f"20{16 + i}-{17 + i}"] = pd.DataFrame({"player_id": np.arange(n), "adp": adp, "proj_total_fp": model,
                                                    "actual_total_fp": y})
    return out


# --------------------------------------------------------------------------- fit

def test_fit_recovers_the_generating_coefficients():
    b = AB.fit_adp_blend(_frames(n_seasons=6), model="m")
    assert b.coef[3] == pytest.approx(0.4, abs=0.05)
    assert b.n_rows == 1200 and b.seasons[0] == "2016-17" and b.model == "m"
    pred = b.predict(np.array([10.0, 100.0]), np.array([3000.0, 1500.0]))
    la = np.log([10.0, 100.0])
    truth = 300 - 900 * la + 60 * la ** 2 + 0.4 * np.array([3000.0, 1500.0])
    assert np.allclose(pred, truth, atol=40)


def test_fit_uses_zero_for_listed_players_who_did_not_play_and_flags_missing_model_totals():
    f = _frames(n_seasons=1)["2016-17"]
    f.loc[:9, "actual_total_fp"] = np.nan        # listed but no games
    f.loc[10:19, "proj_total_fp"] = np.nan       # listed but not projected
    b = AB.fit_adp_blend({"2016-17": f}, model="m")
    assert np.isfinite(b.coef).all() and b.n_rows == len(f)
    assert np.isfinite(b.predict(np.array([30.0]), np.array([np.nan]))).all()


def test_fit_needs_enough_rows():
    with pytest.raises(ValueError, match="at least"):
        AB.fit_adp_blend(_frames(n_seasons=1, n=40), model="m")
    with pytest.raises(ValueError, match="no seasons"):
        AB.fit_adp_blend({}, model="m")


def test_save_and_load_round_trip(tmp_path):
    b = AB.fit_adp_blend(_frames(), model="baseline_hurdle")
    path = b.save(tmp_path / "x" / AB.BLEND_FILE)
    assert AB.AdpBlend.load(path) == b


# --------------------------------------------------------------------------- application

def _proj(n=6):
    return pd.DataFrame({"player_id": np.arange(n), "proj_total_fp": np.linspace(3000, 500, n), "proj_gp": 60.0,
                         "proj_fppg": np.linspace(3000, 500, n) / 60.0})


def test_listed_players_get_the_regression_and_the_rest_keep_the_model_total():
    b = AB.AdpBlend([100.0, 0.0, 0.0, 0.5, 0.0], 1000.0, "m", ["2016-17"], 1)
    proj = _proj()
    adp = pd.DataFrame({"player_id": [0, 2], "adp": [3.0, 40.0]})
    out = AB.blend_totals(proj, adp, b)
    assert out[0] == pytest.approx(100 + 0.5 * 3000) and out[2] == pytest.approx(100 + 0.5 * proj.loc[2, "proj_total_fp"])
    assert out[1] == proj.loc[1, "proj_total_fp"] and out[5] == proj.loc[5, "proj_total_fp"]
    assert (out >= 0).all()


def test_board_columns_are_additive_and_blend_rank_is_a_permutation():
    tables = make_league(n_teams=14, games_per_team=40, seed=8)
    hist = history_for(tables)
    proj = BaselineProjector().project(hist)
    rng = np.random.default_rng(1)
    listed = proj.nlargest(60, "proj_total_fp")["player_id"].to_numpy()
    adp = pd.DataFrame({"player_id": listed, "adp": rng.permutation(np.arange(1, 61)).astype(float)})
    blend = AB.AdpBlend([200.0, -300.0, 30.0, 0.6, 0.0], float(proj["proj_total_fp"].mean()), "baseline", ["2016-17"], 1)
    plain = build_board(proj, hist.players, season_games=40, adp=adp)
    blended = build_board(proj, hist.players, season_games=40, adp=adp, blend=blend)
    assert list(blended.columns) == list(plain.columns) + list(AB.BLEND_COLUMNS)
    pd.testing.assert_frame_equal(blended[plain.columns].reset_index(drop=True), plain.reset_index(drop=True))
    assert sorted(blended["blend_rank"]) == list(range(1, len(blended) + 1))
    top = blended.sort_values("blend_rank")["blend_vorp"]
    assert top.is_monotonic_decreasing
    assert blended.attrs["replacement"] == plain.attrs["replacement"]
    assert not blended["blend_rank"].equals(blended["rank"])           # a different ordering, not a copy
    assert build_board(proj, hist.players, season_games=40, blend=blend).columns.tolist() == build_board(
        proj, hist.players, season_games=40).columns.tolist()          # no ADP, no blend columns


# --------------------------------------------------------------------------- loader

def test_load_blend_modes(tmp_path):
    assert load_blend("off", tmp_path) is None and load_blend(None, tmp_path) is None
    assert load_blend("auto", tmp_path) is None                       # nothing fitted yet: no blend, no error
    b = AB.fit_adp_blend(_frames(), model="m")
    b.save(tmp_path / "processed" / AB.BLEND_FILE)
    assert load_blend("auto", tmp_path) == b
    assert load_blend(str(tmp_path / "processed" / AB.BLEND_FILE), tmp_path) == b
    (tmp_path / "processed" / AB.BLEND_FILE).write_text("{not json", encoding="utf-8")
    assert load_blend("auto", tmp_path) is None                       # a corrupt auto file never blocks the board
    b.save(tmp_path / "processed" / AB.BLEND_FILE)
    assert load_blend("auto", tmp_path, "m") == b                     # fitted for this model: applied
    assert load_blend("auto", tmp_path, "other") is None              # fitted for another model: not applied
    assert load_blend(str(tmp_path / "processed" / AB.BLEND_FILE), tmp_path, "other") == b   # explicit path: warned only
    with pytest.raises(FileNotFoundError):                            # an explicit path that cannot be read is an error
        load_blend(str(tmp_path / "missing.json"), tmp_path)
