"""ADR 0031-0033 report sections: appearance calibration, season-interval coverage, and the board blend vs the model."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from model_testkit import make_league  # noqa: E402

from src.backtest import actuals as act  # noqa: E402
from src.backtest import method_checks as mc  # noqa: E402
from src.backtest.harness import walk_forward  # noqa: E402
from src.models.registry import get_projector  # noqa: E402
from src.value.league import load_league  # noqa: E402

SEASONS = ["2017-18", "2018-19", "2019-20"]
SCORING = load_league()["scoring"]


@pytest.fixture(scope="module")
def tables():
    return make_league()


@pytest.fixture(scope="module")
def result(tables):
    return walk_forward(tables, get_projector("baseline_hurdle"), SEASONS, synthetic=True)


def _adp(tables, noise=0.3):
    rng = np.random.default_rng(0)
    rows = []
    for s in SEASONS:
        a = act.season_actuals(tables["game_logs"], s, SCORING)
        score = a["total_fp"].to_numpy() * (1 + noise * rng.standard_normal(len(a)))
        rows.append(pd.DataFrame({"season": s, "player_id": a["player_id"].to_numpy(),
                                  "adp": pd.Series(-score).rank(method="first").to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def test_the_eval_frame_carries_the_optional_hurdle_and_band_columns(result):
    cols = set(result.players.columns)
    assert {"proj_p_appear", "proj_gp_p10", "proj_gp_p90", "proj_total_fp_p10", "proj_total_fp_p90"} <= cols


def test_a_plain_projector_has_no_optional_columns(tables):
    plain = walk_forward(tables, get_projector("baseline"), SEASONS[:2], synthetic=True)
    assert not {"proj_p_appear", "proj_total_fp_p10"} & set(plain.players.columns)
    assert mc.appearance_calibration(plain) is None
    assert mc.interval_coverage_table(plain, tables["game_logs"], tables["players"]) is None


def test_appearance_calibration_bins_and_brier(result):
    cal = mc.appearance_calibration(result)
    body, brier = cal.iloc[:-1], cal.iloc[-1]
    assert brier["bin"].startswith("Brier") and body["n"].sum() == brier["n"]
    assert ((body["mean_pred"] >= 0) & (body["mean_pred"] <= 1)).all() and ((body["observed"] >= 0) & (body["observed"] <= 1)).all()
    f = result.players
    d = f[f["projected"]]
    assert brier["n"] == len(d)
    assert brier["mean_pred"] == pytest.approx(float(np.mean((d["proj_p_appear"] - d["played"].astype(float)) ** 2)))


def test_interval_coverage_table_counts_actuals_against_the_band(result, tables):
    cov = mc.interval_coverage_table(result, tables["game_logs"], tables["players"])
    allp = cov[cov["group"] == "all projected players"].iloc[0]
    f = result.players
    d = f[f["projected"] & f["proj_total_fp_p10"].notna()]
    inside = ((d["actual_total_fp"] >= d["proj_total_fp_p10"]) & (d["actual_total_fp"] <= d["proj_total_fp_p90"])).mean()
    assert allp["n"] == len(d) and allp["total in band"] == pytest.approx(inside)
    assert allp["below p10"] + allp["above p90"] + allp["total in band"] == pytest.approx(1.0)
    assert 0.0 <= allp["GP in band"] <= 1.0 and allp["mean width (FP)"] > 0
    assert set(cov["group"]) <= {"all projected players", *mc.RISK_GROUPS}


def test_blend_universe_fits_each_season_on_earlier_seasons_only(result, tables, monkeypatch):
    import src.value.adp_blend as AB

    adp = _adp(tables)
    fits = []
    real = AB.fit_adp_blend

    def spy(frames, *, model):
        fits.append(list(frames))
        return real(frames, model=model)

    monkeypatch.setattr(AB, "fit_adp_blend", spy)
    t = mc.blend_universe_table(result, adp, ks=(5, 10))
    assert list(t["metric"]) == ["spearman_total_fp", "top5_hit", "top10_hit", "mae_total_fp"]
    assert t[["model", "blend"]].notna().all().all() and t["seasons won"].str.endswith("/2").all()   # seasons after the first
    assert fits == [SEASONS[:1], SEASONS[:2]]        # season 2 saw only season 1; season 3 saw seasons 1-2, never itself


def test_blend_universe_needs_adp_seasons(result):
    with pytest.raises(ValueError, match="at least two"):
        mc.blend_universe_table(result, pd.DataFrame({"season": ["2010-11"], "player_id": [1], "adp": [1.0]}))


def test_blend_universe_skips_seasons_with_too_little_earlier_adp(result, tables):
    adp = _adp(tables)
    first = adp["season"] == SEASONS[0]
    thin = pd.concat([adp[first].head(40), adp[~first]], ignore_index=True)     # season 1 alone cannot fit the regression
    t = mc.blend_universe_table(result, thin, ks=(5,))
    assert t["seasons won"].str.endswith("/1").all()                            # only the last season is scored
