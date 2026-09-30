"""The debutant / rookie-origin / preseason-availability evaluation summaries, on hand-made frames."""
import numpy as np
import pandas as pd

from src.backtest import debutants as bd
from src.backtest import preseason_availability as pa
from src.backtest import rookie_origin as ro


def frame(n=40, seed=0):
    y = np.random.default_rng(seed).gamma(2, 100, n)
    return pd.DataFrame({"season": ["2020-21"] * (n // 2) + ["2021-22"] * (n - n // 2), "klass": ["stash", "undrafted"] * (n // 2),
                         "actual_total": y, "actual_gp": (y > 60) * 20.0, "actual_mpg": 12.0, "actual_fppg": 9.0,
                         "omit_total": 0.0, "prior_total": y * 1.1, "prior_p_total": y * 1.05, "class_mean_total": y.mean(),
                         "model_total": y * 0.9, "model_mpg": 12.0, "prior_mpg": 13.0, "model_fppg": 9.0, "prior_fppg": 9.5,
                         "p_play": 0.5, "years_since_draft": 2.0, "age": 22.0, "intl": 1.0, "pick": 30.0})


def test_debutant_summary_orders_treatments_sensibly():
    s = bd.summarise(frame())
    assert s["stash"]["n"] == 20 and set(s) >= {"all", "stash", "undrafted"}
    assert s["all"]["model"]["mae"] < s["all"]["prior"]["mae"]            # y*0.9 is closer than y*1.1
    c = s["all"]["model_minus_prior_mae"]
    assert c["ci95"][0] <= c["mean"] <= c["ci95"][1] and c["mean"] < 0
    assert "| model |" in bd.render(s)


def test_rookie_origin_summary_and_render():
    y = np.random.default_rng(1).gamma(2, 150, 60)
    d = pd.DataFrame({"season": ["2020-21"] * 30 + ["2021-22"] * 30, "actual_total": y, "actual_gp": 20.0, "actual_fppg": 10.0,
                      "intl": 0.0, "has_combine": 1.0})
    for v in ro.VARIANTS:
        k = 1.3 if v == "base" else 1.1     # the extra features pull the prediction closer to the outcome
        d[f"{v}_total"], d[f"{v}_fppg"] = y * k, 10.0 * k
    s = ro.summarise(d)
    assert s["origin"]["mae_total_minus_base"]["mean"] < 0 and s["origin"]["seasons_improved"] == 2
    assert "| origin |" in ro.render(s)


def test_preseason_availability_summary_flags_the_confound():
    d = pd.DataFrame({"season": ["2019-20"] * 20 + ["2020-21"] * 20, "s": [2019] * 20 + [2020] * 20, "pre_flag": ["dnp_all"] * 40,
                      "proj_gp": 40.0, "proj_fppg": 30.0, "proj_mpg": 25.0, "team_pre_games": 4.0, "actual_gp": 0.0})
    s = pa.summarise(d)
    assert s["dnp_all"]["share_played_zero"] == 1.0
    assert "Not a validation" in pa.render(s)
