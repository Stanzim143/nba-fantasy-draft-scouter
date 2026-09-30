"""Load-management proxy study (ADR 0024): universe, regression on planted effects, the decision rule, walk-forward without look-ahead."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest import load_report as LR

SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2016, 2026)]
DERIVED = ("in_universe", "err_gp", "err_fp", "ae_gp", "ae_fp", "above", "err_fppg", "age2", "proj_f", "hm_vet", "hi_iso")


def _frame(effect: float, *, n_per=120, seed=0, prepared=True):
    """A synthetic study frame in which err_gp = effect * z(iso) + noise (+ a general bias of -3 GP)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in SEASONS:
        iso = rng.poisson(3.0, n_per).astype(float)
        f = rng.uniform(0.55, 0.95, n_per)
        age = rng.uniform(22, 36, n_per)
        mpg = rng.uniform(20, 36, n_per)
        proj_gp = np.clip(f * 82 - 3 + rng.normal(0, 3, n_per), 10, 80)
        z = (iso - 3.0) / np.sqrt(3.0)
        err = -3.0 + effect * z + rng.normal(0, 10, n_per)
        fppg = rng.uniform(20, 45, n_per)
        act = np.clip(proj_gp + err, 0, 82)
        rows.append(pd.DataFrame({
            "season": s, "player_id": np.arange(n_per), "player_name": "x", "age": age, "proj_gp": proj_gp, "proj_fppg": fppg,
            "proj_total_fp": proj_gp * fppg, "actual_gp": act, "actual_total_fp": act * fppg, "actual_fppg": fppg, "target_len": 82.0,
            "team_id": 1, "n_teams": 1, "L": 82, "gp": np.round(f * 82), "mpg": mpg, "missed": 82 - np.round(f * 82), "iso_n": iso,
            "iso_runs": iso, "rest_n": 0, "iso_n82": iso, "rest_n82": rng.poisson(0.7, n_per).astype(float),
            "iso_frac": rng.uniform(0, 1, n_per), "near65": False, "veteran": True, "f": f}))
    raw = pd.concat(rows, ignore_index=True)
    return LR.prepare(raw) if prepared else raw


def test_universe_applies_the_preregistered_filters():
    raw = _frame(0.0, n_per=20, prepared=False)
    raw["veteran"] = raw["veteran"].astype(object)
    raw.loc[0, "veteran"] = False
    raw.loc[1, "n_teams"] = 2
    raw.loc[2, "mpg"] = 19.0
    raw.loc[3, "gp"] = 40
    raw.loc[4, "age"] = np.nan
    f = LR.prepare(raw)
    assert not f.loc[[0, 1, 2, 3, 4], "in_universe"].any() and f.loc[5:19, "in_universe"].all()


def test_prepare_signs_errors_as_actual_minus_projected_and_flags_the_cells():
    f = _frame(0.0, n_per=10)
    assert (f["err_gp"] == f["actual_gp"] - f["proj_gp"]).all() and (f["err_fp"] == f["actual_total_fp"] - f["proj_total_fp"]).all()
    assert (f["hi_iso"] == (f["iso_n"] >= LR.HI_ISO)).all() and (f["hm_vet"] == ((f["age"] >= 31) & (f["mpg"] >= 30))).all()


def test_regression_recovers_a_planted_effect_with_a_ci_that_excludes_zero():
    f = _frame(-2.0)
    est, lo, hi = LR.proxy_effect(f, "iso_n82", "err_gp", n_boot=300)
    assert lo < est < hi and hi < 0 and est == pytest.approx(-2.0, abs=0.6)
    assert LR.proxy_effect(f, "iso_n82", "err_fp", n_boot=300)[2] < 0
    ps = LR.per_season_effect(f, "iso_n82", "err_gp")
    assert len(ps) == 10 and LR.sign_agreement(ps, est)[0] >= 8


def test_regression_with_no_effect_has_a_ci_around_zero():
    f = _frame(0.0, seed=3)
    est, lo, hi = LR.proxy_effect(f, "iso_n82", "err_gp", n_boot=300)
    assert lo < 0 < hi and abs(est) < 0.8


def test_general_bias_alone_is_not_a_proxy_effect():
    """The fixed effects and controls absorb a uniform over-projection: the proxy coefficient stays about zero."""
    f = _frame(0.0, seed=5)
    assert f["err_gp"].mean() == pytest.approx(-3.0, abs=0.5)
    assert abs(LR.proxy_effect(f, "iso_n82", "err_gp", n_boot=100)[0]) < 0.8


def test_flag_effect_is_per_flagged_player_not_per_sd():
    f = _frame(-2.0)
    assert LR.proxy_effect(f, "hi_iso", "err_gp", n_boot=100)[0] < -1.0, "6+ isolated absences is about z=+1.7, so well below zero at -2 per SD"


def test_decision_rule_is_the_fixed_three_part_conjunction():
    sig = (-50.0, -90.0, -10.0)
    assert LR.decide(sig, (9, 10), (5.0, 1.0, 9.0)) == "incremental"
    assert LR.decide(sig, (9, 10), (-14.0, -22.0, -6.0)) == "descriptive association"
    assert LR.decide(sig, (9, 10), (5.0, -1.0, 9.0)) == "descriptive association"
    assert LR.decide(sig, (6, 10), (5.0, 1.0, 9.0)) == "null", "sign consistency fails: not even descriptive"
    assert LR.decide((-5.0, -40.0, 30.0), (10, 10), (5.0, 1.0, 9.0)) == "null", "CI includes zero"
    assert LR.decide((float("nan"),) * 3, (0, 0), (float("nan"),) * 3) == "null"


def test_walk_forward_correction_trains_only_on_earlier_seasons():
    f = _frame(-2.0)
    wf = LR.walk_forward_correction(f, n_boot=100)
    t = wf["frame"]
    assert set(t["season"]) == set(SEASONS[3:]) and wf["n"] == len(t)
    g = f.copy()          # replace every season after 2020-21 (targets and outcomes) with garbage
    later = g["season"] > "2020-21"
    g.loc[later, "err_gp"] = 999.0
    g.loc[later, "actual_gp"] = g.loc[later, "proj_gp"] + 999.0
    t2 = LR.walk_forward_correction(g, n_boot=50)["frame"]
    a = t[t["season"] <= "2020-21"].reset_index(drop=True)
    b = t2[t2["season"] <= "2020-21"].reset_index(drop=True)
    np.testing.assert_allclose(a["corr_gp"], b["corr_gp"])
    assert (t["gp_c"] >= 0).all() and (t["gp_c"] <= t["target_len"]).all() and np.allclose(t["fp_c"], t["gp_c"] * t["proj_fppg"])


def test_a_large_planted_effect_improves_out_of_sample_error_over_controls_only():
    inc = LR.increment_over_controls(_frame(-4.0), n_boot=200)
    assert inc["n"] > 0 and inc["gp"][1] > 0, "with a large true effect the proxy gains out of sample"
    null = LR.increment_over_controls(_frame(0.0, seed=9), n_boot=200)
    assert null["gp"][1] < 0.2 and null["gp"][2] > -0.2


def test_bins_partition_the_universe():
    f = _frame(0.0, n_per=60)
    b = LR.bin_table(f, n_boot=50)
    assert b["n"].sum() == int(f["in_universe"].sum()) and list(b["iso_n"]) == ["0-1", "2-3", "4-5", "6+"]


def test_study_and_render_smoke():
    f = _frame(-2.0, n_per=80)
    res = LR.study(f, n_boot=60)
    assert res["verdict"] in {"incremental", "descriptive association", "null"} and res["n"] == int(f["in_universe"].sum())
    text = LR.render(f, res, {"other": res}, LR.proxy_table(f, n_boot=40), LR.bin_table(f, n_boot=40), LR.era_table(f, n_boot=40),
                     model="baseline", command="cmd", n_boot=60)
    assert "Verdict under the pre-registered rule" in text and "iso_n82 (primary)" in text and "Post-hoc diagnostics" in text
    assert "contaminated by minor injuries" in text


def test_build_frame_end_to_end_on_the_synthetic_league():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
    from model_testkit import make_league
    from src.models.baseline import BaselineProjector
    from src.value.league import load_league

    tables = make_league()
    seasons = sorted(tables["game_logs"]["season"].unique())[-2:]
    f = LR.build_load_frame(tables, seasons, BaselineProjector(), load_league()["scoring"])
    assert set(f["season"]) <= set(seasons) and len(f) > 0
    assert {"iso_n", "rest_n82", "err_gp", "in_universe"} <= set(LR.prepare(f).columns)
