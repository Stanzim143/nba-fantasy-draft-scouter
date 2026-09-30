"""src.backtest.contract_terms_eval: the pure helpers (subset restriction, the clustered effect-size table, rendering)."""
import numpy as np
import pandas as pd

from src.backtest.contract_terms_eval import GROUPS, effect_table, lift_row, render, restrict, residual_frame, states
from src.backtest.harness import BacktestResult


def _res(players):
    return BacktestResult("x", ["2021-22"], pd.DataFrame({"replacement_level": [1000.0]}, index=["2021-22"]), players, {"ks": (12,), "min_gp": 10})


def test_restrict_keeps_only_the_selected_season_player_rows():
    players = pd.DataFrame({"season": ["2021-22"] * 4, "player_id": [1, 2, 3, 4], "projected": True})
    st = pd.DataFrame({"season": ["2021-22"] * 4, "player_id": [1, 2, 3, 4], "known": [True, False, True, False]})
    out = restrict(_res(players), st, lambda d: d["known"])
    assert list(out.players["player_id"]) == [1, 3]
    assert len(_res(players).players) == 4                       # the original is untouched


def _effect_frame(n=600, shift=1.5, seed=0):
    rng = np.random.default_rng(seed)
    grp = rng.choice(["known_cy", "known_multi", "none"], n, p=[0.2, 0.3, 0.5])
    y = rng.normal(0, 2.5, n) + np.where(grp == "known_cy", shift, 0.0)
    return pd.DataFrame({"season": "2021-22", "player_id": np.arange(n) % 300, "group": grp, "resid_dm": y,
                         "known": grp != "none", "contract_year": grp == "known_cy", "any_wiki": grp != "none",
                         "new_deal": rng.random(n) < 0.2})


def test_effect_table_recovers_a_planted_contract_year_premium_with_a_covering_ci():
    t = effect_table(_effect_frame(), n_boot=300).set_index("group")
    c = t.loc["known final year minus known 2+ years"]
    assert c["lo"] < 1.5 < c["hi"] and c["lo"] > 0                 # the planted +1.5 is inside a CI that excludes zero
    assert set(GROUPS) <= set(t.index)
    null = effect_table(_effect_frame(shift=0.0, seed=1), n_boot=300).set_index("group").loc["known final year minus known 2+ years"]
    assert null["lo"] < 0 < null["hi"]                             # no effect: the CI straddles zero


def test_effect_table_handles_empty_groups():
    f = _effect_frame()
    f = f[f["group"] != "known_cy"]
    t = effect_table(f, n_boot=50).set_index("group")
    assert np.isnan(t.loc["known_cy", "mean_resid"]) and t.loc["known_cy", "n"] == 0
    assert np.isnan(t.loc["known final year minus known 2+ years", "mean_resid"])


def test_lift_row_of_a_result_against_itself_is_zero():
    rng = np.random.default_rng(0)
    n = 80
    players = pd.DataFrame({
        "season": "2021-22", "player_id": np.arange(n), "projected": True, "played": True,
        "proj_total_fp": rng.uniform(200, 2000, n), "proj_fppg": rng.uniform(10, 40, n),
        "actual_total_fp": rng.uniform(200, 2000, n), "actual_fppg": rng.uniform(10, 40, n), "actual_gp": 60.0})
    r = _res(players)
    row = lift_row(r, r, "mae_total_fp", 50)
    assert row["lift"] == 0 and row["verdict"] == "no significant change"


def test_render_produces_the_sections():
    f = _effect_frame(200)
    e = effect_table(f, n_boot=20)
    cov = pd.DataFrame([{"season": "2021-22", "veterans": 10, "known": 4, "no_length": 1, "lapsed": 1, "unknown": 4,
                         "any_event": 6, "contract_year": 1, "new_deal": 2, "extension": 0, "two_way": 0}])
    by = pd.DataFrame([{"group": "none", "n": 5, "bias_base": 0.1, "bias_forced": 0.1, "mae_base": 3.0, "mae_forced": 3.0}])
    rep = {"seasons": ["2021-22"], "n_boot": 20, "leak_check": "passed"}
    lifts = [{"variant": "v", "subset": "league-wide", "metric": "mae_total_fp", "lift": 0.1, "lo": -0.1, "hi": 0.3,
              "p_le0": 0.3, "won": "1/1", "n_players": 10, "verdict": "no significant change"}]
    gate = {"gated": [{"season": "2021-22", "enabled": False, "cv_gain_vs_zero": 0.0001, "reason": "below"}]}
    md = render(rep, cov, lifts, lifts, gate, e, e, e, by)
    for head in ("## Coverage", "## Lift over `baseline`, league-wide", "## Gate history", "## Effect size on covered players",
                 "## Bias and MAE"):
        assert head in md
    assert "future-invariance check passed" in md


def test_states_and_residual_frame_need_real_tables_but_import_cleanly():
    assert callable(states) and callable(residual_frame)
