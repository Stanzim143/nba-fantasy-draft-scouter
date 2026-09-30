"""The ROS walk-forward evaluation: mechanics on the synthetic league, selection and pairing on fabricated rows."""
import numpy as np
import pandas as pd
import pytest
from ise_testkit import SEASON, make_tables

from src.inseason import ros_eval as E
from src.inseason.ros import RosParams


@pytest.fixture(scope="module")
def tables():
    return make_tables()


def test_cutoff_dates_are_the_median_teams_nth_game_and_skip_impossible_cutoffs(tables):
    d = E.cutoff_dates(tables["team_games"], SEASON, [10, 20, 999])
    assert set(d) == {10, 20} and d[10] < d[20]


def test_evaluate_produces_one_row_per_season_cutoff_and_config(tables):
    cfgs = {"prior only": RosParams(mode="prior"), "sample": RosParams(mode="sample"), "blend": RosParams()}
    res = E.evaluate([SEASON], [15, 25], cfgs, tables=tables)
    assert len(res) == 2 * 3 and set(res["config"]) == set(cfgs)
    assert res[["spearman", "mae_total", "top50", "mae_fppg"]].notna().all().all()
    piv = res.pivot_table(index="config", values=["spearman", "mae_total"], aggfunc="mean")
    # the synthetic league has a true latent skill, so season-to-date data must carry information the prior lacks
    assert piv.loc["blend", "spearman"] > piv.loc["prior only", "spearman"]
    assert piv.loc["blend", "mae_total"] < piv.loc["prior only", "mae_total"]


def test_actual_ros_counts_only_games_after_the_cutoff(tables):
    cut = E.cutoff_dates(tables["team_games"], SEASON, [15])[15]
    a = E.actual_ros(tables["game_logs"][tables["game_logs"]["season"] == SEASON], SEASON, cut, {"PTS": 1})
    gl = tables["game_logs"]
    after = gl[(gl["season"] == SEASON) & (gl["game_date"] > cut)]
    assert a["a_games"].sum() == len(after) and a["a_total"].sum() == after["pts"].sum()


def _fake():
    rows = []
    for season, mae_a, mae_b in (("s1", 10, 12), ("s2", 11, 10), ("s3", 9, 12)):
        for cfg, m in (("A", mae_a), ("B", mae_b), ("C", 20)):
            rows.append({"season": season, "cutoff_games": 15, "config": cfg, "mae_total": m, "spearman": 1 / m,
                         "top50": 0.5, "mae_fppg": m})
    return pd.DataFrame(rows)


def test_choose_uses_only_train_seasons_and_paired_counts_seasons():
    res = _fake()
    assert E.choose(res, ["s1", "s3"], ["A", "B"]) == "A"          # train mean 9.5+... A beats B on s1 and s3
    assert E.choose(res, ["s2"], ["A", "B"]) == "B"
    p = E.paired(res, "A", "B", "mae_total", ["s1", "s2", "s3"], True)
    assert p["seasons_won"] == 2 and p["seasons"] == 3 and p["mean_gain"] == pytest.approx((2 - 1 + 3) / 3)
    hi = E.paired(res, "A", "B", "spearman", ["s1", "s2", "s3"], False)
    assert hi["seasons_won"] == 2
    tab = E.summarize(res, ["s1"])
    assert list(tab.columns) == ["spearman", "mae_total", "top50", "mae_fppg"] and np.isfinite(tab.loc["A", "mae_total"])


def test_grid_contains_the_defaults_and_universe_uses_information_at_the_cutoff(tables):
    names = E.grid_configs()
    d = RosParams()
    assert f"blend m{int(d.minutes_pseudo)} g{int(d.games_pseudo)} a{int(d.avail_pseudo)}" in names
    from src.inseason.ros import build_ros

    ros = build_ros(tables, SEASON, E.cutoff_dates(tables["team_games"], SEASON, [15])[15], with_value=False)
    m = E.universe(ros, 40.0)
    assert m.sum() <= E.UNIVERSE_TOP + int((ros["gp"] >= E.UNIVERSE_MIN_GP).sum())
