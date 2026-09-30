"""Walk-forward mini-check on synthetic data: the baseline must beat the naive last-season benchmark.

This is a genuine statistical test, not a smoke test. For each of several fixed seeds we generate a
20-team synthetic league (10 seasons), and for the last three seasons we project season S from
history strictly before S with both models, then compare against what actually happened in S under
the league scoring. Both models are scored on the players they both cover and who played in S.

Why the baseline should win here (and would be a modelling bug if it did not): the naive model
takes last season's noisy per-game line and games played at face value, so a fluky season or an
injury carries straight into the projection. The baseline shrinks small-sample rates to a
role-appropriate mean, applies learned age curves, blends several seasons, and regresses games
played. Injuries in the synthetic league are random and non-persistent, so availability shrinkage
alone is worth a few points of rank correlation.

Margins were chosen from the observed results over the 9 (seed, season) cells below: Spearman on
season total FP improved in all 9 (mean +0.035, min +0.009), the top-50 hit rate by +0.076 on
average, per-game MAE fell by 0.21 FP; the top-100 hit rate is nearly saturated for both models
(+0.009), so it is only required not to be worse. The assertions demand a pooled advantage well
inside the observed margins (not knife-edge) but clearly above zero. Synthetic data only; no
real-data claim.
"""
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.contracts import History, season_str
from src.models.baseline import BaselineProjector
from src.models.naive import NaiveLastSeason
from src.synthetic import make_synthetic_tables
from src.value.frame import fantasy_points_frame
from src.value.league import load_league

SCORING = load_league()["scoring"]
SEEDS = (1, 2, 3)
TARGET_STARTS = (2017, 2018, 2019)


def _evaluate(seed: int) -> list[dict]:
    t = make_synthetic_tables(2010, 2019, n_teams=20, seed=seed)
    gl = t["game_logs"].copy()
    gl["fp"] = fantasy_points_frame(gl, SCORING)
    actual = gl.groupby(["season", "player_id"])["fp"].agg(total="sum", mean="mean").reset_index()
    rows = []
    for start in TARGET_STARTS:
        season = season_str(start)
        h = History.until(t, season)
        a = actual[actual["season"] == season].set_index("player_id")
        proj = {m.name: m.project(h).set_index("player_id") for m in (BaselineProjector(), NaiveLastSeason())}
        common = proj["baseline"].index.intersection(proj["naive_last_season"].index).intersection(a.index)
        for name, p in proj.items():
            pp, aa = p.loc[common], a.loc[common]
            row = dict(seed=seed, season=season, model=name, n=len(common),
                       rho_total=spearmanr(pp["proj_total_fp"], aa["total"]).statistic,
                       rho_fppg=spearmanr(pp["proj_fppg"], aa["mean"]).statistic,
                       mae_fppg=float((pp["proj_fppg"] - aa["mean"]).abs().mean()))
            for n in (50, 100):
                top_actual = set(aa["total"].nlargest(n).index)
                row[f"top{n}"] = len(top_actual & set(pp["proj_total_fp"].nlargest(n).index)) / n
            rows.append(row)
    return rows


@pytest.fixture(scope="module")
def results() -> pd.DataFrame:
    return pd.DataFrame([r for s in SEEDS for r in _evaluate(s)])


def _diff(df: pd.DataFrame, col: str) -> pd.Series:
    piv = df.pivot_table(index=["seed", "season"], columns="model", values=col)
    return piv["baseline"] - piv["naive_last_season"]


def test_baseline_beats_naive_on_rank_correlation(results):
    d = _diff(results, "rho_total")
    assert d.mean() > 0.02, f"mean Spearman lift {d.mean():.3f}"
    assert (d > 0).mean() >= 0.8, "baseline should win the large majority of individual seasons"
    assert _diff(results, "rho_fppg").mean() > 0.005


def test_baseline_beats_naive_on_top_n_hit_rate(results):
    d50, d100 = _diff(results, "top50"), _diff(results, "top100")
    assert d50.mean() > 0.03, f"top-50 lift {d50.mean():.3f}"
    assert d100.mean() > -0.01, f"top-100 hit rate should not be worse (lift {d100.mean():.3f})"


def test_baseline_has_lower_per_game_error(results):
    assert _diff(results, "mae_fppg").mean() < 0.0


def test_baseline_is_absolutely_good(results):
    b = results[results["model"] == "baseline"]
    assert b["rho_total"].min() > 0.80
    assert b["top50"].mean() > 0.55
