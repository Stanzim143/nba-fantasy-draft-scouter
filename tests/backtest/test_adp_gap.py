"""ADP-gap study (docs/research/adp-gap-study.md): synthetic checks so the study script does not rot."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bt_helpers import make_logs
from src.backtest import adp_gap as G

SEASONS = ["2016-17", "2017-18", "2018-19", "2019-20", "2020-21"]


def _frame(seed: int = 0, n: int = 160, young_noise: float = 400.0) -> pd.DataFrame:
    """Listed, projected players whose model is badly wrong for the young and ADP is not (ADP knows more)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in SEASONS:
        true = rng.gamma(4.0, 400.0, n)
        young = rng.random(n) < 0.3
        model = true + rng.normal(0, 150, n) + np.where(young, rng.normal(0, young_noise, n), 0.0)
        adp_score = true + rng.normal(0, 200, n)
        adp = pd.Series(-adp_score).rank().to_numpy()
        rows.append(pd.DataFrame({
            "season": s, "player_id": np.arange(n), "player_name": [f"P{i}" for i in range(n)], "projected": True,
            "played": True, "proj_total_fp": model, "proj_gp": 70.0, "actual_total_fp": true + rng.normal(0, 100, n),
            "actual_gp": 70.0, "adp": adp, "exp": np.where(young, 0, 5)}))
    return pd.concat(rows, ignore_index=True)


def _desc(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame[["season", "player_id", "exp"]].copy()
    d["age"], d["prior_gp_share"], d["team_change"], d["pre_fppg_delta"], d["contract_year"] = 25.0, 0.9, False, 0.0, False
    return d


def test_rank_delta_sign_and_winner():
    f = pd.DataFrame({"season": "2020-21", "player_id": [1, 2, 3], "projected": True,
                      "proj_total_fp": [100.0, 300.0, 200.0], "actual_total_fp": [300.0, 100.0, 200.0],
                      "adp": [1.0, 3.0, 2.0]})
    d = G.disagreement_frame(f).set_index("player_id")
    assert d.loc[1, "rank_delta"] == 2            # model rank 3, ADP rank 1: ADP higher
    assert d.loc[1, "direction"] == "ADP higher"
    assert d.loc[1, "adp_right"] == 1.0 and d.loc[2, "adp_right"] == 1.0
    assert d.loc[3, "adp_right"] == 0.5           # all three agree
    assert G.disagreement_frame(f.assign(adp=np.nan)).empty


def test_win_tables_find_the_planted_segment():
    f = _frame()
    d = G.disagreement_frame(f.drop(columns="exp"), _desc(f))
    t = G.win_tables(d, min_gap=10)
    seg = t["segment"].set_index("segment")
    assert seg.loc["rookie", "adp_right"] > seg.loc["other", "adp_right"]
    assert {"tier", "direction", "overall", "segment_direction"} <= set(t)


def test_corrections_walk_forward_and_lift():
    f = _frame()
    desc = _desc(f)
    arms = G.corrected_totals(f.drop(columns="exp"), desc, min_rows=150)
    assert "2016-17" not in set(arms["season"])          # nothing earlier to fit on
    for a in G.ARMS:
        assert arms[f"arm_{a}"].notna().all()
    # the shrink arm only re-orders the blend's totals within each season
    for _, g in arms.groupby("season"):
        assert np.allclose(np.sort(g["arm_blend"]), np.sort(g["arm_blend_shrink"]))
    full = f.drop(columns="exp")[f["season"].isin(arms["season"].unique())]
    res = G.as_results(arms, full)
    t = G.lift_table(res, "blend", ["blend_feat"], ["spearman_total_fp", "top50_hit"], n_boot=50)
    assert set(t["metric"]) == {"spearman_total_fp", "top50_hit"}
    assert t["lift"].notna().all()


def test_shrink_weights_follow_the_planted_signal():
    f = _frame(young_noise=1500.0)
    d = G.disagreement_frame(f.drop(columns="exp"), _desc(f))
    w = G.fit_shrink_weights(d)
    flagged = [v for k, v in w.items() if k.endswith("|flag")]
    plain = [v for k, v in w.items() if k.endswith("|none")]
    assert np.mean(flagged) > np.mean(plain)


def test_player_descriptors_on_tiny_logs():
    gl = pd.concat([make_logs([(1, 40, 20, 5, 5), (2, 70, 10, 5, 5)], season="2019-20", team_id=1),
                    make_logs([(1, 50, 20, 5, 5)], season="2020-21", team_id=2),
                    make_logs([(2, 60, 10, 5, 5), (3, 60, 10, 5, 5)], season="2020-21", team_id=1)],
                   ignore_index=True)
    players = pd.DataFrame({"player_id": [1, 2, 3], "birthdate": pd.to_datetime(["1988-01-01", "1999-01-01",
                                                                                  "2001-01-01"]),
                            "draft_year": [2008, 2019, 2020], "draft_round": [1, 2, 1], "draft_number": [3, 40, 10]})
    frame = pd.DataFrame({"season": "2020-21", "player_id": [1, 2, 3]})
    d = G.player_descriptors(frame, gl, players).set_index("player_id")
    assert d.loc[1, "team_change"] and not d.loc[2, "team_change"]
    assert d.loc[3, "exp"] == 0 and d.loc[1, "exp"] == 1
    assert d.loc[1, "prior_gp_share"] == pytest.approx(40 / 82)
    assert np.isnan(d.loc[3, "prior_gp_share"])
    assert d.loc[1, "age"] == pytest.approx(32.75, abs=0.01)
