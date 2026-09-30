"""Breakout evaluation: metric helpers, flags, subgroup accuracy, detection statistics, the report and the CLI."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from src.backtest import breakouts as bo  # noqa: E402
from src.backtest.metrics import default_n_rostered  # noqa: E402

CFG = bo.BreakoutConfig()


# --------------------------------------------------------------------------- config

def test_the_rostered_pool_matches_the_backtests_default_replacement_pool():
    assert CFG.rostered_rank == default_n_rostered()                  # teams x roster spots from league.yaml


# --------------------------------------------------------------------------- pure helpers

def test_breakout_needs_both_the_absolute_and_the_relative_margin():
    base = np.array([10.0, 10.0, 30.0, 30.0, 10.0])
    actual = np.array([14.0, 13.9, 34.0, 38.0, np.nan])
    #  +4 and +40%: yes | +3.9: no | +4 but only +13% of 30: no | +8 and +27%: yes | missing: no
    np.testing.assert_array_equal(bo.breakout_flag(actual, base), [True, False, False, True, False])


def test_auc_known_values():
    assert bo.auc([3, 2, 1, 0], [True, True, False, False]) == 1.0
    assert bo.auc([0, 1, 2, 3], [True, True, False, False]) == 0.0
    assert bo.auc([1, 1, 1, 1], [True, False, True, False]) == 0.5           # all ties
    assert np.isnan(bo.auc([1, 2, 3], [True, True, True])) and np.isnan(bo.auc([1, 2], [False, False]))
    assert bo.auc([np.nan, 3, 1], [True, True, False]) == 1.0                # non-finite scores are ignored


def test_auc_matches_the_pairwise_definition():
    rng = np.random.default_rng(0)
    s, y = rng.normal(size=60), rng.random(60) < 0.4
    pos, neg = s[y], s[~y]
    pairs = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    assert bo.auc(s, y) == pytest.approx(pairs)


def test_precision_at_k():
    s = [9, 8, 7, 1, 0]
    y = [True, False, True, True, False]
    assert bo.precision_at_k(s, y, 2) == 0.5 and bo.precision_at_k(s, y, 3) == pytest.approx(2 / 3)
    assert bo.precision_at_k(s, y, 50) == pytest.approx(3 / 5)               # fewer rows than k: uses what exists
    assert np.isnan(bo.precision_at_k([], [], 3))


def test_bootstrap_ci_is_deterministic_ordered_and_handles_empty_input():
    v = np.random.default_rng(1).normal(0.3, 1.0, 80)
    a, b = bo.bootstrap_mean_ci(v), bo.bootstrap_mean_ci(v)
    assert a == b and a[1] < a[0] < a[2] and a[0] == pytest.approx(v.mean())
    assert all(np.isnan(x) for x in bo.bootstrap_mean_ci([np.nan]))


def test_verdicts():
    assert bo.verdict(0.01, 0.2) == "improves" and bo.verdict(-0.2, -0.01) == "hurts"
    assert bo.verdict(-0.1, 0.1) == "no significant change" and bo.verdict(np.nan, 1) == "n/a"


# --------------------------------------------------------------------------- synthetic evaluation frame

def make_frame(seed=0, n_per_season=160, seasons=("2020-21", "2021-22", "2022-23", "2023-24"), skill=1.0, layer_helps=True):
    """Young/old players; the layer's uplift (``score``) genuinely predicts breakouts when ``skill`` > 0."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        n = n_per_season
        age = rng.uniform(19, 33, n)
        base = rng.uniform(8, 32, n)
        uplift_true = rng.normal(0, 5, n) * (age < 24)             # some young players truly jump (or fall)
        actual = base + uplift_true + rng.normal(0, 2, n)
        score = skill * uplift_true + rng.normal(0, 1.5, n) * (age < 24)
        gp = rng.integers(0, 82, n)
        layer = base + score if layer_helps else base
        total = actual * gp
        rows.append(pd.DataFrame({
            "season": s, "player_id": rng.permutation(10_000)[:n] + 1, "player_name": [f"P{i}" for i in range(n)],
            "base_fppg": base, "layer_fppg": layer, "actual_fppg": np.where(gp > 0, actual, np.nan), "actual_gp": gp,
            "actual_total_fp": total, "age": age, "is_rookie": (age < 21) & (rng.random(n) < 0.8),
            "experience": np.where(age < 21, 0, rng.integers(1, 10, n)), "adp_rank": np.where(rng.random(n) < 0.5, rng.integers(1, 200, n), np.nan),
            "sl_gp": np.where(rng.random(n) < 0.3, 4, np.nan), "pre_gp": np.where(rng.random(n) < 0.6, 3, np.nan),
        }))
    f = pd.concat(rows, ignore_index=True)
    f["actual_rank"] = f.groupby("season")["actual_total_fp"].rank(ascending=False, method="first")
    return f


def test_a_breakout_requires_enough_games_played():
    raw = make_frame()
    raw.loc[0, ["base_fppg", "actual_fppg", "actual_gp"]] = [10.0, 30.0, CFG.min_gp - 1]      # a huge jump over 19 games
    raw.loc[1, ["base_fppg", "actual_fppg", "actual_gp"]] = [10.0, 30.0, CFG.min_gp]
    f = bo.with_flags(raw)
    assert not f.loc[0, "breakout"] and f.loc[1, "breakout"]
    assert not f.loc[0, "useful"]


def test_flags_are_defined_as_documented():
    f = bo.with_flags(make_frame())
    assert (f["played"] == (f["actual_gp"] >= CFG.min_gp)).all()
    assert (f["young"] == (f["age"] <= CFG.young_age)).all()
    assert (f["radar"] == (f["adp_rank"].isna() | (f["adp_rank"] > CFG.radar_rank))).all()
    assert (f["useful"] == (f["breakout"] & (f["actual_rank"] <= CFG.rostered_rank))).all()
    assert f["useful"].sum() <= f["breakout"].sum()
    np.testing.assert_allclose(f["score"], f["layer_fppg"] - f["base_fppg"])
    assert (f["has_event"] == (f[["sl_gp", "pre_gp"]].fillna(0).sum(axis=1) > 0)).all()


def test_a_score_that_carries_signal_is_detected():
    f = bo.with_flags(make_frame(skill=1.0))
    det = bo.breakout_detection(f, n_boot=500).set_index("population")
    assert det.at["young", "auc"] > 0.7 and det.at["young", "auc_verdict"] == "improves"
    assert det.at["young", "p@10"] > det.at["young", "base_rate"] and det.at["young", "verdict@10"] == "improves"
    assert det.at["young", "seasons"] == 4
    assert {"young and under the radar", "rookies"} <= set(det.index)


def test_a_score_without_signal_is_not_credited():
    f = bo.with_flags(make_frame(skill=0.0))
    det = bo.breakout_detection(f, n_boot=500).set_index("population")
    assert 0.4 < det.at["young", "auc"] < 0.6
    assert det.at["young", "auc_verdict"] in ("no significant change", "improves", "hurts")
    assert det.at["young", "auc_lo"] < 0.55


def test_detection_supports_the_useful_target():
    f = bo.with_flags(make_frame())
    plain = bo.breakout_detection(f, n_boot=200).set_index("population")
    useful = bo.breakout_detection(f, target="useful", n_boot=200).set_index("population")
    assert len(useful) >= 1
    for pop in useful.index:                                  # a useful breakout is a breakout, so it is rarer
        assert useful.at[pop, "base_rate"] <= plain.at[pop, "base_rate"] + 1e-12


def test_seasons_without_both_outcomes_are_skipped():
    raw = make_frame()
    quiet = raw["season"] == "2020-21"
    raw.loc[quiet, "actual_fppg"] = raw.loc[quiet, "base_fppg"] - 5          # nobody breaks out that season
    det = bo.breakout_detection(bo.with_flags(raw), n_boot=100).set_index("population")
    assert det.at["young", "seasons"] == 3


def test_subgroup_accuracy_credits_a_layer_that_is_closer_to_the_truth():
    good = bo.subgroup_accuracy(bo.with_flags(make_frame(skill=1.0, n_per_season=400)), n_boot=500).set_index("subgroup")
    assert good.at["all players", "mae_gain"] > 0 and good.at["young (<= 23)", "verdict"] == "improves"
    assert good.at["all players", "mae_layer"] < good.at["all players", "mae_base"]
    flat = bo.subgroup_accuracy(bo.with_flags(make_frame(layer_helps=False)), n_boot=500).set_index("subgroup")
    assert flat.at["all players", "mae_gain"] == 0 and flat.at["all players", "verdict"] == "no significant change"
    assert set(good.index) >= {"rookies", "young and under the radar", "young, second year or earlier"}


def test_the_flagged_list_is_sorted_and_labels_the_outcome():
    f = bo.with_flags(make_frame())
    flagged = bo.flagged_by_season(f, top=5)
    assert flagged.groupby("season").size().max() <= 5
    for _, g in flagged.groupby("season"):
        assert g["score"].is_monotonic_decreasing and (g["score"] > 0).all()
    assert set(flagged["outcome"]) <= {"USEFUL BREAKOUT", "breakout, not rosterable", "no breakout", "did not play enough"}
    used = flagged[flagged["outcome"] == "USEFUL BREAKOUT"]
    assert (used["actual_gp"] >= CFG.min_gp).all()


def test_report_contains_every_section_and_the_reproduce_command():
    f = bo.with_flags(make_frame())
    md = bo.render_report(f, CFG, layered_name="baseline_offseason", command="python -m src.backtest.breakouts --x", n_boot=200, top=3)
    for text in ("# Breakout evaluation: baseline_offseason vs baseline", "python -m src.backtest.breakouts --x", "## Definitions",
                 "## FPPG accuracy by subgroup", "## Breakout detection", "### Useful breakouts", "## Who the layer flagged each season",
                 "Useful breakout", "Under the radar"):
        assert text in md


# --------------------------------------------------------------------------- build_frame on a synthetic league

class FakeProjector:
    """Projects last season's FPPG (optionally bumped) for everyone who played then."""

    def __init__(self, bump=0.0, name="fake"):
        self.bump, self.name = bump, name

    def project(self, history):
        gl = history.game_logs
        last = gl[gl["season"] == history.last_season]
        g = last.groupby("player_id").agg(name=("player_name", "last"), gp=("game_id", "size"), pts=("pts", "mean"),
                                          reb=("reb", "mean"), ast=("ast", "mean")).reset_index()
        fppg = g["pts"] + g["reb"] + 2 * g["ast"] + self.bump
        return pd.DataFrame({"player_id": g["player_id"], "player_name": g["name"], "proj_fppg": fppg, "proj_gp": 60.0,
                             "proj_total_fp": fppg * 60.0, "age": 24.0, "is_rookie": False,
                             "offseason_adj": self.bump, "sl_gp": np.nan, "pre_gp": 3.0})


def test_build_frame_joins_projections_outcomes_experience_and_adp():
    from model_testkit import make_league

    from src.value.league import load_league

    tables = make_league()
    scoring = load_league()["scoring"]
    gl = tables["game_logs"]
    projected = gl[gl["season"] == "2017-18"]["player_id"].unique()[:3]           # players the fake projector will project
    adp = pd.DataFrame({"season": ["2018-19"] * 3, "player_id": projected, "adp": [5.0, 1.0, 9.0]})
    f = bo.build_frame(tables, ["2018-19"], FakeProjector(), FakeProjector(bump=2.0), scoring, adp)
    assert set(f["season"]) == {"2018-19"} and f["player_id"].is_unique
    np.testing.assert_allclose(f["layer_fppg"] - f["base_fppg"], 2.0)
    assert (f["actual_gp"] >= 0).all() and f["experience"].max() >= 1
    assert f["adp_rank"].notna().sum() == 3 and sorted(f["adp_rank"].dropna()) == [1.0, 2.0, 3.0]
    played = f[f["actual_gp"] > 0]
    assert played["actual_rank"].notna().all() and played["actual_rank"].min() == 1
    assert f.loc[f["actual_gp"] == 0, "actual_rank"].isna().all()
    flagged = bo.with_flags(f)
    assert {"played", "young", "radar", "breakout", "useful", "score"} <= set(flagged.columns)


def test_build_frame_without_adp_marks_everyone_under_the_radar():
    from model_testkit import make_league

    from src.value.league import load_league

    f = bo.build_frame(make_league(), ["2018-19"], FakeProjector(), FakeProjector(), load_league()["scoring"])
    assert f["adp"].isna().all() and bo.with_flags(f)["radar"].all()


# --------------------------------------------------------------------------- CLI

def test_cli_writes_a_report_and_the_frame(tmp_path, monkeypatch):
    from model_testkit import make_league

    from src.contracts import HISTORY_TABLES
    from src.store import write_table

    data = tmp_path / "data"
    tables = make_league()
    for name in HISTORY_TABLES:
        write_table(tables[name], name, data)
    monkeypatch.setenv("NBA_DATA_DIR", str(data))
    out = tmp_path / "reports"
    code = bo.main(["--seasons", "2018-19", "--model", "baseline", "--run-id", "t", "--out", str(out), "--n-boot", "50"])
    assert code == 0
    md = (out / "t" / "breakouts.md").read_text(encoding="utf-8")
    assert "# Breakout evaluation: baseline vs baseline" in md and "python -m src.backtest.breakouts" in md
    frame = pd.read_parquet(out / "t" / "breakout_frame.parquet")
    assert {"useful", "breakout", "score"} <= set(frame.columns) and (frame["season"] == "2018-19").all()
