"""Walk-forward evidence for the rest-of-season blend: does shrinking the preseason projection toward
season-to-date production beat the preseason projection alone (and season-to-date alone)?

    python -m src.inseason.ros_eval --seasons 2016-17:2025-26 --cutoffs 15,30,45,60 [--out reports/ros_eval]

For each season S and each cutoff (the date the median team had played G games) the preseason projection
is built from ``History.until(tables, S)`` and the season-to-date data is cut at that date, exactly as the
live tool sees it. The target is what the player actually scored after the cutoff (players who did not
play score 0, so availability errors count, as they do in a real league).

Universe (decided from information available at the cutoff only): players in the preseason top 300 by
projected total FP, plus anyone with 10+ games to date.

Metrics per (season, cutoff): Spearman rank correlation of predicted vs actual ROS total FP, MAE of ROS
total FP, top-50 hit rate, and MAE of ROS FPPG among players who played 10+ games after the cutoff
(isolates the rate estimate from availability).

Tuning is out of sample: the pseudo-counts are chosen on the *train* seasons (the first half of the
range) and reported on the *test* seasons (the second half), and the full grid is printed too so the
choice is visible rather than hidden.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

from src.backtest.metrics import mae, spearman, topk_overlap
from src.contracts import HISTORY_TABLES, History, season_start, seasons_between
from src.inseason.ros import RosParams, build_ros
from src.inseason.schedule import from_team_games
from src.models.panel import season_lengths, target_season_games
from src.value.frame import fantasy_points_frame
from src.value.league import load_league

DEFAULT_CUTOFFS = (15, 30, 45, 60)
GRID = {"minutes_pseudo": (100.0, 300.0, 750.0, 1500.0, 3000.0), "games_pseudo": (3.0, 10.0, 20.0, 40.0),
        "avail_pseudo": (5.0, 15.0, 30.0, 60.0, 120.0)}
FPPG_GRID = (2.0, 5.0, 10.0, 20.0, 40.0)
UNIVERSE_TOP = 300
UNIVERSE_MIN_GP = 10


def cutoff_dates(team_games: pd.DataFrame, season: str, cutoffs) -> dict[int, pd.Timestamp]:
    """Date on which the median team had played ``g`` games (``team_games`` of ``season``)."""
    tg = team_games[team_games["season"] == season].sort_values(["team_id", "game_date"])
    n = tg.groupby("team_id").size().min()
    nth = tg.assign(k=tg.groupby("team_id").cumcount() + 1)
    out = {}
    for g in cutoffs:
        if g >= n:
            continue
        d = nth[nth["k"] == g]["game_date"]
        out[int(g)] = pd.Timestamp(d.median()).normalize()
    return out


def actual_ros(game_logs: pd.DataFrame, season: str, as_of, scoring) -> pd.DataFrame:
    gl = game_logs[(game_logs["season"] == season) & (pd.to_datetime(game_logs["game_date"]) > pd.Timestamp(as_of))]
    fp = fantasy_points_frame(gl, scoring) if len(gl) else pd.Series(dtype=float)
    a = gl.assign(fp=fp).groupby("player_id").agg(a_games=("game_id", "size"), a_total=("fp", "sum"))
    a["a_fppg"] = a["a_total"] / a["a_games"]
    return a


def universe(ros: pd.DataFrame, season_games: float) -> pd.Series:
    prior_total = ros["prior_fppg"] * ros["prior_avail"] * season_games
    top = ros["has_prior"] & (prior_total.rank(ascending=False, method="first") <= UNIVERSE_TOP)
    return top | (ros["gp"] >= UNIVERSE_MIN_GP)


def score(ros: pd.DataFrame, actual: pd.DataFrame, mask: pd.Series) -> dict:
    r = ros[mask]
    a_total = actual["a_total"].reindex(r["player_id"]).fillna(0.0).to_numpy()
    a_games = actual["a_games"].reindex(r["player_id"]).fillna(0).to_numpy()
    a_fppg = actual["a_fppg"].reindex(r["player_id"]).to_numpy()
    pred = r["ros_total_fp"].to_numpy()
    ok = a_games >= 10
    return {"spearman": spearman(pred, a_total), "mae_total": mae(pred, a_total),
            "top50": topk_overlap(pred, a_total, 50),
            "mae_fppg": mae(r["ros_fppg"].to_numpy()[ok], a_fppg[ok]) if ok.any() else float("nan"),
            "n": int(mask.sum())}


def evaluate(seasons: list[str], cutoffs, configs: dict[str, RosParams], *, tables=None, cfg=None,
             model: str = "baseline", log=lambda s: None) -> pd.DataFrame:
    """One row per (season, cutoff, config name) with the metrics above."""
    cfg = cfg or load_league()
    scoring = cfg["scoring"]
    if tables is None:
        from src.store import load_tables

        tables = dict(load_tables(HISTORY_TABLES))
    from src.models.registry import get_projector

    rows = []
    for season in seasons:
        hist = History.until(tables, season)
        hist.assert_no_future()
        prior = get_projector(model).project(hist)
        season_games = float(target_season_games(season_lengths(hist, None)))
        s_tables = {"game_logs": tables["game_logs"][tables["game_logs"]["season"] == season],
                    "team_games": tables["team_games"][tables["team_games"]["season"] == season],
                    "players": tables["players"], "player_season_bio": tables["player_season_bio"]}
        schedule = from_team_games(tables["team_games"], season)
        dates = cutoff_dates(tables["team_games"], season, cutoffs)
        log(f"{season}: prior built ({len(prior)} players), cutoffs {dates}")
        for g, as_of in dates.items():
            actual = actual_ros(s_tables["game_logs"], season, as_of, scoring)
            for name, params in configs.items():
                ros = build_ros(s_tables, season, as_of, prior=prior, cfg=cfg, schedule=schedule, params=params,
                                season_games=season_games, with_value=False)
                mask = universe(ros, season_games)
                rows.append({"season": season, "cutoff_games": g, "as_of": as_of.date(), "config": name,
                             **score(ros, actual, mask)})
    return pd.DataFrame(rows)


def grid_configs() -> dict[str, RosParams]:
    out = {}
    for m, gp, a in itertools.product(*GRID.values()):
        out[f"blend m{int(m)} g{int(gp)} a{int(a)}"] = RosParams(minutes_pseudo=m, games_pseudo=gp, avail_pseudo=a)
    return out


def choose(results: pd.DataFrame, train: list[str], names: list[str]) -> str:
    """The config with the lowest mean total-FP MAE on the train seasons."""
    t = results[results["season"].isin(train) & results["config"].isin(names)]
    return t.groupby("config")["mae_total"].mean().idxmin()


def summarize(results: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    r = results[results["season"].isin(seasons)]
    return r.groupby("config")[["spearman", "mae_total", "top50", "mae_fppg"]].mean()


def paired(results: pd.DataFrame, a: str, b: str, metric: str, seasons: list[str], lower_is_better: bool) -> dict:
    """Season-level paired comparison of config ``a`` against ``b`` (mean over cutoffs within a season)."""
    r = results[results["season"].isin(seasons)]
    s = r.pivot_table(index="season", columns="config", values=metric, aggfunc="mean")
    d = (s[b] - s[a]) if lower_is_better else (s[a] - s[b])
    return {"mean_gain": float(d.mean()), "seasons_won": int((d > 0).sum()), "seasons": int(len(d))}


def main(argv: list[str] | None = None) -> int:
    from src.inseason._console import use_utf8_console

    use_utf8_console()
    ap = argparse.ArgumentParser(prog="python -m src.inseason.ros_eval", description=__doc__.split("\n\n")[0])
    ap.add_argument("--seasons", default="2016-17:2025-26")
    ap.add_argument("--cutoffs", default=",".join(map(str, DEFAULT_CUTOFFS)))
    ap.add_argument("--out", type=Path, default=None, help="directory for results.csv")
    ap.add_argument("--model", default="baseline")
    args = ap.parse_args(argv)

    a, _, b = args.seasons.partition(":")
    seasons = seasons_between(season_start(a), season_start(b or a))
    cutoffs = [int(x) for x in args.cutoffs.split(",")]
    half = len(seasons) // 2
    train, test = seasons[:half] or seasons, seasons[half:] or seasons

    configs = {"prior only": RosParams(mode="prior"), "season-to-date only": RosParams(mode="sample")}
    configs.update(grid_configs())
    for k in FPPG_GRID:
        configs[f"blend_fppg k{int(k)}"] = RosParams(mode="blend_fppg", fppg_pseudo=k)
    res = evaluate(seasons, cutoffs, configs, model=args.model, log=lambda s: print(s, file=sys.stderr))
    best = choose(res, train, list(grid_configs()))
    best_fppg = choose(res, train, [f"blend_fppg k{int(k)}" for k in FPPG_GRID])
    print(f"train seasons {train[0]}..{train[-1]}, test seasons {test[0]}..{test[-1]}, cutoffs {cutoffs} team games")
    print(f"config chosen on the train seasons: {best};  direct-FPPG variant: {best_fppg}")
    for label, ss in (("TRAIN", train), ("TEST (out of sample)", test), ("ALL", seasons)):
        tab = summarize(res, ss).loc[["prior only", "season-to-date only", best, best_fppg]]
        print(f"\n{label}")
        print(tab.to_string(float_format=lambda x: f"{x:.3f}"))
    for metric, low in (("mae_total", True), ("spearman", False), ("mae_fppg", True)):
        for opp in ("prior only", "season-to-date only"):
            p = paired(res, best, opp, metric, test, low)
            print(f"TEST paired {metric}: blend vs {opp}: mean gain {p['mean_gain']:+.3f}, "
                  f"blend better in {p['seasons_won']}/{p['seasons']} seasons")
    print("\nTEST by cutoff (mean over test seasons):")
    t = res[res["season"].isin(test) & res["config"].isin(["prior only", "season-to-date only", best])]
    print(t.pivot_table(index="cutoff_games", columns="config", values=["spearman", "mae_total"]).to_string(
        float_format=lambda x: f"{x:.3f}"))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.out / "results.csv", index=False)
        print(f"wrote {args.out / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
