"""Breakout evaluation: does the offseason layer find young players who beat their projection? (ADR 0012)

    python -m src.backtest.breakouts --seasons 2016-17:2025-26 --out reports

The whole-league backtest (``python -m src.backtest``) cannot answer this: most players never appear
in Summer League or preseason, so a layer that only matters for young players barely moves league-wide
metrics. This module scores the population the layer exists for, walk-forward, on exactly the same
projections the draft board would have shown.

Definitions (all fixed up front, see :class:`BreakoutConfig`; none tuned on results)
------------------------------------------------------------------------------------
* **Population**: players the *base* projector projected and who then played at least ``min_gp`` games,
  so actual FPPG exists. (A player who never plays cannot break out; availability is scored elsewhere.)
* **Breakout**: actual FPPG beat the *base* projection by at least ``abs_uplift`` fantasy points per game
  **and** by at least ``rel_uplift`` of the projection. Defined against the base, not the layer under
  test, so the comparison is not circular.
* **Breakout score**: the layer's own adjustment, ``proj_fppg(offseason) - proj_fppg(base)``.
* **Young**: age at season start ``<= young_age``. **Rookie**: the base flagged ``is_rookie``.
* **Under the radar**: ADP rank worse than ``radar_rank`` or no ADP at all.
* **Useful breakout**: a breakout that also finished inside the rostered pool (top ``rostered_rank`` by
  season total fantasy points across the *whole* league). A 9 -> 13 FPPG jump is a breakout by the first
  definition and still leaves a player nobody rosters; the useful definition is what a drafter cares about.

What is reported
----------------
1. Accuracy of FPPG by subgroup, base vs layer, with a paired bootstrap CI on the MAE difference.
2. Breakout detection among young players: AUC of the breakout score, precision@K per season against the
   base rate, and the same restricted to under-the-radar players.
3. A season-by-season list of who the layer flagged and what happened, so the mechanism can be judged by
   eye and not only by a statistic.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.backtest.actuals import season_actuals
from src.backtest.mdutil import df_to_markdown
from src.contracts import HISTORY_TABLES, History, Projector, data_dir, season_start
from src.value.league import load_league, rostered_players


@dataclass(frozen=True)
class BreakoutConfig:
    min_gp: int = 20
    abs_uplift: float = 4.0
    rel_uplift: float = 0.25
    young_age: float = 23.0
    radar_rank: int = 100
    # teams x roster spots from config/league.yaml, the backtest's default replacement pool
    rostered_rank: int = field(default_factory=lambda: int(rostered_players(load_league())))
    top_ks: tuple[int, ...] = (10, 20)


DEFAULT_CONFIG = BreakoutConfig()


# --------------------------------------------------------------------------- pure metric helpers

def breakout_flag(actual_fppg, base_fppg, cfg: BreakoutConfig = DEFAULT_CONFIG) -> np.ndarray:
    """True where actual FPPG beat the base projection by both the absolute and the relative margin."""
    a, b = np.asarray(actual_fppg, float), np.asarray(base_fppg, float)
    uplift = a - b
    return np.isfinite(uplift) & (uplift >= cfg.abs_uplift) & (uplift >= cfg.rel_uplift * np.maximum(b, 0.0))


def auc(score, label) -> float:
    """Probability a random positive outscores a random negative (ties count half). NaN if one class is empty."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def precision_at_k(score, label, k: int) -> float:
    """Share of the ``k`` highest-scoring rows that are positive (fewer rows: uses what exists)."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if len(s) == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")[: min(k, len(s))]
    return float(y[order].mean())


def bootstrap_mean_ci(values, *, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile CI of ``values``; NaNs dropped. (nan, nan, nan) when empty."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def verdict(lo: float, hi: float) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return "n/a"
    return "improves" if lo > 0 else "hurts" if hi < 0 else "no significant change"


# --------------------------------------------------------------------------- evaluation frame

def build_frame(tables: Mapping[str, pd.DataFrame], seasons: Sequence[str], base: Projector, layered: Projector,
                scoring: Mapping[str, float], adp: pd.DataFrame | None = None, *,
                log=lambda msg: None) -> pd.DataFrame:
    """One row per (season, player) projected by ``base``: both projections, the outcome, age, experience, ADP.

    ``adp`` needs ``season``, ``player_id``, ``adp`` (already mapped to player ids).
    """
    gl = tables["game_logs"]
    frames = []
    for s in seasons:
        log(f"[{s}] projecting")
        h = History.until(tables, s)
        pb, pl = base.project(h), layered.project(h)
        act_all = season_actuals(gl, s, scoring)
        league_rank = pd.Series(act_all["total_fp"].rank(ascending=False, method="first").to_numpy(),
                                index=act_all["player_id"].to_numpy())
        act = act_all[["player_id", "gp", "fppg", "total_fp"]].rename(
            columns={"gp": "actual_gp", "fppg": "actual_fppg", "total_fp": "actual_total_fp"})
        cols = ["player_id", "proj_fppg", "proj_gp", "proj_total_fp", "age", "is_rookie"]
        f = pb[[*cols, "player_name"]].rename(columns={"proj_fppg": "base_fppg", "proj_gp": "base_gp",
                                                       "proj_total_fp": "base_total_fp"})
        extra = ["player_id", "proj_fppg", "proj_total_fp", "offseason_adj", "offseason_enabled",
                 "sl_gp", "sl_mpg", "sl_fp36", "sl_z", "pre_gp", "pre_mpg", "pre_fp36", "pre_z"]
        extra = [c for c in extra if c in pl.columns]
        f = f.merge(pl[extra].rename(columns={"proj_fppg": "layer_fppg", "proj_total_fp": "layer_total_fp"}),
                    on="player_id", how="left")
        f = f.merge(act, on="player_id", how="left")
        f["actual_gp"] = f["actual_gp"].fillna(0)
        f["actual_total_fp"] = f["actual_total_fp"].fillna(0.0)
        prior = h.game_logs.groupby("player_id")["season"].nunique()
        f["experience"] = f["player_id"].map(prior).fillna(0).astype(int)
        f["season"] = s
        if adp is not None and len(adp):
            a = adp[adp["season"] == s].drop_duplicates("player_id").set_index("player_id")["adp"]
            f["adp"] = f["player_id"].map(a)
            f["adp_rank"] = f["adp"].rank(method="first")
        else:
            f["adp"] = np.nan
            f["adp_rank"] = np.nan
        f["actual_rank"] = f["player_id"].map(league_rank)   # league-wide, NaN for a player who never played
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out["is_rookie"] = out["is_rookie"].astype(bool)
    return out


def with_flags(frame: pd.DataFrame, cfg: BreakoutConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Add ``played`` (>= min_gp), ``young``, ``radar`` (under the radar), ``breakout``, ``useful`` and ``score``."""
    f = frame.copy()
    f["played"] = f["actual_gp"] >= cfg.min_gp
    f["young"] = f["age"] <= cfg.young_age
    f["radar"] = f["adp_rank"].isna() | (f["adp_rank"] > cfg.radar_rank)
    # A breakout requires having played: FPPG over a handful of games is not a season.
    f["breakout"] = f["played"] & breakout_flag(f["actual_fppg"], f["base_fppg"], cfg)
    f["useful"] = f["breakout"] & (f["actual_rank"] <= cfg.rostered_rank)
    f["score"] = f["layer_fppg"] - f["base_fppg"]
    f["has_event"] = f[[c for c in ("sl_gp", "pre_gp") if c in f.columns]].fillna(0).sum(axis=1) > 0
    return f


# --------------------------------------------------------------------------- reports

def subgroup_accuracy(f: pd.DataFrame, *, n_boot: int = 2000) -> pd.DataFrame:
    """FPPG MAE of base vs layer by subgroup among players who played, with a paired bootstrap CI on the
    MAE improvement (positive = the layer is better)."""
    p = f[f["played"] & f["layer_fppg"].notna()]
    groups = {
        "all players": np.ones(len(p), bool),
        "has summer league or preseason line": p["has_event"].to_numpy(),
        "rookies": p["is_rookie"].to_numpy(),
        "young (<= 23)": p["young"].to_numpy(),
        "young and under the radar": (p["young"] & p["radar"]).to_numpy(),
        "young, second year or earlier": (p["young"] & (p["experience"] <= 2)).to_numpy(),
    }
    rows = []
    for name, m in groups.items():
        d = p[m]
        if len(d) == 0:
            continue
        e_base = (d["actual_fppg"] - d["base_fppg"]).abs().to_numpy()
        e_layer = (d["actual_fppg"] - d["layer_fppg"]).abs().to_numpy()
        gain = e_base - e_layer
        mean, lo, hi = bootstrap_mean_ci(gain, n_boot=n_boot)
        rows.append({"subgroup": name, "n": len(d), "mae_base": e_base.mean(), "mae_layer": e_layer.mean(),
                     "mae_gain": mean, "ci_lo": lo, "ci_hi": hi, "verdict": verdict(lo, hi),
                     "bias_base": float((d["base_fppg"] - d["actual_fppg"]).mean()),
                     "bias_layer": float((d["layer_fppg"] - d["actual_fppg"]).mean())})
    return pd.DataFrame(rows)


def breakout_detection(f: pd.DataFrame, cfg: BreakoutConfig = DEFAULT_CONFIG, *, target: str = "breakout",
                       n_boot: int = 2000) -> pd.DataFrame:
    """Among young players who played: base rate, AUC and precision@K of the breakout score per season.

    ``target`` is ``"breakout"`` or ``"useful"`` (a breakout that also finished in the rostered pool).
    ``lift`` = precision@K minus that season's base rate, averaged over seasons, with a bootstrap CI over
    the seasons' values (each season is one draft). Rows for all young players, the under-the-radar
    subset and rookies.
    """
    rows = []
    pops = {"young": f["played"] & f["young"] & f["layer_fppg"].notna(),
            "young and under the radar": f["played"] & f["young"] & f["radar"] & f["layer_fppg"].notna(),
            "rookies": f["played"] & f["is_rookie"] & f["layer_fppg"].notna()}
    for name, mask in pops.items():
        d = f[mask]
        per_season = []
        for s, g in d.groupby("season"):
            if g[target].sum() == 0 or (~g[target]).sum() == 0:
                continue
            rec = {"season": s, "n": len(g), "base_rate": float(g[target].mean()), "auc": auc(g["score"], g[target])}
            for k in cfg.top_ks:
                rec[f"p@{k}"] = precision_at_k(g["score"], g[target], k)
            per_season.append(rec)
        if not per_season:
            continue
        ps = pd.DataFrame(per_season)
        row = {"population": name, "seasons": len(ps), "players": int(len(d)), "base_rate": ps["base_rate"].mean()}
        m, lo, hi = bootstrap_mean_ci(ps["auc"] - 0.5, n_boot=n_boot)
        row.update({"auc": 0.5 + m, "auc_lo": 0.5 + lo, "auc_hi": 0.5 + hi, "auc_verdict": verdict(lo, hi)})
        for k in cfg.top_ks:
            lift = ps[f"p@{k}"] - ps["base_rate"]
            m, lo, hi = bootstrap_mean_ci(lift, n_boot=n_boot)
            row.update({f"p@{k}": ps[f"p@{k}"].mean(), f"lift@{k}": m, f"lift@{k}_lo": lo, f"lift@{k}_hi": hi,
                        f"verdict@{k}": verdict(lo, hi)})
        rows.append(row)
    return pd.DataFrame(rows)


def flagged_by_season(f: pd.DataFrame, cfg: BreakoutConfig = DEFAULT_CONFIG, top: int = 8) -> pd.DataFrame:
    """Each season's top ``top`` young under-the-radar players by breakout score, and what happened to them."""
    d = f[f["young"] & f["radar"] & f["layer_fppg"].notna() & (f["score"] > 0)]
    rows = []
    for s, g in d.groupby("season"):
        for _, r in g.sort_values("score", ascending=False).head(top).iterrows():
            rows.append({
                "season": s, "player": r["player_name"], "age": round(float(r["age"]), 1),
                "base_fppg": r["base_fppg"], "layer_fppg": r["layer_fppg"], "score": r["score"],
                "actual_gp": int(r["actual_gp"]),
                "actual_fppg": r["actual_fppg"] if r["actual_gp"] > 0 else np.nan,
                "outcome": ("USEFUL BREAKOUT" if bool(r["useful"]) else "breakout, not rosterable" if bool(r["breakout"]) else
                            "did not play enough" if r["actual_gp"] < cfg.min_gp else "no breakout"),
            })
    return pd.DataFrame(rows)


def render_report(f: pd.DataFrame, cfg: BreakoutConfig, *, layered_name: str, command: str,
                  n_boot: int = 2000, top: int = 8) -> str:
    acc = subgroup_accuracy(f, n_boot=n_boot)
    det = breakout_detection(f, cfg, n_boot=n_boot)
    det_useful = breakout_detection(f, cfg, target="useful", n_boot=n_boot)
    flagged = flagged_by_season(f, cfg, top=top)
    seasons = sorted(f["season"].unique(), key=season_start)
    pop = f[f["played"] & f["young"]]
    lines = [
        f"# Breakout evaluation: {layered_name} vs baseline", "",
        f"Seasons {seasons[0]} to {seasons[-1]} ({len(seasons)}), walk-forward: each season is projected from data before it "
        "and scored against what happened.", "",
        "Reproduce:", "", "```", command, "```", "",
        "## Definitions", "",
        f"* **Breakout**: a player who played at least {cfg.min_gp} games and beat the *baseline* projection by at least "
        f"{cfg.abs_uplift:g} FPPG and at least {cfg.rel_uplift:.0%} of it.",
        f"* **Young**: age <= {cfg.young_age:g} at season start. **Under the radar**: ADP rank worse than {cfg.radar_rank} or no ADP.",
        "* **Breakout score**: the layer's projected FPPG minus the baseline's.",
        f"* **Useful breakout**: a breakout that also finished in the top {cfg.rostered_rank} of the league by season "
        "total fantasy points (a rosterable player).",
        f"* Base rate of a breakout among young players who played: {pop['breakout'].mean():.1%} "
        f"({int(pop['breakout'].sum())} of {len(pop)}); of a useful breakout: {pop['useful'].mean():.1%} "
        f"({int(pop['useful'].sum())}).", "",
        "## FPPG accuracy by subgroup", "",
        "MAE in fantasy points per game among players who played; *gain* is base MAE minus layer MAE (positive = layer better), "
        "with a paired bootstrap 95% CI over players.", "",
        df_to_markdown(acc.round(3)), "",
        "## Breakout detection", "",
        "Per season, players are ranked by breakout score; *p@K* is the share of the top K who broke out, *lift* is p@K minus that "
        "season's base rate (mean over seasons, bootstrap CI over seasons). AUC 0.5 = no skill.", "",
        df_to_markdown(det.round(3)) if len(det) else "_not enough breakouts to evaluate_", "",
        "### Useful breakouts (breakout and finished in the rostered pool)", "",
        df_to_markdown(det_useful.round(3)) if len(det_useful) else "_not enough useful breakouts to evaluate_", "",
        "## Who the layer flagged each season", "",
        f"Top {top} young under-the-radar players by breakout score (score > 0), and the outcome.", "",
        df_to_markdown(flagged.round(2)) if len(flagged) else "_none_", "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI

def _load_adp(path: Path | None, base: Path | None) -> pd.DataFrame | None:
    if path is None:
        default = (base or data_dir()) / "processed" / "adp.parquet"
        path = default if default.exists() else None
    if path is None:
        return None
    from src.backtest.benchmarks import load_adp
    from src.store import read_table

    id_map = read_table("player_id_map", base)
    return load_adp(path, id_map).frame


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.backtest.breakouts", description=__doc__.split("\n\n")[0])
    ap.add_argument("--seasons", default="2016-17:2025-26")
    ap.add_argument("--model", default="baseline_offseason", help="the layered projector (registry name)")
    ap.add_argument("--adp-file", type=Path, default=None, help="ADP parquet/CSV (default: the ingested adp.parquet)")
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--preseason-fraction", type=float, default=None,
                    help="keep only the first share of each preseason's game dates (a draft held mid-preseason)")
    ap.add_argument("--save-calibration", action="store_true",
                    help="fit P(breakout) on this run's out-of-sample scores and store it for the watchlist")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--top", type=int, default=8, help="flagged players listed per season")
    args = ap.parse_args(argv)

    from src.backtest.harness import expand_seasons
    from src.models.registry import get_projector
    from src.store import load_tables

    seasons = expand_seasons(args.seasons)
    tables = dict(load_tables(HISTORY_TABLES))
    scoring = load_league()["scoring"]
    kwargs = {} if args.preseason_fraction is None else {"preseason_fraction": args.preseason_fraction}
    base, layered = get_projector("baseline"), get_projector(args.model, **kwargs)
    adp = _load_adp(args.adp_file, None)
    f = build_frame(tables, seasons, base, layered, scoring, adp, log=print)
    cfg = BreakoutConfig()
    f = with_flags(f, cfg)
    run_id = args.run_id or f"breakouts_{args.model}"
    out = args.out / run_id
    out.mkdir(parents=True, exist_ok=True)
    cmd = "python -m src.backtest.breakouts " + " ".join(sys.argv[1:] if argv is None else argv)
    (out / "breakouts.md").write_text(render_report(f, cfg, layered_name=args.model, command=cmd,
                                                    n_boot=args.n_boot, top=args.top), encoding="utf-8")
    f.to_parquet(out / "breakout_frame.parquet", index=False)
    print(f"report: {out / 'breakouts.md'}")
    if args.save_calibration:
        from src.value.breakouts import fit_calibration

        calib = fit_calibration(f, args.model, cfg)
        print(f"calibration for {args.model}: n={calib.breakout.n}; breakout base rate {calib.breakout.base_rate:.3f} "
              f"(score {calib.breakout.coef_score:+.3f}), useful base rate {calib.useful.base_rate:.3f} "
              f"(score {calib.useful.coef_score:+.3f}) -> {calib.save()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
