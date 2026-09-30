"""Secondary analyses for the return-health availability feature (ADR 0022).

    python -m src.backtest.return_feature_eval --seasons 2016-17:2025-26 --out reports/return_feature_2026-09-26

The primary decision (paired-bootstrap ablation of ``baseline_return`` vs ``baseline`` on the whole universe) comes from
``python -m src.backtest --ablate ... --sig-metrics ...``. This module adds the pre-registered *secondary* analyses, which cannot change the
adoption decision:

* lift restricted to ADR 0021's primary cohort (lead block >= 25%, tail >= 15 games at >= 75% health, veteran, single team, >= 15 mpg)
  and to its >= 60%-block subgroup, cohort membership computed point-in-time by ``src.backtest.return_report`` from ``History.until``;
* Tatum's projected GP / total FP before and after, from the live history (data through the last ingested season).

Lift is ``error(base) - error(variant)`` on the same rows (positive = the variant is better), pooled over players and resampled within each
target season (the same percentile bootstrap as ADR 0021). Signed bias is *actual minus projected* (positive = under-projected).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.backtest import return_report as R
from src.backtest.harness import BacktestResult
from src.backtest.mdutil import df_to_markdown
from src.contracts import History

COHORT_COLS = ["season", "player_id", "in_primary", "in_block60"]


def cohort_table(tables: Mapping[str, pd.DataFrame], seasons: Sequence[str]) -> pd.DataFrame:
    """One row per (season, player_id) with the ADR 0021 cohort flags, from ``History.until`` (seasons before the target only)."""
    frames = []
    primary, long_block = R.PRIMARY, R.CohortConfig(t1=0.60)
    for s in seasons:
        h = History.until(tables, s)
        h.assert_no_future()
        f = R.prior_features(h)
        if f.empty:
            continue
        f.insert(0, "season", s)
        f["in_primary"] = R.in_cohort(f, primary)
        f["in_block60"] = R.in_cohort(f, long_block)
        frames.append(f[COHORT_COLS])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COHORT_COLS)


def paired_rows(base: BacktestResult, var: BacktestResult) -> pd.DataFrame:
    """Rows projected by both, with both projections and the actuals (the ablation's comparison universe)."""
    fa = base.players[base.players["projected"]]
    fb = var.players[var.players["projected"]]
    cols = ["season", "player_id", "proj_gp", "proj_total_fp"]
    return fa[cols + ["actual_gp", "actual_total_fp"]].merge(fb[cols], on=["season", "player_id"], suffixes=("_base", "_var"))


def lift_table(rows: pd.DataFrame, cohorts: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Lift (base error - variant error) on games and total FP, plus the signed bias of each, for all rows and each cohort subset."""
    m = rows.merge(cohorts, on=["season", "player_id"], how="left")
    for c in ("in_primary", "in_block60"):
        m[c] = m[c].fillna(False).astype(bool)
    subsets = {"all projected": np.ones(len(m), bool), "ADR 0021 primary cohort": m["in_primary"].to_numpy(),
               "cohort, block >= 60%": m["in_block60"].to_numpy()}
    out = []
    for name, mask in subsets.items():
        g = m[mask]
        if g.empty:
            continue
        s = g["season"].to_numpy()
        for label, act, pb, pv in (("games played", "actual_gp", "proj_gp_base", "proj_gp_var"),
                                   ("total FP", "actual_total_fp", "proj_total_fp_base", "proj_total_fp_var")):
            eb, ev = (g[act] - g[pb]).to_numpy(float), (g[act] - g[pv]).to_numpy(float)
            est, lo, hi = R.boot_mean(np.abs(eb) - np.abs(ev), s, n_boot=n_boot, seed=seed)
            out.append({"subset": name, "n": len(g), "metric": f"MAE {label}", "MAE base": float(np.abs(eb).mean()),
                        "MAE variant": float(np.abs(ev).mean()), "lift": est, "lo": lo, "hi": hi,
                        "bias base": float(eb.mean()), "bias variant": float(ev.mean())})
    return pd.DataFrame(out)


def player_before_after(tables: Mapping[str, pd.DataFrame], target: str, name_part: str,
                        models: Sequence[str]) -> pd.DataFrame:
    """Projected GP / FPPG / total FP of players matching ``name_part`` under each registry model, from the history before ``target``."""
    from src.models.registry import get_projector

    h = History.until(tables, target)
    rows = []
    for reg in models:
        p = get_projector(reg).project(h)
        hit = p[p["player_name"].str.contains(name_part, case=False, na=False)]
        for _, r in hit.iterrows():
            rows.append({"model": reg, "player": r["player_name"], "proj_gp": r["proj_gp"], "proj_fppg": r["proj_fppg"],
                         "proj_total_fp": r["proj_total_fp"]})
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.backtest.return_feature_eval", description=__doc__.split("\n\n")[0])
    ap.add_argument("--seasons", default="2016-17:2025-26")
    ap.add_argument("--pairs", default="baseline:baseline_return,baseline_injury:baseline_injury_return",
                    help="comma list of base:variant registry names")
    ap.add_argument("--target", default=None, help="season projected for the player table (default: the season after the last ingested)")
    ap.add_argument("--player", default="Tatum")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from src.backtest.harness import expand_seasons, walk_forward
    from src.backtest.runner import load_run_tables
    from src.contracts import season_start, season_str
    from src.models.registry import get_projector

    seasons = expand_seasons(args.seasons)
    tables = load_run_tables(argparse.Namespace(synthetic=False))
    cohorts = cohort_table(tables, seasons)
    lines = ["# Return-health feature: secondary analyses (ADR 0022)", "",
             "Lift = base error - variant error on the same rows (positive = variant better); pooled over players, bootstrap resampled "
             f"within target season, {args.n_boot} resamples, 95% percentile CI. Signed bias = actual - projected.", ""]
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = [tuple(x.split(":")) for x in args.pairs.split(",") if x.strip()]
    for base, var in pairs:
        print(f"[{base} vs {var}] walk-forward", flush=True)
        rb = walk_forward(tables, get_projector(base), seasons)
        rv = walk_forward(tables, get_projector(var), seasons)
        t = lift_table(paired_rows(rb, rv), cohorts, n_boot=args.n_boot)
        t.to_csv(args.out / f"lift_{var}.csv", index=False)
        lines += [f"## {var} vs {base}", "", df_to_markdown(t.round(2), index=False, floatfmt="{:.2f}"), ""]
    last = max(season_start(s) for s in tables["game_logs"]["season"].unique())
    target = args.target or season_str(last + 1)
    models = [x for pr in pairs for x in pr] + ["baseline_offseason_debut", "baseline_return_offseason_debut"]
    ba = player_before_after(tables, target, args.player, list(dict.fromkeys(models)))
    ba.to_csv(args.out / "player_before_after.csv", index=False)
    lines += [f"## {args.player}, projection for {target} (history through {season_str(last)})", "",
              df_to_markdown(ba.round(2), index=False, floatfmt="{:.2f}"), ""]
    text = "\n".join(lines)
    (args.out / "return_feature_eval.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
