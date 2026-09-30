"""Walk-forward evaluation of the veteran contract-terms layer, league-wide and conditional on coverage (ADR 0019).

    python -m src.backtest.contract_terms_eval [--seasons 2016-17:2025-26] [--out reports/] [--run-id contract_terms_eval]
                                               [--n-boot 1000] [--leak-check]

Runs, on the same real ten-season walk-forward as every other ablation (``load_run_tables``, so the dated contract
events travel as ``History.extras`` and the leak check covers them):

* ``baseline``                       the reference;
* ``baseline_contract``              the ADR 0013 rookie-scale layer, for the second comparison;
* ``baseline_contract_terms``        the shipped layer (cross-validation gate on, 0.05%);
* ``baseline_contract_terms`` forced the same layer with the gate bypassed (``min_cv_gain=-1``), a diagnostic only:
  it answers "would the terms have helped had the gate let them through?", which the gated run cannot.

and reports, for each layer against ``baseline`` (paired bootstrap over players, stratified by season):

1. league-wide lift (Spearman, top-50 hit rate, MAE of total fantasy points and of FPPG);
2. the same restricted to players whose contract status is known at the start of the season (any known deal, a known
   final year, a fresh deal, any Wikipedia event, and the rest), where the rank metrics are computed within the subset;
3. the gate history (which seasons it opened, the cross-validated gain, the marked training rows);
4. the coverage-conditional effect size: the baseline's own residual (actual minus projected FPPG, players with 20 or
   more games, season-demeaned) by contract state, with a player-clustered bootstrap CI, pooled and by era. This is the
   size of the signal itself, before any model or gate, on exactly the players the source can label.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.backtest.ablation import paired_lift
from src.backtest.harness import BacktestResult, expand_seasons, walk_forward
from src.backtest.runner import load_run_tables
from src.contracts import History, season_start
from src.features.contract import contract_clock
from src.features.contract_terms import coverage_by_season, terms_features
from src.models.contract_baseline import BaselineContractProjector
from src.models.contract_terms_baseline import BaselineContractTermsProjector
from src.models.baseline import BaselineProjector
from src.value.league import load_league

log = logging.getLogger("contract_terms_eval")

DEFAULT_SEASONS = "2016-17:2025-26"
LIFT_METRICS = ("spearman_total_fp", "top50_hit", "mae_total_fp", "mae_fppg")
SUBSET_METRICS = ("spearman_total_fp", "mae_total_fp", "mae_fppg")
GROUPS = ("known_cy", "known_multi", "no_length", "lapsed", "rookie_clock_only", "none")
MIN_RESID_GP = 20


class LoggedTerms(BaselineContractTermsProjector):
    """The shipped projector that also records the fit of every target season (gate state, gain, coefficients)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.gate_log: list[dict] = []

    def project(self, history: History) -> pd.DataFrame:
        out = super().project(history)
        d = dict(self.last_fit.diagnostics) if self.last_fit is not None else {}
        d.update({"season": history.target_season, "enabled": bool(self.last_fit.enabled) if self.last_fit else False,
                  "n_known": int((out["terms_status"] == "known").sum()), "n_players": int(len(out))})
        self.gate_log.append(d)
        return out


def states(tables: dict, seasons: list[str]) -> pd.DataFrame:
    """One row per (season, player) of every player in the target-season universe with the as-of-start terms state."""
    contracts = tables.get("player_contracts")
    rows = []
    for s in seasons:
        h = History.until(tables, s)
        pids = np.array(sorted(tables["game_logs"].loc[tables["game_logs"]["season"] == s, "player_id"].unique()), dtype="int64")
        pids = np.union1d(pids, h.game_logs["player_id"].unique()).astype("int64")
        tf = terms_features(contracts, s, pids, contract_clock(h.players, s, pids))
        grp = np.full(len(pids), "none", dtype=object)
        grp[tf.rookie_cy | tf.rookie_opt] = "rookie_clock_only"
        grp[tf.status == "lapsed"] = "lapsed"
        grp[tf.status == "no_length"] = "no_length"
        grp[tf.status == "known"] = "known_multi"
        grp[tf.contract_year] = "known_cy"
        rows.append(pd.DataFrame({
            "season": s, "player_id": pids, "group": grp, "status": tf.status, "years_remaining": tf.years_remaining,
            "known": tf.status == "known", "contract_year": tf.contract_year, "new_deal": tf.new_deal,
            "extension": tf.extension, "two_way": tf.two_way, "minimum": tf.minimum,
            "any_wiki": tf.status != "unknown"}))
    return pd.concat(rows, ignore_index=True)


SUBSETS = {
    "known deal": lambda d: d["known"],
    "known final year": lambda d: d["contract_year"],
    "known, 2+ years left": lambda d: d["known"] & ~d["contract_year"],
    "signed this offseason": lambda d: d["new_deal"],
    "any Wikipedia event": lambda d: d["any_wiki"],
    "no Wikipedia event": lambda d: ~d["any_wiki"],
}


def restrict(res: BacktestResult, st: pd.DataFrame, mask_fn) -> BacktestResult:
    """``res`` limited to the (season, player) rows selected by ``mask_fn`` over the state frame."""
    keep = st[mask_fn(st)][["season", "player_id"]].assign(_k=True)
    m = res.players.merge(keep, on=["season", "player_id"], how="left")["_k"].fillna(False).to_numpy(bool)
    return replace(res, players=res.players[m].reset_index(drop=True))


def lift_row(a: BacktestResult, b: BacktestResult, metric: str, n_boot: int) -> dict:
    try:
        pl = paired_lift(a, b, metric, n_boot=n_boot)
    except Exception as exc:  # noqa: BLE001 - a subset can be empty in a season
        return {"metric": metric, "lift": np.nan, "lo": np.nan, "hi": np.nan, "p_le0": np.nan, "won": "n/a",
                "n_players": 0, "verdict": f"n/a ({type(exc).__name__})"}
    return {"metric": metric, "lift": pl.estimate, "lo": pl.lo, "hi": pl.hi, "p_le0": pl.p_le_zero,
            "won": f"{pl.wins}/{pl.n_seasons}", "n_players": pl.n_players, "verdict": pl.verdict}


def residual_frame(base: BacktestResult, st: pd.DataFrame) -> pd.DataFrame:
    p = base.players
    p = p[p["projected"] & (p["actual_gp"] >= MIN_RESID_GP)][["season", "player_id", "proj_fppg", "actual_fppg", "actual_gp"]].copy()
    p["resid"] = p["actual_fppg"] - p["proj_fppg"]
    p["resid_dm"] = p["resid"] - p.groupby("season")["resid"].transform("mean")     # season-demeaned: drops league-wide drift
    return p.merge(st, on=["season", "player_id"], how="left")


def effect_table(res: pd.DataFrame, *, n_boot: int, seed: int = 0) -> pd.DataFrame:
    """Mean season-demeaned residual per group with a player-clustered bootstrap CI, plus the contract-year contrasts."""
    rng = np.random.default_rng(seed)
    pid = res["player_id"].to_numpy()
    uniq, inv = np.unique(pid, return_inverse=True)
    rows = []

    def boot_mean(mask: np.ndarray) -> tuple[float, float, float]:
        y, m = res["resid_dm"].to_numpy()[mask], inv[mask]
        if len(y) == 0:
            return np.nan, np.nan, np.nan
        s = np.bincount(m, weights=y, minlength=len(uniq))
        c = np.bincount(m, minlength=len(uniq)).astype(float)
        reps = []
        for _ in range(n_boot):
            w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
            den = float((c * w).sum())
            reps.append(float((s * w).sum() / den) if den else np.nan)
        lo, hi = np.nanquantile(reps, [0.025, 0.975])
        return float(y.mean()), float(lo), float(hi)

    for g in GROUPS:
        mask = (res["group"] == g).to_numpy()
        mean, lo, hi = boot_mean(mask)
        rows.append({"group": g, "n": int(mask.sum()), "n_players": int(len(np.unique(pid[mask]))), "mean_resid": mean, "lo": lo, "hi": hi})
    known = res["known"].to_numpy(bool)
    cy = res["contract_year"].to_numpy(bool)
    for label, a, b in (("known final year minus known 2+ years", cy, known & ~cy),
                        ("known final year minus no Wikipedia event", cy, ~res["any_wiki"].to_numpy(bool)),
                        ("signed this offseason minus not", res["new_deal"].to_numpy(bool), ~res["new_deal"].to_numpy(bool))):
        ya, yb = res["resid_dm"].to_numpy()[a], res["resid_dm"].to_numpy()[b]
        if len(ya) < 3 or len(yb) < 3:
            rows.append({"group": label, "n": int(a.sum()), "n_players": 0, "mean_resid": np.nan, "lo": np.nan, "hi": np.nan})
            continue
        sa = np.bincount(inv[a], weights=res["resid_dm"].to_numpy()[a], minlength=len(uniq))
        ca = np.bincount(inv[a], minlength=len(uniq)).astype(float)
        sb = np.bincount(inv[b], weights=res["resid_dm"].to_numpy()[b], minlength=len(uniq))
        cb = np.bincount(inv[b], minlength=len(uniq)).astype(float)
        reps = []
        for _ in range(n_boot):
            w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
            da, db = float((ca * w).sum()), float((cb * w).sum())
            reps.append(float((sa * w).sum() / da - (sb * w).sum() / db) if da and db else np.nan)
        lo, hi = np.nanquantile(reps, [0.025, 0.975])
        rows.append({"group": label, "n": int(a.sum()), "n_players": int(len(np.unique(pid[a]))),
                     "mean_resid": float(ya.mean() - yb.mean()), "lo": float(lo), "hi": float(hi)})
    return pd.DataFrame(rows)


def run(seasons: list[str], *, n_boot: int, leak_check: bool, out: Path, run_id: str) -> dict:
    tables = load_run_tables(SimpleNamespace(synthetic=False))
    if "player_contracts" not in tables:
        raise SystemExit("player_contracts.parquet not found: run python -m src.ingest.wiki_contracts first")
    scoring = load_league()["scoring"]
    kw = dict(scoring=scoring, leak_check=[seasons[-1]] if leak_check else ())
    cache: dict = {}
    results, gate = {}, {}

    log.info("== baseline")
    results["baseline"] = walk_forward(tables, BaselineProjector(scoring), seasons, **kw)
    log.info("== baseline_contract")
    results["baseline_contract"] = walk_forward(tables, BaselineContractProjector(scoring=scoring), seasons, **kw)
    log.info("== baseline_contract_terms (gated)")
    gated = LoggedTerms(scoring=scoring, walk_forward_cache=cache)
    results["baseline_contract_terms"] = walk_forward(tables, gated, seasons, **kw)
    gate["gated"] = gated.gate_log
    log.info("== baseline_contract_terms (gate forced open)")
    forced = LoggedTerms(scoring=scoring, min_cv_gain=-1.0, walk_forward_cache=cache, name="baseline_contract_terms")
    results["terms_forced"] = walk_forward(tables, forced, seasons, scoring=scoring)
    gate["forced"] = forced.gate_log

    base = results["baseline"]
    st = states(tables, seasons)
    cov = coverage_by_season(tables["player_contracts"], tables["game_logs"], seasons)

    lifts = []
    for name in ("baseline_contract", "baseline_contract_terms", "terms_forced"):
        for m in LIFT_METRICS:
            lifts.append({"variant": name, "subset": "league-wide", **lift_row(base, results[name], m, n_boot)})
    subset_rows = []
    for name in ("baseline_contract_terms", "terms_forced"):
        for label, fn in SUBSETS.items():
            a, b = restrict(base, st, fn), restrict(results[name], st, fn)
            for m in SUBSET_METRICS:
                subset_rows.append({"variant": name, "subset": label, **lift_row(a, b, m, n_boot)})
    res = residual_frame(base, st)
    eff_all = effect_table(res, n_boot=n_boot)
    eff_early = effect_table(res[res["season"].map(season_start) <= 2019], n_boot=n_boot)
    eff_late = effect_table(res[res["season"].map(season_start) >= 2020], n_boot=n_boot)

    forced_res = residual_frame(results["terms_forced"], st)
    layer = res[["season", "player_id", "group", "resid"]].merge(
        forced_res[["season", "player_id", "resid"]].rename(columns={"resid": "resid_layer"}), on=["season", "player_id"])
    by_group = layer.groupby("group").agg(n=("resid", "size"), bias_base=("resid", "mean"), bias_forced=("resid_layer", "mean"),
                                          mae_base=("resid", lambda x: float(np.abs(x).mean())),
                                          mae_forced=("resid_layer", lambda x: float(np.abs(x).mean()))).reset_index()

    report = {
        "seasons": seasons, "n_boot": n_boot, "leak_check": "passed" if leak_check else "not run",
        "coverage": cov.to_dict("records"), "lifts": lifts, "subset_lifts": subset_rows,
        "gate": {k: [{kk: (vv if not isinstance(vv, dict) else vv) for kk, vv in g.items()} for g in v] for k, v in gate.items()},
        "effect_all": eff_all.to_dict("records"), "effect_2016_19": eff_early.to_dict("records"),
        "effect_2020_25": eff_late.to_dict("records"), "by_group_forced": by_group.to_dict("records"),
        "summary": {k: v.summary().to_dict() for k, v in results.items()},
        "metrics_by_season": {k: v.season_metrics[["spearman_total_fp", "top50_hit", "mae_total_fp"]].to_dict() for k, v in results.items()},
    }
    d = out / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "contract_terms_eval.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (d / "report.md").write_text(render(report, cov, lifts, subset_rows, gate, eff_all, eff_early, eff_late, by_group), encoding="utf-8")
    return report


def _f(x, nd=4, sign=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _tbl(df: pd.DataFrame, cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        d = r._asdict()
        lines.append("| " + " | ".join(fmt.get(c, str)(d[c]) for c in cols) + " |")
    return "\n".join(lines)


def render(report, cov, lifts, subset_rows, gate, eff_all, eff_early, eff_late, by_group) -> str:
    L = ["# Contract-terms layer evaluation (ADR 0019)", "",
         f"Seasons {report['seasons'][0]} to {report['seasons'][-1]}; paired bootstrap over players stratified by season, "
         f"{report['n_boot']} replicates; future-invariance check {report['leak_check']}.", "",
         "## Coverage: veterans scored in each season and their contract state at the start of it", "",
         _tbl(cov, ["season", "veterans", "known", "no_length", "lapsed", "unknown", "any_event", "contract_year", "new_deal", "extension", "two_way"]), "",
         "## Lift over `baseline`, league-wide (positive = better; MAE lift = error reduction)", ""]
    ld = pd.DataFrame(lifts)
    L.append(_tbl(ld, ["variant", "metric", "lift", "lo", "hi", "won", "verdict"], {"lift": _f, "lo": _f, "hi": _f}))
    L += ["", "## Lift over `baseline` restricted to players by contract state at the start of the season", "",
          "Rank metrics are computed within the subset; the subset is defined from events dated before opening night only.", ""]
    sd = pd.DataFrame(subset_rows)
    L.append(_tbl(sd, ["variant", "subset", "metric", "lift", "lo", "hi", "won", "n_players", "verdict"], {"lift": _f, "lo": _f, "hi": _f}))
    L += ["", "## Gate history (cross-validated gain over 'no adjustment'; threshold 0.05%)", ""]
    gd = []
    for k, v in gate.items():
        for g in v:
            gd.append({"run": k, "season": g["season"], "enabled": g["enabled"], "gain": g.get("cv_gain_vs_zero", np.nan),
                       "train_rows": g.get("train_rows"), "marked_rows": g.get("marked_rows"), "known_rows": g.get("known_rows"),
                       "reason": str(g.get("reason", ""))[:70]})
    L.append(_tbl(pd.DataFrame(gd), ["run", "season", "enabled", "gain", "train_rows", "marked_rows", "known_rows", "reason"],
                  {"gain": lambda x: _f(x, 5)}))
    for title, e in (("all seasons", eff_all), ("2016-17 to 2019-20", eff_early), ("2020-21 to 2025-26", eff_late)):
        L += ["", f"## Effect size on covered players: baseline residual (actual - projected FPPG, 20+ games, season-demeaned), {title}", "",
              "Player-clustered bootstrap 95% CI. The last three rows are contrasts (positive = the first group beats its projection more).", "",
              _tbl(e, ["group", "n", "n_players", "mean_resid", "lo", "hi"], {"mean_resid": lambda x: _f(x, 3), "lo": lambda x: _f(x, 3), "hi": lambda x: _f(x, 3)})]
    L += ["", "## Bias and MAE of FPPG by contract state, baseline vs the gate-forced layer (20+ games)", "",
          _tbl(by_group, ["group", "n", "bias_base", "bias_forced", "mae_base", "mae_forced"],
               {c: (lambda x: _f(x, 3)) for c in ("bias_base", "bias_forced", "mae_base", "mae_forced")})]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.backtest.contract_terms_eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seasons", default=DEFAULT_SEASONS)
    p.add_argument("--out", default="reports")
    p.add_argument("--run-id", default="contract_terms_eval")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--leak-check", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rep = run(expand_seasons(args.seasons), n_boot=args.n_boot, leak_check=args.leak_check, out=Path(args.out), run_id=args.run_id)
    print(f"report: {Path(args.out) / args.run_id / 'report.md'}")
    for r in rep["lifts"]:
        print(f"{r['variant']:26s} {r['metric']:18s} lift {_f(r['lift'])} [{_f(r['lo'])}, {_f(r['hi'])}] won {r['won']} {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
