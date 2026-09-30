"""Load-management proxy study: does an absence-shape proxy predict next-season games beyond the baseline? (ADR 0024)

    python -m src.backtest.load_report --seasons 2016-17:2025-26 --out reports/load_2026-09-26

There is no label for *why* a game was missed (see ADR 0024), so "load management" is proxied by the shape of last season's absences
(:mod:`src.features.load_management`): games missed in short interior runs (``iso_n82``), one-game absences on the second night of a
back-to-back (``rest_n82``), the isolated share of all absences, a 65..67 GP marker and an age x minutes cell. Every definition,
the universe, the primary regression, the out-of-sample correction test and the decision rule are pre-registered in the ADR.

Leakage: projections and features both come from ``History.until(tables, s)`` (seasons < s); ``s`` only supplies the outcome. The
out-of-sample test fits each target season's correction on the errors of *earlier* target seasons only. Inference is a percentile
bootstrap over rows resampled within each target season (the coefficient is refit in every resample).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.backtest.actuals import season_actuals
from src.backtest.mdutil import df_to_markdown
from src.backtest.return_report import add_outcomes, boot_mean
from src.contracts import History, Projector
from src.features.load_management import prior_load_features
from src.models.panel import season_lengths, target_season_games
from src.value.league import load_league

N_BOOT = 2000
MIN_MPG, MIN_GP = 20.0, 41         # universe: rotation players who were healthy enough to be rested
HM_AGE, HM_MPG = 31.0, 30.0        # the age x minutes cell
FIRST_TEST_SEASON = "2019-20"      # the out-of-sample correction needs three earlier target seasons
BINS = ((0, 1), (2, 3), (4, 5), (6, 999))
PROXIES = ("iso_n82", "rest_n82", "iso_frac", "near65", "hm_vet")
PRIMARY_PROXY = "iso_n82"
CONTROLS = ("f", "age", "age2", "mpg", "proj_f")
MIN_SEASONS_SIGN = 7
HI_ISO = 6                          # the top pre-registered iso_n bin (6+): the descriptive flag threshold
RAW_UNIT = ("hi_iso",)              # 0/1 columns reported per unit (per flagged player), not per SD


# --------------------------------------------------------------------------- walk-forward frame

def build_load_frame(tables: Mapping[str, pd.DataFrame], seasons: Sequence[str], projector: Projector, scoring: Mapping[str, float],
                     *, log=lambda msg: None) -> pd.DataFrame:
    """One row per (target season, projected player who appeared the season before): prior-season proxies, projection and outcome.

    The projector and the features both read ``History.until(tables, s)``, so they cannot disagree about what is known.
    """
    frames = []
    for s in seasons:
        log(f"[{s}] projecting and profiling")
        h = History.until(tables, s)
        h.assert_no_future()
        proj = projector.project(h)
        feat = prior_load_features(h)
        act = season_actuals(tables["game_logs"], s, scoring)[["player_id", "gp", "total_fp", "fppg"]].rename(
            columns={"gp": "actual_gp", "total_fp": "actual_total_fp", "fppg": "actual_fppg"})
        cols = ["player_id", "player_name", "age", "proj_gp", "proj_fppg", "proj_total_fp"]
        f = proj[cols].merge(feat, on="player_id", how="inner").merge(act, on="player_id", how="left")
        f["actual_gp"] = f["actual_gp"].fillna(0.0)
        f["actual_total_fp"] = f["actual_total_fp"].fillna(0.0)
        f["target_len"] = float(target_season_games(season_lengths(h)))
        f.insert(0, "season", s)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def prepare(f: pd.DataFrame) -> pd.DataFrame:
    """Outcomes, derived regressors and the pre-registered universe flag (``in_universe``)."""
    f = add_outcomes(f)
    f["age2"] = f["age"] ** 2
    f["proj_f"] = f["proj_gp"] / f["target_len"]
    f["hm_vet"] = ((f["age"] >= HM_AGE) & (f["mpg"] >= HM_MPG)).astype(float)
    f["near65"] = f["near65"].astype(float)
    f["hi_iso"] = (f["iso_n"] >= HI_ISO).astype(float)
    ok = f["veteran"].astype(bool) & (f["n_teams"] == 1) & (f["mpg"] >= MIN_MPG) & (f["gp"] >= MIN_GP) & f["age"].notna()
    f["in_universe"] = ok & np.isfinite(f[list(CONTROLS)]).all(axis=1)
    return f


# --------------------------------------------------------------------------- regression

def _design(f: pd.DataFrame, proxy: str | None, mu: float, sd: float) -> np.ndarray:
    """Regressors: the standardised proxy first (omitted when ``proxy`` is None: the controls-only model), then the controls."""
    cols = [f[c].to_numpy(float) for c in CONTROLS]
    if proxy is not None:
        x = f[proxy].to_numpy(float)
        cols.insert(0, (x - mu) / sd if sd > 0 else x - mu)
    return np.column_stack(cols)


def _fe_coef(y: np.ndarray, X: np.ndarray, season: np.ndarray) -> np.ndarray:
    """OLS coefficients with season fixed effects (within-season demeaning)."""
    yd, Xd = y.copy(), X.copy()
    for k in np.unique(season):
        m = season == k
        yd[m] -= y[m].mean()
        Xd[m] -= X[m].mean(axis=0)
    return np.linalg.lstsq(Xd, yd, rcond=None)[0]


def standardiser(f: pd.DataFrame, proxy: str | None) -> tuple[float, float]:
    if proxy is None or proxy in RAW_UNIT:
        return 0.0, 1.0
    x = f.loc[f["in_universe"], proxy].to_numpy(float)
    return float(x.mean()), float(x.std())


def proxy_effect(f: pd.DataFrame, proxy: str, outcome: str, *, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """Coefficient on the standardised proxy (per +1 SD) in the fixed-effects regression, with a stratified percentile bootstrap CI."""
    u = f[f["in_universe"] & np.isfinite(f[outcome])].reset_index(drop=True)
    if len(u) < 30:
        return float("nan"), float("nan"), float("nan")
    mu, sd = standardiser(f, proxy)
    y, X, season = u[outcome].to_numpy(float), _design(u, proxy, mu, sd), u["season"].to_numpy()
    est = float(_fe_coef(y, X, season)[0])
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(season == k) for k in np.unique(season)]
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        draws[b] = _fe_coef(y[idx], X[idx], season[idx])[0]
    lo, hi = np.quantile(draws[np.isfinite(draws)], [0.025, 0.975])
    return est, float(lo), float(hi)


def per_season_effect(f: pd.DataFrame, proxy: str, outcome: str) -> pd.DataFrame:
    """The same regression fitted season by season (no bootstrap): the sign-consistency check of the decision rule."""
    mu, sd = standardiser(f, proxy)
    rows = []
    for s, g in f[f["in_universe"] & np.isfinite(f[outcome])].groupby("season", sort=True):
        if len(g) < 30:
            rows.append((s, len(g), float("nan")))
            continue
        X = _design(g, proxy, mu, sd)
        Xd = X - X.mean(axis=0)
        yd = g[outcome].to_numpy(float) - g[outcome].mean()
        rows.append((s, len(g), float(np.linalg.lstsq(Xd, yd, rcond=None)[0][0])))
    return pd.DataFrame(rows, columns=["season", "n", "b"])


def sign_agreement(per_season: pd.DataFrame, pooled: float) -> tuple[int, int]:
    """(seasons whose per-season coefficient has the pooled sign, seasons with a coefficient)."""
    b = per_season["b"].to_numpy(float)
    b = b[np.isfinite(b)]
    return int((np.sign(b) == np.sign(pooled)).sum()), int(len(b))


def bin_table(f: pd.DataFrame, *, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Mean signed error by ``iso_n`` bin (raw count of isolated absences last season), universe rows only."""
    u = f[f["in_universe"]]
    rows = []
    for lo, hi in BINS:
        g = u[(u["iso_n"] >= lo) & (u["iso_n"] <= hi)]
        label = f"{lo}-{hi}" if hi < 999 else f"{lo}+"
        gp, fp = (boot_mean(g["err_gp"], g["season"], n_boot=n_boot), boot_mean(g["err_fp"], g["season"], n_boot=n_boot))
        rows.append({"iso_n": label, "n": len(g), "actual GP": g["actual_gp"].mean(), "proj GP": g["proj_gp"].mean(),
                     "bias GP [95% CI]": _ci(gp), "bias total FP [95% CI]": _ci(fp, "{:+.0f}")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- out-of-sample correction

def walk_forward_correction(f: pd.DataFrame, proxy: str | None = PRIMARY_PROXY, *, first: str = FIRST_TEST_SEASON,
                            n_boot: int = N_BOOT) -> dict:
    """Fit ``err_gp ~ proxy + controls`` on the target seasons before each test season, correct ``proj_gp`` and score against actuals.

    Returns the corrected frame (test rows of the universe) and paired MAE differences (baseline minus corrected; positive = the
    correction helps) for GP and total FP with stratified bootstrap CIs.
    """
    seasons = sorted(f["season"].unique())
    mu, sd = standardiser(f, proxy)
    out = []
    for s in seasons:
        if s < first:
            continue
        tr = f[f["in_universe"] & (f["season"] < s) & np.isfinite(f["err_gp"])]
        te = f[f["in_universe"] & (f["season"] == s)]
        if len(tr) < 100 or te.empty:
            continue
        Xtr = np.column_stack([np.ones(len(tr)), _design(tr, proxy, mu, sd)])
        w = np.linalg.lstsq(Xtr, tr["err_gp"].to_numpy(float), rcond=None)[0]
        pred = np.column_stack([np.ones(len(te)), _design(te, proxy, mu, sd)]) @ w
        t = te.copy()
        t["corr_gp"] = pred
        t["gp_c"] = np.clip(t["proj_gp"] + pred, 0.0, t["target_len"])
        t["fp_c"] = t["gp_c"] * t["proj_fppg"]
        out.append(t)
    if not out:
        return {"frame": pd.DataFrame(), "gp": (np.nan,) * 3, "fp": (np.nan,) * 3, "n": 0}
    t = pd.concat(out, ignore_index=True)
    d_gp = (t["actual_gp"] - t["proj_gp"]).abs() - (t["actual_gp"] - t["gp_c"]).abs()
    d_fp = (t["actual_total_fp"] - t["proj_total_fp"]).abs() - (t["actual_total_fp"] - t["fp_c"]).abs()
    return {"frame": t, "n": len(t), "gp": boot_mean(d_gp, t["season"], n_boot=n_boot), "fp": boot_mean(d_fp, t["season"], n_boot=n_boot),
            "mae_gp": float((t["actual_gp"] - t["proj_gp"]).abs().mean()), "mae_fp": float((t["actual_total_fp"] - t["proj_total_fp"]).abs().mean())}


def increment_over_controls(f: pd.DataFrame, proxy: str = PRIMARY_PROXY, *, n_boot: int = N_BOOT) -> dict:
    """Post-hoc diagnostic (added after the pre-registered out-of-sample test came back negative; NOT part of the verdict).

    The pre-registered test compares the proxy-corrected projection with the *uncorrected* baseline, so it also scores the general
    correction (the intercept and the controls), which lowers mean error but not necessarily mean absolute error (errors are skewed by
    players who never play). This isolates the proxy: the same walk-forward fit with and without the proxy, paired absolute-error
    difference (controls-only minus with-proxy; positive = the proxy helps).
    """
    a, b = walk_forward_correction(f, None, n_boot=n_boot)["frame"], walk_forward_correction(f, proxy, n_boot=n_boot)["frame"]
    if a.empty or b.empty:
        return {"n": 0, "gp": (np.nan,) * 3, "fp": (np.nan,) * 3}
    d_gp = (a["actual_gp"] - a["gp_c"]).abs() - (b["actual_gp"] - b["gp_c"]).abs()
    d_fp = (a["actual_total_fp"] - a["fp_c"]).abs() - (b["actual_total_fp"] - b["fp_c"]).abs()
    mse = lambda t: float(((t["actual_total_fp"] - t["fp_c"]) ** 2).mean() ** 0.5)  # noqa: E731
    return {"n": len(a), "gp": boot_mean(d_gp, a["season"], n_boot=n_boot), "fp": boot_mean(d_fp, a["season"], n_boot=n_boot),
            "rmse_fp_controls": mse(a), "rmse_fp_proxy": mse(b)}


# --------------------------------------------------------------------------- verdict

def decide(primary_fp: tuple[float, float, float], agree: tuple[int, int], wf_fp: tuple[float, float, float]) -> str:
    """The fixed decision rule of ADR 0024: ``incremental`` | ``descriptive association`` | ``null``."""
    est, lo, hi = primary_fp
    if not (np.isfinite(lo) and np.isfinite(hi)) or not (lo > 0 or hi < 0):
        return "null"
    sign_ok = agree[1] > 0 and agree[0] >= MIN_SEASONS_SIGN
    wf_ok = np.isfinite(wf_fp[1]) and wf_fp[1] > 0
    if sign_ok and wf_ok:
        return "incremental"
    return "descriptive association" if sign_ok else "null"


# --------------------------------------------------------------------------- tables / report

def _ci(t: tuple[float, float, float], fmt: str = "{:+.1f}") -> str:
    if not np.isfinite(t[0]):
        return "n/a"
    return f"{fmt.format(t[0])} [{fmt.format(t[1])}, {fmt.format(t[2])}]"


def proxy_table(f: pd.DataFrame, *, n_boot: int = N_BOOT) -> pd.DataFrame:
    rows = []
    for p in PROXIES:
        gp, fp = proxy_effect(f, p, "err_gp", n_boot=n_boot), proxy_effect(f, p, "err_fp", n_boot=n_boot)
        ps = per_season_effect(f, p, "err_fp")
        a = sign_agreement(ps, fp[0]) if np.isfinite(fp[0]) else (0, 0)
        mu, sd = standardiser(f, p)
        rows.append({"proxy": p + (" (primary)" if p == PRIMARY_PROXY else ""), "mean": mu, "sd (1 unit)": sd,
                     "GP per +1 SD [95% CI]": _ci(gp), "total FP per +1 SD [95% CI]": _ci(fp, "{:+.0f}"),
                     "seasons with pooled sign (FP)": f"{a[0]}/{a[1]}"})
    return pd.DataFrame(rows)


def era_table(f: pd.DataFrame, *, n_boot: int = N_BOOT) -> pd.DataFrame:
    rows = []
    for label, m in (("targets 2016-17..2022-23", f["season"] < "2023-24"), ("targets 2023-24..2025-26 (participation policy)", f["season"] >= "2023-24")):
        g = f[m]
        rows.append({"era": label, "n": int(g["in_universe"].sum()), "GP per +1 SD [95% CI]": _ci(proxy_effect(g, PRIMARY_PROXY, "err_gp", n_boot=n_boot)),
                     "total FP per +1 SD [95% CI]": _ci(proxy_effect(g, PRIMARY_PROXY, "err_fp", n_boot=n_boot), "{:+.0f}")})
    return pd.DataFrame(rows)


def study(f: pd.DataFrame, *, n_boot: int = N_BOOT) -> dict:
    """Everything the ADR reports for one projector's frame (already passed through :func:`prepare`)."""
    fp = proxy_effect(f, PRIMARY_PROXY, "err_fp", n_boot=n_boot)
    gp = proxy_effect(f, PRIMARY_PROXY, "err_gp", n_boot=n_boot)
    ps = per_season_effect(f, PRIMARY_PROXY, "err_fp")
    agree = sign_agreement(ps, fp[0]) if np.isfinite(fp[0]) else (0, 0)
    wf = walk_forward_correction(f, n_boot=n_boot)
    u = f[f["in_universe"]]
    return {"increment": increment_over_controls(f, n_boot=n_boot),
            "hi_gp": proxy_effect(f, "hi_iso", "err_gp", n_boot=n_boot), "hi_fp": proxy_effect(f, "hi_iso", "err_fp", n_boot=n_boot),
            "n_hi": int(u["hi_iso"].sum()), "n": int(f["in_universe"].sum()), "n_all": len(f), "primary_fp": fp, "primary_gp": gp, "per_season": ps, "agree": agree, "wf": wf,
            "verdict": decide(fp, agree, wf["fp"]), "resid_gp": float(u["err_gp"].mean()), "resid_fp": float(u["err_fp"].mean()),
            "mae_gp": float(u["ae_gp"].mean()), "mae_fp": float(u["ae_fp"].mean())}


def render(f: pd.DataFrame, res: dict, others: Mapping[str, dict], proxies: pd.DataFrame, bins: pd.DataFrame, era: pd.DataFrame,
           *, model: str, command: str, n_boot: int) -> str:
    wf = res["wf"]
    lines = [
        "# Load-management proxy study", "",
        f"Generated by `{command}` from real data; projector `{model}`; walk-forward, every feature from seasons before the target. "
        f"Percentile bootstrap, {n_boot} resamples, resampled within season. See ADR 0024. **A missed game has no reason label in this data; "
        "every proxy is contaminated by minor injuries.**", "",
        f"* Study rows: {res['n_all']} projected player-seasons; pre-registered universe (veteran, single team, >= {MIN_MPG:g} mpg and >= {MIN_GP} GP "
        f"the season before): **n = {res['n']}**. Baseline residual over the universe: {res['resid_gp']:+.1f} GP, {res['resid_fp']:+.0f} total FP; "
        f"MAE {res['mae_gp']:.1f} GP, {res['mae_fp']:.0f} total FP.",
        f"* Primary proxy `{PRIMARY_PROXY}` (games missed in interior runs of <= 2, per 82 games), per +1 SD, controlling for prior GP fraction, "
        f"age, age^2, mpg and projected fraction, season fixed effects: **{_ci(res['primary_gp'])} GP**, **{_ci(res['primary_fp'], '{:+.0f}')} total FP**.",
        f"* Sign agreement with the pooled total-FP coefficient: {res['agree'][0]} of {res['agree'][1]} target seasons. "
        f"Out-of-sample correction (targets {FIRST_TEST_SEASON}+, n = {wf['n']}): MAE improvement {_ci(wf['gp'])} GP, {_ci(wf['fp'], '{:+.0f}')} total FP "
        "(positive = the correction helps).",
        f"* **Verdict under the pre-registered rule: {res['verdict']}.**",
        f"* Post-hoc diagnostics (added after seeing the above; not part of the verdict): the proxy's out-of-sample gain over the *same* correction "
        f"without the proxy is {_ci(res['increment']['gp'])} GP, {_ci(res['increment']['fp'], '{:+.0f}')} total FP MAE (positive = the proxy helps); "
        f"a player with {HI_ISO}+ isolated absences (n = {res['n_hi']}) is {_ci(res['hi_gp'])} GP and {_ci(res['hi_fp'], '{:+.0f}')} total FP "
        "further under the projection than an otherwise similar player with fewer, controlling as above.", "",
        "## 1. All proxies (each in its own regression, same controls)", "", df_to_markdown(proxies, index=False), "",
        "## 2. Error by number of isolated absences last season (`iso_n` bins)", "",
        "Signed error is actual minus projected (positive = under-projected).", "", df_to_markdown(bins, index=False), "",
        "## 3. Primary proxy by era (reported, not tested)", "", df_to_markdown(era, index=False), "",
        "## 4. Primary coefficient by target season (total FP per +1 SD)", "", df_to_markdown(res["per_season"].round(1), index=False, floatfmt="{:.1f}"), ""]
    if others:
        lines += ["## 5. Other projectors", "", "| projector | n | GP per +1 SD | total FP per +1 SD | seasons with sign | OOS FP MAE gain | verdict |", "|---|---|---|---|---|---|---|"]
        for name, r in others.items():
            lines.append(f"| {name} | {r['n']} | {_ci(r['primary_gp'])} | {_ci(r['primary_fp'], '{:+.0f}')} | {r['agree'][0]}/{r['agree'][1]} | "
                         f"{_ci(r['wf']['fp'], '{:+.0f}')} | {r['verdict']} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.backtest.load_report", description=__doc__.split("\n\n")[0])
    ap.add_argument("--seasons", default="2016-17:2025-26")
    ap.add_argument("--model", default="baseline")
    ap.add_argument("--compare", default="baseline_injury", help="other projectors (comma list, '' for none)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from src.backtest.harness import expand_seasons
    from src.backtest.runner import load_run_tables
    from src.models.registry import get_projector

    seasons = expand_seasons(args.seasons)
    tables = load_run_tables(argparse.Namespace(synthetic=False))
    scoring = load_league()["scoring"]
    f = prepare(build_load_frame(tables, seasons, get_projector(args.model), scoring, log=print))
    res = study(f, n_boot=args.n_boot)
    others = {}
    for name in [x.strip() for x in args.compare.split(",") if x.strip() and x.strip() != args.model]:
        fo = prepare(build_load_frame(tables, seasons, get_projector(name), scoring, log=print))
        others[name] = study(fo, n_boot=args.n_boot)
    cmd = "python -m src.backtest.load_report " + " ".join(sys.argv[1:] if argv is None else argv)
    proxies, bins, era = proxy_table(f, n_boot=args.n_boot), bin_table(f, n_boot=args.n_boot), era_table(f, n_boot=args.n_boot)
    text = render(f, res, others, proxies, bins, era, model=args.model, command=cmd, n_boot=args.n_boot)
    out = args.out or Path("reports") / f"load_{date.today():%Y-%m-%d}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "load_report.md").write_text(text, encoding="utf-8")
    proxies.to_csv(out / "proxies.csv", index=False)
    bins.to_csv(out / "bins.csv", index=False)
    f.to_parquet(out / "study_frame.parquet", index=False)
    print(text)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
