"""Methodology checks layered on a walk-forward result (ADR 0030).

* ``adp_model_arms``: the "ADP + model" arm. ADP is turned into a prior on season total FP and the model's
  projection is tested as an additional regressor. Every coefficient is fit on *earlier* seasons of the run only
  (expanding window), so season ``s`` is scored out of sample; the first season of the run has no training
  data and is dropped from all arms so they stay comparable.
* ``tier_table``: per-tier comparison of model, ADP and ADP+model (top-N hit / capture, paired over seasons).
* ``risk_group_table``: games-played calibration by risk group (rookie, returner, part-season, age 33+, prime).
* ``loso_table``: leave-one-season-out sensitivity of a run's headline means.

Nothing here downloads data; ADP comes from the already-ingested ``adp`` table via ``benchmarks.load_adp``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats as st

from src.backtest import metrics as M
from src.backtest.harness import BacktestResult
from src.contracts import season_start

TIER_KS = (12, 24, 50, 100)
ADP_TIERS = ((1, 12), (13, 36), (37, 72), (73, 120), (121, 10_000))
ARMS = ("adp", "model", "adp_model")


# --------------------------------------------------------------------------- ADP + model

def _design(adp: np.ndarray, model: np.ndarray | None, model_fill: float) -> np.ndarray:
    la = np.log(np.maximum(np.asarray(adp, float), 1.0))
    cols = [np.ones(len(la)), la, la ** 2]
    if model is not None:
        m = np.asarray(model, float)
        miss = ~np.isfinite(m)
        cols += [np.where(miss, model_fill, m), miss.astype(float)]
    return np.column_stack(cols)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, with_model: bool) -> np.ndarray:
    fill = float(np.nanmean(train["proj_total_fp"])) if with_model else 0.0
    Xtr = _design(train["adp"], train["proj_total_fp"] if with_model else None, fill)
    Xte = _design(test["adp"], test["proj_total_fp"] if with_model else None, fill)
    beta, *_ = np.linalg.lstsq(Xtr, train["actual_total_fp"].to_numpy(float), rcond=None)
    return Xte @ beta


@dataclass
class ArmResult:
    seasons: list[str]
    frames: dict[str, dict[str, pd.DataFrame]]      # arm -> season -> eval frame (ADP universe), has adp_rank
    metrics: dict[str, pd.DataFrame]                # arm -> season_metrics

    def summary(self, arm: str) -> pd.Series:
        return M.summarize_seasons(self.metrics[arm])


def adp_model_arms(model: BacktestResult, adp: pd.DataFrame, *, ks: Sequence[int] = TIER_KS,
                   min_gp: int = M.DEFAULT_MIN_GP) -> ArmResult:
    """Build the adp / model / adp+model arms on the ADP universe of every season after the first."""
    seasons = [s for s in model.seasons[1:] if (adp["season"] == s).any()]
    if not seasons:
        raise ValueError("need at least two backtest seasons with ADP to fit an ADP + model arm")
    joined = {}
    for s in model.seasons:
        f = model.season_frame(s)
        a = adp[adp["season"] == s][["player_id", "adp"]].drop_duplicates("player_id")
        j = f.merge(a, on="player_id", how="inner")
        j["actual_total_fp"] = j["actual_total_fp"].fillna(0.0)
        joined[s] = (f, j)
    frames: dict[str, dict[str, pd.DataFrame]] = {a: {} for a in ARMS}
    rows: dict[str, list] = {a: [] for a in ARMS}
    for s in seasons:
        prior = [joined[p][1] for p in model.seasons if season_start(p) < season_start(s)]
        train = pd.concat(prior, ignore_index=True)
        f, test = joined[s]
        preds = {"adp": _fit_predict(train, test, False), "model": test["proj_total_fp"].to_numpy(float),
                 "adp_model": _fit_predict(train, test, True)}
        for arm, p in preds.items():
            fr = f.drop(columns=["proj_total_fp"]).merge(
                pd.DataFrame({"player_id": test["player_id"].to_numpy(), "proj_total_fp": p,
                              "adp": test["adp"].to_numpy(float)}), on="player_id", how="left")
            fr["projected"] = fr["proj_total_fp"].notna()
            fr["adp_rank"] = fr["adp"].rank(method="first")
            fr["actual_total_fp"] = fr["actual_total_fp"].where(fr["played"], 0.0).where(
                fr["projected"] | fr["played"], np.nan)
            m = M.compute_season_metrics(fr, ks=ks, min_gp=min_gp, rank_only=True)
            pr = fr[fr["projected"]]
            m["mae_total_fp"] = M.mae(pr["proj_total_fp"], pr["actual_total_fp"])
            m["season"] = s
            rows[arm].append(m)
            frames[arm][s] = fr
    return ArmResult(seasons, frames, {a: pd.DataFrame(r).set_index("season") for a, r in rows.items()})


def blend_universe_table(model: BacktestResult, adp: pd.DataFrame, *, ks: Sequence[int] = TIER_KS,
                         min_gp: int = M.DEFAULT_MIN_GP) -> pd.DataFrame:
    """The board's ADP blend (``src.value.adp_blend``) against the model it blends, on the *whole* projected universe.

    For each season after the first the blend is fit on earlier seasons only, applied to that season's projections
    (ADP-listed players get the regression, everyone else keeps the model total) and scored like the model itself:
    rows are season means of Spearman, top-N hit and total-FP MAE, then the season-paired difference (blend - model)
    with a t interval and seasons won. This is what a draft-board user would see, including players ADP does not list.
    """
    from src.value.adp_blend import MIN_ROWS, blend_totals, fit_adp_blend

    seasons = [s for s in model.seasons[1:] if (adp["season"] == s).any()]
    if not seasons:
        raise ValueError("need at least two backtest seasons with ADP to fit the blend")
    per_season = {s: model.season_frame(s).merge(adp[adp["season"] == s][["player_id", "adp"]], on="player_id", how="left")
                  for s in model.seasons}
    arms: dict[str, list] = {"model": [], "blend": []}
    for s in seasons:
        prior = {p: per_season[p] for p in model.seasons if season_start(p) < season_start(s)
                 and per_season[p]["adp"].notna().any()}
        if sum(int(fr["adp"].notna().sum()) for fr in prior.values()) < MIN_ROWS:
            continue                                   # too little earlier ADP to fit the regression: season not scored
        blend = fit_adp_blend(prior, model="model")
        f = per_season[s].copy()
        proj = f["projected"].to_numpy()
        blended = f["proj_total_fp"].to_numpy(float).copy()
        listed = f["adp"].notna().to_numpy() & proj
        if listed.any():
            blended[listed] = blend_totals(f.loc[listed], f.loc[listed, ["player_id", "adp"]], blend).to_numpy()
        for arm, vals in (("model", f["proj_total_fp"].to_numpy(float)), ("blend", blended)):
            fr = f.drop(columns=["adp"]).copy()
            fr["proj_total_fp"] = np.where(proj, vals, np.nan)
            m = M.compute_season_metrics(fr, ks=ks, min_gp=min_gp, rank_only=True)
            pr = fr[fr["projected"]]
            m["mae_total_fp"] = M.mae(pr["proj_total_fp"], pr["actual_total_fp"])
            m["season"] = s
            arms[arm].append(m)
    if not arms["model"]:
        raise ValueError("need at least two backtest seasons with ADP to fit the blend")
    mets = {a: pd.DataFrame(r).set_index("season") for a, r in arms.items()}
    rows = []
    for col in ["spearman_total_fp"] + [f"top{k}_hit" for k in ks] + ["mae_total_fp"]:
        p = _paired(mets["model"][col], mets["blend"][col])
        rows.append({"metric": col, "model": float(mets["model"][col].mean()), "blend": float(mets["blend"][col].mean()),
                     "blend - model": p["diff"], "95% CI": f"[{p['lo']:+.3f}, {p['hi']:+.3f}]", "seasons won": p["wins"]})
    return pd.DataFrame(rows)


def _paired(a: pd.Series, b: pd.Series) -> dict:
    d = (b - a).to_numpy(float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "wins": f"0/{n}"}
    se = d.std(ddof=1) / np.sqrt(n)
    t = st.t.ppf(0.975, n - 1)
    return {"diff": float(d.mean()), "lo": float(d.mean() - t * se), "hi": float(d.mean() + t * se),
            "wins": f"{int((d > 0).sum())}/{n}"}


def tier_table(arms: ArmResult, ks: Sequence[int] = TIER_KS) -> pd.DataFrame:
    """Per top-N: each arm's mean hit-rate and value capture, and the season-paired difference vs ADP.

    The interval is a t interval over the per-season differences (seasons are the independent units);
    with ~9 seasons it is wide, which is the honest statement of how much this comparison can resolve.
    """
    out = []
    for k in ks:
        for metric in ("hit", "capture"):
            col = f"top{k}_{metric}"
            r = {"top-N": k, "metric": metric}
            for arm in ARMS:
                r[arm] = float(arms.metrics[arm][col].mean())
            for arm in ("model", "adp_model"):
                p = _paired(arms.metrics["adp"][col], arms.metrics[arm][col])
                r[f"{arm} - adp"] = p["diff"]
                r[f"{arm} 95% CI"] = f"[{p['lo']:+.3f}, {p['hi']:+.3f}]"
                r[f"{arm} seasons won"] = p["wins"]
            out.append(r)
    return pd.DataFrame(out)


def tier_verdict(tiers: pd.DataFrame) -> list[str]:
    """One plain sentence per top-N: does the model beat ADP there, and does ADP+model beat ADP?"""
    lines = []
    for _, r in tiers[tiers["metric"] == "hit"].iterrows():
        def word(arm, r=r):
            lo, hi = (float(x) for x in r[f"{arm} 95% CI"].strip("[]").split(","))
            return "beats" if lo > 0 else ("loses to" if hi < 0 else "is not distinguishable from")
        lines.append(f"top-{int(r['top-N'])} hit rate: model {word('model')} ADP ({r['model - adp']:+.3f}, won "
                     f"{r['model seasons won']} seasons); ADP+model {word('adp_model')} ADP "
                     f"({r['adp_model - adp']:+.3f}, won {r['adp_model seasons won']}).")
    return lines


def adp_tier_error(arms: ArmResult) -> pd.DataFrame:
    """Mean abs error of season total FP by ADP rank band (rank within the season's ADP list), per arm."""
    rows = []
    for lo, hi in ADP_TIERS:
        r = {"ADP rank": f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"}
        for arm in ARMS:
            errs = []
            for fr in arms.frames[arm].values():
                m = fr["projected"] & fr["adp_rank"].between(lo, hi)
                if m.any():
                    errs.append((fr.loc[m, "proj_total_fp"] - fr.loc[m, "actual_total_fp"]).abs().mean())
            r[arm] = float(np.mean(errs)) if errs else float("nan")
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- risk groups

RISK_GROUPS = ("rookie (no prior NBA games)", "returner (missed >50% of last season, or absent)",
               "part-season last year (50-75% of games)", "age 33+", "prime, played 75%+ last year")


def _prev_season(season: str) -> str:
    y = season_start(season) - 1
    return f"{y}-{str(y + 1)[-2:]}"


def _season_len(gl: pd.DataFrame) -> pd.Series:
    """Season length proxy: the most games any player appeared in that season."""
    return gl.groupby(["season", "player_id"])["game_id"].count().groupby("season").max()


def assign_risk_groups(frame: pd.DataFrame, season: str, game_logs: pd.DataFrame, players: pd.DataFrame,
                       season_len: pd.Series) -> pd.Series:
    """Risk group per player of a season's eval frame, from information strictly before the season
    (games played earlier, birthdate)."""
    s0 = season_start(season)
    prior = game_logs[game_logs["season"].map(season_start) < s0]
    gp_last = game_logs[game_logs["season"] == _prev_season(season)].groupby("player_id")["game_id"].count()
    had = prior["player_id"].unique()
    birth = pd.to_datetime(players.drop_duplicates("player_id").set_index("player_id")["birthdate"], errors="coerce")
    b = pd.Series(birth.reindex(frame["player_id"]).to_numpy(), dtype="datetime64[ns]")
    age = ((pd.Timestamp(year=s0, month=10, day=15) - b) / pd.Timedelta(days=365.25)).to_numpy(float)
    L = float(season_len.get(_prev_season(season), 82))
    frac = frame["player_id"].map(gp_last).fillna(0.0).to_numpy(float) / L
    g = np.full(len(frame), RISK_GROUPS[4], dtype=object)
    g[age >= 33] = RISK_GROUPS[3]
    g[(frac < 0.75) & (frac >= 0.5)] = RISK_GROUPS[2]
    g[frac < 0.5] = RISK_GROUPS[1]
    g[~frame["player_id"].isin(had).to_numpy()] = RISK_GROUPS[0]
    return pd.Series(g, index=frame.index)


def risk_group_table(result: BacktestResult, game_logs: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Games-played calibration by risk group, over projected players (zero-game players count as 0 GP).

    ``appear`` is the share of projected players who played at all. The availability model is fit
    conditional on appearing, so a group with low ``appear`` is where ``proj_gp`` is biased high;
    ``bias_gp`` includes the zero-game players, ``bias_when_played`` excludes them.
    """
    slen = _season_len(game_logs)
    parts = []
    for s in result.seasons:
        f = result.season_frame(s)
        f = f[f["projected"]].copy()
        f["group"] = assign_risk_groups(f, s, game_logs, players, slen)
        parts.append(f)
    d = pd.concat(parts, ignore_index=True)
    out = []
    for g in RISK_GROUPS:
        x = d[d["group"] == g]
        if x.empty:
            continue
        pl = x[x["played"]]
        out.append({"group": g, "n": len(x), "appear": float(x["played"].mean()),
                    "proj_gp": float(x["proj_gp"].mean()), "actual_gp": float(x["actual_gp"].mean()),
                    "bias_gp": float((x["proj_gp"] - x["actual_gp"]).mean()),
                    "bias_when_played": float((pl["proj_gp"] - pl["actual_gp"]).mean()) if len(pl) else float("nan"),
                    "mae_gp": float((x["proj_gp"] - x["actual_gp"]).abs().mean())})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- appearance calibration

P_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0001)


def appearance_calibration(result: BacktestResult) -> pd.DataFrame | None:
    """Calibration of the hurdle model's P(appear) (``proj_p_appear``) against who actually played (ADR 0031).

    One row per probability bin over projected players of every season: how many, the mean predicted probability,
    the observed share with at least one game, and a final ``Brier`` row comparing the model with always predicting
    the pooled base rate. ``None`` when the projector did not emit ``proj_p_appear``.
    """
    f = result.players
    if "proj_p_appear" not in f.columns:
        return None
    d = f[f["projected"] & f["proj_p_appear"].notna()]
    if d.empty:
        return None
    p, y = d["proj_p_appear"].to_numpy(float), d["played"].to_numpy(float)
    rows = []
    for lo, hi in zip(P_BINS[:-1], P_BINS[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            rows.append({"bin": f"{lo:.2f}-{min(hi, 1.0):.2f}", "n": int(m.sum()), "mean_pred": float(p[m].mean()),
                         "observed": float(y[m].mean())})
    brier, base = float(np.mean((p - y) ** 2)), float(np.mean((y.mean() - y) ** 2))
    rows.append({"bin": "Brier (model / base rate)", "n": len(p), "mean_pred": brier, "observed": base})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- season-total interval calibration

def interval_coverage_table(result: BacktestResult, game_logs: pd.DataFrame, players: pd.DataFrame,
                            nominal: float = 0.80) -> pd.DataFrame | None:
    """Out-of-sample coverage of the season-total p10-p90 band and the games-played band, by risk group (ADR 0033).

    For every projected player (zero-game players count as 0 FP / 0 GP): the share of actual season totals inside
    ``[proj_total_fp_p10, proj_total_fp_p90]``, the share below p10 and above p90 (nominally ``(1 - nominal) / 2``
    each), the same coverage for games played against ``[proj_gp_p10, proj_gp_p90]``, and the mean band width. The
    games band is closed at both ends because games played is a whole number. ``None`` when the projector emits no
    season-total band.
    """
    if "proj_total_fp_p10" not in result.players.columns:
        return None
    slen = _season_len(game_logs)
    parts = []
    for s in result.seasons:
        f = result.season_frame(s)
        f = f[f["projected"] & f["proj_total_fp_p10"].notna()].copy()
        f["group"] = assign_risk_groups(f, s, game_logs, players, slen)
        parts.append(f)
    d = pd.concat(parts, ignore_index=True)
    rows = []
    for name, x in [("all projected players", d)] + [(g, d[d["group"] == g]) for g in RISK_GROUPS]:
        if x.empty:
            continue
        a, lo, hi = (x[c].to_numpy(float) for c in ("actual_total_fp", "proj_total_fp_p10", "proj_total_fp_p90"))
        r = {"group": name, "n": len(x), "total in band": float(np.mean((a >= lo) & (a <= hi))),
             "below p10": float(np.mean(a < lo)), "above p90": float(np.mean(a > hi)),
             "mean width (FP)": float(np.mean(hi - lo))}
        if "proj_gp_p10" in x.columns:
            g, glo, ghi = (x[c].to_numpy(float) for c in ("actual_gp", "proj_gp_p10", "proj_gp_p90"))
            r["GP in band"] = float(np.mean((g >= np.floor(glo)) & (g <= np.ceil(ghi))))
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- leave-one-season-out

def loso_table(season_metrics: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Mean of each metric over all seasons, and its range when each single season is left out."""
    rows = []
    for m in metrics:
        v = season_metrics[m].dropna()
        if len(v) < 3:
            continue
        loo = {s: float(v.drop(s).mean()) for s in v.index}
        lo_s, hi_s = min(loo, key=loo.get), max(loo, key=loo.get)
        rows.append({"metric": m, "all seasons": float(v.mean()), "min when dropping": loo[lo_s],
                     "(season)": lo_s, "max when dropping": loo[hi_s], "(season) ": hi_s})
    return pd.DataFrame(rows)


def method_checks_markdown(model: BacktestResult, tables: dict, adp: pd.DataFrame | None) -> str:
    """The report section: risk-group calibration, LOSO sensitivity and (with ADP) the ADP + model arms."""
    from src.backtest.mdutil import df_to_markdown

    md = ["## Methodology checks (ADR 0030)", "",
          "### Games played by risk group", "",
          "Availability is fit on player-seasons with at least one game (conditional on appearing); `appear` shows how "
          "often each group does not, `bias_gp` includes those zero-game players and `bias_when_played` excludes them. "
          "Groups use data strictly before the target season; 'part-season' is a games-missed proxy, not a diagnosed "
          "injury.", "",
          df_to_markdown(risk_group_table(model, tables["game_logs"], tables["players"]),
                         formats={"appear": "{:.1%}", "n": "{:.0f}"}, index=False), ""]
    cal = appearance_calibration(model)
    if cal is not None:
        md += ["### Appearance (hurdle) calibration", "",
               "`proj_p_appear` is the modelled chance a projected player plays at least one game; `observed` is how "
               "often that happened (zero-game players are in the frame). The last row compares Brier scores "
               "(model in `mean_pred`, always-the-base-rate in `observed`; lower is better).", "",
               df_to_markdown(cal, formats={"n": "{:.0f}", "mean_pred": "{:.3f}", "observed": "{:.3f}"}, index=False), ""]
    cov = interval_coverage_table(model, tables["game_logs"], tables["players"])
    if cov is not None:
        md += ["### Season-total interval coverage (ADR 0033)", "",
               "Out-of-sample share of realised season totals inside the modelled p10-p90 band (nominal 80%, 10% below, "
               "10% above), by risk group, with the games-played band alongside. Zero-game players are included.", "",
               df_to_markdown(cov, formats={"n": "{:.0f}", "total in band": "{:.1%}", "below p10": "{:.1%}",
                                            "above p90": "{:.1%}", "GP in band": "{:.1%}", "mean width (FP)": "{:.0f}"},
                              index=False), ""]
    names = [m for m in ("spearman_total_fp", "top12_hit", "top50_hit", "mae_total_fp", "mae_gp")
             if m in model.season_metrics]
    md += ["### Leave-one-season-out sensitivity", "",
           "Mean of each metric over seasons, and the range of that mean when any single season is dropped.", "",
           df_to_markdown(loso_table(model.season_metrics, names), index=False), ""]
    if adp is not None:
        arms = adp_model_arms(model, adp)
        tiers = tier_table(arms)
        summ = pd.DataFrame({a: arms.summary(a) for a in ARMS}).loc[
            ["spearman_total_fp"] + [f"top{k}_hit" for k in TIER_KS] + ["mae_total_fp"]]
        md += ["### ADP + model (ADP universe, seasons after the first)", "",
               "`adp` scores by an ADP-only regression of season FP on log-ADP; `model` is the baseline's own "
               "`proj_total_fp`; `adp_model` adds the model projection to that regression. Coefficients are fit on "
               "earlier seasons only and only ADP-listed players are eligible in every arm.", "",
               df_to_markdown(summ), "", "Top-N tiers (hit = overlap with the realised top N; differences are paired "
               "over seasons):", "", df_to_markdown(tiers, index=False), "", "Verdict at each depth:", "",
               *[f"- {v}" for v in tier_verdict(tiers)], "",
               "Season total FP error by ADP rank band (mean abs error, lower is better):", "",
               df_to_markdown(adp_tier_error(arms), index=False, floatfmt="{:.0f}"), "",
               "### Board blend vs the model it blends (whole projected universe, ADR 0032)", "",
               "ADP-listed players get the regression blend, everyone else keeps the model total; fit on earlier seasons "
               "only. `blend - model` is paired over seasons (positive = blend better for Spearman and hit rates; "
               "negative = blend better for MAE).", "",
               df_to_markdown(blend_universe_table(model, adp), index=False), ""]
    return "\n".join(md)
