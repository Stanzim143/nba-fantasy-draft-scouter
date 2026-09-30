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
               df_to_markdown(adp_tier_error(arms), index=False, floatfmt="{:.0f}"), ""]
    return "\n".join(md)
