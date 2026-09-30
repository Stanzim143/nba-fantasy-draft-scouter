"""Markdown report + PNG charts for a backtest run, written to ``reports/<run-id>/``.

Generated reports are deliberately not committed (``reports/.gitignore``).  Charts use the
headless Agg backend; palette is the validated default categorical order (blue, orange, aqua) on
a light surface, with direct labels and recessive gridlines.

Files written: ``report.md``, ``metrics_by_season.csv``, ``players.parquet``, ``run.json``
(from ``BacktestResult.save``), ``misses.csv`` (if analysed), ``ablation.csv`` (if any) and
``metrics_by_season.png``, ``scatter_total_fp.png``, ``calibration.png``, ``ablation.png``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.backtest import metrics as M  # noqa: E402
from src.backtest.ablation import AblationResult, paired_lift  # noqa: E402
from src.backtest.harness import BacktestResult  # noqa: E402
from src.backtest.mdutil import df_to_markdown  # noqa: E402
from src.backtest.misses import MissAnalysis  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#8a8984"]   # blue, orange, aqua, neutral
NOMINAL = {"share_below_p10": 0.10, "share_above_p90": 0.10, "share_in_p10_p90": 0.80}


def default_run_id(result: BacktestResult) -> str:
    h = str(result.metadata.get("data_hash", "nohash"))[:8]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", result.projector)
    return f"{name}_{result.seasons[0]}_{result.seasons[-1]}_{h}"


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)


def _fig(w: float, h: float, **kw):
    fig, axes = plt.subplots(figsize=(w, h), facecolor=SURFACE, **kw)
    return fig, axes


def _short(season: str) -> str:
    return season[2:] if len(season) == 7 else season


# --------------------------------------------------------------------------- charts

def chart_metrics_by_season(results: Sequence[BacktestResult], path: Path) -> None:
    ks = results[0].config["ks"]
    k_mid = 50 if 50 in ks else ks[len(ks) // 2]
    panels = [("spearman_total_fp", "Spearman rank corr. (total FP)"),
              (f"top{k_mid}_hit", f"Top-{k_mid} hit rate"),
              ("mae_total_fp", "MAE of total FP (lower is better)")]
    fig, axes = _fig(12, 3.6, nrows=1, ncols=len(panels))
    for ax, (col, title) in zip(np.atleast_1d(axes), panels):
        for i, r in enumerate(results):
            y = r.season_metrics[col].to_numpy(dtype=float)
            x = np.arange(len(y))
            ax.plot(x, y, color=SERIES[i % len(SERIES)], linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=r.projector)
        _style(ax, title)
        ax.set_xticks(np.arange(len(results[0].seasons)))
        ax.set_xticklabels([_short(s) for s in results[0].seasons], rotation=45 if len(results[0].seasons) > 8 else 0)
    axes_flat = np.atleast_1d(axes)
    if len(results) > 1:
        axes_flat[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def chart_scatter(result: BacktestResult, path: Path) -> bool:
    f = result.players
    ok = f["projected"] & f["played"] & f["proj_total_fp"].notna()
    if not ok.any() or result.config.get("rank_only"):
        return False
    dnp = f["projected"] & ~f["played"]
    fig, ax = _fig(5.6, 5.2)
    ax.scatter(f.loc[ok, "proj_total_fp"], f.loc[ok, "actual_total_fp"], s=9, alpha=0.35,
               color=SERIES[0], linewidths=0, label=f"projected and played (n={int(ok.sum())})")
    if dnp.any():
        ax.scatter(f.loc[dnp, "proj_total_fp"], np.zeros(int(dnp.sum())), s=16, alpha=0.6, marker="x",
                   color=SERIES[1], linewidths=1, label=f"projected, 0 games (n={int(dnp.sum())})")
    hi = float(np.nanmax([f.loc[ok, "proj_total_fp"].max(), f.loc[ok, "actual_total_fp"].max()]))
    ax.plot([0, hi], [0, hi], color=INK2, linewidth=1, linestyle="--", label="perfect")
    rho = M.spearman(f.loc[ok, "proj_total_fp"], f.loc[ok, "actual_total_fp"])
    _style(ax, f"Projected vs actual season total FP\n(all seasons pooled, Spearman {rho:.3f})",
           "projected total FP", "actual total FP")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return True


def chart_calibration(result: BacktestResult, path: Path) -> bool:
    sm = result.season_metrics
    if sm["band_games"].fillna(0).sum() == 0:
        return False
    x = np.arange(len(sm))
    fig, (a1, a2) = _fig(10.5, 3.8, nrows=1, ncols=2)
    a1.bar(x, sm["share_in_p10_p90"], color=SERIES[0], width=0.6)
    a1.axhline(0.80, color=INK, linewidth=1, linestyle="--")
    a1.text(len(sm) - 0.5, 0.815, "nominal 80%", ha="right", fontsize=8, color=INK2)
    _style(a1, "Share of games inside [p10, p90]", ylabel="observed share of games")
    a1.set_ylim(0, 1)
    a2.plot(x, sm["share_below_p10"], color=SERIES[0], marker="o", markersize=5, linewidth=2,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="below p10")
    a2.plot(x, sm["share_above_p90"], color=SERIES[1], marker="o", markersize=5, linewidth=2,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label="above p90")
    a2.axhline(0.10, color=INK, linewidth=1, linestyle="--")
    a2.text(0, 0.105, "nominal 10%", fontsize=8, color=INK2, va="bottom")
    _style(a2, "Tail shares (each should be ~10%)")
    a2.legend(frameon=False, fontsize=8.5, labelcolor=INK2)
    for a in (a1, a2):
        a.set_xticks(x)
        a.set_xticklabels([_short(s) for s in sm.index], rotation=45 if len(sm) > 8 else 0)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return True


def chart_ablation(abl: AblationResult, path: Path) -> bool:
    if len(abl.labels) < 2:
        return False
    metric = abl.sig_metrics[0]
    labs = abl.labels[1:]
    est = np.array([abl.lifts[lab][metric].estimate for lab in labs])
    lo = np.array([abl.lifts[lab][metric].lo for lab in labs])
    hi = np.array([abl.lifts[lab][metric].hi for lab in labs])
    fig, ax = _fig(6.5, 0.9 + 0.7 * len(labs))
    y = np.arange(len(labs))
    ax.errorbar(est, y, xerr=[est - lo, hi - est], fmt="o", color=SERIES[0], ecolor=SERIES[0],
                elinewidth=2, capsize=3, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{lab}\nvs {abl.labels[abl.labels.index(lab) - 1]}" for lab in labs])
    ax.invert_yaxis()
    _style(ax, f"Lift over previous layer: {metric} (95% CI, positive = better)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- markdown

_SUMMARY_LABELS = {
    "spearman_total_fp": "Spearman, total FP", "spearman_fppg": "Spearman, FPPG",
    "mae_fppg": "MAE, FPPG", "mae_gp": "MAE, games played", "mae_total_fp": "MAE, total FP",
    "rmse_total_fp": "RMSE, total FP", "bias_total_fp": "Bias, total FP (pred - actual)",
}


def _label(name: str) -> str:
    if name in _SUMMARY_LABELS:
        return _SUMMARY_LABELS[name]
    m = re.match(r"top(\d+)_hit", name)
    if m:
        return f"Top-{m.group(1)} hit rate"
    m = re.match(r"ndcg_(\d+)", name)
    return f"NDCG@{m.group(1)}" if m else name


def summary_table(result: BacktestResult, n_boot: int, seed: int) -> pd.DataFrame:
    if result.config.get("rank_only"):
        names = ["spearman_total_fp"] + [f"top{k}_hit" for k in result.config["ks"]] + [f"ndcg_{max(result.config['ks'])}"]
    else:
        names = None
    ci = result.summary_ci(names, n_boot=n_boot, seed=seed)
    out = pd.DataFrame({"metric": [_label(n) for n in ci.index], "mean over seasons": ci["estimate"].to_numpy(),
                        "95% CI": [f"[{lo:.3f}, {hi:.3f}]" if not (np.isnan(lo) or np.isnan(hi)) else "n/a"
                                   for lo, hi in zip(ci["lo"], ci["hi"])]})
    return out


def _season_table(result: BacktestResult) -> pd.DataFrame:
    ks = result.config["ks"]
    cols = (["spearman_total_fp", "spearman_fppg"] + [f"top{k}_hit" for k in ks]
            + [f"ndcg_{max(ks)}", "mae_fppg", "mae_gp", "mae_total_fp", "bias_total_fp"])
    t = result.season_metrics[cols].copy()
    t.loc["mean"] = t.mean()
    return t


def _coverage_table(result: BacktestResult) -> pd.DataFrame:
    ks = result.config["ks"]
    cols = ["n_projected", "n_played", "n_projected_no_games", "n_coverage_miss", "coverage_miss_fp_share",
            f"top{max(ks)}_unprojected", "replacement_level"]
    return result.season_metrics[cols].copy()


def _calibration_table(result: BacktestResult) -> pd.DataFrame:
    cs = M.calibration_shares(result.players)     # pooled over all games in all seasons
    rows = [{"band": label, "nominal": NOMINAL.get(col, 0.50), "observed (all games pooled)": cs[col]}
            for col, label in (("share_below_p10", "below p10"), ("share_in_p10_p90", "inside [p10, p90]"),
                               ("share_above_p90", "above p90"), ("share_below_p50", "below p50"))]
    return pd.DataFrame(rows)


def _benchmark_section(result: BacktestResult, benchmarks: Sequence[BacktestResult], n_boot: int, seed: int) -> str:
    ks = result.config["ks"]
    names = ["spearman_total_fp", "spearman_fppg"] + [f"top{k}_hit" for k in ks] + [f"ndcg_{max(ks)}", "mae_fppg",
                                                                                     "mae_gp", "mae_total_fp"]
    cols = {r.projector: r.summary() for r in (result, *benchmarks)}
    side = pd.DataFrame({k: [v.get(n, np.nan) for n in names] for k, v in cols.items()}, index=[_label(n) for n in names])
    lines = ["Mean of each metric across seasons:", "", df_to_markdown(side), ""]
    rows = []
    for b in benchmarks:
        for n in ("spearman_total_fp", f"top{max(ks) if 50 not in ks else 50}_hit", "mae_total_fp"):
            try:
                pl = paired_lift(b, result, n, n_boot=n_boot, seed=seed)
            except ValueError:
                continue
            rows.append({"model": result.projector, "over": b.projector, "metric": n, "lift": pl.estimate,
                         "95% CI": f"[{pl.lo:+.4f}, {pl.hi:+.4f}]", "seasons won": f"{pl.wins}/{pl.n_seasons}",
                         "verdict": pl.verdict})
    if rows:
        lines += ["Paired lift of the model over each benchmark (players projected by both; positive = model better):",
                  "", df_to_markdown(pd.DataFrame(rows), index=False, formats={"lift": "{:+.4f}"})]
    return "\n".join(lines)


NOTES = """\
- Every projection is made from data strictly before the target season (`History.until`, plus a
  sanitised `players` table); see `docs/backtest.md` for the leakage guards.
- Players projected but who played 0 games count as 0 actual FP; players who played but were not
  projected are a coverage miss (see the coverage table). Neither is dropped.
- Confidence intervals resample players within each season (stratified bootstrap); they do not
  capture between-season variation, so treat them as optimistic and check how many seasons agree.
- Miss tags are heuristics computed from the data, not causal explanations.
- {synthetic}"""


def write_report(
    result: BacktestResult,
    out_dir: str | Path,
    *,
    benchmarks: Sequence[BacktestResult] = (),
    ablation: AblationResult | None = None,
    misses: MissAnalysis | None = None,
    run_id: str | None = None,
    n_boot: int = 500,
    seed: int = 0,
    command: str | None = None,
    extra_md: str | None = None,
) -> Path:
    """Write the report and charts under ``out_dir/<run_id>/`` and return that directory."""
    run_id = run_id or default_run_id(result)
    out = Path(out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    result.save(out)
    synthetic = bool(result.metadata.get("synthetic"))
    md: list[str] = [f"# Backtest report: {result.projector}", ""]
    if synthetic:
        md += ["> **SYNTHETIC DATA. These numbers describe a simulated league and are NOT a result. "
               "They exist to smoke-test the pipeline.**", ""]

    meta = result.metadata
    # Column name "value", not "": df_to_markdown resets an unnamed index into a column also
    # called "", which would collide with a same-named data column here and corrupt every cell
    # in it (see the fixed footgun this avoids, documented in mdutil.df_to_markdown).
    info = pd.DataFrame({"value": [
        run_id, result.projector, f"{result.seasons[0]} to {result.seasons[-1]} ({len(result.seasons)} seasons)",
        meta.get("data_hash", "")[:16], meta.get("created_utc", ""), meta.get("harness_version", ""),
        str(meta.get("leak_check") or "not run"), result.config["replacement"],
        f"min_gp={result.config['min_gp']}, K={result.config['ks']}"]},
        index=["run id", "projector", "seasons", "data hash (sha256, first 16)", "created (UTC)", "harness version",
               "future-invariance check", "replacement level", "settings"])
    md += ["## Run", "", df_to_markdown(info), ""]
    if command:
        md += ["Reproduce:", "", "```", command, "```", ""]

    all_results = [result, *benchmarks]
    chart_metrics_by_season(all_results, out / "metrics_by_season.png")
    md += ["## Headline metrics", "", df_to_markdown(summary_table(result, n_boot, seed), index=False),
           "", "Rank metrics score how well the draft order was recovered; error metrics score the "
           "size of the projection. CIs are a stratified bootstrap over players.", "",
           "## By season", "", df_to_markdown(_season_table(result)), "",
           "![metrics by season](metrics_by_season.png)", ""]

    md += ["## Evaluation universe and coverage", "",
           df_to_markdown(_coverage_table(result), formats={
               "coverage_miss_fp_share": "{:.1%}", "replacement_level": "{:.0f}", "n_projected": "{:.0f}",
               "n_played": "{:.0f}", "n_projected_no_games": "{:.0f}", "n_coverage_miss": "{:.0f}",
               f"top{max(result.config['ks'])}_unprojected": "{:.0f}"}),
           "", "`n_projected_no_games` players are scored as 0 FP; `n_coverage_miss` players played but had no "
           "projection (rookies, returners); the last column counts them inside the actual top-K.", ""]

    if benchmarks:
        md += ["## Benchmarks", "", _benchmark_section(result, benchmarks, n_boot, seed), ""]

    if chart_scatter(result, out / "scatter_total_fp.png"):
        md += ["## Predicted vs actual", "", "![predicted vs actual](scatter_total_fp.png)", ""]

    if chart_calibration(result, out / "calibration.png"):
        md += ["## Floor and ceiling calibration", "",
               "`fppg_p10`/`fppg_p90` are game-level quantiles, so they are checked against each actual "
               "game's fantasy points (not the season mean). A calibrated band leaves ~10% of games below "
               "p10, ~10% above p90.", "", df_to_markdown(_calibration_table(result), floatfmt="{:.3f}", index=False), "",
               "![calibration](calibration.png)", ""]

    if ablation is not None:
        ablation.table.to_csv(out / "ablation.csv")
        md += ["## Ablation", "", ablation.to_markdown(), ""]
        if chart_ablation(ablation, out / "ablation.png"):
            md += ["![ablation](ablation.png)", ""]

    if misses is not None:
        misses.table.to_csv(out / "misses.csv", index=False)
        md += ["## Biggest misses", "", misses.to_markdown(), ""]

    if extra_md:
        md += [extra_md, ""]

    md += ["## Notes", "", NOTES.format(synthetic=(
        "This run used synthetic data." if synthetic else "This run used real ingested data.")), ""]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    return out
