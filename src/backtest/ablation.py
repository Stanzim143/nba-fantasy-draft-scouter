"""Ablation: run several projectors over the same seasons and test each layer's lift.

A typical ordered variant list is ``baseline, +injury, +roster, +contract``.  Each variant is a
plain ``Projector``; this module knows nothing about how they differ.

Statistics
----------
For every variant after the first, and for each metric in ``sig_metrics``, the *lift over the
previous variant* is estimated with a **paired bootstrap over players, stratified by season**:
each replicate resamples the same players (same row indices) for both variants within every
season, computes ``metric(B) - metric(A)`` per season and averages across seasons.  A positive
lift always means "better" (error metrics are sign-flipped).

* Comparison universe: players **projected by both variants**, so both are scored on the same
  rows.  Coverage differences are therefore invisible to the paired test; the un-paired table
  columns (full universe, incl. ``top{K}_hit`` and coverage counts) do show them.
* ``lift_*_lo/hi``: 95% percentile-bootstrap CI of the mean lift; ``lift_*_p``: one-sided
  bootstrap p-value for "lift <= 0"; ``lift_*_wins``: seasons with a positive point lift out of
  the seasons compared (a resampling-free sanity check on consistency).
* Verdict: ``improves`` if the CI is entirely above 0, ``hurts`` if entirely below, otherwise
  ``no significant change``.
* Rank metrics (``top{K}_hit``, ``top{K}_capture``, ``ndcg_K``) are bootstrapped over *seasons* instead of players: a
  player resampled twice would fill several of the K slots and distort the metric, so the whole season (player set intact)
  is the resampling unit. With ~10 seasons that interval is coarse but honest; the other metrics resample players.
* ``lift_*_p`` is the share of bootstrap replicates of the observed lift at or below zero, not a null-centred p-value, and
  is not corrected for the many metrics and variants tested.
* Caveats (see docs/backtest.md): players are resampled independently although the same player
  appears in several seasons and layers are compared sequentially (multiple comparisons), so treat
  CIs as optimistic; consistency across seasons (``wins``) matters as much as the p-value.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.backtest import metrics as M
from src.backtest.harness import BacktestResult, walk_forward
from src.backtest.mdutil import df_to_markdown
from src.contracts import Projector

DEFAULT_SIG_METRICS = ("spearman_total_fp", "top50_hit", "mae_total_fp")


@functools.lru_cache(maxsize=512)
def _fn_for(name: str, ks: tuple[int, ...], r: float):
    return M.metric_spec(name, ks, r).fn


def verdict(lo: float, hi: float) -> str:
    """Classify a lift CI (positive = better)."""
    if np.isnan(lo) or np.isnan(hi):
        return "n/a"
    if lo > 0:
        return "improves"
    if hi < 0:
        return "hurts"
    return "no significant change"


@dataclass(frozen=True)
class PairedLift:
    metric: str
    estimate: float       # mean over seasons of (B - A), positive = B better
    lo: float
    hi: float
    p_le_zero: float
    wins: int             # seasons where B strictly beat A
    n_seasons: int        # seasons with a defined lift
    n_players: int        # total common player-seasons compared

    @property
    def verdict(self) -> str:
        return verdict(self.lo, self.hi)


def paired_lift(a: BacktestResult, b: BacktestResult, metric: str, *, n_boot: int = 1000,
                seed: int = 0, level: float = 0.95) -> PairedLift:
    """Lift of ``b`` over ``a`` on ``metric`` (positive = b better); see the module docstring."""
    if a.seasons != b.seasons:
        raise ValueError(f"results cover different seasons: {a.seasons} vs {b.seasons}")
    ks = tuple(a.config["ks"])
    min_gp = int(a.config["min_gp"])
    spec0 = M.metric_spec(metric, ks, 0.0)
    hib = spec0.higher_is_better
    if hib is None:
        raise ValueError(f"{metric!r} has no better/worse direction (bias); pick another metric")
    sign = 1.0 if hib else -1.0

    groups, point, n_players = [], [], 0
    for s in a.seasons:
        fa, fb = a.season_frame(s), b.season_frame(s)
        common = set(fa.loc[fa["projected"], "player_id"]) & set(fb.loc[fb["projected"], "player_id"])
        fa = fa[fa["player_id"].isin(common)].reset_index(drop=True)
        fb = fb[fb["player_id"].isin(common)].reset_index(drop=True)
        r = float(a.season_metrics.loc[s, "replacement_level"])
        sp = M.metric_spec(metric, ks, r)
        pa, aa = M.select_arrays(fa, sp, min_gp)
        pb, ab = M.select_arrays(fb, sp, min_gp)
        if not np.array_equal(aa, ab, equal_nan=True):  # pragma: no cover - would mean corrupt inputs
            raise ValueError(f"season {s}: the two results disagree on the actuals")
        groups.append((pa, pb, aa, np.full(len(pa), r)))
        n_players += len(pa)
        point.append(sign * (sp.fn(pb, aa) - sp.fn(pa, aa)))

    def stat(g):
        fn = _fn_for(metric, ks, float(g[3][0]) if len(g[3]) else 0.0)
        return sign * (fn(g[1], g[2]) - fn(g[0], g[2]))

    boot = M.bootstrap_groups(groups, stat, n_boot=n_boot, seed=seed, level=level,
                              resample="groups" if M.is_rank_metric(metric) else "players")
    pt = np.array(point)
    ok = ~np.isnan(pt)
    return PairedLift(metric, boot.estimate, boot.lo, boot.hi, boot.p_le_zero,
                      int((pt[ok] > 0).sum()), int(ok.sum()), n_players)


# --------------------------------------------------------------------------- ablation table

def _table_metrics(ks: Sequence[int]) -> list[str]:
    return (["spearman_total_fp", "spearman_fppg"] + [f"top{k}_hit" for k in ks] + [f"ndcg_{max(ks)}"]
            + ["mae_fppg", "mae_gp", "mae_total_fp", "rmse_total_fp", "bias_total_fp", "vorp_weighted_mae",
               "n_coverage_miss"])


@dataclass
class AblationResult:
    labels: list[str]
    results: dict[str, BacktestResult]
    table: pd.DataFrame           # index = label; mean metrics + lift columns vs previous row
    lifts: dict[str, dict[str, PairedLift]]   # label -> metric -> lift over the previous variant
    sig_metrics: list[str]

    def to_markdown(self) -> str:
        """Two tables: mean metrics per variant, and lift over the previous variant."""
        metric_cols = [c for c in self.table.columns if not c.startswith("lift_") and c != "verdict"]
        out = ["**Mean metrics across seasons**", "", df_to_markdown(self.table[metric_cols])]
        rows = []
        for lab in self.labels[1:]:
            for m in self.sig_metrics:
                pl = self.lifts[lab][m]
                rows.append({"variant": lab, "over": self.labels[self.labels.index(lab) - 1], "metric": m,
                             "lift": pl.estimate, "95% CI": f"[{pl.lo:+.4f}, {pl.hi:+.4f}]",
                             "p (lift<=0)": pl.p_le_zero, "seasons won": f"{pl.wins}/{pl.n_seasons}",
                             "verdict": pl.verdict})
        if rows:
            out += ["", "**Lift over the previous variant** (positive = better; paired bootstrap over "
                    "players, stratified by season)", "",
                    df_to_markdown(pd.DataFrame(rows), index=False,
                                   formats={"lift": "{:+.4f}", "p (lift<=0)": "{:.3f}"})]
        return "\n".join(out)


def ablation_from_results(
    named: Sequence[tuple[str, BacktestResult]],
    *,
    sig_metrics: Sequence[str] = DEFAULT_SIG_METRICS,
    n_boot: int = 500,
    seed: int = 0,
    level: float = 0.95,
) -> AblationResult:
    """Build the ablation table from already-computed results (order = layer order)."""
    labels = [n for n, _ in named]
    if len(set(labels)) != len(labels):
        raise ValueError(f"variant labels must be unique: {labels}")
    if len(named) < 1:
        raise ValueError("need at least one variant")
    results = dict(named)
    ks = tuple(named[0][1].config["ks"])
    cols = _table_metrics(ks)
    sig = list(sig_metrics)
    rows: dict[str, dict] = {}
    lifts: dict[str, dict[str, PairedLift]] = {}
    for i, (lab, res) in enumerate(named):
        summ = res.summary()
        row = {"n_seasons": float(len(res.seasons))}
        row.update({c: float(summ.get(c, np.nan)) for c in cols})
        if i > 0:
            prev = named[i - 1][1]
            lifts[lab] = {}
            for m in sig:
                pl = paired_lift(prev, res, m, n_boot=n_boot, seed=seed, level=level)
                lifts[lab][m] = pl
                row.update({f"lift_{m}": pl.estimate, f"lift_{m}_lo": pl.lo, f"lift_{m}_hi": pl.hi,
                            f"lift_{m}_p": pl.p_le_zero, f"lift_{m}_wins": float(pl.wins)})
            row["verdict"] = lifts[lab][sig[0]].verdict if sig else "n/a"
        rows[lab] = row
    table = pd.DataFrame.from_dict(rows, orient="index")
    if "verdict" not in table.columns:
        table["verdict"] = "baseline"
    table["verdict"] = table["verdict"].fillna("baseline")
    return AblationResult(labels, results, table, lifts, sig)


def run_ablation(
    tables: Mapping[str, pd.DataFrame],
    variants: Sequence[Projector | tuple[str, Projector]],
    seasons: Sequence[str],
    *,
    sig_metrics: Sequence[str] = DEFAULT_SIG_METRICS,
    n_boot: int = 500,
    seed: int = 0,
    level: float = 0.95,
    **walk_forward_kwargs,
) -> AblationResult:
    """Walk-forward every variant over the same seasons and build the ablation table.

    ``variants`` are Projectors (label = ``.name``) or ``(label, projector)`` pairs, in layer
    order.  Extra keyword arguments go to ``walk_forward`` (scoring, ks, cache_dir, ...).
    """
    named: list[tuple[str, BacktestResult]] = []
    for v in variants:
        label, proj = v if isinstance(v, tuple) else (getattr(v, "name", type(v).__name__), v)
        res = walk_forward(tables, proj, seasons, **walk_forward_kwargs)
        res.projector = label
        named.append((label, res))
    return ablation_from_results(named, sig_metrics=sig_metrics, n_boot=n_boot, seed=seed, level=level)
