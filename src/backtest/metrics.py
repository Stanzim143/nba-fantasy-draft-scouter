"""Backtest metrics.  Array functions are pure, vectorised and NaN/edge-case safe.

Conventions
-----------
* ``pred`` / ``actual`` are 1-D array-likes of equal length.  ``bias = mean(pred - actual)``:
  positive means the model over-projects.
* Error metrics and Spearman drop pairs where either side is NaN; an empty or degenerate input
  returns NaN (never raises, never 0.0-as-a-lie).
* Top-K selection breaks ties deterministically by position (stable sort), so callers should
  pass rows in a canonical order (the harness sorts by ``player_id``).
* Top-K / NDCG treat a NaN *prediction* as "not selectable / ranked last" (a coverage miss) and a
  NaN *actual* as "not eligible for the actual top-K" (a player who did not play).

The frame-level layer (``MetricSpec``, ``build_specs``, ``compute_season_metrics``) is the single
source of truth for what each named metric means: the harness, the bootstrap CIs and the
ablation significance tests all go through it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

DEFAULT_KS = (12, 50, 100)
DEFAULT_MIN_GP = 10          # FPPG metrics ignore players with fewer games (tiny-sample noise)

Replacement = float | Callable[[np.ndarray], float] | None


# --------------------------------------------------------------------------- helpers

def _f64(x) -> np.ndarray:
    return np.asarray(x, dtype="float64").ravel()


def _pair(pred, actual) -> tuple[np.ndarray, np.ndarray]:
    p, a = _f64(pred), _f64(actual)
    if p.shape != a.shape:
        raise ValueError(f"pred and actual differ in length: {p.shape} vs {a.shape}")
    ok = ~(np.isnan(p) | np.isnan(a))
    return p[ok], a[ok]


def _check_k(k: int) -> int:
    if int(k) != k or k <= 0:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return int(k)


def _top_indices(score: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest non-NaN scores, ties broken by position (stable)."""
    idx = np.flatnonzero(~np.isnan(score))
    order = np.argsort(-score[idx], kind="stable")
    return idx[order[:k]]


# --------------------------------------------------------------------------- rank / error metrics

def _average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given their average rank (same as scipy.stats.rankdata)."""
    n = len(x)
    order = np.argsort(x, kind="stable")
    xs = x[order]
    new = np.concatenate([[True], xs[1:] != xs[:-1]])
    starts = np.flatnonzero(new)
    ends = np.concatenate([starts[1:], [n]])
    avg = (starts + ends + 1) / 2.0
    ranks = np.empty(n)
    ranks[order] = avg[np.cumsum(new) - 1]
    return ranks


def spearman(pred, actual) -> float:
    """Spearman rank correlation (Pearson on average ranks, so ties are handled)."""
    p, a = _pair(pred, actual)
    if len(p) < 2:
        return float("nan")
    rp, ra = _average_ranks(p), _average_ranks(a)
    sp, sa = rp.std(), ra.std()
    if sp == 0 or sa == 0:
        return float("nan")
    return float(np.mean((rp - rp.mean()) * (ra - ra.mean())) / (sp * sa))


def mae(pred, actual) -> float:
    p, a = _pair(pred, actual)
    return float(np.mean(np.abs(p - a))) if len(p) else float("nan")


def rmse(pred, actual) -> float:
    p, a = _pair(pred, actual)
    return float(np.sqrt(np.mean((p - a) ** 2))) if len(p) else float("nan")


def bias(pred, actual) -> float:
    """Mean signed error, ``mean(pred - actual)``."""
    p, a = _pair(pred, actual)
    return float(np.mean(p - a)) if len(p) else float("nan")


# --------------------------------------------------------------------------- top-K metrics

def topk_overlap(pred, actual, k: int) -> float:
    """Hit rate: |predicted top-K  intersect  actual top-K| / min(K, #eligible actual players).

    If fewer than K players are projected the hit rate is capped accordingly (a model that
    covers only 40 players cannot score above 40/K), which is what makes coverage visible.
    """
    k = _check_k(k)
    p, a = _f64(pred), _f64(actual)
    if p.shape != a.shape:
        raise ValueError(f"pred and actual differ in length: {p.shape} vs {a.shape}")
    denom = min(k, int((~np.isnan(a)).sum()))
    if denom == 0:
        return float("nan")
    hit = np.intersect1d(_top_indices(p, k), _top_indices(a, k))
    return float(len(hit) / denom)


def topk_value_capture(pred, actual, k: int) -> float:
    """Draft-value captured: actual FP of the predicted top-K / actual FP of the true top-K.

    1.0 = the model's K picks were worth as much as the best possible K.  Actual FP is floored
    at 0 and NaN (did not play) counts as 0, so a projected-but-absent player is a dead pick.
    """
    k = _check_k(k)
    p, a = _f64(pred), _f64(actual)
    if p.shape != a.shape:
        raise ValueError(f"pred and actual differ in length: {p.shape} vs {a.shape}")
    val = np.maximum(np.nan_to_num(a, nan=0.0), 0.0)
    ideal = np.sort(val)[::-1][:k].sum()
    if ideal <= 0:
        return float("nan")
    return float(val[_top_indices(p, k)].sum() / ideal)


def ndcg(pred, actual, k: int | None = None) -> float:
    """NDCG@k with linear gains = actual total FP (floored at 0), discount 1/log2(rank+1).

    Top-heavy: a swap at rank 3 costs far more than a swap at rank 90.  Ties in ``pred`` get
    the *expected* DCG under random tie-breaking (order-invariant); NaN predictions are ranked
    last, tied.  Returns NaN when the ideal DCG is 0.
    """
    p, a = _f64(pred), _f64(actual)
    if p.shape != a.shape:
        raise ValueError(f"pred and actual differ in length: {p.shape} vs {a.shape}")
    n = len(p)
    if n == 0:
        return float("nan")
    kk = n if k is None else min(_check_k(k), n)
    gain = np.maximum(np.nan_to_num(a, nan=0.0), 0.0)
    disc = np.zeros(n)
    disc[:kk] = 1.0 / np.log2(np.arange(kk) + 2.0)
    ideal = float(np.sort(gain)[::-1] @ disc)
    if ideal <= 0:
        return float("nan")
    cd = np.concatenate([[0.0], np.cumsum(disc)])
    key = np.where(np.isnan(p), -np.inf, p)
    order = np.argsort(-key, kind="stable")
    sk = key[order]
    new = np.concatenate([[True], sk[1:] != sk[:-1]])
    starts = np.flatnonzero(new)
    ends = np.concatenate([starts[1:], [n]])
    expected_disc = (cd[ends] - cd[starts]) / (ends - starts)
    dcg = float(gain[order] @ expected_disc[np.cumsum(new) - 1])
    return dcg / ideal


# --------------------------------------------------------------------------- VORP-weighted error

def replacement_level(values, n_rostered: int) -> float:
    """Value of the best *un-rostered* player: the (n_rostered + 1)-th highest value.

    Falls back to 0.0 when the universe is no bigger than the rostered pool (nobody is below
    replacement, so replacement is the zero-production floor).
    """
    v = np.sort(_f64(values)[~np.isnan(_f64(values))])[::-1]
    return float(v[n_rostered]) if 0 <= n_rostered < len(v) else 0.0


def default_n_rostered() -> int:
    """teams x roster size from config/league.yaml (182 for the current 14-slot league: 13 managers +
    1 placeholder, per the live ESPN league; see ADR 0008)."""
    from src.value.league import load_league, rostered_players
    return int(rostered_players(load_league()))


def default_replacement(values) -> float:
    """Documented default: the (N+1)-th best realised season total FP, N = rostered players.

    ``values`` are the actual season totals of *every player who played*.  Replacement is taken
    from realised results (what a manager could really have picked up off waivers that year),
    not from the model's own predictions.  Swap in the value engine's replacement level later
    by passing any callable ``values -> float`` (or a float) as ``replacement``.
    """
    return replacement_level(values, default_n_rostered())


def resolve_replacement(replacement: Replacement, actual_played_totals) -> float:
    if replacement is None:
        return default_replacement(actual_played_totals)
    if callable(replacement):
        return float(replacement(_f64(actual_played_totals)))
    return float(replacement)


def vorp_weighted_error(pred, actual, replacement_value: float, kind: str = "mae") -> float:
    """Error weighted by how much value is at stake.

    weight_i = max(max(pred_i, actual_i) - R, 0): a player matters if the model *or* reality
    put him above replacement level R, so both false stars and missed breakouts count, while
    errors among replacement-level filler are ignored.  ``kind='mae'`` -> sum(w|e|)/sum(w);
    ``kind='bias'`` -> sum(w(pred-actual))/sum(w).  NaN if every weight is 0.
    """
    if kind not in ("mae", "bias"):
        raise ValueError("kind must be 'mae' or 'bias'")
    p, a = _pair(pred, actual)
    if not len(p):
        return float("nan")
    w = np.maximum(np.maximum(p, a) - replacement_value, 0.0)
    tot = w.sum()
    if tot <= 0:
        return float("nan")
    e = np.abs(p - a) if kind == "mae" else (p - a)
    return float((w * e).sum() / tot)


# --------------------------------------------------------------------------- bootstrap

@dataclass(frozen=True)
class Bootstrap:
    """Percentile-bootstrap summary of a statistic.

    ``p_le_zero`` is the one-sided bootstrap p-value that the statistic is <= 0
    ((1 + #{boot <= 0}) / (n_boot + 1)); it is meaningful for paired *differences* (lift). It is the share of replicates of
    the *observed* statistic at or below zero, not a p-value from a null-centred (permutation-style) distribution, and no
    multiplicity correction is applied across the many metrics and variants a report tests: read it as a rough ordering of
    evidence, with the CI and the seasons-won count, not as a calibrated significance level.
    """
    estimate: float
    lo: float
    hi: float
    se: float
    p_le_zero: float
    n_boot: int
    n_groups: int


def is_rank_metric(name: str) -> bool:
    """Top-K hit / capture and NDCG: metrics where a player resampled twice would fill several of the K slots."""
    return bool(re.match(r"^(top\d+_|ndcg)", name))


def bootstrap_groups(
    groups: Sequence[Sequence[np.ndarray]],
    stat: Callable[[Sequence[np.ndarray]], float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
    resample: str = "players",
) -> Bootstrap:
    """Bootstrap of the mean-over-groups of ``stat``; ``resample`` picks the unit (default: players within each group).

    ``resample="groups"`` resamples whole groups (seasons) with replacement instead and averages their (precomputed)
    ``stat`` values. Use it for top-K / capture / NDCG metrics: with player-level resampling a player drawn twice
    appears as duplicate rows that can fill several of the K slots, which distorts the metric; resampling seasons keeps
    every group's player set intact (at the price of a coarse, honest interval when there are few seasons). With fewer
    than three non-empty groups it falls back to resampling players.

    Player-level (default): stratified by group (season).

    ``groups[g]`` is a tuple of equal-length arrays (e.g. ``(pred, actual)``).  Each replicate
    resamples every group's rows with replacement (same row indices for every array in the
    group, so paired predictions stay paired), evaluates ``stat`` per group and averages the
    non-NaN group values.  Deterministic given ``seed``.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    if resample not in ("players", "groups"):
        raise ValueError("resample must be 'players' or 'groups'")
    arrs = [[np.asarray(x) for x in g] for g in groups]
    est_vals = [stat(g) for g in arrs]
    estimate = float(np.nanmean(est_vals)) if any(not np.isnan(v) for v in est_vals) else float("nan")
    rng = np.random.default_rng(seed)
    if resample == "groups" and sum(1 for g in arrs if len(g[0])) >= 3:
        ev = np.asarray(est_vals, float)
        pick = rng.integers(0, len(ev), size=(n_boot, len(ev)))
        vals = ev[pick]
        cnt = (~np.isnan(vals)).sum(axis=1)
        reps = np.where(cnt > 0, np.nansum(vals, axis=1) / np.maximum(cnt, 1), np.nan)
        return _summarise(reps, estimate, n_boot, len(arrs), level)
    idx = [rng.integers(0, len(g[0]), size=(n_boot, len(g[0]))) if len(g[0]) else None for g in arrs]
    reps = np.full(n_boot, np.nan)
    for b in range(n_boot):
        vals = []
        for g, ix in zip(arrs, idx):
            if ix is None:
                continue
            v = stat([x[ix[b]] for x in g])
            if not np.isnan(v):
                vals.append(v)
        if vals:
            reps[b] = np.mean(vals)
    return _summarise(reps, estimate, n_boot, len(arrs), level)


def _summarise(reps: np.ndarray, estimate: float, n_boot: int, n_groups: int, level: float) -> Bootstrap:
    ok = reps[~np.isnan(reps)]
    if len(ok) < 2:
        nan = float("nan")
        return Bootstrap(estimate, nan, nan, nan, nan, n_boot, n_groups)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(ok, [alpha, 1 - alpha])
    p = (1 + int((ok <= 0).sum())) / (len(ok) + 1)
    return Bootstrap(estimate, float(lo), float(hi), float(ok.std(ddof=1)), float(p), n_boot, n_groups)


def bootstrap_ci(pred, actual, metric: Callable[[np.ndarray, np.ndarray], float], *,
                 n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> Bootstrap:
    """Percentile CI for ``metric(pred, actual)`` by resampling players with replacement."""
    p, a = _f64(pred), _f64(actual)
    return bootstrap_groups([(p, a)], lambda g: metric(g[0], g[1]), n_boot=n_boot, seed=seed, level=level)


def paired_bootstrap_diff(
    groups: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    higher_is_better: bool = True,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> Bootstrap:
    """Paired bootstrap of the *improvement* of prediction B over prediction A.

    ``groups[g] = (pred_a, pred_b, actual)`` over the same players.  The statistic is
    ``metric(B) - metric(A)`` (sign flipped if lower is better), so a positive estimate always
    means B is better, and ``p_le_zero`` is the p-value against "B is no better than A".
    """
    sign = 1.0 if higher_is_better else -1.0

    def stat(g):
        return sign * (metric(g[1], g[2]) - metric(g[0], g[2]))

    return bootstrap_groups(groups, stat, n_boot=n_boot, seed=seed, level=level)


# --------------------------------------------------------------------------- frame-level metrics

@dataclass(frozen=True)
class MetricSpec:
    """How one named metric is read off an evaluation frame."""
    name: str
    pred: str                      # frame column holding the prediction
    actual: str                    # frame column holding the outcome
    subset: str                    # 'projected' | 'fppg' | 'universe'
    fn: Callable[[np.ndarray, np.ndarray], float]
    higher_is_better: bool | None  # None: not monotone (bias)


def select_arrays(frame: pd.DataFrame, spec: MetricSpec, min_gp: int = DEFAULT_MIN_GP
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Rows of ``frame`` that ``spec`` scores, as (pred, actual) arrays.

    * ``projected``: every projected player (zero-game players count with actual 0).
    * ``fppg``: projected players with at least ``min_gp`` games (FPPG is noise below that).
    * ``universe``: projected + unprojected-but-played; the actual side is NaN for players who
      did not play, and the pred side is NaN for the unprojected (coverage misses).
    """
    if spec.subset == "projected":
        m = frame["projected"].to_numpy()
    elif spec.subset == "fppg":
        m = frame["projected"].to_numpy() & (frame["actual_gp"].to_numpy() >= min_gp)
    elif spec.subset == "universe":
        m = np.ones(len(frame), dtype=bool)
    else:
        raise ValueError(f"unknown subset {spec.subset!r}")
    p = _f64(frame[spec.pred].to_numpy())[m]
    a = _f64(frame[spec.actual].to_numpy())[m]
    if spec.subset == "universe":
        a = np.where(frame["played"].to_numpy()[m], a, np.nan)
    return p, a


def build_specs(ks: Sequence[int] = DEFAULT_KS, replacement_value: float | None = None
                ) -> dict[str, MetricSpec]:
    """Registry of every named per-season metric.  VORP metrics need ``replacement_value``."""
    S: dict[str, MetricSpec] = {}

    def add(name, pred, actual, subset, fn, hib):
        S[name] = MetricSpec(name, pred, actual, subset, fn, hib)

    add("spearman_total_fp", "proj_total_fp", "actual_total_fp", "projected", spearman, True)
    add("spearman_fppg", "proj_fppg", "actual_fppg", "fppg", spearman, True)
    for k in ks:
        add(f"top{k}_hit", "proj_total_fp", "actual_total_fp", "universe",
            lambda p, a, k=k: topk_overlap(p, a, k), True)
        add(f"top{k}_capture", "proj_total_fp", "actual_total_fp", "universe",
            lambda p, a, k=k: topk_value_capture(p, a, k), True)
        add(f"ndcg_{k}", "proj_total_fp", "actual_total_fp", "universe",
            lambda p, a, k=k: ndcg(p, a, k), True)
    for fam, (pc, ac, sub) in {"fppg": ("proj_fppg", "actual_fppg", "fppg"),
                               "gp": ("proj_gp", "actual_gp", "projected"),
                               "total_fp": ("proj_total_fp", "actual_total_fp", "projected")}.items():
        add(f"mae_{fam}", pc, ac, sub, mae, False)
        add(f"rmse_{fam}", pc, ac, sub, rmse, False)
        add(f"bias_{fam}", pc, ac, sub, bias, None)   # 'better' is 'closer to 0', not monotone
    if replacement_value is not None:
        r = float(replacement_value)
        add("vorp_weighted_mae", "proj_total_fp", "actual_total_fp", "projected",
            lambda p, a: vorp_weighted_error(p, a, r, "mae"), False)
        add("vorp_weighted_bias", "proj_total_fp", "actual_total_fp", "projected",
            lambda p, a: vorp_weighted_error(p, a, r, "bias"), None)
    return S


_TOPK_RE = re.compile(r"^(?:top(\d+)_(?:hit|capture)|ndcg_(\d+))$")


def metric_spec(name: str, frame_ks: Sequence[int] = DEFAULT_KS, replacement_value: float | None = None
                ) -> MetricSpec:
    """Look up one metric by name (parsing arbitrary K out of ``top{K}_hit`` etc.)."""
    m = _TOPK_RE.match(name)
    ks = tuple(frame_ks) + ((int(m.group(1) or m.group(2)),) if m else ())
    specs = build_specs(ks, replacement_value)
    if name not in specs:
        raise KeyError(f"unknown metric {name!r}; known: {sorted(specs)}")
    return specs[name]


def compute_season_metrics(
    frame: pd.DataFrame,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    replacement: Replacement = None,
    min_gp: int = DEFAULT_MIN_GP,
    rank_only: bool = False,
) -> dict[str, float]:
    """All metrics + coverage counts + calibration shares for ONE season's evaluation frame."""
    played_tot = frame.loc[frame["played"], "actual_total_fp"].to_numpy()
    r = resolve_replacement(replacement, played_tot)
    specs = build_specs(ks, r)
    out: dict[str, float] = {}
    for name, spec in specs.items():
        if rank_only and not (name.startswith(("spearman_total", "top", "ndcg"))):
            out[name] = float("nan")
            continue
        p, a = select_arrays(frame, spec, min_gp)
        out[name] = spec.fn(p, a)

    projected, played = frame["projected"].to_numpy(), frame["played"].to_numpy()
    unproj = played & ~projected
    tot_played = float(played_tot.sum())
    out["n_projected"] = float(projected.sum())
    out["n_played"] = float(played.sum())
    out["n_projected_no_games"] = float((projected & ~played).sum())
    out["n_coverage_miss"] = float(unproj.sum())
    out["coverage_miss_fp_share"] = (float(frame.loc[unproj, "actual_total_fp"].sum()) / tot_played
                                     if tot_played > 0 else float("nan"))
    a_rank = np.where(played, _f64(frame["actual_total_fp"].to_numpy()), np.nan)
    for k in ks:
        top = _top_indices(a_rank, k)
        out[f"top{k}_unprojected"] = float((~projected[top]).sum())
    out["replacement_level"] = r
    out.update(calibration_shares(frame))
    return out


def calibration_shares(frame: pd.DataFrame) -> dict[str, float]:
    """Game-level calibration of the floor/median/ceiling: observed share of games below p10,
    below p50, above p90, and inside [p10, p90] (nominal 0.10 / 0.50 / 0.10 / 0.80)."""
    bg = float(frame["band_games"].sum())
    pg = float(frame["p50_games"].sum())
    below, above = float(frame["n_below_p10"].sum()), float(frame["n_above_p90"].sum())
    nan = float("nan")
    return {
        "band_games": bg,
        "share_below_p10": below / bg if bg else nan,
        "share_above_p90": above / bg if bg else nan,
        "share_in_p10_p90": 1 - (below + above) / bg if bg else nan,
        "share_below_p50": float(frame["n_below_p50"].sum()) / pg if pg else nan,
    }


def summarize_seasons(season_metrics: pd.DataFrame) -> pd.Series:
    """Unweighted mean across seasons (each season = one draft, so each counts once)."""
    return season_metrics.mean(numeric_only=True)


def metric_higher_is_better(name: str) -> bool | None:
    """True/False for ordered metrics; None where 'better' is not monotone (bias, counts)."""
    if name.startswith(("bias_", "vorp_weighted_bias")):
        return None
    if name.startswith(("mae_", "rmse_", "vorp_weighted_mae")):
        return False
    if name.startswith(("spearman", "top", "ndcg")) and not name.endswith("_unprojected"):
        return True
    return None
