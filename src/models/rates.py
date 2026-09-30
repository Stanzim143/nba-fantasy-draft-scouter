"""Per-minute / per-attempt rate estimation: recency weighting, age adjustment, shrinkage.

One machinery serves every modelled quantity (``Spec``): minutes per game, eight per-minute
counting rates, three shooting percentages, and the game-level FP volatility ratio.

For a target player-season and lags k = 1..K (K seasons back) with numerator ``x_k`` and exposure
``e_k`` (minutes, attempts or games)::

    est   = sum_k d^(k-1) * (x_k + e_k * shift_k) / sum_k d^(k-1) * e_k       (age-adjusted, recency-weighted)
    n_eff = sum_k d^(k-1) * e_k
    rate  = prior + n_eff / (n_eff + kappa) * (est - prior)                  (empirical-Bayes shrinkage)

``shift_k`` is the fitted age curve's expected change between the player's age in lag k and now.
The decay ``d`` and ``kappa`` are chosen per spec by minimising next-season squared error (exposure
weighted) over every (player, season) in the history that has at least one earlier season. The
``prior`` is a weighted regression of rate on position group and minutes per game ("role-appropriate
mean"); minutes and volatility use a constant prior.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.age import AgeCurve, FLAT, consecutive_pairs, fit_age_curve
from src.models.config import BaselineConfig
from src.models.panel import index_panel, lag_matrix
from src.models.shrink import fit_kappa, shrink, wls


@dataclass(frozen=True)
class Spec:
    name: str
    num: str
    den: str
    kind: str              # rate | pct | minutes | ratio
    age: bool = True       # apply an age curve
    prior: str = "regression"   # regression | constant | one


SPECS: dict[str, Spec] = {s.name: s for s in (
    Spec("fga", "fga", "min", "rate"),
    Spec("fta", "fta", "min", "rate"),
    Spec("fg3a", "fg3a", "min", "rate"),
    Spec("reb", "reb", "min", "rate"),
    Spec("ast", "ast", "min", "rate"),
    Spec("stl", "stl", "min", "rate"),
    Spec("blk", "blk", "min", "rate"),
    Spec("tov", "tov", "min", "rate"),
    Spec("fg_pct", "fgm", "fga", "pct"),
    Spec("ft_pct", "ftm", "fta", "pct"),
    Spec("fg3_pct", "fg3m", "fg3a", "pct"),
)}
MINUTES = Spec("mpg", "min", "gp", "minutes", prior="constant")
VOLATILITY = Spec("vol", "z_gp", "gpz", "ratio", age=False, prior="one")

BOUNDS = {"rate": (0.0, 5.0), "pct": (0.0, 1.0), "minutes": (0.0, 48.0), "ratio": (0.2, 5.0)}


def design(group: np.ndarray, mpg: np.ndarray | None) -> np.ndarray:
    """Prior-regression design: intercept, guard, center, minutes per game (forward is the baseline)."""
    group = np.asarray(group)
    cols = [np.ones(len(group)), (group == "G").astype(float), (group == "C").astype(float)]
    if mpg is not None:
        cols.append(np.asarray(mpg, float) / 10.0)
    return np.column_stack(cols)


@dataclass
class Frame:
    """Lagged view of the panel for a set of (player, target season) rows."""

    pids: np.ndarray
    target_s: np.ndarray
    age: np.ndarray            # age in the target season (NaN if unknown)
    group: np.ndarray
    lags: dict[str, np.ndarray]   # col -> (n, K), NaN where the season is absent
    n_lags: int

    def col(self, name: str) -> np.ndarray:
        return np.nan_to_num(self.lags[name])

    def has_history(self) -> np.ndarray:
        return (self.col("gp") > 0).any(axis=1)

    def subset(self, mask: np.ndarray) -> "Frame":
        return Frame(self.pids[mask], self.target_s[mask], self.age[mask], self.group[mask],
                     {k: v[mask] for k, v in self.lags.items()}, self.n_lags)


def lag_columns() -> list[str]:
    cols = {"gp", "f", "min"}
    for sp in list(SPECS.values()) + [MINUTES, VOLATILITY]:
        cols.update((sp.num, sp.den))
    return sorted(cols)


def frame_from_panel_rows(panel: pd.DataFrame, cfg: BaselineConfig) -> Frame:
    """Training frame: every panel row, looking back at earlier seasons."""
    pidx = index_panel(panel)
    pids = panel["player_id"].to_numpy("int64")
    s = panel["s"].to_numpy("int64")
    return Frame(pids, s, panel["age"].to_numpy(float), panel["pos_group"].to_numpy(),
                 lag_matrix(pidx, pids, s, cfg.n_lags, lag_columns()), cfg.n_lags)


def frame_for_target(panel: pd.DataFrame, pids: np.ndarray, target_s: int, ages: np.ndarray,
                     groups: np.ndarray, cfg: BaselineConfig) -> Frame:
    pidx = index_panel(panel)
    pids = pids.astype("int64")
    ts = np.full(len(pids), target_s, dtype="int64")
    return Frame(pids, ts, ages.astype(float), np.asarray(groups),
                 lag_matrix(pidx, pids, ts, cfg.n_lags, lag_columns()), cfg.n_lags)


@dataclass
class SpecFit:
    spec: Spec
    decay: float
    kappa: float
    curve: AgeCurve
    beta: np.ndarray
    fitted: bool           # decay/kappa learned from data (False = documented fall-back)
    n_train: int
    loss: float

    def prior_for(self, group: np.ndarray, mpg: np.ndarray | None) -> np.ndarray:
        if self.spec.prior == "regression":
            return design(group, mpg) @ self.beta
        return np.full(len(group), float(self.beta[0]))


def estimate_raw(fit: SpecFit, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    """Recency-weighted, age-adjusted raw rate and its effective exposure."""
    sp = fit.spec
    num, den = frame.col(sp.num), frame.col(sp.den)
    ks = np.arange(1, frame.n_lags + 1)
    w = fit.decay ** (ks - 1)
    if sp.age and not fit.curve.is_flat:
        shift = fit.curve.shift(frame.age[:, None] - ks[None, :], frame.age[:, None])
    else:
        shift = 0.0
    n_eff = (den * w).sum(axis=1)
    est = np.where(n_eff > 0, ((num + den * shift) * w).sum(axis=1) / np.maximum(n_eff, 1e-12), np.nan)
    lo, hi = BOUNDS[sp.kind]
    return np.clip(est, lo, hi), n_eff


def predict_rate(fit: SpecFit, frame: Frame, mpg: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Shrunk rate for each frame row and the raw estimate's effective exposure."""
    est, n_eff = estimate_raw(fit, frame)
    prior = fit.prior_for(frame.group, mpg)
    est = np.where(np.isnan(est), prior, est)
    lo, hi = BOUNDS[fit.spec.kind]
    return np.clip(shrink(est, prior, n_eff, fit.kappa), lo, hi), n_eff


def fit_spec(spec: Spec, panel: pd.DataFrame, train: Frame, cfg: BaselineConfig,
             mpg_pred: np.ndarray | None = None) -> SpecFit:
    """Estimate the age curve, prior regression, decay and kappa for one quantity.

    ``mpg_pred`` (aligned with ``train``'s rows) is the minutes model's history-only prediction (lag inputs only; the minutes model is fit on the same rows, so in-sample fit) for every
    training row; with ``cfg.tune_with_predicted_mpg`` it replaces the target season's actual MPG as the prior
    input while tuning decay/kappa, matching forecasting (``predict_rate``). Without it, actual MPG is used.
    """
    # 1. age curve (delta method on consecutive seasons)
    curve = FLAT
    if spec.age:
        curve = fit_age_curve(consecutive_pairs(panel, spec.num, spec.den),
                              degree=cfg.age_degree, min_pairs=cfg.age_min_pairs)
    # 2. prior mean
    den = panel[spec.den].to_numpy(float)
    rate = np.where(den > 0, panel[spec.num].to_numpy(float) / np.where(den > 0, den, 1.0), np.nan)
    if spec.prior == "regression":
        X = design(panel["pos_group"].to_numpy(), panel["mpg"].to_numpy(float))
        beta = wls(X, rate, den)
    elif spec.prior == "constant":
        ok = den > 0
        beta = np.array([float((rate[ok] * den[ok]).sum() / den[ok].sum())])
    else:
        beta = np.array([1.0])
    proto = SpecFit(spec, cfg.default_decay, cfg.default_kappa[spec.name], curve, beta, False, 0, float("inf"))

    # 3. decay and kappa by next-season prediction error inside the history
    rows = train.has_history()
    if rows.sum() < cfg.min_fit_rows:
        return proto
    sub = train.subset(rows)
    y_den = panel[spec.den].to_numpy(float)[rows]
    y_num = panel[spec.num].to_numpy(float)[rows]
    y = np.where(y_den > 0, y_num / np.where(y_den > 0, y_den, 1.0), np.nan)
    if cfg.tune_with_predicted_mpg and mpg_pred is not None:
        mpg_in = np.asarray(mpg_pred, float)[rows]
    else:
        mpg_in = panel["mpg"].to_numpy(float)[rows]
    prior = proto.prior_for(sub.group, mpg_in)
    best: SpecFit | None = None
    for d in cfg.decay_grid:
        cand = SpecFit(spec, d, proto.kappa, curve, beta, True, int(rows.sum()), float("inf"))
        est, n_eff = estimate_raw(cand, sub)
        kappa, loss, ok = fit_kappa(est, prior, n_eff, y, y_den, default=proto.kappa, min_rows=cfg.min_fit_rows)
        if not ok:
            return proto
        cand.kappa, cand.loss = kappa, loss
        if best is None or loss < best.loss:
            best = cand
    assert best is not None
    return best
