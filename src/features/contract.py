"""Contract feature layer: the rookie-scale clock, the only contract signal that is point-in-time (ADR 0013).

No source in this project carries dated salary or contract data (ADR 0005 D4 rejected Spotrac,
RealGM, HoopsHype and Basketball-Reference; see ADR 0013 for what the ingested data does and does not
contain). What *can* be derived honestly is the rookie-scale clock, from two facts that are public on
draft night and never change: the draft year and the draft slot. Everything here is a pure function of
``players`` (``draft_year``, ``draft_round``, ``draft_number``, already sanitised by ``History``) and the
target season, so there is nothing that could leak: no game log, no outcome, no date after the draft.

CBA basis (2023 CBA, in force through the 2029-30 season; mark every line below as an ASSUMPTION about
how the target player's contract actually looks, because we never see the contract itself):

* A **first-round pick** signs a rookie-scale contract: 2 guaranteed seasons plus 2 team-option seasons,
  4 seasons in all. ``scale_year`` = ``target_season - draft_year + 1`` (1 = the rookie season).
  ``scale_year == 3`` is the first *option* year (the club had to pick it up by Oct 31 of the year
  before), ``scale_year == 4`` the final scale season: the "contract year". Eligibility for a rookie-scale
  extension opens after the 3rd season and closes just before the 4th tips off, so ``scale_year == 4``
  is also the extension-eligible season. A player not extended plays that season and then hits
  restricted free agency, which is the ``scale_year == 5`` state here: he is on a new (or qualifying)
  contract whose length and size we cannot see.
* A **second-round pick or undrafted player** has no scale. First contracts run from a non-guaranteed
  minimum deal up to 4 years, so the clock is unknowable; we only keep the first four seasons since the
  draft for second-rounders as a weak "early, cheap, insecure" marker. Undrafted players carry no draft
  year in ``players`` at all, so they get no clock (every feature is zero).
* Veterans (more than 5 seasons since the draft) carry no information at all: their contract year is
  invisible without contract data, which is exactly why this layer cannot test the "veterans play harder
  in contract years" hypothesis PLANNING.md was originally after.

Known ways the nominal clock is wrong (uncertainty, not fixed here): draft-and-stash players sign years
after the draft (the clock starts at signing); first-rounders are waived or traded before the option
year; option years are declined; extensions signed a year early move the walk year; a first-rounder can
sign for less than the full scale. The layer measures whether the nominal clock is nevertheless useful.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.contracts import season_start
from src.models.rookies import pick_effective, to_float
from src.models.shrink import wls

SCALE_YEARS = 4
TOP_PICK = 10             # picks 1..10: near-certain option pickup, extension offers
LATE_PICK = 20            # picks 20..30: options are the ones most often declined
ROUND2_YEARS = 4          # second-rounders: the first four seasons since the draft carry a weak marker

FEATURE_NAMES = (
    "r1_y1", "r1_y2", "r1_y3", "r1_y4", "r1_y5",
    "r2_y1", "r2_y2", "r2_y3", "r2_y4",
    "r1_y4_top", "r1_y4_late",
)

MIN_FIT_ROWS = 150        # fewer usable training rows than this -> no adjustment
MIN_TRAIN_GP = 10         # a training residual needs at least this many actual games
MIN_MARKED_ROWS = 40      # fewer training rows carrying any non-zero clock feature -> no adjustment
MIN_CV_GAIN = 0.0005      # required relative CV improvement over "no adjustment" (0.05%)
RIDGE_GRID = (1.0, 10.0, 100.0, 1000.0)
WEIGHT_GP_CAP = 60.0
FACTOR_LO, FACTOR_HI = 0.8, 1.25   # a rookie-scale clock never justifies more than a modest scaling


# --------------------------------------------------------------------------- the clock

@dataclass(frozen=True)
class ContractClock:
    """Per-player nominal rookie-scale clock for one target season. All arrays share one length."""

    player_id: np.ndarray
    years_since_draft: np.ndarray    # 1 = rookie season, NaN when the draft year is unknown or in the future
    draft_round: np.ndarray          # 1, 2, or 0 = unknown / undrafted
    pick: np.ndarray                 # overall pick (round midpoint when only the round is known), 61 undrafted
    scale_year: np.ndarray           # 1..4 for first-rounders within the scale, else NaN
    is_option_year: np.ndarray       # first-rounder in year 3
    is_contract_year: np.ndarray     # first-rounder in year 4: the final scale season
    is_extension_eligible: np.ndarray  # same season as the contract year (window closes before opening night)
    is_post_scale_year: np.ndarray   # first-rounder in year 5: RFA / re-signed, contract not observable


def contract_clock(players: pd.DataFrame, target_season: str,
                   player_ids: np.ndarray | None = None) -> ContractClock:
    """Nominal rookie-scale clock of every requested player in ``target_season``.

    ``players`` is used only for ``draft_year``, ``draft_round`` and ``draft_number``. A draft year after
    the target season (only possible if the caller skipped ``History``'s sanitising) yields NaN, never a
    clock, so nothing from the future can reach a feature.
    """
    p = players.drop_duplicates("player_id").set_index("player_id")
    pids = np.asarray(p.index if player_ids is None else player_ids, dtype="int64")
    q = p.reindex(pids)
    s = season_start(target_season)
    dy = to_float(q["draft_year"])
    yrs = np.where(np.isfinite(dy) & (dy <= s), s - dy + 1.0, np.nan)
    pick = pick_effective(q["draft_number"], q["draft_round"])
    rnd = to_float(q["draft_round"])
    num = to_float(q["draft_number"])
    round_ = np.where(np.isfinite(rnd) & (rnd >= 1), np.minimum(rnd, 2.0),
                      np.where(np.isfinite(num) & (num >= 1), np.where(num <= 30, 1.0, 2.0), 0.0))
    round_ = np.where(np.isfinite(yrs), round_, 0.0)
    first = round_ == 1
    scale = np.where(first & (yrs >= 1) & (yrs <= SCALE_YEARS), yrs, np.nan)
    return ContractClock(
        player_id=pids, years_since_draft=yrs, draft_round=round_, pick=pick, scale_year=scale,
        is_option_year=first & (yrs == 3), is_contract_year=first & (yrs == SCALE_YEARS),
        is_extension_eligible=first & (yrs == SCALE_YEARS), is_post_scale_year=first & (yrs == SCALE_YEARS + 1))


def clock_design(clock: ContractClock) -> np.ndarray:
    """Indicator matrix, columns as in :data:`FEATURE_NAMES`. Veterans, undrafted and unknown rows are all zero."""
    y, r, pick = clock.years_since_draft, clock.draft_round, clock.pick
    cols = [((r == 1) & (y == k)).astype(float) for k in range(1, SCALE_YEARS + 2)]
    cols += [((r == 2) & (y == k)).astype(float) for k in range(1, ROUND2_YEARS + 1)]
    final = (r == 1) & (y == SCALE_YEARS)
    cols += [(final & (pick <= TOP_PICK)).astype(float), (final & (pick >= LATE_PICK)).astype(float)]
    return np.column_stack(cols)


# --------------------------------------------------------------------------- fitted adjustment

@dataclass
class ContractFit:
    """The fitted adjustment: fantasy points per game to add, relative to a player with no clock."""

    coefs: np.ndarray | None          # per standardised-feature coefficient (no intercept)
    scale: np.ndarray | None
    ridge: float
    enabled: bool
    diagnostics: dict = field(default_factory=dict)

    def adjustment(self, clock: ContractClock) -> np.ndarray:
        n = len(clock.player_id)
        if not self.enabled or self.coefs is None or n == 0:
            return np.zeros(n)
        return (clock_design(clock) / self.scale) @ self.coefs

    def adjustment_from_design(self, design: np.ndarray) -> np.ndarray:
        """Same adjustment for a caller-supplied design matrix (ADR 0019's terms features share this fit)."""
        n = len(design)
        if not self.enabled or self.coefs is None or n == 0:
            return np.zeros(n)
        return (design / self.scale) @ self.coefs


def _cv_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, groups: np.ndarray, ridge: float) -> float:
    """Leave-one-season-out weighted squared error of the ridge fit (intercept included, then dropped at
    prediction time exactly as the projector will use it: the veteran/no-clock group is the reference)."""
    tot = 0.0
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if tr.sum() < 30 or te.sum() == 0:
            return float("inf")
        scale = X[tr].std(axis=0)
        scale = np.where(scale < 1e-9, 1.0, scale)
        beta = wls(np.column_stack([np.ones(tr.sum()), X[tr] / scale]), y[tr], w[tr], ridge=ridge / tr.sum())
        pred = (X[te] / scale) @ beta[1:]
        tot += float(np.sum(w[te] * (y[te] - pred) ** 2))
    return tot / float(w.sum())


def fit_adjustment(X: np.ndarray, resid: np.ndarray, weight: np.ndarray, season: np.ndarray, *,
                   min_rows: int = MIN_FIT_ROWS, names: tuple[str, ...] = FEATURE_NAMES,
                   min_gain: float = MIN_CV_GAIN) -> ContractFit:
    """Fit ``resid ~ intercept + clock features`` and decide whether it earns its place.

    ``resid`` is actual minus base-projected FPPG for one historical player-season, ``season`` its start
    year (leave-one-season-out folds). The intercept absorbs the base model's average bias for players
    with no clock and is *not* applied: only the contrast against them is. Switched off (zero adjustment)
    unless cross-validation beats "no adjustment" by :data:`MIN_CV_GAIN`. ``names`` labels the columns of ``X``
    in the diagnostics (default: this module's rookie-scale features; ADR 0019 reuses the fit with its own);
    ``min_gain`` is the required cross-validated gain (a diagnostic override, e.g. -1 to force the layer on).
    """
    ok = np.isfinite(resid) & np.isfinite(weight) & (weight > 0)
    diag: dict = {"train_rows": int(ok.sum()), "train_seasons": int(len(np.unique(np.asarray(season)[ok]))),
                  "marked_rows": int((X[ok].sum(axis=1) > 0).sum())}
    if ok.sum() < min_rows or diag["train_seasons"] < 2 or diag["marked_rows"] < MIN_MARKED_ROWS:
        diag["reason"] = "too few training rows, seasons or clock-marked rows"
        return ContractFit(None, None, RIDGE_GRID[1], False, diag)
    Xo, y, w, s = X[ok], np.asarray(resid, float)[ok], np.minimum(np.asarray(weight, float)[ok], WEIGHT_GP_CAP), np.asarray(season)[ok]
    zero_loss = float(np.sum(w * y ** 2) / w.sum())
    # The honest reference is "apply nothing": the same fold-wise loss when the contrast is dropped, i.e. the
    # residual around its (unapplied) intercept is what the base model already carries.
    best = (float("inf"), RIDGE_GRID[1])
    for ridge in RIDGE_GRID:
        loss = _cv_loss(Xo, y, w, s, ridge)
        if loss < best[0]:
            best = (loss, ridge)
    cv_loss, ridge = best
    gain = (zero_loss - cv_loss) / zero_loss if zero_loss > 0 else 0.0
    diag.update({"zero_adjustment_loss": zero_loss, "cv_loss": cv_loss, "cv_gain_vs_zero": gain, "ridge": ridge})
    scale = Xo.std(axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    beta = wls(np.column_stack([np.ones(len(y)), Xo / scale]), y, w, ridge=ridge / len(y))
    diag["coefficients_fppg"] = {n: float(b / sc) for n, b, sc in zip(names, beta[1:], scale)}
    if not np.isfinite(cv_loss) or gain < min_gain:
        diag["reason"] = f"cross-validated gain {gain:.5f} below the {min_gain} threshold"
        return ContractFit(None, None, ridge, False, diag)
    diag["reason"] = "enabled"
    return ContractFit(beta[1:], scale, ridge, True, diag)


@dataclass
class TrainingSet:
    design: np.ndarray
    resid: np.ndarray
    weight: np.ndarray
    season: np.ndarray


def build_training_set(players: pd.DataFrame, seasons: list[str], project: Callable[[str], pd.DataFrame | None],
                       actual: pd.DataFrame) -> TrainingSet | None:
    """Stack the walk-forward residuals of ``seasons``.

    ``project(s)`` is the base projection of season ``s`` computed from data before ``s`` (columns
    ``player_id``, ``proj_fppg``); ``actual`` has ``player_id``, ``s``, ``gp``, ``fppg`` for completed
    seasons. The clock of a historical row uses only its own season and the static draft facts.
    """
    Xs, ys, ws, ss = [], [], [], []
    for s in seasons:
        sy = season_start(s)
        proj = project(s)
        if proj is None or proj.empty:
            continue
        act = actual[(actual["s"] == sy) & (actual["gp"] >= MIN_TRAIN_GP)][["player_id", "gp", "fppg"]]
        m = proj[["player_id", "proj_fppg"]].merge(act, on="player_id", how="inner")
        if m.empty:
            continue
        Xs.append(clock_design(contract_clock(players, s, m["player_id"].to_numpy("int64"))))
        ys.append((m["fppg"] - m["proj_fppg"]).to_numpy(float))
        ws.append(m["gp"].to_numpy(float))
        ss.append(np.full(len(m), sy))
    if not Xs:
        return None
    return TrainingSet(np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), np.concatenate(ss))


def factor_from_adjustment(proj_fppg: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """Multiplier turning ``proj_fppg`` into ``proj_fppg + adj``, clipped to ``[FACTOR_LO, FACTOR_HI]``.

    A projection at or below ~one fantasy point per game cannot be scaled meaningfully and is left alone.
    """
    p = np.asarray(proj_fppg, float)
    safe = np.where(p > 1.0, p, np.nan)
    f = (safe + np.asarray(adj, float)) / safe
    return np.clip(np.where(np.isfinite(f), f, 1.0), FACTOR_LO, FACTOR_HI)
