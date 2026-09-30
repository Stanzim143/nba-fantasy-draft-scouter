"""Rookie / no-history path: a prior by draft slot learned from the history's own rookie seasons.

A *rookie season* in the history is a player-season whose season start equals the player's
``draft_year`` (so the player is a genuine first-year player, not merely first seen in the data).
For each such row we know the draft pick (``draft_number``; if missing, a round-based midpoint)
and position group. Minutes per game, each per-minute rate, each shooting percentage and the
games-played fraction are regressed (weighted ridge) on::

    intercept, log(pick) - mean, guard, center, age - mean

and the fitted regression is the prior for every rookie of the target season. There is no
individual history to blend in, so a rookie projection is the slot prior and is flagged
``confidence = "low"``. The games-played fraction is additionally capped at the mean fraction of the
history's top-10 picks (``RookiePrior.f_cap``): the log-linear fit otherwise over-projects the very top
slots, whose availability plateaus (they are also the rookies most often held out with injuries).

When the history holds fewer than ``min_rookie_rows`` rookies (a short history, or a synthetic
league with random draft slots) the draft-slot regression is not trusted and every rookie gets the
explicit **replacement-level prior**: the weighted 20th-percentile minutes per game of the league
and the veteran rate prior evaluated at that role. Undrafted players and drafted players with no
history who are not rookies of the target season are excluded, not guessed at (see ADR 0003).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.shrink import wls

UNDRAFTED_PICK = 61.0
RIDGE = 1e-3
TOP_PICK_PLATEAU = 10   # picks 1..this define the games-played plateau (see RookiePrior.f_cap)
MIN_PLATEAU_ROWS = 5    # fewer rookies than this among those picks -> no cap


def to_float(values) -> np.ndarray:
    """Numeric column (float, int, nullable Int64, object) -> float64 with NaN for missing."""
    return pd.to_numeric(pd.Series(values), errors="coerce").astype("float64").to_numpy()


def pick_effective(draft_number, draft_round) -> np.ndarray:
    """Overall pick, with a round midpoint when only the round is known and 61 for undrafted.

    Non-positive picks or rounds (real data has 0 for "not available") count as missing.
    """
    num = to_float(draft_number)
    rnd = to_float(draft_round)
    mid = np.where(np.isfinite(rnd) & (rnd >= 1), (rnd - 1) * 30 + 15.5, UNDRAFTED_PICK)
    return np.where(np.isfinite(num) & (num >= 1), num, mid)


def rookie_design(pick: np.ndarray, group: np.ndarray, age: np.ndarray, centers: tuple[float, float]) -> np.ndarray:
    lp, ag = centers
    age_c = np.where(np.isfinite(age), age - ag, 0.0)
    return np.column_stack([
        np.ones(len(pick)),
        np.log(pick) - lp,
        (np.asarray(group) == "G").astype(float),
        (np.asarray(group) == "C").astype(float),
        age_c,
    ])


@dataclass
class RookiePrior:
    mode: str                                   # "draft_slot" | "replacement"
    n_rookies: int
    centers: tuple[float, float]
    coefs: dict[str, np.ndarray]                # target name -> regression coefficients
    fallback: dict[str, dict[str, float]]       # replacement-level value by target name, then position group
    f_cap: float | None = None                  # ceiling on the predicted games-played fraction, see below

    def predict(self, name: str, pick: np.ndarray, group: np.ndarray, age: np.ndarray) -> np.ndarray:
        if self.mode == "replacement":
            table = self.fallback[name]
            return np.array([table[g] for g in np.asarray(group)], dtype=float)
        out = rookie_design(pick, group, age, self.centers) @ self.coefs[name]
        if name == "f" and self.f_cap is not None:
            out = np.minimum(out, self.f_cap)
        return out

    @property
    def slot_slope(self) -> dict[str, float]:
        """Fitted coefficient on log(pick) per target (negative = later picks produce less)."""
        return {k: float(v[1]) for k, v in self.coefs.items()} if self.mode == "draft_slot" else {}


def fit_rookie_prior(panel: pd.DataFrame, players: pd.DataFrame, rate_targets: dict[str, tuple[str, str]],
                     fallback: dict[str, dict[str, float]], min_rows: int) -> RookiePrior:
    """Fit the draft-slot prior. ``rate_targets`` maps a target name to its (numerator, denominator) panel columns.

    ``fallback`` holds the replacement-level value of every target ("mpg", "f" and the rate names) for
    each position group, used when there are fewer than ``min_rows`` usable rookie rows.
    """
    pl = players.drop_duplicates("player_id").set_index("player_id")
    dy = to_float(pl["draft_year"].reindex(panel["player_id"]).to_numpy())
    is_rookie = np.isfinite(dy) & (dy == panel["s"].to_numpy(float))
    r = panel[is_rookie].copy()
    if len(r) < min_rows:
        return RookiePrior("replacement", int(len(r)), (0.0, 0.0), {}, fallback)
    pick = pick_effective(pl["draft_number"].reindex(r["player_id"]).to_numpy(),
                          pl["draft_round"].reindex(r["player_id"]).to_numpy())
    age = r["age"].to_numpy(float)
    centers = (float(np.mean(np.log(pick))), float(np.nanmean(age)) if np.isfinite(age).any() else 21.0)
    X = rookie_design(pick, r["pos_group"].to_numpy(), age, centers)
    coefs: dict[str, np.ndarray] = {}
    coefs["mpg"] = wls(X, r["mpg"].to_numpy(float), r["gp"].to_numpy(float), ridge=RIDGE)
    f = r["f"].to_numpy(float)
    coefs["f"] = wls(X, f, np.ones(len(r)), ridge=RIDGE)
    # Availability does not keep improving toward pick 1: the games-played fraction is regressed linearly
    # on log(pick), which extrapolates past what the top slots actually deliver (walk-forward 2017-18 to
    # 2025-26: projected ~78 games for picks 1-3 against ~59 played, +18 games; a cap at the mean of the top-10
    # picks' own fraction cuts that to +5 with every other slot unchanged). The cap is estimated from the
    # history's own rookies, like everything else here.
    plateau = (pick <= TOP_PICK_PLATEAU) & np.isfinite(f)
    f_cap = float(np.mean(f[plateau])) if plateau.sum() >= MIN_PLATEAU_ROWS else None
    for name, (num, den) in rate_targets.items():
        d = r[den].to_numpy(float)
        rate = np.where(d > 0, r[num].to_numpy(float) / np.where(d > 0, d, 1.0), np.nan)
        coefs[name] = wls(X, rate, d, ridge=RIDGE)
    return RookiePrior("draft_slot", int(len(r)), centers, coefs, fallback, f_cap)
