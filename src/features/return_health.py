"""'Returned and healthy' availability features derived purely from ``History`` (ADR 0022).

The baseline availability model sees one season-level fraction of the schedule played. A player who missed a long lead block
(games before his first appearance in the season) and then played nearly every team game after returning looks, at the season level,
like a chronically absent player (Jayson Tatum 2025-26: first game 2026-03-06, 16 of the 20 team games after; 16 / 82 = 0.20).
These features expose the shape: how much of the season was a lead block, how healthy the tail after it was, and a flag for Task A's
"long lead block, then healthy" cohort (ADR 0021).

Everything is a pure function of ``History.game_logs`` / ``History.team_games`` / ``History.players`` (all sliced to seasons before the
target by ``History.until``), so it is point-in-time by construction, the same argument as ``src.features.injury``.

Per player-season (:func:`build_return_panel`), on the player's *primary* team's schedule (the team he played most for; ``L`` = its team
games, the same denominator as ``src.features.injury`` and ``src.backtest.return_report``):

* ``lead_frac``  = games missed before the first appearance / ``L`` (the lead block).
* ``tail_len``   = ``L`` - games missed before the first appearance (team games from the first appearance to the season end).
* ``tail_gp``    = games played (all of them are after the first appearance).
* ``tail_health`` (raw)  = ``tail_gp / tail_len``; ``tail_health_s`` (shrunk) = ``(tail_gp + K * H0) / (tail_len + K)`` with
  ``K = SHRINK_K`` games and ``H0 = NEUTRAL_HEALTH``, so a two-game tail is not read as 100% health.
* ``lead_season`` = the season has a lead block worth speaking of: ``lead_frac >= LEAD_MIN``.
* ``return_flag`` = Task A's definitions on the raw tail: ``lead_frac >= 0.25``, ``tail_len >= 15``, ``tail_health >= 0.75``.

A season counts as *no lead block* (``lead_frac = 0``, no flag) if the player was not a veteran that season (no earlier NBA season in the
history or ``players.from_year``: a debut, call-up or mid-season signing is not "absence") or played for more than one team (a trade makes
the earlier team's games look missed; ADR 0006 limitation, ADR 0021 uses single-team players for the same reason).

:class:`ReturnFeatures` turns the panel into seven columns for ``AvailabilityModel(extra=...)``, recency-weighted with the same
``n_lags`` / ``decay`` the availability model uses (weight ``decay**k`` on the season ``k+1`` years before the target; seasons the player
did not appear in carry no weight):

0. ``lead_bar``          sum(w * lead_frac) / sum(w) over the seasons present; 0 when there is no lead block anywhere.
1. ``tail_health_bar``   weighted mean of the shrunk tail health over the *lead* seasons only; ``NEUTRAL_HEALTH`` when there is none.
2. ``tail_len_bar``      sum(w * lead_season * tail_len / L) / sum(w): how much of the season the healthy tail covers; 0 with no lead season.
3. ``return_bar``        sum(w * return_flag) / sum(w).
4. ``lead_last``         last season's ``lead_frac`` (0 if he did not appear last season).
5. ``tail_health_last``  last season's shrunk tail health if it was a lead season, else ``NEUTRAL_HEALTH``.
6. ``return_last``       last season's ``return_flag`` (0 / 1).

The matrix is never NaN and always has :data:`N_COLS` columns (missing values take the conventions above), so the availability model keeps
every training row and the fit and predict widths always agree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from src.contracts import History
from src.models.panel import lag_matrix, season_starts

PANEL_COLS = ("lead_frac", "tail_health_s", "tail_len_frac", "lead_season", "return_flag")
FEATURE_NAMES = ("lead_bar", "tail_health_bar", "tail_len_bar", "return_bar", "lead_last", "tail_health_last", "return_last")
N_COLS = len(FEATURE_NAMES)

LEAD_MIN = 0.10          # a lead block of at least 10% of the schedule makes the season a "lead season"
SHRINK_K = 10.0          # pseudo-games of neutral health added to every tail
NEUTRAL_HEALTH = 0.75    # the shrink prior, the fill value with no lead season, and Task A's health threshold
FLAG_BLOCK = 0.25        # Task A (ADR 0021) primary-cohort definitions
FLAG_MIN_TAIL = 15
FLAG_HEALTH = 0.75


def build_return_panel(history: History) -> pd.DataFrame:
    """One row per (player_id, s) with the lead-block / tail-health profile of the primary team's schedule.

    Only seasons in which the player appears in ``game_logs`` produce a row. Columns: ``player_id, s, L, n_teams, veteran, gp,
    first_idx`` (games missed before the first appearance) and the derived :data:`PANEL_COLS`.
    """
    cols = ["player_id", "s", "L", "n_teams", "veteran", "gp", "first_idx", *PANEL_COLS]
    gl, tg = history.game_logs, history.team_games
    if gl.empty or tg.empty:
        return pd.DataFrame(columns=cols)
    gl = gl[["season", "game_id", "player_id", "team_id"]].drop_duplicates(["player_id", "game_id", "team_id"]).copy()
    gl["s"] = season_starts(gl["season"])
    tg = tg[["season", "game_id", "game_date", "team_id"]].copy()
    tg["s"] = season_starts(tg["season"])
    tg = tg.sort_values(["team_id", "s", "game_date", "game_id"], kind="mergesort")
    tg["idx"] = tg.groupby(["team_id", "s"]).cumcount()
    sched_len = tg.groupby(["team_id", "s"]).size().rename("L").reset_index()

    counts = gl.groupby(["player_id", "s", "team_id"]).size().rename("n").reset_index()
    n_teams = counts.groupby(["player_id", "s"]).size().rename("n_teams")
    primary = (counts.sort_values(["player_id", "s", "n", "team_id"], ascending=[True, True, False, True])
                     .drop_duplicates(["player_id", "s"]).drop(columns="n"))
    mine = gl.merge(primary, on=["player_id", "s", "team_id"]).merge(
        tg[["team_id", "s", "game_id", "idx"]], on=["team_id", "s", "game_id"], how="inner")
    if mine.empty:
        return pd.DataFrame(columns=cols)
    agg = mine.groupby(["player_id", "s", "team_id"]).agg(gp=("idx", "size"), first_idx=("idx", "min")).reset_index()
    agg = agg.merge(sched_len, on=["team_id", "s"], how="left").merge(n_teams.reset_index(), on=["player_id", "s"], how="left")

    frm = pd.to_numeric(history.players.drop_duplicates("player_id").set_index("player_id")["from_year"], errors="coerce")
    first_seen = gl.groupby("player_id")["s"].min()
    f = frm.reindex(agg["player_id"]).to_numpy(dtype="float64")
    g = first_seen.reindex(agg["player_id"]).to_numpy(dtype="float64")
    s = agg["s"].to_numpy(float)
    agg["veteran"] = (np.nan_to_num(f, nan=np.inf) < s) | (np.nan_to_num(g, nan=np.inf) < s)

    L = agg["L"].to_numpy(float)
    gp = agg["gp"].to_numpy(float)
    first = agg["first_idx"].to_numpy(float)
    tail_len = L - first
    usable = agg["veteran"].to_numpy(bool) & (agg["n_teams"].to_numpy() == 1) & (L > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lead_frac = np.where(usable, first / L, 0.0)
        raw_health = np.where(tail_len > 0, gp / tail_len, 0.0)
        tail_len_frac = np.where(usable, tail_len / L, 0.0)
    agg["lead_frac"] = lead_frac
    agg["tail_health_s"] = (gp + SHRINK_K * NEUTRAL_HEALTH) / (tail_len + SHRINK_K)
    agg["lead_season"] = usable & (lead_frac >= LEAD_MIN)
    agg["tail_len_frac"] = tail_len_frac
    agg["return_flag"] = usable & (lead_frac >= FLAG_BLOCK) & (tail_len >= FLAG_MIN_TAIL) & (raw_health >= FLAG_HEALTH)
    for c in ("lead_season", "return_flag"):
        agg[c] = agg[c].astype(float)
    return agg[cols].reset_index(drop=True)


def _weights(lags: np.ndarray, decay: float) -> tuple[np.ndarray, np.ndarray]:
    present = np.isfinite(lags)
    return np.where(present, decay ** np.arange(lags.shape[1]), 0.0), present


@dataclass
class ReturnFeatures:
    """Built once per ``History`` by ``BaselineReturnProjector.fit``; reused for training rows and the target-season frame via
    :meth:`build`. Columns are :data:`FEATURE_NAMES` (see the module docstring)."""

    panel: pd.DataFrame
    n_lags: int
    decay: float
    n_cols: int = field(default=N_COLS, init=False)

    @classmethod
    def fit(cls, history: History, *, n_lags: int, decay: float) -> "ReturnFeatures":
        return cls(build_return_panel(history), n_lags, decay)

    def build(self, pids: np.ndarray, target_s: np.ndarray, age: np.ndarray | None = None) -> np.ndarray:
        """(n, 7) extra-feature matrix for ``AvailabilityModel``, aligned with ``pids`` / ``target_s`` (``age`` is unused; it is
        accepted so the signature matches ``InjuryFeatures.build``)."""
        n = len(pids)
        out = np.zeros((n, N_COLS))
        out[:, [1, 5]] = NEUTRAL_HEALTH
        if not len(self.panel) or n == 0:
            return out
        pidx = self.panel.set_index(["player_id", "s"]).sort_index()
        lags = lag_matrix(pidx, np.asarray(pids, "int64"), np.asarray(target_s, "int64"), self.n_lags, list(PANEL_COLS))
        w, present = _weights(lags["lead_frac"], self.decay)
        wsum = w.sum(axis=1)
        safe = np.maximum(wsum, 1e-12)
        z = lambda a: np.where(present, a, 0.0)  # noqa: E731  (absent seasons carry no weight; NaN -> 0 before the product)
        lead, leadseason = z(lags["lead_frac"]), z(lags["lead_season"])
        has = wsum > 0
        out[:, 0] = np.where(has, (w * lead).sum(axis=1) / safe, 0.0)
        wl = w * leadseason
        wlsum = wl.sum(axis=1)
        out[:, 1] = np.where(wlsum > 0, (wl * z(lags["tail_health_s"])).sum(axis=1) / np.maximum(wlsum, 1e-12), NEUTRAL_HEALTH)
        out[:, 2] = np.where(has, (wl * z(lags["tail_len_frac"])).sum(axis=1) / safe, 0.0)
        out[:, 3] = np.where(has, (w * z(lags["return_flag"])).sum(axis=1) / safe, 0.0)
        out[:, 4] = lead[:, 0]
        out[:, 5] = np.where(leadseason[:, 0] > 0, z(lags["tail_health_s"])[:, 0], NEUTRAL_HEALTH)
        out[:, 6] = z(lags["return_flag"])[:, 0]
        return out


@dataclass
class ConcatFeatures:
    """Stack several feature builders (each with ``build(pids, target_s, age)``) column-wise, in order, for one ``extra`` matrix."""

    parts: Sequence[object]

    @property
    def n_cols(self) -> int:
        return sum(p.n_cols for p in self.parts)

    def build(self, pids: np.ndarray, target_s: np.ndarray, age: np.ndarray) -> np.ndarray:
        return np.column_stack([p.build(pids, target_s, age) for p in self.parts])
