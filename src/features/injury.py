"""Injury/availability history features derived purely from ``History`` (ADR 0006).

Everything here is a pure function of ``History.game_logs`` and ``History.team_games`` -- both
already sliced to seasons strictly before the target season by ``History.until`` -- so it is
leakage-safe by construction, the same way ``src.models.panel`` is. No new raw data source is
needed: "games missed" is already derivable as ``team_games`` minus ``game_logs`` (ADR 0001,
rule 5), and the *order* teams played those games in (also in ``team_games``) is enough to find
absence streaks, which a season-total games-missed count cannot distinguish.

Two shortcomings of the games-missed total the baseline's ``AvailabilityModel`` already uses
(``f = gp / season_len``, recency-weighted into ``f_bar``):

* It cannot tell a player who missed 20 games in one 3-week stretch (one injury) from a player who
  missed 20 games in ones and twos (load management / minor knocks) -- these carry different
  forward-looking risk. ``longest_streak`` / ``n_streaks`` (below) capture the shape.
* It has no notion of whether a player's absence rate is normal *for his age* -- ``age_residual``
  compares a player's own recency-weighted absence rate against the league's empirical age curve.

Team attribution note: a player's "season schedule" for streak purposes is his *primary* team for
that season (the team he played the most games for). A player traded mid-season will show
misattributed absences around the trade (games for the other team count as "missed"); this is a
documented approximation (see docs/adr/0006-injury-layer.md), not a correctness bug -- there is no
point-in-time-correct roster-tenure table in the current contract to do better with.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.contracts import History
from src.models.panel import lag_matrix, season_starts

# Columns produced by build_injury_panel, keyed on (player_id, s).
STREAK_COLS = ("missed_frac", "longest_streak_frac", "n_streaks")

LONG_ABSENCE_THRESHOLD = 0.15  # >= 15% of a season missed in one streak (~12 games of 82)


def _streak_stats(missed: np.ndarray) -> tuple[int, int]:
    """(longest run of consecutive True, number of runs) in a boolean array."""
    if missed.size == 0 or not missed.any():
        return 0, 0
    padded = np.concatenate(([False], missed, [False])).astype(np.int8)
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    lengths = ends - starts
    return int(lengths.max()), int(len(lengths))


def build_injury_panel(history: History) -> pd.DataFrame:
    """One row per (player_id, s) with absence-streak features, ``s`` = season start year.

    Only seasons in which the player actually appears in ``game_logs`` produce a row (a player
    who misses an *entire* season contributes no row here either -- consistent with how
    ``src.models.panel.build_panel`` treats a fully-missed season: it "simply contributes zero
    exposure", never a fabricated streak of the full schedule).
    """
    gl, tg = history.game_logs, history.team_games
    cols = ["player_id", "s", "season_len", *STREAK_COLS, "gp"]
    if gl.empty or tg.empty:
        return pd.DataFrame(columns=cols)

    gl = gl.copy()
    gl["s"] = season_starts(gl["season"])
    tg = tg.copy()
    tg["s"] = season_starts(tg["season"])

    # Primary team per player-season: the team_id the player has the most game_logs rows for.
    counts = gl.groupby(["player_id", "s", "team_id"]).size().rename("n").reset_index()
    primary = (counts.sort_values(["player_id", "s", "n"], ascending=[True, True, False])
                      .drop_duplicates(["player_id", "s"])
                      .rename(columns={"team_id": "primary_team_id"})[["player_id", "s", "primary_team_id"]])

    played_ids = gl.groupby(["player_id", "s"])["game_id"].agg(frozenset)
    tg_sorted = tg.sort_values(["team_id", "s", "game_date", "game_id"], kind="mergesort")

    rows = []
    for (team_id, s), sched in tg_sorted.groupby(["team_id", "s"], sort=False):
        team_players = primary[(primary["primary_team_id"] == team_id) & (primary["s"] == s)]
        if team_players.empty:
            continue
        game_ids = sched["game_id"].to_numpy()
        season_len = len(game_ids)
        for pid in team_players["player_id"]:
            present = played_ids.get((pid, s), frozenset())
            missed = ~np.isin(game_ids, list(present))
            gp = int(len(game_ids) - missed.sum())
            longest, n_streaks = _streak_stats(missed)
            rows.append((int(pid), int(s), season_len,
                        float(missed.sum()) / season_len if season_len else 0.0,
                        float(longest) / season_len if season_len else 0.0,
                        n_streaks, gp))
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=["player_id", "s", "season_len", "missed_frac",
                                       "longest_streak_frac", "n_streaks", "gp"])


def fit_age_absence_curve(panel: pd.DataFrame, min_rows: int = 60) -> tuple[float, float, float]:
    """Weighted least-squares ``missed_frac ~ a + b*age + c*age^2`` over the whole injury panel.

    Weighted by ``season_len`` (more games = more reliable an estimate of that season's absence
    rate). Returns ``(a, b, c)``; falls back to a flat curve at the panel's mean when there is too
    little data to trust a quadratic fit.
    """
    if len(panel) < min_rows or "age" not in panel.columns:
        mean = float(panel["missed_frac"].mean()) if len(panel) else 0.15
        return (mean, 0.0, 0.0)
    age = panel["age"].to_numpy(float)
    y = panel["missed_frac"].to_numpy(float)
    w = panel["season_len"].to_numpy(float)
    ok = np.isfinite(age) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if ok.sum() < min_rows:
        mean = float(np.average(y[ok], weights=w[ok])) if ok.any() else 0.15
        return (mean, 0.0, 0.0)
    age, y, w = age[ok], y[ok], w[ok]
    X = np.column_stack([np.ones_like(age), age, age ** 2])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return (float(coef[0]), float(coef[1]), float(coef[2]))


def _recency_weighted(lags: np.ndarray, decay: float) -> np.ndarray:
    present = np.isfinite(lags)
    v0 = np.where(present, lags, 0.0)
    w = decay ** np.arange(lags.shape[1]) * present
    wsum = w.sum(axis=1)
    return np.where(wsum > 0, (w * v0).sum(axis=1) / np.maximum(wsum, 1e-12), np.nan)


@dataclass
class InjuryFeatures:
    """Built once per ``History`` by ``BaselineInjuryProjector.fit``; reused for training rows and
    the target-season frame via :meth:`build`.

    Columns of :meth:`build`'s output, in order:

    0. ``streak_bar``     recency-weighted longest-absence-streak fraction of the schedule
    1. ``freq_bar``       recency-weighted number of distinct absence streaks (frequency, not size)
    2. ``long_absence``   1.0 if last season's longest streak was >= ``LONG_ABSENCE_THRESHOLD``
                          of the schedule, else 0.0 ("coming off a long absence last season")
    3. ``age_residual``   recency-weighted absence rate minus the league's expected absence rate
                          for the player's age (positive = missing more than age-typical)
    """

    panel: pd.DataFrame              # player_id, s, season_len, missed_frac, longest_streak_frac, n_streaks
    n_lags: int
    decay: float
    age_curve: tuple[float, float, float]
    n_cols: int = field(default=4, init=False)

    @classmethod
    def fit(cls, history: History, ages: np.ndarray, panel_pids: np.ndarray, panel_s: np.ndarray,
            *, n_lags: int, decay: float) -> "InjuryFeatures":
        """``ages``/``panel_pids``/``panel_s`` are the baseline panel's own age/id/season columns
        (same rows, same order) so the age curve is fit on age values already computed by the
        baseline -- no separate age lookup needed here."""
        panel = build_injury_panel(history)
        if len(panel) and len(panel_pids):
            age_by_key = pd.Series(ages, index=pd.MultiIndex.from_arrays([panel_pids, panel_s]))
            panel = panel.copy()
            panel["age"] = age_by_key.reindex(
                pd.MultiIndex.from_arrays([panel["player_id"], panel["s"]])).to_numpy()
        else:
            panel = panel.assign(age=np.nan) if len(panel) else panel
        curve = fit_age_absence_curve(panel)
        return cls(panel, n_lags, decay, curve)

    def _index(self) -> pd.DataFrame:
        return self.panel.set_index(["player_id", "s"]).sort_index()

    def build(self, pids: np.ndarray, target_s: np.ndarray, age: np.ndarray) -> np.ndarray:
        """(n, 4) extra-feature matrix for ``AvailabilityModel``, aligned with ``pids``/``target_s``.

        Columns: ``streak_bar, freq_bar, long_absence, age_residual`` (see the class docstring).
        """
        n = len(pids)
        if not len(self.panel):
            return np.full((n, self.n_cols), np.nan)
        pidx = self._index()
        lags = lag_matrix(pidx, np.asarray(pids, "int64"), np.asarray(target_s, "int64"),
                          self.n_lags, list(STREAK_COLS))
        streak_bar = _recency_weighted(lags["longest_streak_frac"], self.decay)
        freq_bar = _recency_weighted(lags["n_streaks"], self.decay)
        streak_last = np.where(np.isfinite(lags["longest_streak_frac"][:, 0]),
                               lags["longest_streak_frac"][:, 0], 0.0)
        long_absence = (streak_last >= LONG_ABSENCE_THRESHOLD).astype(float)
        missed_bar = _recency_weighted(lags["missed_frac"], self.decay)
        a, b, c = self.age_curve
        age = np.asarray(age, float)
        expected = a + b * age + c * age ** 2
        age_residual = np.where(np.isfinite(missed_bar) & np.isfinite(expected), missed_bar - expected, np.nan)
        return np.column_stack([streak_bar, freq_bar, long_absence, age_residual])
