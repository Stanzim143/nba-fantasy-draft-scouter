"""Roster-transactions feature layer: ``changed_team`` and ``team_departures_lost`` (ADR 0011).

Unlike ``src.features.roster`` (ADR 0010), which had no dated source and so could only use *lagged*
team-context from completed seasons, this layer is fed a real, dated source
(``src.ingest.wiki_transactions.read_team_transactions``, frozen schema: ``season``, ``team_id``,
``player_id``, ``direction`` (``"in"``/``"out"``), ``source_kind``, ``txn_date`` (nullable)) and uses
each season's OWN transactions directly, the same way a rookie's ``draft_year == target`` is used
directly rather than lagged: both are facts dated before the season, not facts *about* the season's
outcome.

**Leakage cutoff.** A transaction only counts if ``txn_date`` is not null and falls before October 1
of the season's start year (``season_start(season)``). This single, fixed rule is used identically
for historical training rows and the real (as-yet-unplayed) target season -- see ADR 0011 D1/D4 for
why a uniform rule, rather than each season's true (but only-sometimes-knowable) opening date, is the
right choice: the target season's real opening date is never available (``History.until`` excludes
its games entirely), so using two different cutoff rules would mismatch training and prediction.

This module is an independent SIBLING of ``src.features.roster``, not a subclass or extension of it:
it does not import ``RosterFeatures``, but does independently reuse the ``role_share`` *concept*
(recomputed here from ``History.game_logs``/``History.team_games``, duplicated rather than imported,
matching this project's per-module self-containment convention -- see ``src.features.roster``'s and
``src.features.injury``'s own docstrings for the same convention applied to ``_recency_weighted`` and
the primary-team rule).

Two pieces, for the two different callers:

* :func:`build_transactions_panel` -- one row per (player_id, s) for player-seasons that actually
  appear in ``History.game_logs`` (the shape a training panel needs).
* :class:`TransactionContext` -- the underlying, reusable index (team resolution + departure
  weights, keyed by (team_id, season)), queryable for an *arbitrary* set of (pids, target_s) via
  :meth:`TransactionContext.for_players` -- this is what lets the same machinery serve the real
  target season, whose players have not played yet and so have no row of their own to look up.
* :class:`TransactionFeatures` -- ``TransactionContext`` plus one small fitted WLS regression (same
  shape as ``RosterFeatures``, see below for why fitting was chosen over a fixed coefficient).

**Fitted vs. fixed adjustment.** This project's stated principle is "everything estimated from the
history, not hand-tuned" (see ``BaselineConfig``'s search grids and ADR 0006/0010's own fitted
adjustments). Although ``changed_team``/``team_departures_lost`` are themselves direct, deterministic
features (no WLS involved in computing them), the mapping from those two raw features to an additive
minutes adjustment still needs a coefficient, and picking that by hand would be exactly the kind of
hand-tuning the rest of the model avoids. ``TransactionFeatures.fit`` therefore mirrors
``RosterFeatures.fit`` exactly: one WLS regression of ``actual_mpg - context_free_mpg_estimate`` on
``[changed_team, team_departures_lost]``, weighted by each row's actual games played, with the same
zero-fallback-below-``MIN_FIT_ROWS`` philosophy and the same +/-``MPG_SHIFT_CAP`` clip (reusing
ADR 0010's numeric values, 60 rows / 4.0 minutes -- there is no better-justified alternative for this
layer, and keeping the same magnitudes makes the two layers' adjustments directly comparable in the
ablation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.contracts import History
from src.models.panel import season_starts
from src.models.shrink import wls

TRANSACTIONS_COLS = ("changed_team", "team_departures_lost")

MIN_FIT_ROWS = 60          # fewer usable training rows than this -> zero adjustment (fit fallback)
MPG_SHIFT_CAP = 4.0        # +/- minutes the fitted adjustment is clipped to


def _primary_team_panel(gl: pd.DataFrame) -> pd.DataFrame:
    """player_id, s -> primary_team_id (the team_id with the most game_logs rows that season).

    Duplicated from ``src.features.roster``/``src.features.injury`` rather than imported -- same
    per-module self-containment convention those modules already document.
    """
    counts = gl.groupby(["player_id", "s", "team_id"]).size().rename("n").reset_index()
    return (counts.sort_values(["player_id", "s", "n"], ascending=[True, True, False])
                  .drop_duplicates(["player_id", "s"])
                  .rename(columns={"team_id": "primary_team_id"})[["player_id", "s", "primary_team_id"]])


def _role_share_panel(gl: pd.DataFrame) -> pd.DataFrame:
    """player_id, s -> role_share, the exact ``player_mpg / (team_min_per_game / 5)`` formula from
    ``src.features.roster.build_roster_panel`` (duplicated here, not imported -- see module docstring)."""
    primary = _primary_team_panel(gl)
    game_min = gl.groupby(["team_id", "s", "game_id"], sort=False)["min"].sum().rename("min_sum").reset_index()
    team_min_pg = (game_min.groupby(["team_id", "s"])["min_sum"].mean()
                   .rename("team_min_per_game").reset_index()
                   .rename(columns={"team_id": "primary_team_id"}))
    player_mpg = gl.groupby(["player_id", "s"]).agg(min_sum=("min", "sum"), gp=("game_id", "size")).reset_index()
    player_mpg["player_mpg"] = player_mpg["min_sum"] / player_mpg["gp"]

    panel = primary.merge(team_min_pg, on=["primary_team_id", "s"], how="left")
    panel = panel.merge(player_mpg[["player_id", "s", "player_mpg"]], on=["player_id", "s"], how="left")
    panel["role_share"] = panel["player_mpg"] / (panel["team_min_per_game"] / 5.0)
    return panel[["player_id", "s", "role_share"]]


def _empty_indexed_series(names: list[str], dtype) -> pd.Series:
    return pd.Series(dtype=dtype, index=pd.MultiIndex.from_tuples([], names=names))


@dataclass
class TransactionContext:
    """Built once per ``(History, transactions)`` pair; queryable for any (pids, target_s).

    * ``primary_team`` -- ``(player_id, s) -> primary_team_id``, from ``History.game_logs``.
    * ``role_share`` -- ``(player_id, s) -> role_share``, from ``History.game_logs``.
    * ``in_team`` -- ``(player_id, s) -> team_id``, the resolved incoming team from a dated
      ``direction == "in"`` transaction before that season's October-1 cutoff (the latest such
      transaction if more than one, tie-broken by team_id for determinism).
    * ``departures`` -- ``(team_id, s) -> [(departing_player_id, weight), ...]``, weight being that
      departing player's own *previous*-season (``s - 1``) ``role_share`` (0.0 if unresolvable).
    """

    primary_team: pd.Series
    role_share: pd.Series
    in_team: pd.Series
    departures: dict

    @classmethod
    def build(cls, history: History, transactions: pd.DataFrame) -> "TransactionContext":
        gl = history.game_logs
        if gl.empty:
            primary_team = _empty_indexed_series(["player_id", "s"], "int64")
            role_share = _empty_indexed_series(["player_id", "s"], "float64")
        else:
            gl = gl.copy()
            gl["s"] = season_starts(gl["season"])
            primary_team = _primary_team_panel(gl).set_index(["player_id", "s"])["primary_team_id"]
            role_share = _role_share_panel(gl).set_index(["player_id", "s"])["role_share"]

        txn = transactions.copy() if transactions is not None else pd.DataFrame(
            columns=["season", "team_id", "player_id", "direction", "source_kind", "txn_date"])

        if len(txn):
            txn["s"] = season_starts(txn["season"])
            cutoff = pd.to_datetime(txn["s"].astype(str) + "-10-01")
            txn_date = pd.to_datetime(txn["txn_date"])
            before_cutoff = txn_date.notna() & (txn_date < cutoff)
            txn_valid = txn[before_cutoff]
        else:
            txn_valid = txn.assign(s=pd.Series(dtype="int64"))

        txn_in = txn_valid[txn_valid["direction"] == "in"]
        txn_out = txn_valid[txn_valid["direction"] == "out"]

        if len(txn_in):
            txn_in_sorted = txn_in.sort_values(["player_id", "s", "txn_date", "team_id"], kind="mergesort")
            in_team = (txn_in_sorted.drop_duplicates(["player_id", "s"], keep="last")
                       .set_index(["player_id", "s"])["team_id"])
        else:
            in_team = _empty_indexed_series(["player_id", "s"], "int64")

        departures: dict = {}
        for row in txn_out.itertuples(index=False):
            team_id, s, pid = int(row.team_id), int(row.s), int(row.player_id)
            w = role_share.get((pid, s - 1))
            w = float(w) if w is not None and pd.notna(w) else 0.0
            departures.setdefault((team_id, s), []).append((pid, w))

        return cls(primary_team, role_share, in_team, departures)

    # ---------------------------------------------------------------- per-player resolution

    def _resolve(self, pid: int, s: int) -> tuple[int | None, float]:
        """(resolved_team_id or None, changed_team) for one (player_id, s)."""
        prev = self.primary_team.get((pid, s - 1))
        prev_team = int(prev) if prev is not None and pd.notna(prev) else None
        inc = self.in_team.get((pid, s))
        in_team = int(inc) if inc is not None and pd.notna(inc) else None

        if in_team is not None:
            resolved = in_team
        elif prev_team is not None:
            resolved = prev_team
        else:
            return None, 0.0

        if prev_team is None:
            changed = 1.0 if in_team is not None else 0.0
        else:
            changed = 1.0 if resolved != prev_team else 0.0
        return resolved, changed

    def _departures_lost(self, team_id: int, s: int, exclude_pid: int | None) -> float:
        total = 0.0
        for dep_pid, w in self.departures.get((team_id, s), ()):
            if exclude_pid is not None and dep_pid == exclude_pid:
                continue
            total += w
        return total

    def for_players(self, pids: np.ndarray, target_s: np.ndarray) -> np.ndarray:
        """(n, 2) ``[changed_team, team_departures_lost]``. Never NaN -- 0.0 is the safe default
        for a player with no team assignment at all (no previous season, no recorded transaction)."""
        pids = np.asarray(pids, "int64")
        target_s = np.asarray(target_s, "int64")
        n = len(pids)
        out = np.zeros((n, 2), dtype="float64")
        for i in range(n):
            pid, s = int(pids[i]), int(target_s[i])
            resolved, changed = self._resolve(pid, s)
            out[i, 0] = changed
            out[i, 1] = self._departures_lost(resolved, s, pid) if resolved is not None else 0.0
        return out


def build_transactions_panel(history: History, transactions: pd.DataFrame) -> pd.DataFrame:
    """One row per (player_id, s) present in ``History.game_logs``: ``changed_team``,
    ``team_departures_lost`` (see module docstring / ADR 0011 D4 for the exact rules).

    Only player-seasons that actually appear in ``History.game_logs`` produce a row here (consistent
    with ``build_roster_panel``/``build_injury_panel``: a fully-missed season contributes no row) --
    this is the shape a *training* panel needs. The real target season's players (who have not
    played yet) are served directly by ``TransactionContext.for_players``, not by this function.
    """
    gl = history.game_logs
    cols = ["player_id", "s", *TRANSACTIONS_COLS]
    if gl.empty:
        return pd.DataFrame(columns=cols)

    gl = gl.copy()
    gl["s"] = season_starts(gl["season"])
    player_seasons = gl[["player_id", "s"]].drop_duplicates().reset_index(drop=True)

    ctx = TransactionContext.build(history, transactions)
    feats = ctx.for_players(player_seasons["player_id"].to_numpy(), player_seasons["s"].to_numpy())

    out = player_seasons.copy()
    out["changed_team"] = feats[:, 0]
    out["team_departures_lost"] = feats[:, 1]
    out["player_id"] = out["player_id"].astype("int64")
    out["s"] = out["s"].astype("int64")
    return out[cols].reset_index(drop=True)


@dataclass
class TransactionFeatures:
    """Built once per ``(History, transactions)`` by ``BaselineTransactionsProjector.fit``; reused
    for training rows and the target-season frame via :meth:`build`.

    ``build``'s output is a single ``(n,)`` array: the fitted, clipped additive minutes-per-game
    adjustment for each (pid, target_s) row -- added directly to ``mpg_hat``, exactly like
    ``RosterFeatures.build`` (see module docstring for why this is fitted, not a fixed coefficient,
    and why there is no lagging/recency-weighting here unlike ``RosterFeatures``: these features are
    already dated directly to the target season, not proxies from an earlier one).
    """

    ctx: TransactionContext
    beta: np.ndarray   # [intercept, changed_team_coef, team_departures_lost_coef]
    n_cols: int = field(default=len(TRANSACTIONS_COLS), init=False)

    @classmethod
    def fit(cls, history: History, transactions: pd.DataFrame, pids: np.ndarray, target_s: np.ndarray,
            mpg_est: np.ndarray, actual_mpg: np.ndarray, weight: np.ndarray) -> "TransactionFeatures":
        """Fit ``residual = actual_mpg - mpg_est ~ intercept + changed_team + team_departures_lost``
        by ``wls``, weighted by ``weight`` (that row's actual games played).

        ``pids``/``target_s``/``mpg_est``/``actual_mpg``/``weight`` are the same historical training
        rows, same length, same order (mirrors ``RosterFeatures.fit``'s pattern).

        Falls back to ``beta = zeros`` (a no-op adjustment) when there are fewer than
        ``MIN_FIT_ROWS`` usable rows, or ``wls`` otherwise can't be trusted -- no history means no
        adjustment, never a crash.
        """
        ctx = TransactionContext.build(history, transactions)
        beta = np.zeros(len(TRANSACTIONS_COLS) + 1)
        feats = cls(ctx, beta)

        pids = np.asarray(pids, "int64")
        target_s = np.asarray(target_s, "int64")
        mpg_est = np.asarray(mpg_est, float)
        actual_mpg = np.asarray(actual_mpg, float)
        weight = np.asarray(weight, float)

        if len(pids) == 0:
            return feats

        X_raw = ctx.for_players(pids, target_s)
        residual = actual_mpg - mpg_est
        ok = (np.isfinite(residual) & np.isfinite(weight) & (weight > 0)
              & np.isfinite(X_raw).all(axis=1))
        if ok.sum() < MIN_FIT_ROWS:
            return feats

        X = np.column_stack([np.ones(ok.sum()), X_raw[ok]])
        try:
            fitted_beta = wls(X, residual[ok], weight[ok])
        except (ValueError, np.linalg.LinAlgError):
            return feats
        if not np.all(np.isfinite(fitted_beta)):
            return feats
        return cls(ctx, fitted_beta)

    def build(self, pids: np.ndarray, target_s: np.ndarray) -> np.ndarray:
        """(n,) additive mpg adjustment, clipped to +/- ``MPG_SHIFT_CAP``. 0.0 (never NaN) for any
        row with no usable transaction context."""
        pids = np.asarray(pids, "int64")
        target_s = np.asarray(target_s, "int64")
        n = len(pids)
        X_raw = self.ctx.for_players(pids, target_s)
        X = np.column_stack([np.ones(n), X_raw])
        adj = X @ self.beta
        return np.clip(adj, -MPG_SHIFT_CAP, MPG_SHIFT_CAP)
