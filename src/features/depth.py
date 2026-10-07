"""Departed-minutes and playoff-closing-role minutes layer (ADR 0035; tests the ADR 0034 flags as a projection input).

Two per-player features, both known before the target season starts:

* ``open_net`` -- rotation minutes per team game that left the player's team before the season (dated ``out`` transactions)
  net of those that arrived (``in``), each weighted by that player's *previous-season* minutes per team game, in tens of
  minutes. Only for players who stayed on their team (``changed_team`` was ADR 0011's feature and a mover's role is not
  this signal) and whose own previous-season minutes are in :data:`MPG_RANGE` (a star is not the one helped).
* ``po_uplift`` -- how far the player's last :data:`RECENT_GAMES` playoff games of the previous season exceeded his
  regular-season minutes, in tens of minutes (``max(0, .)``), needing :data:`MIN_GAMES` playoff games and a regular-season
  average under :data:`ROLE_MAX_MPG`.

``DepthFeatures.fit`` regresses ``actual_mpg - context_free_mpg`` on ``[1, open_net, po_uplift]`` (weighted by games played)
and ``build`` returns the clipped additive minutes shift, the same contract as ``TransactionFeatures``. Same leakage rule as
ADR 0011: a transaction counts only if dated before October 1 of the season's start year; playoffs of season ``s - 1``
precede season ``s``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.contracts import History
from src.models.panel import season_starts
from src.models.shrink import wls

MPG_RANGE = (8.0, 30.0)
MIN_GAMES = 5
RECENT_GAMES = 5
ROLE_MAX_MPG = 28.0
MIN_FIT_ROWS = 60
MPG_SHIFT_CAP = 4.0
FEATURES = ("open_net", "po_uplift")


def _prior_mpg(gl: pd.DataFrame, tg: pd.DataFrame) -> pd.Series:
    """``(player_id, s) -> minutes per team game`` (total minutes / the median team-season length)."""
    n_games = tg.assign(s=season_starts(tg["season"])).groupby(["s", "team_id"]).size().groupby("s").median()
    tot = gl.assign(s=season_starts(gl["season"])).groupby(["player_id", "s"])["min"].sum().reset_index()
    tot["mpg"] = tot["min"] / tot["s"].map(n_games)
    return tot.set_index(["player_id", "s"])["mpg"]


def _primary_team(gl: pd.DataFrame) -> pd.Series:
    g = gl.assign(s=season_starts(gl["season"]))
    n = g.groupby(["player_id", "s", "team_id"]).size().rename("n").reset_index()
    n = n.sort_values(["player_id", "s", "n"], ascending=[True, True, False]).drop_duplicates(["player_id", "s"])
    return n.set_index(["player_id", "s"])["team_id"]


def _playoff_uplift(playoffs: pd.DataFrame, gl: pd.DataFrame) -> pd.Series:
    """``(player_id, s_playoffs) -> max(0, closing playoff mpg - regular-season mpg)`` for qualifying role players."""
    if playoffs is None or playoffs.empty or gl.empty:
        return pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=["player_id", "s"]))
    reg = gl.assign(s=season_starts(gl["season"])).groupby(["player_id", "s"])["min"].mean()
    po = playoffs.assign(s=season_starts(playoffs["season"])).sort_values(["game_date", "game_id"])
    out = {}
    for (pid, s), g in po.groupby(["player_id", "s"]):
        rm = reg.get((pid, s))
        if len(g) < MIN_GAMES or rm is None or not np.isfinite(rm) or rm >= ROLE_MAX_MPG:
            continue
        out[(int(pid), int(s))] = max(0.0, float(g["min"].tail(RECENT_GAMES).mean()) - float(rm))
    idx = pd.MultiIndex.from_tuples(list(out), names=["player_id", "s"])
    return pd.Series(list(out.values()), index=idx, dtype=float)


@dataclass
class DepthContext:
    prior_mpg: pd.Series
    primary_team: pd.Series
    in_team: dict          # (pid, s) -> team_id
    moves: dict            # (team_id, s) -> {"out": [(pid, w)], "in": [(pid, w)]}
    po_uplift: pd.Series

    @classmethod
    def build(cls, history: History, transactions: pd.DataFrame | None, playoffs: pd.DataFrame | None) -> "DepthContext":
        gl, tg = history.game_logs, history.team_games
        if gl.empty or tg.empty:
            e = pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=["player_id", "s"]))
            return cls(e, e, {}, {}, e)
        prior = _prior_mpg(gl, tg)
        primary = _primary_team(gl)
        playoffs = playoffs[playoffs["season"] < history.target_season] if playoffs is not None and len(playoffs) else playoffs
        in_team: dict = {}
        moves: dict = {}
        if transactions is not None and len(transactions):
            t = transactions.copy()
            t["s"] = season_starts(t["season"])
            d = pd.to_datetime(t["txn_date"])
            t = t[d.notna() & (d < pd.to_datetime(t["s"].astype(str) + "-10-01"))].sort_values(["txn_date", "team_id"], kind="mergesort")
            for r in t.itertuples(index=False):
                pid, s, tid = int(r.player_id), int(r.s), int(r.team_id)
                w = prior.get((pid, s - 1))
                w = float(w) if w is not None and np.isfinite(w) else 0.0
                m = moves.setdefault((tid, s), {"out": [], "in": []})
                if r.direction == "in":
                    m["in"].append((pid, w))
                    in_team[(pid, s)] = tid
                elif r.direction == "out":
                    m["out"].append((pid, w))
        return cls(prior, primary, in_team, moves, _playoff_uplift(playoffs, gl))

    def for_players(self, pids: np.ndarray, target_s: np.ndarray) -> np.ndarray:
        """(n, 2) ``[open_net, po_uplift]``, in tens of minutes; 0 where the signal does not apply. Never NaN."""
        pids, target_s = np.asarray(pids, "int64"), np.asarray(target_s, "int64")
        out = np.zeros((len(pids), 2))
        lo, hi = MPG_RANGE
        for i, (pid, s) in enumerate(zip(pids.tolist(), target_s.tolist())):
            m = self.prior_mpg.get((pid, s - 1))
            prev = self.primary_team.get((pid, s - 1))
            if m is None or prev is None or not np.isfinite(m):
                continue
            up = self.po_uplift.get((pid, s - 1))
            if up is not None and np.isfinite(up):
                out[i, 1] = up / 10.0
            inc = self.in_team.get((pid, s))
            if (inc is None or inc == int(prev)) and lo <= m <= hi:
                mv = self.moves.get((int(prev), s))
                if mv:
                    net = sum(w for p, w in mv["out"] if p != pid) - sum(w for p, w in mv["in"] if p != pid)
                    out[i, 0] = net / 10.0
        return out


@dataclass
class DepthFeatures:
    ctx: DepthContext
    beta: np.ndarray       # [intercept, open_net, po_uplift]

    @classmethod
    def fit(cls, history: History, transactions: pd.DataFrame | None, playoffs: pd.DataFrame | None, pids: np.ndarray,
            target_s: np.ndarray, mpg_est: np.ndarray, actual_mpg: np.ndarray, weight: np.ndarray) -> "DepthFeatures":
        ctx = DepthContext.build(history, transactions, playoffs)
        zero = cls(ctx, np.zeros(len(FEATURES) + 1))
        if len(pids) == 0:
            return zero
        X = ctx.for_players(pids, target_s)
        resid = np.asarray(actual_mpg, float) - np.asarray(mpg_est, float)
        w = np.asarray(weight, float)
        ok = np.isfinite(resid) & np.isfinite(w) & (w > 0)
        if ok.sum() < MIN_FIT_ROWS:
            return zero
        try:
            beta = wls(np.column_stack([np.ones(ok.sum()), X[ok]]), resid[ok], w[ok])
        except (ValueError, np.linalg.LinAlgError):
            return zero
        return cls(ctx, beta) if np.all(np.isfinite(beta)) else zero

    def build(self, pids: np.ndarray, target_s: np.ndarray) -> np.ndarray:
        """(n,) additive mpg shift clipped to +/- ``MPG_SHIFT_CAP`` (the fitted intercept is not applied, as in ADR 0011)."""
        return np.clip(self.ctx.for_players(pids, target_s) @ self.beta[1:], -MPG_SHIFT_CAP, MPG_SHIFT_CAP)
