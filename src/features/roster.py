"""Roster-context feature layer: pace, role share, positional crowding (ADR 0010).

Everything here is a pure function of ``History.game_logs``/``History.team_games`` -- both already
sliced to seasons strictly before the target season by ``History.until`` -- so ``build_roster_panel``
is leakage-safe by construction, the same way ``src.models.panel`` and ``src.features.injury`` are.
``RosterFeatures.build`` only ever looks up *lagged* rows (``target_s - 1 .. target_s - n_lags``) via
``src.models.panel.lag_matrix``, never the target season itself, so the fitted per-row minutes
adjustment cannot see anything about the season it is adjusting.

Documented limitation (ADR 0010 "Context"): there is no point-in-time preseason roster/depth-chart
source in this project's data contract, so there is no way to build a genuine forward-looking
"a star arrived/departed this offseason" signal -- that would require knowing, *before* target
season S tips off, who is actually on a team's roster for S, and the only roster information this
contract carries (``game_logs``/``team_games``) describes what already happened *during* a season.
Using season S's own games would be exactly the leakage the ``History.until`` cutoff exists to
prevent. This layer instead uses the most recently *completed* season's pace/role/crowding as a
lagged proxy for the same context in the target season -- team systems and rotations are reasonably
persistent year over year, so this is a real, useful signal, but it is NOT the arrival/departure
signal PLANNING.md originally described, and it will not react to an offseason trade or signing
until a season has actually been played under the new roster. Modeling the true signal is future
work gated on ingesting a dated preseason roster/transactions source (see ADR 0010 D5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.contracts import History
from src.models.panel import lag_matrix, season_starts
from src.models.shrink import wls
from src.value.positions import position_group

# Columns produced by build_roster_panel, keyed on (player_id, s).
ROSTER_COLS = ("pace", "role_share", "pos_crowding")

CROWD_MPG_THRESHOLD = 15.0   # mpg a teammate must clear to count as "crowding" the position
MIN_FIT_ROWS = 60            # fewer usable training rows than this -> zero adjustment (fit fallback)
MPG_SHIFT_CAP = 4.0          # +/- minutes the fitted adjustment is clipped to


def _recency_weighted(lags: np.ndarray, decay: float) -> np.ndarray:
    """Copied verbatim from ``src.features.injury`` (see that module's docstring for why each
    feature-layer module keeps its own copy rather than importing a private helper across
    modules -- independently-owned per the repo's track-ownership convention)."""
    present = np.isfinite(lags)
    v0 = np.where(present, lags, 0.0)
    w = decay ** np.arange(lags.shape[1]) * present
    wsum = w.sum(axis=1)
    return np.where(wsum > 0, (w * v0).sum(axis=1) / np.maximum(wsum, 1e-12), np.nan)


def build_roster_panel(history: History) -> pd.DataFrame:
    """One row per (player_id, s): ``pace``, ``role_share``, ``pos_crowding``.

    * ``pace`` -- the player's *primary* team's (most game_logs rows for that team_id that season,
      same rule ``src.features.injury`` uses) estimated possessions per game that season:
      ``POSS ~= FGA - OREB + TOV + 0.44*FTA``, summed across all of a team's players within each
      game_id, then averaged over the team's games that season.
    * ``role_share`` -- ``player_mpg / (team_min_per_game / 5)``. ``player_mpg`` is the player's own
      minutes per game that season, computed across ALL teams he played for that season (simpler and
      more representative of his actual role than restricting to just the primary team's games when
      he was traded -- documented choice, see module docstring's primary-team caveat for the
      analogous ``pace``/denominator side, which does use the primary team only).
      ``team_min_per_game`` is the *primary* team's average of (sum of all players' ``min`` in a
      game_id) across that team's games that season.
    * ``pos_crowding`` -- computed at player-TEAM-season granularity: for player p's primary
      team+season, the count of *other* player_ids who logged >= ``CROWD_MPG_THRESHOLD`` mpg while
      playing for that same team_id in that same season, at the same ``position_group`` as p.

    Only seasons in which the player actually appears in ``game_logs`` produce a row (consistent
    with ``build_injury_panel``/``build_panel``: a fully-missed season contributes no row).
    """
    gl, tg = history.game_logs, history.team_games
    cols = ["player_id", "s", *ROSTER_COLS]
    if gl.empty or tg.empty:
        return pd.DataFrame(columns=cols)

    # Canonical row order (the tables' unique keys): float sums/means below are order-sensitive in the last
    # bits, and the fit amplifies that, so the panel must not depend on how the caller happened to order rows.
    gl = gl.sort_values(["game_id", "player_id"], kind="mergesort").reset_index(drop=True)
    tg = tg.sort_values(["game_id", "team_id"], kind="mergesort").reset_index(drop=True)
    gl["s"] = season_starts(gl["season"])

    # --- primary team per player-season (same rule as src.features.injury) ---
    counts = gl.groupby(["player_id", "s", "team_id"]).size().rename("n").reset_index()
    primary = (counts.sort_values(["player_id", "s", "n"], ascending=[True, True, False])
                      .drop_duplicates(["player_id", "s"])
                      .rename(columns={"team_id": "primary_team_id"})[["player_id", "s", "primary_team_id"]])

    # --- pace: team-game possessions, averaged to team-season, joined via primary team ---
    game_totals = gl.groupby(["team_id", "s", "game_id"], sort=False).agg(
        fga=("fga", "sum"), oreb=("oreb", "sum"), tov=("tov", "sum"), fta=("fta", "sum"),
    ).reset_index()
    game_totals["poss"] = (game_totals["fga"] - game_totals["oreb"] + game_totals["tov"]
                            + 0.44 * game_totals["fta"])
    team_pace = game_totals.groupby(["team_id", "s"])["poss"].mean().rename("pace").reset_index()

    # --- team total minutes per game (sum of all players' min in a game_id), team-season avg ---
    game_min = gl.groupby(["team_id", "s", "game_id"], sort=False)["min"].sum().rename("min_sum").reset_index()
    team_min_pg = game_min.groupby(["team_id", "s"])["min_sum"].mean().rename("team_min_per_game").reset_index()

    # --- player mpg: across all teams played for that season (role_share numerator) ---
    player_mpg = gl.groupby(["player_id", "s"]).agg(min_sum=("min", "sum"), gp=("game_id", "size")).reset_index()
    player_mpg["player_mpg"] = player_mpg["min_sum"] / player_mpg["gp"]

    # --- pos_crowding: player-team-season mpg + position group ---
    ptm = gl.groupby(["player_id", "team_id", "s"]).agg(min_sum=("min", "sum"), gp=("game_id", "size")).reset_index()
    ptm["team_mpg"] = ptm["min_sum"] / ptm["gp"]
    pos = history.players.drop_duplicates("player_id").set_index("player_id")["position"]
    ptm["pos_group"] = [position_group(p) for p in pos.reindex(ptm["player_id"]).to_numpy()]

    crowd_rows = []
    for (team_id, s), grp in ptm.groupby(["team_id", "s"], sort=False):
        qualifying = grp[grp["team_mpg"] >= CROWD_MPG_THRESHOLD]
        counts_by_group = qualifying.groupby("pos_group")["player_id"].size()
        for _, row in grp.iterrows():
            pg = row["pos_group"]
            n_same = int(counts_by_group.get(pg, 0))
            # exclude self if this player himself qualifies
            if row["team_mpg"] >= CROWD_MPG_THRESHOLD:
                n_same -= 1
            crowd_rows.append((int(row["player_id"]), int(team_id), int(s), n_same))
    crowding = pd.DataFrame(crowd_rows, columns=["player_id", "team_id", "s", "pos_crowding"])

    # --- assemble: one row per (player_id, s), via primary team ---
    team_pace = team_pace.rename(columns={"team_id": "primary_team_id"})
    team_min_pg = team_min_pg.rename(columns={"team_id": "primary_team_id"})
    crowding = crowding.rename(columns={"team_id": "primary_team_id"})

    panel = primary.merge(team_pace, on=["primary_team_id", "s"], how="left")
    panel = panel.merge(team_min_pg, on=["primary_team_id", "s"], how="left")
    panel = panel.merge(player_mpg[["player_id", "s", "player_mpg"]], on=["player_id", "s"], how="left")
    panel = panel.merge(crowding, on=["player_id", "primary_team_id", "s"], how="left")

    panel["role_share"] = panel["player_mpg"] / (panel["team_min_per_game"] / 5.0)
    panel["pos_crowding"] = panel["pos_crowding"].fillna(0.0)

    out = panel[["player_id", "s", "pace", "role_share", "pos_crowding"]].copy()
    out["player_id"] = out["player_id"].astype("int64")
    out["s"] = out["s"].astype("int64")
    return out.reset_index(drop=True)


@dataclass
class RosterFeatures:
    """Built once per ``History`` by ``BaselineRosterProjector.fit``; reused for training rows and
    the target-season frame via :meth:`build`.

    ``build``'s output is a single ``(n,)`` array: the fitted, recency-weighted, clipped additive
    minutes-per-game adjustment for each (pid, target_s) row -- added directly to ``mpg_hat``.
    """

    panel: pd.DataFrame   # player_id, s, pace, role_share, pos_crowding
    n_lags: int
    decay: float
    beta: np.ndarray      # [intercept, pace_coef, role_share_coef, pos_crowding_coef]

    @classmethod
    def fit(cls, history: History, pids: np.ndarray, target_s: np.ndarray,
            mpg_est: np.ndarray, actual_mpg: np.ndarray, weight: np.ndarray,
            *, n_lags: int, decay: float) -> "RosterFeatures":
        """Fit ``residual = actual_mpg - mpg_est ~ intercept + pace_bar + role_share_bar +
        pos_crowding_bar`` by ``wls``, weighted by ``weight`` (that row's actual games played).

        ``pids``/``target_s``/``mpg_est``/``actual_mpg``/``weight`` are the same historical
        training rows, same length, same order (mirrors ``InjuryFeatures.fit``'s
        ``ages``/``panel_pids``/``panel_s`` pattern).

        Falls back to ``beta = zeros`` (a no-op adjustment) when there are fewer than
        ``MIN_FIT_ROWS`` usable rows, or ``wls`` otherwise can't be trusted -- no history means no
        adjustment, never a crash (mirrors ``InjuryFeatures``'s NaN-safe philosophy).
        """
        panel = build_roster_panel(history)
        n_cols = len(ROSTER_COLS) + 1  # intercept + 3 features
        beta = np.zeros(n_cols)
        feats = cls(panel, n_lags, decay, beta)

        pids = np.asarray(pids, "int64")
        target_s = np.asarray(target_s, "int64")
        mpg_est = np.asarray(mpg_est, float)
        actual_mpg = np.asarray(actual_mpg, float)
        weight = np.asarray(weight, float)

        if not len(panel) or len(pids) == 0:
            return feats

        X_context = feats._context_matrix(pids, target_s)
        residual = actual_mpg - mpg_est
        ok = (np.isfinite(residual) & np.isfinite(weight) & (weight > 0)
              & np.isfinite(X_context).all(axis=1))
        if ok.sum() < MIN_FIT_ROWS:
            return feats

        X = np.column_stack([np.ones(ok.sum()), X_context[ok]])
        try:
            fitted_beta = wls(X, residual[ok], weight[ok])
        except (ValueError, np.linalg.LinAlgError):
            return feats
        if not np.all(np.isfinite(fitted_beta)):
            return feats
        return cls(panel, n_lags, decay, fitted_beta)

    def _index(self) -> pd.DataFrame:
        return self.panel.set_index(["player_id", "s"]).sort_index()

    def _context_matrix(self, pids: np.ndarray, target_s: np.ndarray) -> np.ndarray:
        """(n, 3) recency-weighted lagged [pace, role_share, pos_crowding]; NaN where no history."""
        n = len(pids)
        if not len(self.panel):
            return np.full((n, len(ROSTER_COLS)), np.nan)
        pidx = self._index()
        lags = lag_matrix(pidx, np.asarray(pids, "int64"), np.asarray(target_s, "int64"),
                          self.n_lags, list(ROSTER_COLS))
        pace_bar = _recency_weighted(lags["pace"], self.decay)
        role_share_bar = _recency_weighted(lags["role_share"], self.decay)
        pos_crowding_bar = _recency_weighted(lags["pos_crowding"], self.decay)
        return np.column_stack([pace_bar, role_share_bar, pos_crowding_bar])

    def build(self, pids: np.ndarray, target_s: np.ndarray) -> np.ndarray:
        """(n,) additive mpg adjustment, clipped to +/- ``MPG_SHIFT_CAP``. 0.0 (never NaN) for any
        row with no usable context history."""
        pids = np.asarray(pids, "int64")
        target_s = np.asarray(target_s, "int64")
        n = len(pids)
        X_context = self._context_matrix(pids, target_s)
        ok = np.isfinite(X_context).all(axis=1)
        adj = np.zeros(n)
        if ok.any():
            X = np.column_stack([np.ones(ok.sum()), X_context[ok]])
            adj[ok] = X @ self.beta
        return np.clip(adj, -MPG_SHIFT_CAP, MPG_SHIFT_CAP)
