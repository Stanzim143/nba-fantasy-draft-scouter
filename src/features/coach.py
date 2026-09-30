"""Coach features: how each head coach runs a team, and what changes when a team gets a new one (ADR 0020).

Three pieces, all pure functions of frames (tested offline):

* :func:`team_style` -- one row per team-season describing *how the team was run*, from box scores only:
  ``star`` (minutes of the team's top minute-getter, averaged over games), ``top5`` (mean minutes of the five biggest
  minute-getters), ``depth10`` (players per game who play 10+ minutes), ``pace`` (possessions per game, estimated from
  the box score), ``three`` (share of shots from three), ``young`` (share of team minutes played by players aged 23 or
  under: the "develops young players" axis) and ``old`` (share by players 31+).
* :func:`opening_coaches` -- the head coach each team *began* a season with (``src.ingest.wiki_coaches``), the one a
  preseason decision can know.
* :class:`CoachContext` -- for a target season, each team's expected style shift from a **coaching change**: a coach's
  prior style (his own earlier seasons, at any team, never the target season or later) minus the team's previous-season
  style, damped by the number of seasons behind it.

**What the data shows** (real 2016-17..2025-26 team-seasons; numbers in ``docs/coaches.md`` and ADR 0020): style follows the
coach. Year to year the correlation of a team's style is 0.59 to 0.70 when the coach stays and falls to 0.13 to 0.61 when he
changes (pace 0.61 vs 0.14, star minutes 0.66 vs 0.34; a raw persistence contrast, which the transfer test below confirms only for some axes), and when a coach moves to a new team the team's style shift is
predicted by his prior style (28 new-coach team-seasons with a measurable coach history; coefficient on the coach's earlier style in
team style ~ coach's earlier style + team's last style: top-five minutes +0.43 [+0.02, +0.70], rotation depth +0.42 [+0.03, +0.78] and three-point share +0.32 [+0.13, +0.63] follow a coach (bootstrap interval excludes 0 and permutation p < 0.05); star minutes +0.58 has permutation p = 0.001 but a bootstrap interval reaching -0.10, so probably but not certainly; pace (+0.07), youth share (-0.08) and veteran share (+0.10) do not). A team's youth movement is mostly its roster, not its
coach, which contradicts the premise that some coaches "develop" young players in a way that travels.

Nothing here reads the target season's games: :meth:`CoachContext.build` is given a ``History`` (seasons before the target
only) and the coach table, and looks up only the target season's *opening* coach (announced before the season) and earlier
seasons' styles.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.contracts import History, season_start

STYLE_COLS = ("star", "top5", "depth10", "pace", "three", "young", "old")
SHIFT_COLS = ("star", "top5", "depth10", "pace", "three")     # axes a shift is computed for; which follow a coach is in coach_report
YOUNG_AGE = 23
OLD_AGE = 31
MIN_GAMES = 20                     # a team-season needs this many games to count as a style observation
PRIOR_SEASONS_K = 1.0              # shrinkage: a coach with n prior seasons counts n / (n + K) of his measured style
LEAGUE_MIN_MPG = 10.0              # "rotation player" threshold for depth10


def team_style(game_logs: pd.DataFrame, bio: pd.DataFrame) -> pd.DataFrame:
    """Per (season, team): the style columns above plus ``s`` (start year), ``games`` and league-relative ``*_x`` columns
    (each style minus that season's league mean, so 2016 and 2025 are comparable: pace and three-point volume drifted).

    A traded player counts for the team whose game he played in. Games where a team's box score has fewer than five
    players are ignored (bad rows), not repaired.
    """
    if game_logs.empty:
        return pd.DataFrame(columns=["season", "team_id", "s", "games", *STYLE_COLS, *(f"{c}_x" for c in STYLE_COLS)])
    gl = game_logs[["season", "team_id", "game_id", "player_id", "min", "fga", "fta", "oreb", "tov", "fg3a"]].copy()
    ages = bio[["season", "player_id", "age_at_season_start"]]
    gl = gl.merge(ages, on=["season", "player_id"], how="left")
    gl = gl.sort_values(["season", "team_id", "game_id", "min"], ascending=[True, True, True, False], kind="mergesort")
    key = ["season", "team_id", "game_id"]
    gl["rk"] = gl.groupby(key).cumcount() + 1
    per_game = pd.DataFrame({
        "star": gl[gl["rk"] == 1].groupby(key)["min"].sum(),
        "top5": gl[gl["rk"] <= 5].groupby(key)["min"].mean(),
        "depth10": gl[gl["min"] >= LEAGUE_MIN_MPG].groupby(key).size(),
        "n": gl.groupby(key).size(),
    })
    per_game["depth10"] = per_game["depth10"].fillna(0)
    per_game = per_game[per_game["n"] >= 5]
    ts = per_game.groupby(["season", "team_id"]).agg(star=("star", "mean"), top5=("top5", "mean"),
                                                     depth10=("depth10", "mean"), games=("n", "size"))
    tot = gl.groupby(["season", "team_id"]).agg(minutes=("min", "sum"), fga=("fga", "sum"), fta=("fta", "sum"),
                                                oreb=("oreb", "sum"), tov=("tov", "sum"), fg3a=("fg3a", "sum"))
    ts = ts.join(tot)
    ts["pace"] = (ts["fga"] + 0.44 * ts["fta"] - ts["oreb"] + ts["tov"]) / ts["games"]
    ts["three"] = ts["fg3a"] / ts["fga"].replace(0, np.nan)
    young = gl[gl["age_at_season_start"] <= YOUNG_AGE].groupby(["season", "team_id"])["min"].sum()
    old = gl[gl["age_at_season_start"] >= OLD_AGE].groupby(["season", "team_id"])["min"].sum()
    ts["young"] = (young.reindex(ts.index).fillna(0) / ts["minutes"])
    ts["old"] = (old.reindex(ts.index).fillna(0) / ts["minutes"])
    out = ts.reset_index()
    out = out[out["games"] >= MIN_GAMES].copy()
    out["s"] = out["season"].map(season_start)
    for c in STYLE_COLS:
        out[f"{c}_x"] = out[c] - out.groupby("s")[c].transform("mean")
    return out[["season", "team_id", "s", "games", *STYLE_COLS, *(f"{c}_x" for c in STYLE_COLS)]].reset_index(drop=True)


def opening_coaches(team_coaches: pd.DataFrame, *, through_start: int | None = None) -> pd.DataFrame:
    """``season, s, team_id, coach_key, coach_name, single`` (``single``: no mid-season change listed) for each team-season's
    opening coach. ``through_start`` drops seasons that start after that year (leakage guard for the caller)."""
    oc = team_coaches[team_coaches["is_opening"]].copy()
    oc["s"] = oc["season"].map(season_start)
    if through_start is not None:
        oc = oc[oc["s"] <= through_start]
    oc["single"] = oc["n_coaches"] == 1
    return oc[["season", "s", "team_id", "coach_key", "coach_name", "single"]].reset_index(drop=True)


@dataclass
class CoachContext:
    """Built once per ``(History, team_coaches)``; answers, for any team and target season, the coach and the style shift a
    coaching change implies.

    * ``coach_of[(team_id, s)]`` -- opening coach key (all seasons up to and including the target).
    * ``closing_of[(team_id, s)]`` -- the last-listed coach of that season (who the roster last played for).
    * ``prior[(coach_key, s)]`` -- his own earlier seasons' league-relative style ``{col: mean}`` and ``n`` (single-coach
      seasons before ``s`` only).
    * ``prev_style[(team_id, s)]`` -- the team's league-relative style in season ``s - 1``.
    """

    coach_of: dict = field(default_factory=dict)
    closing_of: dict = field(default_factory=dict)
    closing_status: dict = field(default_factory=dict)
    name_of: dict = field(default_factory=dict)
    prior: dict = field(default_factory=dict)
    prev_style: dict = field(default_factory=dict)

    @classmethod
    def build(cls, history: History, team_coaches: pd.DataFrame, *, extra_style: pd.DataFrame | None = None) -> "CoachContext":
        target_s = season_start(history.target_season)
        style = team_style(history.game_logs, history.player_season_bio)
        if extra_style is not None and len(extra_style):
            style = pd.concat([style, extra_style[style.columns]], ignore_index=True)
        oc = opening_coaches(team_coaches, through_start=target_s)
        ctx = cls()
        for r in oc.itertuples(index=False):
            ctx.coach_of[(int(r.team_id), int(r.s))] = r.coach_key
            ctx.name_of[r.coach_key] = r.coach_name
        last = team_coaches.sort_values("seq").drop_duplicates(["season", "team_id"], keep="last")
        for r in last.itertuples(index=False):
            if season_start(r.season) <= target_s:
                ctx.closing_of[(int(r.team_id), season_start(r.season))] = r.coach_key
                ctx.closing_status[(int(r.team_id), season_start(r.season))] = r.status
                ctx.name_of[r.coach_key] = r.coach_name
        style = style[style["s"] < target_s]                      # never the target season or later
        joined = style.merge(oc[["s", "team_id", "coach_key", "single"]], on=["s", "team_id"], how="inner")
        joined = joined[joined["single"]]
        by_coach = {k: g.sort_values("s") for k, g in joined.groupby("coach_key")}
        seasons = sorted(set(oc["s"]) | {target_s})
        for coach, g in by_coach.items():
            for s in seasons:
                prior_rows = g[g["s"] < s]
                if len(prior_rows):
                    ctx.prior[(coach, s)] = {"n": int(len(prior_rows)),
                                             **{c: float(prior_rows[f"{c}_x"].mean()) for c in STYLE_COLS}}
        for r in style.itertuples(index=False):
            ctx.prev_style[(int(r.team_id), int(r.s) + 1)] = {c: float(getattr(r, f"{c}_x")) for c in STYLE_COLS}
        return ctx

    def coach(self, team_id: int, s: int) -> str | None:
        return self.coach_of.get((int(team_id), int(s)))

    def is_new(self, team_id: int, s: int) -> bool:
        """True when the team's opening coach is not the one it finished last season with (needs both seasons in the table)."""
        a, b = self.coach_of.get((int(team_id), int(s))), self.closing_of.get((int(team_id), int(s) - 1))
        return bool(a and b and a != b)

    def shift(self, team_id: int, s: int) -> dict[str, float]:
        """Expected style shift (league-relative minutes, possessions or share) from a coaching change; all zeros when the
        coach is unchanged, or the new coach has no earlier season to measure (a first-time head coach) or the team has no
        previous season."""
        zero = {c: 0.0 for c in SHIFT_COLS}
        if not self.is_new(team_id, s):
            return zero
        coach = self.coach_of[(int(team_id), int(s))]
        prior, prev = self.prior.get((coach, int(s))), self.prev_style.get((int(team_id), int(s)))
        if not prior or not prev:
            return zero
        damp = prior["n"] / (prior["n"] + PRIOR_SEASONS_K)
        return {c: damp * (prior[c] - prev[c]) for c in SHIFT_COLS}

    def debut_coach(self, team_id: int, s: int) -> bool:
        """A new coach with no earlier head-coaching season in the table (style unknown, only the 'new coach' flag applies)."""
        return self.is_new(team_id, s) and (self.coach_of[(int(team_id), int(s))], int(s)) not in self.prior


# --------------------------------------------------------------------------- the projection feature

COACH_COLS = ("top5_x_high", "top5_x_low", "star_x_high", "depth_x_low", "new_coach")
MIN_FIT_ROWS = 60                  # fewer usable training rows than this: no adjustment (same fallback as ADR 0010 / 0011)
MPG_SHIFT_CAP = 4.0                # +/- minutes the fitted adjustment is clipped to (same as ADR 0010 / 0011)
HIGH_MPG_LO, HIGH_MPG_HI = 22.0, 32.0   # last-season minutes at which a player counts 0% .. 100% as a "high-minute" player


def _high(mpg_last: np.ndarray) -> np.ndarray:
    """0 for a player at or below 22 minutes last season, 1 at or above 32, linear between: the weight with which a change in
    how a coach concentrates minutes reaches him."""
    return np.clip((np.asarray(mpg_last, float) - HIGH_MPG_LO) / (HIGH_MPG_HI - HIGH_MPG_LO), 0.0, 1.0)


@dataclass
class CoachFeatures:
    """The minutes adjustment from a coaching change (``baseline_coach``; ADR 0020).

    For each (player, target season): the player's team is resolved the way ``src.features.transactions`` resolves it (a dated
    arrival before October 1, else last season's primary team; for the live season, optionally the current roster snapshot),
    the team's coach and expected style shift come from :class:`CoachContext`, and the player's last-season minutes decide how
    much of a change in *concentration* reaches him. Features (all zero for a team whose coach did not change):

    * ``top5_x_high`` / ``top5_x_low`` -- the expected shift in the top-five players' minutes times the player's high-minute
      weight / one minus it (a coach who concentrates minutes gives more to stars and takes them from the bench);
    * ``star_x_high`` -- the expected shift in the top player's minutes times the high-minute weight;
    * ``depth_x_low`` -- the expected shift in rotation depth (players over 10 minutes) times one minus the weight;
    * ``new_coach`` -- a coaching change happened (1), whatever the coach's history.

    ``beta`` is fitted by weighted least squares on the base model's historical minute residuals (see ``fit``), so nothing
    here is hand-tuned; below ``MIN_FIT_ROWS`` usable rows it is zero and the layer is a no-op.
    """

    ctx: CoachContext
    primary_team: pd.Series
    in_team: pd.Series
    mpg_last: pd.Series
    target_s: int
    live_team: dict = field(default_factory=dict)
    beta: np.ndarray = field(default_factory=lambda: np.zeros(len(COACH_COLS) + 1))

    @classmethod
    def build_context(cls, history: History, team_coaches: pd.DataFrame, transactions: pd.DataFrame | None = None, *,
                      live_team: dict | None = None) -> "CoachFeatures":
        from src.features.transactions import TransactionContext

        ctx = CoachContext.build(history, team_coaches)
        gl = history.game_logs
        if transactions is not None and len(transactions):
            tctx = TransactionContext.build(history, transactions)
            primary, in_team = tctx.primary_team, tctx.in_team
        elif len(gl):
            from src.features.transactions import _empty_indexed_series, _primary_team_panel
            from src.models.panel import season_starts

            g = gl.copy()
            g["s"] = season_starts(g["season"])
            primary = _primary_team_panel(g).set_index(["player_id", "s"])["primary_team_id"]
            in_team = _empty_indexed_series(["player_id", "s"], "int64")
        else:
            from src.features.transactions import _empty_indexed_series

            primary = _empty_indexed_series(["player_id", "s"], "int64")
            in_team = _empty_indexed_series(["player_id", "s"], "int64")
        if len(gl):
            from src.models.panel import season_starts

            g = gl[["player_id", "season", "min"]].copy()
            g["s"] = season_starts(g["season"])
            mpg_last = g.groupby(["player_id", "s"])["min"].mean()
        else:
            mpg_last = pd.Series(dtype="float64", index=pd.MultiIndex.from_tuples([], names=["player_id", "s"]))
        return cls(ctx, primary, in_team, mpg_last, season_start(history.target_season), dict(live_team or {}))

    # ------------------------------------------------------------------ per-row features
    def resolve_team(self, pid: int, s: int) -> int | None:
        if s == self.target_s and pid in self.live_team:
            return int(self.live_team[pid])
        arrived = self.in_team.get((pid, s))
        if arrived is not None and pd.notna(arrived):
            return int(arrived)
        prev = self.primary_team.get((pid, s - 1))
        return int(prev) if prev is not None and pd.notna(prev) else None

    def features(self, pids: np.ndarray, target_s: np.ndarray) -> np.ndarray:
        """``(n, len(COACH_COLS))``. Never NaN: a player with no team, no last season or an unchanged coach gets zeros."""
        pids = np.asarray(pids, "int64")
        target_s = np.asarray(target_s, "int64")
        out = np.zeros((len(pids), len(COACH_COLS)))
        mpg = np.array([self.mpg_last.get((int(p), int(s) - 1), np.nan) for p, s in zip(pids, target_s)], float)
        high = np.where(np.isfinite(mpg), _high(np.nan_to_num(mpg)), 0.0)
        have_mpg = np.isfinite(mpg)
        for i, (pid, s) in enumerate(zip(pids, target_s)):
            team = self.resolve_team(int(pid), int(s))
            if team is None or not self.ctx.is_new(team, int(s)):
                continue
            sh = self.ctx.shift(team, int(s))
            h = float(high[i])
            if have_mpg[i]:
                out[i, 0] = sh["top5"] * h
                out[i, 1] = sh["top5"] * (1.0 - h)
                out[i, 2] = sh["star"] * h
                out[i, 3] = sh["depth10"] * (1.0 - h)
            out[i, 4] = 1.0
        return out

    # ------------------------------------------------------------------ fit / build
    def fit(self, pids: np.ndarray, target_s: np.ndarray, mpg_est: np.ndarray, actual_mpg: np.ndarray,
            weight: np.ndarray) -> "CoachFeatures":
        """``actual_mpg - mpg_est ~ 1 + features`` by weighted least squares (weight = games played); zero beta when there are
        too few usable rows or the solve fails. The rows are the same historical training rows every layer uses."""
        from src.models.shrink import wls

        self.beta = np.zeros(len(COACH_COLS) + 1)
        pids = np.asarray(pids, "int64")
        target_s = np.asarray(target_s, "int64")
        resid = np.asarray(actual_mpg, float) - np.asarray(mpg_est, float)
        w = np.asarray(weight, float)
        X_raw = self.features(pids, target_s)
        ok = np.isfinite(resid) & np.isfinite(w) & (w > 0) & np.isfinite(X_raw).all(axis=1)
        if ok.sum() < MIN_FIT_ROWS:
            return self
        try:
            beta = wls(np.column_stack([np.ones(ok.sum()), X_raw[ok]]), resid[ok], w[ok])
        except (ValueError, np.linalg.LinAlgError):
            return self
        if np.all(np.isfinite(beta)):
            self.beta = beta
        return self

    def build(self, pids: np.ndarray, target_s: np.ndarray) -> np.ndarray:
        """``(n,)`` additive minutes-per-game adjustment, clipped to +/- ``MPG_SHIFT_CAP``. The intercept is fitted (so the
        coefficients measure what a coaching change adds *beyond* the base model's average residual) but never applied: a
        uniform correction belongs to the base model, and applying it here would make the layer a bias fix that only touches
        the teams with a new coach."""
        adj = self.features(pids, target_s) @ self.beta[1:]
        return np.clip(adj, -MPG_SHIFT_CAP, MPG_SHIFT_CAP)
