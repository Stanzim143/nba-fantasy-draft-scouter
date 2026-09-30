"""Offseason feature layer: what Summer League and preseason tell us beyond last season's box scores (ADR 0012).

The question this layer answers, per player: *given the baseline projection, does what he did this
summer and preseason move the number, and by how much?* It is a **stacked residual model**, not a
rewrite of the baseline:

1. For every earlier season ``s`` in the history, the base projector is re-run on ``History.until(s)``
   (walk-forward, so it sees nothing from ``s`` on) and its ``proj_fppg`` is compared with what the
   player actually averaged in ``s``. The gap is the residual the layer must explain.
2. Each ``(player, s)`` residual is regressed on features built from the Summer League and preseason
   that preceded ``s`` (never from ``s`` itself): production per 36 minutes relative to that summer's
   cohort, shrunk by minutes played, and the minutes the player was given; each also interacted with
   youth and rookie status, because that is where the evidence should matter.
3. The fitted adjustment is applied to the target season's projection as a per-player multiplier on
   every counting stat, so box-score identities keep holding and fantasy points scale exactly.

Why a residual on top of the baseline instead of new inputs to it: the baseline already prices
age, draft slot, minutes history and regression to the mean. The only honest claim an offseason signal
can make is that it explains *what is left over*; and stacking lets the ablation answer that claim
directly (paired against the unadjusted baseline, same seasons, same players).

Leakage. ``History.until`` keeps a row only if its ``season`` tag is before the target season, and the
offseason tables tag July 2026 as ``"2025-26"`` (the season it follows), so events that happened before
the target season starts are visible and nothing else is. :func:`slice_offseason` re-applies the same
rule for callers that read the parquet directly. Preseason games can additionally be cut at ``as_of``
for live use (a draft held before the last preseason game must not see it).

Honesty gate. Hyperparameters (the minutes-shrinkage constant ``K`` and the ridge strength) are chosen
by leave-one-season-out cross-validation on the training rows, and the adjustment is switched **off**
unless that cross-validation beats "no adjustment" by a margin. A layer with no demonstrated signal
returns the baseline unchanged and says so in :attr:`OffseasonFit.diagnostics`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from src.contracts import season_start
from src.models.shrink import wls
from src.value.frame import fantasy_points_frame

SUMMER_LEAGUE = "summer_league"
PRESEASON = "preseason"

MIN_COHORT_MINUTES = 40.0       # a summer's cohort is the players with at least this many minutes
Z_CLIP = 3.0                    # cohort z-scores are clipped so one weird line cannot dominate
K_GRID = (60.0, 120.0, 240.0, 480.0)         # minutes of evidence at which a player's own line gets half weight
RIDGE_GRID = (3.0, 30.0, 300.0, 3000.0)   # ridge strength in pseudo-rows (wls takes it relative to the data scale)
MIN_FIT_ROWS = 150              # fewer usable training rows than this -> no adjustment
MIN_TRAIN_GP = 10               # a training residual needs at least this many actual games
MIN_CV_GAIN = 0.002             # required relative CV improvement over "no adjustment" (0.2%)
FACTOR_LO, FACTOR_HI = 0.6, 1.6  # the fitted per-player multiplier is clipped to this range
WEIGHT_GP_CAP = 60.0

# Standard box-score components, each z-scored against the event's cohort (see ``_cohort_z``):
# scoring efficiency, shot usage, playmaking, rebounding and defensive events.
COMPONENTS = ("ts", "usg", "ast", "reb", "stk")
_EVENT_STATS = ["gp", "minutes", "mpg", "gp_share", "fp36", "z", *[f"{c}_z" for c in COMPONENTS]]


def event_columns(components: bool = False) -> list[str]:
    """Columns of the per-event table the design matrix reads (``sl_*`` and ``pre_*``)."""
    base = ["minutes", "z", "mpg"]
    extra = [f"{c}_z" for c in COMPONENTS] if components else []
    return [f"{p}_{c}" for p in ("sl", "pre") for c in (*base, *extra)]


def feature_names(components: bool = False) -> tuple[str, ...]:
    names: list[str] = []
    for p in ("sl", "pre"):
        names += [f"has_{p}", f"{p}_z", f"{p}_z_youth", f"{p}_z_rookie", f"{p}_mpg", f"{p}_mpg_youth"]
        if components:
            for c in COMPONENTS:
                names += [f"{p}_{c}_z", f"{p}_{c}_z_youth"]
    return tuple(names)


FEATURE_NAMES = feature_names(False)


# --------------------------------------------------------------------------- leakage

def slice_offseason(logs: pd.DataFrame, target_season: str) -> pd.DataFrame:
    """Rows whose ``season`` tag (the season the event *follows*) is before ``target_season``.

    This is exactly ``History.until``'s rule, restated for callers that read the parquet directly.
    """
    cutoff = season_start(target_season)
    if logs.empty:
        return logs
    return logs[logs["season"].map(season_start) < cutoff].reset_index(drop=True)


def assert_offseason_no_future(logs: pd.DataFrame, target_season: str) -> None:
    cutoff = season_start(target_season)
    if len(logs) and (logs["season"].map(season_start) >= cutoff).any():
        raise AssertionError(f"leakage: offseason rows tagged for seasons >= {target_season}")


# --------------------------------------------------------------------------- per-event player table

def _event_rows(logs: pd.DataFrame, team_games: pd.DataFrame | None, scoring: Mapping[str, float],
                context: str, as_of: date | None) -> pd.DataFrame:
    """One row per (player_id, event_season) for a single context, with raw usage and production."""
    d = logs[logs["context"] == context]
    if as_of is not None:
        d = d[d["game_date"] <= pd.Timestamp(as_of)]
    if d.empty:
        return pd.DataFrame(columns=["player_id", "event_season", "gp", "minutes", "fp_total", "gp_share", "mpg", "fp36",
                                     "ts", "usg36", "ast36", "reb36", "stk36"])
    d = d.assign(fp=fantasy_points_frame(d, scoring))
    g = d.groupby(["player_id", "event_season"]).agg(
        gp=("game_id", "size"), minutes=("min", "sum"), fp_total=("fp", "sum"),
        pts=("pts", "sum"), fga=("fga", "sum"), fta=("fta", "sum"), tov=("tov", "sum"), ast=("ast", "sum"),
        reb=("reb", "sum"), stl=("stl", "sum"), blk=("blk", "sum")).reset_index()
    if team_games is not None and len(team_games):
        tg = team_games[team_games["context"] == context]
        if as_of is not None:
            tg = tg[tg["game_date"] <= pd.Timestamp(as_of)]
        tcount = tg.groupby(["event_season", "team_id"]).size().rename("team_games")
        stints = d.groupby(["player_id", "event_season", "team_id"]).size().rename("n").reset_index()
        stints = stints.merge(tcount.reset_index(), on=["event_season", "team_id"], how="left")
        denom = stints.groupby(["player_id", "event_season"])["team_games"].sum().rename("team_games_played_for")
        g = g.merge(denom.reset_index(), on=["player_id", "event_season"], how="left")
        g["gp_share"] = np.where(g["team_games_played_for"] > 0, g["gp"] / g["team_games_played_for"], np.nan)
    else:
        g["gp_share"] = np.nan
    g["mpg"] = g["minutes"] / g["gp"]
    per36 = np.where(g["minutes"] > 0, 36.0 / g["minutes"].where(g["minutes"] > 0), np.nan)
    g["fp36"] = g["fp_total"] * per36
    shot_pts = 2.0 * (g["fga"] + 0.44 * g["fta"])
    g["ts"] = np.where(shot_pts > 0, g["pts"] / shot_pts.where(shot_pts > 0), np.nan)
    g["usg36"] = (g["fga"] + 0.44 * g["fta"] + g["tov"]) * per36
    g["ast36"] = g["ast"] * per36
    g["reb36"] = g["reb"] * per36
    g["stk36"] = (g["stl"] + g["blk"]) * per36
    return g


_Z_SOURCE = {"z": "fp36", "ts_z": "ts", "usg_z": "usg36", "ast_z": "ast36", "reb_z": "reb36", "stk_z": "stk36"}


def _cohort_z(g: pd.DataFrame) -> pd.DataFrame:
    """Add cohort-relative z-scores: ``z`` for fp36 and one per component (see :data:`COMPONENTS`).

    Each is the value minus the same event's minutes-weighted cohort mean, over the cohort's weighted sd,
    clipped at :data:`Z_CLIP`. The cohort is the event's players with at least :data:`MIN_COHORT_MINUTES`.
    """
    g = g.copy()
    for z_col in _Z_SOURCE:
        g[z_col] = np.nan
    for idx in g.groupby("event_season").groups.values():
        sub_ = g.loc[idx]
        for z_col, src in _Z_SOURCE.items():
            c = sub_[(sub_["minutes"] >= MIN_COHORT_MINUTES) & sub_[src].notna()]
            if len(c) < 5:
                continue
            w = c["minutes"].to_numpy(float)
            mu = float(np.average(c[src], weights=w))
            sd = float(np.sqrt(np.average((c[src] - mu) ** 2, weights=w)))
            if sd > 1e-9:
                g.loc[idx, z_col] = ((sub_[src] - mu) / sd).clip(-Z_CLIP, Z_CLIP)
    return g


def _cut_preseason_fraction(logs: pd.DataFrame, frac: float) -> pd.DataFrame:
    """Keep each preseason's first ``frac`` of game *dates*: a draft held midway through the preseason has
    only seen those games. ``frac >= 1`` keeps everything."""
    if frac >= 1.0 or logs.empty:
        return logs
    pre = logs["context"] == PRESEASON
    keep = ~pre.to_numpy()
    out = [logs[keep]]
    for _, d in logs[pre].groupby("event_season"):
        dates = np.sort(d["game_date"].unique())
        n = max(int(np.ceil(frac * len(dates))), 1) if len(dates) else 0
        out.append(d[d["game_date"] <= dates[n - 1]] if n else d.iloc[0:0])
    return pd.concat(out, ignore_index=True)


def event_player_table(logs: pd.DataFrame, team_games: pd.DataFrame | None, scoring: Mapping[str, float], *,
                       as_of: date | None = None, contexts: tuple[str, ...] = (SUMMER_LEAGUE, PRESEASON),
                       preseason_fraction: float = 1.0) -> pd.DataFrame:
    """Per (player_id, event_season): Summer League and preseason usage and cohort-relative production.

    Columns: ``sl_*`` and ``pre_*`` for ``gp``, ``minutes``, ``mpg``, ``gp_share``, ``fp36``, ``z`` and the
    component z-scores ``ts_z``, ``usg_z``, ``ast_z``, ``reb_z``, ``stk_z`` (NaN where the player has no
    such event, or the context is not requested). ``as_of`` cuts
    *preseason* games dated after it; ``preseason_fraction`` keeps only the first share of each
    preseason's game dates (a sensitivity for drafts held before the preseason ends).
    """
    logs = _cut_preseason_fraction(logs, preseason_fraction)
    parts = []
    for ctx, prefix, cut in ((SUMMER_LEAGUE, "sl", None), (PRESEASON, "pre", as_of)):
        if ctx in contexts:
            g = _cohort_z(_event_rows(logs, team_games, scoring, ctx, cut))
        else:
            g = pd.DataFrame(columns=["player_id", "event_season", *_EVENT_STATS])
        g = g[["player_id", "event_season", *_EVENT_STATS]]
        g.columns = ["player_id", "event_season", *[f"{prefix}_{c}" for c in _EVENT_STATS]]
        parts.append(g)
    out = parts[0].merge(parts[1], on=["player_id", "event_season"], how="outer")
    return out.sort_values(["event_season", "player_id"], kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- design matrix

def youth_weight(age: np.ndarray) -> np.ndarray:
    """1 at age <= 20, 0 at age >= 26, linear between; unknown age counts as 0 (no youth effect)."""
    a = np.asarray(age, float)
    return np.where(np.isfinite(a), np.clip((26.0 - a) / 6.0, 0.0, 1.0), 0.0)


def design(events: pd.DataFrame, age: np.ndarray, rookie: np.ndarray, K: float, components: bool = False) -> np.ndarray:
    """Feature matrix (rows aligned with ``events``), columns as in :func:`feature_names`.

    ``events`` holds the ``sl_*``/``pre_*`` columns for the player-season. Missing events give zeros
    in every column of that block, including the ``has_*`` indicator, so "no evidence" is exactly
    "no adjustment from that block". With ``components`` each box-score component z-score enters too,
    shrunk by minutes like the blended one and interacted with youth.
    """
    y = youth_weight(age)
    r = np.asarray(rookie, float)

    def shrunk(col: str, has: np.ndarray, m: np.ndarray) -> np.ndarray:
        v = events[col].to_numpy(float)
        return np.where(has & np.isfinite(v), v, 0.0) * (m / (m + K))

    cols: list[np.ndarray] = []
    for prefix, centre in (("sl", 25.0), ("pre", 20.0)):
        minutes = events[f"{prefix}_minutes"].to_numpy(float)
        has = np.isfinite(minutes) & (minutes > 0)
        m = np.where(has, minutes, 0.0)
        z = shrunk(f"{prefix}_z", has, m)
        mpg = np.where(has & np.isfinite(events[f"{prefix}_mpg"].to_numpy(float)),
                       (events[f"{prefix}_mpg"].to_numpy(float) - centre) / 10.0, 0.0)
        cols += [has.astype(float), z, z * y, z * r, mpg, mpg * y]
        if components:
            for c in COMPONENTS:
                zc = shrunk(f"{prefix}_{c}_z", has, m)
                cols += [zc, zc * y]
    return np.column_stack(cols)


# --------------------------------------------------------------------------- fitted adjustment

@dataclass
class OffseasonFit:
    """The fitted adjustment. ``adjust`` returns fantasy points per game to add for each player."""

    coefs: np.ndarray | None
    mean: np.ndarray | None
    scale: np.ndarray | None
    K: float
    ridge: float
    enabled: bool
    diagnostics: dict = field(default_factory=dict)
    components: bool = False

    def adjustment(self, events: pd.DataFrame, age: np.ndarray, rookie: np.ndarray) -> np.ndarray:
        n = len(events)
        if not self.enabled or self.coefs is None or n == 0:
            return np.zeros(n)
        X = (design(events, age, rookie, self.K, self.components) - self.mean) / self.scale
        # A row with no events at all gets exactly zero (the intercept only ever describes event holders).
        has_any = (events["sl_minutes"].fillna(0).to_numpy(float) > 0) | (events["pre_minutes"].fillna(0).to_numpy(float) > 0)
        return np.where(has_any, self.coefs[0] + X @ self.coefs[1:], 0.0)


def _standardise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    return (X - mean) / scale, mean, scale


def _cv_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, groups: np.ndarray, ridge: float) -> float:
    """Leave-one-season-out weighted squared error of the ridge fit (standardised inside each fold)."""
    tot = 0.0
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if tr.sum() < 30 or te.sum() == 0:
            return float("inf")
        Xs, mean, scale = _standardise(X[tr])
        beta = wls(np.column_stack([np.ones(tr.sum()), Xs]), y[tr], w[tr], ridge=ridge / tr.sum())
        pred = beta[0] + ((X[te] - mean) / scale) @ beta[1:]
        tot += float(np.sum(w[te] * (y[te] - pred) ** 2))
    return tot / float(w.sum())


def fit_adjustment(events: pd.DataFrame, age: np.ndarray, rookie: np.ndarray, resid: np.ndarray,
                   weight: np.ndarray, season: np.ndarray, *, min_rows: int = MIN_FIT_ROWS,
                   components: bool = False) -> OffseasonFit:
    """Fit the residual regression on training rows and decide whether it earns its place.

    ``events`` has one row per training player-season (the ``sl_*``/``pre_*`` columns), ``resid`` is the
    actual-minus-projected FPPG, ``season`` the start year used for leave-one-season-out folds.
    """
    has_any = (events["sl_minutes"].fillna(0).to_numpy(float) > 0) | (events["pre_minutes"].fillna(0).to_numpy(float) > 0)
    ok = has_any & np.isfinite(resid) & np.isfinite(weight) & (weight > 0)
    diag: dict = {"train_rows": int(ok.sum()), "train_seasons": int(len(np.unique(np.asarray(season)[ok])))}
    if ok.sum() < min_rows or diag["train_seasons"] < 2:
        diag["reason"] = "too few training rows or seasons"
        return OffseasonFit(None, None, None, K_GRID[1], RIDGE_GRID[1], False, diag, components)
    ev, a, r = events[ok].reset_index(drop=True), np.asarray(age, float)[ok], np.asarray(rookie, float)[ok]
    y, w, s = np.asarray(resid, float)[ok], np.minimum(np.asarray(weight, float)[ok], WEIGHT_GP_CAP), np.asarray(season)[ok]

    null_loss = float(np.sum(w * (y - np.average(y, weights=w)) ** 2) / w.sum())
    zero_loss = float(np.sum(w * y ** 2) / w.sum())
    best = (float("inf"), K_GRID[1], RIDGE_GRID[1])
    for K in K_GRID:
        X = design(ev, a, r, K, components)
        for ridge in RIDGE_GRID:
            loss = _cv_loss(X, y, w, s, ridge)
            if loss < best[0]:
                best = (loss, K, ridge)
    cv_loss, K, ridge = best
    gain = (zero_loss - cv_loss) / zero_loss if zero_loss > 0 else 0.0
    diag.update({"zero_adjustment_loss": zero_loss, "constant_only_loss": null_loss, "cv_loss": cv_loss,
                 "cv_gain_vs_zero": gain, "K": K, "ridge": ridge})
    if not np.isfinite(cv_loss) or gain < MIN_CV_GAIN:
        diag["reason"] = f"cross-validated gain {gain:.4f} below the {MIN_CV_GAIN} threshold"
        return OffseasonFit(None, None, None, K, ridge, False, diag, components)
    X = design(ev, a, r, K, components)
    Xs, mean, scale = _standardise(X)
    beta = wls(np.column_stack([np.ones(len(y)), Xs]), y, w, ridge=ridge / len(y))
    diag["coefficients"] = {n: float(b) for n, b in zip(("intercept", *feature_names(components)), beta)}
    diag["reason"] = "enabled"
    return OffseasonFit(beta, mean, scale, K, ridge, True, diag, components)


# --------------------------------------------------------------------------- applying an adjustment

def factor_from_adjustment(proj_fppg: np.ndarray, adj: np.ndarray) -> np.ndarray:
    """Multiplier turning ``proj_fppg`` into ``proj_fppg + adj``, clipped to ``[FACTOR_LO, FACTOR_HI]``.

    A projection at or below ~zero cannot be scaled meaningfully and is left alone (factor 1).
    """
    p = np.asarray(proj_fppg, float)
    safe = np.where(p > 1.0, p, np.nan)
    f = (safe + np.asarray(adj, float)) / safe
    return np.clip(np.where(np.isfinite(f), f, 1.0), FACTOR_LO, FACTOR_HI)


@dataclass
class TrainingSet:
    """Everything the fit needs, aligned row for row."""

    events: pd.DataFrame
    age: np.ndarray
    rookie: np.ndarray
    resid: np.ndarray
    weight: np.ndarray
    season: np.ndarray


def build_training_set(
    history_tables: Mapping[str, pd.DataFrame],
    seasons: list[str],
    project: Callable[[str], pd.DataFrame],
    events_by_season: Callable[[str], pd.DataFrame],
    age_at: Callable[[np.ndarray, int], np.ndarray],
    actual: pd.DataFrame,
    components: bool = False,
) -> TrainingSet | None:
    """Stack the walk-forward residuals of ``seasons``.

    ``project(s)`` returns the base projection for season ``s`` computed from data before ``s``
    (columns ``player_id``, ``proj_fppg``, ``is_rookie``); ``events_by_season(s)`` the event table rows of
    event season ``s``; ``actual`` has ``player_id``, ``s``, ``gp``, ``fppg`` for completed seasons.
    """
    frames = []
    for s in seasons:
        sy = season_start(s)
        proj = project(s)
        if proj is None or proj.empty:
            continue
        act = actual[(actual["s"] == sy) & (actual["gp"] >= MIN_TRAIN_GP)][["player_id", "gp", "fppg"]]
        m = proj[["player_id", "proj_fppg", "is_rookie"]].merge(act, on="player_id", how="inner")
        ev = events_by_season(s)
        m = m.merge(ev, on="player_id", how="left")
        m["season_start"] = sy
        m["age"] = age_at(m["player_id"].to_numpy(), sy)
        frames.append(m)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    cols = event_columns(components)
    for col in cols:
        if col not in d.columns:
            d[col] = np.nan
    return TrainingSet(
        events=d[cols].reset_index(drop=True),
        age=d["age"].to_numpy(float), rookie=d["is_rookie"].to_numpy(bool).astype(float),
        resid=(d["fppg"] - d["proj_fppg"]).to_numpy(float), weight=d["gp"].to_numpy(float),
        season=d["season_start"].to_numpy(int),
    )
