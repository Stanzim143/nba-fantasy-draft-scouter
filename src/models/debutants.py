"""Debutant path: stash players (drafted earlier, only now arriving) and undrafted signees (ADR 0016).

The baseline projects veterans from their own history and the *current* draft class from a draft-slot prior. A player
who is neither (a 2022 second-rounder arriving from Turkey, a 2025 lottery pick who missed his rookie year, an
undrafted two-way signee) was named on the board's notes and left unprojected. This module projects them, always
flagged low confidence:

    lines     the slot prior of ``src.models.rookies`` at an *effective* pick (an old pick is worth less: the pick is
              pulled toward "undrafted" by ``decay ** years_since_draft``), with minutes scaled by a class multiplier
    games     ``p_play * share`` of the schedule: the chance he plays at all, times the share of games he plays if he does
    p_play    stash: an explicit assumed constant (see below); undrafted: a small logistic fit on preseason usage
              (games share, minutes) and Summer League minutes, fitted on earlier seasons' camp participants

Everything estimated (class multipliers, games share, decay, the logistic) is fitted on **earlier seasons'** analogues
only (:func:`build_analogs`), so the walk-forward backtest (``python -m src.backtest.debutants``) can score it.

The stash ``p_play`` is an assumption, not an estimate. A backtest can only recognise a stash as "drafted earlier" if
he has an index record, and only people who eventually played have one, so every historical stash analogue played
(36 of 36): the true rate for a *rostered* stash is not identifiable from history. ``STASH_P_PLAY`` is a documented
constant; players on an NBA roster on draft day are overwhelmingly ones who play at least some games.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from src.contracts import History, season_start, season_str
from src.features.debutants import CLASSES, STASH, UNDRAFTED, find_candidates
from src.features.offseason import event_player_table
from src.models.baseline import BaselineProjector, FittedBaseline
from src.models.rookies import UNDRAFTED_PICK
from src.value.frame import fantasy_points_frame

STASH_P_PLAY = 0.90
DECAY_GRID = (1.0, 0.85, 0.70, 0.50)
MIN_DECAY_GAIN = 0.10        # a pick discount must cut training error by this much (relative) to be used
SHRINK_ROWS = 6.0            # pseudo-rows pulling the class multipliers toward 1
P_CLIP = (0.02, 0.97)
EVIDENCE = ("pre_gp_share", "pre_mpg", "sl_mpg")
LOGIT_RIDGE = 1.0
MIN_LOGIT_ROWS = 60
MIN_CLASS_PLAYED = 5         # fewer training debutants of a class than this: use the pooled class (or unit multipliers)


# --------------------------------------------------------------------------- analogues (training rows)

def season_actuals(game_logs: pd.DataFrame, scoring: Mapping[str, float]) -> pd.DataFrame:
    """Per (start-year, player): games, minutes per game, fantasy points per game and total."""
    gl = game_logs
    fp = fantasy_points_frame(gl, scoring).to_numpy()
    d = pd.DataFrame({"s": gl["season"].map(season_start).to_numpy(), "player_id": gl["player_id"].to_numpy(),
                      "min": gl["min"].to_numpy(float), "fp": fp})
    g = d.groupby(["s", "player_id"]).agg(gp=("fp", "size"), mpg=("min", "mean"), fppg=("fp", "mean"),
                                          tot=("fp", "sum")).reset_index()
    return g


def build_analogs(game_logs: pd.DataFrame, offseason_logs: pd.DataFrame, offseason_team_games: pd.DataFrame | None,
                  profiles: pd.DataFrame, scoring: Mapping[str, float], seasons: list[str],
                  players: pd.DataFrame | None = None, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Camp participants with no earlier NBA game and their realised season, for each of ``seasons``.

    Every input must already be sliced to what was knowable (``game_logs`` only through the last of ``seasons``; nothing
    from a later season is read). Players who never appear in that season's game logs get ``gp = tot = 0``.
    """
    if events is None:
        events = event_player_table(offseason_logs, offseason_team_games, scoring)
    act = season_actuals(game_logs, scoring)
    starts = game_logs["season"].map(season_start).to_numpy()
    from src.features.debutants import camp_ids

    window = int(starts.min()) if len(starts) else None
    rows = []
    for s in seasons:
        sy = season_start(s)
        prior = np.unique(game_logs["player_id"].to_numpy()[starts < sy])
        c = find_candidates(s, prior_player_ids=prior, profiles=profiles, source_ids=camp_ids(events, s),
                            players=players, events=events, window_start=window)
        if c.empty:
            continue
        c["s"] = sy
        rows.append(c)
    if not rows:
        return pd.DataFrame()
    A = pd.concat(rows, ignore_index=True).merge(act, on=["s", "player_id"], how="left")
    for col in ("gp", "tot"):
        A[col] = A[col].fillna(0.0)
    A["played"] = A["gp"] > 0
    return A


# --------------------------------------------------------------------------- fitted parameters

@dataclass
class DebutantParams:
    """What was learned from earlier seasons' analogues. ``enabled`` is False when there was nothing to learn from."""

    mpg_scale: dict[str, float] = field(default_factory=lambda: {STASH: 1.0, UNDRAFTED: 1.0})
    stash_intl_scale: float = 1.0     # extra multiplier for internationally-developed stashes (shrunk toward 1)
    gp_share: dict[str, float] = field(default_factory=lambda: {UNDRAFTED: 0.20})
    decay: float = 1.0
    p_stash: float = STASH_P_PLAY
    logit: np.ndarray | None = None
    logit_mean: np.ndarray | None = None
    logit_scale: np.ndarray | None = None
    p_base: float = 0.30
    evidence_means: np.ndarray | None = None
    n_train: dict[str, int] = field(default_factory=dict)
    enabled: bool = False
    notes: list[str] = field(default_factory=list)


def effective_pick(pick: np.ndarray, years: np.ndarray, decay: float) -> np.ndarray:
    """An old pick counts for less: the slot is pulled toward undrafted (61) by ``decay ** years``."""
    p = np.asarray(pick, float)
    y = np.where(np.isfinite(years), years, 0.0)
    return UNDRAFTED_PICK - (UNDRAFTED_PICK - p) * np.power(decay, y)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _evidence_matrix(df: pd.DataFrame, means: np.ndarray | None = None):
    X = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(float) for c in EVIDENCE])
    X[:, 0] = np.clip(X[:, 0], 0, 1)
    has = np.isfinite(X[:, 0]) | np.isfinite(X[:, 1])
    if means is None:
        means = np.nanmean(X, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
    X = np.where(np.isfinite(X), X, means)
    return np.column_stack([X, has.astype(float)]), means


def _fit_logit(X: np.ndarray, y: np.ndarray, ridge: float = LOGIT_RIDGE, iters: int = 50) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(Xb.shape[1])
    pen = np.eye(Xb.shape[1]) * ridge
    pen[0, 0] = 0.0
    for _ in range(iters):
        p = _sigmoid(Xb @ beta)
        grad = Xb.T @ (y - p) - pen @ beta
        hess = (Xb.T * (p * (1 - p))) @ Xb + pen + 1e-9 * np.eye(len(beta))
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def fit_params(fitted: FittedBaseline, analogs: pd.DataFrame) -> DebutantParams:
    """Fit the class multipliers, games share, pick decay and undrafted play probability on earlier analogues."""
    prm = DebutantParams()
    if analogs is None or analogs.empty:
        prm.notes.append("no analogues: prior used as-is")
        return prm
    A = analogs.copy()
    played = A[A["played"]]
    prm.n_train = {c: int(((A["klass"] == c) & A["played"]).sum()) for c in CLASSES}
    prm.n_train.update({f"{c}_camp": int((A["klass"] == c).sum()) for c in CLASSES})
    L = float(fitted.season_games)

    def prior_for(d: pd.DataFrame, decay: float) -> pd.DataFrame:
        pick = effective_pick(d["pick"].to_numpy(float), d["years_since_draft"].to_numpy(float), decay)
        r = fitted.prior_rows(d["player_id"].to_numpy("int64"), pick, d["group"].to_numpy(), d["age"].to_numpy(float))
        r["prior_fppg"] = fitted._fppg(r)
        return r

    st = played[played["klass"] == STASH]
    if len(st) >= MIN_CLASS_PLAYED:
        base_loss = None
        best = (np.inf, 1.0)
        for dcy in DECAY_GRID:
            r = prior_for(st, dcy)
            loss = float(np.mean(np.abs(r["prior_fppg"].to_numpy() - st["fppg"].to_numpy(float))))
            if dcy == 1.0:
                base_loss = loss
            if loss < best[0]:
                best = (loss, dcy)
        if base_loss and (base_loss - best[0]) / base_loss >= MIN_DECAY_GAIN:
            prm.decay = best[1]
    # Stash lines are the slot prior as it is: the walk-forward evaluation (ADR 0016) found no pick, age or years-since-draft
    # effect in its residuals and a plain prior beat every fitted multiplier. The one candidate signal, overseas
    # development, is a shrunk multiplier on the prior's production.
    st_all = played[played["klass"] == STASH]
    if len(st_all) >= MIN_CLASS_PLAYED:
        r = prior_for(st_all, prm.decay)
        intl = st_all["intl"].to_numpy(float) > 0
        if intl.sum() >= 3:
            w = np.minimum(st_all["gp"].to_numpy(float), 30.0)[intl]
            ratio = np.sum(w * st_all["fppg"].to_numpy(float)[intl]) / np.sum(w * r["prior_fppg"].to_numpy(float)[intl])
            n = float(intl.sum())
            prm.stash_intl_scale = float((n * ratio + SHRINK_ROWS * 1.0) / (n + SHRINK_ROWS))
    else:
        prm.notes.append(f"stash: only {len(st_all)} training debutants, prior used as-is")
    d = played[played["klass"] == UNDRAFTED]
    if len(d) >= MIN_CLASS_PLAYED:
        r = prior_for(d, 1.0)
        w = np.minimum(d["gp"].to_numpy(float), 30.0)
        pred, act = r["proj_mpg"].to_numpy(float), d["mpg"].to_numpy(float)
        k = SHRINK_ROWS * float(np.mean(pred))
        prm.mpg_scale[UNDRAFTED] = float((np.sum(w * act) / np.sum(w) * len(d) + k) / (np.sum(w * pred) / np.sum(w) * len(d) + k))
        prm.gp_share[UNDRAFTED] = float((( d["gp"].to_numpy(float) / L).sum() + SHRINK_ROWS * 0.25) / (len(d) + SHRINK_ROWS))
    else:
        prm.notes.append(f"undrafted: only {len(d)} training debutants, multipliers left at their defaults")
    und = A[A["klass"] == UNDRAFTED]
    prm.p_base = float(und["played"].mean()) if len(und) else prm.p_base
    if len(und) >= MIN_LOGIT_ROWS and und["played"].nunique() == 2:
        X, means = _evidence_matrix(und)
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        prm.logit = _fit_logit((X - mu) / sd, und["played"].to_numpy(float))
        prm.logit_mean, prm.logit_scale = np.concatenate([mu]), sd
        prm.evidence_means = means
    prm.enabled = True
    return prm


def p_play(prm: DebutantParams, cand: pd.DataFrame) -> np.ndarray:
    p = np.where(cand["klass"].to_numpy() == STASH, prm.p_stash, prm.p_base)
    if prm.logit is not None:
        X, _ = _evidence_matrix(cand, prm.evidence_means)
        pu = _sigmoid(prm.logit[0] + ((X - prm.logit_mean) / prm.logit_scale) @ prm.logit[1:])
        p = np.where(cand["klass"].to_numpy() == UNDRAFTED, pu, p)
    return np.clip(p, *P_CLIP)


def predict_rows(fitted: FittedBaseline, prm: DebutantParams, cand: pd.DataFrame) -> pd.DataFrame:
    """Component rows (same shape as ``FittedBaseline.prior_rows``) plus ``p_play`` and ``klass`` for ``cand``."""
    if cand.empty:
        return pd.DataFrame()
    stash = cand["klass"].to_numpy() == STASH
    pick = np.where(stash, effective_pick(cand["pick"].to_numpy(float), cand["years_since_draft"].to_numpy(float), prm.decay),
                    UNDRAFTED_PICK)
    intl = cand["intl"].to_numpy(float) > 0
    scale = np.where(stash, np.where(intl, prm.stash_intl_scale, 1.0), prm.mpg_scale[UNDRAFTED])
    pp = p_play(prm, cand)
    group, age, pids = cand["group"].to_numpy(), cand["age"].to_numpy(float), cand["player_id"].to_numpy("int64")
    prior_f = np.clip(fitted.rookie_prior.predict("f", pick, group, age), 0.02, 0.985)
    share = np.where(stash, prior_f, prm.gp_share[UNDRAFTED])   # a stash plays the prior's share; an undrafted signee the class share
    rows = fitted.prior_rows(pids, pick, group, age, mpg_scale=scale, mu_f=np.clip(pp * share, 0.005, 0.985))
    rows["p_play"] = pp
    rows["klass"] = cand["klass"].to_numpy()
    return rows


# --------------------------------------------------------------------------- the projector

class DebutantBaselineProjector(BaselineProjector):
    """``baseline`` plus projected stash and undrafted debutants. Registered as ``"baseline_debut"``.

    Veterans and the current draft class are **exactly** the baseline's rows (same numbers); the only change is extra
    rows for debutants, with ``projection_class`` (``veteran`` / ``rookie`` / ``stash`` / ``undrafted``), ``p_play`` and
    ``confidence = "low"``. Inputs beyond the ``History``: ``player_profiles`` and (live only) the newest roster
    snapshot, from ``History.extras`` or the parquet store; with neither, no debutant is added.
    """

    name = "baseline_debut"

    def __init__(self, *args, profiles: pd.DataFrame | None = None, roster: pd.DataFrame | None = None,
                 offseason_logs: pd.DataFrame | None = None, offseason_team_games: pd.DataFrame | None = None,
                 min_analog_seasons: int = 2, add_debutants: bool = True, origin: bool = False,
                 combine: pd.DataFrame | None = None, **kw):
        super().__init__(*args, **kw)
        self.add_debutants, self.origin, self._combine = add_debutants, origin, combine
        self._profiles, self._roster = profiles, roster
        self._logs, self._tg = offseason_logs, offseason_team_games
        self.min_analog_seasons = min_analog_seasons
        self.last_params: DebutantParams | None = None
        self.last_candidates: pd.DataFrame | None = None

    # -- data access (History.extras first, then the store; missing -> None)
    def _get(self, history: History, key: str, explicit, loader):
        if explicit is not None:
            return explicit
        if key in history.extras:
            return history.extras[key]
        try:
            return loader()
        except (FileNotFoundError, ImportError, OSError):
            return None

    def _inputs(self, history: History):
        from src.ingest import nba_offseason, nba_profiles

        profiles = self._get(history, "player_profiles", self._profiles, nba_profiles.read_profiles)
        logs = self._get(history, "offseason_logs", self._logs, nba_offseason.read_offseason_logs)
        tg = self._get(history, "offseason_team_games", self._tg, nba_offseason.read_offseason_team_games)
        if logs is not None:
            from src.features.offseason import assert_offseason_no_future, slice_offseason

            logs = slice_offseason(logs, history.target_season)
            tg = slice_offseason(tg, history.target_season) if tg is not None else None
            assert_offseason_no_future(logs, history.target_season)
        return profiles, logs, tg

    def _live_roster_ids(self, history: History):
        """Player ids on the newest roster snapshot **if it belongs to the target season**, else None (backtest)."""
        snap = self._roster
        if snap is None:
            snap = history.extras.get("roster_snapshots")
        if snap is None:
            try:
                from src.ingest.nba_incoming import read_roster_snapshots

                snap = read_roster_snapshots()
            except (FileNotFoundError, ImportError, OSError):
                return None
        if snap is None or snap.empty:
            return None
        cur = snap[snap["season"] == history.target_season]
        if cur.empty:
            return None
        latest = cur[cur["snapshot_date"] == cur["snapshot_date"].max()]
        self._roster_names = latest.drop_duplicates("player_id").set_index("player_id")["player_name"]
        return latest["player_id"].to_numpy("int64")

    def project(self, history: History) -> pd.DataFrame:
        fitted = self.fit(history)
        if self.origin:
            self._attach_origin(history, fitted)
        out = fitted.predict()
        veterans = out["is_rookie"].to_numpy(bool)
        out["projection_class"] = np.where(veterans, "rookie", "veteran")
        out["p_play"] = np.nan
        extra = self._debutant_rows(history, fitted) if self.add_debutants else None
        if extra is not None and len(extra):
            out = pd.concat([out, extra], ignore_index=True).sort_values("player_id", kind="mergesort").reset_index(drop=True)
        out["model"] = self.name
        from src.contracts import validate_table

        return validate_table(out, "projections")

    def _attach_origin(self, history: History, fitted: FittedBaseline) -> None:
        """Origin-aware rookie prior (``src.models.rookie_origin``): needs profiles; the combine is optional."""
        from src.ingest import nba_profiles
        from src.models.rookie_origin import make_adjuster

        profiles = self._get(history, "player_profiles", self._profiles, nba_profiles.read_profiles)
        if profiles is None:
            return
        combine = self._get(history, "draft_combine", self._combine, nba_profiles.read_combine)
        if combine is None:
            combine = pd.DataFrame(columns=nba_profiles.COMBINE_COLUMNS)
        fitted.rookie_adjust = make_adjuster(fitted, history, profiles, combine, self.scoring)

    def _debutant_rows(self, history: History, fitted: FittedBaseline) -> pd.DataFrame | None:
        profiles, logs, tg = self._inputs(history)
        if profiles is None:
            return None
        scoring = self.scoring
        events = event_player_table(logs, tg, scoring) if logs is not None and len(logs) else None
        gl = history.game_logs
        prior = gl["player_id"].unique()
        roster_ids = self._live_roster_ids(history)
        target = history.target_season
        if roster_ids is not None:
            source = roster_ids
        elif events is not None:
            from src.features.debutants import camp_ids

            source = camp_ids(events, target)
        else:
            return None
        names = fitted.names
        if logs is not None and len(logs):
            # camp participants who never got an NBA record have no profile row (and so no name there): the box
            # scores name them. Requiring a profile row would silently keep only future NBA players.
            ln = logs.drop_duplicates("player_id", keep="last").set_index("player_id")["player_name"]
            names = pd.concat([names, ln[~ln.index.isin(names.index)]])
        snap_names = getattr(self, "_roster_names", None)
        if snap_names is not None:
            names = pd.concat([names, snap_names[~snap_names.index.isin(names.index)]])
        cand = find_candidates(target, prior_player_ids=prior, profiles=profiles, source_ids=source, names=names,
                               players=history.players, events=events,
                               window_start=int(gl["season"].map(season_start).min()) if len(gl) else None)
        if cand.empty:
            return None
        # drop anyone who is already a projected rookie of this draft class
        cand = cand[~cand["player_id"].isin(fitted.rookie_player_ids()) & cand["player_name"].notna()].reset_index(drop=True)
        analogs = pd.DataFrame()
        if logs is not None and len(logs) and events is not None:
            start = season_start(target)
            first = int(gl["season"].map(season_start).min()) if len(gl) else start
            seasons = [season_str(y) for y in range(first + 1, start)]
            analogs = build_analogs(gl, logs, tg, profiles, scoring, seasons, players=history.players, events=events)
            if len(seasons) < self.min_analog_seasons:
                analogs = pd.DataFrame()
        prm = fit_params(fitted, analogs)
        self.last_params, self.last_candidates = prm, cand
        rows = predict_rows(fitted, prm, cand)
        if rows.empty:
            return None
        d = rows.copy()
        out = fitted._assemble(d.drop(columns=["p_play", "klass"]))
        m = d.set_index("player_id")
        out["projection_class"] = m["klass"].reindex(out["player_id"]).to_numpy()
        out["p_play"] = m["p_play"].reindex(out["player_id"]).to_numpy()
        out["confidence"] = "low"
        out["position"] = cand.set_index("player_id")["position"].reindex(out["player_id"]).to_numpy()
        out["age"] = cand.set_index("player_id")["age"].reindex(out["player_id"]).to_numpy()
        nm = cand.set_index("player_id")["player_name"].reindex(out["player_id"])
        out["player_name"] = np.where(pd.isna(out["player_name"]), nm.to_numpy(), out["player_name"])
        return out
