"""Rookie appearance stage: the chance a drafted rookie plays at all in his first season (ADR 0038).

The draft-slot prior (``src.models.rookies``) is fit on rookie seasons with at least one game, and the hurdle stage of
availability (``src.models.appearance``, ADR 0031) covers veterans only, so every rookie was projected as certain to
appear. History says otherwise, and says it most clearly through the preseason: walk-forward 2015-16..2025-26, drafted
rookies with a preseason game appear in about 98% of cases, those without one in about 39% (mostly stashed second-
rounders and injured picks). This module fits::

    P(appear)   ridge logistic on log(pick), played-in-preseason, preseason minutes per game,
                played-in-Summer-League and Summer League minutes per game

on earlier seasons' draft classes (rows: players with ``draft_year`` equal to a history season's start year; label: any
game in that season), and the projector multiplies the rookie's ``p_appear`` by it, so ``proj_gp = p * mu_f * L`` exactly
as for a veteran in the hurdle models and the season-total band gets its point mass at zero.

**Partial preseason (live only).** A draft held during the preseason has seen only part of it. On the live path (the
projector passes ``live=True`` when a roster snapshot of the target season exists) the target's preseason game dates so far
are compared with the median of the last ``RECENT_SEASONS`` training preseasons, and every training season's preseason is cut
to that share of its game dates (``src.features.offseason._cut_preseason_fraction``) before the features are built, so "no
preseason game yet" means the same thing in training as on draft day. In the walk-forward backtest the target preseason is
complete, so nothing is cut: preseason length varies too much (22 game dates in 2015-16, 9 in 2020-21) for a date count to
tell a finished short preseason from an unfinished one. With no target preseason game at all the preseason columns are
dropped on either path.

**Known bias.** Draftees who have never played an NBA game have no ``players`` row, so they are missing from the training
rows (and from the backtest's scoring universe). The fitted rate is therefore, if anything, too high, which makes the stage
conservative. Everything is a pure function of the ``History`` and the offseason logs sliced to the target, so it is
leakage-safe like the rest of the debutant path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.contracts import season_start
from src.features.offseason import PRESEASON, SUMMER_LEAGUE, _cut_preseason_fraction
from src.models.rookies import pick_effective, to_float

MIN_ROWS = 100            # fewer training rookies than this (or fewer than MIN_NEG non-appearances): no stage
MIN_NEG = 10
P_CLIP = (0.05, 0.995)
RIDGE = 1.0
COLUMNS = ("log_pick", "has_pre", "pre_mpg", "has_sl", "sl_mpg")
PRESEASON_COLUMNS = ("has_pre", "pre_mpg")
RECENT_SEASONS = 3        # live: the reference preseason length is the median of this many latest training preseasons


def evidence(logs: pd.DataFrame | None) -> pd.DataFrame:
    """Per (event_season, player_id): games and minutes per game in the preseason and the Summer League."""
    cols = ["event_season", "player_id", "pre_gp", "pre_mpg", "sl_gp", "sl_mpg"]
    if logs is None or logs.empty:
        return pd.DataFrame(columns=cols)
    parts = []
    for ctx, prefix in ((PRESEASON, "pre"), (SUMMER_LEAGUE, "sl")):
        d = logs[logs["context"] == ctx]
        g = d.groupby(["event_season", "player_id"]).agg(gp=("game_id", "nunique"), minutes=("min", "sum")).reset_index()
        g[f"{prefix}_gp"] = g["gp"].astype(float)
        g[f"{prefix}_mpg"] = np.where(g["gp"] > 0, g["minutes"] / g["gp"].where(g["gp"] > 0), np.nan)
        parts.append(g[["event_season", "player_id", f"{prefix}_gp", f"{prefix}_mpg"]])
    out = parts[0].merge(parts[1], on=["event_season", "player_id"], how="outer")
    return out[cols]


def preseason_dates(logs: pd.DataFrame | None, season: str) -> int:
    if logs is None or logs.empty:
        return 0
    d = logs[(logs["context"] == PRESEASON) & (logs["event_season"] == season)]
    return int(d["game_date"].nunique())


def target_fraction(logs: pd.DataFrame | None, target: str, train_seasons: list[str], live: bool = False) -> float:
    """Share of the target's preseason already seen, in [0, 1]: 0 with no target preseason game, 1 when not ``live``.

    Live, the reference length is the median game dates of the last ``RECENT_SEASONS`` training preseasons.
    """
    n = preseason_dates(logs, target)
    if n == 0:
        return 0.0
    if not live:
        return 1.0
    full = [k for k in (preseason_dates(logs, s) for s in train_seasons) if k > 0][-RECENT_SEASONS:]
    if not full:
        return 1.0
    return float(min(1.0, n / float(np.median(full))))


def cut_logs(logs: pd.DataFrame | None, fraction: float) -> pd.DataFrame | None:
    """Offseason logs with every preseason cut to its first ``fraction`` of game dates (0 drops the preseason)."""
    if logs is None or logs.empty or fraction >= 1.0:
        return logs
    if fraction <= 0.0:
        return logs[logs["context"] != PRESEASON].reset_index(drop=True)
    return _cut_preseason_fraction(logs, fraction)


def design(pick: np.ndarray, ev: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Raw feature matrix (before standardising); ``ev`` holds ``pre_gp/pre_mpg/sl_gp/sl_mpg`` aligned with ``pick``."""
    pre_gp = to_float(ev["pre_gp"]) if "pre_gp" in ev else np.full(len(pick), np.nan)
    sl_gp = to_float(ev["sl_gp"]) if "sl_gp" in ev else np.full(len(pick), np.nan)
    has_pre = np.nan_to_num(pre_gp) > 0
    has_sl = np.nan_to_num(sl_gp) > 0
    feats = {
        "log_pick": np.log(np.asarray(pick, float)),
        "has_pre": has_pre.astype(float),
        "pre_mpg": np.where(has_pre, np.nan_to_num(to_float(ev["pre_mpg"])), 0.0),
        "has_sl": has_sl.astype(float),
        "sl_mpg": np.where(has_sl, np.nan_to_num(to_float(ev["sl_mpg"])), 0.0),
    }
    return np.column_stack([feats[c] for c in columns])


@dataclass
class RookieAppearance:
    """A fitted stage; ``enabled`` False means every rookie keeps ``p_appear = 1``."""

    columns: tuple[str, ...] = COLUMNS
    beta: np.ndarray | None = None
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    fraction: float = 1.0
    base_rate: float = 1.0
    n_train: int = 0
    n_appeared: int = 0
    enabled: bool = False
    notes: list[str] = field(default_factory=list)

    def predict(self, pick: np.ndarray, ev: pd.DataFrame) -> np.ndarray:
        if not self.enabled:
            return np.ones(len(pick))
        from src.models.debutants import _sigmoid

        X = (design(pick, ev, self.columns) - self.mean) / self.scale
        return np.clip(_sigmoid(self.beta[0] + X @ self.beta[1:]), *P_CLIP)


def training_rows(players: pd.DataFrame, game_logs: pd.DataFrame, ev: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """One row per drafted rookie of each of ``seasons``: pick, evidence and whether he played that season."""
    pl = players.drop_duplicates("player_id")
    dy = to_float(pl["draft_year"])
    played = set(zip(game_logs["season"].to_numpy(), game_logs["player_id"].to_numpy("int64")))
    rows = []
    for s in seasons:
        r = pl[dy == season_start(s)]
        if r.empty:
            continue
        pids = r["player_id"].to_numpy("int64")
        rows.append(pd.DataFrame({
            "season": s, "player_id": pids,
            "pick": pick_effective(r["draft_number"].to_numpy(), r["draft_round"].to_numpy()),
            "appeared": [(s, int(p)) in played for p in pids],
        }))
    if not rows:
        return pd.DataFrame(columns=["season", "player_id", "pick", "appeared", "pre_gp", "pre_mpg", "sl_gp", "sl_mpg"])
    out = pd.concat(rows, ignore_index=True)
    return out.merge(ev.rename(columns={"event_season": "season"}), on=["season", "player_id"], how="left")


def fit(players: pd.DataFrame, game_logs: pd.DataFrame, logs: pd.DataFrame | None, target: str,
        live: bool = False) -> RookieAppearance:
    """Fit on every history season before ``target`` (``game_logs``/``players``/``logs`` already sliced to it).

    ``live``: the target preseason may be unfinished (draft day), so the training preseasons are cut to match it.
    """
    from src.models.debutants import _fit_logit

    seasons = sorted({s for s in game_logs["season"].unique() if season_start(s) < season_start(target)}, key=season_start)
    frac = target_fraction(logs, target, seasons, live)
    columns = COLUMNS if frac > 0 else tuple(c for c in COLUMNS if c not in PRESEASON_COLUMNS)
    T = training_rows(players, game_logs, evidence(cut_logs(logs, frac)), seasons)
    res = RookieAppearance(columns=columns, fraction=frac, n_train=len(T), n_appeared=int(T["appeared"].sum()))
    res.base_rate = float(T["appeared"].mean()) if len(T) else 1.0
    if frac <= 0:
        res.notes.append("no preseason game of the target season yet: preseason columns dropped")
    if len(T) < MIN_ROWS or (len(T) - res.n_appeared) < MIN_NEG:
        res.notes.append(f"too few training rookies ({len(T)}, {len(T) - res.n_appeared} non-appearances): stage off")
        return res
    X = design(T["pick"].to_numpy(float), T, columns)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    res.beta = _fit_logit((X - mu) / sd, T["appeared"].to_numpy(float), ridge=RIDGE)
    res.mean, res.scale, res.enabled = mu, sd, True
    return res


def make_hook(stage: RookieAppearance, players: pd.DataFrame, logs: pd.DataFrame | None, target: str):
    """``callable(pids) -> p_appear`` for the target season's rookies (the ``FittedBaseline.rookie_p_appear`` slot)."""
    ev = evidence(logs)
    ev = ev[ev["event_season"] == target].drop(columns="event_season").drop_duplicates("player_id").set_index("player_id")
    pl = players.drop_duplicates("player_id").set_index("player_id")

    def hook(pids: np.ndarray) -> np.ndarray:
        pids = np.asarray(pids, "int64")
        r = pl.reindex(pids)
        pick = pick_effective(r["draft_number"].to_numpy(), r["draft_round"].to_numpy())
        return stage.predict(pick, ev.reindex(pids).reset_index(drop=True))

    return hook
