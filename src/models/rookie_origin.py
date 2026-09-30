"""Origin-aware rookie prior: draft slot + position + age + country / previous organisation + combine (ADR 0016).

The shipped rookie prior (``src.models.rookies``) is a set of per-stat regressions on slot, position group and age. The
walk-forward test in ``src.backtest.rookie_origin`` measures what adding two origin flags (``intl``, ``non_college``) and
the draft-combine measurements does. This module holds the shared pieces:

* :func:`feature_frame`: the features for a set of players (point-in-time safe: country, previous organisation and
  combine results are fixed before the draft);
* :func:`predict_variants`: refit the slot regression **on the direct targets** (minutes, games fraction, FP per game)
  with and without the extra features, on earlier drafts' rookies, and predict the target season's rookies;
* :func:`make_adjuster`: turns that into multipliers on the shipped prior's minutes, games and per-minute rates
  (``FittedBaseline.rookie_adjust``), so a stat line keeps every box-score identity and the shipped prior stays the anchor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.contracts import History
from src.models.baseline import FittedBaseline
from src.models.rookies import pick_effective, to_float
from src.models.shrink import wls
from src.value.frame import fantasy_points_frame

VARIANTS = ("base", "origin", "combine", "both")
RIDGE = 1.0
MIN_TRAIN_ROOKIES = 60
FACTOR_CLIP = (0.6, 1.6)
COMBINE_ATTENDANCE = 0.6     # centring constant for the attendance indicator (about 60% attend)


def feature_frame(pids: np.ndarray, draft_years: np.ndarray, profiles: pd.DataFrame, combine: pd.DataFrame) -> pd.DataFrame:
    """Origin and combine features for ``pids`` (the combine is matched on player and draft year)."""
    pr = profiles.drop_duplicates("player_id").set_index("player_id")
    f = pd.DataFrame({"player_id": pids})
    f["intl"] = (pr["origin"].reindex(pids).to_numpy() == "intl").astype(float)
    f["non_college"] = (pr["prev_org_type"].reindex(pids).to_numpy() == "other").astype(float)
    f["draft_year"] = draft_years
    cb = combine.drop_duplicates(["draft_year", "player_id"])
    f = f.merge(cb, on=["player_id", "draft_year"], how="left")
    wing = (f["wingspan_in"] - f["height_wo_shoes_in"]).to_numpy(float)
    vert = f["max_vertical_in"].to_numpy(float)
    has = np.isfinite(wing) & np.isfinite(vert)
    f["has_combine"] = has.astype(float)
    f["wing_minus_height"] = np.where(has, wing, np.nan)   # missing stays NaN; ``_design`` imputes the training mean
    f["max_vertical"] = np.where(has, vert, np.nan)
    return f[["player_id", "intl", "non_college", "has_combine", "wing_minus_height", "max_vertical"]]


def _design(pick, group, age, centers, feats: pd.DataFrame, variant: str, cmeans=(0.0, 0.0)) -> np.ndarray:
    lp, ag = centers
    age_c = np.where(np.isfinite(age), age - ag, 0.0)
    cols = [np.ones(len(pick)), np.log(pick) - lp, (np.asarray(group) == "G").astype(float),
            (np.asarray(group) == "C").astype(float), age_c]
    if variant in ("origin", "both"):
        cols += [feats["intl"].to_numpy(float), feats["non_college"].to_numpy(float)]
    if variant in ("combine", "both"):
        w = np.where(np.isfinite(feats["wing_minus_height"].to_numpy(float)), feats["wing_minus_height"].to_numpy(float), cmeans[0])
        v = np.where(np.isfinite(feats["max_vertical"].to_numpy(float)), feats["max_vertical"].to_numpy(float), cmeans[1])
        cols += [feats["has_combine"].to_numpy(float) - COMBINE_ATTENDANCE, (w - cmeans[0]) / 3.0, (v - cmeans[1]) / 3.0]
    return np.column_stack(cols)


def predict_variants(fitted: FittedBaseline, history: History, profiles: pd.DataFrame, combine: pd.DataFrame,
                     scoring, *, variants=VARIANTS, ridge: float = RIDGE) -> pd.DataFrame | None:
    """Direct-target predictions (``{variant}_mpg / _f / _fppg`` per rookie of the target season), or ``None`` when the
    history holds fewer than :data:`MIN_TRAIN_ROOKIES` earlier rookies."""
    panel = fitted.panel
    sy = fitted.target_start
    pl = history.players.drop_duplicates("player_id").set_index("player_id")
    dy = to_float(pl["draft_year"].reindex(panel["player_id"]).to_numpy())
    is_r = np.isfinite(dy) & (dy == panel["s"].to_numpy(float))
    tr = panel[is_r].copy()
    if len(tr) < MIN_TRAIN_ROOKIES:
        return None
    rids = fitted.rookie_player_ids()
    if not len(rids):
        return None
    tr_dy = dy[is_r].astype("int64")
    sums = tr[["pts", "fgm", "fga", "fg3m", "ftm", "fta", "reb", "ast", "stl", "blk", "tov"]]
    tr["fppg"] = fantasy_points_frame(sums, scoring).to_numpy() / tr["gp"].to_numpy(float)
    tpick = pick_effective(pl["draft_number"].reindex(tr["player_id"]).to_numpy(), pl["draft_round"].reindex(tr["player_id"]).to_numpy())
    tage = tr["age"].to_numpy(float)
    centers = (float(np.mean(np.log(tpick))), float(np.nanmean(tage)))
    tf_ = feature_frame(tr["player_id"].to_numpy("int64"), tr_dy, profiles, combine)
    rl = pl.reindex(rids)
    rpick = pick_effective(rl["draft_number"].to_numpy(), rl["draft_round"].to_numpy())
    rgroup = fitted._positions(rids)
    rage = fitted.panel_data.ages.at(rids, sy)
    rf = feature_frame(rids, np.full(len(rids), sy, dtype="int64"), profiles, combine)
    has = tf_["has_combine"].to_numpy() > 0
    cmeans = (float(np.nanmean(tf_["wing_minus_height"])) if has.any() else 0.0,
              float(np.nanmean(tf_["max_vertical"])) if has.any() else 0.0)
    out = pd.DataFrame({"player_id": rids})
    for v in variants:
        X = _design(tpick, tr["pos_group"].to_numpy(), tage, centers, tf_, v, cmeans)
        Xr = _design(rpick, rgroup, rage, centers, rf, v, cmeans)
        out[f"{v}_mpg"] = np.clip(Xr @ wls(X, tr["mpg"].to_numpy(float), tr["gp"].to_numpy(float), ridge=ridge), 0.5, 40)
        out[f"{v}_f"] = np.clip(Xr @ wls(X, tr["f"].to_numpy(float), np.ones(len(tr)), ridge=ridge), 0.02, 0.985)
        out[f"{v}_fppg"] = np.clip(Xr @ wls(X, tr["fppg"].to_numpy(float), tr["gp"].to_numpy(float), ridge=ridge), 0.0, 80)
    out["intl"] = rf["intl"].to_numpy()
    out["has_combine"] = rf["has_combine"].to_numpy()
    return out


def make_adjuster(fitted: FittedBaseline, history: History, profiles: pd.DataFrame, combine: pd.DataFrame, scoring,
                  *, target: str = "both"):
    """``fitted.rookie_adjust`` callable: multipliers (minutes, games, per-minute count rates) for the rookies, from the
    ratio of the ``target`` variant's regression to the same regression without the extra features. ``None`` when
    there is too little history."""
    pred = predict_variants(fitted, history, profiles, combine, scoring, variants=("base", target))
    if pred is None:
        return None
    m = np.clip(pred[f"{target}_mpg"] / pred["base_mpg"], *FACTOR_CLIP).to_numpy()
    f = np.clip(pred[f"{target}_f"] / pred["base_f"], *FACTOR_CLIP).to_numpy()
    fp = np.clip(pred[f"{target}_fppg"] / pred["base_fppg"].clip(lower=1.0), *FACTOR_CLIP).to_numpy()
    idx = {int(p): i for i, p in enumerate(pred["player_id"])}

    def adjust(pids):
        ii = np.array([idx.get(int(p), -1) for p in pids])
        ok = ii >= 0
        mm, ff, kk = np.ones(len(ii)), np.ones(len(ii)), np.ones(len(ii))
        mm[ok], ff[ok] = m[ii[ok]], f[ii[ok]]
        kk[ok] = fp[ii[ok]] / m[ii[ok]]          # production per game scales by fp; minutes already carry m
        return mm, ff, kk

    return adjust
