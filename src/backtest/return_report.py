"""Return-from-injury study: does the baseline over-discount a long absence followed by a healthy tail? (ADR 0021)

    python -m src.backtest.return_report --seasons 2016-17:2025-26 --out reports/return_2026-09-26 [--board reports/daily/draft_board_....csv]

The baseline's availability model sees one season-level fraction of the schedule played. A player who missed the first 76% of last
season (an Achilles injury) and then played 16 of the last 20 games looks the same as one who was hurt on and off all year. This module
asks, on the ten real walk-forward seasons and only from data before each target season, whether such players' *next* season is
under-projected, in games played and in total fantasy points (the metric a points-league draft runs on).

Definitions are fixed in ADR 0021 before any outcome was inspected (:class:`CohortConfig` defaults are the primary cohort; the
sensitivity grid is reported whole, never selected from):

* **Season profile** (:func:`season_profiles`): per player-season, the primary team's schedule in date order; the *lead block* is the run
  of missed games before the first appearance, the *interior block* the longest missed run that starts after game 1 and ends before the
  last game; *tail health* is games played / team games after the block.
* **Cohort**: veteran, single-team, >= ``min_mpg``, block >= ``t1`` of the schedule, tail >= ``min_tail`` games, tail health >= ``tail_x``.
* **Matched controls**: same target season, not in the cohort, absence spread over many short runs, k nearest on prior fraction played,
  age and minutes (hard calipers).
* **Outcomes**: signed *actual minus projected* (positive = under-projected) for GP and total FP, MAE, share of rows above projection.

Leakage: every feature comes from ``History.until(tables, s)`` (seasons < s, the same object the projector sees); ``s`` only supplies the
outcome. Inference is a percentile bootstrap over rows, resampled within each target season.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.backtest.actuals import season_actuals
from src.backtest.mdutil import df_to_markdown
from src.contracts import History, Projector, season_start, season_str
from src.models.panel import season_lengths, season_starts, target_season_games
from src.value.league import load_league

N_BOOT = 2000
K_CONTROLS = 3
SPREAD_SHARE = 0.60          # a "spread" absence: longest missed run <= 60% of the games missed
CALIPER_F, CALIPER_AGE, CALIPER_MPG = 0.10, 2.0, 5.0
MIN_GP_FPPG = 10             # FPPG bias needs a real sample of games


@dataclass(frozen=True)
class CohortConfig:
    """One cohort definition. The defaults are the primary (pre-registered) cohort of ADR 0021."""
    kind: str = "lead"            # "lead": block before the first appearance; "mid": interior block, then a return
    t1: float = 0.25              # block length >= t1 * team games
    tail_x: float = 0.75          # played >= tail_x of the games outside the block
    min_tail: int = 15            # games after the block must be at least this many
    min_mpg: float = 15.0         # prior-season minutes per game
    single_team: bool = True      # a mid-season trade misattributes the other team's games as missed (ADR 0006)
    exclude_targets: tuple[str, ...] = ()
    controls: str = "spread"      # "spread" (many short runs) or "any" (every non-cohort player)

    def label(self) -> str:
        extra = ("" if self.single_team else ", multi-team ok") + ("" if self.min_mpg == 15.0 else f", mpg>={self.min_mpg:g}") \
            + (f", excl {','.join(self.exclude_targets)}" if self.exclude_targets else "") \
            + ("" if self.controls == "spread" else ", controls any")
        return f"{self.kind} block>={self.t1:.0%} tail>={self.tail_x:.0%} of>={self.min_tail}g{extra}"


PRIMARY = CohortConfig()


# --------------------------------------------------------------------------- season profiles (point in time)

def _runs(missed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(starts, lengths) of each run of consecutive True in a boolean array."""
    if missed.size == 0 or not missed.any():
        return np.zeros(0, int), np.zeros(0, int)
    d = np.diff(np.concatenate(([0], missed.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return starts, ends - starts


PROFILE_COLS = ["player_id", "s", "team_id", "n_teams", "L", "gp", "mpg", "missed", "first_idx", "n_runs", "longest_run",
                "block_start", "block_len", "block_tail"]


def season_profiles(history: History) -> pd.DataFrame:
    """One row per (player_id, s) with the shape of the games the player missed, from ``history`` alone.

    ``L`` = the primary team's games (the team he played most for; a traded player's other games are the ADR 0006 approximation, flagged
    by ``n_teams > 1``), ``gp`` = games played for that team, ``first_idx`` = games missed before the first appearance (the lead block),
    ``longest_run`` = longest missed run anywhere, ``block_*`` = the longest *interior* run (starts after game 1, ends before the last
    game: the player was playing, was out for a long block, and returned) with ``block_tail`` the team games after it.
    """
    gl, tg = history.game_logs, history.team_games
    if gl.empty or tg.empty:
        return pd.DataFrame(columns=PROFILE_COLS)
    gl = gl[["season", "game_id", "player_id", "team_id", "min"]].copy()
    gl["s"] = season_starts(gl["season"])
    tg = tg[["season", "game_id", "game_date", "team_id"]].copy()
    tg["s"] = season_starts(tg["season"])
    tg = tg.sort_values(["team_id", "s", "game_date", "game_id"], kind="mergesort")
    tg["idx"] = tg.groupby(["team_id", "s"]).cumcount()
    sched_len = tg.groupby(["team_id", "s"]).size().rename("L")

    counts = gl.groupby(["player_id", "s", "team_id"]).size().rename("n").reset_index()
    n_teams = counts.groupby(["player_id", "s"]).size().rename("n_teams")
    primary = (counts.sort_values(["player_id", "s", "n", "team_id"], ascending=[True, True, False, True])
                     .drop_duplicates(["player_id", "s"]))
    tot = gl.groupby(["player_id", "s"]).agg(min_sum=("min", "sum"), gp_all=("game_id", "size"))
    mine = gl.merge(primary[["player_id", "s", "team_id"]], on=["player_id", "s", "team_id"])
    mine = mine.merge(tg[["team_id", "s", "game_id", "idx"]], on=["team_id", "s", "game_id"], how="inner")

    rows = []
    for (pid, s, team), g in mine.groupby(["player_id", "s", "team_id"], sort=True):
        L = int(sched_len.loc[(team, s)])
        present = np.zeros(L, bool)
        present[g["idx"].to_numpy()] = True
        gp = int(present.sum())
        starts, lens = _runs(~present)
        first_idx = int(np.argmax(present)) if gp else L
        block_start = block_len = block_tail = 0
        interior = (starts > 0) & (starts + lens < L)
        if interior.any():
            j = int(np.flatnonzero(interior)[np.argmax(lens[interior])])
            block_start, block_len, block_tail = int(starts[j]), int(lens[j]), int(L - starts[j] - lens[j])
        t = tot.loc[(pid, s)]
        rows.append((int(pid), int(s), int(team), int(n_teams.loc[(pid, s)]), L, gp,
                     float(t["min_sum"]) / max(int(t["gp_all"]), 1), L - gp, first_idx, len(lens),
                     int(lens.max()) if len(lens) else 0, block_start, block_len, block_tail))
    return pd.DataFrame(rows, columns=PROFILE_COLS)


def _veteran_flags(history: History, pids: np.ndarray, s: np.ndarray) -> np.ndarray:
    """True if the player had an NBA season before ``s``: ``players.from_year`` (sanitised) or any earlier game log in the history."""
    frm = pd.to_numeric(history.players.drop_duplicates("player_id").set_index("player_id")["from_year"], errors="coerce")
    first_seen = history.game_logs.assign(s=season_starts(history.game_logs["season"])).groupby("player_id")["s"].min()
    f = frm.reindex(pids).to_numpy(dtype="float64")
    g = first_seen.reindex(pids).to_numpy(dtype="float64")
    return (np.nan_to_num(f, nan=np.inf) < s) | (np.nan_to_num(g, nan=np.inf) < s)


def veteran_flags(history: History, pids: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Public name for :func:`_veteran_flags` (same result)."""
    return _veteran_flags(history, pids, s)


def prior_features(history: History) -> pd.DataFrame:
    """Prior-season (``s - 1``) absence profile of every player who appeared then, keyed on ``player_id``.

    A pure function of ``history`` (seasons < target); columns are :data:`PROFILE_COLS` less the key plus ``veteran``, ``f``.
    """
    tgt = season_start(history.target_season)
    prof = season_profiles(history)
    prof = prof[prof["s"] == tgt - 1].copy()
    prof["veteran"] = _veteran_flags(history, prof["player_id"].to_numpy(), prof["s"].to_numpy())
    prof["f"] = prof["gp"] / prof["L"]
    return prof.drop(columns="s").reset_index(drop=True)


# --------------------------------------------------------------------------- cohort membership

def in_universe(df: pd.DataFrame, cfg: CohortConfig) -> np.ndarray:
    ok = df["veteran"].to_numpy(bool) & (df["mpg"].to_numpy(float) >= cfg.min_mpg) & (df["L"].to_numpy() > 0)
    if cfg.single_team:
        ok &= df["n_teams"].to_numpy() == 1
    if cfg.exclude_targets:
        ok &= ~df["season"].isin(cfg.exclude_targets).to_numpy()
    return ok


def in_cohort(df: pd.DataFrame, cfg: CohortConfig) -> np.ndarray:
    """Boolean mask of the rows that are 'a long absence block, then a healthy tail' under ``cfg`` (and inside the universe)."""
    L, gp = df["L"].to_numpy(float), df["gp"].to_numpy(float)
    if cfg.kind == "lead":
        block, tail = df["first_idx"].to_numpy(float), L - df["first_idx"].to_numpy(float)
        outside = tail                                    # everything after the first appearance
        lead_ok = df["gp"].to_numpy() > 0
    elif cfg.kind == "mid":
        block, tail = df["block_len"].to_numpy(float), df["block_tail"].to_numpy(float)
        outside = L - block                               # health is judged over every game outside the block
        lead_ok = (df["block_len"].to_numpy() > 0) & (df["first_idx"].to_numpy() < df["block_start"].to_numpy())   # he was playing before it
    else:
        raise ValueError(f"unknown cohort kind {cfg.kind!r}")
    with np.errstate(divide="ignore", invalid="ignore"):
        health = np.where(outside > 0, gp / outside, 0.0)
    return in_universe(df, cfg) & lead_ok & (block >= cfg.t1 * L) & (tail >= cfg.min_tail) & (health >= cfg.tail_x)


def control_pool(df: pd.DataFrame, cfg: CohortConfig, cohort: np.ndarray) -> np.ndarray:
    """Universe rows that are not in the cohort and (default) whose absence is spread over many short runs."""
    pool = in_universe(df, cfg) & ~cohort
    if cfg.controls == "spread":
        missed = df["missed"].to_numpy(float)
        pool &= (missed > 0) & (df["longest_run"].to_numpy(float) <= SPREAD_SHARE * missed)
    return pool


def match_controls(df: pd.DataFrame, cohort: np.ndarray, pool: np.ndarray, k: int = K_CONTROLS) -> list[np.ndarray]:
    """For each cohort row (in order), positions of its <= ``k`` nearest controls in the same target season within the calipers.

    Distance is the sum of squared caliper-scaled differences in prior fraction played, age and minutes per game; rows with a NaN
    matching variable never match. Matching is with replacement.
    """
    X = np.column_stack([df["f"].to_numpy(float) / CALIPER_F, df["age"].to_numpy(float) / CALIPER_AGE, df["mpg"].to_numpy(float) / CALIPER_MPG])
    season = df["season"].to_numpy()
    out = []
    for i in np.flatnonzero(cohort):
        cand = np.flatnonzero(pool & (season == season[i]))
        if len(cand):
            d = np.abs(X[cand] - X[i])
            cand = cand[(d <= 1.0).all(axis=1) & np.isfinite(d).all(axis=1)]
            d = ((X[cand] - X[i]) ** 2).sum(axis=1)
            cand = cand[np.argsort(d, kind="stable")[:k]]
        out.append(cand)
    return out


# --------------------------------------------------------------------------- walk-forward frame

def build_study_frame(tables: Mapping[str, pd.DataFrame], seasons: Sequence[str], projector: Projector,
                      scoring: Mapping[str, float], *, log=lambda msg: None) -> pd.DataFrame:
    """One row per (target season, projected player who appeared the season before): prior-season profile, projection and outcome.

    ``History.until`` builds what the projector and the features both see, so the two cannot disagree about what is known.
    Projected players who did not play the target season count 0 GP / 0 FP (the harness rule); players with no prior-season game
    are outside the study (no absence profile to condition on).
    """
    frames = []
    for s in seasons:
        log(f"[{s}] projecting and profiling")
        h = History.until(tables, s)
        h.assert_no_future()
        proj = projector.project(h)
        feat = prior_features(h)
        act = season_actuals(tables["game_logs"], s, scoring)[["player_id", "gp", "total_fp", "fppg"]].rename(
            columns={"gp": "actual_gp", "total_fp": "actual_total_fp", "fppg": "actual_fppg"})
        cols = ["player_id", "player_name", "age", "proj_gp", "proj_fppg", "proj_total_fp"]
        f = proj[cols].merge(feat, on="player_id", how="inner").merge(act, on="player_id", how="left")
        f["actual_gp"] = f["actual_gp"].fillna(0.0)
        f["actual_total_fp"] = f["actual_total_fp"].fillna(0.0)
        f["target_len"] = float(target_season_games(season_lengths(h)))
        f.insert(0, "season", s)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def add_outcomes(f: pd.DataFrame) -> pd.DataFrame:
    """Signed errors (actual minus projected; positive = under-projected), absolute errors and the above-projection flag."""
    f = f.copy()
    f["err_gp"] = f["actual_gp"] - f["proj_gp"]
    f["err_fp"] = f["actual_total_fp"] - f["proj_total_fp"]
    f["ae_gp"], f["ae_fp"] = f["err_gp"].abs(), f["err_fp"].abs()
    f["above"] = (f["actual_gp"] > f["proj_gp"]).astype(float)
    f["err_fppg"] = np.where(f["actual_gp"] >= MIN_GP_FPPG, f["actual_fppg"] - f["proj_fppg"], np.nan)
    return f


# --------------------------------------------------------------------------- inference

def boot_mean(values, strata, *, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """Pooled mean and 95% percentile CI, resampling rows within each stratum (season). NaNs dropped; (nan, nan, nan) if empty."""
    v, st = np.asarray(values, float), np.asarray(strata)
    ok = np.isfinite(v)
    v, st = v[ok], st[ok]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    tot = np.zeros(n_boot)
    for k in np.unique(st):
        g = v[st == k]
        tot += g[rng.integers(0, len(g), size=(n_boot, len(g)))].sum(axis=1)
    lo, hi = np.quantile(tot / len(v), [0.025, 0.975])
    return float(v.mean()), float(lo), float(hi)


METRICS = (("bias_gp", "err_gp"), ("bias_fp", "err_fp"), ("mae_gp", "ae_gp"), ("mae_fp", "ae_fp"),
           ("above_share", "above"), ("bias_fppg", "err_fppg"))


def _ctrl_mean(col: np.ndarray, ctrls: list[np.ndarray]) -> np.ndarray:
    return np.array([np.nanmean(col[c]) if len(c) and np.isfinite(col[c]).any() else np.nan for c in ctrls])


def evaluate(f: pd.DataFrame, cfg: CohortConfig = PRIMARY, *, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Cohort vs matched controls under ``cfg``. Returns n's and, per metric, cohort / control / paired-difference (est, lo, hi).

    ``cohort`` uses every cohort row; ``control`` and ``diff`` use the cohort rows that found at least one control (``n_matched``),
    the diff being row-wise cohort minus the mean of that row's controls, so the season mix is the cohort's.
    """
    f = f.reset_index(drop=True)
    coh = in_cohort(f, cfg)
    pool = control_pool(f, cfg, coh)
    ctrls = match_controls(f, coh, pool)
    rows = np.flatnonzero(coh)
    matched = np.array([len(c) > 0 for c in ctrls], bool)
    out = {"cfg": cfg, "n": int(coh.sum()), "n_matched": int(matched.sum()), "n_pool": int(pool.sum()),
           "n_seasons": int(f.loc[coh, "season"].nunique()), "rows": rows, "metrics": {}}
    season = f["season"].to_numpy()
    for name, col in METRICS:
        v = f[col].to_numpy(float)
        cv = v[rows]
        ctl = _ctrl_mean(v, ctrls)
        m = matched & np.isfinite(cv) & np.isfinite(ctl)
        out["metrics"][name] = {
            "cohort": boot_mean(cv, season[rows], n_boot=n_boot, seed=seed),
            "control": boot_mean(ctl[m], season[rows][m], n_boot=n_boot, seed=seed),
            "diff": boot_mean((cv - ctl)[m], season[rows][m], n_boot=n_boot, seed=seed),
            "n_diff": int(m.sum()),
        }
    return out


def verdict(res: dict) -> str:
    """The pre-registered rule: 'too harsh' needs the paired total-FP difference CI above zero (ADR 0021)."""
    if res["n_matched"] == 0:
        return "no controls"
    _, lo, hi = res["metrics"]["bias_fp"]["diff"]
    if np.isnan(lo):
        return "n/a"
    return "too harsh" if lo > 0 else "too lenient" if hi < 0 else "inconclusive"


# --------------------------------------------------------------------------- tables

def _ci(t: tuple[float, float, float], fmt: str = "{:+.1f}") -> str:
    if np.isnan(t[0]):
        return "n/a"
    return f"{fmt.format(t[0])} [{fmt.format(t[1])}, {fmt.format(t[2])}]"


def primary_table(res: dict) -> pd.DataFrame:
    rows = []
    for name, label, fmt in (("bias_gp", "bias GP (actual - proj)", "{:+.1f}"), ("bias_fp", "bias total FP (actual - proj)", "{:+.0f}"),
                             ("mae_gp", "MAE GP", "{:.1f}"), ("mae_fp", "MAE total FP", "{:.0f}"),
                             ("above_share", "share actual GP > proj GP", "{:.0%}"), ("bias_fppg", "bias FPPG (>=10 GP)", "{:+.2f}")):
        m = res["metrics"][name]
        rows.append({"metric": label, "cohort [95% CI]": _ci(m["cohort"], fmt), "matched controls [95% CI]": _ci(m["control"], fmt),
                     "cohort - controls [95% CI]": _ci(m["diff"], fmt), "n (paired)": m["n_diff"]})
    return pd.DataFrame(rows)


def per_season_table(f: pd.DataFrame, res: dict) -> pd.DataFrame:
    d = f.iloc[res["rows"]]
    g = d.groupby("season").agg(n=("player_id", "size"), proj_gp=("proj_gp", "mean"), actual_gp=("actual_gp", "mean"),
                                bias_gp=("err_gp", "mean"), bias_fp=("err_fp", "mean"), above_share=("above", "mean"))
    return g.reset_index()


def sensitivity_grid(f: pd.DataFrame, *, n_boot: int = N_BOOT) -> pd.DataFrame:
    """Every cell reported as it falls: block threshold x tail health for the lead type, the mid type, and the robustness variants."""
    cells: list[CohortConfig] = []
    for t1 in (0.25, 0.40, 0.60):
        for x in (0.60, 0.75, 0.90):
            cells.append(replace(PRIMARY, t1=t1, tail_x=x))
    for t1 in (0.25, 0.40):
        cells.append(replace(PRIMARY, kind="mid", t1=t1))
    cells += [replace(PRIMARY, kind="mid", t1=0.25, tail_x=0.90),
              replace(PRIMARY, single_team=False), replace(PRIMARY, min_mpg=0.0), replace(PRIMARY, exclude_targets=("2020-21",)),
              replace(PRIMARY, min_tail=10), replace(PRIMARY, min_tail=25), replace(PRIMARY, controls="any")]
    rows = []
    for cfg in cells:
        r = evaluate(f, cfg, n_boot=n_boot)
        m = r["metrics"]
        rows.append({"cohort": cfg.label(), "primary": cfg == PRIMARY, "n": r["n"], "n_matched": r["n_matched"],
                     "bias GP": _ci(m["bias_gp"]["cohort"]), "bias FP": _ci(m["bias_fp"]["cohort"], "{:+.0f}"),
                     "GP vs controls": _ci(m["bias_gp"]["diff"]), "FP vs controls": _ci(m["bias_fp"]["diff"], "{:+.0f}"),
                     "verdict": verdict(r)})
    return pd.DataFrame(rows)


def leave_one_season_out(f: pd.DataFrame, cfg: CohortConfig = PRIMARY, *, n_boot: int = N_BOOT) -> pd.DataFrame:
    """The primary result with each target season dropped in turn: is one season (a shortened one, a wave of retirements) driving it?

    Added after the per-season table showed one season far from the rest; it is a fixed procedure over every season, not a choice of
    which season to drop, and it does not enter the verdict.
    """
    rows = []
    for s in sorted(f["season"].unique()):
        r = evaluate(f[f["season"] != s], cfg, n_boot=n_boot)
        m = r["metrics"]
        rows.append({"dropped season": s, "n": r["n"], "n_matched": r["n_matched"], "bias GP": _ci(m["bias_gp"]["cohort"]),
                     "bias FP": _ci(m["bias_fp"]["cohort"], "{:+.0f}"), "GP vs controls": _ci(m["bias_gp"]["diff"]),
                     "FP vs controls": _ci(m["bias_fp"]["diff"], "{:+.0f}"), "verdict": verdict(r)})
    return pd.DataFrame(rows)


def calibration_table(f: pd.DataFrame, cohort_rows: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Projected-GP decile of the whole study universe vs actual GP, with the primary cohort's rows inside each decile."""
    d = f.copy()
    d["in_cohort"] = False
    d.loc[d.index[cohort_rows], "in_cohort"] = True
    d["bin"] = pd.qcut(d["proj_gp"].rank(method="first"), n_bins, labels=False) + 1
    g = d.groupby("bin")
    out = pd.DataFrame({"n": g.size(), "proj_gp": g["proj_gp"].mean(), "actual_gp": g["actual_gp"].mean(),
                        "bias_gp": g["err_gp"].mean(), "bias_fp": g["err_fp"].mean()})
    c = d[d["in_cohort"]].groupby("bin")
    out["cohort_n"] = c.size()
    out["cohort_bias_gp"] = c["err_gp"].mean()
    out["cohort_bias_fp"] = c["err_fp"].mean()
    out["cohort_actual_gp"] = c["actual_gp"].mean()
    out["cohort_proj_gp"] = c["proj_gp"].mean()
    out["cohort_n"] = out["cohort_n"].fillna(0).astype(int)
    return out.reset_index()


def proj_gp_bands(f: pd.DataFrame, cohort_rows: np.ndarray) -> pd.DataFrame:
    """Cohort rows only, three projected-GP bands (a cohort too small for deciles): where the miss lives."""
    d = f.iloc[cohort_rows].copy()
    d["band"] = pd.cut(d["proj_gp"], [0, 35, 50, 1e9], labels=["proj GP < 35", "35 <= proj GP < 50", "proj GP >= 50"], right=False)
    g = d.groupby("band", observed=True)
    return pd.DataFrame({"n": g.size(), "proj_gp": g["proj_gp"].mean(), "actual_gp": g["actual_gp"].mean(),
                         "bias_gp": g["err_gp"].mean(), "bias_fp": g["err_fp"].mean()}).reset_index()


# --------------------------------------------------------------------------- current cohort (informational)

def current_cohort(tables: Mapping[str, pd.DataFrame], projector: Projector, res: dict, board: pd.DataFrame | None = None,
                   cfg: CohortConfig = PRIMARY) -> tuple[str, pd.DataFrame]:
    """Players in the primary cohort as of the latest data, with the study's implied correction. Changes no model or board output.

    ``corr_gp_raw`` is the cohort's own mean GP miss (what the projection would need to add to be unbiased on this group);
    ``corr_gp_excess`` is the part beyond matched controls. Both are capped at the schedule when applied. ``board`` (columns
    ``player_id, proj_gp, proj_fppg, proj_total_fp``, optional ``rank``, ``adp``) replaces the live ``projector`` output as the
    'current' projection.
    """
    latest = max(season_start(s) for s in tables["game_logs"]["season"].unique())
    target = season_str(latest + 1)
    h = History.until(tables, target)
    feat = prior_features(h)
    proj = projector.project(h)
    cur = proj[["player_id", "player_name", "age", "proj_gp", "proj_fppg", "proj_total_fp"]].merge(feat, on="player_id", how="inner")
    cur["season"] = target
    cur = cur[in_cohort(cur, cfg)].copy()
    src = f"live `{getattr(projector, 'name', 'projector')}` projection for {target}"
    if board is not None:
        keep = [c for c in ("player_id", "rank", "adp", "proj_gp", "proj_fppg", "proj_total_fp") if c in board.columns]
        cur = cur.drop(columns=["proj_gp", "proj_fppg", "proj_total_fp"]).merge(board[keep], on="player_id", how="left")
        src = f"draft board CSV ({len(board)} rows) for {target}"
    m = res["metrics"]
    raw, exc = m["bias_gp"]["cohort"][0], m["bias_gp"]["diff"][0]
    cap = float(target_season_games(season_lengths(h)))
    cur["prior_gp"], cur["prior_L"] = cur["gp"], cur["L"]
    cur["block_games"] = cur["first_idx"]
    cur["tail_health"] = cur["gp"] / (cur["L"] - cur["first_idx"])
    cur["corr_gp_raw"] = raw
    cur["corr_gp_excess"] = exc
    cur["gp_if_raw"] = np.minimum(cur["proj_gp"] + raw, cap)
    cur["fp_if_raw"] = cur["gp_if_raw"] * cur["proj_fppg"]
    cur["fp_delta_raw"] = cur["fp_if_raw"] - cur["proj_gp"] * cur["proj_fppg"]
    cur["fp_delta_excess"] = (np.minimum(cur["proj_gp"] + exc, cap) - cur["proj_gp"]) * cur["proj_fppg"]
    cur["block_share"] = cur["first_idx"] / cur["L"]
    cols = [c for c in ("rank", "player_name", "age", "adp", "prior_gp", "prior_L", "block_games", "block_share", "tail_health", "proj_gp",
                        "proj_fppg", "gp_if_raw", "fp_delta_raw", "fp_delta_excess") if c in cur.columns]
    cols = list(dict.fromkeys(cols))
    sort = "rank" if "rank" in cur.columns and cur["rank"].notna().any() else "proj_total_fp"
    return src, cur.sort_values(sort, ascending=sort == "rank")[cols].reset_index(drop=True)


# --------------------------------------------------------------------------- report

def render(f: pd.DataFrame, res: dict, grid: pd.DataFrame, per_season: pd.DataFrame, loo: pd.DataFrame, calib: pd.DataFrame,
           bands: pd.DataFrame, others: Mapping[str, dict], current: tuple[str, pd.DataFrame] | None, *, model: str, command: str, n_boot: int) -> str:
    m = res["metrics"]
    all_bias = (f["err_gp"].mean(), f["err_fp"].mean())
    lines = [
        "# Return-from-injury study", "",
        f"Generated by `{command}` from real data; projector `{model}`; walk-forward, every feature from seasons before the target. "
        f"Percentile bootstrap, {n_boot} resamples, resampled within season. See ADR 0021.", "",
        f"* Study universe: {len(f)} projected player-seasons over {f['season'].nunique()} target seasons (all players, before the "
        f"veteran / single-team / minutes filters). Mean miss over everyone: {all_bias[0]:+.1f} GP, {all_bias[1]:+.0f} total FP.",
        f"* Primary cohort ({res['cfg'].label()}): **n = {res['n']}** in {res['n_seasons']} seasons; {res['n_matched']} have at least one matched "
        f"control (pool {res['n_pool']} spread-absence controls). **Verdict under the pre-registered rule: {verdict(res)}.**", "",
        "## 1. Primary cohort vs matched controls", "",
        "Signed error is actual minus projected: positive means the model under-projected. *cohort - controls* is the row-wise paired "
        "difference against each row's own matched controls (same target season, similar age, minutes and prior fraction played).", "",
        df_to_markdown(primary_table(res), index=False), "",
        "## 2. By target season (primary cohort)", "", df_to_markdown(per_season.round(2), index=False, floatfmt="{:.2f}"), "",
        "### Leave one season out", "",
        "The primary result with each target season dropped in turn (added after the per-season table; it does not enter the verdict).", "",
        df_to_markdown(loo, index=False), "",
        "## 3. Sensitivity grid (every cell, no selection)", "",
        "Block threshold `t1` x tail health `X` for the lead type, the mid type (long block after playing, then a return), and robustness "
        "variants. *vs controls* is the paired difference; the verdict applies the pre-registered total-FP rule to each cell.", "",
        df_to_markdown(grid, index=False), "",
        "## 4. Calibration: projected GP vs actual GP", "",
        "Deciles of `proj_gp` over the whole study universe, with the primary cohort's rows inside each decile.", "",
        df_to_markdown(calib.round(2), index=False, floatfmt="{:.1f}",
                                                                   formats={"bin": "{:.0f}", "n": "{:.0f}", "cohort_n": "{:.0f}"}), "",
        "Cohort rows only, by projected-GP band:", "", df_to_markdown(bands.round(2), index=False, floatfmt="{:.1f}"), ""]
    if others:
        lines += ["## 5. Other projectors on the same primary cohort", "",
                  "| projector | n | bias GP [95% CI] | bias total FP [95% CI] | cohort - controls, FP [95% CI] | verdict |", "|---|---|---|---|---|---|"]
        for name, r in others.items():
            mm = r["metrics"]
            lines.append(f"| {name} | {r['n']} | {_ci(mm['bias_gp']['cohort'])} | {_ci(mm['bias_fp']['cohort'], '{:+.0f}')} | "
                         f"{_ci(mm['bias_fp']['diff'], '{:+.0f}')} | {verdict(r)} |")
        lines.append("")
    if current is not None:
        src, cur = current
        lines += ["## 6. Players in the primary cohort now (informational only)", "",
                  f"Source of the current projection: {src}. `corr_gp_raw` = the cohort's mean GP miss in the walk-forward "
                  f"({m['bias_gp']['cohort'][0]:+.1f}), `corr_gp_excess` = its excess over matched controls ({m['bias_gp']['diff'][0]:+.1f}); "
                  "`gp_if_raw` applies the raw correction (capped at the schedule) and `fp_delta_raw` is the resulting change in total FP at the "
                  "current FPPG; `fp_delta_excess` is the same change if the whole excess over controls were applied (an upper reading). "
                  "**Nothing here changes the model or the board.**", "",
                  df_to_markdown(cur.round(2), index=False, floatfmt="{:.2f}") if len(cur) else "_no player currently qualifies_", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.backtest.return_report", description=__doc__.split("\n\n")[0])
    ap.add_argument("--seasons", default="2016-17:2025-26")
    ap.add_argument("--model", default="baseline", help="projector under test (registry name)")
    ap.add_argument("--compare", default="baseline_injury,baseline_offseason_debut", help="other projectors for the primary cohort (comma list, '' for none)")
    ap.add_argument("--board", type=Path, default=None, help="draft board CSV to read the current proj_gp/proj_fppg from")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from src.backtest.harness import expand_seasons
    from src.backtest.runner import load_run_tables
    from src.models.registry import get_projector

    seasons = expand_seasons(args.seasons)
    tables = load_run_tables(argparse.Namespace(synthetic=False))
    scoring = load_league()["scoring"]
    projector = get_projector(args.model)
    f = add_outcomes(build_study_frame(tables, seasons, projector, scoring, log=print))
    res = evaluate(f, PRIMARY, n_boot=args.n_boot)
    grid = sensitivity_grid(f, n_boot=args.n_boot)
    others = {}
    for name in [x.strip() for x in args.compare.split(",") if x.strip() and x.strip() != args.model]:
        fo = add_outcomes(build_study_frame(tables, seasons, get_projector(name), scoring, log=print))
        others[name] = evaluate(fo, PRIMARY, n_boot=args.n_boot)
    board = pd.read_csv(args.board) if args.board else None
    current = current_cohort(tables, projector, res, board)
    cmd = "python -m src.backtest.return_report " + " ".join(sys.argv[1:] if argv is None else argv)
    loo = leave_one_season_out(f, n_boot=args.n_boot)
    text = render(f, res, grid, per_season_table(f, res), loo, calibration_table(f, res["rows"]), proj_gp_bands(f, res["rows"]), others,
                  current, model=args.model, command=cmd, n_boot=args.n_boot)
    out = args.out or Path("reports") / f"return_{date.today():%Y-%m-%d}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "return_report.md").write_text(text, encoding="utf-8")
    primary_table(res).to_csv(out / "primary.csv", index=False)
    grid.to_csv(out / "sensitivity.csv", index=False)
    calibration_table(f, res["rows"]).to_csv(out / "calibration.csv", index=False)
    current[1].to_csv(out / "current_cohort.csv", index=False)
    f.to_parquet(out / "study_frame.parquet", index=False)
    print(text)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
