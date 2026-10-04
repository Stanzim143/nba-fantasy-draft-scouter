"""Walk-forward backtest harness.

For each target season S the harness

1. builds ``History`` from seasons < S only (``leakage.build_history``: sliced tables, sanitised
   ``players``, ``assert_no_future`` + an ``extras`` check),
2. runs the projector on that history and nothing else,
3. validates the returned frame against the ``projections`` contract and its semantics,
4. scores it against season S's realised fantasy points under the league scoring
   (``actuals.build_eval_frame`` documents the evaluation universe),
5. records per-season metrics and the player-level frame.

It is deterministic: no randomness is consumed here, players are processed in ``player_id`` order,
and the result carries a content hash of the input data so a rerun can be verified.
"""
from __future__ import annotations

import json
import logging
import platform
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.backtest import actuals as act
from src.backtest import metrics as M
from src.backtest.errors import BacktestError, ProjectionValidationError
from src.backtest.leakage import assert_projector_ignores_future, build_history, data_hash
from src.contracts import (HISTORY_TABLES, PROJECTION_STATS, ContractError, Projector, season_start,
                           season_str, validate_table)
from src.value.frame import fantasy_points_frame
from src.value.league import load_league

log = logging.getLogger("src.backtest")

HARNESS_VERSION = "1.0"
MAX_SANE_GP = 100.0
RANK_ONLY_COLUMNS = ("season", "player_id", "player_name", "model", "proj_total_fp")


# --------------------------------------------------------------------------- seasons

def expand_seasons(spec: str) -> list[str]:
    """'2016-17:2025-26' -> inclusive range; '2016-17,2018-19' -> list; '2020-21' -> one season."""
    out: list[str] = []
    for part in (p.strip() for p in spec.split(",") if p.strip()):
        if ":" in part:
            a, b = (season_start(x.strip()) for x in part.split(":"))
            if b < a:
                raise ValueError(f"season range {part!r} runs backwards")
            out.extend(season_str(y) for y in range(a, b + 1))
        else:
            season_start(part)
            out.append(part)
    if not out:
        raise ValueError("no seasons given")
    return out


# --------------------------------------------------------------------------- projection validation

def validate_projections(
    proj: pd.DataFrame,
    *,
    season: str,
    projector_name: str,
    rank_only: bool = False,
    scoring: Mapping[str, float] | None = None,
    check_scoring: bool = True,
) -> pd.DataFrame:
    """Raise ``ProjectionValidationError`` (naming projector and season) unless ``proj`` is usable.

    Checks: contract columns/dtypes/nulls; every row is for ``season``; one row per player
    (across ALL ``model`` values, not just per (season, player, model)); finite, sane values;
    ``proj_total_fp == proj_fppg * proj_gp``; ``proj_fppg`` equals the league-scoring value of the
    projected stat line (catches a model scored with the wrong league); ordered floor <= median
    <= ceiling.  ``rank_only`` projectors need only ``RANK_ONLY_COLUMNS``.
    """
    who = f"projector {projector_name!r}, season {season}"

    def bad(msg: str):
        raise ProjectionValidationError(f"{who}: {msg}")

    if not isinstance(proj, pd.DataFrame):
        bad(f"project() returned {type(proj).__name__}, expected a DataFrame")
    if proj.empty:
        bad("project() returned no players")
    if rank_only:
        missing = [c for c in RANK_ONLY_COLUMNS if c not in proj.columns]
        if missing:
            bad(f"rank-only projection missing columns {missing}")
        if proj[list(RANK_ONLY_COLUMNS)].isna().any().any():
            bad("rank-only projection has nulls in required columns")
    else:
        try:
            validate_table(proj, "projections")
        except ContractError as exc:
            raise ProjectionValidationError(f"{who}: violates the projections contract: {exc}") from exc
    wrong = sorted(set(proj["season"].astype(str)) - {season})
    if wrong:
        bad(f"rows for seasons {wrong}; expected only {season}")
    dup = proj["player_id"].duplicated()
    if dup.any():
        bad(f"{int(dup.sum())} duplicate player_id rows (e.g. {proj.loc[dup, 'player_id'].head(3).tolist()})")
    if not np.isfinite(proj["proj_total_fp"].to_numpy(dtype="float64")).all():
        bad("non-finite proj_total_fp")
    if rank_only:
        return proj

    for c in ("proj_gp", "proj_mpg", "proj_fppg", *PROJECTION_STATS):
        if not np.isfinite(proj[c].to_numpy(dtype="float64")).all():
            bad(f"non-finite values in {c}")
    if (proj["proj_gp"] < 0).any() or (proj["proj_gp"] > MAX_SANE_GP).any():
        bad(f"proj_gp outside [0, {MAX_SANE_GP:g}]")
    if not np.allclose(proj["proj_total_fp"], proj["proj_fppg"] * proj["proj_gp"], rtol=1e-6, atol=1e-3):
        bad("proj_total_fp != proj_fppg * proj_gp (the contract defines it as their product)")
    if check_scoring:
        sc = scoring if scoring is not None else load_league()["scoring"]
        expect = fantasy_points_frame(proj, sc, prefix="proj_")
        if not np.allclose(proj["proj_fppg"], expect, rtol=1e-3, atol=0.05):
            worst = float((proj["proj_fppg"] - expect).abs().max())
            bad(f"proj_fppg does not match the league scoring applied to the projected stat line "
                f"(max abs gap {worst:.2f} FP); was the wrong scoring used?")
    q = proj[["fppg_p10", "fppg_p50", "fppg_p90"]].to_numpy(dtype="float64")
    lo_hi = ~np.isnan(q[:, [0, 2]]).any(axis=1)
    if (q[lo_hi, 0] > q[lo_hi, 2]).any():
        bad("floor exceeds ceiling (fppg_p10 > fppg_p90)")
    mid = ~np.isnan(q).any(axis=1)
    if ((q[mid, 0] > q[mid, 1]) | (q[mid, 1] > q[mid, 2])).any():
        bad("quantiles are not ordered p10 <= p50 <= p90")
    return proj


# --------------------------------------------------------------------------- result

@dataclass
class BacktestResult:
    """Everything a walk-forward run produced.

    ``season_metrics``: one row per season (index = season) of every metric in
    ``metrics.build_specs`` plus coverage counts and calibration shares.
    ``players``: the long player-level evaluation frame (all seasons stacked).
    ``config`` / ``metadata``: what was run and how to reproduce it.
    """
    projector: str
    seasons: list[str]
    season_metrics: pd.DataFrame
    players: pd.DataFrame
    config: dict
    metadata: dict = field(default_factory=dict)

    # ---- views
    def summary(self) -> pd.Series:
        """Mean of every metric across seasons (each season = one draft, weight 1)."""
        return M.summarize_seasons(self.season_metrics)

    def season_frame(self, season: str) -> pd.DataFrame:
        return self.players[self.players["season"] == season].reset_index(drop=True)

    def summary_ci(self, names: Sequence[str] | None = None, *, n_boot: int = 500,
                   seed: int = 0, level: float = 0.95) -> pd.DataFrame:
        """Bootstrap CI (over players, stratified by season) of each metric's season-mean."""
        names = list(names) if names is not None else _DEFAULT_CI_METRICS(self.config["ks"])
        ks, min_gp = tuple(self.config["ks"]), int(self.config["min_gp"])
        rows = {}
        for name in names:
            groups = []
            for s in self.seasons:
                r = float(self.season_metrics.loc[s, "replacement_level"])
                spec = M.metric_spec(name, ks, r)
                p, a = M.select_arrays(self.season_frame(s), spec, min_gp)
                groups.append((p, a, np.full(len(p), r)))    # r rides along so each season keeps its own R
            if name.startswith("vorp_weighted"):
                kind = "bias" if name.endswith("bias") else "mae"
                def stat(g, kind=kind):
                    return M.vorp_weighted_error(g[0], g[1], float(g[2][0]) if len(g[2]) else 0.0, kind)
            else:
                def stat(g, fn=spec.fn):
                    return fn(g[0], g[1])
            b = M.bootstrap_groups(groups, stat, n_boot=n_boot, seed=seed, level=level,
                                   resample="groups" if M.is_rank_metric(name) else "players")
            rows[name] = {"estimate": b.estimate, "lo": b.lo, "hi": b.hi, "se": b.se}
        return pd.DataFrame(rows).T

    # ---- persistence
    def save(self, path: str | Path) -> Path:
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        self.season_metrics.to_csv(d / "metrics_by_season.csv")
        self.players.to_parquet(d / "players.parquet", index=False)
        (d / "run.json").write_text(json.dumps(
            {"projector": self.projector, "seasons": self.seasons, "config": self.config,
             "metadata": self.metadata}, indent=2, default=_json_default), encoding="utf-8")
        return d

    @classmethod
    def load(cls, path: str | Path) -> "BacktestResult":
        d = Path(path)
        meta = json.loads((d / "run.json").read_text(encoding="utf-8"))
        sm = pd.read_csv(d / "metrics_by_season.csv", index_col=0)
        sm.index = sm.index.astype(str)
        return cls(meta["projector"], meta["seasons"], sm, pd.read_parquet(d / "players.parquet"),
                   meta["config"], meta["metadata"])


def _DEFAULT_CI_METRICS(ks: Sequence[int]) -> list[str]:
    return (["spearman_total_fp", "spearman_fppg"] + [f"top{k}_hit" for k in ks]
            + [f"ndcg_{max(ks)}", "mae_fppg", "mae_gp", "mae_total_fp", "rmse_total_fp", "bias_total_fp"])


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


# --------------------------------------------------------------------------- walk-forward

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def _history_hash(history) -> str:
    return data_hash({"game_logs": history.game_logs, "team_games": history.team_games,
                      "players": history.players, "player_season_bio": history.player_season_bio,
                      **{f"extras_{k}": v for k, v in history.extras.items()}})


def _describe_replacement(r) -> str:
    if r is None:
        return f"default: best un-rostered player, N={M.default_n_rostered()} rostered (league.yaml)"
    if callable(r):
        return f"callable {getattr(r, '__name__', repr(r))}"
    return f"fixed {float(r):g} FP"


def walk_forward(
    tables: Mapping[str, pd.DataFrame],
    projector: Projector,
    seasons: Sequence[str],
    *,
    scoring: Mapping[str, float] | None = None,
    ks: Sequence[int] = M.DEFAULT_KS,
    min_gp: int = M.DEFAULT_MIN_GP,
    replacement: M.Replacement = None,
    sanitize_players: bool = True,
    check_scoring: bool = True,
    cache_dir: str | Path | None = None,
    leak_check: Sequence[str] = (),
    synthetic: bool = False,
    extra_metadata: Mapping | None = None,
) -> BacktestResult:
    """Run ``projector`` walk-forward over ``seasons`` and score it against realised results.

    ``tables`` is the *full* dict (it must contain the target seasons, that is where actuals
    come from); the projector only ever receives ``History`` objects built from seasons before the
    target.  ``scoring`` defaults to ``config/league.yaml``.  ``leak_check`` lists seasons on which
    to additionally run ``assert_projector_ignores_future`` (3 extra projector runs each).
    ``cache_dir`` caches projections per (projector, fingerprint, history-content hash, season):
    it requires ``projector.fingerprint`` (any string that changes when the model's code or
    parameters change) so a stale cache can never be served silently.
    """
    seasons = list(seasons)
    if not seasons:
        raise ValueError("no seasons to backtest")
    if len(set(seasons)) != len(seasons):
        raise ValueError(f"duplicate seasons in {seasons}")
    for s in seasons:
        season_start(s)
    if seasons != sorted(seasons, key=season_start):
        raise ValueError("seasons must be in chronological order")
    missing_tables = [t for t in HISTORY_TABLES if t not in tables]
    if missing_tables:
        raise BacktestError(f"tables missing {missing_tables}")
    for t in HISTORY_TABLES:
        validate_table(tables[t], t)
    have = set(tables["game_logs"]["season"].unique())
    no_actuals = [s for s in seasons if s not in have]
    if no_actuals:
        raise BacktestError(f"no game_logs for requested seasons {no_actuals}; nothing to score against")
    first_data = min(have, key=season_start)
    if any(season_start(s) <= season_start(first_data) for s in seasons):
        raise BacktestError(
            f"season {[s for s in seasons if season_start(s) <= season_start(first_data)][0]} has no earlier "
            f"history in the data (first data season is {first_data}); start the backtest one season later")

    name = getattr(projector, "name", type(projector).__name__)
    rank_only = bool(getattr(projector, "rank_only", False))
    scoring = dict(scoring if scoring is not None else load_league()["scoring"])
    fingerprint = getattr(projector, "fingerprint", None)
    if cache_dir is not None and not fingerprint:
        raise ValueError(f"projector {name!r} has no `fingerprint` attribute; refusing to cache its "
                         "projections (a stale cache could silently score old model code)")
    cache = Path(cache_dir) if cache_dir is not None else None
    for s in leak_check:
        if s not in seasons:
            raise ValueError(f"leak_check season {s} is not among the backtest seasons")

    rows, frames, timings, leak_results = [], [], {}, {}
    for season in seasons:
        t0 = time.perf_counter()
        history = build_history(tables, season, sanitize=sanitize_players)
        if history.game_logs.empty:
            raise BacktestError(f"empty history for {season}")

        proj = None
        cache_file = None
        if cache is not None:
            cache_file = cache / f"{_slug(name)}__{_slug(str(fingerprint))}__{_history_hash(history)[:16]}__{season}.parquet"
            if cache_file.exists():
                proj = pd.read_parquet(cache_file)
                log.info("%s %s: projections from cache", name, season)
        if proj is None:
            try:
                proj = projector.project(history)
            except BacktestError:
                raise
            except Exception as exc:
                raise BacktestError(f"projector {name!r} raised {type(exc).__name__} for season {season}: {exc}") from exc
            validate_projections(proj, season=season, projector_name=name, rank_only=rank_only,
                                 scoring=scoring, check_scoring=check_scoring)
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                proj.to_parquet(cache_file, index=False)
        else:
            validate_projections(proj, season=season, projector_name=name, rank_only=rank_only,
                                 scoring=scoring, check_scoring=check_scoring)

        if season in leak_check:
            assert_projector_ignores_future(projector, tables, season, sanitize=sanitize_players)
            leak_results[season] = "passed"

        actual = act.season_actuals(tables["game_logs"], season, scoring)
        games = act.game_fp(tables["game_logs"], season, scoring)
        frame = act.build_eval_frame(proj, actual, games, season, rank_only=rank_only)
        m = M.compute_season_metrics(frame, ks=ks, replacement=replacement, min_gp=min_gp, rank_only=rank_only)
        m["season"] = season
        rows.append(m)
        frames.append(frame)
        timings[season] = round(time.perf_counter() - t0, 3)
        log.info("%s %s: spearman_total_fp=%.3f top50=%.2f n_proj=%d coverage_miss=%d", name, season,
                 m["spearman_total_fp"], m.get("top50_hit", float("nan")), m["n_projected"], m["n_coverage_miss"])

    season_metrics = pd.DataFrame(rows).set_index("season")
    players = pd.concat(frames, ignore_index=True)
    config = {
        "scoring": scoring, "ks": list(ks), "min_gp": int(min_gp),
        "replacement": _describe_replacement(replacement), "sanitize_players": bool(sanitize_players),
        "check_scoring": bool(check_scoring), "rank_only": rank_only,
        "metric_conventions": "bias = mean(pred - actual); FPPG metrics need actual_gp >= min_gp; "
                              "zero-game projected players count as 0 total FP",
    }
    metadata = {
        "harness_version": HARNESS_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_hash": data_hash({t: tables[t] for t in HISTORY_TABLES}),
        "projector_class": f"{type(projector).__module__}.{type(projector).__qualname__}",
        "projector_fingerprint": fingerprint,
        "synthetic": bool(synthetic),
        "leak_check": leak_results,
        "seconds_per_season": timings,
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        **(dict(extra_metadata) if extra_metadata else {}),
    }
    return BacktestResult(name, seasons, season_metrics, players, config, metadata)
