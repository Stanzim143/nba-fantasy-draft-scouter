"""Breakout watchlist: young, under-the-radar players whose offseason line points above their projection (ADR 0012).

    python -m src.value.breakouts --season 2026-27 --out reports/watchlist.csv [--top 30] [--all-ages]

What it is. The board ranks everyone by projected value; the watchlist re-reads the same projections for one
question: *which young players does the offseason evidence (Summer League, preseason) move the most, and
which of those is the draft market not already paying for?* Each row carries the evidence (his own Summer
League and preseason lines), the model's uplift over the plain baseline, where the market drafts him, and a
calibrated probability that he breaks out.

What it is not. It is not a prediction that a listed player *will* break out. The backtest (ADR 0012,
``python -m src.backtest.breakouts``) measured a real but modest signal: the top ten flagged young
under-the-radar players broke out about 35% of the time against a 21% base rate. Read the probability as
"odds a bit better than a coin-flip-adjusted base rate", never as a lock.

Definitions match the backtest exactly (``src.backtest.breakouts.BreakoutConfig``): a *breakout* is beating
the baseline projection by >= 4 FPPG and >= 25% while playing >= 20 games; *young* is age <= 23; *under the
radar* is no ADP or an ADP rank worse than 100. The probability is conditional on playing 20+ games (see
``proj_gp`` for how likely that is), estimated by a logistic fit on the walk-forward, out-of-sample scores.

Information-matched model. Before any preseason game exists for the season, ``baseline_summer_league`` is
used (the evidence that exists), then ``baseline_offseason`` once preseason games are in. The backtest showed
the preseason carries most of the signal, so the watchlist is materially better after the preseason: re-run it
just before the draft (see ``docs/offseason.md`` for the refresh runbook).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.backtest.breakouts import DEFAULT_CONFIG, BreakoutConfig
from src.contracts import data_dir as data_dir_default
from src.contracts import season_start

CALIBRATION_TEMPLATE = "breakout_calibration_{model}.json"
FULL_MODEL = "baseline_offseason"
SUMMER_ONLY_MODEL = "baseline_summer_league"


# --------------------------------------------------------------------------- calibrated probability

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def logistic_fit(X: np.ndarray, y: np.ndarray, *, ridge: float = 1e-2, iters: int = 60) -> np.ndarray:
    """Newton-Raphson logistic regression with a small ridge on the non-intercept terms. Column 0 of ``X`` is the intercept."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    pen = np.eye(X.shape[1]) * ridge
    pen[0, 0] = 0.0
    for _ in range(iters):
        p = _sigmoid(X @ beta)
        grad = X.T @ (y - p) - pen @ beta
        hess = (X.T * (p * (1 - p))) @ X + pen + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


TARGETS = ("breakout", "useful")


@dataclass
class LogisticParams:
    intercept: float
    coef_score: float
    coef_rookie: float
    n: int
    base_rate: float


@dataclass
class BreakoutCalibration:
    """P(outcome | played >= min_gp) for a young player from the model's out-of-sample uplift score, for the
    two targets of the evaluation: ``breakout`` and ``useful`` (a breakout that finished in the rostered pool)."""

    model: str
    seasons: list[str]
    breakout: LogisticParams
    useful: LogisticParams
    config: dict = field(default_factory=dict)

    def probability(self, score, is_rookie, target: str = "breakout") -> np.ndarray:
        if target not in TARGETS:
            raise ValueError(f"unknown target {target!r}; expected one of {TARGETS}")
        q: LogisticParams = getattr(self, target)
        return _sigmoid(q.intercept + q.coef_score * np.asarray(score, float) + q.coef_rookie * np.asarray(is_rookie, float))

    def save(self, base: Path | None = None) -> Path:
        path = calibration_path(self.model, base)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return path


def calibration_path(model: str, base: Path | None = None) -> Path:
    return (base or data_dir_default()) / "processed" / CALIBRATION_TEMPLATE.format(model=model)


def load_calibration(model: str, base: Path | None = None) -> BreakoutCalibration | None:
    path = calibration_path(model, base)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    try:
        return BreakoutCalibration(model=raw["model"], seasons=raw["seasons"], config=raw.get("config", {}),
                                   breakout=LogisticParams(**raw["breakout"]), useful=LogisticParams(**raw["useful"]))
    except (KeyError, TypeError):
        return None  # an older calibration file with a different shape: treated as absent, never guessed at


def fit_calibration(frame: pd.DataFrame, model: str, cfg: BreakoutConfig = DEFAULT_CONFIG) -> BreakoutCalibration:
    """Fit on a backtest evaluation frame (``src.backtest.breakouts.with_flags`` output): young players who played."""
    d = frame[frame["played"] & frame["young"] & frame["score"].notna()]
    if len(d) < 50:
        raise ValueError(f"too few young players ({len(d)}) to calibrate")
    X = np.column_stack([np.ones(len(d)), d["score"].to_numpy(float), d["is_rookie"].to_numpy(float)])
    fitted = {}
    for target in TARGETS:
        y = d[target].to_numpy(float)
        if len(np.unique(y)) < 2:
            raise ValueError(f"only one outcome class for {target!r}; cannot calibrate")
        b = logistic_fit(X, y)
        fitted[target] = LogisticParams(float(b[0]), float(b[1]), float(b[2]), int(len(d)), float(y.mean()))
    return BreakoutCalibration(model=model, seasons=sorted(d["season"].unique(), key=season_start),
                               config={k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
                               **fitted)


# --------------------------------------------------------------------------- model choice

def choose_model(offseason_logs: pd.DataFrame | None, season: str) -> str:
    """``baseline_offseason`` once any preseason game of ``season`` exists, else ``baseline_summer_league``.

    The preseason carries most of the backtested signal; before it exists, a model fitted on Summer League
    alone is the one whose coefficients describe the evidence actually available.
    """
    if offseason_logs is None or offseason_logs.empty:
        return SUMMER_ONLY_MODEL
    has_pre = ((offseason_logs["event_season"] == season) & (offseason_logs["context"] == "preseason")).any()
    return FULL_MODEL if has_pre else SUMMER_ONLY_MODEL


# --------------------------------------------------------------------------- watchlist

LINE_COLUMNS = ["sl_gp", "sl_mpg", "sl_fp36", "sl_z", "pre_gp", "pre_mpg", "pre_fp36", "pre_z"]
SORT_OPTIONS = ("useful_prob", "breakout_prob", "uplift", "sl_z", "pre_z")


def _line(gp, mpg, z, label: str) -> str:
    if not np.isfinite(gp) or gp <= 0:
        return ""
    zs = f" z{z:+.1f}" if np.isfinite(z) else ""
    return f"{label} {int(gp)}g {mpg:.0f}mpg{zs}"


def build_watchlist(projections: pd.DataFrame, board: pd.DataFrame, players: pd.DataFrame, *,
                    roster: pd.DataFrame | None = None, last_team: pd.Series | None = None,
                    calibration: BreakoutCalibration | None = None, cfg: BreakoutConfig = DEFAULT_CONFIG,
                    young_only: bool = True, under_radar_only: bool = True, min_uplift: float = 0.0,
                    sort_by: str = "useful_prob", select: bool = True) -> pd.DataFrame:
    """Rank players by breakout evidence.

    With ``select=False`` every projected player is returned with the ``young`` / ``under_radar`` flags and no
    filtering, so a caller (the app) can re-filter and re-sort it instantly with :func:`select_watchlist`.

    ``projections`` is a layered (offseason) projections table with ``offseason_adj``; ``board`` its
    ``build_board`` output (rank, vorp, ADP columns when an ADP frame was supplied); ``players`` supplies
    draft slots. ``roster`` (latest ``roster_snapshots`` day) adds current team; ``last_team`` (player_id ->
    last season's team_id) flags a move. Only players whose uplift exceeds ``min_uplift`` are listed (no
    positive evidence, no flag). ``sort_by`` is one of ``useful_prob`` (default; falls back to ``uplift``
    without a calibration), ``breakout_prob``, ``uplift``, ``sl_z`` or ``pre_z`` (raw performance, model-free).
    """
    if sort_by not in SORT_OPTIONS:
        raise ValueError(f"unknown sort_by {sort_by!r}; expected one of {SORT_OPTIONS}")
    p = projections.copy()
    for c in ("offseason_adj", *LINE_COLUMNS):
        if c not in p.columns:
            p[c] = np.nan if c != "offseason_adj" else 0.0
    keep = ["player_id", "age", "is_rookie", "proj_fppg", "proj_gp", "proj_total_fp", "offseason_adj", *LINE_COLUMNS]
    if "n_hist_seasons" in p.columns:
        keep.append("n_hist_seasons")
    b = board[[c for c in ("player_id", "rank", "name", "position", "vorp", "tier", "adp", "adp_gap") if c in board.columns]]
    w = p[keep].merge(b, on="player_id", how="inner")
    w = w.rename(columns={"rank": "board_rank", "proj_fppg": "layer_fppg", "proj_gp": "proj_gp"})
    w["base_fppg"] = w["layer_fppg"] - w["offseason_adj"]
    w["uplift"] = w["offseason_adj"]
    if "adp" in w.columns:
        w["adp_rank"] = w["adp"].rank(method="first")
    else:
        w["adp"] = np.nan
        w["adp_gap"] = np.nan
        w["adp_rank"] = np.nan
    w["under_radar"] = w["adp_rank"].isna() | (w["adp_rank"] > cfg.radar_rank)
    w["young"] = w["age"] <= cfg.young_age

    pl = players.drop_duplicates("player_id").set_index("player_id")
    w["draft_pick"] = w["player_id"].map(pd.to_numeric(pl["draft_number"], errors="coerce"))
    if roster is not None and len(roster):
        r = roster.drop_duplicates("player_id").set_index("player_id")
        w["team"] = w["player_id"].map(r["team_abbr"])
        if last_team is not None:
            prior = w["player_id"].map(last_team)
            now = w["player_id"].map(r["team_id"])
            w["changed_team"] = (prior.notna() & now.notna() & (prior != now))
        else:
            w["changed_team"] = False
    else:
        w["team"] = None
        w["changed_team"] = False

    if calibration is not None:
        w["breakout_prob"] = calibration.probability(w["uplift"].to_numpy(float), w["is_rookie"].to_numpy(bool), "breakout")
        w["useful_prob"] = calibration.probability(w["uplift"].to_numpy(float), w["is_rookie"].to_numpy(bool), "useful")
    else:
        w["breakout_prob"] = np.nan
        w["useful_prob"] = np.nan
    w["evidence"] = [
        " | ".join(x for x in (_line(a, b_, c, "SL"), _line(d, e, f, "pre")) if x) or "no offseason line"
        for a, b_, c, d, e, f in zip(w["sl_gp"], w["sl_mpg"], w["sl_z"], w["pre_gp"], w["pre_mpg"], w["pre_z"])]

    w = w.reset_index(drop=True)
    if not select:
        return w
    return select_watchlist(w, young_only=young_only, under_radar_only=under_radar_only, min_uplift=min_uplift,
                            sort_by=sort_by)


WATCH_COLUMNS = ["watch_rank", "name", "team", "position", "age", "is_rookie", "draft_pick", "board_rank", "adp", "adp_gap",
                 "base_fppg", "layer_fppg", "uplift", "proj_gp", "proj_total_fp", "vorp", "useful_prob", "breakout_prob",
                 "changed_team", "risk_flags", "contract_flag", "sl_z", "pre_z", "evidence", "player_id"]


def select_watchlist(full: pd.DataFrame, *, young_only: bool = True, under_radar_only: bool = True,
                     min_uplift: float = 0.0, sort_by: str = "useful_prob") -> pd.DataFrame:
    """Filter and rank a full watchlist frame (``build_watchlist(select=False)``): cheap enough to run on every click.

    Only players whose uplift exceeds ``min_uplift`` are listed (no positive evidence, no flag). A probability sort
    falls back to ``uplift`` when the frame carries no calibrated probabilities.
    """
    if sort_by not in SORT_OPTIONS:
        raise ValueError(f"unknown sort_by {sort_by!r}; expected one of {SORT_OPTIONS}")
    keep = full["uplift"] > min_uplift
    if young_only:
        keep = keep & full["young"]
    if under_radar_only:
        keep = keep & full["under_radar"]
    w = full[keep]
    calibrated = bool(full["useful_prob"].notna().any())
    sort_key = sort_by if (calibrated or sort_by not in ("useful_prob", "breakout_prob")) else "uplift"
    w = w.sort_values([sort_key, "uplift"], ascending=False, kind="mergesort", na_position="last").reset_index(drop=True)
    w.insert(0, "watch_rank", np.arange(1, len(w) + 1))
    return w[[c for c in WATCH_COLUMNS if c in w.columns]]


# --------------------------------------------------------------------------- assembly (shared by the CLI and the app)

@dataclass
class WatchlistResult:
    watchlist: pd.DataFrame
    model: str
    notes: list[str] = field(default_factory=list)
    calibrated: bool = False


def assemble_watchlist(season: str, *, model: str | None = None, data_dir: Path | None = None, teams: int | None = None,
                       adp_path: Path | None = None, select: bool = True, cfg: BreakoutConfig = DEFAULT_CONFIG,
                       **select_kwargs) -> WatchlistResult:
    """Everything the watchlist needs, from the data store: history, the information-matched projection, the board,
    ADP, roster snapshot and calibration. Raises ``FileNotFoundError`` when the NBA history is not ingested; every other
    missing input degrades to a note (and a blank column) instead of an error.

    ``select=False`` returns the full unfiltered frame (see :func:`build_watchlist`) for callers that re-filter it.
    """
    from src.backtest.errors import BacktestError
    from src.contracts import HISTORY_TABLES, History
    from src.ingest.nba_incoming import latest_snapshot, read_roster_snapshots
    from src.ingest.nba_offseason import read_offseason_logs, read_offseason_team_games
    from src.models.panel import season_lengths, target_season_games
    from src.models.registry import get_projector
    from src.store import load_tables
    from src.value.board import build_board, load_adp_for_board

    notes: list[str] = []
    tables = dict(load_tables(HISTORY_TABLES, base=data_dir))
    logs = None
    try:
        logs = read_offseason_logs(data_dir)
        tables["offseason_logs"], tables["offseason_team_games"] = logs, read_offseason_team_games(data_dir)
    except FileNotFoundError:
        notes.append("No offseason tables: run `python -m src.ingest.nba_offseason`. Without them the list is empty.")
    history = History.until(tables, season)
    chosen = model or choose_model(logs, season)
    if model is None:
        notes.append(f"Model {chosen}: " + ("preseason games exist, so the full model is used." if chosen == FULL_MODEL
                     else "only Summer League exists so far; the preseason carries most of the signal, re-run after it."))
    layered = get_projector(chosen).project(history)

    adp = None
    adp_file = adp_path or (data_dir or data_dir_default()) / "processed" / "adp.parquet"
    if Path(adp_file).exists():
        try:
            adp, _ = load_adp_for_board(Path(adp_file), season, data_dir)
        except (FileNotFoundError, BacktestError) as exc:
            notes.append(f"ADP could not be used ({exc}); every player is treated as unpriced.")
    else:
        notes.append("No ADP file: every player is treated as unpriced.")
    games = target_season_games(season_lengths(history, None)) if len(history.team_games) else 82
    board = build_board(layered, history.players, teams=teams, adp=adp, season_games=games)

    roster = None
    try:
        roster = latest_snapshot(read_roster_snapshots(data_dir))
    except FileNotFoundError:
        notes.append("No roster snapshot: team left blank (run `python -m src.ingest.nba_incoming`).")
    calibration = load_calibration(chosen, data_dir)
    if calibration is None:
        notes.append(f"No calibration for {chosen}: ranking by uplift, probabilities blank "
                     "(run `python -m src.backtest.breakouts --save-calibration`).")
    risk_cols = None
    try:
        from src.value.risk import compute_risk, season_is_live

        if season_is_live(season):
            overlay, risk_notes = compute_risk(layered, board, history, season, data_dir)
            risk_cols = overlay[["player_id", "risk_level", "risk_flags"]]
            notes.extend(risk_notes)
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        notes.append(f"draft-day risk flags unavailable ({exc})")
    wl = build_watchlist(layered, board, history.players, roster=roster,
                         last_team=last_season_team(history.player_season_bio), calibration=calibration, cfg=cfg,
                         select=select, **select_kwargs)
    if risk_cols is not None:
        wl = wl.merge(risk_cols, on="player_id", how="left")
        wl["risk_flags"] = wl["risk_flags"].fillna("")
    try:   # ADR 0019: display-only, unvalidated contract flags; the ranking columns are untouched
        from src.value.contract_flags import compute_contract_flags

        cf, cf_notes = compute_contract_flags(board, history, season, data_dir)
        wl = wl.merge(cf[["player_id", "contract_flag"]], on="player_id", how="left")
        wl["contract_flag"] = wl["contract_flag"].fillna("")
        wl["contract_validation"] = "unvalidated_display_only"   # ADR 0019: the flag is display only
        notes.extend(cf_notes)
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        notes.append(f"contract flags unavailable ({exc})")
    return WatchlistResult(wl, chosen, notes, calibration is not None)


def last_season_team(player_season_bio: pd.DataFrame) -> pd.Series:
    """player_id -> team_id at the end of his most recent completed season."""
    b = player_season_bio.dropna(subset=["team_id"]).copy()
    b["s"] = b["season"].map(season_start)
    last = b.sort_values(["player_id", "s"]).groupby("player_id").tail(1)
    return pd.Series(last["team_id"].astype(int).to_numpy(), index=last["player_id"].to_numpy())


# --------------------------------------------------------------------------- CLI

def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.value.breakouts", description=__doc__.split("\n\n")[0])
    ap.add_argument("--season", required=True, help="target season, e.g. 2026-27")
    ap.add_argument("--model", default=None, help="offseason projector (default: chosen from the data available)")
    ap.add_argument("--out", type=Path, default=None, help="CSV path to write")
    ap.add_argument("--adp", type=Path, default=None, help="raw ingested ADP table or a player_id/adp CSV (default: adp.parquet)")
    ap.add_argument("--teams", type=int, default=None, help="override league.teams")
    ap.add_argument("--top", type=int, default=25, help="rows to print")
    ap.add_argument("--sort", choices=SORT_OPTIONS, default="useful_prob",
                    help="ranking: calibrated probability (default), raw uplift, or raw Summer League / preseason z-score")
    ap.add_argument("--min-uplift", type=float, default=0.0, help="list only players whose uplift exceeds this (FPPG)")
    ap.add_argument("--all-ages", action="store_true", help="do not restrict to young players")
    ap.add_argument("--include-priced", action="store_true", help="do not restrict to under-the-radar players")
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    try:
        result = assemble_watchlist(args.season, model=args.model, data_dir=args.data_dir, teams=args.teams,
                                    adp_path=args.adp, young_only=not args.all_ages,
                                    under_radar_only=not args.include_priced, min_uplift=args.min_uplift,
                                    sort_by=args.sort)
    except FileNotFoundError as e:
        _safe_print(f"error: {e}")
        return 2
    for note in result.notes:
        _safe_print(f"note: {note}")
    wl = result.watchlist
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        wl.to_csv(args.out, index=False, encoding="utf-8")
    _safe_print(f"breakout watchlist for {args.season} ({result.model}): {len(wl)} players"
                + (f" -> {args.out}" if args.out else ""))
    _safe_print(wl.drop(columns=["player_id"]).head(args.top).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
