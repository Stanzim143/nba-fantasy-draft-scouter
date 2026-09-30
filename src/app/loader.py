"""Board loading for the app: wraps ``src.value.board.build_board`` and the model registry with
plain-English error handling, so the app can show a message instead of a traceback.

No Streamlit import here on purpose — this module is a plain function, unit-testable without a
running app (see ``tests/app/test_loader.py``). It reuses the exact board-building logic the CLI
(``python -m src.value.board``) uses; it does not reimplement any of it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.contracts import HISTORY_TABLES, History, season_start
from src.models.panel import season_lengths, target_season_games
from src.models.registry import available_projectors, get_projector
from src.value.board import build_board
from src.value.vorp import POSITIONAL_MODES


class BoardUnavailable(Exception):
    """Raised when a board cannot be built for the given inputs. The message is user-facing and
    safe to show verbatim in the app."""


def _load_history(season: str, synthetic: bool, data_dir: Path | None) -> History:
    if synthetic:
        from src.synthetic import make_synthetic_tables

        start = season_start(season)
        tables = make_synthetic_tables(first_start=start - 6, last_start=start - 1, n_teams=14,
                                       games_per_team=60, seed=0)
    else:
        from src.store import load_tables

        tables = load_tables(HISTORY_TABLES, base=data_dir)
    return History.until(tables, season)


def _load_adp(season: str, data_dir: Path | None) -> tuple[pd.DataFrame | None, str | None]:
    """The season's ADP mapped to ``player_id``, or ``(None, reason)``. ADP is an extra: a missing or unusable file
    never blocks the board, it only leaves the ``adp`` / ``adp_gap`` columns out."""
    from src.backtest.errors import BacktestError
    from src.contracts import data_dir as default_dir
    from src.value.board import load_adp_for_board

    path = (data_dir or default_dir()) / "processed" / "adp.parquet"
    if not path.exists():
        return None, "no ADP file ingested (python -m src.ingest.espn_adp), so the board has no ADP columns"
    try:
        frame, _ = load_adp_for_board(path, season, data_dir)
    except (FileNotFoundError, BacktestError, KeyError, ValueError) as exc:
        return None, f"ADP could not be used ({exc})"
    if frame.empty:
        return None, f"the ADP file has no rows for {season}"
    return frame, None


def _with_risk(board: pd.DataFrame, proj: pd.DataFrame, history: History, season: str, data_dir: Path | None) -> pd.DataFrame:
    """Attach the draft-day risk overlay (ADR 0016) during the season's live window; the overlay is extra, never fatal."""
    from src.value.risk import attach_risk, compute_risk, season_is_live

    if not season_is_live(season):
        return board
    try:
        overlay, notes = compute_risk(proj, board, history, season, data_dir)
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        board.attrs["risk_notes"] = [f"risk overlay unavailable ({exc})"]
        return board
    out = attach_risk(board, overlay)
    out.attrs["risk_notes"] = notes
    return out


def _with_contract_flags(board: pd.DataFrame, history: History, season: str, data_dir: Path | None) -> pd.DataFrame:
    """Attach the display-only, unvalidated contract flags (ADR 0019); extra, never fatal, never changes a rank."""
    from src.value.contract_flags import attach_contract_flags, compute_contract_flags

    try:
        overlay, notes = compute_contract_flags(board, history, season, data_dir)
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        board.attrs["risk_notes"] = [*board.attrs.get("risk_notes", []), f"contract flags unavailable ({exc})"]
        return board
    out = attach_contract_flags(board, overlay)
    out.attrs["risk_notes"] = [*board.attrs.get("risk_notes", []), *notes]
    return out


def _with_return_flag(board: pd.DataFrame, history: History, season_games: float) -> pd.DataFrame:
    """Attach the advisory return-from-absence flag (ADR 0023); extra, never fatal, never changes a projection or a rank."""
    from src.value.return_flag import attach_return_flag, compute_return_flag

    try:
        overlay, notes = compute_return_flag(board, history, season_games=season_games)
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        board.attrs["risk_notes"] = [*board.attrs.get("risk_notes", []), f"return flag unavailable ({exc})"]
        return board
    out = attach_return_flag(board, overlay)
    out.attrs["risk_notes"] = [*board.attrs.get("risk_notes", []), *notes]
    return out


def _with_load_flag(board: pd.DataFrame, history: History, season_games: float) -> pd.DataFrame:
    """Attach the advisory short-absence flag (ADR 0024); extra, never fatal, never changes a projection or a rank."""
    from src.value.load_flag import attach_load_flag, compute_load_flag

    try:
        overlay, notes = compute_load_flag(board, history, season_games=season_games)
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        board.attrs["risk_notes"] = [*board.attrs.get("risk_notes", []), f"short-absence flag unavailable ({exc})"]
        return board
    out = attach_load_flag(board, overlay)
    out.attrs["risk_notes"] = [*board.attrs.get("risk_notes", []), *notes]
    return out


def load_board(season: str, model: str = "baseline", *, teams: int | None = None,
               synthetic: bool = False, data_dir: Path | None = None,
               positional: str = "auto") -> pd.DataFrame:
    """Build a ranked draft board for ``season``, or raise ``BoardUnavailable`` with a message
    that is safe to show directly in the UI.

    ``synthetic=True`` uses the deterministic synthetic league (no real data needed) — useful as
    a demo / smoke-test path when real data has not been ingested yet. With real data the board also carries
    ``adp`` and ``adp_gap`` when an ADP file exists (``board.attrs["adp_note"]`` says why when it does not).
    """
    try:
        season_start(season)  # validates the format
    except ValueError as e:
        raise BoardUnavailable(f"'{season}' is not a valid season (expected e.g. '2026-27'): {e}") from e

    if model not in available_projectors():
        raise BoardUnavailable(f"unknown model '{model}'; available models: {available_projectors()}")

    if positional not in POSITIONAL_MODES:
        raise BoardUnavailable(f"positional mode must be one of {POSITIONAL_MODES}")

    try:
        history = _load_history(season, synthetic, data_dir)
    except FileNotFoundError as e:
        raise BoardUnavailable(
            f"real data isn't available for {season}: {e}. Ingest it first with "
            "`python -m src.ingest.nba_stats`, or turn on the synthetic demo mode in the sidebar."
        ) from e

    if len(history.game_logs) == 0 and len(history.team_games) == 0:
        raise BoardUnavailable(
            f"no historical data is available before {season} to project from "
            "(nothing in game_logs/team_games strictly before this season's start)."
        )

    projector = get_projector(model)
    proj = projector.project(history)
    if proj.empty:
        raise BoardUnavailable(f"the '{model}' projector produced no players for {season}.")

    games = target_season_games(season_lengths(history, None)) if len(history.team_games) else 82.0
    adp, adp_note = (None, "synthetic data has no ADP") if synthetic else _load_adp(season, data_dir)
    try:
        board = build_board(proj, history.players, teams=teams, adp=adp, season_games=games, positional=positional)
    except (ValueError, KeyError) as e:
        raise BoardUnavailable(f"could not build the board for {season}: {e}") from e
    board.attrs["adp_note"] = adp_note
    board.attrs["risk_notes"] = []
    if not synthetic:
        board = _with_risk(board, proj, history, season, data_dir)
        board = _with_return_flag(board, history, float(games))
        board = _with_load_flag(board, history, float(games))
        board = _with_contract_flags(board, history, season, data_dir)

    if board.empty:
        raise BoardUnavailable(f"the board for {season} came back empty.")
    return board


def load_watchlist(season: str, *, model: str | None = None, teams: int | None = None, synthetic: bool = False,
                   data_dir: Path | None = None):
    """The full (unfiltered) breakout watchlist for ``season`` as a ``WatchlistResult``.

    The caller filters and sorts it with ``src.value.breakouts.select_watchlist``, so changing a filter in the app never
    re-projects anything. Raises ``BoardUnavailable`` with a user-facing message when it cannot be built.
    """
    from src.value.breakouts import assemble_watchlist

    try:
        season_start(season)
    except ValueError as e:
        raise BoardUnavailable(f"'{season}' is not a valid season (expected e.g. '2026-27'): {e}") from e
    if synthetic:
        raise BoardUnavailable("The breakout watchlist is built from real Summer League and preseason data; it is not "
                               "available in the synthetic demo mode.")
    if model is not None and model not in available_projectors():
        raise BoardUnavailable(f"unknown model '{model}'; available models: {available_projectors()}")
    try:
        return assemble_watchlist(season, model=model, data_dir=data_dir, teams=teams, select=False)
    except FileNotFoundError as e:
        raise BoardUnavailable(f"real data isn't available for {season}: {e}. Ingest it first with "
                               "`python -m src.ingest.nba_stats`.") from e
