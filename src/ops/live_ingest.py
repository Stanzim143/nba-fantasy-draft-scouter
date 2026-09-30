"""Incremental, idempotent ingest of the live season's game logs and team games (ADR 0018).

``python -m src.ingest.nba_stats`` rebuilds a whole season from three season-long pulls; that is right for history and
too heavy (and too blunt) for a nightly job. This module pulls only a **date window** (``DateFrom`` = the newest game
already stored minus a few days of overlap), replaces exactly that window in the season's ``game_logs`` and
``team_games`` and leaves everything else alone. It reuses the ingest's own parameter builders, transforms, validation
and table writer, so a row means exactly what it means in the full ingest.

Properties (tests in ``tests/ops/test_ops_live_ingest.py``):

* **Idempotent**: the same payload merged twice gives identical tables; an unchanged table is not rewritten.
* **Self-healing**: the overlap re-pulls the last days every night, so a game that was missing or corrected upstream, or a
  run that happened mid game-day, is repaired by the next run.
* **Polite**: two requests a night (player and team logs) plus, only when a player appears who is not in ``players`` yet, the
  player index and the season bio pull. The season opener is skipped without any request while nothing can have happened.
* **Final games only**: a game is stored only when both team rows carry a result (``WL``) and the player rows add up to each
  team's points (an exact invariant of every stored season). A game still in progress, or half published, is left out and
  reported as pending; the next night's overlap picks it up. A stored copy of such a game is never removed.
* **Safe**: a window that comes back much smaller than what is stored refuses to write rather than shrinking the tables (the
  message names stored games absent from the pull); a stored game that vanished upstream is never deleted; player rows without
  their team game hold only that game pending (all games orphaned = broken team endpoint = refuse); one rolling ``.prev_nightly`` copy of a table is kept before it is rewritten.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.contracts import data_dir, season_start, table_path, validate_table
from src.ingest import nba_stats as ns
from src.ingest import nba_transform as tf
from src.ingest.nba_client import NBAClient, NBAClientError
from src.store import table_exists, write_table

OVERLAP_DAYS = 3
MAX_BIRTHDATE_FETCH = 80      # CommonPlayerInfo calls (one per new player) in a single night
WINDOW_CACHE_KEEP_DAYS = 14   # raw cache files of the nightly date windows are never re-used; older ones are pruned
WINDOW_ENDPOINTS = ("playergamelogs", "leaguegamelog")
SHRINK_TOLERANCE = 0.8       # a window with fewer than this share of the stored rows for the same dates is refused
MIN_ROWS_TO_GUARD = 8         # smaller stored windows (the first nights of a season) are not guarded
BACKUP_SUFFIX = ".prev_nightly"


class LiveIngestError(RuntimeError):
    pass


@dataclass
class LiveIngestResult:
    season: str
    skipped: str | None = None
    date_from: str | None = None
    last_game_date: str | None = None          # newest game date stored after this run
    new_games: int = 0                         # game ids in the tables that were not there before
    new_player_rows: int = 0
    new_players: int = 0
    tables: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_games: list[str] = field(default_factory=list)   # in the window but not final / not consistent: left for the next run
    vanished_games: list[str] = field(default_factory=list)  # stored in the window but absent from the pull: kept as stored
    network_requests: int = 0
    cache_hits: int = 0


def nba_date(d: date) -> str:
    return d.strftime("%m/%d/%Y")


def windowed(params: dict[str, Any], date_from: date | None, date_to: date | None = None) -> dict[str, Any]:
    out = dict(params)
    if date_from is not None:
        out["DateFrom"] = nba_date(date_from)
    if date_to is not None:
        out["DateTo"] = nba_date(date_to)
    return out


def final_game_ids(team_payload: dict[str, Any], new_tg: pd.DataFrame, new_gl: pd.DataFrame) -> tuple[set[str], set[str]]:
    """``(final, pending)`` game ids among the team games pulled.

    Final = two team rows, each with a result (``WL``), and the player rows of each team add up to that team's points.
    """
    raw = tf.result_set(team_payload)
    played = raw[raw["WL"].notna() & (raw["WL"].astype(str).str.strip() != "")] if "WL" in raw.columns else raw
    with_result = played.groupby(raw.loc[played.index, "GAME_ID"].astype(str).str.strip()).size()
    have_result = set(with_result[with_result >= 2].index)
    sums = new_gl.groupby(["game_id", "team_id"])["pts"].sum()
    ok = set()
    for gid, grp in new_tg.groupby("game_id"):
        if gid not in have_result or len(grp) != 2:
            continue
        if all(int(sums.get((gid, int(t)), -1)) == int(p) for t, p in zip(grp["team_id"], grp["pts_for"])):
            ok.add(gid)
    return ok, set(new_tg["game_id"]) - ok


def replace_window_keep(existing: pd.DataFrame | None, new: pd.DataFrame, season: str, date_from: date | None,
                        sort_cols: list[str], keep_ids: set[str], date_to: date | None = None) -> pd.DataFrame:
    """``replace_window`` that never drops a stored game whose id is in ``keep_ids`` (its fresh copy was not final)."""
    merged = replace_window(existing, new, season, date_from, sort_cols, date_to)
    if not keep_ids or existing is None or existing.empty:
        return merged
    lost = existing[existing["game_id"].isin(keep_ids) & ~existing["game_id"].isin(set(merged["game_id"]))]
    if lost.empty:
        return merged
    return ns._sorted_by_season(pd.concat([merged, lost], ignore_index=True), sort_cols)


def replace_window(existing: pd.DataFrame | None, new: pd.DataFrame, season: str, date_from: date | None,
                   sort_cols: list[str], date_to: date | None = None) -> pd.DataFrame:
    """``existing`` with this season's rows on/after ``date_from`` (and on/before ``date_to``) replaced by ``new``."""
    if existing is None or existing.empty:
        return ns._sorted_by_season(new, sort_cols) if len(new) else new
    dates = pd.to_datetime(existing["game_date"])
    cut = pd.Timestamp(date_from) if date_from is not None else pd.Timestamp.min
    keep = (existing["season"] != season) | (dates < cut)
    if date_to is not None:
        keep = keep | (dates > pd.Timestamp(date_to))
    kept = existing[keep]
    frames = [f for f in (kept, new) if len(f)]
    merged = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].copy()
    return ns._sorted_by_season(merged, sort_cols)


def _backup_then_write(df: pd.DataFrame, name: str, base: Path) -> Path:
    path = table_path(name, base)
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + BACKUP_SUFFIX))
    return write_table(df, name, base)


def prune_window_cache(client: Any, now_ts: float | None = None, keep_days: int = WINDOW_CACHE_KEEP_DAYS) -> int:
    """Delete raw cache files of old nightly date windows (their names carry ``MM-DD-YYYY``; full-season pulls do not)."""
    import re
    import time

    root = getattr(client, "cache_dir", None)
    if root is None:
        return 0
    cutoff = (now_ts or time.time()) - keep_days * 86400
    n = 0
    for endpoint in WINDOW_ENDPOINTS:
        for f in Path(root, endpoint).glob("*.json"):
            if re.match(r"^\d{2}-\d{2}-\d{4}_", f.name) and f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
                n += 1
    return n


def _season_rows(df: pd.DataFrame | None, season: str) -> pd.DataFrame:
    return df[df["season"] == season] if df is not None and len(df) else (df if df is not None else pd.DataFrame())


def run_live_ingest(season: str, client: NBAClient, base: Path | None = None, *, completed_through: date,
                    opening_night: date | None = None, date_to: date | None = None, overlap_days: int = OVERLAP_DAYS, refresh: bool = True,
                    log: Callable[[str], None] = print) -> LiveIngestResult:
    """Pull and merge the live ``season``'s games up to (and a little past) ``completed_through``.

    ``refresh`` bypasses the client's disk cache (the window is new every night); offline runs pass ``False``.
    ``date_to`` bounds the window on the right; a live night leaves it open, a replay of a past night sets it.
    """
    season_start(season)
    base = base or data_dir()
    res = LiveIngestResult(season)
    old_gl = ns._existing("game_logs", base) if table_exists("game_logs", base) else None
    old_tg = ns._existing("team_games", base) if table_exists("team_games", base) else None
    have_tg = _season_rows(old_tg, season)
    last = pd.Timestamp(have_tg["game_date"].max()).date() if len(have_tg) else None
    if last is None and opening_night is not None and completed_through < opening_night:
        res.skipped = f"the season opens {opening_night}; nothing to pull before then"
        log(f"[{season}] {res.skipped}")
        return res
    date_from = None if last is None else last - timedelta(days=overlap_days)
    res.date_from = date_from.isoformat() if date_from else None

    reqs0, hits0 = client.stats.network_requests, client.stats.cache_hits
    log(f"[{season}] pulling games from {res.date_from or 'the start of the season'}")
    gl_payload = client.get("playergamelogs", windowed(ns.player_gamelogs_params(season), date_from, date_to), refresh=refresh)
    tm_payload = client.get("leaguegamelog", windowed(ns.gamelog_params(season, "T"), date_from, date_to), refresh=refresh)
    new_gl, gl_rep = tf.transform_game_logs(gl_payload, season)
    new_tg, tg_rep = tf.transform_team_games(tm_payload, season)
    if date_from is not None:                       # the source honours DateFrom; keep the contract explicit anyway
        cut = pd.Timestamp(date_from)
        new_gl = new_gl[new_gl["game_date"] >= cut].reset_index(drop=True)
        new_tg = new_tg[new_tg["game_date"] >= cut].reset_index(drop=True)
    if date_to is not None:
        top = pd.Timestamp(date_to)
        new_gl = new_gl[new_gl["game_date"] <= top].reset_index(drop=True)
        new_tg = new_tg[new_tg["game_date"] <= top].reset_index(drop=True)

    pulled_ids = set(new_gl["game_id"]) | set(new_tg["game_id"])
    pending: set[str] = set()
    # Player rows whose (game, team) has no team row: that game is half published (or the team endpoint lags). Hold only
    # those games back, like a game with no WL. If EVERY game in the window is like that the team endpoint is broken: fail.
    if len(new_gl):
        team_keys = set(zip(new_tg["game_id"], new_tg["team_id"])) if len(new_tg) else set()
        orphan_games = {g for g, t in zip(new_gl["game_id"], new_gl["team_id"]) if (g, t) not in team_keys}
        if orphan_games and orphan_games >= pulled_ids:
            raise LiveIngestError(f"all {len(orphan_games)} game(s) in the pulled window have player rows but no team rows; refusing to "
                                  "write (is the team-game endpoint down or empty? the next run repairs it)")
        if orphan_games:
            log(f"[{season}] {len(orphan_games)} game(s) have player rows but no team rows yet, left for the next run: {sorted(orphan_games)[:6]}")
            pending |= orphan_games
            new_gl = new_gl[~new_gl["game_id"].isin(orphan_games)].reset_index(drop=True)
            new_tg = new_tg[~new_tg["game_id"].isin(orphan_games)].reset_index(drop=True)
    if len(new_tg):
        final, not_final = final_game_ids(tm_payload, new_tg, new_gl)
        if not_final:
            log(f"[{season}] {len(not_final)} game(s) not final or not consistent yet, left for the next run: {sorted(not_final)[:6]}")
            pending |= not_final
            new_tg = new_tg[new_tg["game_id"].isin(final)].reset_index(drop=True)
            new_gl = new_gl[new_gl["game_id"].isin(final)].reset_index(drop=True)
    res.pending_games = sorted(pending)
    cons = tf.check_consistency(new_gl, new_tg) if len(new_gl) and len(new_tg) else {"orphan_player_rows": 0}
    if cons["orphan_player_rows"]:
        raise LiveIngestError(f"{cons['orphan_player_rows']} player rows reference a (game, team) missing from the team games "
                              "pulled; refusing to write (a half-published game day: the next run repairs it)")
    for label, old, new in (("game_logs", old_gl, new_gl), ("team_games", old_tg, new_tg)):
        if old is None or not len(old):
            continue
        cut = pd.Timestamp(date_from) if date_from is not None else pd.Timestamp.min
        window_old = old[(old["season"] == season) & (pd.to_datetime(old["game_date"]) >= cut)]
        if date_to is not None:
            window_old = window_old[pd.to_datetime(window_old["game_date"]) <= pd.Timestamp(date_to)]
        window_old = window_old[~window_old["game_id"].isin(pending)]        # games held back as not final are not "missing"
        if len(window_old) >= MIN_ROWS_TO_GUARD and len(new) < SHRINK_TOLERANCE * len(window_old):
            gone = sorted(set(window_old["game_id"]) - pulled_ids)
            hint = (f" {len(gone)} stored game(s) are absent from the pull altogether ({gone[:6]}); stored games are never deleted, "
                    "so a game that vanished upstream keeps this guard tripped only until the window moves past it (overlap days), "
                    "or run the full ingest." if gone else "")
            raise LiveIngestError(f"{label}: the pulled window has {len(new)} rows but {len(window_old)} are stored for the same "
                                  f"dates; refusing to shrink the table (partial upstream response?).{hint}")

    keep = set(pending)
    if old_tg is not None and len(old_tg):          # a stored game that vanished from the pull is kept, never deleted
        cut = pd.Timestamp(date_from) if date_from is not None else pd.Timestamp.min
        stored = old_tg[(old_tg["season"] == season) & (pd.to_datetime(old_tg["game_date"]) >= cut)]
        if date_to is not None:
            stored = stored[pd.to_datetime(stored["game_date"]) <= pd.Timestamp(date_to)]
        res.vanished_games = sorted(set(stored["game_id"]) - pulled_ids)
        if res.vanished_games:
            log(f"[{season}] {len(res.vanished_games)} stored game(s) absent from the pull, kept as stored: {res.vanished_games[:6]}")
            keep |= set(res.vanished_games)
    game_logs = replace_window_keep(old_gl, new_gl, season, date_from, ["game_date", "game_id", "team_id", "player_id"], keep, date_to)
    team_games = replace_window_keep(old_tg, new_tg, season, date_from, ["game_date", "game_id", "team_id"], keep, date_to)
    if len(_season_rows(game_logs, season)) == 0 or len(_season_rows(team_games, season)) == 0:
        res.skipped = "the source has no regular-season games for this season yet"
        res.network_requests = client.stats.network_requests - reqs0
        res.cache_hits = client.stats.cache_hits - hits0
        log(f"[{season}] {res.skipped}")
        return res
    validate_table(game_logs, "game_logs")
    validate_table(team_games, "team_games")

    before_ids = set(have_tg["game_id"]) if len(have_tg) else set()
    now_ids = set(_season_rows(team_games, season)["game_id"])
    res.new_games = len(now_ids - before_ids)
    old_rows = len(_season_rows(old_gl, season))
    res.new_player_rows = len(_season_rows(game_logs, season)) - old_rows
    res.last_game_date = pd.Timestamp(_season_rows(team_games, season)["game_date"].max()).date().isoformat()

    # ---- players / season bio: only when a player shows up who is not in the players table yet
    old_players = ns._existing("players", base) if table_exists("players", base) else None
    season_ids = set(_season_rows(game_logs, season)["player_id"])
    known = set(old_players["player_id"]) if old_players is not None else set()
    missing = season_ids - known
    players = old_players
    bio = ns._existing("player_season_bio", base) if table_exists("player_season_bio", base) else None
    if missing or old_players is None:
        log(f"[{season}] {len(missing)} new player(s); refreshing player index and season bio")
        index_payload = client.get("playerindex", ns.player_index_params(season), refresh=refresh)
        bio_payload = client.get("leaguedashplayerbiostats", ns.bio_stats_params(season), refresh=refresh)
        bio_frame = tf.bio_stats_frame(bio_payload)
        birth = {}
        if old_players is not None:
            birth = {int(p): b for p, b in zip(old_players["player_id"], old_players["birthdate"]) if pd.notna(b)}
        extra_rows: list[dict[str, Any]] = []
        for pid in sorted(missing)[:MAX_BIRTHDATE_FETCH]:      # exact birthdates as the full ingest has them (ages depend on it)
            try:
                row = tf.player_attributes_from_common_player_info(client.get("commonplayerinfo", ns.common_player_info_params(pid)))
            except (NBAClientError, tf.TransformError) as exc:
                log(f"[{season}] no CommonPlayerInfo for player {pid} ({type(exc).__name__}); birthdate stays unknown")
                continue
            extra_rows.append(row)
            if row.get("birthdate") is not None:
                birth[pid] = row["birthdate"]
        extra = pd.DataFrame(extra_rows) if extra_rows else None
        players_new = tf.build_players(game_logs, tf.player_index_frame(index_payload), {season: bio_frame}, birth, extra)
        players = ns.merge_players(old_players, players_new)
        validate_table(players, "players")
        run_logs = _season_rows(game_logs, season)
        run_birth = {pid: bd for pid, bd in birth.items() if pid in season_ids}
        bio_new, _ = tf.build_player_season_bio(run_logs, run_birth, {season: bio_frame})
        bio = ns.merge_seasons(bio, bio_new, [season], ["player_id"])
        validate_table(bio, "player_season_bio")
        res.new_players = len(missing)

    for name, df, old in (("game_logs", game_logs, old_gl), ("team_games", team_games, old_tg),
                          ("players", players, old_players),
                          ("player_season_bio", bio, ns._existing("player_season_bio", base) if table_exists("player_season_bio", base) else None)):
        if df is None:
            continue
        changed = not ns.frames_equal(old, df)
        path = _backup_then_write(df, name, base) if changed else table_path(name, base)
        res.tables[name] = {"rows": len(df), "changed": changed, "path": str(path)}
        log(f"  {name}: {len(df):,} rows ({'written' if changed else 'unchanged'})")
    res.network_requests = client.stats.network_requests - reqs0
    res.cache_hits = client.stats.cache_hits - hits0
    if refresh:
        try:
            prune_window_cache(client)
        except OSError:
            pass
    del gl_rep, tg_rep
    return res
