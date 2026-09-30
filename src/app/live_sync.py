"""Draft-day live sync: poll the ESPN league sync for newly drafted players and reconcile them
onto the in-app draft state, instead of requiring a manual "Mark drafted" click for every
opponent pick. See ``docs/adr/0025-draft-live-sync.md`` for the design decisions (polling vs
push, conflict resolution between manual and auto marks, failure modes).

No Streamlit import here on purpose, same convention as ``src.app.state``/``src.app.loader``:
this module is plain, unit-testable functions and dataclasses (``tests/app/test_live_sync.py``).
``draft_board.py`` owns the ``st.session_state`` wiring and renders the status indicator; nothing
in here decides *how* it is displayed.

Design in one paragraph
------------------------
Streamlit re-runs the whole script on every interaction and has no background thread of its own,
so "live" here means "poll ESPN's draft-detail endpoint again, on demand or on the next rerun,
never faster than a floor interval" -- not a push subscription (ESPN's fantasy API exposes none).
Each poll asks :func:`sync_once` for every pick with a filled ``espn_player_id`` that was not
already seen this session (:func:`diff_new_picks`), maps it from ESPN's player id onto this
project's own ``player_id`` via the ``player_id_map`` contract table (built by
``src.ingest.espn_league``'s sibling ``src.ingest.espn_adp``), attributes it to "me" or
"opponent" from the league's team id, and applies it through the exact same ``draft_player``
function the manual form uses (:func:`apply_detected_picks`), so a duplicate -- the same player
already marked by hand -- is simply skipped rather than double-counted. Any failure (network,
auth, an un-ingested id map, ESPN rate-limiting) degrades to a returned error rather than raising:
the manual "Mark drafted" control is never disabled and is always the fallback.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from src.app.state import ME, OPPONENT, DraftError, DraftState, draft_player
from src.ingest.espn_league import ESPNLeagueClient, ESPNLeagueError, fetch_draft_detail, parse_draft
from src.ops.runlock import LockBusy, RunLock

#: Floor on how often the app itself will re-poll ESPN, independent of (and >=) the
#: ``ESPNLeagueClient``'s own ``min_interval``/backoff, which only governs a single request burst.
DEFAULT_MIN_POLL_INTERVAL = 20.0

#: A poll lock older than this is assumed to belong to a crashed/closed session and is reclaimed,
#: so a stale lock can never block a live draft indefinitely. Much shorter than ``RunLock``'s own
#: 3-hour default (built for a once-a-day batch job): a live-draft poll is expected roughly every
#: ``DEFAULT_MIN_POLL_INTERVAL`` seconds.
LOCK_STALE_SECONDS = 60.0

#: Settings files ``src.ops.nightly``/``src.ops.daily_refresh`` already use to persist "which ESPN
#: team is mine" (``python -m src.ops.nightly --set-team N``). Read-only here, so a user who set
#: their team after a previous draft never has to enter it again for this app.
_TEAM_ID_SETTINGS_FILES = ("nightly.json", "daily_refresh.json")


def load_dotenv_file(repo_root: Path) -> bool:
    """Load ``<repo_root>/.env`` into ``os.environ`` without overriding variables already set, so
    ``streamlit run`` picks up ``ESPN_LEAGUE_ID``/``ESPN_TEAM_ID`` the same as the CLIs do. A missing
    file or a missing ``python-dotenv`` just means "nothing loaded"; returns whether a file was read."""
    env_file = Path(repo_root) / ".env"
    if not env_file.is_file():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return bool(load_dotenv(env_file, override=False))


def league_id_from_env(env: Mapping[str, str] | None = None) -> int | None:
    val = (env if env is not None else os.environ).get("ESPN_LEAGUE_ID", "").strip()
    return int(val) if val else None


def team_id_from_env(env: Mapping[str, str] | None = None) -> int | None:
    """The ESPN ``team_id`` that identifies "me" for pick attribution, from ``$ESPN_TEAM_ID`` --
    the same env var ``src.ops.nightly``/``src.inseason`` already use (see CONTRIBUTING.md's env
    var conventions; never committed)."""
    val = (env if env is not None else os.environ).get("ESPN_TEAM_ID", "").strip()
    return int(val) if val else None


def team_id_from_settings(data_dir: Path | None = None) -> int | None:
    """Fallback for :func:`team_id_from_env`: the ``team_id`` already saved by
    ``python -m src.ops.nightly --set-team`` into ``<data_dir>/nightly.json`` (or the older
    ``daily_refresh.json``). Never raises: a missing/corrupt file just means "not configured"."""
    import json

    from src.contracts import data_dir as default_data_dir

    base = data_dir or default_data_dir()
    for name in _TEAM_ID_SETTINGS_FILES:
        path = base / name
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        tid = data.get("team_id")
        if tid in (None, ""):
            continue
        try:
            return int(tid)
        except (TypeError, ValueError):
            continue
    return None


def default_team_id(env: Mapping[str, str] | None = None, data_dir: Path | None = None) -> int | None:
    """``$ESPN_TEAM_ID`` if set, else whatever ``--set-team`` last saved, else ``None`` (every
    detected pick is then attributed to the opponent side -- see :func:`classify_team`)."""
    env_val = team_id_from_env(env)
    return env_val if env_val is not None else team_id_from_settings(data_dir)


# --------------------------------------------------------------------------- pure diff / attribution

@dataclass(frozen=True)
class DetectedPick:
    overall_pick: int | None
    round: int | None
    espn_player_id: int
    team_id: int | None
    drafted_by: str  # ME or OPPONENT, see classify_team
    player_id: int | None  # resolved internal player_id, or None if id_map can't place it yet
    name: str | None
    position: str | None


def classify_team(team_id: int | None, my_team_id: int | None) -> str:
    """"me" only when ``team_id`` matches the configured ``my_team_id``; every other team
    (including an unknown/unset ``my_team_id``) is "opponent" -- the safer default, since the
    manual form is still how the user's own picks normally get logged in real time."""
    if my_team_id is not None and team_id == my_team_id:
        return ME
    return OPPONENT


def resolve_player(espn_player_id: int, id_map: Mapping[str, int] | None, *,
                   board_names: Mapping[int, str] | None = None,
                   board_positions: Mapping[int, str] | None = None,
                   ) -> tuple[int | None, str | None, str | None]:
    """``(player_id, name, position)`` for an ESPN player id, or ``(None, None, None)`` if the
    ``player_id_map`` table has no entry for it yet (an unresolved pick is still surfaced to the
    user -- see :func:`apply_detected_picks` -- never silently dropped)."""
    if not id_map:
        return None, None, None
    pid = id_map.get(str(espn_player_id))
    if pid is None:
        return None, None, None
    name = (board_names or {}).get(pid)
    position = (board_positions or {}).get(pid)
    return pid, name, position


def diff_new_picks(picks: list[dict], seen_overall_picks: frozenset[int]) -> list[dict]:
    """Picks from ``parse_draft()["picks"]`` that are filled (``espn_player_id`` is not ``None``)
    and were not already seen this session, in draft order. Pure and order-stable so repeated
    polls of an unchanged draft always produce an empty list."""
    out = [p for p in picks
          if p.get("espn_player_id") is not None and p.get("overall_pick") not in seen_overall_picks]
    out.sort(key=lambda p: (p.get("overall_pick") is None, p.get("overall_pick")))
    return out


def build_detected_picks(new_picks: list[dict], *, my_team_id: int | None,
                         id_map: Mapping[str, int] | None = None,
                         board_names: Mapping[int, str] | None = None,
                         board_positions: Mapping[int, str] | None = None) -> list[DetectedPick]:
    out = []
    for p in new_picks:
        pid, name, position = resolve_player(p["espn_player_id"], id_map, board_names=board_names,
                                             board_positions=board_positions)
        out.append(DetectedPick(
            overall_pick=p.get("overall_pick"), round=p.get("round"),
            espn_player_id=p["espn_player_id"], team_id=p.get("team_id"),
            drafted_by=classify_team(p.get("team_id"), my_team_id),
            player_id=pid, name=name, position=position,
        ))
    return out


def apply_detected_picks(
    state: DraftState, detected: list[DetectedPick],
) -> tuple[DraftState, list[DetectedPick], list[DetectedPick], list[DetectedPick]]:
    """Apply resolved picks onto ``state`` through the same ``draft_player`` the manual form uses.

    Returns ``(new_state, applied, duplicate, unresolved)``:

    * ``applied`` -- newly marked drafted.
    * ``duplicate`` -- already drafted in ``state`` (typically because the user's own manual
      "Mark drafted" click already logged this player, or a previous poll already applied it) --
      skipped so a pick is never double-counted regardless of which path saw it first.
    * ``unresolved`` -- ``espn_player_id`` has no ``player_id_map`` entry yet; returned so the
      caller can tell the user to mark that player by hand rather than silently dropping the pick.
    """
    applied: list[DetectedPick] = []
    duplicate: list[DetectedPick] = []
    unresolved: list[DetectedPick] = []
    for d in detected:
        if d.player_id is None:
            unresolved.append(d)
            continue
        if d.player_id in state.drafted_ids:
            duplicate.append(d)
            continue
        try:
            state = draft_player(state, d.player_id, d.name or f"ESPN player {d.espn_player_id}",
                                 d.position, d.drafted_by)
        except DraftError:
            duplicate.append(d)  # lost a race against another apply in the same batch
            continue
        applied.append(d)
    return state, applied, duplicate, unresolved


# --------------------------------------------------------------------------- polling gate / lock

def should_poll(last_synced_at: datetime | None, now: datetime, *,
               min_interval: float = DEFAULT_MIN_POLL_INTERVAL) -> bool:
    """Whether enough time has passed since ``last_synced_at`` to poll ESPN again -- the app-level
    floor that keeps a click-happy user (or an auto-refresh loop) from hammering ESPN, on top of
    the ``ESPNLeagueClient``'s own per-request rate limit/backoff."""
    if last_synced_at is None:
        return True
    return (now - last_synced_at).total_seconds() >= min_interval


def poll_lock(path: Path, *, stale_after: float = LOCK_STALE_SECONDS) -> RunLock:
    """An exclusive, stale-aware lock so at most one session/browser tab polls ESPN for a given
    league at a time -- reusing ``src.ops.runlock.RunLock`` (already this repo's "only one worker
    downloads ... at a time" mechanism for the nightly job) instead of a bespoke one, applied here
    to repeated draft-day polling rather than a single batch run. ``stale_after`` is much shorter
    than ``RunLock``'s own 3-hour default since a live-draft poll is expected roughly every
    ``DEFAULT_MIN_POLL_INTERVAL`` seconds, not once a day.

    Call :meth:`RunLock.acquire`; it raises :class:`~src.ops.runlock.LockBusy` when another
    session already holds it -- callers here catch that and skip the poll (see :func:`sync_once`
    callers in ``draft_board.py``), never blocking or crashing the draft."""
    return RunLock(Path(path), stale_after=stale_after)


# --------------------------------------------------------------------------- sync status + orchestration

@dataclass
class LiveSyncStatus:
    """Session-persisted status for the UI indicator: last synced time, connected/error state."""

    last_synced_at: datetime | None = None
    last_error: str | None = None
    consecutive_failures: int = 0
    seen_overall_picks: frozenset[int] = field(default_factory=frozenset)
    last_skipped_locked: bool = False

    @property
    def connected(self) -> bool:
        """True once at least one poll has succeeded and the most recent one did not error."""
        return self.last_synced_at is not None and self.last_error is None


@dataclass(frozen=True)
class SyncResult:
    ok: bool
    fetched_at: datetime
    error: str | None
    detected: tuple[DetectedPick, ...] = ()
    draft_completed: bool = False
    seen_overall_picks: frozenset[int] = frozenset()
    locked_out: bool = False  # another session currently holds the poll lock; nothing was fetched


def sync_once(client: ESPNLeagueClient, league_id: int, season_id: int, *,
             seen_overall_picks: frozenset[int], my_team_id: int | None,
             id_map: Mapping[str, int] | None = None,
             board_names: Mapping[int, str] | None = None,
             board_positions: Mapping[int, str] | None = None,
             now: datetime | None = None,
             lock_path: Path | None = None) -> SyncResult:
    """Fetch the current draft state and return the picks new since ``seen_overall_picks``.

    Never raises: every ``ESPNLeagueError`` (auth, not-found, offline-cache-miss, rate-limited/
    retries exhausted, ...) is caught and turned into ``SyncResult(ok=False, error=...)`` so a
    caller can show the error in the status indicator and keep drafting with the manual control.
    When ``lock_path`` is given, the fetch is wrapped in :func:`poll_lock`; another session already
    holding it produces ``SyncResult(ok=True, locked_out=True)`` (a benign skip, not an error --
    see the "only one worker downloads ... at a time" convention this mirrors) rather than raising
    or counting as a failure.
    """
    fetched_at = now or datetime.now(timezone.utc)
    lock = poll_lock(lock_path) if lock_path is not None else None
    try:
        if lock is not None:
            lock.acquire()
        # refresh=True: a live poll must never be served the *first* poll's cached snapshot
        # forever (ESPNLeagueClient otherwise caches a request's response on disk indefinitely --
        # see fetch_draft_detail's docstring).
        payload = fetch_draft_detail(client, season_id, league_id, refresh=True)
    except LockBusy:
        return SyncResult(ok=True, fetched_at=fetched_at, error=None,
                          seen_overall_picks=seen_overall_picks, locked_out=True)
    except ESPNLeagueError as exc:
        return SyncResult(ok=False, fetched_at=fetched_at, error=str(exc),
                          seen_overall_picks=seen_overall_picks)
    finally:
        if lock is not None:
            lock.release()
    draft = parse_draft(payload)
    picks = draft.get("picks") or []
    new_picks = diff_new_picks(picks, seen_overall_picks)
    detected = build_detected_picks(new_picks, my_team_id=my_team_id, id_map=id_map,
                                    board_names=board_names, board_positions=board_positions)
    updated_seen = seen_overall_picks | {
        p["overall_pick"] for p in new_picks if p.get("overall_pick") is not None
    }
    return SyncResult(ok=True, fetched_at=fetched_at, error=None, detected=tuple(detected),
                      draft_completed=bool(draft.get("drafted")), seen_overall_picks=updated_seen)


def load_id_map(data_dir: Path | None = None) -> dict[str, int] | None:
    """``{espn_source_id: player_id}`` from the ``player_id_map`` contract table (built by
    ``python -m src.ingest.espn_adp``), or ``None`` if it has not been ingested yet -- a missing
    map degrades every detected pick to "unresolved" (see :func:`apply_detected_picks`), never a
    crash."""
    from src.store import read_table, table_exists

    if not table_exists("player_id_map", base=data_dir):
        return None
    df = read_table("player_id_map", base=data_dir)
    df = df[df["source"] == "espn"]
    return dict(zip(df["source_id"].astype(str), df["player_id"].astype(int)))
