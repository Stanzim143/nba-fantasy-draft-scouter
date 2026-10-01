"""ESPN league sync: pull *this project's real league* settings, rosters, free agents, draft and
transactions, and reconcile them against ``config/league.yaml``.

    python -m src.ingest.espn_league --league-id 1234567890 [--season 2026-27] [--offline]

Unlike ``src.ingest.nba_stats`` (the global NBA stats ingest) this module talks to ESPN's
*fantasy* API for **one specific league**. It is a separate, thin ``requests`` client (per
ADR 0005/0008, not the ``espn-api`` package) that follows ``nba_client.py``'s caching pattern
loosely: every response is cached verbatim on disk under ``raw_dir("espn_league")`` and re-served
from there, so a rerun (or ``--offline``) never re-hits the network for the same request.

What it produces
-----------------
A single JSON file, ``data_dir()/processed/espn_league/<league_id>_<espn_season_id>.json`` (path
overridable with ``--out``), shaped like this (all keys always present; empty lists/dicts, not
missing keys, when ESPN has nothing to say -- e.g. before a draft):

``fetched_at``           ISO-8601 UTC timestamp of the pull
``league_id``             the ESPN league id (int)
``season``                our season string, e.g. ``"2026-27"``
``espn_season_id``        the ESPN season id actually used (end year, e.g. 2027)
``is_public``             ``settings.isPublic`` from ESPN (bool)
``settings``              parsed league settings, see ``parse_settings``
``teams``                 list of ``{team_id, name, abbrev, division_id, owner_ids, roster}``;
                           ``roster`` is a list of rostered players (empty before a draft)
``members``               list of ``{id, display_name}`` (ESPN's own pseudonymous display name;
                           real first/last names are deliberately NOT persisted here)
``draft``                 ``{drafted, in_progress, picks}``; ``picks[].espn_player_id`` is
                           ``None`` for an unfilled pick (draft not yet run)
``free_agents``           list of free-agent/waiver players (their own ADP/ownership snapshot);
                           before a draft this is effectively the whole player pool
``transactions``          list of transactions (empty if none have happened yet)
``draft_completed``       convenience bool, ``draft.drafted``
``reconciliation``        list of ``{field, config, espn, match, note}`` rows comparing every
                           ``config/league.yaml`` setting this module can check against the live
                           ESPN value (see ``reconcile``)

This is deliberately NOT one of the ``src.contracts`` ``TableSpec`` tables (it is per-league,
not per-player-season, and mixes several nested shapes); see docs/adr/0008-espn-league-sync.md
for why a lightweight documented JSON shape was chosen instead of forcing it into one.

Safety
------
Read-only. This module only ever issues GET requests. It must never be extended with a
transaction/roster-move/message-posting call. Real ESPN member names are intentionally dropped
during parsing (see ``parse_members``); only ESPN's own pseudonymous ``displayName`` is kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.contracts import data_dir, raw_dir, season_start
from src.ingest.http_cache import _is_network_error
from src.value.league import load_league

BASE_URL = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season_id}"
    "/segments/0/leagues/{league_id}"
)
USER_AGENT = "nba-fantasy-2026-league-sync/0.1 (personal, non-commercial; see docs/adr/0008)"
SOURCE = "espn_league"  # raw_dir source name

# ESPN lineup slot id -> name (docs/research/espn-api-findings.md section 6, verified on-league)
SLOT_NAMES: dict[int, str] = {
    0: "PG", 1: "SG", 2: "SF", 3: "PF", 4: "C", 5: "G", 6: "F", 7: "SG/SF", 8: "G/F",
    9: "PF/C", 10: "F/C", 11: "UTIL", 12: "BE", 13: "IR", 14: "IR2",
}
BENCH_SLOT, IR_SLOT = 12, 13

# ESPN stat id -> config/league.yaml scoring key (espn-api STATS_MAP; verified against our league's
# real scoringItems in docs/adr/0008)
STAT_ID_TO_KEY: dict[int, str] = {
    0: "PTS", 1: "BLK", 2: "STL", 3: "AST", 6: "REB", 11: "TO",
    13: "FGM", 14: "FGA", 15: "FTM", 16: "FTA", 17: "3PM",
}


# --------------------------------------------------------------------------- exceptions

class ESPNLeagueError(RuntimeError):
    """Base class for every error this module raises deliberately."""


class ESPNAuthError(ESPNLeagueError):
    """401/403: the league is not readable without cookies. Never retried automatically."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


class ESPNNotFoundError(ESPNLeagueError):
    """404: no such league/season."""


class ESPNOfflineCacheMiss(ESPNLeagueError):
    """Offline mode is on and the requested response is not in the raw cache."""


class ESPNCacheCorruptError(ESPNLeagueError):
    """A cache file exists but is not valid JSON and we are offline (cannot re-fetch)."""


# --------------------------------------------------------------------------- helpers

def offline_from_env(env: Mapping[str, str] | None = None) -> bool:
    val = (env if env is not None else os.environ).get("NBA_OFFLINE", "")
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def cache_key(views: list[str], params: Mapping[str, Any] | None, headers: Mapping[str, str] | None) -> str:
    """Stable, filesystem-safe file stem for a (views, params, X-Fantasy-Filter) request."""
    canon = {
        "views": sorted(views),
        "params": _canonical(dict(params or {})),
        "filter": (headers or {}).get("X-Fantasy-Filter"),
    }
    digest = hashlib.sha1(json.dumps(canon, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    readable = "-".join(sorted(views))[:60]
    return f"{readable}__{digest}"


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass
class ClientStats:
    cache_hits: int = 0
    network_requests: int = 0
    retries: int = 0
    log: list[dict] = field(default_factory=list)


RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


# --------------------------------------------------------------------------- client

class ESPNLeagueClient:
    """Rate-limited, disk-cached, read-only client for one league's ESPN fantasy endpoints.

    Deliberately does not send ``espn_s2``/``SWID`` cookies: this module is only built and
    verified for a *public* league (see docs/adr/0008). A 401/403 raises :class:`ESPNAuthError`
    immediately -- it is never retried, and this client never attempts to guess credentials.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        *,
        offline: bool | None = None,
        min_interval: float = 2.0,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
        timeout: tuple[float, float] = (10.0, 30.0),
        session: Any = None,
        clock=time.monotonic,
        sleep=time.sleep,
        headers: Mapping[str, str] | None = None,
        base_url: str = BASE_URL,
    ):
        if min_interval < 0 or max_retries < 0:
            raise ValueError("min_interval/max_retries must be >= 0")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else raw_dir(SOURCE)
        self.offline = offline_from_env() if offline is None else offline
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.timeout = timeout
        self._session = session
        self._clock = clock
        self._sleep = sleep
        self.headers = {"User-Agent": USER_AGENT, **(headers or {})}
        self.base_url = base_url
        self._last_request_at: float | None = None
        self.stats = ClientStats()

    # ----- cache

    def cache_path(self, league_id: int, season_id: int, views: list[str],
                   params: Mapping[str, Any] | None = None, headers: Mapping[str, str] | None = None) -> Path:
        key = cache_key(views, params, headers)
        return self.cache_dir / f"{league_id}_{season_id}" / f"{key}.json"

    def _read_cache(self, path: Path) -> dict | None:
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        try:
            return json.loads(raw)
        except ValueError as exc:
            if self.offline:
                raise ESPNCacheCorruptError(f"corrupt cache file {path}: {exc}") from exc
            path.replace(path.with_suffix(".corrupt"))
            return None

    def peek(self, league_id: int, season_id: int, views: list[str], **kw) -> dict | None:
        """Cached payload or ``None``. Never touches the network."""
        return self._read_cache(self.cache_path(league_id, season_id, views, **kw))

    # ----- public API

    def get(self, league_id: int, season_id: int, views: list[str], *,
             params: Mapping[str, Any] | None = None, headers: Mapping[str, str] | None = None,
             refresh: bool = False) -> dict:
        path = self.cache_path(league_id, season_id, views, params, headers)
        if refresh and self.offline:
            raise ESPNOfflineCacheMiss(f"offline mode: cannot refresh {views}")
        cached = None if refresh else self._read_cache(path)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached
        if self.offline:
            raise ESPNOfflineCacheMiss(
                f"offline mode: league {league_id} season {season_id} views={views} is not cached "
                f"at {path}. Run once online (no --offline) to populate {self.cache_dir}."
            )
        body = self._fetch(league_id, season_id, views, params, headers)
        payload = json.loads(body)
        atomic_write_bytes(path, body)
        return payload

    # ----- network

    def _get_session(self):
        if self._session is None:
            import requests  # local import: tests inject a fake session and never need requests

            self._session = requests.Session()
        return self._session

    def _respect_rate_limit(self) -> None:
        if self._last_request_at is None:
            return
        wait = self.min_interval - (self._clock() - self._last_request_at)
        if wait > 0:
            self._sleep(wait)

    def _fetch(self, league_id: int, season_id: int, views: list[str],
               params: Mapping[str, Any] | None, headers: Mapping[str, str] | None) -> bytes:
        url = self.base_url.format(season_id=season_id, league_id=league_id)
        query: dict[str, Any] = {"view": list(views)}
        if params:
            query.update(params)
        hdrs = {**self.headers, **(headers or {})}
        last_problem = "no attempt made"
        for attempt in range(self.max_retries + 1):
            self._respect_rate_limit()
            self._last_request_at = self._clock()
            self.stats.network_requests += 1
            try:
                resp = self._get_session().get(url, params=query, headers=hdrs, timeout=self.timeout)
            except Exception as exc:  # noqa: BLE001 - network layer: timeouts, resets, DNS, TLS ...
                if not _is_network_error(exc):
                    raise
                self.stats.log.append({"views": list(views), "status": None, "attempt": attempt})
                last_problem = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    self.stats.retries += 1
                    self._sleep(min(self.backoff_max, self.backoff_base ** (attempt + 1)))
                    continue
                raise ESPNLeagueError(
                    f"league {league_id} season {season_id} views={views}: network error after "
                    f"{self.max_retries + 1} attempts ({last_problem})") from exc
            status = resp.status_code
            self.stats.log.append({"views": list(views), "status": status, "attempt": attempt})
            if status == 200:
                try:
                    json.loads(resp.content)
                    return resp.content
                except ValueError as exc:    # an HTML/maintenance page served with 200
                    last_problem = f"HTTP 200 with a non-JSON body: {exc}"
                    if attempt < self.max_retries:
                        self.stats.retries += 1
                        self._sleep(min(self.backoff_max, self.backoff_base ** (attempt + 1)))
                        continue
                    raise ESPNLeagueError(
                        f"league {league_id} season {season_id} views={views}: {last_problem} "
                        f"(after {self.max_retries + 1} attempts)") from exc
            if status in (401, 403):
                raise ESPNAuthError(
                    f"league {league_id} season {season_id} views={views}: HTTP {status}. This league is "
                    "not readable without cookies (or ESPN is rejecting anonymous access right now).",
                    status,
                )
            if status == 404:
                raise ESPNNotFoundError(
                    f"league {league_id} season {season_id}: HTTP 404 (no such league/season, or the "
                    "season id is wrong -- ESPN season ids are the END year of the season)"
                )
            if status in RETRYABLE_STATUS and attempt < self.max_retries:
                last_problem = f"HTTP {status}"
                self.stats.retries += 1
                delay = min(self.backoff_max, self.backoff_base ** (attempt + 1))
                self._sleep(delay)
                continue
            snippet = resp.content[:200].decode("utf-8", "replace") if resp.content else ""
            raise ESPNLeagueError(f"league {league_id} season {season_id} views={views}: HTTP {status}: {snippet}")
        raise ESPNLeagueError(f"views={views}: gave up after {self.max_retries + 1} attempts ({last_problem})")


def _limit(v: Any) -> Any:
    """ESPN's "no limit" is a negative number (or absent); a real 0 is a limit of zero, not "unlimited"."""
    return None if v is None or v < 0 else v


# --------------------------------------------------------------------------- fetch helpers

def fetch_settings_teams_rosters(client: ESPNLeagueClient, season_id: int, league_id: int) -> dict:
    """Settings, teams, rosters and standings in one request (views merge server-side)."""
    return client.get(league_id, season_id, ["mSettings", "mTeam", "mRoster", "mStandings"])


def fetch_draft_detail(client: ESPNLeagueClient, season_id: int, league_id: int, *,
                       refresh: bool = False) -> dict:
    """``refresh=True`` bypasses the on-disk cache and re-hits the network even though a cached
    response already exists for this exact (league, season) request -- needed by draft-day live
    sync (``src.app.live_sync``, ADR 0025), which polls this same request repeatedly and must see
    each new pick rather than the first poll's cached snapshot forever. The default stays
    ``False`` so the CLI's own cache-forever-per-request behavior (``ESPNLeagueClient.get``'s
    docstring) is unchanged for every other caller."""
    return client.get(league_id, season_id, ["mDraftDetail"], refresh=refresh)


def fetch_free_agents(client: ESPNLeagueClient, season_id: int, league_id: int, *,
                       scoring_period_id: int = 1, limit: int = 200) -> dict:
    flt = {
        "players": {
            "filterStatus": {"value": ["FREEAGENT", "WAIVERS"]},
            "limit": limit,
            "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
        }
    }
    headers = {"X-Fantasy-Filter": json.dumps(flt)}
    return client.get(league_id, season_id, ["kona_player_info"],
                       params={"scoringPeriodId": scoring_period_id}, headers=headers)


def fetch_transactions(client: ESPNLeagueClient, season_id: int, league_id: int) -> dict:
    flt = {"transactions": {"filterType": {"value": ["FREEAGENT", "WAIVER", "WAIVER_ERROR"]}}}
    headers = {"X-Fantasy-Filter": json.dumps(flt)}
    return client.get(league_id, season_id, ["mTransactions2"], headers=headers)


# --------------------------------------------------------------------------- parsing

def parse_settings(payload: dict) -> dict:
    """Extract the settings this project cares about from an ``mSettings`` (+mTeam) payload."""
    s = payload.get("settings", {}) or {}
    roster = s.get("rosterSettings", {}) or {}
    counts = roster.get("lineupSlotCounts", {}) or {}
    slots = {SLOT_NAMES.get(int(k), f"SLOT_{k}"): v for k, v in counts.items() if v}
    starters = {k: v for k, v in slots.items() if k not in ("BE", "IR", "IR2")}

    scoring: dict[str, float] = {}
    for item in (s.get("scoringSettings", {}) or {}).get("scoringItems", []) or []:
        key = STAT_ID_TO_KEY.get(item.get("statId"))
        if key is not None:
            scoring[key] = item.get("points")

    draft_s = s.get("draftSettings", {}) or {}
    acq_s = s.get("acquisitionSettings", {}) or {}
    trade_s = s.get("tradeSettings", {}) or {}
    sched_s = s.get("scheduleSettings", {}) or {}
    draft_detail = payload.get("draftDetail", {}) or {}

    return {
        "name": s.get("name"),
        "is_public": s.get("isPublic"),
        "size": s.get("size"),
        "team_count": len(payload.get("teams", []) or []) or s.get("size"),
        "scoring_type": (s.get("scoringSettings", {}) or {}).get("scoringType"),
        "scoring": scoring,
        "roster_slots": slots,
        "starters": starters,
        "bench": slots.get("BE", 0),
        "ir": slots.get("IR", 0) + slots.get("IR2", 0),
        # matches config/league.yaml's "roster.size" semantics: starters + bench, IR counted separately
        "roster_size": sum(starters.values()) + slots.get("BE", 0),
        "draft": {
            "type": draft_s.get("type"),
            "order_type": draft_s.get("orderType"),
            "seconds_per_pick": draft_s.get("timePerSelection"),
            "pick_trading": draft_s.get("isTradingEnabled"),
            "pick_order": draft_s.get("pickOrder"),
            "keeper_count": draft_s.get("keeperCount"),
            "drafted": draft_detail.get("drafted"),
            "in_progress": draft_detail.get("inProgress"),
        },
        "acquisition": {
            "type": acq_s.get("acquisitionType"),
            "waiver_hours": acq_s.get("waiverHours"),
            "season_limit": _limit(acq_s.get("acquisitionLimit")),
            "matchup_limit": acq_s.get("matchupAcquisitionLimit"),
            "matchup_limit_per_scoring_period": acq_s.get("matchupLimitPerScoringPeriod"),
        },
        "trades": {
            "limit": _limit(trade_s.get("max")),
            "deadline_epoch_ms": trade_s.get("deadlineDate"),
            "review_hours": trade_s.get("revisionHours"),
            "votes_to_veto": trade_s.get("vetoVotesRequired"),
        },
        "schedule": {
            "matchup_period_count": sched_s.get("matchupPeriodCount"),
            "playoff_team_count": sched_s.get("playoffTeamCount"),
            "playoff_reseed": sched_s.get("playoffReseed"),
            "divisions": sched_s.get("divisions"),
        },
    }


def parse_members(payload: dict) -> list[dict]:
    """ESPN's own pseudonymous display name only -- real first/last names are dropped on purpose."""
    return [{"id": m.get("id"), "display_name": m.get("displayName")} for m in payload.get("members", []) or []]


def _extract_player(entry: dict) -> dict:
    return (entry.get("playerPoolEntry") or {}).get("player") or entry.get("player") or {}


def parse_roster_entries(entries: list[dict]) -> list[dict]:
    out = []
    for e in entries or []:
        player = _extract_player(e)
        out.append({
            "espn_player_id": player.get("id", e.get("playerId")),
            "name": player.get("fullName"),
            "lineup_slot_id": e.get("lineupSlotId"),
            "lineup_slot": SLOT_NAMES.get(e.get("lineupSlotId")),
            "default_position_id": player.get("defaultPositionId"),
            "pro_team_id": player.get("proTeamId"),
            "injury_status": player.get("injuryStatus"),
        })
    return out


def parse_teams(payload: dict) -> list[dict]:
    out = []
    for t in payload.get("teams", []) or []:
        out.append({
            "team_id": t.get("id"),
            "name": t.get("name"),
            "abbrev": t.get("abbrev"),
            "division_id": t.get("divisionId"),
            "owner_ids": t.get("owners", []),
            "roster": parse_roster_entries((t.get("roster") or {}).get("entries", [])),
        })
    return out


def parse_draft(payload: dict) -> dict:
    dd = payload.get("draftDetail", {}) or {}
    picks = []
    for p in dd.get("picks", []) or []:
        player_id = p.get("playerId")
        picks.append({
            "overall_pick": p.get("overallPickNumber"),
            "round": p.get("roundId"),
            "round_pick": p.get("roundPickNumber"),
            "team_id": p.get("teamId"),
            "espn_player_id": None if player_id is None or player_id < 0 else player_id,
            "keeper": p.get("keeper"),
        })
    return {"drafted": dd.get("drafted"), "in_progress": dd.get("inProgress"), "picks": picks}


def parse_free_agents(payload: dict) -> list[dict]:
    out = []
    for e in payload.get("players", []) or []:
        player = e.get("player") or {}
        own = player.get("ownership") or {}
        out.append({
            "espn_player_id": player.get("id", e.get("id")),
            "name": player.get("fullName"),
            "default_position_id": player.get("defaultPositionId"),
            "pro_team_id": player.get("proTeamId"),
            "injury_status": player.get("injuryStatus"),
            "percent_owned": own.get("percentOwned"),
            "adp": own.get("averageDraftPosition"),
            "on_team_id": e.get("onTeamId"),
        })
    return out


def parse_transactions(payload: dict) -> list[dict]:
    out = []
    for t in payload.get("transactions") or []:
        out.append({
            "id": t.get("id"),
            "type": t.get("type"),
            "status": t.get("status"),
            "team_id": t.get("teamId"),
            "scoring_period_id": t.get("scoringPeriodId"),
            "items": t.get("items"),
        })
    return out


# --------------------------------------------------------------------------- reconciliation

def _hours_to_days(hours) -> float | None:
    return None if hours is None else hours / 24.0


def reconcile(cfg: dict, settings: dict) -> list[dict]:
    """Compare every ``config/league.yaml`` field this module can check against the live ESPN
    ``settings`` (as returned by :func:`parse_settings`). Neither source is trusted blindly:
    every row says both values and whether they agree, so a human can see the ground truth."""
    lg = cfg["league"]
    rows: list[dict] = []

    def add(field_name: str, config_val, espn_val, note: str = "") -> None:
        rows.append({"field": field_name, "config": config_val, "espn": espn_val,
                     "match": config_val == espn_val, "note": note})

    add("teams", lg["teams"], settings["team_count"])
    add("roster.size", lg["roster"]["size"], settings["roster_size"])
    add("roster.starters", lg["roster"]["starters"], settings["starters"])
    add("roster.bench", lg["roster"]["bench"], settings["bench"])
    add("roster.ir", lg["roster"]["ir"], settings["ir"])

    add("draft.type", lg["draft"]["type"], (settings["draft"]["type"] or "").lower() or None)
    add("draft.seconds_per_pick", lg["draft"]["seconds_per_pick"], settings["draft"]["seconds_per_pick"])
    add("draft.order", lg["draft"]["order"], (settings["draft"]["order_type"] or "").lower() or None)
    add("draft.pick_trading", lg["draft"]["pick_trading"], settings["draft"]["pick_trading"])

    add("acquisition.season_limit", lg["acquisition"]["season_limit"], settings["acquisition"]["season_limit"])
    add("acquisition.waiver_period_days", lg["acquisition"]["waiver_period_days"],
        _hours_to_days(settings["acquisition"]["waiver_hours"]))
    add("acquisition.matchup_limit", lg["acquisition"]["matchup_limit"], settings["acquisition"]["matchup_limit"],
        note=(f"ESPN represents this as a per-scoring-period cap "
              f"(matchupLimitPerScoringPeriod={settings['acquisition']['matchup_limit_per_scoring_period']}), "
              "not directly the same unit as config's per-matchup-week number; compare with judgement"))

    add("trades.limit", lg["trades"]["limit"], settings["trades"]["limit"])
    add("trades.review_days", lg["trades"]["review_days"], _hours_to_days(settings["trades"]["review_hours"]))
    add("trades.votes_to_veto", lg["trades"]["votes_to_veto"], settings["trades"]["votes_to_veto"])
    espn_deadline = None
    if settings["trades"]["deadline_epoch_ms"] is not None:
        espn_deadline = datetime.fromtimestamp(
            settings["trades"]["deadline_epoch_ms"] / 1000, tz=timezone.utc
        ).date().isoformat()
    add("trades.deadline", str(lg["trades"]["deadline"]), espn_deadline,
        note="epoch ms converted to a UTC date; may be off by one day vs. the league's local timezone")

    keeper_count = settings["draft"]["keeper_count"]
    add("keepers", lg["keepers"], None if keeper_count is None else keeper_count > 0)

    add("schedule.regular_season_matchups", lg["schedule"]["regular_season_matchups"],
        settings["schedule"]["matchup_period_count"])
    add("schedule.playoff_teams", lg["schedule"]["playoff_teams"], settings["schedule"]["playoff_team_count"])
    add("schedule.playoff_reseeding", lg["schedule"]["playoff_reseeding"], settings["schedule"]["playoff_reseed"])

    for key, weight in (cfg.get("scoring") or {}).items():
        add(f"scoring.{key}", weight, settings["scoring"].get(key))

    return rows


# --------------------------------------------------------------------------- report assembly

def build_report(cfg: dict, league_id: int, season: str, season_id: int, *,
                  settings_payload: dict, draft_payload: dict, fa_payload: dict,
                  tx_payload: dict, fetched_at: datetime | None = None) -> dict:
    settings = parse_settings(settings_payload)
    draft = parse_draft(draft_payload)
    return {
        "fetched_at": (fetched_at or datetime.now(timezone.utc)).isoformat(),
        "league_id": league_id,
        "season": season,
        "espn_season_id": season_id,
        "is_public": settings.get("is_public"),
        "settings": settings,
        "teams": parse_teams(settings_payload),
        "members": parse_members(settings_payload),
        "draft": draft,
        "free_agents": parse_free_agents(fa_payload),
        "transactions": parse_transactions(tx_payload),
        "draft_completed": bool(draft.get("drafted")),
        "reconciliation": reconcile(cfg, settings),
    }


# --------------------------------------------------------------------------- CLI

def _safe_print(text: str, *, file=None) -> None:
    """``print`` that never crashes on a non-UTF-8 console (see src.value.board._safe_print)."""
    stream = file or sys.stdout
    try:
        print(text, file=stream)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding), file=stream)


def _print_report(report: dict, out_path: Path) -> None:
    _safe_print(f"\nWrote {out_path}")
    settings = report["settings"]
    _safe_print(f"\nLeague {report['league_id']} ({settings.get('name')!r}), season {report['season']} "
                f"(ESPN seasonId {report['espn_season_id']}): "
                f"{'PUBLIC' if report['is_public'] else 'not flagged public'}, "
                "reachable without cookies.")
    _safe_print(f"Teams: {len(report['teams'])} (ESPN settings.size={settings.get('size')})")
    if not report["draft_completed"]:
        _safe_print("Draft: NOT completed yet -- every roster is empty; free agents below are "
                    "effectively the whole player pool.")
    else:
        _safe_print("Draft: completed.")
    _safe_print(f"Free agents pulled: {len(report['free_agents'])}")
    _safe_print(f"Transactions: {len(report['transactions'])}")

    _safe_print("\nReconciliation: config/league.yaml vs live ESPN settings")
    _safe_print(f"{'field':30s} {'config':>22} {'espn':>22}  match")
    mismatches = [r for r in report["reconciliation"] if not r["match"]]
    for row in report["reconciliation"]:
        mark = "OK" if row["match"] else "DIFFERS"
        line = f"{row['field']:30s} {str(row['config']):>22} {str(row['espn']):>22}  {mark}"
        if row["note"]:
            line += f"   ({row['note']})"
        _safe_print(line)
    if mismatches:
        _safe_print(f"\n{len(mismatches)} discrepancy(ies) found -- see rows marked DIFFERS above.")
    else:
        _safe_print("\nNo discrepancies found.")


def main(argv: list[str] | None = None, *, client: ESPNLeagueClient | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.ingest.espn_league",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--league-id", type=int, default=None,
                    help="ESPN league id (default: $ESPN_LEAGUE_ID)")
    ap.add_argument("--season", default=None, help="e.g. 2026-27 (default: config/league.yaml)")
    ap.add_argument("--offline", action="store_true", help="never touch the network; cache only")
    ap.add_argument("--data-dir", type=Path, default=None, help="data root (default: NBA_DATA_DIR)")
    ap.add_argument("--out", type=Path, default=None, help="override the output JSON path")
    ap.add_argument("--config", type=Path, default=None, help="league.yaml path override")
    ap.add_argument("--free-agent-limit", type=int, default=200)
    args = ap.parse_args(argv)

    league_id = args.league_id if args.league_id is not None else _int_env("ESPN_LEAGUE_ID")
    if league_id is None:
        _safe_print("error: --league-id is required (or set ESPN_LEAGUE_ID in .env)", file=sys.stderr)
        return 2

    cfg = load_league(args.config) if args.config else load_league()
    season = args.season or cfg["league"]["season"]
    try:
        season_id = season_start(season) + 1
    except ValueError as exc:
        _safe_print(f"error: {exc}", file=sys.stderr)
        return 2

    if client is None:
        base = (args.data_dir / "raw" / SOURCE) if args.data_dir is not None else raw_dir(SOURCE)
        client = ESPNLeagueClient(cache_dir=base, offline=args.offline)

    season_ids_to_try = [season_id]
    if season_id - 1 not in season_ids_to_try:
        season_ids_to_try.append(season_id - 1)  # per ADR 0005: the league may still show last season

    settings_payload = None
    used_season_id = None
    last_exc: Exception | None = None
    for sid in season_ids_to_try:
        try:
            settings_payload = fetch_settings_teams_rosters(client, sid, league_id)
            used_season_id = sid
            break
        except ESPNAuthError as exc:
            _safe_print(f"error: league {league_id} requires authentication (HTTP {exc.status}). "
                        "This league is not publicly viewable without cookies. If it should be, add "
                        "ESPN_S2 and ESPN_SWID to your OWN .env (see .env.example) -- never paste them "
                        "into chat or commit them -- then re-run. No further data was pulled.",
                        file=sys.stderr)
            return 3
        except ESPNNotFoundError as exc:
            last_exc = exc
            continue
        except ESPNOfflineCacheMiss as exc:
            _safe_print(f"error: {exc}", file=sys.stderr)
            return 5
    if settings_payload is None:
        _safe_print(f"error: league {league_id} not found for ESPN season id(s) {season_ids_to_try} "
                    f"({last_exc}). Double-check the league id and season.", file=sys.stderr)
        return 4
    if used_season_id != season_id:
        _safe_print(f"note: season id {season_id} (derived from {season!r}) was not found; "
                    f"used {used_season_id} instead.")

    draft_payload = fetch_draft_detail(client, used_season_id, league_id)
    fa_payload = fetch_free_agents(client, used_season_id, league_id, limit=args.free_agent_limit)
    try:
        tx_payload = fetch_transactions(client, used_season_id, league_id)
    except ESPNLeagueError as exc:
        _safe_print(f"warning: transactions fetch failed ({exc}); continuing without transactions",
                    file=sys.stderr)
        tx_payload = {}

    report = build_report(cfg, league_id, season, used_season_id, settings_payload=settings_payload,
                          draft_payload=draft_payload, fa_payload=fa_payload, tx_payload=tx_payload)

    out_root = args.data_dir if args.data_dir is not None else data_dir()
    out_path = args.out or (out_root / "processed" / "espn_league" / f"{league_id}_{used_season_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    _print_report(report, out_path)
    return 0


def _int_env(name: str) -> int | None:
    val = os.environ.get(name, "").strip()
    return int(val) if val else None


if __name__ == "__main__":
    raise SystemExit(main())
