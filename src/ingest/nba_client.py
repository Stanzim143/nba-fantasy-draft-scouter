"""Polite, cached, retrying HTTP layer for stats.nba.com (the unofficial NBA stats API).

Design goals
------------
* **Never download the same response twice.** Every response is stored verbatim as JSON under
  ``raw_dir("nba_api")/<endpoint>/<key>.json`` and re-served from disk on later calls. The key is
  derived from the endpoint plus the full parameter set.
* **Offline reproducibility.** With ``offline=True`` (or ``NBA_OFFLINE=1``) a cache miss raises
  :class:`NBAOfflineCacheMiss` instead of touching the network, so a backtest can be rebuilt from
  the raw cache alone.
* **Polite.** At least ``min_interval`` seconds between *network* requests (cache hits are free),
  a real browser User-Agent, timeouts, and exponential backoff (honouring ``Retry-After``).
* **Atomic.** Files are written to a temp file in the same directory and renamed, so an
  interrupted run never leaves a half-written cache entry. Unparseable entries are treated as
  corrupt (quarantined and re-fetched online, an error offline).
* **Testable.** Session, clock, sleep and jitter RNG are injectable; no test needs the network.

This module knows nothing about tables; see ``nba_transform`` for that.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from src.contracts import raw_dir

BASE_URL = "https://stats.nba.com/stats"
SOURCE = "nba_api"  # raw_dir source name
OFFLINE_ENV = "NBA_OFFLINE"

# Browser-like headers. stats.nba.com silently drops connections from clients it does not like: in
# our tests a Windows Chrome/131 User-Agent was reset immediately while the header set below (the
# one the nba_api project ships, a current Chrome UA plus Sec-Ch-Ua hints) was accepted. If this
# ever starts failing, bump the Chrome version here first. (Accept-Encoding is left to requests.)
DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
    "Sec-Ch-Ua": '"Not:A-Brand";v="99", "Google Chrome";v="145", "Chromium";v="145"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Fetch-Dest": "empty",
}

RETRYABLE_STATUS = frozenset({403, 408, 425, 429, 500, 502, 503, 504})


# --------------------------------------------------------------------------- exceptions

class NBAClientError(RuntimeError):
    """Base class for every error this client raises deliberately."""


class NBAOfflineCacheMiss(NBAClientError):
    """Offline mode is on and the requested response is not in the raw cache."""


class NBAHTTPError(NBAClientError):
    """A non-retryable HTTP status (e.g. 400 for bad parameters), or retries were exhausted."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class NBAResponseError(NBAClientError):
    """The server answered 200 but the body is not a valid stats.nba.com result payload."""


class NBACacheCorruptError(NBAClientError):
    """A cache file exists but is not valid JSON and we are offline (cannot re-fetch)."""


# --------------------------------------------------------------------------- helpers

def offline_from_env(env: Mapping[str, str] | None = None) -> bool:
    val = (env if env is not None else os.environ).get(OFFLINE_ENV, "")
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _canonical_params(params: Mapping[str, Any]) -> dict[str, str]:
    return {str(k): "" if v is None else str(v) for k, v in sorted(params.items())}


def cache_key(endpoint: str, params: Mapping[str, Any]) -> str:
    """Stable, filesystem-safe, human-skimmable file stem for (endpoint, params)."""
    canon = _canonical_params(params)
    digest = hashlib.sha1(
        json.dumps([endpoint, canon], sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    readable = "_".join(v for v in canon.values() if v)
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", readable).strip("-")[:70]
    return f"{readable}__{digest}" if readable else digest


def validate_payload(payload: Any, *, where: str = "response") -> dict:
    """A stats.nba.com payload is a JSON object with ``resultSets`` (list) or ``resultSet``."""
    if not isinstance(payload, dict):
        raise NBAResponseError(f"{where}: expected a JSON object, got {type(payload).__name__}")
    sets = payload.get("resultSets", payload.get("resultSet"))
    if sets is None:
        raise NBAResponseError(f"{where}: no 'resultSets'/'resultSet' key (keys: {sorted(payload)})")
    return payload


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
    slept_for_rate_limit: float = 0.0
    slept_for_backoff: float = 0.0
    log: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- client

class NBAClient:
    """Rate-limited, retrying, disk-cached client. See module docstring."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        *,
        offline: bool | None = None,
        min_interval: float = 1.0,
        max_retries: int = 6,
        backoff_base: float = 2.0,
        backoff_max: float = 120.0,
        jitter: float = 0.25,
        timeout: tuple[float, float] = (10.0, 60.0),
        session: Any = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        headers: Mapping[str, str] | None = None,
        base_url: str = BASE_URL,
        write_fetch_log: bool = True,
    ):
        if min_interval < 0 or max_retries < 0 or backoff_base < 1:
            raise ValueError("min_interval/max_retries must be >= 0 and backoff_base >= 1")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else raw_dir(SOURCE)
        self.offline = offline_from_env() if offline is None else offline
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.jitter = jitter
        self.timeout = timeout
        self._session = session
        self._clock = clock
        self._sleep = sleep
        self._rng = rng or random.Random()
        self.headers = {**DEFAULT_HEADERS, **(headers or {})}
        self.base_url = base_url.rstrip("/")
        self.write_fetch_log = write_fetch_log
        self._last_request_at: float | None = None
        self.stats = ClientStats()

    # ----- cache

    def cache_path(self, endpoint: str, params: Mapping[str, Any]) -> Path:
        return self.cache_dir / endpoint / f"{cache_key(endpoint, params)}.json"

    def _read_cache(self, path: Path) -> dict | None:
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        try:
            return validate_payload(json.loads(raw), where=str(path))
        except (ValueError, NBAResponseError) as exc:
            if self.offline:
                raise NBACacheCorruptError(f"corrupt cache file {path}: {exc}") from exc
            # Keep the evidence, then fall through to a re-fetch.
            path.replace(path.with_suffix(".corrupt"))
            return None

    # ----- public API

    def peek(self, endpoint: str, params: Mapping[str, Any]) -> dict | None:
        """Cached payload or ``None``. Never touches the network (works online and offline)."""
        return self._read_cache(self.cache_path(endpoint, params))

    def get(self, endpoint: str, params: Mapping[str, Any], *, refresh: bool = False) -> dict:
        """Return the parsed JSON for ``endpoint``/``params``, from cache if possible.

        ``refresh=True`` bypasses the cache and re-downloads (atomically replacing the old file);
        it is refused offline. Use it only for data that changed upstream (an in-progress season).
        """
        path = self.cache_path(endpoint, params)
        if refresh and self.offline:
            raise NBAOfflineCacheMiss(f"offline mode: cannot refresh {endpoint} {dict(params)}")
        cached = None if refresh else self._read_cache(path)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached
        if self.offline:
            raise NBAOfflineCacheMiss(
                f"offline mode: {endpoint} {dict(params)} is not cached at {path}. "
                f"Run the ingest online once (unset {OFFLINE_ENV}) to populate {self.cache_dir}."
            )
        body = self._fetch(endpoint, params)
        payload = validate_payload(json.loads(body), where=f"{endpoint} {dict(params)}")
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
            self.stats.slept_for_rate_limit += wait
            self._sleep(wait)

    def _backoff_delay(self, attempt: int, retry_after: float | None) -> float:
        delay = min(self.backoff_max, self.backoff_base ** (attempt + 1))
        if retry_after is not None:
            delay = max(delay, min(retry_after, self.backoff_max))
        if self.jitter:
            delay *= 1 + self._rng.uniform(0, self.jitter)
        return delay

    def _fetch(self, endpoint: str, params: Mapping[str, Any]) -> bytes:
        url = f"{self.base_url}/{endpoint}"
        query = _canonical_params(params)
        last_problem = "no attempt made"
        for attempt in range(self.max_retries + 1):
            self._respect_rate_limit()
            started = self._clock()
            self._last_request_at = started
            self.stats.network_requests += 1
            status: int | None = None
            retry_after: float | None = None
            problem: str | None = None
            body = b""
            try:
                resp = self._get_session().get(url, params=query, headers=self.headers, timeout=self.timeout)
                status, body = resp.status_code, resp.content
                ra = (getattr(resp, "headers", None) or {}).get("Retry-After")
                if ra is not None:
                    try:
                        retry_after = float(ra)
                    except ValueError:
                        retry_after = None
            except Exception as exc:  # network layer: timeouts, resets, DNS, TLS ...
                if not _is_network_error(exc):
                    raise
                problem = f"{type(exc).__name__}: {exc}"
            self._log(endpoint, query, attempt, status, len(body), self._clock() - started, problem)

            if problem is None:
                if status == 200:
                    try:
                        validate_payload(json.loads(body), where=f"{endpoint} {query}")
                        return body
                    except (ValueError, NBAResponseError) as exc:
                        problem = f"invalid body from server: {exc}"
                elif status in RETRYABLE_STATUS:
                    problem = f"HTTP {status}"
                else:
                    snippet = body[:200].decode("utf-8", "replace")
                    raise NBAHTTPError(f"{endpoint} {query}: HTTP {status} (not retryable): {snippet}", status)
            last_problem = problem
            if attempt < self.max_retries:
                delay = self._backoff_delay(attempt, retry_after)
                self.stats.retries += 1
                self.stats.slept_for_backoff += delay
                self._sleep(delay)
        raise NBAHTTPError(
            f"{endpoint} {query}: gave up after {self.max_retries + 1} attempts; last problem: {last_problem}. "
            "stats.nba.com may be throttling or blocking this IP; wait and re-run (cached responses are kept).",
            None,
        )

    def _log(self, endpoint, query, attempt, status, nbytes, elapsed, problem) -> None:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "endpoint": endpoint,
            "params": query,
            "attempt": attempt,
            "status": status,
            "bytes": nbytes,
            "elapsed_s": round(elapsed, 3),
            "problem": problem,
        }
        self.stats.log.append(entry)
        if self.write_fetch_log:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                with open(self.cache_dir / "_fetch_log.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")
            except OSError:
                pass  # the log is diagnostic only


def _is_network_error(exc: Exception) -> bool:
    """True for transport-level failures worth retrying (requests exceptions, OS-level errors)."""
    try:
        import requests

        if isinstance(exc, requests.exceptions.RequestException):
            return True
    except ImportError:  # pragma: no cover
        pass
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))
