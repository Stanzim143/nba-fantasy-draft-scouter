"""Polite, cached, retrying HTTP layer shared by the ESPN ADP ingest (and, in principle, any other
non-stats.nba.com source this project talks to).

This is a deliberately independent sibling of ``nba_client.py``, not a subclass or an import of it:
ESPN and FantasyPros have a different auth model (none), a different rate-limit posture (>= 2s between
requests, per ADR 0005/CLAUDE.md), different response shapes (arbitrary JSON or HTML, not
``stats.nba.com``'s ``resultSets`` envelope), and callers need to reach more than one host and content
type from the same process. The *shape* of the solution (disk cache keyed by endpoint+params, atomic
writes, offline mode via ``NBA_OFFLINE``, exponential backoff with jitter, injectable clock/session for
tests) intentionally mirrors ``nba_client.py`` so both are recognizable, but nothing is shared at the
code level to keep the two sources decoupled.

Design goals (same rationale as ``nba_client.py``):
* Never re-download an identical request; cache verbatim response bytes under
  ``raw_dir(source)/<label>/<key>.<ext>``.
* Offline reproducibility via ``NBA_OFFLINE=1`` (reusing the existing project-wide convention).
* Polite: a minimum interval between *network* requests (cache hits are free), a descriptive
  User-Agent, timeouts, exponential backoff honouring ``Retry-After``.
* Atomic writes, corrupt-cache quarantine.
* Testable: session, clock, sleep and jitter RNG are injectable.
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

OFFLINE_ENV = "NBA_OFFLINE"  # reuse the project-wide offline convention (see nba_client.py)

_MISS = object()  # sentinel: a cached JSON ``null`` is a valid hit, not a miss

RETRYABLE_STATUS = frozenset({403, 408, 425, 429, 500, 502, 503, 504})


# --------------------------------------------------------------------------- exceptions

class HttpCacheError(RuntimeError):
    """Base class for every error this client raises deliberately."""


class OfflineCacheMiss(HttpCacheError):
    """Offline mode is on and the requested response is not in the raw cache."""


class HttpFetchError(HttpCacheError):
    """A non-retryable HTTP status, or retries were exhausted."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class CacheCorruptError(HttpCacheError):
    """A cache file exists but cannot be read and we are offline (cannot re-fetch)."""


# --------------------------------------------------------------------------- helpers

def offline_from_env(env: Mapping[str, str] | None = None) -> bool:
    val = (env if env is not None else os.environ).get(OFFLINE_ENV, "")
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _canonical_params(params: Mapping[str, Any]) -> dict[str, str]:
    return {str(k): "" if v is None else str(v) for k, v in sorted(params.items())}


def cache_key(label: str, params: Mapping[str, Any]) -> str:
    """Stable, filesystem-safe, human-skimmable file stem for (label, params).

    The URL is deliberately not part of the key: raw caches hold point-in-time data that cannot be refetched, so changing the
    key would orphan them. The contract is one label per URL (every caller builds its label from a fixed endpoint), and
    ``test_every_http_cache_label_maps_to_one_url`` in tests/ingest guards it.
    """
    canon = _canonical_params(params)
    digest = hashlib.sha1(json.dumps([label, canon], sort_keys=True).encode("utf-8")).hexdigest()[:12]
    readable = "_".join(v for v in canon.values() if v)
    readable = re.sub(r"[^A-Za-z0-9._-]+", "-", readable).strip("-")[:70]
    return f"{readable}__{digest}" if readable else digest


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

class CachedHttpClient:
    """Rate-limited, retrying, disk-cached client for arbitrary GET requests.

    Unlike ``NBAClient`` (one fixed ``base_url``), each call supplies its own full URL, so one
    instance can serve several hosts (e.g. ESPN's fantasy API and FantasyPros) while sharing one
    cache directory, rate limiter and politeness policy.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        offline: bool | None = None,
        min_interval: float = 2.0,
        max_retries: int = 5,
        backoff_base: float = 2.0,
        backoff_max: float = 120.0,
        jitter: float = 0.25,
        timeout: tuple[float, float] = (10.0, 60.0),
        session: Any = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        headers: Mapping[str, str] | None = None,
        write_fetch_log: bool = True,
    ):
        if min_interval < 0 or max_retries < 0 or backoff_base < 1:
            raise ValueError("min_interval/max_retries must be >= 0 and backoff_base >= 1")
        self.cache_dir = Path(cache_dir)
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
        self.headers = dict(headers or {})
        self.write_fetch_log = write_fetch_log
        self._last_request_at: float | None = None
        self.stats = ClientStats()

    # ----- cache

    def cache_path(self, label: str, params: Mapping[str, Any], *, ext: str = "json") -> Path:
        return self.cache_dir / label / f"{cache_key(label, params)}.{ext}"

    def _read_cache(self, path: Path, *, parse: Callable[[bytes], Any]) -> Any:
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return _MISS
        try:
            return parse(raw)
        except Exception as exc:  # noqa: BLE001 - any parse failure means "not usable"
            if self.offline:
                raise CacheCorruptError(f"corrupt cache file {path}: {exc}") from exc
            path.replace(path.with_suffix(path.suffix + ".corrupt"))
            return _MISS

    # ----- public API

    def get_json(self, label: str, url: str, params: Mapping[str, Any] | None = None, *,
                 headers: Mapping[str, str] | None = None, refresh: bool = False) -> Any:
        """GET ``url`` (JSON body), cached under ``label``. Returns the parsed JSON."""
        params = params or {}
        path = self.cache_path(label, params, ext="json")
        body = self._get_bytes(label, url, params, path, headers=headers, refresh=refresh,
                                parse=lambda b: json.loads(b.decode("utf-8")))
        return json.loads(body.decode("utf-8"))

    def get_text(self, label: str, url: str, params: Mapping[str, Any] | None = None, *,
                 headers: Mapping[str, str] | None = None, refresh: bool = False) -> str:
        """GET ``url`` (text/HTML body), cached under ``label``. Returns the decoded text."""
        params = params or {}
        path = self.cache_path(label, params, ext="html")
        body = self._get_bytes(label, url, params, path, headers=headers, refresh=refresh,
                                parse=lambda b: b.decode("utf-8", "replace"))
        return body.decode("utf-8", "replace")

    def _get_bytes(self, label: str, url: str, params: Mapping[str, Any], path: Path, *,
                    headers: Mapping[str, str] | None, refresh: bool,
                    parse: Callable[[bytes], Any]) -> bytes:
        if refresh and self.offline:
            raise OfflineCacheMiss(f"offline mode: cannot refresh {label} {dict(params)}")
        cached = _MISS if refresh else self._read_cache(path, parse=parse)
        if cached is not _MISS:
            self.stats.cache_hits += 1
            return path.read_bytes()
        if self.offline:
            raise OfflineCacheMiss(
                f"offline mode: {label} {dict(params)} is not cached at {path}. "
                f"Run once online (unset {OFFLINE_ENV}) to populate {self.cache_dir}."
            )
        body = self._fetch(label, url, params, headers=headers, parse=parse)
        atomic_write_bytes(path, body)
        return body

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

    def _fetch(self, label: str, url: str, params: Mapping[str, Any], *,
               headers: Mapping[str, str] | None, parse: Callable[[bytes], Any]) -> bytes:
        req_headers = {**self.headers, **(headers or {})}
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
                resp = self._get_session().get(url, params=query, headers=req_headers, timeout=self.timeout)
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
            self._log(label, url, query, attempt, status, len(body), self._clock() - started, problem)

            if problem is None:
                if status == 200:
                    try:
                        parse(body)
                        return body
                    except Exception as exc:  # noqa: BLE001
                        problem = f"invalid body from server: {exc}"
                elif status in RETRYABLE_STATUS:
                    problem = f"HTTP {status}"
                else:
                    snippet = body[:200].decode("utf-8", "replace")
                    raise HttpFetchError(f"{label} {url} {query}: HTTP {status} (not retryable): {snippet}", status)
            last_problem = problem
            if attempt < self.max_retries:
                delay = self._backoff_delay(attempt, retry_after)
                self.stats.retries += 1
                self.stats.slept_for_backoff += delay
                self._sleep(delay)
        raise HttpFetchError(
            f"{label} {url} {query}: gave up after {self.max_retries + 1} attempts; last problem: {last_problem}. "
            "The source may be throttling or blocking this IP; wait and re-run (cached responses are kept).",
            None,
        )

    def _log(self, label, url, query, attempt, status, nbytes, elapsed, problem) -> None:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "label": label, "url": url, "params": query, "attempt": attempt, "status": status,
            "bytes": nbytes, "elapsed_s": round(elapsed, 3), "problem": problem,
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
    try:
        import requests

        if isinstance(exc, requests.exceptions.RequestException):
            return True
    except ImportError:  # pragma: no cover
        pass
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


def default_cache_dir(source: str) -> Path:
    return raw_dir(source)
