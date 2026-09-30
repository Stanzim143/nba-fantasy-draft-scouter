"""NBAClient: caching, atomicity, offline mode, retries/backoff, rate limiting. No network."""
import json
import os
import random

import pytest
import requests

from ingest_fakes import FakeClock, FakeResponse, FakeSession, wrap
from src.ingest import nba_client as nc
from src.ingest.nba_client import (
    NBAClient, NBACacheCorruptError, NBAHTTPError, NBAOfflineCacheMiss, cache_key, validate_payload,
)

OK = wrap("Thing", ["A", "B"], [[1, 2]])
PARAMS = {"Season": "2018-19", "PlayerOrTeam": "P", "Empty": ""}


def make(tmp_path, script, **kw):
    clock = FakeClock()
    kw.setdefault("min_interval", 0)
    kw.setdefault("jitter", 0)
    session = FakeSession(script)
    client = NBAClient(tmp_path / "raw", session=session, clock=clock, sleep=clock.sleep, offline=False, **kw)
    return client, session, clock


# ------------------------------------------------------------------ cache keys

def test_cache_key_ignores_param_order_and_types():
    assert cache_key("e", {"a": 1, "b": "x"}) == cache_key("e", {"b": "x", "a": "1"})


def test_cache_key_differs_by_endpoint_and_params():
    base = cache_key("e", {"Season": "2018-19"})
    assert base != cache_key("f", {"Season": "2018-19"})
    assert base != cache_key("e", {"Season": "2019-20"})
    # an empty-string param is part of the identity (the server treats "" and absent alike, we don't guess)
    assert base != cache_key("e", {"Season": "2018-19", "DateFrom": ""})


def test_cache_key_is_filesystem_safe():
    k = cache_key("e", {"x": 'a/b\\c:d*e?f"g<h>i|j', "y": "Regular Season"})
    assert not set(k) & set('/\\:*?"<>|')
    assert len(k) < 120


# ------------------------------------------------------------------ cache hit / miss

def test_miss_downloads_then_hit_never_touches_network(tmp_path):
    client, session, _ = make(tmp_path, [FakeResponse(200, OK)])
    a = client.get("thing", PARAMS)
    b = client.get("thing", PARAMS)
    assert a == b == OK
    assert len(session.calls) == 1
    assert client.stats.cache_hits == 1 and client.stats.network_requests == 1


def test_cache_file_is_verbatim_response_bytes(tmp_path):
    body = json.dumps(OK, ensure_ascii=False).encode("utf-8") + b" "  # odd whitespace kept
    client, _, _ = make(tmp_path, [FakeResponse(200, body)])
    client.get("thing", PARAMS)
    assert client.cache_path("thing", PARAMS).read_bytes() == body


def test_cache_survives_new_client_instance(tmp_path):
    c1, s1, _ = make(tmp_path, [FakeResponse(200, OK)])
    c1.get("thing", PARAMS)
    c2, s2, _ = make(tmp_path, [FakeResponse(500)])
    assert c2.get("thing", PARAMS) == OK
    assert s2.calls == []


def test_non_ascii_payload_roundtrips(tmp_path):
    payload = wrap("T", ["NAME"], [["Nikola Jokić"], ["Dennis Schröder"]])
    client, _, _ = make(tmp_path, [FakeResponse(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"))])
    client.get("t", {})
    fresh, _, _ = make(tmp_path, [FakeResponse(500)])
    assert fresh.get("t", {})["resultSets"][0]["rowSet"][0][0] == "Nikola Jokić"


def test_different_params_are_different_entries(tmp_path):
    client, session, _ = make(tmp_path, [FakeResponse(200, OK)])
    client.get("thing", {"Season": "2018-19"})
    client.get("thing", {"Season": "2019-20"})
    assert len(session.calls) == 2


def test_peek_never_hits_network(tmp_path):
    client, session, _ = make(tmp_path, [FakeResponse(200, OK)])
    assert client.peek("thing", PARAMS) is None
    client.get("thing", PARAMS)
    assert client.peek("thing", PARAMS) == OK
    assert len(session.calls) == 1


def test_refresh_redownloads_and_replaces(tmp_path):
    new = wrap("Thing", ["A", "B"], [[9, 9]])
    client, session, _ = make(tmp_path, [FakeResponse(200, OK), FakeResponse(200, new)])
    client.get("thing", PARAMS)
    assert client.get("thing", PARAMS, refresh=True) == new
    assert client.get("thing", PARAMS) == new
    assert len(session.calls) == 2


def test_refresh_is_refused_offline(tmp_path):
    client, _, _ = make(tmp_path, [FakeResponse(200, OK)])
    client.offline = True
    with pytest.raises(NBAOfflineCacheMiss, match="cannot refresh"):
        client.get("thing", PARAMS, refresh=True)


# ------------------------------------------------------------------ offline mode

def test_offline_cache_miss_raises_clear_error_and_no_request(tmp_path):
    client, session, _ = make(tmp_path, [FakeResponse(200, OK)])
    client.offline = True
    with pytest.raises(NBAOfflineCacheMiss) as ei:
        client.get("thing", PARAMS)
    msg = str(ei.value)
    assert "offline" in msg and "thing" in msg and "NBA_OFFLINE" in msg and str(client.cache_dir) in msg
    assert session.calls == []


def test_offline_serves_cached_responses(tmp_path):
    online, _, _ = make(tmp_path, [FakeResponse(200, OK)])
    online.get("thing", PARAMS)
    off, session, _ = make(tmp_path, [FakeResponse(500)])
    off.offline = True
    assert off.get("thing", PARAMS) == OK
    assert session.calls == []


def test_offline_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_OFFLINE", "1")
    assert NBAClient(tmp_path).offline is True
    monkeypatch.setenv("NBA_OFFLINE", "0")
    assert NBAClient(tmp_path).offline is False
    monkeypatch.delenv("NBA_OFFLINE")
    assert NBAClient(tmp_path).offline is False
    monkeypatch.setenv("NBA_OFFLINE", "true")
    assert NBAClient(tmp_path).offline is True
    # an explicit argument beats the environment
    assert NBAClient(tmp_path, offline=False).offline is False


def test_default_cache_dir_is_under_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path))
    assert NBAClient().cache_dir == tmp_path / "raw" / "nba_api"


# ------------------------------------------------------------------ atomic writes / corruption

def test_failed_write_leaves_no_partial_file(tmp_path, monkeypatch):
    client, _, _ = make(tmp_path, [FakeResponse(200, OK)])
    real_replace = os.replace

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(nc.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        client.get("thing", PARAMS)
    monkeypatch.setattr(nc.os, "replace", real_replace)
    d = client.cache_path("thing", PARAMS).parent
    assert not client.cache_path("thing", PARAMS).exists()
    assert list(d.glob("*.tmp")) == []


def test_failed_write_keeps_previous_good_file(tmp_path, monkeypatch):
    path = tmp_path / "f.json"
    nc.atomic_write_bytes(path, b'{"resultSets": []}')

    def boom(src, dst):
        raise OSError("nope")

    monkeypatch.setattr(nc.os, "replace", boom)
    with pytest.raises(OSError):
        nc.atomic_write_bytes(path, b"NEW")
    assert path.read_bytes() == b'{"resultSets": []}'
    assert list(tmp_path.glob("*.tmp")) == []


def test_corrupt_cache_file_is_quarantined_and_refetched_online(tmp_path):
    client, session, _ = make(tmp_path, [FakeResponse(200, OK)])
    path = client.cache_path("thing", PARAMS)
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"resultSets": [tru')  # truncated by a crash
    assert client.get("thing", PARAMS) == OK
    assert len(session.calls) == 1
    assert path.with_suffix(".corrupt").exists()  # evidence kept


def test_corrupt_cache_file_offline_is_a_clear_error(tmp_path):
    client, _, _ = make(tmp_path, [FakeResponse(200, OK)])
    path = client.cache_path("thing", PARAMS)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not json")
    client.offline = True
    with pytest.raises(NBACacheCorruptError, match="corrupt"):
        client.get("thing", PARAMS)


# ------------------------------------------------------------------ retries and backoff

def test_retries_5xx_with_exponential_backoff_then_succeeds(tmp_path):
    client, session, clock = make(tmp_path, [FakeResponse(500), FakeResponse(503), FakeResponse(200, OK)])
    assert client.get("thing", PARAMS) == OK
    assert len(session.calls) == 3
    assert clock.sleeps == [2.0, 4.0]  # base 2: 2^1, 2^2
    assert client.stats.retries == 2


def test_retries_connection_errors_and_timeouts(tmp_path):
    script = [requests.exceptions.ConnectionError("reset"), requests.exceptions.ReadTimeout("slow"), FakeResponse(200, OK)]
    client, session, clock = make(tmp_path, script)
    assert client.get("thing", PARAMS) == OK
    assert len(session.calls) == 3 and len(clock.sleeps) == 2


def test_backoff_is_capped(tmp_path):
    client, _, clock = make(tmp_path, [FakeResponse(500)] * 5 + [FakeResponse(200, OK)],
                            max_retries=6, backoff_max=10)
    client.get("thing", PARAMS)
    assert clock.sleeps == [2.0, 4.0, 8.0, 10.0, 10.0]


def test_retry_after_header_is_honoured(tmp_path):
    client, _, clock = make(tmp_path, [FakeResponse(429, headers={"Retry-After": "30"}), FakeResponse(200, OK)])
    client.get("thing", PARAMS)
    assert clock.sleeps == [30.0]


def test_retry_after_is_capped_by_backoff_max(tmp_path):
    client, _, clock = make(tmp_path, [FakeResponse(429, headers={"Retry-After": "9999"}), FakeResponse(200, OK)],
                            backoff_max=60)
    client.get("thing", PARAMS)
    assert clock.sleeps == [60.0]


def test_jitter_stays_within_bounds_and_is_seedable(tmp_path):
    client, _, clock = make(tmp_path, [FakeResponse(500), FakeResponse(500), FakeResponse(200, OK)],
                            jitter=0.5, rng=random.Random(7))
    client.get("thing", PARAMS)
    base = [2.0, 4.0]
    assert all(b <= s <= b * 1.5 for b, s in zip(base, clock.sleeps))
    client2, _, clock2 = make(tmp_path / "b", [FakeResponse(500), FakeResponse(500), FakeResponse(200, OK)],
                              jitter=0.5, rng=random.Random(7))
    client2.get("thing", PARAMS)
    assert clock.sleeps == clock2.sleeps


def test_gives_up_after_max_retries_and_caches_nothing(tmp_path):
    client, session, _ = make(tmp_path, [FakeResponse(503)], max_retries=3)
    with pytest.raises(NBAHTTPError, match="gave up after 4 attempts") as ei:
        client.get("thing", PARAMS)
    assert "throttling" in str(ei.value)
    assert len(session.calls) == 4
    assert not client.cache_path("thing", PARAMS).exists()


def test_max_retries_zero_means_single_attempt(tmp_path):
    client, session, clock = make(tmp_path, [FakeResponse(500)], max_retries=0)
    with pytest.raises(NBAHTTPError):
        client.get("thing", PARAMS)
    assert len(session.calls) == 1 and clock.sleeps == []


def test_non_retryable_status_fails_fast_with_status(tmp_path):
    client, session, clock = make(tmp_path, [FakeResponse(400, b"Bad Season")])
    with pytest.raises(NBAHTTPError, match="HTTP 400") as ei:
        client.get("thing", PARAMS)
    assert ei.value.status == 400 and "Bad Season" in str(ei.value)
    assert len(session.calls) == 1 and clock.sleeps == []


@pytest.mark.parametrize("body", [b"<html>Access Denied</html>", b"[1,2,3]", b'{"nope": 1}', b""])
def test_bad_200_bodies_are_retried_not_cached(tmp_path, body):
    client, session, _ = make(tmp_path, [FakeResponse(200, body), FakeResponse(200, OK)])
    assert client.get("thing", PARAMS) == OK
    assert len(session.calls) == 2


def test_persistent_bad_body_raises_with_reason(tmp_path):
    client, _, _ = make(tmp_path, [FakeResponse(200, b'{"nope": 1}')], max_retries=1)
    with pytest.raises(NBAHTTPError, match="invalid body"):
        client.get("thing", PARAMS)
    assert not client.cache_path("thing", PARAMS).exists()


def test_programming_errors_are_not_swallowed_as_network_errors(tmp_path):
    client, session, _ = make(tmp_path, [RuntimeError("bug")])
    with pytest.raises(RuntimeError, match="bug"):
        client.get("thing", PARAMS)
    assert len(session.calls) == 1


def test_403_is_retried_then_reported_as_possible_block(tmp_path):
    client, session, _ = make(tmp_path, [FakeResponse(403)], max_retries=2)
    with pytest.raises(NBAHTTPError, match="HTTP 403"):
        client.get("thing", PARAMS)
    assert len(session.calls) == 3


# ------------------------------------------------------------------ rate limiting

def test_consecutive_requests_are_spaced_by_min_interval(tmp_path):
    client, session, clock = make(tmp_path, [FakeResponse(200, OK)], min_interval=1.5)
    stamps = []
    for season in ("2015-16", "2016-17", "2017-18"):
        client.get("thing", {"Season": season})
        stamps.append(clock.now)
    assert len(session.calls) == 3
    assert clock.sleeps == [1.5, 1.5]  # first request is free, then one full interval each time
    assert stamps[1] - stamps[0] >= 1.5 and stamps[2] - stamps[1] >= 1.5


def test_time_already_elapsed_counts_towards_interval(tmp_path):
    client, _, clock = make(tmp_path, [FakeResponse(200, OK)], min_interval=2.0)
    client.get("thing", {"Season": "a"})
    clock.tick(1.25)
    client.get("thing", {"Season": "b"})
    assert clock.sleeps == [pytest.approx(0.75)]
    client.get("thing", {"Season": "c"})
    clock.tick(5)
    client.get("thing", {"Season": "d"})
    assert len(clock.sleeps) == 2  # no wait needed after a long gap


def test_cache_hits_are_free_and_do_not_reset_the_rate_limit_clock(tmp_path):
    client, _, clock = make(tmp_path, [FakeResponse(200, OK)], min_interval=1.0)
    client.get("thing", {"Season": "a"})
    for _ in range(5):
        client.get("thing", {"Season": "a"})  # hits
    assert clock.sleeps == []
    assert client.stats.cache_hits == 5


def test_retries_also_respect_the_rate_limit(tmp_path):
    client, _, clock = make(tmp_path, [FakeResponse(500), FakeResponse(200, OK)], min_interval=1.0, backoff_base=1)
    client.get("thing", PARAMS)
    # backoff sleep of 1.0 already satisfies the 1.0 interval, so no extra rate-limit sleep
    assert clock.sleeps == [1.0]


# ------------------------------------------------------------------ request shape

def test_request_carries_browser_headers_url_params_and_timeout(tmp_path):
    client, session, _ = make(tmp_path, [FakeResponse(200, OK)])
    client.get("leaguegamelog", {"Season": "2018-19", "Counter": 0, "Nothing": None})
    call = session.calls[0]
    assert call["url"] == "https://stats.nba.com/stats/leaguegamelog"
    assert call["params"] == {"Counter": "0", "Nothing": "", "Season": "2018-19"}
    assert "Mozilla" in call["headers"]["User-Agent"]
    assert call["headers"]["Referer"] == "https://www.nba.com/"
    assert call["timeout"] == client.timeout and all(t > 0 for t in client.timeout)


def test_custom_headers_override_defaults(tmp_path):
    client, session, _ = make(tmp_path, [FakeResponse(200, OK)], headers={"User-Agent": "test-agent"})
    client.get("thing", {})
    assert session.calls[0]["headers"]["User-Agent"] == "test-agent"
    assert "Referer" in session.calls[0]["headers"]


def test_fetch_log_records_every_attempt(tmp_path):
    client, _, _ = make(tmp_path, [FakeResponse(500), FakeResponse(200, OK)])
    client.get("thing", PARAMS)
    lines = (client.cache_dir / "_fetch_log.jsonl").read_text().splitlines()
    entries = [json.loads(x) for x in lines]
    assert [e["status"] for e in entries] == [500, 200]
    assert entries[0]["endpoint"] == "thing" and entries[1]["attempt"] == 1


def test_constructor_rejects_nonsense_settings(tmp_path):
    for bad in ({"min_interval": -1}, {"max_retries": -1}, {"backoff_base": 0.5}):
        with pytest.raises(ValueError):
            NBAClient(tmp_path, **bad)


def test_validate_payload_accepts_both_result_set_spellings():
    assert validate_payload({"resultSets": []})
    assert validate_payload({"resultSet": {"name": "x"}})
    with pytest.raises(nc.NBAResponseError):
        validate_payload({"other": 1})
    with pytest.raises(nc.NBAResponseError):
        validate_payload([])
