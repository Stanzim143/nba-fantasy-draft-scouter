"""CachedHttpClient: caching, atomicity, offline mode, retries/backoff, rate limiting. No network."""

import pytest

from ingest_fakes import FakeClock, FakeResponse, FakeSession
from src.ingest.http_cache import (
    CachedHttpClient, CacheCorruptError, HttpFetchError, OfflineCacheMiss, cache_key,
)

URL = "https://example.com/players"
PARAMS = {"season": "2023-24", "limit": 5}


def make(tmp_path, script, **kw):
    clock = FakeClock()
    kw.setdefault("min_interval", 0)
    kw.setdefault("jitter", 0)
    kw.setdefault("offline", False)
    session = FakeSession(script)
    client = CachedHttpClient(tmp_path / "raw", session=session, clock=clock, sleep=clock.sleep, **kw)
    return client, session, clock


# ------------------------------------------------------------------ cache keys

def test_cache_key_ignores_param_order_and_types():
    assert cache_key("e", {"a": 1, "b": "x"}) == cache_key("e", {"b": "x", "a": "1"})


def test_cache_key_differs_by_label_and_params():
    base = cache_key("e", {"season": "2023-24"})
    assert base != cache_key("f", {"season": "2023-24"})
    assert base != cache_key("e", {"season": "2024-25"})


# ------------------------------------------------------------------ json cache hit/miss

def test_json_miss_downloads_then_hit_never_touches_network(tmp_path):
    body = {"a": 1}
    client, session, _ = make(tmp_path, [FakeResponse(200, body)])
    a = client.get_json("players", URL, PARAMS)
    b = client.get_json("players", URL, PARAMS)
    assert a == b == body
    assert len(session.calls) == 1
    assert client.stats.cache_hits == 1 and client.stats.network_requests == 1


def test_json_cache_survives_new_client_instance(tmp_path):
    c1, _, _ = make(tmp_path, [FakeResponse(200, {"a": 1})])
    c1.get_json("players", URL, PARAMS)
    c2, s2, _ = make(tmp_path, [FakeResponse(500)])
    assert c2.get_json("players", URL, PARAMS) == {"a": 1}
    assert s2.calls == []


def test_headers_are_sent_and_merged(tmp_path):
    client, session, _ = make(tmp_path, [FakeResponse(200, {"a": 1})], headers={"X-Base": "b"})
    client.get_json("players", URL, PARAMS, headers={"X-Fantasy-Filter": "{}"})
    sent = session.calls[0]["headers"]
    assert sent["X-Base"] == "b"
    assert sent["X-Fantasy-Filter"] == "{}"


def test_refresh_redownloads_and_replaces(tmp_path):
    client, session, _ = make(tmp_path, [FakeResponse(200, {"a": 1}), FakeResponse(200, {"a": 2})])
    client.get_json("players", URL, PARAMS)
    fresh = client.get_json("players", URL, PARAMS, refresh=True)
    assert fresh == {"a": 2}
    assert len(session.calls) == 2


# ------------------------------------------------------------------ text (HTML) cache

def test_text_cache_roundtrips(tmp_path):
    client, session, _ = make(tmp_path, [FakeResponse(200, "<table>hi</table>")])
    a = client.get_text("adp", URL, {"year": 2024})
    b = client.get_text("adp", URL, {"year": 2024})
    assert a == b == "<table>hi</table>"
    assert len(session.calls) == 1


# ------------------------------------------------------------------ offline mode

def test_offline_cache_miss_raises_clear_error(tmp_path):
    client, _, _ = make(tmp_path, [], offline=True)
    with pytest.raises(OfflineCacheMiss, match="offline"):
        client.get_json("players", URL, PARAMS)


def test_offline_cache_hit_never_touches_network(tmp_path):
    online, _, _ = make(tmp_path, [FakeResponse(200, {"a": 1})])
    online.get_json("players", URL, PARAMS)
    offline, session, _ = make(tmp_path, [FakeResponse(500)], offline=True)
    assert offline.get_json("players", URL, PARAMS) == {"a": 1}
    assert session.calls == []


def test_offline_refresh_refused(tmp_path):
    client, _, _ = make(tmp_path, [], offline=True)
    with pytest.raises(OfflineCacheMiss):
        client.get_json("players", URL, PARAMS, refresh=True)


def test_offline_corrupt_cache_raises_instead_of_silently_refetching(tmp_path):
    client, _, _ = make(tmp_path, [FakeResponse(200, {"a": 1})])
    path = client.cache_path("players", PARAMS, ext="json")
    client.get_json("players", URL, PARAMS)
    path.write_bytes(b"not json{{{")
    offline, _, _ = make(tmp_path, [], offline=True)
    with pytest.raises(CacheCorruptError):
        offline.get_json("players", URL, PARAMS)


def test_cached_json_null_is_a_hit_not_a_miss(tmp_path):
    client, session, _ = make(tmp_path, [FakeResponse(200, "null")])
    assert client.get_json("players", URL, PARAMS) is None
    assert client.get_json("players", URL, PARAMS) is None
    assert len(session.calls) == 1 and client.stats.cache_hits == 1


def test_online_corrupt_cache_is_quarantined_and_refetched(tmp_path):
    client, session, _ = make(tmp_path, [FakeResponse(200, {"a": 1})])
    path = client.cache_path("players", PARAMS, ext="json")
    client.get_json("players", URL, PARAMS)
    path.write_bytes(b"not json{{{")
    client2, session2, _ = make(tmp_path, [FakeResponse(200, {"a": 2})])
    assert client2.get_json("players", URL, PARAMS) == {"a": 2}
    assert len(session2.calls) == 1
    assert path.with_suffix(".json.corrupt").exists()


# ------------------------------------------------------------------ retries / backoff / rate limit

def test_retries_on_retryable_status_then_succeeds(tmp_path):
    client, session, clock = make(tmp_path, [FakeResponse(500), FakeResponse(200, {"a": 1})])
    result = client.get_json("players", URL, PARAMS)
    assert result == {"a": 1}
    assert len(session.calls) == 2
    assert client.stats.retries == 1


def test_non_retryable_status_raises_immediately(tmp_path):
    client, session, _ = make(tmp_path, [FakeResponse(400, b"bad request")])
    with pytest.raises(HttpFetchError):
        client.get_json("players", URL, PARAMS)
    assert len(session.calls) == 1


def test_gives_up_after_max_retries(tmp_path):
    client, session, _ = make(tmp_path, [FakeResponse(503)], max_retries=2)
    with pytest.raises(HttpFetchError, match="gave up"):
        client.get_json("players", URL, PARAMS)
    assert len(session.calls) == 3  # initial + 2 retries


def test_rate_limit_sleeps_between_network_requests(tmp_path):
    clock = FakeClock()
    session = FakeSession([FakeResponse(200, {"a": 1}), FakeResponse(200, {"a": 2})])
    client = CachedHttpClient(tmp_path / "raw", session=session, clock=clock, sleep=clock.sleep,
                              offline=False, min_interval=2.0, jitter=0)
    client.get_json("players", URL, {"x": 1})
    client.get_json("players", URL, {"x": 2})
    assert clock.sleeps and clock.sleeps[0] == pytest.approx(2.0)


def test_default_min_interval_is_at_least_two_seconds(tmp_path):
    client = CachedHttpClient(tmp_path / "raw")
    assert client.min_interval >= 2.0
