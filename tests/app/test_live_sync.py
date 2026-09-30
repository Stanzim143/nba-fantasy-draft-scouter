"""Unit coverage for ``src.app.live_sync``: the draft-day live-sync diff/attribution/apply logic
(ADR 0025), deterministic and network-free throughout. No Streamlit import here on purpose, same
as the module under test."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.app import live_sync as ls
from src.app.state import ME, OPPONENT, DraftState, draft_player
from src.ingest.espn_league import ESPNLeagueClient
from src.ops.runlock import LockBusy


# --------------------------------------------------------------------------- tiny local fakes
# (mirrors tests/ingest/ingest_fakes.FakeResponse/FakeSession; kept local so this test file has
# no cross-directory import dependency)

class FakeResponse:
    def __init__(self, status: int = 200, body: dict | bytes | None = None):
        if isinstance(body, dict):
            body = json.dumps(body).encode("utf-8")
        self.status_code = status
        self.content = body if body is not None else b""


class FakeSession:
    def __init__(self, script: list):
        self.script = list(script)
        self.calls: list[dict] = []
        self._i = 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        item = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClock:
    """Injectable clock+sleep: sleeping advances time, so retry backoff never actually sleeps."""

    def __init__(self, start: float = 1000.0):
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(tmp_path, script, **kw):
    session = FakeSession(script)
    clock = FakeClock()
    kw.setdefault("min_interval", 0)
    kw.setdefault("offline", False)
    return ESPNLeagueClient(tmp_path / "raw", session=session, clock=clock, sleep=clock.sleep, **kw), session


def draft_payload(picks: list[dict]) -> dict:
    return {"draftDetail": {"drafted": False, "inProgress": True, "picks": picks}}


def pick(overall, team_id, player_id, round_=1):
    return {"id": overall, "overallPickNumber": overall, "roundId": round_, "roundPickNumber": overall,
           "teamId": team_id, "playerId": player_id, "keeper": False}


# --------------------------------------------------------------------------- classify_team

def test_classify_team_matches_my_team_id():
    assert ls.classify_team(5, 5) == ME
    assert ls.classify_team(6, 5) == OPPONENT


def test_classify_team_unset_my_team_id_is_always_opponent():
    assert ls.classify_team(5, None) == OPPONENT
    assert ls.classify_team(None, None) == OPPONENT


# --------------------------------------------------------------------------- env / settings resolution

def test_team_id_from_env(monkeypatch):
    assert ls.team_id_from_env({"ESPN_TEAM_ID": "7"}) == 7
    assert ls.team_id_from_env({}) is None


def test_team_id_from_settings_reads_nightly_json(tmp_path):
    (tmp_path / "nightly.json").write_text(json.dumps({"team_id": 4}), encoding="utf-8")
    assert ls.team_id_from_settings(tmp_path) == 4


def test_team_id_from_settings_falls_back_to_daily_refresh_json(tmp_path):
    (tmp_path / "daily_refresh.json").write_text(json.dumps({"team_id": 9}), encoding="utf-8")
    assert ls.team_id_from_settings(tmp_path) == 9


def test_team_id_from_settings_missing_file_is_none(tmp_path):
    assert ls.team_id_from_settings(tmp_path) is None


def test_team_id_from_settings_corrupt_file_is_none_not_a_crash(tmp_path):
    (tmp_path / "nightly.json").write_text("{not json", encoding="utf-8")
    assert ls.team_id_from_settings(tmp_path) is None


def test_default_team_id_prefers_env_over_settings(tmp_path):
    (tmp_path / "nightly.json").write_text(json.dumps({"team_id": 4}), encoding="utf-8")
    assert ls.default_team_id({"ESPN_TEAM_ID": "7"}, tmp_path) == 7
    assert ls.default_team_id({}, tmp_path) == 4


# --------------------------------------------------------------------------- diff_new_picks

def test_diff_new_picks_skips_unfilled_and_already_seen():
    picks = [
        {"overall_pick": 1, "espn_player_id": None, "team_id": 1},   # unfilled
        {"overall_pick": 2, "espn_player_id": 100, "team_id": 2},    # already seen
        {"overall_pick": 3, "espn_player_id": 200, "team_id": 3},    # new
    ]
    out = ls.diff_new_picks(picks, frozenset({2}))
    assert [p["overall_pick"] for p in out] == [3]


def test_diff_new_picks_is_order_stable_by_overall_pick():
    picks = [
        {"overall_pick": 5, "espn_player_id": 500, "team_id": 1},
        {"overall_pick": 3, "espn_player_id": 300, "team_id": 2},
        {"overall_pick": 4, "espn_player_id": 400, "team_id": 3},
    ]
    out = ls.diff_new_picks(picks, frozenset())
    assert [p["overall_pick"] for p in out] == [3, 4, 5]


def test_diff_new_picks_empty_when_nothing_new():
    picks = [{"overall_pick": 1, "espn_player_id": 10, "team_id": 1}]
    assert ls.diff_new_picks(picks, frozenset({1})) == []


# --------------------------------------------------------------------------- resolve_player / build_detected_picks

def test_resolve_player_maps_through_id_map_and_board_lookups():
    id_map = {"9999": 42}
    pid, name, pos = ls.resolve_player(9999, id_map, board_names={42: "Star Player"}, board_positions={42: "PG"})
    assert (pid, name, pos) == (42, "Star Player", "PG")


def test_resolve_player_unmapped_is_all_none():
    assert ls.resolve_player(1234, {"9999": 42}) == (None, None, None)


def test_resolve_player_no_id_map_is_all_none():
    assert ls.resolve_player(9999, None) == (None, None, None)


def test_build_detected_picks_attributes_and_resolves():
    new_picks = [
        {"overall_pick": 1, "round": 1, "espn_player_id": 9999, "team_id": 5},
        {"overall_pick": 2, "round": 1, "espn_player_id": 8888, "team_id": 6},
    ]
    id_map = {"9999": 42}
    detected = ls.build_detected_picks(new_picks, my_team_id=5, id_map=id_map,
                                       board_names={42: "Star Player"}, board_positions={42: "PG"})
    assert detected[0].drafted_by == ME and detected[0].player_id == 42 and detected[0].name == "Star Player"
    assert detected[1].drafted_by == OPPONENT and detected[1].player_id is None  # unresolved


# --------------------------------------------------------------------------- apply_detected_picks

def _dp(player_id, name, drafted_by, position=None, espn_id=0, overall=1):
    return ls.DetectedPick(overall_pick=overall, round=1, espn_player_id=espn_id or player_id or 0,
                           team_id=1, drafted_by=drafted_by, player_id=player_id, name=name, position=position)


def test_apply_detected_picks_marks_resolved_players():
    state = DraftState()
    detected = [_dp(1, "Player A", OPPONENT), _dp(2, "Player B", ME)]
    new_state, applied, duplicate, unresolved = ls.apply_detected_picks(state, detected)
    assert new_state.drafted_ids == {1, 2}
    assert len(applied) == 2 and not duplicate and not unresolved
    assert new_state.opponent_ids == {1}
    assert new_state.my_ids == {2}


def test_apply_detected_picks_dedups_against_a_prior_manual_mark():
    """The core anti-double-count requirement: a player already marked drafted by hand (or an
    earlier poll) must be skipped, not drafted a second time (which would raise DraftError if not
    guarded)."""
    state = draft_player(DraftState(), 1, "Player A", "PG", OPPONENT)  # manual click already logged this
    detected = [_dp(1, "Player A", OPPONENT), _dp(2, "Player B", ME)]
    new_state, applied, duplicate, unresolved = ls.apply_detected_picks(state, detected)
    assert len(new_state) == 2  # not 3: player 1 was not re-added
    assert [d.player_id for d in applied] == [2]
    assert [d.player_id for d in duplicate] == [1]
    assert not unresolved


def test_apply_detected_picks_surfaces_unresolved_without_touching_state():
    state = DraftState()
    detected = [_dp(None, None, OPPONENT, espn_id=555)]
    new_state, applied, duplicate, unresolved = ls.apply_detected_picks(state, detected)
    assert len(new_state) == 0
    assert not applied and not duplicate
    assert unresolved and unresolved[0].espn_player_id == 555


def test_apply_detected_picks_never_double_counts_a_repeated_batch():
    state = DraftState()
    detected = [_dp(1, "Player A", OPPONENT)]
    state, applied, _, _ = ls.apply_detected_picks(state, detected)
    assert len(applied) == 1
    # Simulate the same pick surviving into a second poll's detected batch (e.g. a bug upstream):
    # applying it again must be a no-op dedup, never a second Pick.
    state, applied2, duplicate2, _ = ls.apply_detected_picks(state, detected)
    assert not applied2
    assert [d.player_id for d in duplicate2] == [1]
    assert len(state) == 1


# --------------------------------------------------------------------------- should_poll

def test_should_poll_true_when_never_synced():
    assert ls.should_poll(None, datetime.now(timezone.utc)) is True


def test_should_poll_false_within_interval_true_after():
    now = datetime(2026, 10, 17, 12, 0, 0, tzinfo=timezone.utc)
    last = now - timedelta(seconds=5)
    assert ls.should_poll(last, now, min_interval=20) is False
    later = now + timedelta(seconds=16)
    assert ls.should_poll(last, later, min_interval=20) is True


# --------------------------------------------------------------------------- poll_lock / sync_once

def test_poll_lock_second_acquire_fails_until_released(tmp_path):
    path = tmp_path / "league.lock"
    lock_a = ls.poll_lock(path)
    lock_a.acquire()
    lock_b = ls.poll_lock(path)
    with pytest.raises(LockBusy):
        lock_b.acquire()
    lock_a.release()
    lock_b.acquire()  # now free
    lock_b.release()


def test_sync_once_returns_locked_out_when_another_session_holds_the_lock(tmp_path):
    lock_path = tmp_path / "league.lock"
    other = ls.poll_lock(lock_path)
    other.acquire()
    try:
        client, session = make_client(tmp_path, [FakeResponse(200, draft_payload([]))])
        result = ls.sync_once(client, 1, 2027, seen_overall_picks=frozenset(), my_team_id=None,
                              lock_path=lock_path)
        assert result.locked_out is True
        assert result.ok is True
        assert session.calls == []  # never reached the network: the lock was checked first
    finally:
        other.release()


def test_sync_once_happy_path_detects_new_picks(tmp_path):
    payload = draft_payload([pick(1, 5, 9999), pick(2, 6, None)])  # pick 2 unfilled
    client, session = make_client(tmp_path, [FakeResponse(200, payload)])
    result = ls.sync_once(client, 1, 2027, seen_overall_picks=frozenset(), my_team_id=5,
                          id_map={"9999": 42}, board_names={42: "Star Player"})
    assert result.ok is True
    assert result.locked_out is False
    assert len(result.detected) == 1
    d = result.detected[0]
    assert d.player_id == 42 and d.drafted_by == ME and d.name == "Star Player"
    assert result.seen_overall_picks == frozenset({1})


def test_sync_once_repeated_call_with_updated_seen_finds_nothing_new(tmp_path):
    payload = draft_payload([pick(1, 5, 9999)])
    client, session = make_client(tmp_path, [FakeResponse(200, payload), FakeResponse(200, payload)])
    first = ls.sync_once(client, 1, 2027, seen_overall_picks=frozenset(), my_team_id=None)
    second = ls.sync_once(client, 1, 2027, seen_overall_picks=first.seen_overall_picks, my_team_id=None)
    assert len(first.detected) == 1
    assert len(second.detected) == 0


def test_sync_once_always_bypasses_the_on_disk_cache(tmp_path):
    """Regression guard: ``ESPNLeagueClient`` caches a request's response on disk indefinitely by
    default (same league/season/views -> same cache file, served forever without hitting the
    network again). A live poll must never be served the *first* poll's stale snapshot -- it has
    to re-fetch every time, so ``sync_once`` must pass ``refresh=True`` through to
    ``fetch_draft_detail``. Proven here by two ``sync_once`` calls against the exact same
    (league_id, season_id): the fake session must see two real requests, not one served from
    cache."""
    payload_1 = draft_payload([pick(1, 5, 9999)])
    payload_2 = draft_payload([pick(1, 5, 9999), pick(2, 6, 8888)])
    client, session = make_client(tmp_path, [FakeResponse(200, payload_1), FakeResponse(200, payload_2)])
    first = ls.sync_once(client, 1, 2027, seen_overall_picks=frozenset(), my_team_id=None)
    second = ls.sync_once(client, 1, 2027, seen_overall_picks=first.seen_overall_picks, my_team_id=None)
    assert len(session.calls) == 2, "the second poll was served from the on-disk cache, not re-fetched"
    assert len(first.detected) == 1
    assert [p.overall_pick for p in second.detected] == [2]  # the newly-appeared pick 2, not pick 1 again


def test_sync_once_never_raises_on_auth_error_degrades_to_error_result(tmp_path):
    client, _ = make_client(tmp_path, [FakeResponse(401, b"nope")])
    result = ls.sync_once(client, 1, 2027, seen_overall_picks=frozenset(), my_team_id=None)
    assert result.ok is False
    assert result.error and "401" in result.error
    assert result.detected == ()


def test_sync_once_never_raises_on_5xx_exhausted_retries(tmp_path):
    client, _ = make_client(
        tmp_path,
        [FakeResponse(500), FakeResponse(500), FakeResponse(500), FakeResponse(500)],
        max_retries=3, backoff_base=1,
    )
    result = ls.sync_once(client, 1, 2027, seen_overall_picks=frozenset(), my_team_id=None)
    assert result.ok is False
    assert result.error


# --------------------------------------------------------------------------- load_id_map

def test_load_id_map_returns_none_when_table_missing(tmp_path):
    assert ls.load_id_map(tmp_path) is None


def test_load_id_map_filters_to_espn_source_and_keys_by_source_id_str(tmp_path, monkeypatch):
    df = pd.DataFrame({
        "player_id": [1, 2, 3],
        "source": ["espn", "espn", "spotrac"],
        "source_id": ["100", "200", "300"],
        "source_name": ["A", "B", "C"],
        "match_method": ["exact", "exact", "exact"],
        "confidence": [1.0, 1.0, 1.0],
    })

    def fake_table_exists(name, base=None):
        return name == "player_id_map"

    def fake_read_table(name, base=None, validate=True):
        return df

    monkeypatch.setattr("src.store.table_exists", fake_table_exists)
    monkeypatch.setattr("src.store.read_table", fake_read_table)

    id_map = ls.load_id_map(tmp_path)
    assert id_map == {"100": 1, "200": 2}  # spotrac row excluded


def test_load_dotenv_file_sets_missing_but_never_overrides(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("ESPN_LEAGUE_ID=123\nESPN_TEAM_ID=7\n", encoding="utf-8")
    monkeypatch.delenv("ESPN_LEAGUE_ID", raising=False)
    monkeypatch.setenv("ESPN_TEAM_ID", "1")
    assert ls.load_dotenv_file(tmp_path) is True
    assert ls.league_id_from_env() == 123
    assert ls.team_id_from_env() == 1          # a real env var wins over the file


def test_load_dotenv_file_missing_file_is_a_noop(tmp_path):
    assert ls.load_dotenv_file(tmp_path) is False
