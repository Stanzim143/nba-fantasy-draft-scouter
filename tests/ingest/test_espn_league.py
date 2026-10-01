"""ESPN league sync: client (cache/offline/auth/retries), parsing, reconciliation, CLI. No network."""
from __future__ import annotations

import json

import pytest
import yaml

import espn_league_fakes as el_fakes
from espn_league_fakes import (
    FakeClock, FakeResponse, FakeSession, make_draft_payload, make_free_agent, make_free_agents_payload,
    make_roster_entry, make_settings_payload, make_team, make_transactions_payload,
)
from src.ingest import espn_league as el

LEAGUE_ID = 1234567890
SEASON_ID = 2027


def make_client(tmp_path, script, **kw):
    clock = FakeClock()
    kw.setdefault("min_interval", 0)
    kw.setdefault("offline", False)
    session = FakeSession(script)
    client = el.ESPNLeagueClient(tmp_path / "raw", session=session, clock=clock, sleep=clock.sleep, **kw)
    return client, session, clock


def minimal_cfg(**overrides) -> dict:
    cfg = {
        "league": {
            "name": "Example League", "platform": "espn", "season": "2026-27", "format": "h2h_points",
            "teams": 10,
            "draft": {"type": "snake", "date": None, "seconds_per_pick": 90, "order": "manual", "pick_trading": True},
            "roster": {"size": 13, "starters": {"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1, "G": 1, "F": 1, "UTIL": 3},
                       "bench": 3, "ir": 1},
            "lineups": "daily",
            "acquisition": {"system": "waivers", "season_limit": None, "matchup_limit": 7, "waiver_period_days": 1},
            "trades": {"limit": None, "deadline": "2027-03-13", "review_days": 2, "votes_to_veto": 4},
            "keepers": False,
            "schedule": {"weeks_per_matchup": 1, "regular_season_matchups": 20, "playoff_teams": 8,
                        "playoff_rounds_weeks": [1, 1, 1], "playoff_reseeding": False},
        },
        "scoring": {"PTS": 1, "REB": 1, "AST": 2, "STL": 4, "BLK": 4, "TO": -2, "FGM": 2, "FGA": -1,
                    "FTM": 1, "FTA": -1, "3PM": 1},
    }
    cfg["league"].update(overrides)
    return cfg


# --------------------------------------------------------------------------- client: cache / offline / errors

def test_cache_key_stable_and_distinguishes_requests():
    a = el.cache_key(["mSettings"], None, None)
    b = el.cache_key(["mSettings"], {"scoringPeriodId": 1}, None)
    c = el.cache_key(["mSettings"], None, {"X-Fantasy-Filter": "{}"})
    assert len({a, b, c}) == 3


def test_miss_downloads_then_hit_never_touches_network(tmp_path):
    payload = make_settings_payload()
    client, session, _ = make_client(tmp_path, [FakeResponse(200, payload)])
    a = client.get(LEAGUE_ID, SEASON_ID, ["mSettings", "mTeam"])
    b = client.get(LEAGUE_ID, SEASON_ID, ["mSettings", "mTeam"])
    assert a == b == payload
    assert len(session.calls) == 1
    assert client.stats.cache_hits == 1 and client.stats.network_requests == 1


def test_request_uses_repeated_view_params_and_descriptive_ua(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(200, make_settings_payload())])
    client.get(LEAGUE_ID, SEASON_ID, ["mSettings", "mTeam", "mRoster"])
    call = session.calls[0]
    assert set(call["params"]["view"]) == {"mSettings", "mTeam", "mRoster"}
    assert "nba-fantasy-2026-league-sync" in call["headers"]["User-Agent"]
    assert str(LEAGUE_ID) in call["url"] and str(SEASON_ID) in call["url"]


def test_cache_survives_new_client_instance(tmp_path):
    payload = make_settings_payload()
    c1, _, _ = make_client(tmp_path, [FakeResponse(200, payload)])
    c1.get(LEAGUE_ID, SEASON_ID, ["mSettings"])
    c2, s2, _ = make_client(tmp_path, [FakeResponse(500)])
    assert c2.get(LEAGUE_ID, SEASON_ID, ["mSettings"]) == payload
    assert s2.calls == []


def test_offline_cache_hit_ok_cache_miss_raises(tmp_path):
    payload = make_settings_payload()
    c1, _, _ = make_client(tmp_path, [FakeResponse(200, payload)])
    c1.get(LEAGUE_ID, SEASON_ID, ["mSettings"])
    offline, session, _ = make_client(tmp_path, [], offline=True)
    assert offline.get(LEAGUE_ID, SEASON_ID, ["mSettings"]) == payload
    assert session.calls == []
    with pytest.raises(el.ESPNOfflineCacheMiss):
        offline.get(LEAGUE_ID, SEASON_ID, ["mDraftDetail"])


def test_401_raises_auth_error_and_is_never_retried(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(401, b"nope")])
    with pytest.raises(el.ESPNAuthError) as exc_info:
        client.get(LEAGUE_ID, SEASON_ID, ["mSettings"])
    assert exc_info.value.status == 401
    assert len(session.calls) == 1  # not retried


def test_403_raises_auth_error(tmp_path):
    client, _, _ = make_client(tmp_path, [FakeResponse(403, b"forbidden")])
    with pytest.raises(el.ESPNAuthError) as exc_info:
        client.get(LEAGUE_ID, SEASON_ID, ["mSettings"])
    assert exc_info.value.status == 403


def test_404_raises_not_found(tmp_path):
    client, _, _ = make_client(tmp_path, [FakeResponse(404, b"not found")])
    with pytest.raises(el.ESPNNotFoundError):
        client.get(LEAGUE_ID, SEASON_ID, ["mSettings"])


def test_retryable_status_is_retried_then_succeeds(tmp_path):
    payload = make_settings_payload()
    client, session, clock = make_client(tmp_path, [FakeResponse(503), FakeResponse(200, payload)])
    result = client.get(LEAGUE_ID, SEASON_ID, ["mSettings"])
    assert result == payload
    assert len(session.calls) == 2
    assert client.stats.retries == 1


def test_rate_limit_sleeps_between_network_calls(tmp_path):
    client, _, clock = make_client(tmp_path, [FakeResponse(200, make_settings_payload()),
                                              FakeResponse(200, make_draft_payload())], min_interval=5)
    client.get(LEAGUE_ID, SEASON_ID, ["mSettings"])
    client.get(LEAGUE_ID, SEASON_ID, ["mDraftDetail"])
    assert clock.sleeps and clock.sleeps[0] == pytest.approx(5)


def test_corrupt_cache_quarantined_online_raised_offline(tmp_path):
    client, _, _ = make_client(tmp_path, [FakeResponse(200, make_settings_payload())])
    path = client.cache_path(LEAGUE_ID, SEASON_ID, ["mSettings"])
    path.parent.mkdir(parents=True)
    path.write_bytes(b"{not json")
    # online: quarantined and re-fetched
    result = client.get(LEAGUE_ID, SEASON_ID, ["mSettings"])
    assert result == make_settings_payload()
    assert path.with_suffix(".corrupt").exists()
    # offline: corrupt cache is fatal
    bad_path = client.cache_path(LEAGUE_ID, SEASON_ID, ["mDraftDetail"])
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_bytes(b"{not json")
    offline, _, _ = make_client(tmp_path, [], offline=True)
    with pytest.raises(el.ESPNCacheCorruptError):
        offline.get(LEAGUE_ID, SEASON_ID, ["mDraftDetail"])


# --------------------------------------------------------------------------- parsing: settings

def test_parse_settings_team_count_and_scoring():
    payload = make_settings_payload(team_count=13)
    settings = el.parse_settings(payload)
    assert settings["team_count"] == 13
    assert settings["size"] == 13
    assert settings["scoring"] == {"PTS": 1.0, "REB": 1.0, "AST": 2.0, "STL": 4.0, "BLK": 4.0, "TO": -2.0,
                                   "FGM": 2.0, "FGA": -1.0, "FTM": 1.0, "FTA": -1.0, "3PM": 1.0}


def test_parse_settings_roster_slots():
    settings = el.parse_settings(make_settings_payload())
    assert settings["starters"] == {"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1, "G": 1, "F": 1, "UTIL": 3}
    assert settings["bench"] == 3
    assert settings["ir"] == 1
    assert settings["roster_size"] == 13


def test_parse_settings_draft_acquisition_trades():
    settings = el.parse_settings(make_settings_payload(drafted=False, matchup_acquisition_limit=1.0,
                                                        waiver_hours=24, trade_max=-1, veto_votes=4,
                                                        review_hours=48))
    assert settings["draft"] == {
        "type": "SNAKE", "order_type": "MANUAL", "seconds_per_pick": 90, "pick_trading": True,
        "pick_order": [t for t in range(1, 14)], "keeper_count": 0, "drafted": False, "in_progress": False,
    }
    assert settings["acquisition"]["waiver_hours"] == 24
    assert settings["acquisition"]["season_limit"] is None  # -1 -> None
    assert settings["trades"]["limit"] is None  # -1 -> None
    assert settings["trades"]["votes_to_veto"] == 4


def test_parse_settings_handles_missing_keys_gracefully():
    # a maximally sparse payload should not crash
    settings = el.parse_settings({})
    assert settings["team_count"] is None
    assert settings["starters"] == {}
    assert settings["scoring"] == {}


# --------------------------------------------------------------------------- parsing: teams / rosters

def test_parse_teams_and_members_drops_real_names():
    payload = make_settings_payload(teams=[
        make_team(1, "Amsterdammer Abzocker", "OWNER-A"),
        make_team(2, "Copenhagen Beers", "OWNER-B"),
    ], members=[{"id": "OWNER-A", "displayName": "ESPNFAN123", "firstName": "Real", "lastName": "Name"}])
    teams = el.parse_teams(payload)
    assert [t["name"] for t in teams] == ["Amsterdammer Abzocker", "Copenhagen Beers"]
    assert teams[0]["owner_ids"] == ["OWNER-A"]
    members = el.parse_members(payload)
    assert members == [{"id": "OWNER-A", "display_name": "ESPNFAN123"}]
    assert "firstName" not in members[0] and "lastName" not in members[0]


def test_parse_teams_empty_roster_before_draft():
    payload = make_settings_payload(teams=[make_team(1, "Team 1", "OWNER-1", roster_entries=[])])
    teams = el.parse_teams(payload)
    assert teams[0]["roster"] == []


def test_parse_teams_populated_roster_after_draft():
    entry = make_roster_entry(9999, "Star Player", lineup_slot_id=0, default_position_id=1, pro_team_id=5)
    payload = make_settings_payload(teams=[make_team(1, "Team 1", "OWNER-1", roster_entries=[entry])])
    teams = el.parse_teams(payload)
    [player] = teams[0]["roster"]
    assert player["espn_player_id"] == 9999
    assert player["name"] == "Star Player"
    assert player["lineup_slot"] == "PG"
    assert player["pro_team_id"] == 5


# --------------------------------------------------------------------------- parsing: draft / free agents / tx

def test_parse_draft_undrafted_picks_are_none():
    payload = make_draft_payload(drafted=False, team_ids=[16, 27, 2])
    draft = el.parse_draft(payload)
    assert draft["drafted"] is False
    assert [p["espn_player_id"] for p in draft["picks"]] == [None, None, None]
    assert [p["team_id"] for p in draft["picks"]] == [16, 27, 2]


def test_parse_draft_completed_picks_have_players():
    payload = make_draft_payload(drafted=True, picks=[
        {"id": 1, "overallPickNumber": 1, "roundId": 1, "roundPickNumber": 1, "teamId": 16,
         "playerId": 5104157, "keeper": False},
    ])
    draft = el.parse_draft(payload)
    assert draft["drafted"] is True
    assert draft["picks"][0]["espn_player_id"] == 5104157


def test_parse_free_agents():
    payload = make_free_agents_payload([make_free_agent(1, "Player A", adp=3.0, pct_owned=99.0)])
    fas = el.parse_free_agents(payload)
    assert fas == [{
        "espn_player_id": 1, "name": "Player A", "default_position_id": 1, "pro_team_id": 1,
        "injury_status": "ACTIVE", "percent_owned": 99.0, "adp": 3.0, "on_team_id": 0,
    }]


def test_parse_transactions_missing_key_is_empty_list():
    assert el.parse_transactions(make_transactions_payload(transactions=None)) == []


def test_parse_transactions_present():
    payload = make_transactions_payload(transactions=[
        {"id": "t1", "type": "WAIVER", "status": "EXECUTED", "teamId": 1, "scoringPeriodId": 5, "items": []},
    ])
    txs = el.parse_transactions(payload)
    assert txs == [{"id": "t1", "type": "WAIVER", "status": "EXECUTED", "team_id": 1,
                    "scoring_period_id": 5, "items": []}]


# --------------------------------------------------------------------------- reconciliation

def test_reconcile_flags_team_count_mismatch():
    cfg = minimal_cfg(teams=10)
    settings = el.parse_settings(make_settings_payload(team_count=13))
    rows = el.reconcile(cfg, settings)
    row = next(r for r in rows if r["field"] == "teams")
    assert row["config"] == 10 and row["espn"] == 13 and row["match"] is False


def test_reconcile_matches_when_config_is_correct():
    cfg = minimal_cfg(teams=13)
    settings = el.parse_settings(make_settings_payload(team_count=13))
    rows = el.reconcile(cfg, settings)
    row = next(r for r in rows if r["field"] == "teams")
    assert row["match"] is True


def test_reconcile_scoring_matches_real_league_config():
    cfg = minimal_cfg(teams=13)
    settings = el.parse_settings(make_settings_payload())
    rows = el.reconcile(cfg, settings)
    scoring_rows = [r for r in rows if r["field"].startswith("scoring.")]
    assert scoring_rows and all(r["match"] for r in scoring_rows)


def test_reconcile_roster_starters_and_bench_and_ir():
    cfg = minimal_cfg(teams=13)
    settings = el.parse_settings(make_settings_payload())
    rows = {r["field"]: r for r in el.reconcile(cfg, settings)}
    assert rows["roster.starters"]["match"] is True
    assert rows["roster.bench"]["match"] is True
    assert rows["roster.ir"]["match"] is True


def test_reconcile_flags_mismatched_scoring():
    cfg = minimal_cfg(teams=13)
    bad_items = [dict(i) for i in el_fakes.DEFAULT_SCORING_ITEMS]
    for item in bad_items:
        if item["statId"] == 0:  # PTS
            item["points"] = 2.0  # config says 1
    settings = el.parse_settings(make_settings_payload(scoring_items=bad_items))
    rows = {r["field"]: r for r in el.reconcile(cfg, settings)}
    assert rows["scoring.PTS"]["match"] is False
    assert rows["scoring.PTS"]["config"] == 1 and rows["scoring.PTS"]["espn"] == 2.0


def test_reconcile_trades_deadline_converts_epoch():
    cfg = minimal_cfg(teams=13)
    settings = el.parse_settings(make_settings_payload(deadline_epoch_ms=1804870800000))
    rows = {r["field"]: r for r in el.reconcile(cfg, settings)}
    assert rows["trades.deadline"]["espn"] in ("2027-03-12", "2027-03-13")  # tz-dependent day boundary


def test_reconcile_matchup_limit_carries_a_unit_caveat_note():
    cfg = minimal_cfg(teams=13)
    settings = el.parse_settings(make_settings_payload())
    rows = {r["field"]: r for r in el.reconcile(cfg, settings)}
    assert rows["acquisition.matchup_limit"]["note"]  # non-empty: units differ, flagged explicitly


# --------------------------------------------------------------------------- CLI

def cli_client(tmp_path, script, **kw):
    clock = FakeClock()
    session = FakeSession(script)
    return el.ESPNLeagueClient(tmp_path / "cache", session=session, clock=clock, sleep=clock.sleep,
                               offline=False, min_interval=0, **kw)


def test_cli_end_to_end_writes_report_and_prints_reconciliation(tmp_path, capsys):
    data_dir = tmp_path / "data"
    cfg_path = tmp_path / "league.yaml"
    cfg_path.write_text(yaml.safe_dump(minimal_cfg(teams=10)), encoding="utf-8")
    settings_payload = make_settings_payload(team_count=13, drafted=False)
    script = [
        FakeResponse(200, settings_payload),
        FakeResponse(200, make_draft_payload(drafted=False)),
        FakeResponse(200, make_free_agents_payload()),
        FakeResponse(200, make_transactions_payload(transactions=None)),
    ]
    client = cli_client(tmp_path, script)
    rc = el.main(["--league-id", str(LEAGUE_ID), "--season", "2026-27", "--data-dir", str(data_dir),
                 "--config", str(cfg_path)], client=client)
    assert rc == 0
    out_path = data_dir / "processed" / "espn_league" / f"{LEAGUE_ID}_2027.json"
    assert out_path.exists()
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["settings"]["team_count"] == 13
    assert report["draft_completed"] is False
    assert any(r["field"] == "teams" and not r["match"] for r in report["reconciliation"])

    captured = capsys.readouterr()
    assert "13" in captured.out
    assert "DIFFERS" in captured.out
    assert "NOT completed yet" in captured.out


def test_cli_no_draft_yet_case_does_not_crash(tmp_path):
    data_dir = tmp_path / "data"
    cfg_path = tmp_path / "league.yaml"
    cfg_path.write_text(yaml.safe_dump(minimal_cfg(teams=13)), encoding="utf-8")
    teams = [make_team(i, f"Team {i}", f"OWNER-{i}", roster_entries=[]) for i in range(1, 14)]
    settings_payload = make_settings_payload(teams=teams, drafted=False)
    script = [
        FakeResponse(200, settings_payload),
        FakeResponse(200, make_draft_payload(drafted=False, team_ids=[t["id"] for t in teams])),
        FakeResponse(200, make_free_agents_payload()),
        FakeResponse(200, make_transactions_payload(transactions=None)),
    ]
    client = cli_client(tmp_path, script)
    rc = el.main(["--league-id", str(LEAGUE_ID), "--season", "2026-27", "--data-dir", str(data_dir),
                 "--config", str(cfg_path)], client=client)
    assert rc == 0
    out_path = data_dir / "processed" / "espn_league" / f"{LEAGUE_ID}_2027.json"
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert all(t["roster"] == [] for t in report["teams"])
    assert report["draft"]["picks"] and all(p["espn_player_id"] is None for p in report["draft"]["picks"])


def test_cli_private_league_reports_clearly_and_stops(tmp_path, capsys):
    cfg_path = tmp_path / "league.yaml"
    cfg_path.write_text(yaml.safe_dump(minimal_cfg(teams=13)), encoding="utf-8")
    client = cli_client(tmp_path, [FakeResponse(403, b"forbidden")])
    rc = el.main(["--league-id", str(LEAGUE_ID), "--season", "2026-27", "--data-dir", str(tmp_path / "data"),
                 "--config", str(cfg_path)], client=client)
    assert rc == 3
    err = capsys.readouterr().err
    assert "cookies" in err.lower() and ".env" in err
    # no output file should have been written -- we stop before pulling anything else
    assert not (tmp_path / "data" / "processed" / "espn_league").exists()


def test_cli_league_not_found(tmp_path, capsys):
    cfg_path = tmp_path / "league.yaml"
    cfg_path.write_text(yaml.safe_dump(minimal_cfg(teams=13)), encoding="utf-8")
    client = cli_client(tmp_path, [FakeResponse(404, b"nope"), FakeResponse(404, b"nope")])
    rc = el.main(["--league-id", "999999999", "--season", "2026-27", "--data-dir", str(tmp_path / "data"),
                 "--config", str(cfg_path)], client=client)
    assert rc == 4
    assert "not found" in capsys.readouterr().err.lower()


def test_cli_missing_league_id_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("ESPN_LEAGUE_ID", raising=False)
    rc = el.main(["--season", "2026-27", "--data-dir", str(tmp_path / "data")])
    assert rc == 2


def test_cli_league_id_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ESPN_LEAGUE_ID", str(LEAGUE_ID))
    cfg_path = tmp_path / "league.yaml"
    cfg_path.write_text(yaml.safe_dump(minimal_cfg(teams=13)), encoding="utf-8")
    script = [
        FakeResponse(200, make_settings_payload()),
        FakeResponse(200, make_draft_payload()),
        FakeResponse(200, make_free_agents_payload()),
        FakeResponse(200, make_transactions_payload()),
    ]
    client = cli_client(tmp_path, script)
    rc = el.main(["--season", "2026-27", "--data-dir", str(tmp_path / "data"), "--config", str(cfg_path)],
                client=client)
    assert rc == 0


def test_cli_offline_without_cache_reports_error(tmp_path, capsys):
    cfg_path = tmp_path / "league.yaml"
    cfg_path.write_text(yaml.safe_dump(minimal_cfg(teams=13)), encoding="utf-8")
    client = el.ESPNLeagueClient(tmp_path / "cache", offline=True)
    rc = el.main(["--league-id", str(LEAGUE_ID), "--season", "2026-27", "--data-dir", str(tmp_path / "data"),
                 "--config", str(cfg_path)], client=client)
    assert rc == 5
    assert "offline" in capsys.readouterr().err.lower()


def test_cli_offline_serves_from_cache(tmp_path):
    cfg_path = tmp_path / "league.yaml"
    cfg_path.write_text(yaml.safe_dump(minimal_cfg(teams=13)), encoding="utf-8")
    data_dir = tmp_path / "data"
    script = [
        FakeResponse(200, make_settings_payload()),
        FakeResponse(200, make_draft_payload()),
        FakeResponse(200, make_free_agents_payload()),
        FakeResponse(200, make_transactions_payload()),
    ]
    online_client = cli_client(tmp_path, script)
    rc = el.main(["--league-id", str(LEAGUE_ID), "--season", "2026-27", "--data-dir", str(data_dir),
                 "--config", str(cfg_path)], client=online_client)
    assert rc == 0

    offline_client = el.ESPNLeagueClient(tmp_path / "cache", offline=True)
    rc2 = el.main(["--league-id", str(LEAGUE_ID), "--season", "2026-27", "--data-dir", str(data_dir),
                  "--config", str(cfg_path), "--offline"], client=offline_client)
    assert rc2 == 0


def test_cli_falls_back_to_prior_season_id_when_current_not_found(tmp_path, capsys):
    cfg_path = tmp_path / "league.yaml"
    cfg_path.write_text(yaml.safe_dump(minimal_cfg(teams=13)), encoding="utf-8")
    script = [
        FakeResponse(404, b"nope"),  # 2027 not found
        FakeResponse(200, make_settings_payload()),  # 2026 found
        FakeResponse(200, make_draft_payload()),
        FakeResponse(200, make_free_agents_payload()),
        FakeResponse(200, make_transactions_payload()),
    ]
    client = cli_client(tmp_path, script)
    rc = el.main(["--league-id", str(LEAGUE_ID), "--season", "2026-27", "--data-dir", str(tmp_path / "data"),
                 "--config", str(cfg_path)], client=client)
    assert rc == 0
    out_path = tmp_path / "data" / "processed" / "espn_league" / f"{LEAGUE_ID}_2026.json"
    assert out_path.exists()
    assert "used 2026 instead" in capsys.readouterr().out


def test_cli_transactions_failure_is_a_warning_not_a_crash(tmp_path, capsys):
    cfg_path = tmp_path / "league.yaml"
    cfg_path.write_text(yaml.safe_dump(minimal_cfg(teams=13)), encoding="utf-8")
    script = [
        FakeResponse(200, make_settings_payload()),
        FakeResponse(200, make_draft_payload()),
        FakeResponse(200, make_free_agents_payload()),
        FakeResponse(500, b"server error"),
        FakeResponse(500, b"server error"),
        FakeResponse(500, b"server error"),
        FakeResponse(500, b"server error"),
    ]
    client = cli_client(tmp_path, script, max_retries=3, backoff_base=1)
    rc = el.main(["--league-id", str(LEAGUE_ID), "--season", "2026-27", "--data-dir", str(tmp_path / "data"),
                 "--config", str(cfg_path)], client=client)
    assert rc == 0
    assert "warning" in capsys.readouterr().err.lower()


def test_parse_settings_keeps_a_real_zero_limit():
    s = el.parse_settings(make_settings_payload(trade_max=0, season_limit=0))
    assert s["acquisition"]["season_limit"] == 0 and s["trades"]["limit"] == 0
    s = el.parse_settings(make_settings_payload(trade_max=None, season_limit=None))
    assert s["acquisition"]["season_limit"] is None and s["trades"]["limit"] is None


def test_network_errors_are_retried_then_succeed(tmp_path):
    payload = make_settings_payload()
    client, session, _ = make_client(tmp_path, [ConnectionError("reset"), FakeResponse(200, payload)])
    assert client.get(LEAGUE_ID, SEASON_ID, ["mSettings"]) == payload
    assert len(session.calls) == 2 and client.stats.retries == 1


def test_persistent_network_error_raises_league_error(tmp_path):
    client, _, _ = make_client(tmp_path, [TimeoutError("slow")], max_retries=2)
    with pytest.raises(el.ESPNLeagueError, match="network error"):
        client.get(LEAGUE_ID, SEASON_ID, ["mSettings"])


def test_non_json_200_is_retried_and_never_cached(tmp_path):
    payload = make_settings_payload()
    client, session, _ = make_client(tmp_path, [FakeResponse(200, b"<html>maintenance</html>"), FakeResponse(200, payload)])
    assert client.get(LEAGUE_ID, SEASON_ID, ["mSettings"]) == payload
    bad, _, _ = make_client(tmp_path / "b", [FakeResponse(200, b"<html>")], max_retries=1)
    with pytest.raises(el.ESPNLeagueError, match="non-JSON"):
        bad.get(LEAGUE_ID, SEASON_ID, ["mSettings"])
    assert not bad.cache_path(LEAGUE_ID, SEASON_ID, ["mSettings"]).exists()
