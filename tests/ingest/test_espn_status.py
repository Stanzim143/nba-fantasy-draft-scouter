"""espn_status: dated, append-only ESPN injury-status archive."""
from datetime import date

import pandas as pd
import pytest

from src.ingest import espn_status as es

RAW = [{"id": 1, "fullName": "A One", "proTeamId": 3, "injuryStatus": "OUT", "injured": True, "lastNewsDate": 1790000000000},
       {"id": 2, "fullName": "B Two", "proTeamId": 4, "injuryStatus": None, "injured": False},
       {"id": None, "fullName": "skipped"}]
IDMAP = pd.DataFrame({"source": ["espn"], "source_id": ["1"], "player_id": [111]})


def test_frame_maps_ids_and_defaults_status():
    f = es.status_frame(RAW, "2026-27", date(2026, 9, 24), IDMAP)
    assert len(f) == 2 and f.loc[0, "player_id"] == 111 and pd.isna(f.loc[1, "player_id"])
    assert list(f["injury_status"]) == ["OUT", "UNKNOWN"] and pd.notna(f.loc[0, "last_news_date"]) and pd.isna(f.loc[1, "last_news_date"])


def test_days_append_and_a_rerun_replaces_the_day(tmp_path):
    d1 = es.status_frame(RAW, "2026-27", date(2026, 9, 24), IDMAP)
    d2 = es.status_frame(RAW, "2026-27", date(2026, 9, 25), IDMAP)
    es.write_snapshot(d1, tmp_path)
    es.write_snapshot(d2, tmp_path)
    es.write_snapshot(d2, tmp_path)
    all_ = es.read_status(tmp_path)
    assert len(all_) == 4 and len(es.latest_status(all_)) == 2
    with pytest.raises(ValueError):
        es.write_snapshot(pd.concat([d1, d1]), tmp_path)


def test_missing_archive_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        es.read_status(tmp_path)


def _client_with_payload(tmp_path, payload):
    import json
    import os
    import time

    from ingest_fakes import FakeClock, FakeResponse, FakeSession
    from src.ingest import espn_adp
    from src.ingest.http_cache import CachedHttpClient

    clock = FakeClock()
    c = CachedHttpClient(tmp_path / "raw", session=FakeSession([FakeResponse(200, json.dumps(payload))]), clock=clock,
                         sleep=clock.sleep, offline=False, min_interval=0, jitter=0)
    sid = espn_adp.espn_season_id("2026-27")
    c.get_json(f"players/{sid}", "https://x/", espn_adp.players_params())
    c.offline = True
    return c, c.cache_path(f"players/{sid}", espn_adp.players_params()), os, time


def test_snapshot_is_dated_by_cache_fetch_time_not_today(tmp_path):
    c, path, os, _ = _client_with_payload(tmp_path, RAW)
    os.utime(path, (1_790_000_000, 1_790_000_000))  # 2026-09-21 ~
    from datetime import datetime
    fetched = datetime.fromtimestamp(1_790_000_000)
    es.run_ingest("2026-27", c, tmp_path / "d", now=fetched, log=lambda *_: None)
    assert es.read_status(tmp_path / "d")["snapshot_date"].iloc[0].date() == fetched.date()


def test_stale_cached_payload_is_refused(tmp_path):
    from datetime import datetime, timedelta

    from src.ingest import espn_adp
    c, path, os, _ = _client_with_payload(tmp_path, RAW)
    os.utime(path, (1_790_000_000, 1_790_000_000))
    with pytest.raises(espn_adp.EspnAdpError, match="refresh it first"):
        es.run_ingest("2026-27", c, tmp_path / "d", now=datetime.fromtimestamp(1_790_000_000) + timedelta(days=3),
                      log=lambda *_: None)
    assert not es.table_path(tmp_path / "d").exists()


def test_empty_payload_is_a_clean_error(tmp_path):
    from src.ingest import espn_adp
    c, *_ = _client_with_payload(tmp_path, [])
    with pytest.raises(espn_adp.EspnAdpError, match="no usable players"):
        es.run_ingest("2026-27", c, tmp_path / "d", log=lambda *_: None)


def test_main_puts_the_raw_cache_under_data_dir(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(es, "run_ingest", lambda season, client, base, **kw: seen.append(client.cache_dir) or
                        {"players": 0, "mapped": 0})
    assert es.main(["--data-dir", str(tmp_path), "--offline"]) == 0
    assert seen[0] == tmp_path / "raw" / "espn"
