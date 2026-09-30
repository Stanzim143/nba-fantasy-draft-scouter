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
