"""Real-data checks (skip cleanly when the data directory is not populated)."""
from pathlib import Path

import pandas as pd
import pytest

from src.contracts import data_dir
from src.value.league import load_league

pytestmark = pytest.mark.real_data

REAL = Path.home() / "dev-data" / "nba-fantasy-2026"


def _need(name):
    path = REAL / "processed" / name
    if not path.exists():
        pytest.skip(f"{path} not ingested")
    return pd.read_parquet(path)


def test_espn_team_id_table_matches_the_real_team_games_abbreviations_and_ids():
    from src.inseason.schedule import ABBR_OF_TEAM_ID

    tg = _need("team_games.parquet")
    real = tg.drop_duplicates("team_id").set_index("team_id")["team_abbr"]
    latest = tg[tg["season"] == tg["season"].max()].drop_duplicates("team_id").set_index("team_id")["team_abbr"]
    assert set(latest.index) == set(ABBR_OF_TEAM_ID)
    assert all(ABBR_OF_TEAM_ID[t] == a for t, a in latest.items()), (real.to_dict(), ABBR_OF_TEAM_ID)


def test_stored_espn_schedule_is_eighty_games_per_team_and_matches_the_league_shape():
    from src.inseason.schedule import build_calendar, read_schedule, weekly_team_table

    try:
        sch = read_schedule(REAL, "2026-27")
    except FileNotFoundError:
        pytest.skip("schedule_games not ingested")
    per = pd.concat([sch["home_team_id"], sch["away_team_id"]]).value_counts()
    assert len(per) == 30 and per.min() == per.max() == 80
    cal = build_calendar(sch, "2026-27", cfg=load_league())
    if cal.source == "derived":
        assert len(cal.weeks) == 23 and str(cal.last_day) == "2027-04-04"   # ESPN's finalScoringPeriod 167
    wk = weekly_team_table(sch, cal, "2026-09-24")
    assert 2200 < wk["games"].sum() <= 2400   # the fantasy season ends before the last NBA week


def test_realised_2025_26_schedule_rebuilt_from_team_games_is_complete():
    from src.inseason.schedule import from_team_games

    tg = _need("team_games.parquet")
    sch = from_team_games(tg, "2025-26")
    assert len(sch) == 30 * 82 // 2
    assert data_dir()  # the real data dir is only ever read here
