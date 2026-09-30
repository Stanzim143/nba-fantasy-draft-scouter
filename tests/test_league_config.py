from src.value.league import load_league, rostered_players, starter_slots


def test_scoring_matches_espn_settings():
    cfg = load_league()
    assert cfg["scoring"] == {
        "PTS": 1, "REB": 1, "AST": 2, "STL": 4, "BLK": 4, "TO": -2,
        "FGM": 2, "FGA": -1, "FTM": 1, "FTA": -1, "3PM": 1,
    }


def test_roster_is_consistent():
    cfg = load_league()
    assert starter_slots(cfg) == 10
    assert starter_slots(cfg) + cfg["league"]["roster"]["bench"] == cfg["league"]["roster"]["size"]
    assert rostered_players(cfg) == cfg["league"]["teams"] * 13


def test_draft_time_is_set_and_unambiguous():
    from datetime import date, datetime, timedelta, timezone
    from zoneinfo import ZoneInfo

    d = load_league()["league"]["draft"]
    assert d["date"] == date(2026, 10, 17) and d["timezone"] == "Pacific/Auckland"
    start = datetime.fromisoformat(d["start_utc"].replace("Z", "+00:00"))
    assert start == datetime(2026, 10, 16, 18, 0, tzinfo=timezone.utc)
    # the configured local date is the date of the start instant in the configured zone, and it is NZ morning
    nz = start.astimezone(ZoneInfo(d["timezone"]))
    assert nz.date() == d["date"] and nz.hour == 7 and nz.utcoffset() == timedelta(hours=13)
    # ... which is the evening before in CEST (UTC+2)
    cest = start.astimezone(ZoneInfo("Europe/Berlin"))
    assert (cest.date(), cest.hour, cest.utcoffset()) == (date(2026, 10, 16), 20, timedelta(hours=2))
