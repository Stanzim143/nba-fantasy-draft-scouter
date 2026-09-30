"""Matchup calendar: ESPN periods when real, the derived fallback otherwise."""
from datetime import date

import pandas as pd

from src.inseason.weeks import calendar_from_settings, derive, espn_periods_look_real, from_espn


def _gpd(first=date(2026, 10, 20), days=175, base=8, low_start=date(2027, 2, 15)):
    idx = pd.date_range(first, periods=days)
    vals = [base] * days
    for i, d in enumerate(idx):
        if low_start <= d.date() <= date(2027, 2, 21):
            vals[i] = 1
    return pd.Series(vals, index=idx)


def test_placeholder_periods_are_rejected_and_real_ones_accepted():
    placeholder = {str(i): [i] for i in range(1, 24)}
    assert not espn_periods_look_real(placeholder, 167)
    assert not espn_periods_look_real(placeholder)
    assert not espn_periods_look_real({}, 167)
    real = {"1": list(range(1, 7)), "2": list(range(7, 14)), "3": list(range(14, 168))}
    assert espn_periods_look_real(real, 167)


def test_from_espn_maps_scoring_periods_to_dates_and_marks_playoffs():
    cal = from_espn({"1": [1, 2, 3, 4, 5, 6], "2": list(range(7, 14)), "3": list(range(14, 21))}, date(2026, 10, 20),
                    regular_periods=2)
    assert cal.source == "espn"
    w1, w2, w3 = cal.weeks
    assert (w1.start, w1.end, w1.n_days) == (date(2026, 10, 20), date(2026, 10, 25), 6)
    assert w2.start == date(2026, 10, 26) and w3.kind == "playoff" and w2.kind == "regular"


def test_derive_reproduces_the_league_shape_for_2026_27():
    cal = derive(date(2026, 10, 20), 23, regular_periods=20, games_per_day=_gpd(), last_day=date(2027, 4, 4))
    assert cal.source == "derived" and len(cal.weeks) == 23
    assert cal.weeks[0].n_days == 6 and cal.first_day == date(2026, 10, 20)
    assert cal.last_day == date(2027, 4, 4)                       # ESPN's finalScoringPeriod 167
    assert sum(w.n_days for w in cal.weeks) == 167
    long_weeks = [w for w in cal.weeks if w.n_days == 14]
    assert len(long_weeks) == 1 and long_weeks[0].contains(date(2027, 2, 18))   # All-Star break merged
    assert [w.kind for w in cal.weeks].count("playoff") == 3
    for a, b in zip(cal.weeks, cal.weeks[1:]):                     # weeks tile the calendar with no gaps
        assert (b.start - a.end).days == 1


def test_derive_without_game_counts_merges_nothing():
    cal = derive(date(2026, 10, 20), 5, regular_periods=4)
    assert [w.n_days for w in cal.weeks] == [6, 7, 7, 7, 7]


def test_calendar_from_settings_falls_back_when_placeholder_and_uses_espn_when_real():
    cfg = {"league": {"schedule": {"regular_season_matchups": 20, "playoff_rounds_weeks": [1, 1, 1]}}}
    placeholder = {"settings": {"scheduleSettings": {"matchupPeriodCount": 20,
                                                     "matchupPeriods": {str(i): [i] for i in range(1, 24)}}},
                   "status": {"finalScoringPeriod": 167}}
    cal = calendar_from_settings(placeholder, date(2026, 10, 20), cfg=cfg, games_per_day=_gpd())
    assert cal.source == "derived" and cal.last_day == date(2027, 4, 4)
    real = {"settings": {"scheduleSettings": {"matchupPeriodCount": 2,
                                              "matchupPeriods": {"1": list(range(1, 8)), "2": list(range(8, 15)),
                                                                 "3": list(range(15, 22))}}},
            "status": {"finalScoringPeriod": 21}}
    cal = calendar_from_settings(real, date(2026, 10, 20), cfg=cfg)
    assert cal.source == "espn" and cal.weeks[2].kind == "playoff"
    assert calendar_from_settings(None, date(2026, 10, 20), cfg=cfg).source == "derived"


def test_current_and_next_week_edges():
    cal = derive(date(2026, 10, 20), 3, regular_periods=2)
    assert cal.current_week(date(2026, 9, 24)).week == 1          # before the season: the first week
    assert cal.current_week(date(2026, 10, 27)).week == 2
    assert cal.next_week(date(2026, 10, 27)).week == 3
    assert cal.next_week(date(2026, 11, 5)) is None               # last week has no next
    assert cal.current_week(date(2027, 6, 1)) is None             # after the season
    assert list(cal.as_frame()["week"]) == [1, 2, 3]
