"""The league's matchup weeks: which calendar days belong to which H2H matchup period.

Two sources, in order of trust:

1. **ESPN** ``settings.scheduleSettings.matchupPeriods`` from the league sync (``{matchupPeriodId:
   [scoringPeriodIds]}``; a scoring period is one day, 1 = opening night). This is authoritative and is
   used whenever it looks real.
2. **Derived** (documented approximation) when ESPN has not published real periods yet. On 2026-09-24
   ESPN still returned a placeholder for our league: 23 periods of exactly one scoring period each
   (``{1: [1], 2: [2], ...}``) while ``finalScoringPeriod`` was 167. :func:`espn_periods_look_real`
   rejects that. The derivation is: Monday-Sunday calendar weeks starting at opening night (so week 1 is
   a partial week), the one interior week with clearly fewer games (the All-Star break) merged into the
   week before it, then truncated to the league's period count (regular-season matchups + playoff rounds).
   For the 2026-27 season this reproduces the league's own ``finalScoringPeriod`` (167 = 2027-04-04)
   exactly, which is the one independent check available before the season starts.

Playoff periods are the last ``sum(playoff_rounds_weeks)`` periods; only the top ``playoff_teams``
play in them, so a week's ``kind`` is ``"playoff"`` for the tools to label (they still project it).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Mapping

import pandas as pd

MIN_REAL_PERIOD_DAYS = 100  # a real league covers most of a season; the placeholder covers 23 days


@dataclass(frozen=True)
class Week:
    week: int            # matchup period id, 1-based
    start: date
    end: date            # inclusive
    kind: str            # "regular" | "playoff"

    @property
    def n_days(self) -> int:
        return (self.end - self.start).days + 1

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end


def as_date(x) -> date:
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, date):
        return x
    return pd.Timestamp(x).date()


@dataclass(frozen=True)
class MatchupCalendar:
    weeks: tuple[Week, ...]
    source: str          # "espn" | "derived"

    def week_for(self, day) -> Week | None:
        d = as_date(day)
        for w in self.weeks:
            if w.contains(d):
                return w
        return None

    def current_week(self, as_of) -> Week | None:
        """The week containing ``as_of``; the first week before the season starts; ``None`` after it ends."""
        d = as_date(as_of)
        if not self.weeks:
            return None
        if d < self.weeks[0].start:
            return self.weeks[0]
        return self.week_for(d)

    def next_week(self, as_of) -> Week | None:
        cur = self.current_week(as_of)
        if cur is None:
            return None
        nxt = [w for w in self.weeks if w.week == cur.week + 1]
        return nxt[0] if nxt else None

    @property
    def first_day(self) -> date:
        return self.weeks[0].start

    @property
    def last_day(self) -> date:
        return self.weeks[-1].end

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"week": w.week, "start": w.start, "end": w.end, "n_days": w.n_days, "kind": w.kind}
                             for w in self.weeks])


# --------------------------------------------------------------------------- ESPN periods

def espn_periods_look_real(matchup_periods: Mapping, final_scoring_period: int | None = None) -> bool:
    """False for ESPN's not-yet-published placeholder (see the module docstring)."""
    ids = [int(s) for v in (matchup_periods or {}).values() for s in v]
    if not ids:
        return False
    if final_scoring_period:
        return max(ids) >= final_scoring_period - 1
    return len(ids) >= MIN_REAL_PERIOD_DAYS


def from_espn(matchup_periods: Mapping, day_one, *, regular_periods: int | None = None) -> MatchupCalendar:
    """Calendar from ESPN's ``{matchupPeriodId: [scoringPeriodIds]}``; scoring period 1 falls on ``day_one``."""
    d1 = as_date(day_one)
    weeks: list[Week] = []
    for mp, sps in sorted(((int(k), v) for k, v in matchup_periods.items()), key=lambda kv: kv[0]):
        sps = sorted(int(s) for s in sps)
        if not sps:
            continue
        start = d1 + timedelta(days=sps[0] - 1)
        end = d1 + timedelta(days=sps[-1] - 1)
        kind = "playoff" if (regular_periods is not None and mp > regular_periods) else "regular"
        weeks.append(Week(mp, start, end, kind))
    return MatchupCalendar(tuple(weeks), "espn")


# --------------------------------------------------------------------------- derived fallback

def derive(first_day, n_periods: int, *, regular_periods: int | None = None,
           games_per_day: pd.Series | None = None, last_day=None) -> MatchupCalendar:
    """Monday-Sunday weeks from ``first_day`` with the All-Star week merged, cut to ``n_periods``.

    ``games_per_day`` (index = date, value = games league-wide that day) lets the derivation find the
    All-Star week (the interior calendar week with clearly the fewest games); without it no week is
    merged. ``last_day`` bounds the calendar (default: enough weeks to hold ``n_periods + 3``).
    """
    d0 = as_date(first_day)
    end_cap = as_date(last_day) if last_day is not None else d0 + timedelta(days=7 * (n_periods + 3))
    cal: list[tuple[date, date]] = []
    start = d0
    while start <= end_cap:
        end = start + timedelta(days=(6 - start.weekday()))  # Sunday
        cal.append((start, min(end, end_cap)))
        start = end + timedelta(days=1)

    if games_per_day is not None and len(games_per_day) and len(cal) > n_periods:
        gpd = games_per_day.copy()
        gpd.index = pd.to_datetime(gpd.index).date
        totals = [int(sum(v for d, v in gpd.items() if a <= d <= b)) for a, b in cal]
        lo, hi = int(0.45 * len(cal)), int(0.85 * len(cal))  # the break is mid-late season; early Cup weeks are incomplete
        interior = [i for i in range(max(lo, 1), min(hi, len(cal) - 1)) if (cal[i][1] - cal[i][0]).days == 6]
        if interior:
            med = float(pd.Series([totals[i] for i in interior]).median())
            low = min(interior, key=lambda i: totals[i])
            if totals[low] < 0.75 * med:
                cal[low - 1] = (cal[low - 1][0], cal[low][1])
                del cal[low]
    cal = cal[:n_periods]
    weeks = []
    for i, (a, b) in enumerate(cal, start=1):
        kind = "playoff" if (regular_periods is not None and i > regular_periods) else "regular"
        weeks.append(Week(i, a, b, kind))
    return MatchupCalendar(tuple(weeks), "derived")


def calendar_from_settings(settings_payload: dict | None, day_one, *, cfg: dict | None = None,
                           games_per_day: pd.Series | None = None) -> MatchupCalendar:
    """Best available calendar: ESPN's periods when real, else the derived fallback.

    ``settings_payload`` is the raw league ``mSettings`` payload (or ``None`` offline); ``cfg`` is
    ``config/league.yaml`` (for the period counts when ESPN's are a placeholder).
    """
    d1 = as_date(day_one)
    sched = (((settings_payload or {}).get("settings") or {}).get("scheduleSettings")) or {}
    status = (settings_payload or {}).get("status") or {}
    reg = sched.get("matchupPeriodCount")
    periods = sched.get("matchupPeriods") or {}
    final_sp = status.get("finalScoringPeriod")
    if reg is None and cfg:
        reg = cfg["league"]["schedule"]["regular_season_matchups"]
    if periods and espn_periods_look_real(periods, final_sp):
        return from_espn(periods, d1, regular_periods=reg)
    playoff_weeks = sum(cfg["league"]["schedule"].get("playoff_rounds_weeks", [])) if cfg else 3
    reg = reg or 20
    last = d1 + timedelta(days=int(final_sp) - 1) if final_sp else None
    return derive(d1, reg + playoff_weeks, regular_periods=reg, games_per_day=games_per_day, last_day=last)
