"""Opt-in: run the backtest harness on real ingested data when it exists (skipped otherwise).

Regression coverage for two real bugs, both the same class: real data's pandas nullable
Int64/boolean dtypes for players.draft_year/from_year/to_year produced actual pd.NA entries in
boolean arrays built from them, and code that expected plain True/False crashed -- but every other
test in this suite used synthetic fixtures whose draft_year is plain non-nullable int64, so this
class of bug hid behind an all-synthetic test suite twice in a row:

1. build_history/sanitize_players: `~drop.to_numpy()` bitwise-negated an object array of Python
   bools (KeyError). Fixed 2026-09-22.
2. leakage._scramble_players (the --leak-check / assert_projector_ignores_future path): a boolean
   array built from `draft.isna() & (frm >= cutoff)` could itself be pd.NA for a player with both
   draft_year and from_year unknown, and `if future.sum() > 1` raised `TypeError: boolean value of
   NA is ambiguous`. Found by a third round of adversarial verification. Fixed 2026-09-22.

This file exists so both are covered directly against real data going forward.
"""
import pytest

from src.backtest.benchmarks import NaiveLastSeason
from src.backtest.harness import walk_forward
from src.backtest.leakage import assert_projector_ignores_future, build_history, sanitize_players
from src.contracts import HISTORY_TABLES, season_start, season_str, table_path
from src.models.registry import get_projector
from src.store import load_tables

_MISSING = [t for t in HISTORY_TABLES if not table_path(t).exists()]
pytestmark = pytest.mark.skipif(bool(_MISSING), reason=f"real data not ingested yet (missing {_MISSING})")


@pytest.fixture(scope="module")
def real():
    return load_tables(HISTORY_TABLES)


def _second_to_last_season(tables):
    starts = sorted({season_start(s) for s in tables["game_logs"]["season"].unique()})
    return season_str(starts[-2])


def test_sanitize_players_does_not_crash_on_real_nullable_dtypes(real):
    """The exact call that raised KeyError on real data before the fix."""
    season = _second_to_last_season(real)
    assert str(real["players"]["draft_year"].dtype) in ("Int64", "int64")
    out = sanitize_players(real["players"], season)
    assert len(out) > 0
    cutoff = season_start(season)
    assert not (out["from_year"].dropna() >= cutoff).any()
    assert not (out["to_year"].dropna() >= cutoff).any()


def test_build_history_works_on_real_data(real):
    season = _second_to_last_season(real)
    h = build_history(real, season)
    h.assert_no_future()
    assert len(h.players) > 0
    assert len(h.game_logs) > 0


def test_walk_forward_runs_on_real_data(real):
    """The actual CLI's code path (python -m src.backtest), on real data, for one season."""
    season = _second_to_last_season(real)
    result = walk_forward(real, NaiveLastSeason(), [season])
    assert len(result.season_frame(season)) > 100  # a real season has hundreds of scored players
    assert season in result.season_metrics.index


def test_walk_forward_with_registry_baseline_runs_on_real_data(real):
    season = _second_to_last_season(real)
    baseline = get_projector("baseline")
    result = walk_forward(real, baseline, [season])
    assert len(result.season_frame(season)) > 100


def test_leak_check_does_not_crash_on_real_nullable_dtypes(real):
    """The exact call behind `python -m src.backtest --leak-check` on real data.

    Also requires the fixture actually contain a player with unknown draft_year (the specific
    condition that produced a real pd.NA rather than a plain bool in _scramble_players).
    """
    assert real["players"]["draft_year"].isna().any(), "fixture must contain an undrafted player"
    season = _second_to_last_season(real)
    assert_projector_ignores_future(NaiveLastSeason(), real, season)  # must not raise
