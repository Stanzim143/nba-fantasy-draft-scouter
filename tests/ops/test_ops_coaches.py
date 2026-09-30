"""The coaches step of the daily refresh (ADR 0020): the worker, the coach-change diff and the report section."""
import argparse

from src.ops import daily_refresh as dr
from src.ops.daily_report import render_report


def test_step_is_registered_with_a_timeout_and_a_worker():
    assert "coaches" in dr.STEP_ORDER and "coaches" in dr.DEFAULT_TIMEOUTS and "coaches" in dr.WORKERS


def test_worker_returns_the_team_to_coach_map_as_state(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "_refresh_step", lambda *a, **k: {"teams": 2, "coaches": {"1": "A One", "2": "B Two"}})
    a = argparse.Namespace(season="2026-27", data_dir=tmp_path, reports_dir=tmp_path, offline=True)
    summary, state = dr.worker_coaches(a)
    assert summary == {"teams": 2} and state == {"coach_map": {"1": "A One", "2": "B Two"}}


def test_a_changed_coach_shows_in_the_diff_and_the_report():
    prev = {"coach_map": {"1610612741": "A One", "1610612742": "B Two"}}
    diffs = dr.build_diffs([prev], {"coach_map": {"1610612741": "A One", "1610612742": "C Three"}}, None)
    assert diffs["coaches"] == [{"team_id": "1610612742", "team": "DAL", "from": "B Two", "to": "C Three"}]
    text = render_report({"date": "2026-09-25", "outcome": "ok", "run_id": "r", "season": "2026-27", "steps": [], "diffs": diffs})
    assert "Head coach changes" in text and "DAL: B Two -> C Three" in text
    assert "none" in render_report({"date": "d", "outcome": "ok", "run_id": "r", "season": "s", "steps": [], "diffs": {"coaches": []}})


def test_no_previous_run_means_no_coach_diff():
    assert "coaches" not in dr.build_diffs([], {"coach_map": {"1": "A"}}, None)
