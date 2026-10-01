"""The nightly job's injury-report step (ADR 0033): it is wired into the step list and calls the incremental ingest."""
import argparse
from datetime import date
from pathlib import Path

import pytest

from src.ingest import nba_injury_reports as ir
from src.ops import nightly as nt


def _args(tmp_path, offline=False):
    return argparse.Namespace(data_dir=Path(tmp_path), season="2026-27", through="2026-10-21", offline=offline)


def test_step_is_registered_between_games_and_the_rest():
    assert nt.STEP_ORDER.index("games") < nt.STEP_ORDER.index("injuries") < nt.STEP_ORDER.index("roster")
    assert "injuries" in nt.WORKERS and nt.DEFAULT_TIMEOUTS["injuries"] >= 120


def test_worker_calls_the_ingest_for_the_live_season_with_a_request_budget(tmp_path, monkeypatch):
    seen = {}

    def fake_ingest(base, **kw):
        seen.update(kw, base=base)
        return ir.IngestResult(n_dates=3, n_fetched=2, n_absent=0, n_pending=1, requests=4, n_rows=120,
                               unmatched_names=["A (BOS)"])

    monkeypatch.setattr(ir, "ingest", fake_ingest)
    summary, state = nt.worker_injuries(_args(tmp_path))
    assert seen["seasons"] == ["2026-27"] and seen["until"] == date(2026, 10, 21) and seen["offline"] is False
    assert seen["max_requests"] <= 100 and seen["base"] == Path(tmp_path)
    assert summary["fetched"] == 2 and summary["pending"] == 1 and summary["unmatched_players"] == 1
    assert state == {"injury_reports": summary}


def test_worker_fails_loudly_on_parse_problems(tmp_path, monkeypatch):
    monkeypatch.setattr(ir, "ingest", lambda base, **kw: ir.IngestResult(parse_failures=["2026-10-20 05PM: no rows parsed"]))
    with pytest.raises(nt.StepError, match="no rows parsed"):
        nt.worker_injuries(_args(tmp_path))
