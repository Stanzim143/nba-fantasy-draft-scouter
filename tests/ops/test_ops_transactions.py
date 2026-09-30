"""The transactions step of the daily refresh (ADR 0017): the worker, the diff hook and the report section."""
import argparse
from pathlib import Path

import pandas as pd

from src.ingest import espn_transactions as et
from src.ops import daily_refresh as dr
from src.ops.daily_report import render_report


def seed(base: Path, reports: Path):
    feed = [{"date": "2026-09-23T07:00Z", "description": "Acquired G Buddy Hield from Atlanta.", "team": {"abbreviation": "CHA"}},
            {"date": "2026-09-23T07:00Z", "description": "Waived G Camp Guy.", "team": {"abbreviation": "DEN"}},
            {"date": "2026-06-23T07:00Z", "description": "Hired Micah Nori as head coach.", "team": {"abbreviation": "POR"}}]
    recs, _ = et.feed_rows_to_records(feed)
    for r in recs:
        r["player_id"] = 1 if r["person"] == "Buddy Hield" else None
    old = et._frame([r for r in recs if r["kind"] == "hired"], pd.Timestamp("2026-09-01", tz="UTC").to_pydatetime())
    new = et._frame([r for r in recs if r["kind"] != "hired"], pd.Timestamp("2026-09-24 08:00", tz="UTC").to_pydatetime())
    et._atomic_parquet(pd.concat([old, new], ignore_index=True), et.ledger_path(base))
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"player_id": [1], "rank": [30], "name": ["Buddy Hield"], "position": ["G"], "vorp": [500.0], "adp": [31.0]}).to_csv(
        reports / "draft_board_baseline.csv", index=False)


def args(base, reports):
    return argparse.Namespace(season="2026-27", data_dir=base, reports_dir=reports, offline=True)


def test_worker_reports_only_rows_new_this_run_and_ranks_them(tmp_path, monkeypatch):
    base, rep = tmp_path / "data", tmp_path / "rep"
    seed(base, rep)
    monkeypatch.setattr(dr, "_refresh_step", lambda *a, **k: {"new_rows": 3, "ledger_rows": 4})
    summary, state = dr.worker_transactions(args(base, rep))
    tx = state["transactions"]
    assert tx["new_rows"] == 3 and tx["ranked"] and tx["ledger_through"] == "2026-09-23"
    assert any("Buddy Hield (G, #30): ATL -> CHA" in x for x in tx["lines"])
    assert not any("Camp Guy" in x for x in tx["lines"]) and any("1 other moves" in x for x in tx["lines"])
    assert tx["staff"] == []                                   # the coach hire was in an earlier run, not this one


def test_worker_with_nothing_new_and_no_board(tmp_path, monkeypatch):
    base, rep = tmp_path / "data", tmp_path / "rep"
    seed(base, rep)
    (rep / "draft_board_baseline.csv").unlink()
    monkeypatch.setattr(dr, "_refresh_step", lambda *a, **k: {"new_rows": 0})
    _, state = dr.worker_transactions(args(base, rep))
    tx = state["transactions"]
    assert tx["new_rows"] == 0 and tx["lines"] == [] and not tx["ranked"]


def test_worker_without_a_board_lists_every_move_unranked(tmp_path, monkeypatch):
    base, rep = tmp_path / "data", tmp_path / "rep"
    seed(base, rep)
    (rep / "draft_board_baseline.csv").unlink()
    monkeypatch.setattr(dr, "_refresh_step", lambda *a, **k: {"new_rows": 3})
    _, state = dr.worker_transactions(args(base, rep))
    tx = state["transactions"]
    assert not tx["ranked"] and tx["new_rows"] == 3
    assert any("Buddy Hield (unranked)" in x for x in tx["lines"]) and any("Camp Guy" in x for x in tx["lines"])


def test_step_is_registered_after_the_board_with_a_timeout_and_a_worker():
    assert dr.STEP_ORDER.index("transactions") > dr.STEP_ORDER.index("board") and "transactions" in dr.DEFAULT_TIMEOUTS and "transactions" in dr.WORKERS


def test_diff_carries_the_state_and_the_report_renders_it():
    tx = {"new_rows": 2, "ranked": True, "ledger_through": "2026-09-23", "staff": ["POR hired Micah Nori (head coach)"],
          "lines": ["Sep 23  TRADE     Buddy Hield (G, #30): ATL -> CHA", "            CHA best teammates: X (#5)", "(1 other moves involve players outside the board's top 180 or unranked)"]}
    diffs = dr.build_diffs([], {"transactions": tx}, None)
    assert diffs["transactions"] == tx
    run = {"date": "2026-09-25", "outcome": "ok", "run_id": "r1", "season": "2026-27", "steps": [], "diffs": diffs}
    text = render_report(run)
    assert "League transactions (2 new since the previous run; feed through 2026-09-23)" in text
    assert "- Sep 23  TRADE     Buddy Hield (G, #30): ATL -> CHA" in text and "  CHA best teammates: X (#5)" in text
    assert "- staff: POR hired Micah Nori (head coach)" in text
    empty = render_report({**run, "diffs": {"transactions": {"new_rows": 0, "lines": [], "staff": [], "ranked": True}}})
    assert "no new transactions" in empty
