"""The Transactions tab: its data prep, and a headless run of the draft-board app with a ledger on disk."""
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.app import txn_view as V
from src.ingest import espn_transactions as et

TODAY = date(2026, 9, 25)


def ledger(base=None):
    feed = [{"date": "2026-09-23T07:00Z", "description": "Acquired G Buddy Hield from Atlanta.", "team": {"abbreviation": "CHA"}},
            {"date": "2026-09-22T07:00Z", "description": "Signed G Bench Guy to a two-way contract.", "team": {"abbreviation": "MIA"}},
            {"date": "2026-08-01T07:00Z", "description": "Waived G Old News.", "team": {"abbreviation": "MIA"}},
            {"date": "2026-06-23T07:00Z", "description": "Hired Micah Nori as head coach.", "team": {"abbreviation": "POR"}}]
    recs, _ = et.feed_rows_to_records(feed)
    for r in recs:
        r["player_id"] = {"Buddy Hield": 1, "Bench Guy": 2, "Old News": 3}.get(r["person"])
    df = et._frame(recs, pd.Timestamp("2026-09-24 08:00", tz="UTC").to_pydatetime())
    if base is not None:
        et._atomic_parquet(df, et.ledger_path(base))
    return df


BOARD = pd.DataFrame({"player_id": [1, 2, 3], "rank": [20, 500, 30], "name": ["Buddy Hield", "Bench Guy", "Old News"],
                      "position": ["G", "G", "G"], "vorp": [1.0, 0.0, 1.0], "adp": [20.0, float("nan"), 30.0]})


def test_moves_window_relevance_and_sorting():
    mv = V.moves(ledger(), BOARD, None, days=14, today=TODAY)
    assert list(mv["person"]) == ["Buddy Hield"]                       # Bench Guy is outside the top 180, Old News outside the window
    every = V.moves(ledger(), BOARD, None, days=14, relevant_only=False, today=TODAY)
    assert list(every["person"]) == ["Buddy Hield", "Bench Guy"]
    wide = V.moves(ledger(), BOARD, None, days=90, teams=["mia"], relevant_only=False, today=TODAY)
    assert set(wide["person"]) == {"Bench Guy", "Old News"}
    assert list(V.moves(ledger(), BOARD, None, days=90, player="bench", today=TODAY)["person"]) == ["Bench Guy"]   # a search ignores relevance


def test_without_a_board_everything_is_listed():
    mv = V.moves(ledger(), None, None, days=14, today=TODAY)
    assert len(mv) == 2 and mv["rank"].isna().all()


def test_staff_view_and_freshness():
    sf = V.staff(ledger(), days=120, today=TODAY)
    assert list(sf["person"]) == ["Micah Nori"] and sf.iloc[0]["role"] == "head_coach"
    assert V.staff(ledger(), days=120, teams=["ORL"], today=TODAY).empty
    assert "Feed through 2026-09-23" in V.freshness(ledger(), TODAY) and "days ago" not in V.freshness(ledger(), TODAY)
    assert "days ago" in V.freshness(ledger(), date(2026, 10, 5))


def test_missing_ledger_is_a_message_not_a_crash(tmp_path):
    with pytest.raises(V.LedgerUnavailable) as exc:
        V.load_ledger(tmp_path)
    assert "espn_transactions" in str(exc.value)
    assert V.load_roster(tmp_path) is None


def test_app_shows_the_transactions_tab_headlessly(tmp_path, monkeypatch):
    at_mod = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path))
    ledger(tmp_path)
    page = Path(__file__).resolve().parents[2] / "src" / "app" / "draft_board.py"
    at = at_mod.AppTest.from_file(str(page), default_timeout=180)
    at.run()
    at.sidebar.checkbox[0].check()                                   # synthetic demo data: nothing real needed
    at.sidebar.button[0].click()
    at.run()
    assert not at.exception, at.exception
    assert len(at.tabs) >= 8 and at.tabs[7].label == "Transactions"
    text = " ".join(x.value for x in at.tabs[7].markdown) + " ".join(x.value for x in at.tabs[7].caption)
    assert "ADR 0017" in text
