"""The Next pick tab (ADR 0040) renders headless: asks for the slot when unknown, then shows the availability
table; works on a board with no ADP (synthetic demo) and never changes the board."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PAGE = Path(__file__).resolve().parents[2] / "src" / "app" / "draft_board.py"


def _board(n=40) -> pd.DataFrame:
    return pd.DataFrame({
        "rank": np.arange(1, n + 1), "player_id": np.arange(1, n + 1), "name": [f"P{i}" for i in range(n)],
        "position": "PG", "tier": 1, "proj_fppg": np.linspace(50, 20, n), "proj_gp": 70.0,
        "proj_total_fp": np.linspace(3500, 1400, n), "vorp": np.linspace(900, -10, n),
        "fppg_p10": 30.0, "fppg_p50": 40.0, "fppg_p90": 50.0,
    })


def test_next_pick_tab_asks_for_slot_then_shows_probabilities(monkeypatch, tmp_path):
    at_mod = pytest.importorskip("streamlit.testing.v1")
    from src.app import live_sync, loader

    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(live_sync, "load_dotenv_file", lambda *_a, **_k: False)
    for var in ("ESPN_LEAGUE_ID", "ESPN_TEAM_ID"):
        monkeypatch.delenv(var, raising=False)
    board = _board()
    before = board.copy(deep=True)
    monkeypatch.setattr(loader, "load_board", lambda *_a, **_k: board)

    at = at_mod.AppTest.from_file(str(PAGE), default_timeout=120)
    at.run()
    [b for b in at.sidebar.button if b.label.startswith("Load")][0].click()
    at.run()
    assert not at.exception, at.exception
    tab = next(t for t in at.tabs if t.label == "Next pick")
    assert "unknown" in " ".join(i.value for i in tab.info)
    assert not tab.dataframe

    tab.number_input(key="ns_slot").set_value(5)
    at.run()
    assert not at.exception, at.exception
    tab = next(t for t in at.tabs if t.label == "Next pick")
    shown = tab.dataframe[0].value
    assert {"p_avail_next", "p_avail_after", "vorp_at_risk"} <= set(shown.columns)
    assert shown["p_avail_next"].between(0, 1).all() and len(shown) == 40
    assert "board rank" in " ".join(c.value for c in tab.caption)  # no ADP: the proxy is announced
    pd.testing.assert_frame_equal(board, before)
