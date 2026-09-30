"""The advisory return flag in the app's plain functions (ADR 0023): display columns, the filter, the flagged table, the rank caption,
and a loader that never fails because of the flag. No Streamlit."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.app import loader
from src.app.state import BOARD_COLUMNS, RANK_CAPTION, display_columns, filter_board, return_flag_table
from src.value import return_flag as RF


def _board(with_flag=True):
    b = pd.DataFrame({"rank": [1, 2, 3], "player_id": [10, 11, 12], "name": ["A", "B", "Tatum-like"], "position": ["PG", "C", "SF"],
                      "tier": [1, 1, 2], "proj_fppg": [50.0, 45.0, 41.0], "proj_gp": [70.0, 60.0, 50.0],
                      "proj_total_fp": [3500.0, 2700.0, 2050.0], "vorp": [900.0, 700.0, 400.0], "adp": [1.0, 5.0, 9.0],
                      "fppg_p10": 30.0, "fppg_p50": 40.0, "fppg_p90": 50.0})
    if with_flag:
        b["return_flag"] = ["", "", RF.FLAG_LABEL]
        b["return_block_pct"] = [np.nan, np.nan, 75.6]
        b["return_tail"] = ["", "", "16/20"]
        b["return_gp_upside_adv"] = [0.0, 0.0, 5.0]
        b["return_fp_upside_adv"] = [0.0, 0.0, 205.0]
    return b


def test_rank_is_shown_by_total_fp_next_to_fppg_and_games():
    cols = display_columns(_board(False))
    i = cols.index("proj_fppg")
    assert cols[i:i + 3] == ["proj_fppg", "proj_gp", "proj_total_fp"] and cols[i + 3] != "proj_fppg"
    assert cols.index("vorp") > cols.index("proj_total_fp") and cols[0] == "rank"
    assert BOARD_COLUMNS[:7] == ["rank", "name", "position", "tier", "proj_fppg", "proj_gp", "proj_total_fp"]
    assert "TOTAL fantasy points above replacement (VORP)" in RANK_CAPTION and "FPPG x games" in RANK_CAPTION
    assert "A high FPPG player who misses games ranks lower" in RANK_CAPTION


def test_flag_columns_are_shown_only_when_the_board_has_them():
    assert "return_flag" not in display_columns(_board(False))
    cols = display_columns(_board())
    assert "return_flag" in cols and "return_tail" in cols
    assert set(display_columns(_board(False))) < set(cols)


def test_returned_only_filter_keeps_flagged_players_and_is_a_noop_without_the_column():
    assert filter_board(_board(), returned_only=True)["name"].tolist() == ["Tatum-like"]
    assert len(filter_board(_board(), returned_only=False)) == 3
    assert len(filter_board(_board(False), returned_only=True)) == 3
    assert filter_board(_board(), position="SF", returned_only=True)["name"].tolist() == ["Tatum-like"]
    assert filter_board(_board(), position="PG", returned_only=True).empty


def test_flagged_table_hides_drafted_players_and_shows_total_fp_first():
    t = return_flag_table(_board(), drafted_ids=[])
    assert t["name"].tolist() == ["Tatum-like"]
    for c in ("return_block_pct", "return_tail", "proj_gp", "proj_fppg", "proj_total_fp", "return_gp_upside_adv", "return_fp_upside_adv"):
        assert c in t.columns
    assert return_flag_table(_board(), drafted_ids=[12]).empty
    assert return_flag_table(_board(False)).empty


def test_flagging_does_not_reorder_or_change_the_displayed_board():
    plain, flagged = _board(False), _board()
    cols = [c for c in display_columns(plain)]
    assert filter_board(flagged)[cols].to_csv(index=False) == filter_board(plain)[cols].to_csv(index=False)


def test_loader_turns_a_flag_failure_into_a_note_and_keeps_the_board(monkeypatch):
    b = _board(False)
    b.attrs["risk_notes"] = ["earlier note"]

    def boom(*a, **k):
        raise FileNotFoundError("nope")

    monkeypatch.setattr(RF, "compute_return_flag", boom)
    out = loader._with_return_flag(b, None, 82.0)
    assert out is b and "return_flag" not in out.columns
    assert out.attrs["risk_notes"] == ["earlier note", "return flag unavailable (nope)"]


def test_loader_attaches_the_flag_and_its_notes_without_touching_the_board(monkeypatch):
    b = _board(False)
    b.attrs["risk_notes"] = []
    b.attrs["replacement"] = {"rank": 1}
    o = pd.DataFrame({"player_id": [12], "return_flag": [RF.FLAG_LABEL], "return_block_pct": [75.6], "return_tail": ["16/20"],
                      "return_gp_upside_adv": [5.0], "return_fp_upside_adv": [205.0], "return_text": ["returned"]})
    monkeypatch.setattr(RF, "compute_return_flag", lambda *a, **k: (o, ["a note"]))
    out = loader._with_return_flag(b, None, 82.0)
    assert out["return_flag"].tolist() == ["", "", RF.FLAG_LABEL] and out.attrs["risk_notes"] == ["a note"]
    assert out.attrs["replacement"] == {"rank": 1}
    assert out[list(b.columns)].equals(b)


def test_synthetic_boards_carry_no_flag_columns():
    b = loader.load_board("2021-22", "baseline", synthetic=True)
    assert "return_flag" not in b.columns


def test_app_smoke_synthetic_board_shows_the_total_fp_caption_and_no_exception():
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "src" / "app" / "draft_board.py"), default_timeout=120)
    at.run()
    [c for c in at.sidebar.checkbox if c.label.startswith("Use synthetic")][0].check()
    [b for b in at.sidebar.button if b.label.startswith("Load")][0].click()
    at.run()
    assert not at.exception and not at.error
    assert any(RANK_CAPTION in c.value for c in at.caption)
    # no flag on synthetic data, so the filter is not offered
    assert not [c for c in at.checkbox if c.label == "Returned-healthy only"]


def test_flag_columns_sit_right_after_vorp():
    cols = display_columns(_board())
    i = cols.index("vorp")
    assert cols[i + 1:i + 3] == ["return_flag", "return_tail"]
