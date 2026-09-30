"""The advisory short-absence flag in the app's plain functions (ADR 0024): display column, filter, flagged table, loader never fails
because of the flag. No Streamlit."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.app import loader
from src.app.state import display_columns, filter_board, load_flag_table
from src.value import load_flag as LF


def _board(with_flag=True):
    b = pd.DataFrame({"rank": [1, 2, 3], "player_id": [10, 11, 12], "name": ["A", "B", "Frequent"], "position": ["PG", "C", "SF"],
                      "tier": [1, 1, 2], "proj_fppg": [50.0, 45.0, 41.0], "proj_gp": [70.0, 60.0, 50.0],
                      "proj_total_fp": [3500.0, 2700.0, 2050.0], "vorp": [900.0, 700.0, 400.0], "adp": [1.0, 5.0, 9.0],
                      "fppg_p10": 30.0, "fppg_p50": 40.0, "fppg_p90": 50.0})
    if with_flag:
        b["lm_flag"] = ["", "", LF.FLAG_LABEL]
        b["lm_iso_n"] = [2.0, np.nan, 7.0]
        b["lm_rest_n"] = [0.0, np.nan, 2.0]
        b["lm_gp_risk_adv"] = [0.0, 0.0, -2.0]
        b["lm_fp_risk_adv"] = [0.0, 0.0, -82.0]
    return b


def test_flag_column_is_shown_only_when_the_board_has_it():
    assert "lm_flag" not in display_columns(_board(False))
    cols = display_columns(_board())
    assert "lm_flag" in cols and set(display_columns(_board(False))) < set(cols)
    assert cols.index("lm_flag") > cols.index("vorp"), "next to the other advisory flags, after the projection columns"


def test_short_absences_filter_keeps_flagged_players_and_is_a_noop_without_the_column():
    assert filter_board(_board(), short_absences_only=True)["name"].tolist() == ["Frequent"]
    assert len(filter_board(_board(), short_absences_only=False)) == 3
    assert len(filter_board(_board(False), short_absences_only=True)) == 3
    assert filter_board(_board(), position="SF", short_absences_only=True)["name"].tolist() == ["Frequent"]
    assert filter_board(_board(), position="PG", short_absences_only=True).empty


def test_flagged_table_lists_undrafted_players_in_board_order_with_advisory_columns():
    t = load_flag_table(_board())
    assert t["name"].tolist() == ["Frequent"] and t.iloc[0]["lm_gp_risk_adv"] == -2.0 and t.iloc[0]["lm_fp_risk_adv"] == -82.0
    assert {"lm_iso_n", "lm_rest_n", "proj_gp", "proj_total_fp", "vorp"} <= set(t.columns)
    assert load_flag_table(_board(), drafted_ids=[12]).empty
    assert load_flag_table(_board(False)).empty


def test_loader_wiring_is_never_fatal_and_leaves_the_board_untouched(monkeypatch):
    b = _board(False)
    b.attrs["risk_notes"] = []

    def boom(*a, **k):
        raise ValueError("bad history")

    monkeypatch.setattr(LF, "compute_load_flag", boom)
    out = loader._with_load_flag(b, history=None, season_games=82.0)
    assert out is b and "lm_flag" not in out.columns
    assert out.attrs["risk_notes"] == ["short-absence flag unavailable (bad history)"]
