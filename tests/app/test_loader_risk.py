"""The board loader attaches the risk overlay only in the live window and never fails because of it."""
from datetime import date

import pandas as pd

from src.app import loader
from src.value import risk as vr


def test_season_is_live_window():
    assert vr.season_is_live("2026-27", date(2026, 9, 24)) and vr.season_is_live("2026-27", date(2026, 12, 1))
    assert not vr.season_is_live("2026-27", date(2026, 6, 1)) and not vr.season_is_live("2020-21", date(2026, 9, 24))


def test_outside_the_live_window_the_board_is_untouched(monkeypatch):
    b = pd.DataFrame({"player_id": [1], "rank": [1]})
    monkeypatch.setattr(vr, "season_is_live", lambda season: False)
    assert loader._with_risk(b, b, None, "2019-20", None) is b


def test_an_overlay_failure_becomes_a_note_not_an_error(monkeypatch):
    b = pd.DataFrame({"player_id": [1], "rank": [1]})
    monkeypatch.setattr(vr, "season_is_live", lambda season: True)

    def boom(*a, **k):
        raise FileNotFoundError("nope")

    monkeypatch.setattr(vr, "compute_risk", boom)
    out = loader._with_risk(b, b, None, "2026-27", None)
    assert "risk overlay unavailable" in out.attrs["risk_notes"][0] and "risk_level" not in out.columns


def test_attach_keeps_rows_and_columns_and_fills_blanks():
    b = pd.DataFrame({"player_id": [1, 2], "rank": [1, 2]})
    o = pd.DataFrame({"player_id": [1], "risk_level": ["watch"], "risk_flags": ["x"], "risk_gp_haircut": [0.0], "risk_gp": [50.0]})
    out = vr.attach_risk(b, o)
    assert list(out["player_id"]) == [1, 2] and out["risk_flags"].tolist() == ["x", ""] and out["risk_level"].tolist() == ["watch", ""]
