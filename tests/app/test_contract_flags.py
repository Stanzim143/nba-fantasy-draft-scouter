"""Display-only contract flags (ADR 0019): they add columns, never change a row or a rank, and degrade to notes."""
import numpy as np
import pandas as pd

from src.app import loader
from src.app.state import display_columns
from src.contracts import History
from src.ingest.wiki_contracts import PLAYER_CONTRACTS_COLUMNS
from src.value import contract_flags as cf


def _history():
    players = pd.DataFrame({"player_id": [1, 2, 3], "draft_year": pd.array([2010, 2010, 2022], dtype="Int64"),
                            "draft_round": pd.array([1, 1, 1], dtype="Int64"), "draft_number": pd.array([5, 9, 12], dtype="Int64")})
    return History("2025-26", pd.DataFrame(), pd.DataFrame(), players, pd.DataFrame())


def _contracts():
    row = dict(season="2024-25", page_season="2025-26", start_year=2025, team_id=1, player_id=1, event="sign", contract_type="standard",
               years=1, days=pd.NA, amount_usd=np.nan, signed_date=pd.Timestamp("2025-07-06"), date_source="cell", is_two_way=False,
               is_ten_day=False, is_exhibit10=False, is_minimum=False, is_extension=False, is_rookie=False, non_guaranteed=False,
               has_option=False, multiyear_unspecified=False, section="add", terms_source="cell")
    df = pd.DataFrame([row])[PLAYER_CONTRACTS_COLUMNS]
    df["years"] = pd.array(df["years"], dtype="Int64")
    return df


def test_flags_are_added_without_touching_rows_or_ranks(monkeypatch, tmp_path):
    import src.ingest.wiki_contracts as wc

    monkeypatch.setattr(wc, "read_player_contracts", lambda base=None: _contracts())
    board = pd.DataFrame({"player_id": [3, 1, 2, 99], "rank": [1, 2, 3, 4], "name": list("abcd")})
    ov, notes = cf.compute_contract_flags(board, _history(), "2025-26")
    out = cf.attach_contract_flags(board, ov)
    assert list(out["player_id"]) == [3, 1, 2, 99] and list(out["rank"]) == [1, 2, 3, 4]
    r = out.set_index("player_id")
    assert r.loc[1, "contract_flag"] == "contract year"
    assert r.loc[1, "contract_status"] == "known" and r.loc[1, "contract_years_left"] == 1
    assert r.loc[2, "contract_status"] == "unknown" and r.loc[2, "contract_flag"] == ""     # blank = not known to be
    assert r.loc[99, "contract_status"] == "unknown"
    assert notes and "unvalidated" in notes[0]
    assert (out["contract_validation"] == "unvalidated_display_only").all()       # visible to a CSV consumer
    assert list(out.columns[:3]) == ["player_id", "rank", "name"]                 # existing columns unchanged


def test_missing_table_falls_back_to_rookie_scale_and_a_note(monkeypatch):
    import src.ingest.wiki_contracts as wc

    def boom(base=None):
        raise FileNotFoundError("x")

    monkeypatch.setattr(wc, "read_player_contracts", boom)
    board = pd.DataFrame({"player_id": [1, 3], "rank": [1, 2]})
    ov, notes = cf.compute_contract_flags(board, _history(), "2025-26")
    assert ov.set_index("player_id").loc[3, "contract_flag"] == "rookie final year (nominal)"
    assert any("no player_contracts" in n for n in notes)


def test_loader_wrapper_never_fails_and_display_column_is_listed(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("nope")

    monkeypatch.setattr(cf, "compute_contract_flags", boom)
    b = pd.DataFrame({"player_id": [1], "rank": [1]})
    out = loader._with_contract_flags(b, _history(), "2025-26", None)
    assert "contract_flag" not in out.columns and "contract flags unavailable" in out.attrs["risk_notes"][-1]
    b2 = pd.DataFrame({c: [1] for c in ("rank", "name", "position", "tier", "proj_fppg", "proj_gp", "proj_total_fp", "vorp", "fppg_p10",
                                        "fppg_p50", "fppg_p90", "contract_flag")})
    assert display_columns(b2)[-1] == "contract_flag"
