import numpy as np
import pandas as pd

from src.app.state import display_columns
from src.value.rank_reason import add_rank_reason, reason_for


def _row(**kw):
    base = dict(proj_fppg=33.0, proj_gp=41.8, proj_total_fp=1376.7, vorp=144.7, projection_class="veteran", age=30.0)
    base.update(kw)
    return pd.Series(base)


def test_reason_states_the_arithmetic_and_low_availability():
    r = reason_for(_row())
    assert "33.0 FP/game x 42 games = 1377 FP" in r and "+145 vs a replacement player" in r
    assert "~40 missed games" in r


def test_reason_flags_adp_disagreement_in_both_directions():
    assert "market drafts him 78 spots higher (ADP 68)" in reason_for(_row(adp=68.0, adp_gap=-78.0))
    assert "model ranks him 30 spots above the market" in reason_for(_row(adp=150.0, adp_gap=30.0))
    assert "ADP" not in reason_for(_row(adp=140.0, adp_gap=5.0))


def test_reason_includes_risk_text_truncated_and_ignores_nan():
    r = reason_for(_row(risk_flags="x" * 200, return_flag=np.nan, proj_gp=75.0))
    assert "durable" in r and r.endswith("…") and "nan" not in r.lower()


def test_missing_columns_never_raise_and_every_row_gets_a_string():
    b = pd.DataFrame({"rank": [1, 2], "name": ["a", "b"]})
    out = add_rank_reason(b)
    assert out["rank_reason"].tolist() == ["", ""] and "rank_reason" not in b.columns


def test_rank_reason_sits_right_after_the_position_column():
    b = pd.DataFrame({c: [1] for c in ("rank", "name", "position", "tier", "proj_fppg", "proj_gp", "proj_total_fp",
                                       "vorp", "fppg_p10", "fppg_p50", "fppg_p90", "rank_reason")})
    cols = display_columns(b)
    assert cols[:4] == ["rank", "name", "position", "rank_reason"]
