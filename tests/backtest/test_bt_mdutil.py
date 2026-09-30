"""Tests for the tiny markdown table renderer (src/backtest/mdutil.py)."""
import numpy as np
import pandas as pd

from src.backtest.mdutil import df_to_markdown


def test_basic_render_with_index():
    df = pd.DataFrame({"a": [1, 2], "b": [3.5, 4.25]}, index=["x", "y"])
    md = df_to_markdown(df)
    lines = md.splitlines()
    assert lines[0] == "|  | a | b |"
    assert lines[2] == "| x | 1 | 3.500 |"
    assert lines[3] == "| y | 2 | 4.250 |"


def test_without_index():
    df = pd.DataFrame({"a": [1, 2]})
    md = df_to_markdown(df, index=False)
    assert md.splitlines()[0] == "| a |"


def test_nan_renders_as_na():
    df = pd.DataFrame({"a": [1.0, np.nan]})
    md = df_to_markdown(df, index=False)
    assert "n/a" in md.splitlines()[3]


def test_pipe_and_newline_in_value_are_escaped():
    df = pd.DataFrame({"a": ["x|y", "line1\nline2"]})
    md = df_to_markdown(df, index=False)
    lines = md.splitlines()
    assert "x\\|y" in lines[2]
    assert "\n" not in lines[3]
    assert "line1 line2" in lines[3]


def test_custom_format_string_and_callable():
    df = pd.DataFrame({"pct": [0.5], "note": ["x"]}, index=["r"])
    md = df_to_markdown(df, formats={"pct": "{:.1%}", "note": lambda v: v.upper()})
    line = md.splitlines()[2]
    assert "50.0%" in line and "X" in line


def test_single_unnamed_value_column_does_not_collide_with_reset_index():
    """Regression: a caller's own column literally named "" collides with the "" that
    df_to_markdown gives an unnamed index after reset_index(). Before the fix this made every
    cell in that column render as a pandas Series repr ("Name: 0, dtype: ...") instead of the
    value -- exactly what write_report's "Run" table did in production before it was renamed to
    "value". Covers both spellings so this can't silently regress either way."""
    for col in ("", "value"):
        df = pd.DataFrame({col: ["alpha", "beta", "gamma"]}, index=["row1", "row2", "row3"])
        md = df_to_markdown(df)
        assert "Name:" not in md
        assert "dtype:" not in md
        lines = md.splitlines()
        assert lines[2] == "| row1 | alpha |"
        assert lines[3] == "| row2 | beta |"
        assert lines[4] == "| row3 | gamma |"


def test_survives_a_genuinely_duplicate_named_column():
    """Two columns can legitimately share a name after some upstream operation; the renderer
    must not silently corrupt one of them (row[c] would return a Series; row.iloc[i] can't)."""
    df = pd.DataFrame([[1, 2], [3, 4]])
    df.columns = ["x", "x"]
    md = df_to_markdown(df, index=False)
    lines = md.splitlines()
    assert lines[0] == "| x | x |"
    assert lines[2] == "| 1 | 2 |"
    assert lines[3] == "| 3 | 4 |"
    assert "Name:" not in md
