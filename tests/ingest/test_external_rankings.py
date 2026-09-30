"""Coverage for the shared external-rankings contract (src.ingest.external_rankings): every input
row survives into the resolved frame (matched or not), ambiguous names are labelled distinctly from
genuinely unmatched ones, and the required-column check is enforced."""
from __future__ import annotations

import pandas as pd
import pytest

from src.ingest.external_rankings import RESOLVED_COLUMNS, resolve_players, unresolved_rows


def _players():
    return pd.DataFrame({
        "player_id": [1, 2, 3],
        "player_name": ["Nikola Jokic", "Luka Doncic", "Duplicate Name"],
        "from_year": [2014, 2018, 2010],
        "to_year": [None, None, None],
    })


def _parsed(rows):
    cols = ["source_id", "source_name_raw", "name_parsed", "team", "positions", "status_tag",
            "ext_rank", "adp", "ecr_vs_adp"]
    return pd.DataFrame(rows, columns=cols)


def test_every_input_row_survives_matched_or_not():
    parsed = _parsed([
        ["1", "Nikola Jokić", "Nikola Jokić", "DEN", "C", None, 1, 1.0, None],
        ["2", "Some Totally Unknown Player", "Some Totally Unknown Player", "FA", "G", None, 999, None, None],
    ])
    result = resolve_players(parsed, _players(), source="test")
    assert len(result.frame) == 2
    assert list(result.frame.columns) == list(RESOLVED_COLUMNS)
    matched_row = result.frame[result.frame["source_id"] == "1"].iloc[0]
    assert matched_row["matched"]
    assert matched_row["player_id"] == 1
    assert matched_row["match_method"] == "exact"  # accent-insensitive normalize_name treats this as exact

    unmatched_row = result.frame[result.frame["source_id"] == "2"].iloc[0]
    assert not unmatched_row["matched"]
    assert pd.isna(unmatched_row["player_id"])
    assert unmatched_row["match_method"] == "unmatched"


def test_unresolved_rows_helper_returns_only_the_unmatched_ones():
    parsed = _parsed([
        ["1", "Nikola Jokić", "Nikola Jokić", "DEN", "C", None, 1, 1.0, None],
        ["2", "Unknown Guy", "Unknown Guy", "FA", "G", None, 999, None, None],
    ])
    result = resolve_players(parsed, _players(), source="test")
    unresolved = unresolved_rows(result.frame)
    assert len(unresolved) == 1
    assert unresolved.iloc[0]["source_id"] == "2"


def test_missing_required_column_raises_clearly():
    bad = pd.DataFrame({"source_id": ["1"], "name_parsed": ["A"]})
    with pytest.raises(ValueError, match="missing columns"):
        resolve_players(bad, _players(), source="test")


def test_report_is_the_id_map_match_report():
    parsed = _parsed([
        ["1", "Nikola Jokić", "Nikola Jokić", "DEN", "C", None, 1, 1.0, None],
    ])
    result = resolve_players(parsed, _players(), source="test")
    assert result.report.n_total == 1
    assert result.report.n_matched == 1
