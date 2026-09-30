"""Coverage for the external-rankings comparison join (src.value.rankings_compare): the rank_delta
sign convention, honest handling of unmatched/one-source-only players, and the biggest-disagreement
sort helper."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ingest.external_rankings import RESOLVED_COLUMNS
from src.value.rankings_compare import (
    DISAGREEMENT_THRESHOLD,
    biggest_disagreements,
    build_comparison,
    flag_disagreements,
    unmatched_report,
)


def _resolved_row(*, source, source_id, name, player_id=None, matched=True, method="exact",
                  team="DEN", positions="C", status_tag=None, ext_rank=1, adp=None, ecr_vs_adp=None,
                  confidence=1.0):
    return {
        "source": source, "source_id": source_id, "source_name_raw": name, "name_parsed": name,
        "team": team, "positions": positions, "status_tag": status_tag, "ext_rank": ext_rank,
        "adp": adp, "ecr_vs_adp": ecr_vs_adp, "player_id": player_id, "match_method": method,
        "confidence": confidence, "matched": matched,
    }


def _resolved_frame(rows):
    df = pd.DataFrame(rows, columns=RESOLVED_COLUMNS)
    df["player_id"] = df["player_id"].astype("Int64")
    return df


def _board():
    return pd.DataFrame({
        "player_id": [1, 2, 3],
        "name": ["Player One", "Player Two", "Player Three"],
        "rank": [1, 2, 3],
    })


def test_rank_delta_sign_convention_positive_means_source_ranks_earlier():
    board = _board()
    # our_rank=2 for player 2; yahoo ranks him 1st (earlier) -> rank_delta_yahoo should be +1.
    yahoo = _resolved_frame([
        _resolved_row(source="yahoo", source_id="1", name="Player Two", player_id=2, ext_rank=1),
    ])
    cmp = build_comparison(board, {"yahoo": yahoo})
    row = cmp[cmp["player_id"] == 2].iloc[0]
    assert row["yahoo_rank"] == 1
    assert row["rank_delta_yahoo"] == 1  # our_rank(2) - yahoo_rank(1) = +1: yahoo likes him earlier


def test_rank_delta_negative_when_we_rank_earlier_than_source():
    board = _board()
    # our_rank=1 for player 1; yahoo ranks him 5th (later) -> rank_delta_yahoo should be -4.
    yahoo = _resolved_frame([
        _resolved_row(source="yahoo", source_id="1", name="Player One", player_id=1, ext_rank=5),
    ])
    cmp = build_comparison(board, {"yahoo": yahoo})
    row = cmp[cmp["player_id"] == 1].iloc[0]
    assert row["rank_delta_yahoo"] == 1 - 5 == -4


def test_player_absent_from_a_source_gets_null_rank_and_false_flag_not_zero():
    board = _board()
    yahoo = _resolved_frame([
        _resolved_row(source="yahoo", source_id="1", name="Player One", player_id=1, ext_rank=1),
    ])
    cmp = build_comparison(board, {"yahoo": yahoo})
    row2 = cmp[cmp["player_id"] == 2].iloc[0]
    assert row2["on_yahoo"] == False  # noqa: E712
    assert pd.isna(row2["yahoo_rank"])
    assert pd.isna(row2["rank_delta_yahoo"])
    # no fantasypros frame supplied at all
    assert row2["on_fantasypros"] == False  # noqa: E712
    assert pd.isna(row2["fantasypros_rank"])


def test_player_present_in_external_source_but_not_our_board_is_kept_not_dropped():
    board = _board()
    fp = _resolved_frame([
        _resolved_row(source="fantasypros", source_id="9", name="Outside Player", player_id=99, ext_rank=9),
    ])
    cmp = build_comparison(board, {"fantasypros": fp})
    row = cmp[cmp["player_id"] == 99]
    assert len(row) == 1
    row = row.iloc[0]
    assert row["on_our_board"] == False  # noqa: E712
    assert pd.isna(row["our_rank"])
    assert row["on_fantasypros"] == True  # noqa: E712
    assert row["name"] == "Outside Player"  # name fallback from the source when we have no board name


def test_unmatched_rows_are_reported_not_silently_dropped_from_the_join():
    board = _board()
    yahoo = _resolved_frame([
        _resolved_row(source="yahoo", source_id="1", name="Player One", player_id=1, ext_rank=1),
        _resolved_row(source="yahoo", source_id="2", name="Some Unmatchable Guy", player_id=None,
                      matched=False, method="unmatched", ext_rank=50, confidence=np.nan),
    ])
    cmp = build_comparison(board, {"yahoo": yahoo})
    # the unmatched row never had a player_id, so it cannot appear in the player_id-keyed comparison
    assert "Some Unmatchable Guy" not in cmp["name"].to_numpy()
    # but it must show up in the dedicated unmatched report, not vanish
    report = unmatched_report({"yahoo": yahoo})
    assert len(report) == 1
    assert report.iloc[0]["source_name_raw"] == "Some Unmatchable Guy"
    assert report.iloc[0]["match_method"] == "unmatched"


def test_ambiguous_rows_also_appear_in_unmatched_report():
    yahoo = _resolved_frame([
        _resolved_row(source="yahoo", source_id="3", name="Ambiguous Name", player_id=None,
                      matched=False, method="ambiguous", ext_rank=30, confidence=np.nan),
    ])
    report = unmatched_report({"yahoo": yahoo})
    assert len(report) == 1
    assert report.iloc[0]["match_method"] == "ambiguous"


def test_unmatched_report_empty_when_everything_resolved():
    yahoo = _resolved_frame([
        _resolved_row(source="yahoo", source_id="1", name="Player One", player_id=1, ext_rank=1),
    ])
    report = unmatched_report({"yahoo": yahoo})
    assert report.empty
    assert list(report.columns)  # still has the right shape even when empty


def test_biggest_disagreements_sorts_by_absolute_delta_both_directions():
    board = pd.DataFrame({
        "player_id": [1, 2, 3],
        "name": ["A", "B", "C"],
        "rank": [1, 2, 3],
    })
    yahoo = _resolved_frame([
        _resolved_row(source="yahoo", source_id="1", name="A", player_id=1, ext_rank=1),   # delta 0
        _resolved_row(source="yahoo", source_id="2", name="B", player_id=2, ext_rank=20),  # delta -18
        _resolved_row(source="yahoo", source_id="3", name="C", player_id=3, ext_rank=1),   # delta +2
    ])
    cmp = build_comparison(board, {"yahoo": yahoo})
    top = biggest_disagreements(cmp, n=2, source="yahoo")
    assert list(top["player_id"]) == [2, 3]  # |-18| > |+2| > |0|


def test_biggest_disagreements_default_uses_max_abs_across_sources():
    board = pd.DataFrame({"player_id": [1], "name": ["A"], "rank": [10]})
    yahoo = _resolved_frame([_resolved_row(source="yahoo", source_id="1", name="A", player_id=1, ext_rank=50)])
    fp = _resolved_frame([_resolved_row(source="fantasypros", source_id="1", name="A", player_id=1, ext_rank=11)])
    cmp = build_comparison(board, {"yahoo": yahoo, "fantasypros": fp})
    row = cmp.iloc[0]
    assert row["rank_delta_yahoo"] == 10 - 50
    assert row["rank_delta_fantasypros"] == 10 - 11
    assert row["max_abs_rank_delta"] == 40


def test_unknown_source_key_raises():
    import pytest

    board = _board()
    with pytest.raises(ValueError):
        build_comparison(board, {"espn": pd.DataFrame()})


# --------------------------------------------------------------------------------------------
# Advisory disagreement flag (ADR 0029)
# --------------------------------------------------------------------------------------------

def test_build_comparison_includes_disagreement_columns_by_default():
    board = pd.DataFrame({"player_id": [1], "name": ["A"], "rank": [10]})
    yahoo = _resolved_frame([_resolved_row(source="yahoo", source_id="1", name="A", player_id=1, ext_rank=50)])
    cmp = build_comparison(board, {"yahoo": yahoo})
    for col in ("rankings_disagreement", "rankings_disagreement_direction", "rankings_disagreement_text"):
        assert col in cmp.columns


def test_flag_disagreements_flags_at_or_above_threshold_and_not_below():
    board = pd.DataFrame({"player_id": [1, 2], "name": ["A", "B"], "rank": [10, 10]})
    yahoo = _resolved_frame([
        _resolved_row(source="yahoo", source_id="1", name="A", player_id=1,
                      ext_rank=10 - DISAGREEMENT_THRESHOLD),   # exactly at threshold: delta = +threshold
        _resolved_row(source="yahoo", source_id="2", name="B", player_id=2,
                      ext_rank=10 - (DISAGREEMENT_THRESHOLD - 1)),  # one below threshold
    ])
    cmp = build_comparison(board, {"yahoo": yahoo})
    row1 = cmp[cmp["player_id"] == 1].iloc[0]
    row2 = cmp[cmp["player_id"] == 2].iloc[0]
    assert row1["rankings_disagreement"] == True  # noqa: E712
    assert row2["rankings_disagreement"] == False  # noqa: E712
    assert row2["rankings_disagreement_direction"] == ""
    assert row2["rankings_disagreement_text"] == ""


def test_flag_disagreements_direction_market_favors_when_external_ranks_earlier():
    # our_rank=100, yahoo ranks him 1st (much earlier) -> positive delta -> market favors him more than we do.
    board = pd.DataFrame({"player_id": [1], "name": ["A"], "rank": [100]})
    yahoo = _resolved_frame([_resolved_row(source="yahoo", source_id="1", name="A", player_id=1, ext_rank=1)])
    cmp = build_comparison(board, {"yahoo": yahoo})
    row = cmp.iloc[0]
    assert row["rankings_disagreement"] == True  # noqa: E712
    assert row["rankings_disagreement_direction"] == "market_favors"
    assert "yahoo" in row["rankings_disagreement_text"]
    assert "#1" in row["rankings_disagreement_text"] and "#100" in row["rankings_disagreement_text"]


def test_flag_disagreements_direction_we_favor_when_we_rank_earlier():
    # our_rank=1, yahoo ranks him 200th -> negative delta -> we favor him more than the market does.
    board = pd.DataFrame({"player_id": [1], "name": ["A"], "rank": [1]})
    yahoo = _resolved_frame([_resolved_row(source="yahoo", source_id="1", name="A", player_id=1, ext_rank=200)])
    cmp = build_comparison(board, {"yahoo": yahoo})
    row = cmp.iloc[0]
    assert row["rankings_disagreement_direction"] == "we_favor"
    assert "our board" in row["rankings_disagreement_text"]


def test_flag_disagreements_uses_the_larger_magnitude_source():
    # yahoo delta = 10-190 = -180 (|180|); fantasypros delta = 10-20 = -10 (|10|) -> yahoo drives it.
    board = pd.DataFrame({"player_id": [1], "name": ["A"], "rank": [10]})
    yahoo = _resolved_frame([_resolved_row(source="yahoo", source_id="1", name="A", player_id=1, ext_rank=190)])
    fp = _resolved_frame([_resolved_row(source="fantasypros", source_id="1", name="A", player_id=1, ext_rank=20)])
    cmp = build_comparison(board, {"yahoo": yahoo, "fantasypros": fp})
    row = cmp.iloc[0]
    assert row["rankings_disagreement"] == True  # noqa: E712
    assert "yahoo" in row["rankings_disagreement_text"]
    assert "fantasypros" not in row["rankings_disagreement_text"]


def test_flag_disagreements_blank_when_player_has_no_external_rank_at_all():
    board = pd.DataFrame({"player_id": [1], "name": ["A"], "rank": [1]})
    cmp = build_comparison(board, {})
    row = cmp.iloc[0]
    assert row["rankings_disagreement"] == False  # noqa: E712
    assert row["rankings_disagreement_direction"] == ""
    assert row["rankings_disagreement_text"] == ""


def test_flag_disagreements_respects_a_custom_threshold():
    board = pd.DataFrame({"player_id": [1], "name": ["A"], "rank": [10]})
    yahoo = _resolved_frame([_resolved_row(source="yahoo", source_id="1", name="A", player_id=1, ext_rank=5)])
    cmp = build_comparison(board, {"yahoo": yahoo})  # delta = +5, below the default threshold
    assert cmp.iloc[0]["rankings_disagreement"] == False  # noqa: E712
    tight = flag_disagreements(cmp, threshold=5.0)
    assert tight.iloc[0]["rankings_disagreement"] == True  # noqa: E712
