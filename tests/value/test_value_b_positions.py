"""Dataset position string -> ESPN slot eligibility."""
import numpy as np
import pandas as pd
import pytest

from src.value.positions import (
    ALL_SLOTS, ESPN_POSITIONS, eligible_positions, eligible_slots, is_known, position_group, position_groups,
)


@pytest.mark.parametrize("pos, expected", [
    ("G", ("PG", "SG")),
    ("F", ("SF", "PF")),
    ("C", ("C",)),
    ("G-F", ("SG", "SF")),
    ("F-G", ("SG", "SF")),
    ("F-C", ("PF", "C")),
    ("C-F", ("PF", "C")),
    ("PG", ("PG",)),
    ("sf", ("SF",)),
    ("PG-SG", ("PG", "SG")),
    ("G-C", ("PG", "SG", "C")),
    ("Guard-Forward", ("SG", "SF")),
    ("Forward-Center", ("PF", "C")),
    ("Center", ("C",)),
    ("G/F", ("SG", "SF")),
])
def test_eligible_positions(pos, expected):
    assert eligible_positions(pos) == expected


@pytest.mark.parametrize("bad", [None, np.nan, pd.NA, "", "  ", "??", "X-Y", float("nan")])
def test_unknown_positions_are_never_invented(bad):
    assert eligible_positions(bad) == ()
    assert not is_known(bad)
    assert eligible_slots(bad) == ("UTIL",)          # only UTIL: we do not guess
    assert position_group(bad) == "U"


def test_flex_slots_follow_eligibility():
    assert eligible_slots("G") == ("PG", "SG", "G", "UTIL")
    assert eligible_slots("F-C") == ("PF", "C", "F", "UTIL")
    assert eligible_slots("C") == ("C", "UTIL")           # a pure center has no flex slot except UTIL
    assert eligible_slots("G-F") == ("SG", "SF", "G", "F", "UTIL")
    assert set(eligible_slots("G-F")) <= set(ALL_SLOTS)


def test_every_player_can_fill_util():
    for pos in ["G", "F", "C", "G-F", "F-C", None, "PG"]:
        assert "UTIL" in eligible_slots(pos)


def test_position_group_uses_first_token():
    assert position_group("F-C") == "F" and position_group("C-F") == "C"
    assert position_group("G-F") == "G" and position_group("F-G") == "F"
    assert position_group("PG") == "G" and position_group("PF") == "F"
    assert position_groups(["G", None, "C"]) == ["G", "U", "C"]


def test_eligible_positions_are_valid_espn_positions_and_ordered():
    for pos in ["G", "F", "C", "G-F", "F-C", "G-C", "PG-C"]:
        e = eligible_positions(pos)
        assert set(e) <= set(ESPN_POSITIONS)
        assert list(e) == [p for p in ESPN_POSITIONS if p in e]
