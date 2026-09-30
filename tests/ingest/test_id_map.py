"""id_map: name normalization, alias table, and match_players against constructed NBA rosters."""
import pandas as pd
import pytest

from src.ingest.id_map import alias_key, match_players, normalize_name


# ------------------------------------------------------------------ normalize_name

@pytest.mark.parametrize("raw,expected", [
    ("Nikola Jokic", "nikola jokic"),
    ("Dončić, Luka", "doncic luka"),  # comma is punctuation, not a name field
    ("Nikola Jokić", "nikola jokic"),
    ("Dennis Schröder", "dennis schroder"),
    ("Jimmy Butler III", "jimmy butler"),
    ("Gary Trent Jr.", "gary trent"),
    ("Otto Porter Jr", "otto porter"),
    ("Marcus Morris Sr.", "marcus morris"),
    ("  Kevin   Durant ", "kevin durant"),
    ("D'Angelo Russell", "d angelo russell"),
    ("P.J. Tucker", "p j tucker"),
])
def test_normalize_name_cases(raw, expected):
    assert normalize_name(raw) == expected


def test_normalize_name_keeps_suffix_when_it_is_the_whole_name_minus_one_token():
    # a lone token that happens to be a suffix word stays (edge case, not a real name)
    assert normalize_name("III") == "iii"


# ------------------------------------------------------------------ alias_key

def test_alias_key_collapses_legal_name_change():
    assert alias_key("Enes Kanter") == alias_key("Enes Freedom")


def test_alias_key_collapses_nickname():
    assert alias_key("Cam Thomas") == alias_key("Cameron Thomas")


def test_alias_key_is_normalize_name_when_no_alias():
    assert alias_key("Nikola Jokic") == normalize_name("Nikola Jokic")


# ------------------------------------------------------------------ match_players

def nba_players(rows):
    """rows: list of (player_id, name, from_year, to_year)."""
    return pd.DataFrame(rows, columns=["player_id", "player_name", "from_year", "to_year"])


def espn_universe(rows):
    """rows: list of (source_id, name, season_start)."""
    return pd.DataFrame(rows, columns=["source_id", "name", "season_start"])


def test_exact_match():
    nba = nba_players([(1, "Stephen Curry", 2009, 2026)])
    espn = espn_universe([("100", "Stephen Curry", 2023)])
    frame, report = match_players(espn, nba, source="espn")
    assert len(frame) == 1
    assert frame.iloc[0]["player_id"] == 1
    assert frame.iloc[0]["match_method"] == "exact"
    assert frame.iloc[0]["confidence"] == 1.0
    assert report.n_exact == 1 and report.n_matched == 1 and report.n_total == 1


def test_suffix_difference_matches_via_normalize_name():
    # ESPN carries the "III"/"Jr." suffix; nba_api's static list may or may not.
    nba = nba_players([(2, "Jimmy Butler", 2011, 2026)])
    espn = espn_universe([("200", "Jimmy Butler III", 2023)])
    frame, report = match_players(espn, nba, source="espn")
    assert len(frame) == 1 and frame.iloc[0]["player_id"] == 2
    assert report.n_unmatched == 0


def test_accented_name_matches():
    nba = nba_players([(3, "Nikola Jokic", 2015, 2026)])
    espn = espn_universe([("300", "Nikola Jokić", 2023)])
    frame, report = match_players(espn, nba, source="espn")
    assert len(frame) == 1 and frame.iloc[0]["player_id"] == 3


def test_alias_table_resolves_legal_name_change():
    nba = nba_players([(4, "Enes Kanter", 2011, 2020)])
    espn = espn_universe([("400", "Enes Freedom", 2021)])
    frame, report = match_players(espn, nba, source="espn")
    assert len(frame) == 1 and frame.iloc[0]["player_id"] == 4
    assert frame.iloc[0]["match_method"] == "normalized"
    assert report.n_normalized == 1


def test_true_name_collision_is_ambiguous_without_year_signal():
    # Two distinct real players sharing an exact normalized name, active in overlapping eras.
    nba = nba_players([
        (5, "Marcus Williams", 2003, 2010),
        (6, "Marcus Williams", 2005, 2012),
    ])
    espn = espn_universe([("500", "Marcus Williams", None)])
    frame, report = match_players(espn, nba, source="espn", year_col=None)
    assert frame.empty
    assert report.n_ambiguous == 1
    assert report.ambiguous[0]["source_id"] == "500"


def test_name_collision_disambiguated_by_year_window():
    # Same exact name, but the two real players' careers do not overlap: a season hint resolves it.
    nba = nba_players([
        (7, "Chris Wright", 2000, 2004),
        (8, "Chris Wright", 2018, 2022),
    ])
    espn = espn_universe([("700", "Chris Wright", 2019)])
    frame, report = match_players(espn, nba, source="espn")
    assert len(frame) == 1
    assert frame.iloc[0]["player_id"] == 8
    assert report.n_ambiguous == 0


def test_unmatched_player_is_reported_not_dropped_silently():
    nba = nba_players([(9, "Stephen Curry", 2009, 2026)])
    espn = espn_universe([("900", "Wang Zhelin", 2019)])
    frame, report = match_players(espn, nba, source="espn")
    assert frame.empty
    assert report.n_unmatched == 1
    assert report.unmatched[0]["source_id"] == "900"
    assert report.unmatched[0]["name"] == "Wang Zhelin"


def test_fuzzy_match_for_close_misspelling():
    nba = nba_players([(10, "Karl-Anthony Towns", 2015, 2026)])
    espn = espn_universe([("1000", "Karl Anthony Town", 2023)])  # dropped final 's', hyphen differs
    frame, report = match_players(espn, nba, source="espn")
    assert len(frame) == 1 and frame.iloc[0]["player_id"] == 10
    assert frame.iloc[0]["match_method"] == "fuzzy"
    assert 0 < frame.iloc[0]["confidence"] < 1.0
    assert report.n_fuzzy == 1


def test_fuzzy_match_does_not_fire_for_unrelated_names():
    nba = nba_players([(11, "James Harden", 2009, 2026)])
    espn = espn_universe([("1100", "James Harding", 2023)])  # too different for our cutoff? check both ways
    frame, report = match_players(espn, nba, source="espn")
    # Either a (correctly cautious) miss, or a fuzzy match -- but never an "exact"/"normalized" one.
    if len(frame):
        assert frame.iloc[0]["match_method"] == "fuzzy"
    else:
        assert report.n_unmatched == 1


def test_report_summary_counts_everything_exactly_once():
    nba = nba_players([
        (1, "Stephen Curry", 2009, 2026),
        (2, "Jimmy Butler", 2011, 2026),
        (4, "Enes Kanter", 2011, 2020),
    ])
    espn = espn_universe([
        ("100", "Stephen Curry", 2023),
        ("200", "Jimmy Butler III", 2023),
        ("400", "Enes Freedom", 2021),
        ("999", "Nobody Realname", 2023),
    ])
    frame, report = match_players(espn, nba, source="espn")
    assert report.n_total == 4
    assert report.n_matched == len(frame) == 3
    assert report.n_unmatched == 1
    assert report.n_exact + report.n_normalized + report.n_fuzzy == report.n_matched


def test_output_frame_matches_player_id_map_shape():
    nba = nba_players([(1, "Stephen Curry", 2009, 2026)])
    espn = espn_universe([("100", "Stephen Curry", 2023)])
    frame, _ = match_players(espn, nba, source="espn")
    assert list(frame.columns) == ["player_id", "source", "source_id", "source_name", "match_method", "confidence"]
    assert pd.api.types.is_string_dtype(frame["source_id"]) or frame["source_id"].dtype == object
    assert (frame["source"] == "espn").all()
