"""fantasypros_rankings: PLAYER NAME field parsing, ECR VS. ADP sentinel handling, and end-to-end
resolution onto NBA player_id."""
from __future__ import annotations

import pandas as pd
import pytest

from src.ingest import fantasypros_rankings as fpr

HEADER = 'RK,"PLAYER NAME",TEAM,BEST,WORST,AVG.,STD.DEV,"ECR VS. ADP"'


def write_csv(tmp_path, rows: list[str]):
    path = tmp_path / "fantasypros.csv"
    path.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- parse_fantasypros_csv

def test_plain_no_tag_row(tmp_path):
    path = write_csv(tmp_path, ['"1","Nikola Jokic (DEN - C)",,"1","1","1.0","0.0","0"'])
    df = fpr.parse_fantasypros_csv(path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["source_id"] == "1"
    assert row["source_name_raw"] == "Nikola Jokic (DEN - C)"
    assert row["name_parsed"] == "Nikola Jokic"
    assert row["team"] == "DEN"
    assert row["positions"] == "C"
    assert row["status_tag"] is None
    assert row["ext_rank"] == 1
    assert row["adp"] is None
    assert row["ecr_vs_adp"] == 0


def test_out_tag(tmp_path):
    path = write_csv(tmp_path, ['"8","Anthony Davis (WAS - PF,C) OUT",,"8","15","9.8","2.7","+4"'])
    df = fpr.parse_fantasypros_csv(path)
    row = df.iloc[0]
    assert row["name_parsed"] == "Anthony Davis"
    assert row["team"] == "WAS"
    assert row["positions"] == "PF,C"
    assert row["status_tag"] == "OUT"
    assert row["ecr_vs_adp"] == 4


def test_dtd_tag_and_dash_sentinel_becomes_null(tmp_path):
    path = write_csv(tmp_path, ['"183","Jaden Ivey (FA - PG,SG) DTD",,"121","184","167.8","27.0","-"'])
    df = fpr.parse_fantasypros_csv(path)
    row = df.iloc[0]
    assert row["name_parsed"] == "Jaden Ivey"
    assert row["team"] == "FA"
    assert row["positions"] == "PG,SG"
    assert row["status_tag"] == "DTD"
    assert row["ecr_vs_adp"] is None or pd.isna(row["ecr_vs_adp"])


def test_two_way_tag(tmp_path):
    path = write_csv(tmp_path, ['"211","Cam Whitmore (DEN - SF,PF) TWO-WAY",,"158","211","196.8","22.4","+11"'])
    df = fpr.parse_fantasypros_csv(path)
    row = df.iloc[0]
    assert row["name_parsed"] == "Cam Whitmore"
    assert row["team"] == "DEN"
    assert row["positions"] == "SF,PF"
    assert row["status_tag"] == "TWO-WAY"
    assert row["ecr_vs_adp"] == 11


def test_ret_tag_with_suffix_in_name(tmp_path):
    path = write_csv(tmp_path, ['"176","Russell Westbrook III (FA - PG,SG) RET",,"132","176","159.3","18.3","+16"'])
    df = fpr.parse_fantasypros_csv(path)
    row = df.iloc[0]
    assert row["name_parsed"] == "Russell Westbrook III"  # suffix retained; id_map strips downstream
    assert row["team"] == "FA"
    assert row["positions"] == "PG,SG"
    assert row["status_tag"] == "RET"
    assert row["ecr_vs_adp"] == 16


def test_jr_suffix_no_tag(tmp_path):
    path = write_csv(tmp_path, ['"161","Darius Acuff Jr. (FA - G)",,"87","161","140.3","30.9","-"'])
    df = fpr.parse_fantasypros_csv(path)
    row = df.iloc[0]
    assert row["name_parsed"] == "Darius Acuff Jr."
    assert row["team"] == "FA"
    assert row["positions"] == "G"
    assert row["status_tag"] is None
    assert row["ecr_vs_adp"] is None or pd.isna(row["ecr_vs_adp"])


def test_fa_team_preserved_as_string_not_null(tmp_path):
    path = write_csv(tmp_path, ['"156","Cam Thomas (FA - PG,SF,SG)",,"76","156","135.8","34.5","+9"'])
    df = fpr.parse_fantasypros_csv(path)
    row = df.iloc[0]
    assert row["team"] == "FA"
    assert isinstance(row["team"], str)
    assert row["positions"] == "PG,SF,SG"


def test_signed_positive_and_negative_ecr_vs_adp(tmp_path):
    path = write_csv(tmp_path, [
        '"50","Some Player (HOU - PG)",,"40","60","50.0","5.0","+25"',
        '"51","Other Player (HOU - SG)",,"41","61","51.0","5.0","-29"',
    ])
    df = fpr.parse_fantasypros_csv(path)
    assert df.iloc[0]["ecr_vs_adp"] == 25
    assert df.iloc[1]["ecr_vs_adp"] == -29


def test_adp_column_always_none(tmp_path):
    path = write_csv(tmp_path, [
        '"1","Nikola Jokic (DEN - C)",,"1","1","1.0","0.0","0"',
        '"2","Luka Doncic (LAL - PG,SG)",,"2","4","3.0","0.6","+1"',
    ])
    df = fpr.parse_fantasypros_csv(path)
    assert df["adp"].apply(lambda v: v is None).all()


def test_multiple_positions_all_parsed(tmp_path):
    path = write_csv(tmp_path, ['"3","Luka Doncic (LAL - PG,SG)",,"2","4","3.0","0.6","+1"'])
    df = fpr.parse_fantasypros_csv(path)
    assert df.iloc[0]["positions"] == "PG,SG"


def test_malformed_player_name_raises(tmp_path):
    path = write_csv(tmp_path, ['"5","GarbageNoParens",,"5","5","5.0","0.0","0"'])
    with pytest.raises(fpr.FantasyProsParseError):
        fpr.parse_fantasypros_csv(path)


def test_missing_required_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text('RK,"PLAYER NAME"\n"1","Nikola Jokic (DEN - C)"\n', encoding="utf-8")
    with pytest.raises(fpr.FantasyProsParseError):
        fpr.parse_fantasypros_csv(path)


# --------------------------------------------------------------------------- load_fantasypros_rankings

def make_players(rows):
    """rows: list of (player_id, player_name, from_year, to_year)."""
    return pd.DataFrame(rows, columns=["player_id", "player_name", "from_year", "to_year"])


def test_load_fantasypros_rankings_exact_alias_and_unmatched(tmp_path):
    players = make_players([
        (1, "Nikola Jokic", 2015, 2026),
        (2, "Cameron Thomas", 2021, 2026),  # id_map.ALIAS_GROUPS: "cam thomas" / "cameron thomas"
    ])
    path = write_csv(tmp_path, [
        '"1","Nikola Jokic (DEN - C)",,"1","1","1.0","0.0","0"',            # exact match
        '"2","Cam Thomas (BKN - SG) OUT",,"20","30","25.0","3.0","-5"',     # alias match
        '"3","Totally Unknown Player (FA - PG)",,"200","250","220.0","10.0","-"',  # unmatched
    ])

    result = fpr.load_fantasypros_rankings(path, players)
    frame = result.frame

    assert len(frame) == 3  # never silently dropped

    jokic = frame[frame["source_id"] == "1"].iloc[0]
    assert jokic["player_id"] == 1
    assert bool(jokic["matched"]) is True
    assert jokic["match_method"] == "exact"

    trent = frame[frame["source_id"] == "2"].iloc[0]
    assert trent["player_id"] == 2
    assert bool(trent["matched"]) is True
    assert trent["match_method"] == "normalized"
    assert trent["status_tag"] == "OUT"

    unknown = frame[frame["source_id"] == "3"].iloc[0]
    assert pd.isna(unknown["player_id"])
    assert bool(unknown["matched"]) is False
    assert unknown["match_method"] == "unmatched"

    assert result.report.n_total == 3
    assert result.report.n_unmatched == 1
