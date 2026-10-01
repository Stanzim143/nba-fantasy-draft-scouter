"""Labelled-injury features (ADR 0033): reason classification, the per-season panel, the feature matrix, and leakage."""
import numpy as np
import pandas as pd
import pytest

from src.features.injury_labels import (MIN_SEASON_COVERAGE, InjuryLabelFeatures, assert_labels_no_future,
                                        build_label_panel, classify_reason)


# --------------------------------------------------------------------------- classification

@pytest.mark.parametrize("category, detail, expected", [
    ("Injury/Illness", "Right Ankle; Sprain", ("injury", "foot_ankle")),
    ("Injury/Illness", "Left Knee; Sprain", ("injury", "knee")),
    ("Injury/Illness", "Left hamstring tightness", ("injury", "leg_hip")),
    ("Injury/Illness", "Low back surgery", ("recovery", "back")),
    ("Injury/Illness", "Right Shoulder; Injury Recovery", ("recovery", "upper_limb")),
    ("Injury/Illness", "Concussion Protocol", ("injury", "head_neck")),
    ("Injury/Illness", "N/a; Non-Covid related illness", ("illness", "")),
    ("Injury/Illness", "Left Knee; Injury Management", ("rest", "")),
    ("Injury/Illness", "Foot; Load Management", ("rest", "")),
    ("Rest", "", ("rest", "")),
    ("G League Team", "", ("g_league", "")),
    ("G League", "Two-Way", ("g_league", "")),
    ("G League", "On Assignment", ("g_league", "")),
    ("Personal Reasons", "", ("personal", "")),
    ("Not With Team", "", ("not_with_team", "")),
    ("Health and Safety Protocols", "", ("covid", "")),
    ("Suspension", "", ("suspension", "")),
    ("Injury/Illness", "Something unusual", ("injury", "other")),
    ("Return to Competition Reconditioning", "", ("recovery", "")),
    ("Something new", "unheard of", ("other", "")),
    (None, None, ("other", "")),
])
def test_classify_reason(category, detail, expected):
    assert classify_reason(category, detail) == expected


# --------------------------------------------------------------------------- panel


def _team_games(dates, season="2019-20", team="AAA"):
    return pd.DataFrame({"season": season, "team_abbr": team, "game_date": pd.to_datetime(dates),
                         "game_id": [f"{team}{i}" for i in range(len(dates))]})


def _reports(rows, season="2019-20"):
    df = pd.DataFrame(rows, columns=["game_date", "player_id", "status", "category", "detail", "team_abbr"])
    df["season"] = season
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["player_id"] = pd.array(df["player_id"], dtype="Int64")
    return df


DATES = [f"2019-11-{d:02d}" for d in range(1, 11)]


def test_panel_shares_longest_spell_and_number_of_spells():
    tg = _team_games(DATES)
    # player 1: Out (injury) games 2-4 and 7; rest on game 9; G League on game 10. Player 2 is only on the report for game 1.
    rows = [(DATES[i], 1, "Out", "Injury/Illness", "Left Knee; Sprain", "AAA") for i in (1, 2, 3, 6)]
    rows += [(DATES[8], 1, "Out", "Rest", "", "AAA"), (DATES[9], 1, "Out", "G League", "On Assignment", "AAA"),
             (DATES[0], 2, "Questionable", "Injury/Illness", "Right Ankle; Sprain", "AAA"),
             (DATES[0], 3, "Out", "Injury/Illness", "Right Ankle; Sprain", "AAA")]
    rep = _reports(rows)
    # covered = dates that have a report: only the rows' dates are "covered"; add a report row for every date by a third player
    rep = pd.concat([rep, _reports([(d, 9, "Available", "", "", "AAA") for d in DATES])], ignore_index=True)
    panel, cov = build_label_panel(rep, tg)
    p = panel.set_index("player_id")
    assert cov.loc[0, "share"] == 1.0
    assert p.loc[1, "inj_out"] == pytest.approx(4 / 10) and p.loc[1, "rest_out"] == pytest.approx(1 / 10)
    assert p.loc[1, "other_out"] == pytest.approx(1 / 10)
    assert p.loc[1, "longest"] == pytest.approx(3 / 10) and p.loc[1, "n_spells"] == 2
    assert p.loc[3, "inj_out"] == pytest.approx(1 / 10) and p.loc[3, "n_spells"] == 1
    assert 2 not in p.index and 9 not in p.index                         # never Out: no row (the feature layer treats it as zero)


def test_panel_uses_only_dates_that_have_a_report():
    tg = _team_games(DATES)
    covered = DATES[:5]                                                  # reports exist only for the first five game dates
    rep = _reports([(covered[0], 1, "Out", "Injury/Illness", "Left Knee; Sprain", "AAA"),
                    (covered[1], 1, "Out", "Injury/Illness", "Left Knee; Sprain", "AAA")] +
                   [(d, 9, "Available", "", "", "AAA") for d in covered])
    panel, cov = build_label_panel(rep, tg)
    assert cov.loc[0, "share"] == pytest.approx(0.5)
    assert panel.set_index("player_id").loc[1, "inj_out"] == pytest.approx(2 / 5)    # 2 of the 5 covered games, not of 10


def test_panel_handles_empty_inputs():
    tg = _team_games(DATES)
    assert build_label_panel(_reports([]), tg)[0].empty
    assert build_label_panel(_reports([(DATES[0], 1, "Available", "", "", "AAA")]), tg)[0].empty


# --------------------------------------------------------------------------- features

def _feats(decay=0.5):
    tg = pd.concat([_team_games(DATES, "2019-20"), _team_games([d.replace("2019", "2020") for d in DATES], "2020-21")],
                   ignore_index=True)
    rows19 = [(DATES[i], 1, "Out", "Injury/Illness", "Left Knee; Sprain", "AAA") for i in range(4)]
    rows19 += [(d, 9, "Available", "", "", "AAA") for d in DATES]
    rows20 = [(d.replace("2019", "2020"), 9, "Available", "", "", "AAA") for d in DATES]
    rep = pd.concat([_reports(rows19, "2019-20"), _reports(rows20, "2020-21")], ignore_index=True)
    return InjuryLabelFeatures.from_reports(rep, tg, "2021-22", n_lags=3, decay=decay)


def test_feature_columns_recency_weighting_and_zero_for_healthy_covered_seasons():
    f = _feats()
    assert f.covered_seasons == frozenset({2019, 2020})
    out = f.build([1, 9, 77], [2021, 2021, 2021])
    # player 1: lag1 = 2020-21 (healthy, zero), lag2 = 2019-20 (inj_out 0.4): weights 1, 0.5 -> 0.4 * 0.5 / 1.5
    assert out[0, 0] == 1.0 and out[0, 1] == pytest.approx(0.4 * 0.5 / 1.5) and out[0, 4] == pytest.approx(0.4 * 0.5 / 1.5)
    assert out[0, 5] == pytest.approx(1 * 0.5 / 1.5)                        # n_spells
    assert (out[1, 1:] == 0).all() and out[1, 0] == 1.0                      # listed only as Available: healthy, covered
    assert (out[2, 1:] == 0).all() and out[2, 0] == 1.0                      # unknown player in covered seasons: zeros


def test_uncovered_seasons_are_all_zero_and_flagged():
    f = _feats()
    early = f.build([1], [2014])[0]
    assert early.tolist() == [0.0] * 6                                       # lags 2013, 2012, 2011: no reports at all
    only_old = f.build([1], [2020])[0]                                       # lag1 = 2019 covered, lags 2018.. not
    assert only_old[0] == 1.0 and only_old[1] == pytest.approx(0.4)          # the uncovered lags do not dilute the covered one


def test_low_coverage_season_is_not_trusted():
    tg = _team_games(DATES)
    rep = _reports([(DATES[0], 1, "Out", "Injury/Illness", "Left Knee; Sprain", "AAA")])       # 1 of 10 dates covered
    f = InjuryLabelFeatures.from_reports(rep, tg, "2020-21", n_lags=2, decay=0.5)
    assert f.covered_seasons == frozenset() and MIN_SEASON_COVERAGE > 0.1
    assert f.build([1], [2020]).tolist() == [[0.0] * 6]


def test_future_reports_are_refused():
    tg = _team_games(DATES)
    rep = _reports([(DATES[0], 1, "Out", "Injury/Illness", "Left Knee; Sprain", "AAA")])
    with pytest.raises(AssertionError, match="leakage"):
        InjuryLabelFeatures.from_reports(rep, tg, "2019-20", n_lags=2, decay=0.5)
    with pytest.raises(AssertionError, match="leakage"):
        assert_labels_no_future(rep, "2019-20")
    assert_labels_no_future(rep, "2020-21")
    assert np.isfinite(_feats().build([1], [2021])).all()
