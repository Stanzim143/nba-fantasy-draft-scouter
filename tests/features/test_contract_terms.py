"""src.features.contract_terms: the as-of-season-start contract state, its unknown handling, the leakage
guard, extension chains and the combination with the rookie-scale clock (ADR 0019)."""
import numpy as np
import pandas as pd
import pytest

from src.contracts import season_str
from src.features.contract import contract_clock
from src.features.contract_terms import (
    TERMS_FEATURE_NAMES, assert_contracts_no_future, coverage_by_season, fit_terms_adjustment, slice_contracts,
    terms_features,
)
from src.ingest.wiki_contracts import PLAYER_CONTRACTS_COLUMNS, coverage_start_year, tag_start_year


def ev(pid, event, date, years=None, **kw):
    """One player_contracts row for an event dated ``date`` (tag and coverage year derived exactly as the ingest does)."""
    d = pd.Timestamp(date) if date else pd.NaT
    tag, cov = (2015, 2015) if pd.isna(d) else (tag_start_year(d), coverage_start_year(d))
    row = dict(season=season_str(tag), page_season=season_str(tag),
               start_year=cov, team_id=1, player_id=pid, event=event, contract_type="standard",
               years=years if years is not None else pd.NA, days=pd.NA, amount_usd=np.nan, signed_date=d,
               date_source="cell", is_two_way=False, is_ten_day=False, is_exhibit10=False, is_minimum=False,
               is_extension=event == "extension", is_rookie=False, non_guaranteed=False, has_option=False,
               multiyear_unspecified=False, section="add", terms_source="cell")
    row.update(kw)
    return row


def table(*rows):
    df = pd.DataFrame(list(rows))[PLAYER_CONTRACTS_COLUMNS]
    df["years"] = pd.array(df["years"], dtype="Int64")
    return df


def feats(rows, target, pids, clock=None):
    return terms_features(table(*rows) if rows else None, target, np.array(pids), clock)


# --------------------------------------------------------------------------- the state machine

def test_contract_year_is_the_final_season_of_a_known_deal():
    f = feats([ev(1, "sign", "2023-07-06", 3)], "2025-26", [1])          # 2023-24, 2024-25, 2025-26
    assert f.status[0] == "known" and f.years_remaining[0] == 1 and f.contract_year[0]
    assert feats([ev(1, "sign", "2023-07-06", 3)], "2024-25", [1]).years_remaining[0] == 2
    assert feats([ev(1, "sign", "2023-07-06", 3)], "2023-24", [1]).years_remaining[0] == 3
    f = feats([ev(1, "sign", "2023-07-06", 3)], "2026-27", [1])          # ran out before this season
    assert f.status[0] == "lapsed" and not f.contract_year[0] and np.isnan(f.years_remaining[0])


def test_unknown_is_never_not_a_contract_year():
    f = feats([ev(1, "sign", "2023-07-06", 3)], "2025-26", [1, 2, 3])
    assert list(f.status) == ["known", "unknown", "unknown"]
    assert np.isnan(f.years_remaining[1:]).all()                          # NaN, not 0 or 5
    assert not f.contract_year[1:].any() and not f.design()[1:].any()     # every indicator zero, no separate "not CY" class
    assert feats([], "2025-26", [7]).status[0] == "unknown"               # no table at all
    empty = table(ev(9, "sign", "2023-07-06", 1)).iloc[0:0]
    assert feats(empty.to_dict("records"), "2025-26", [7]).status[0] == "unknown"


def test_a_signing_with_no_stated_length_is_not_a_contract_year():
    f = feats([ev(1, "sign", "2025-07-06", None)], "2025-26", [1])
    assert f.status[0] == "no_length" and not f.contract_year[0] and np.isnan(f.years_remaining[0])
    assert f.new_deal[0]                                                    # but the fresh signing is known


def test_new_deal_means_signed_between_july_and_the_target_seasons_first_day():
    rows = [ev(1, "sign", "2025-07-06", 2), ev(2, "sign", "2025-02-01", 2), ev(3, "resign", "2025-09-30", 1)]
    f = feats(rows, "2025-26", [1, 2, 3])
    assert list(f.new_deal) == [True, False, True]
    assert list(f.years_remaining) == [2, 1, 1]      # the Feb 2025 deal covers 2024-25 and 2025-26
    f2 = feats(rows + [ev(4, "sign", "2025-10-01", 2)], "2025-26", [4])       # Oct 1 or later: not known yet
    assert f2.status[0] == "unknown"


def test_latest_deal_wins_and_end_events_lapse_it():
    rows = [ev(1, "sign", "2021-07-01", 2), ev(1, "resign", "2023-07-05", 3)]
    f = feats(rows, "2024-25", [1])
    assert f.years_remaining[0] == 2                                        # re-signed deal covers 2023-24..2025-26
    rows = [ev(2, "sign", "2023-07-01", 4), ev(2, "waived", "2024-01-10")]
    assert feats(rows, "2024-25", [2]).status[0] == "lapsed"                # waived: the old deal is gone
    rows += [ev(2, "sign", "2024-02-01", 1)]
    assert feats(rows, "2024-25", [2]).status[0] == "lapsed"                # 1-year deal from Feb 2024 covered 2023-24 only
    assert feats(rows, "2023-24", [2]).years_remaining[0] == 4               # at 2023-24's start only the July deal is known
    rows = [ev(3, "sign", "2023-07-01", 4), ev(3, "waived", "2024-01-10"), ev(3, "claimed", "2024-01-11")]
    assert feats(rows, "2024-25", [3]).years_remaining[0] == 3              # claimed off waivers: contract follows him
    assert feats([ev(4, "sign", "2023-07-01", 4), ev(4, "expired", "2024-07-01")], "2025-26", [4]).status[0] == "lapsed"


def test_ten_day_contracts_never_set_the_status():
    r10 = ev(1, "sign", "2025-02-01", None, is_ten_day=True, contract_type="ten_day")
    assert feats([r10], "2025-26", [1]).status[0] == "unknown"
    assert feats([ev(1, "sign", "2023-07-01", 4), r10], "2025-26", [1]).years_remaining[0] == 2


def test_extension_adds_years_after_the_running_deal():
    rows = [ev(1, "sign", "2022-07-01", 2), ev(1, "extension", "2023-10-20", 3)]      # deal ends 2023-24, +3 -> 2026-27
    f = feats(rows, "2024-25", [1])
    assert f.years_remaining[0] == 3 and f.extension[0] and not f.contract_year[0]
    assert feats(rows, "2026-27", [1]).years_remaining[0] == 1
    # no earlier deal known: assume the extension starts after the season in which it was signed
    f = feats([ev(2, "extension", "2023-10-23", 3)], "2025-26", [2])
    assert f.years_remaining[0] == 2                                                  # 2024-25, 25-26, 26-27
    # extension with unstated length: only the fact that he was extended is known
    f = feats([ev(3, "sign", "2021-07-01", 2), ev(3, "extension", "2022-08-10", None)], "2022-23", [3])
    assert f.status[0] == "no_length" and f.extension[0] and not f.contract_year[0]
    # the flag fades once the extension is more than a season old
    assert not feats(rows, "2026-27", [1]).extension[0]


def test_two_way_and_minimum_flags_follow_the_running_deal():
    rows = [ev(1, "sign", "2025-07-01", 2, is_two_way=True, contract_type="two_way"),
            ev(2, "sign", "2025-07-01", 1, is_minimum=True, contract_type="minimum"),
            ev(3, "sign", "2025-07-01", None, is_two_way=True, contract_type="two_way")]
    f = feats(rows, "2025-26", [1, 2, 3])
    assert list(f.two_way) == [True, False, True] and list(f.minimum) == [False, True, False]
    assert not feats(rows, "2026-27", [1]).two_way[0] or feats(rows, "2026-27", [1]).years_remaining[0] == 1


def test_undated_rows_are_ordered_by_their_page_season():
    undated = ev(1, "sign", None, 2, date_source="none")
    f = feats([undated], "2016-17", [1])
    assert f.years_remaining[0] == 1                                                 # 2015-16 and 2016-17
    assert feats([undated], "2015-16", [1]).status[0] == "unknown"                   # page season: invisible to itself


# --------------------------------------------------------------------------- leakage

def test_rows_tagged_at_or_after_the_target_are_ignored_even_if_unsliced():
    rows = [ev(1, "sign", "2023-07-06", 3)]
    future = [ev(1, "resign", "2025-12-01", 5), ev(2, "sign", "2026-02-01", 1), ev(3, "sign", "2025-10-01", 2)]
    clean = feats(rows, "2025-26", [1, 2, 3])
    dirty = feats(rows + future, "2025-26", [1, 2, 3])                     # the caller forgot to slice
    for a in ("status", "years_remaining", "contract_year", "new_deal", "extension"):
        np.testing.assert_array_equal(getattr(clean, a), getattr(dirty, a))
    with pytest.raises(AssertionError, match="leakage"):
        assert_contracts_no_future(table(*rows, *future), "2025-26")
    assert len(slice_contracts(table(*rows, *future), "2025-26")) == 1


def test_bending_only_future_rows_changes_no_feature():
    """Scramble every future row (years, dates, players); features of the target season stay bit-identical."""
    rows = [ev(1, "sign", "2023-07-06", 3), ev(2, "sign", "2024-07-06", 2), ev(3, "extension", "2024-11-01", 2)]
    future = [ev(1, "resign", "2025-11-05", 4), ev(2, "waived", "2026-01-05"), ev(3, "sign", "2025-10-30", 1)]
    t1, t2 = table(*rows, *future), table(*rows, *future)
    m = t2["season"].map(lambda s: int(s[:4])) >= 2025
    t2.loc[m, "years"] = pd.array([1] * int(m.sum()), dtype="Int64")
    t2.loc[m, "player_id"] = t2.loc[m, "player_id"].to_numpy()[::-1]
    t2.loc[m, "event"] = "expired"
    a, b = terms_features(t1, "2025-26", np.array([1, 2, 3])), terms_features(t2, "2025-26", np.array([1, 2, 3]))
    np.testing.assert_array_equal(a.design(), b.design())
    np.testing.assert_array_equal(a.years_remaining, b.years_remaining)


def test_an_event_dated_before_opening_night_is_visible_and_a_later_one_is_not():
    known = feats([ev(1, "sign", "2025-09-30", 1)], "2025-26", [1])
    assert known.status[0] == "known" and known.contract_year[0] and known.new_deal[0]
    hidden = feats([ev(1, "sign", "2025-10-01", 1)], "2025-26", [1])
    assert hidden.status[0] == "unknown"


# --------------------------------------------------------------------------- rookie clock

def _players(rows):
    df = pd.DataFrame(rows, columns=["player_id", "draft_year", "draft_round", "draft_number"])
    for c in df.columns:
        df[c] = df[c].astype("Int64")
    return df


def test_rookie_scale_clock_only_where_no_wiki_deal_is_known():
    players = _players([(1, 2022, 1, 10), (2, 2022, 1, 12), (3, 2023, 1, 8), (4, 2018, 1, 3)])
    pids = np.array([1, 2, 3, 4])
    clock = contract_clock(players, "2025-26", pids)
    f = terms_features(table(ev(2, "extension", "2025-07-10", 4)), "2025-26", pids, clock)   # player 2 extended
    assert list(f.rookie_cy) == [True, False, False, False]        # 1: nominal final scale year, unknown deal
    assert list(f.rookie_opt) == [False, False, True, False]       # 3: nominal option year
    assert f.status[1] == "known" and f.extension[1]               # 2: the wiki-stated deal beats the nominal scale
    assert list(f.basis()) == ["rookie_scale", "wiki", "rookie_scale", ""]
    assert list(f.flag()) == ["rookie_contract_year", "extension", "rookie_option_year", ""]


def test_design_matrix_shape_and_columns():
    f = feats([ev(1, "sign", "2023-07-06", 3), ev(2, "sign", "2024-07-06", 3), ev(3, "sign", "2025-07-06", 1)],
              "2025-26", [1, 2, 3, 4])
    X = f.design()
    assert X.shape == (4, len(TERMS_FEATURE_NAMES)) and set(np.unique(X)) <= {0.0, 1.0}
    col = {n: i for i, n in enumerate(TERMS_FEATURE_NAMES)}
    assert X[0, col["cy"]] == 1 and X[1, col["yr2"]] == 1 and X[2, col["cy"]] == 1 and X[2, col["new_deal"]] == 1
    assert X[3].sum() == 0


# --------------------------------------------------------------------------- fit gate and coverage

def test_fit_is_off_without_enough_marked_rows_and_on_for_a_planted_effect():
    rng = np.random.default_rng(0)
    n = 1200
    X = np.zeros((n, len(TERMS_FEATURE_NAMES)))
    X[: n // 4, 0] = 1                                              # a quarter are contract-year rows
    season = np.tile(np.arange(2016, 2022), n // 6 + 1)[:n]
    w = np.full(n, 60.0)
    y = rng.normal(0, 3, n)
    from src.features.contract_terms import TermsTrainingSet
    null = fit_terms_adjustment(TermsTrainingSet(X, y, w, season, np.ones(n, bool)))
    assert not null.enabled                                          # noise: the gate stays shut
    planted = fit_terms_adjustment(TermsTrainingSet(X, y + 2.5 * X[:, 0], w, season, np.ones(n, bool)))
    assert planted.enabled and planted.diagnostics["coefficients_fppg"]["cy"] == pytest.approx(2.5, abs=0.4)
    sparse = np.zeros_like(X)
    sparse[:5, 0] = 1
    off = fit_terms_adjustment(TermsTrainingSet(sparse, y + 2.5 * sparse[:, 0], w, season, np.ones(n, bool)))
    assert not off.enabled and "clock-marked" in off.diagnostics["reason"]


def test_coverage_by_season_counts_only_scored_veterans():
    gl = pd.DataFrame({
        "player_id": [1] * 20 + [2] * 20 + [3] * 20 + [1] * 20 + [2] * 5 + [3] * 20,
        "season": ["2023-24"] * 60 + ["2024-25"] * 45,
    })
    contracts = table(ev(1, "sign", "2023-07-06", 3), ev(3, "sign", "2024-07-06", 1, is_minimum=True))
    cov = coverage_by_season(contracts, gl, ["2024-25"], min_gp=10)
    r = cov.iloc[0]
    assert r["veterans"] == 2                      # player 2 played 5 games: not scored
    assert r["known"] == 2 and r["contract_year"] == 1 and r["new_deal"] == 1 and r["minimum"] == 1
    assert r["known_pct"] == 1.0
