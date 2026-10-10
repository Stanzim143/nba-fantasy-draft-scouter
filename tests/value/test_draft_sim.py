"""Draft-availability simulator (ADR 0040): snake math, monotonicity, determinism, edge cases, advisory-only."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.app import glossary as G
from src.value import draft_sim as DS


def _board(n=40, *, with_adp=True):
    b = pd.DataFrame({"rank": np.arange(1, n + 1), "player_id": np.arange(100, 100 + n), "name": [f"P{i}" for i in range(n)],
                      "position": "PG", "proj_fppg": np.linspace(50, 20, n), "vorp": np.linspace(900, -50, n), "tier": 1})
    if with_adp:
        b["adp"] = np.arange(1, n + 1, dtype=float)
    return b


# --------------------------------------------------------------------------- snake math (13 teams, turn edges)

def test_snake_pick_numbers_13_teams_first_middle_last_slot():
    assert DS.my_pick_numbers(1, 13, 4) == [1, 26, 27, 52]
    assert DS.my_pick_numbers(13, 13, 4) == [13, 14, 39, 40]
    assert DS.my_pick_numbers(7, 13, 3) == [7, 20, 33]
    every = sorted(p for s in range(1, 14) for p in DS.my_pick_numbers(s, 13, 13))
    assert every == list(range(1, 13 * 13 + 1))


def test_slot_of_pick_inverts_snake_pick():
    for teams in (13, 14):
        for slot in range(1, teams + 1):
            for p in DS.my_pick_numbers(slot, teams, 13):
                assert DS.slot_of_pick(p, teams) == slot


def test_bad_snake_inputs_raise():
    with pytest.raises(ValueError):
        DS.snake_pick(1, 14, 13)
    with pytest.raises(ValueError):
        DS.snake_pick(0, 1, 13)


def test_slot_from_espn_pick_order():
    order = [16, 27, 2, 5]
    assert DS.slot_from_pick_order(order, 2) == 3
    assert DS.slot_from_pick_order(order, 99) is None
    assert DS.slot_from_pick_order(None, 2) is None
    assert DS.slot_from_pick_order(order, None) is None


def test_load_pick_order_from_league_snapshot(tmp_path):
    d = tmp_path / "processed" / "espn_league"
    d.mkdir(parents=True)
    (d / "42_2027.json").write_text('{"settings": {"draft": {"pick_order": [16, 27, 2]}}}', encoding="utf-8")
    assert DS.load_pick_order(42, "2026-27", tmp_path) == [16, 27, 2]
    assert DS.load_pick_order(43, "2026-27", tmp_path) is None
    assert DS.load_pick_order(None, "2026-27", tmp_path) is None
    (d / "44_2027.json").write_text("not json", encoding="utf-8")
    assert DS.load_pick_order(44, "2026-27", tmp_path) is None


def test_resolve_slot_priority():
    assert DS.resolve_slot(14, manual=5, pick_order=[3, 4], team_id=4, my_pick_nos=[9]) == (5, "entered by hand")
    assert DS.resolve_slot(14, manual=0, pick_order=[3, 4], team_id=4, my_pick_nos=[9])[0] == 2
    assert DS.resolve_slot(14, manual=None, pick_order=[3, 4], team_id=99, my_pick_nos=[20, 9])[0] == 9
    assert DS.resolve_slot(14, my_pick_nos=[20])[0] == 9  # round 2, snake back: slot 9 picks 20th
    assert DS.resolve_slot(14, manual=15)[0] is None


def test_upcoming_turns_and_opponent_pick_counts_at_the_turn():
    mine = DS.my_pick_numbers(13, 13, 13)  # 13, 14, 39, 40, ...
    assert DS.upcoming_turns(5, mine) == (False, [13, 14])
    assert DS.opponent_picks_before(13, 5, mine) == 8
    assert DS.opponent_picks_before(14, 5, mine) == 8  # back-to-back at the turn: no opponent in between
    on, targets = DS.upcoming_turns(13, mine)
    assert on and targets == [14, 39]
    assert DS.opponent_picks_before(14, 13, mine) == 0
    assert DS.opponent_picks_before(39, 13, mine) == 24
    first = DS.my_pick_numbers(1, 13, 13)  # 1, 26, 27, ...
    assert DS.upcoming_turns(1, first) == (True, [26, 27])
    assert DS.opponent_picks_before(26, 1, first) == 24


# --------------------------------------------------------------------------- simulation properties

def test_probabilities_are_monotone_in_adp_and_in_the_number_of_picks():
    m = np.arange(1, 81, dtype=float)
    p = DS.survival_probabilities(m, [10, 25], n_sims=4000, seed=1)
    assert p.shape == (2, 80)
    assert (p[1] <= p[0] + 1e-12).all(), "a later turn can never be more likely than an earlier one"
    for row in p:
        assert (np.diff(row) >= -0.03).all(), "survival should rise with ADP up to Monte Carlo noise"
        assert row[0] < 0.01 and row[-1] > 0.99
    smooth = np.convolve(p[0], np.ones(5) / 5, mode="valid")
    assert (np.diff(smooth) >= -0.005).all()


def test_seed_determinism():
    m = np.arange(1, 60, dtype=float)
    a = DS.survival_probabilities(m, [12], seed=7)
    assert np.array_equal(a, DS.survival_probabilities(m, [12], seed=7))
    assert not np.array_equal(a, DS.survival_probabilities(m, [12], seed=8))


def test_zero_picks_means_certain_and_pool_smaller_than_picks_means_gone():
    m = np.arange(1, 6, dtype=float)
    p = DS.survival_probabilities(m, [0, 5, 9])
    assert (p[0] == 1).all()
    assert (p[1] == 0).all() and (p[2] == 0).all()
    assert DS.survival_probabilities(np.array([]), [3]).shape == (1, 0)


def test_fast_enough_for_a_streamlit_rerun():
    m = np.arange(1, 301, dtype=float)
    t0 = time.perf_counter()
    DS.survival_probabilities(m, [20, 40], n_sims=3000)
    assert time.perf_counter() - t0 < 1.0


# --------------------------------------------------------------------------- market ADP handling

def test_missing_adp_goes_late_in_board_order_and_no_adp_falls_back_to_rank():
    b = _board(6)
    b.loc[[1, 4], "adp"] = np.nan
    m, note = DS.market_adp(b)
    assert m[1] > np.nanmax(b["adp"]) and m[4] > m[1]
    assert "no ADP" in note
    m2, note2 = DS.market_adp(_board(6, with_adp=False))
    assert np.allclose(m2, np.arange(1, 7), atol=1e-3) and "board rank" in note2


def test_adp_ties_are_broken_by_board_rank():
    b = _board(4)
    b["adp"] = 139.9
    m, _ = DS.market_adp(b)
    assert (np.diff(m) > 0).all()


# --------------------------------------------------------------------------- the table

def test_table_mid_draft_and_vorp_at_risk():
    b = _board(60)
    r = DS.availability_table(b, drafted_ids=[100, 101, 102], current_pick=4, slot=13, teams=13, rounds=13)
    assert not r.on_the_clock and r.targets == [13, 14] and r.opponent_picks == [9, 9]
    t = r.table
    assert len(t) == 57 and list(t.columns) == ["player_id", *DS.TABLE_COLUMNS]
    assert t["rank"].tolist() == sorted(t["rank"].tolist())
    assert t.iloc[0]["p_avail_next"] < 0.05 and t.iloc[-1]["p_avail_next"] > 0.95
    exp = (t["vorp"].clip(lower=0) * (1 - t["p_avail_next"])).round(1)
    assert np.allclose(t["vorp_at_risk"], exp)


def test_table_on_the_clock_waits_for_the_following_turn():
    r = DS.availability_table(_board(60), [], current_pick=13, slot=13, teams=13, rounds=13)
    assert r.on_the_clock and r.targets == [14, 39] and r.opponent_picks == [0, 24]
    assert (r.table["p_avail_next"] == 1).all() and (r.table["vorp_at_risk"] == 0).all()


def test_table_last_pick_and_end_of_draft():
    r = DS.availability_table(_board(10), [], current_pick=13 * 13, slot=13, teams=13, rounds=13)
    assert r.on_the_clock and r.targets == [] and r.table["p_avail_next"].isna().all()
    assert any("no picks left" in n for n in r.notes)
    r = DS.availability_table(_board(10), [], current_pick=13 * 13 + 1, slot=1, teams=13, rounds=13)
    assert r.targets == [] and any("draft is over" in n for n in r.notes)
    r = DS.availability_table(_board(10), [], current_pick=160, slot=13, teams=13, rounds=13)  # one turn left
    assert r.targets == [169] and r.table["p_avail_after"].isna().all()


def test_empty_pool_and_board_without_adp():
    b = _board(5)
    r = DS.availability_table(b, b["player_id"], current_pick=6, slot=2, teams=13, rounds=13)
    assert r.table.empty
    r = DS.availability_table(_board(30, with_adp=False), [], current_pick=1, slot=5, teams=13, rounds=13)
    assert r.table["adp"].isna().all() and r.table["p_avail_next"].between(0, 1).all()


def test_advisory_only_board_is_not_changed():
    b = _board(50)
    b["proj_total_fp"] = b["proj_fppg"] * 70
    before = b.copy(deep=True)
    r = DS.availability_table(b, [100, 105], current_pick=3, slot=4, teams=13, rounds=13)
    pd.testing.assert_frame_equal(b, before)
    merged = before.merge(r.table[["player_id", "rank", "vorp"]], on="player_id", suffixes=("", "_sim"))
    assert (merged["rank"] == merged["rank_sim"]).all() and (merged["vorp"] == merged["vorp_sim"]).all()


def test_every_shown_column_has_a_glossary_entry():
    missing = [c for c in DS.TABLE_COLUMNS if c not in G.COLUMN_GLOSSARY]
    assert not missing, missing
