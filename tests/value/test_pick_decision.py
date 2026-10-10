"""Take now or wait (ADR 0044): the two-pick expectation on hand-computable draws, a brute-force check,
edge cases (back-to-back, last pick, not on the clock, no ADP, playoff tiebreak) and advisory-only."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from src.app import glossary as G
from src.value import draft_sim as DS
from src.value import pick_decision as PD


def _avail(values, order, k, *, names=None, positions=None, on_clock=True, targets=(10,), extra=None):
    n = len(values)
    order = np.asarray(order)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(n)[None, :], axis=1)
    table = pd.DataFrame({"player_id": np.arange(n), "rank": np.arange(1, n + 1),
                          "name": names or [f"P{i}" for i in range(n)], "position": positions or ["G"] * n,
                          "vorp": np.asarray(values, float), "adp": np.nan,
                          "p_avail_next": (rank >= k).mean(axis=0) if targets else np.nan})
    for c, v in (extra or {}).items():
        table[c] = v
    return DS.AvailabilityResult(table=table, current_pick=1, on_the_clock=on_clock, targets=list(targets),
                                 opponent_picks=[k] * len(targets), notes=[], order=order)


def test_three_players_two_draws_by_hand():
    # A=100, B=60, C=50; one opponent pick before my next turn.
    # Draw 0 the opponent prefers A, then B, then C; draw 1 prefers B, then A, then C.
    # Take A: d0 opp takes B -> C (50); d1 opp takes B -> C (50).        next_best 50, two_pick 150
    # Take B: d0 opp takes A -> C (50); d1 opp takes A -> C (50).        next_best 50, two_pick 110
    # Take C: d0 opp takes A -> B (60); d1 opp takes B -> A (100).       next_best 80, two_pick 130
    av = _avail([100, 60, 50], [[0, 1, 2], [1, 0, 2]], 1, names=["A", "B", "C"])
    np.testing.assert_allclose(PD.expected_next_best(av.order, np.array([100., 60, 50]), 1), [50, 50, 80])
    d = PD.decide(av)
    t = d.table.set_index("name")
    assert d.status == "ok" and list(d.table["name"]) == ["A", "C", "B"]
    assert t.loc["A", "two_pick"] == 150 and t.loc["C", "two_pick"] == 130 and t.loc["B", "two_pick"] == 110
    assert t.loc["A", "wait_cost"] == 20 and t.loc["C", "wait_cost"] == -20 and t.loc["B", "wait_cost"] == -40
    assert t.loc["A", "p_avail_next"] == 0.5
    assert d.table["reason"].iloc[0].startswith("Take now: 50% gone")


def _brute_next_best(order, values, k, c):
    out = []
    for row in order:
        rest = [j for j in row if j != c]
        left = rest[k:]
        out.append(max((values[j] for j in left), default=0.0))
    return float(np.mean(out))


@pytest.mark.parametrize("k", [0, 1, 3, 7, 9, 12])
def test_vectorised_expectation_matches_brute_force(k):
    rng = np.random.default_rng(3)
    n = 10
    values = rng.normal(50, 30, n)
    order = np.vstack([rng.permutation(n) for _ in range(40)])
    got = PD.expected_next_best(order, values, k)
    want = [_brute_next_best(order, values, k, c) for c in range(n)]
    np.testing.assert_allclose(got, want)
    pos = np.array(["G", "F"] * 5)
    for g in ("G", "F"):
        m = pos == g
        got_g = PD.expected_next_best(order, values, k, mask=m)
        for c in range(n):
            per = []
            for row in order:
                left = [j for j in row if j != c][k:]
                cand = [values[j] for j in left if m[j]]
                if cand:
                    per.append(max(cand))
            if per:
                assert got_g[c] == pytest.approx(np.mean(per))
            else:
                assert np.isnan(got_g[c])


def test_back_to_back_at_the_turn_waiting_costs_nothing():
    av = _avail([100, 60, 50], [[0, 1, 2], [2, 1, 0]], 0, names=["A", "B", "C"])
    d = PD.decide(av)
    t = d.table.set_index("name")
    assert d.status == "back_to_back" and "costs nothing" in d.message
    assert t.loc["A", "wait_cost"] == 0 and t.loc["B", "wait_cost"] == 0
    assert t.loc["C", "wait_cost"] == -10
    assert set(d.table["name"].iloc[:2]) == {"A", "B"}


def test_playoff_fp_breaks_ties_only_and_absence_changes_nothing():
    order = [[0, 1, 2], [2, 1, 0]]
    plain = PD.decide(_avail([100, 60, 50], order, 0, names=["A", "B", "C"]))
    assert "playoff_fp" not in plain.table.columns and plain.table["name"].iloc[0] == "A"  # tie -> board rank
    tb = PD.decide(_avail([100, 60, 50], order, 0, names=["A", "B", "C"]), tiebreak=np.array([10.0, 30.0, 999.0]))
    assert list(tb.table["name"]) == ["B", "A", "C"]  # B wins the exact tie; C's huge playoff_fp does not lift it
    pd.testing.assert_frame_equal(tb.table.drop(columns="playoff_fp").sort_values("name").reset_index(drop=True)
                                  .drop(columns="reason"),
                                  plain.table.sort_values("name").reset_index(drop=True).drop(columns="reason"))
    # off a tie, a large playoff_fp never overturns the objective
    d = PD.decide(_avail([100, 60, 50], [[0, 1, 2], [1, 0, 2]], 1, names=["A", "B", "C"]),
                  tiebreak=np.array([0.0, 0.0, 1e9]))
    assert d.table["name"].iloc[0] == "A"


def test_last_pick_and_not_on_clock_and_empty():
    last = PD.decide(_avail([10, 30, 20], [[0, 1, 2]], 0, targets=()))
    assert last.status == "last_pick" and last.table["name"].iloc[0] == "P1"
    assert last.table["wait_cost"].iloc[0] == 10 and "last pick" in last.message.lower()
    off = PD.decide(_avail([10, 30, 20], [[0, 1, 2]], 2, on_clock=False))
    assert off.status == "not_on_clock" and off.table.empty
    empty = DS.AvailabilityResult(table=pd.DataFrame(columns=["player_id", "vorp"]), current_pick=1,
                                  on_the_clock=True, targets=[5], opponent_picks=[3])
    assert PD.decide(empty).status == "empty"


def test_need_values_replace_vorp():
    av = _avail([100, 60, 50], [[0, 1, 2], [1, 0, 2]], 1, names=["A", "B", "C"])
    d = PD.decide(av, values=np.array([100.0, 160.0, 50.0]), value_col="need_adj_vorp")
    assert d.table["name"].iloc[0] == "B" and d.value_col == "need_adj_vorp"


def _board(n=60, *, with_adp=True):
    b = pd.DataFrame({"rank": np.arange(1, n + 1), "player_id": np.arange(100, 100 + n),
                      "name": [f"P{i}" for i in range(n)], "position": np.array(["G", "F", "C", "G-F"])[np.arange(n) % 4],
                      "vorp": np.linspace(900, -50, n), "tier": 1})
    if with_adp:
        b["adp"] = np.arange(1, n + 1, dtype=float)
    return b


@pytest.mark.parametrize("with_adp", [True, False])
def test_end_to_end_on_the_simulator_never_touches_the_board(with_adp):
    board = _board(with_adp=with_adp)
    before = board.copy(deep=True)
    teams, rounds, slot = 14, 13, 5
    res = DS.availability_table(board, [], current_pick=5, slot=slot, teams=teams, rounds=rounds)
    d = PD.decide(res, alt_orders=PD.band_orders(res))
    assert d.status == "ok" and len(d.table) == PD.TOP_N
    assert d.opponent_picks == DS.opponent_picks_before(d.next_pick, 5, DS.my_pick_numbers(slot, teams, rounds))
    assert set(d.band) == set(PD.BAND_LABELS)
    assert PD.band_orders(res)["high noise"] is res.band_orders[-1]  # the sim's own draws, not re-simulated
    assert (d.table["wait_cost"].iloc[1:] <= 0).all() and d.table["wait_cost"].iloc[0] >= 0
    # p_avail_next is the simulator's own number for the same draws
    sim = res.table.set_index("player_id")["p_avail_next"]
    assert np.allclose(d.table.set_index("player_id")["p_avail_next"], sim.loc[d.table["player_id"]])
    pd.testing.assert_frame_equal(board, before)


def test_band_flip_flag_when_the_noise_changes_the_pick():
    # A=100, B=95, C=10, D=0, one opponent pick. Main draws: the opponent prefers B, then C, so taking B now
    # still leaves A (195) while taking A leaves only C (110): take B. Alternative draws prefer A, then B:
    # taking A leaves C (110), taking B leaves C (105): take A. The recommendation flips.
    av = _avail([100, 95, 10, 0], [[1, 2, 0, 3]] * 4, 1, names=["A", "B", "C", "D"])
    assert PD.decide(av).table["name"].iloc[0] == "B"
    stay = PD.decide(av, alt_orders={"low noise": av.order})
    assert not stay.band_flip and "same pick" in stay.message
    flip = PD.decide(av, alt_orders={"high noise": np.array([[0, 1, 2, 3]] * 4)})
    assert flip.band_flip and flip.band["high noise"] == "A" and "Sensitive" in flip.message


def test_refactored_simulator_reproduces_survival_and_order_is_kept():
    m = np.arange(1, 50, dtype=float)
    order = DS.simulate_order(m, n_sims=500, seed=4)
    np.testing.assert_array_equal(DS.survival_from_order(order, [5, 12]),
                                  DS.survival_probabilities(m, [5, 12], n_sims=500, seed=4))
    res = DS.availability_table(_board(), [], current_pick=1, slot=1, teams=14, rounds=13)
    assert res.order.shape == (DS.DEFAULT_SIMS, len(res.table))


def test_fast_enough_for_a_streamlit_rerun():
    board = _board(700)
    t0 = time.perf_counter()
    res = DS.availability_table(board, [], current_pick=20, slot=6, teams=14, rounds=13)
    PD.decide(res, alt_orders=PD.band_orders(res))
    assert time.perf_counter() - t0 < 1.5


def test_glossary_covers_the_panel_columns():
    for c in PD.DECISION_COLUMNS:
        assert G.tooltip(c), c
