"""Slot-aware roster value and the trade analyzer, on hand-built rest-of-season frames with known answers."""
import pandas as pd
import pytest

from src.inseason.context import InSeasonContext, RosterChoice
from src.inseason.lineup import (
    NameError_, fit_to_capacity, lineup_value, normalize_name, resolve_name, resolve_names, roster_frame,
)
from src.inseason.trade import (
    CONFIDENCE_BUCKETS, analyze, confidence_bucket, confidence_label, describe, evaluate_trade, replacement_value,
)
from src.value.league import load_league
from src.value.replacement import league_shape

CFG = load_league()
SHAPE = league_shape(CFG)          # 13 teams, PG SG SF PF C G F UTIL x3, 3 bench


def frame(rows):
    """rows: (id, name, position, ros_total) -> a ROS-like frame."""
    df = pd.DataFrame(rows, columns=["player_id", "name", "position", "ros_total_fp"])
    df["ros_fppg"] = df["ros_total_fp"] / 40.0
    df["ros_games"] = 40.0
    df["avail"] = 1.0
    df["team_id"] = 1610612737
    df["team_games_left"] = 40.0
    df["gp"] = 10
    return df


# ten starters (one per slot type) + three bench, values descending with id
BASE = [
    (1, "Guard One", "PG", 1000), (2, "Guard Two", "SG", 950), (3, "Wing Three", "SF", 900),
    (4, "Big Four", "PF", 850), (5, "Center Five", "C", 800), (6, "Guard Six", "PG", 750),
    (7, "Wing Seven", "SF", 700), (8, "Util Eight", "SG", 650), (9, "Util Nine", "PF", 600),
    (10, "Util Ten", "C", 550), (11, "Bench Eleven", "PG", 300), (12, "Bench Twelve", "SG", 250),
    (13, "Bench Thirteen", "SF", 200),
]


def idx_of(rows):
    return frame(rows).set_index("player_id")


def test_full_lineup_value_is_starters_plus_weighted_bench():
    idx = idx_of(BASE)
    res = lineup_value(roster_frame(idx, range(1, 14)), SHAPE, 0.4)
    assert res.starter_value == sum(v for _, _, _, v in BASE[:10])
    assert res.bench_value == pytest.approx(0.4 * (300 + 250 + 200))
    assert res.value == pytest.approx(res.starter_value + res.bench_value)
    assert not res.empty_slots and sorted(res.bench) == [11, 12, 13] and res.excess == []
    assert set(v for v in res.starters.values()) == set(range(1, 11))


def test_positional_eligibility_leaves_a_slot_empty_and_a_center_fills_it():
    rows = [(i, f"G{i}", "PG", 1000 - 10 * i) for i in range(1, 8)] + [(20, "Only Center", "C", 400)]
    idx = idx_of(rows)
    no_c = lineup_value(roster_frame(idx, range(1, 8)), SHAPE, 0.0)
    assert "C" in no_c.empty_slots and "F" in no_c.empty_slots and "SF" in no_c.empty_slots
    with_c = lineup_value(roster_frame(idx, list(range(1, 8)) + [20]), SHAPE, 0.0)
    assert with_c.starters["C"] == 20 and with_c.value == pytest.approx(no_c.value + 400)


def test_matching_is_globally_optimal_not_greedy():
    # A is PG/SG eligible ('G'), B only a PG. Best: B -> PG, A -> SG (both start, guard slots G and UTIL free too).
    rows = [(1, "A", "G", 900), (2, "B", "PG", 800)]
    res = lineup_value(roster_frame(idx_of(rows), [1, 2]), SHAPE, 0.0)
    assert res.value == 1700 and all(res.starters[s] is not None for s in ("PG", "SG"))


def test_ir_player_uses_no_roster_spot_and_is_valued_like_the_bench():
    idx = idx_of(BASE + [(14, "Injured Star", "PF", 900)])
    ids = list(range(1, 15))
    res = lineup_value(roster_frame(idx, ids), SHAPE, 0.5, ir_ids=[14])
    assert res.ir == [14] and 14 not in res.bench
    base = lineup_value(roster_frame(idx, range(1, 14)), SHAPE, 0.5)
    assert res.value == pytest.approx(base.value + 0.5 * 900)     # the IR player adds exactly w * his ROS total
    fitted, dropped, added = fit_to_capacity(roster_frame(idx, ids), SHAPE, 0.5, None, 500.0, ir_ids=[14])
    assert dropped == [] and added == [] and len(fitted) == 14    # 13 active + 1 IR is a legal roster


def test_fit_to_capacity_drops_the_least_valuable_and_fills_with_the_best_free_agent():
    idx = idx_of(BASE + [(14, "Extra", "SF", 100)])
    fitted, dropped, added = fit_to_capacity(roster_frame(idx, range(1, 15)), SHAPE, 0.4, None, 500.0)
    assert dropped == [14] and len(fitted) == 13
    short = roster_frame(idx, range(1, 13))
    pool = frame([(30, "Good FA", "SF", 400), (31, "Better FA", "SF", 420), (32, "Poor FA", "PG", 50)]).set_index("player_id")
    fitted, dropped, added = fit_to_capacity(short, SHAPE, 0.4, roster_frame(pool, [30, 31, 32]), 100.0)
    assert added == [31] and dropped == [] and len(fitted) == 13
    fitted, _, added = fit_to_capacity(short, SHAPE, 0.4, None, 100.0)
    assert added == [-1]                                            # generic replacement-level player


def test_one_for_one_upgrade_is_worth_exactly_the_difference():
    rows = BASE + [(40, "Star PG", "PG", 1200)]
    idx = idx_of(rows).reset_index()
    res = evaluate_trade(idx, range(1, 14), [1], [40], SHAPE, 0.0, repl_value=100.0)
    assert res.delta == pytest.approx(200.0) and res.dropped == [] and res.added == []
    assert res.verdict == "favours you" and res.incoming_ros == 1200 and res.outgoing_ros == 1000


def test_two_for_one_fills_the_freed_spot_with_the_best_free_agent_not_a_phantom_zero():
    rows = BASE + [(40, "Star PF", "PF", 1500), (50, "FA Guard", "PG", 500), (51, "FA Wing", "SF", 450)]
    idx = idx_of(rows).reset_index()
    # give the two bench-adjacent util guys (650 + 600), get one 1500 star; the freed spot goes to a 500 FA
    res = evaluate_trade(idx, range(1, 14), [8, 9], [40], SHAPE, 0.0, pool_ids=[50, 51], repl_value=100.0)
    assert res.added == [50]                                        # the better of the two free agents
    assert res.delta == pytest.approx(1500 + 500 - 650 - 600)
    # with no free-agent pool the filler is a replacement-level player of the stated value
    res2 = evaluate_trade(idx, range(1, 14), [8, 9], [40], SHAPE, 0.0, pool_ids=None, repl_value=300.0)
    assert res2.added == [-1]
    assert res2.delta == pytest.approx(1500 + 300 - 650 - 600)


def test_one_for_two_overfills_the_roster_and_names_the_forced_drop():
    rows = BASE + [(40, "New A", "SF", 620), (41, "New B", "PG", 610)]
    idx = idx_of(rows).reset_index()
    res = evaluate_trade(idx, range(1, 14), [10], [40, 41], SHAPE, 0.4, repl_value=100.0)
    assert res.dropped == [13]                                     # least valuable player on the new roster
    assert res.after.value > 0


def test_trading_away_your_only_center_costs_the_slot():
    rows = BASE[:4] + [(5, "Center Five", "C", 800)] + BASE[5:10] + [(41, "Extra Guard", "PG", 820)]
    idx = idx_of(rows).reset_index()
    res = evaluate_trade(idx, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [5], [41], SHAPE, 0.0, repl_value=0.0)
    # the C-eligible pool is just player 10 (550), who moves to C; the 820 guard takes a UTIL slot
    assert res.delta == pytest.approx(820 - 800)


def test_invalid_trades_raise():
    idx = idx_of(BASE + [(40, "X", "PF", 500)]).reset_index()
    with pytest.raises(ValueError, match="not on the roster"):
        evaluate_trade(idx, range(1, 14), [40], [], SHAPE, 0.4)
    with pytest.raises(ValueError, match="already on the roster"):
        evaluate_trade(idx, range(1, 14), [1], [2], SHAPE, 0.4)
    with pytest.raises(ValueError, match="at least one"):
        evaluate_trade(idx, range(1, 14), [], [], SHAPE, 0.4)
    with pytest.raises(KeyError):
        roster_frame(idx.set_index("player_id"), [999])


def test_verdict_is_even_inside_the_tolerance():
    rows = BASE + [(40, "Same PG", "PG", 1001)]
    idx = idx_of(rows).reset_index()
    res = evaluate_trade(idx, range(1, 14), [1], [40], SHAPE, 0.4, repl_value=100.0)
    assert res.verdict == "roughly even" and res.delta == pytest.approx(1.0)


# ---------------------------------------------------------------- confidence bucketing (ADR 0027)

def test_confidence_bucket_is_roughly_even_inside_the_tolerance_band():
    assert confidence_bucket(0.0, 100.0) == "roughly_even"
    assert confidence_bucket(100.0, 100.0) == "roughly_even"          # exactly at the edge counts as inside
    assert confidence_bucket(-99.0, 100.0) == "roughly_even"


def test_confidence_bucket_leans_between_one_and_three_times_tolerance():
    assert confidence_bucket(150.0, 100.0) == "lean_your_way"
    assert confidence_bucket(-150.0, 100.0) == "lean_other_way"
    assert confidence_bucket(300.0, 100.0) == "lean_your_way"         # exactly at the clear_mult edge is still a lean


def test_confidence_bucket_is_clear_beyond_the_clear_multiple():
    assert confidence_bucket(301.0, 100.0) == "clear_win"
    assert confidence_bucket(-301.0, 100.0) == "clear_loss"
    assert confidence_bucket(1000.0, 100.0, clear_mult=2.0) == "clear_win"  # a custom multiplier is honoured


def test_confidence_bucket_is_nan_safe_and_never_raises():
    assert confidence_bucket(float("nan"), 100.0) == "roughly_even"    # non-finite delta: no signal
    assert confidence_bucket(float("inf"), 100.0) == "roughly_even"
    assert confidence_bucket(50.0, float("nan")) == "clear_win"        # no usable band: sign-only
    assert confidence_bucket(-50.0, 0.0) == "clear_loss"
    assert confidence_bucket(0.0, 0.0) == "roughly_even"


def test_confidence_label_covers_every_bucket_key():
    for key in CONFIDENCE_BUCKETS:
        label = confidence_label(key)
        assert label and label != key and "_" not in label  # every real key maps to readable, space-separated prose
    assert confidence_label("not_a_real_bucket") == "not_a_real_bucket"  # falls back rather than raising


def test_replacement_value_prefers_the_engine_value_when_attached():
    idx = idx_of(BASE).reset_index()
    idx.attrs["replacement"] = {"total": 123.0}
    assert replacement_value(idx, SHAPE) == 123.0
    assert replacement_value(frame(BASE), SHAPE) >= 0


# ---------------------------------------------------------------- names, analyze(), CLI

def test_name_resolution_is_accent_insensitive_and_reports_ambiguity():
    ros = frame([(1, "Nikola Jokić", "C", 1), (2, "Nikola Vučević", "C", 1), (3, "Luka Dončić", "G", 1),
                 (4, "Jalen Williams", "G", 1), (5, "Jaylin Williams", "F", 1)])
    assert normalize_name("Dončić") == "doncic"
    assert resolve_name(ros, "doncic") == 3 and resolve_name(ros, "Nikola Jokic") == 1
    assert resolve_names(ros, "Luka Doncic, jokic") == [3, 1]
    with pytest.raises(NameError_, match="ambiguous"):
        resolve_name(ros, "nikola")
    with pytest.raises(NameError_, match="no player"):
        resolve_name(ros, "Nobody Here")
    assert resolve_name(ros, "Jalen Williams") == 4                 # exact beats substring


def _ctx(rows):
    ros = frame(rows)
    ros.attrs["replacement"] = {"total": 200.0}
    return InSeasonContext(season="2026-27", as_of=pd.Timestamp("2026-12-01"), cfg=CFG, tables={}, ros=ros)


def test_analyze_and_describe_use_the_free_agent_pool_and_the_partner_side():
    rows = BASE + [(40, "Star PF", "PF", 1500), (50, "FA Guard", "PG", 500), (60, "Partner Scrub", "PG", 100)]
    ctx = _ctx(rows)
    ctx.ros["gp"] = 10
    mine = list(range(1, 14))
    theirs = [40, 60]
    choice = RosterChoice(mine, [], set(mine) | set(theirs), "me", {"other": theirs})
    out = analyze(ctx, choice, [8, 9], [40], partner="other")
    assert out["mine"].delta > 0 and out["partner"].delta < 0        # they lose the star for two mid players
    assert out["mine"].added == [50]                                 # the only free agent fills the freed spot
    text = describe(ctx, out["mine"], "You")
    assert "Star PF" in text and "FA Guard" in text
    with pytest.raises(Exception, match="unknown partner"):
        analyze(ctx, choice, [8], [40], partner="nobody")
