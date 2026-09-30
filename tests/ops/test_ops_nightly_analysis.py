"""The nightly artifacts on the synthetic league: waivers, alerts, week tables, lineup hints, isolation, leakage, week anchor."""
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inseason"))
from ise_testkit import CFG13, SEASON, date_after_games, make_tables  # noqa: E402

from src.contracts import HISTORY_TABLES, History  # noqa: E402
from src.inseason import context as C  # noqa: E402
from src.models.registry import get_projector  # noqa: E402
from src.ops import nightly_analysis as na  # noqa: E402
from src.value.replacement import league_shape  # noqa: E402


@pytest.fixture(scope="module")
def tables():
    return make_tables()


@pytest.fixture(scope="module")
def prior(tables):
    return get_projector("baseline").project(History.until({k: tables[k] for k in HISTORY_TABLES}, SEASON))


@pytest.fixture(scope="module")
def through(tables):
    return date_after_games(tables, 22)


def league_ctx(tables, prior, through, slate=None):
    """A synthetic context that also carries a 'synced league': every team's mock-draft roster, ESPN ids = 1000 + player id."""
    ctx = C.load_context(SEASON, through, synthetic=True, tables=tables, prior=prior, cfg=CFG13)
    ctx.week_anchor = pd.Timestamp(slate if slate is not None else through + timedelta(days=1))
    shape = league_shape(ctx.cfg)
    pool = ctx.ros.drop_duplicates("player_id").set_index("player_id")["ros_total_fp"]
    pool = pool[pool.index.isin(C.active_universe(ctx))]
    draft = C.mock_snake_draft(pool, shape.teams, shape.roster_size)
    ctx.league = {"teams": [{"team_id": t + 1, "name": f"Team {t + 1}", "roster": [{"espn_player_id": 1000 + p, "lineup_slot": "BE"} for p in ids]}
                            for t, ids in draft.items()], "free_agents": []}
    ctx.id_map = {str(1000 + p): int(p) for p in ctx.ros["player_id"]}
    return ctx, draft


def test_full_artifact_set_for_a_synced_league(tables, prior, through):
    ctx, draft = league_ctx(tables, prior, through)
    slate = (through + timedelta(days=1)).date()
    art = na.build_artifacts(ctx, slate, team_id=3, top=10)
    assert art.errors == {} and art.meta["team_label"].startswith("Team 3")
    for name in ("ros", "waivers_adds", "waivers_streams", "waivers_streams_next", "alerts_rising", "alerts_beneficiaries", "week_this_week", "lineup"):
        assert name in art.frames, name
    mine = set(draft[2])
    adds = art.frames["waivers_adds"]
    assert len(adds) == 10 and not set(adds["player_id"]) & mine                      # never recommends someone already on a roster
    rostered = {p for ids in draft.values() for p in ids}
    assert not set(adds["player_id"]) & rostered
    assert set(adds["drop"].dropna()) <= set(ctx.ros["name"])                          # drops are real players of yours
    lineup = art.frames["lineup"]
    assert set(lineup["player_id"]) == mine and lineup["slot_this_week"].isin(
        ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL#1", "UTIL#2", "UTIL#3", "BENCH", "IR"]).all()
    assert (lineup["slot_this_week"].isin(["BENCH", "IR"]) | (lineup["week_exp_fp"] >= 0)).all()
    assert art.state["my_roster"] == sorted(mine) and "ros_rank" in art.state and len(art.state["ros_rank"]) == na.STATE_ROS_RANKS
    wk = art.meta["week"]
    assert wk["this_week"]["week"] >= 1 and wk["calendar_source"] == "derived" and wk["slate_games"] >= 0
    assert art.frames["ros"]["rank"].tolist() == list(range(1, len(art.frames["ros"]) + 1))
    assert art.frames["week_this_week"]["mine"].any()


def test_no_team_configured_skips_team_artifacts_but_keeps_league_wide_ones(tables, prior, through):
    ctx, _ = league_ctx(tables, prior, through)
    art = na.build_artifacts(ctx, (through + timedelta(days=1)).date(), team_id=None)
    assert art.errors == {} and "no team id configured" in art.meta["team_skipped"]
    assert "waivers_adds" not in art.frames and "lineup" not in art.frames
    assert {"ros", "alerts_rising", "week_this_week", "free_agents_top"} <= set(art.frames) and art.state["my_roster"] is None
    ctx2 = C.load_context(SEASON, through, synthetic=True, tables=tables, prior=prior, cfg=CFG13)            # not synced at all
    art2 = na.build_artifacts(ctx2, (through + timedelta(days=1)).date(), team_id=3)
    assert "no synced league" in art2.meta["team_skipped"] and any("no league rosters" in n for n in art2.notes)


def test_a_team_id_before_the_draft_is_a_stated_reason_not_a_crash(tables, prior, through):
    ctx, _ = league_ctx(tables, prior, through)
    for t in ctx.league["teams"]:
        t["roster"] = []
    art = na.build_artifacts(ctx, (through + timedelta(days=1)).date(), team_id=3)
    assert "draft has not happened" in art.meta["team_skipped"] and art.errors == {} and "ros" in art.frames


def test_one_failing_artifact_does_not_stop_the_others(tables, prior, through, monkeypatch):
    ctx, _ = league_ctx(tables, prior, through)
    monkeypatch.setattr(na, "make_alerts", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("alerts blew up")))
    monkeypatch.setattr(na, "make_lineup", lambda *a, **k: (_ for _ in ()).throw(ValueError("lineup blew up")))
    art = na.build_artifacts(ctx, (through + timedelta(days=1)).date(), team_id=3)
    assert set(art.errors) == {"alerts", "lineup"} and "alerts blew up" in art.errors["alerts"]
    assert {"ros", "waivers_adds", "week_this_week"} <= set(art.frames)


def test_the_artifacts_do_not_depend_on_games_after_the_completed_day(tables, prior, through):
    """Leakage guard: rebuild the context from tables with every later game corrupted; outputs must be identical."""
    ctx, _ = league_ctx(tables, prior, through)
    slate = (through + timedelta(days=1)).date()
    a = na.build_artifacts(ctx, slate, team_id=3, top=8)
    bad = {k: v.copy() for k, v in tables.items()}
    late = bad["game_logs"]["game_date"] > through
    assert late.any()
    for col in ("pts", "reb", "ast", "min"):
        bad["game_logs"].loc[late, col] = bad["game_logs"].loc[late, col] * 3 + 5
    ctx2, _ = league_ctx(bad, prior, through)
    b = na.build_artifacts(ctx2, slate, team_id=3, top=8)
    for name in ("ros", "waivers_adds", "alerts_rising", "alerts_beneficiaries", "lineup"):
        pd.testing.assert_frame_equal(a.frames[name].reset_index(drop=True), b.frames[name].reset_index(drop=True))
    assert a.state == b.state


def test_the_week_follows_the_next_slate_not_the_last_completed_day(tables, prior, through):
    """On the day a new matchup week starts, the last completed day is still in the old week; the recommendations must be for the new one."""
    ctx, _ = league_ctx(tables, prior, through)
    cal = ctx.calendar
    old = cal.weeks[3]
    ctx.as_of = pd.Timestamp(old.end)                      # everything up to the old week's last day is done
    ctx.week_anchor = pd.Timestamp(old.end) + pd.Timedelta(days=1)
    from src.inseason.waivers import pick_week

    assert pick_week(ctx).week == old.week + 1
    ctx.week_anchor = None
    assert pick_week(ctx).week == old.week                   # the default (as_of) is unchanged for the ADR 0015 tools
    ctx.week_anchor = pd.Timestamp(old.end) + pd.Timedelta(days=1)
    assert pick_week(ctx, next_week=True).week == old.week + 2


def test_rising_minutes_and_beneficiaries_are_tagged_by_ownership(tables, prior, through):
    ctx, draft = league_ctx(tables, prior, through)
    own = na.ownership(type("C", (), {"my_ids": draft[2]})(), ctx)
    fa = set(C.free_agent_ids(ctx, set(own)))
    frames = na.make_alerts(ctx, own, fa, top=20)
    seen = set()
    for key in ("rising", "beneficiaries", "absent"):
        f = frames[key]
        if len(f):
            seen |= set(f["status"])
            assert set(f["status"]) <= {"mine", "other team", "free agent", "?"}
    assert seen, "the synthetic league should produce at least one flag"
    mine_rows = pd.concat([f[f["status"] == "mine"] for f in frames.values() if len(f)])
    assert set(mine_rows["player_id"]) <= set(draft[2])


def test_lineup_hints_use_the_next_slate_and_flag_out_players(tables, prior, through):
    ctx, draft = league_ctx(tables, prior, through)
    slate = (through + timedelta(days=1)).date()
    star = max(draft[2], key=lambda p: float(ctx.ros.drop_duplicates("player_id").set_index("player_id").at[p, "ros_total_fp"]))
    ctx.injuries = pd.Series({star: "OUT"}, dtype=object)
    choice = C.resolve_roster(ctx, team_id=3)
    team_ids = set(ctx.ros["team_id"].dropna().astype(int))
    playing = set(list(team_ids)[: len(team_ids) // 2])
    table, meta = na.make_lineup(ctx, choice, slate, playing, None)
    row = table[table["player_id"] == star].iloc[0]
    assert "OUT" in row["hint"] and meta["slate"] == slate.isoformat()
    assert table[table["player_id"] == star]["plays_today"].iloc[0] == (int(ctx.ros.drop_duplicates("player_id").set_index("player_id").at[star, "team_id"]) in playing)
    none_play = na.make_lineup(ctx, choice, slate, set(), None)[1]
    assert none_play["today_players"] == 0 and none_play["today_expected_fp"] == 0.0
    assert np.isfinite(meta["week_expected_fp"])
