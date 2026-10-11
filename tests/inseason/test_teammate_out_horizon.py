"""ADR 0042: the teammate-out boost covers one (the next) game, even when a player has 3+ games left in the week."""
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ise_testkit import CFG13, SEASON, date_after_games, make_tables

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ops"))
from src.contracts import HISTORY_TABLES, History  # noqa: E402
from src.inseason import context as C  # noqa: E402
from src.inseason import waivers as W  # noqa: E402
from src.models.registry import get_projector  # noqa: E402
from src.ops import nightly_analysis as na  # noqa: E402

BOOST = 7.0


@pytest.fixture(scope="module")
def setup():
    tables = make_tables()
    prior = get_projector("baseline").project(History.until({k: tables[k] for k in HISTORY_TABLES}, SEASON))
    return tables, prior, date_after_games(tables, 22)


def _flat_boost(monkeypatch, module):
    seen = {}

    def fake(gl, tg, injuries, positions, ros, *, as_of, beta=None):
        seen["called"] = True
        return pd.Series(BOOST, index=ros["player_id"].drop_duplicates().to_numpy())
    monkeypatch.setattr(module, "status_boost", fake)
    return seen


def _fresh(ctx, ids):
    s = pd.Series("OUT", index=list(ids))
    s.attrs["fetched_at"] = ctx.as_of + pd.Timedelta(hours=30)
    return s


def test_waiver_week_total_adds_exactly_one_game_of_boost(setup, monkeypatch):
    tables, prior, as_of = setup
    ctx = C.load_context(SEASON, as_of, synthetic=True, tables=tables, prior=prior, cfg=CFG13)
    ctx.week_anchor = pd.Timestamp(as_of + timedelta(days=1))          # the next slate, as the nightly job anchors it
    ctx.injuries = _fresh(ctx, [])
    seen = _flat_boost(monkeypatch, W)
    from tests.inseason.test_waivers_context import _mine  # noqa: E402
    s = W.find_waivers(ctx, _mine(ctx), top=60, candidates=80).streams
    multi = (s["week_team_games"] >= 3) & (s["week_exp_games"] > 1.5)
    assert seen and multi.any(), "need a streamer with 3+ team games and well over one expected game"
    np.testing.assert_allclose(s["week_exp_fp"], s["week_exp_games"] * s["ros_fppg"]
                               + np.minimum(s["week_exp_games"], 1.0) * BOOST, rtol=1e-9)


def test_nightly_week_total_adds_exactly_one_game_of_boost(setup, monkeypatch):
    tables, prior, through = setup
    from tests.ops.test_ops_nightly_analysis import league_ctx  # noqa: E402
    ctx, draft = league_ctx(tables, prior, through)
    ctx.injuries = _fresh(ctx, [])
    seen = _flat_boost(monkeypatch, na)
    choice = C.resolve_roster(ctx, team_id=3)
    teams = set(ctx.ros["team_id"].dropna().astype(int))
    table, _ = na.make_lineup(ctx, choice, (through + timedelta(days=1)).date(), teams, None)
    avail = ctx.ros.drop_duplicates("player_id").set_index("player_id")["avail"].reindex(table["player_id"]).to_numpy()
    g = table["games_left_week"].to_numpy() * avail
    assert seen and (table["games_left_week"] >= 3).any(), "need a rostered player with 3+ games left"
    assert (table["out_boost_fppg"] == BOOST).all()
    np.testing.assert_allclose(table["week_exp_fp"], g * table["ros_fppg"] + np.minimum(g, 1.0) * BOOST, rtol=1e-9)
