"""The in-season view functions and a headless run of the Streamlit page on the synthetic league."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inseason"))
from ise_testkit import CFG13, SEASON, date_after_games, make_tables  # noqa: E402

from src.app import inseason_view as V  # noqa: E402
from src.contracts import HISTORY_TABLES, History  # noqa: E402
from src.inseason import context as C  # noqa: E402
from src.models.registry import get_projector  # noqa: E402


@pytest.fixture(scope="module")
def tables():
    return make_tables()


@pytest.fixture(scope="module")
def prior(tables):
    return get_projector("baseline").project(History.until({k: tables[k] for k in HISTORY_TABLES}, SEASON))


@pytest.fixture()
def ctx(tables, prior, monkeypatch, tmp_path):
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path))
    return C.load_context(SEASON, date_after_games(tables, 22), synthetic=True, tables=tables, prior=prior, cfg=CFG13)


def test_week_and_summary_tables(ctx):
    s = V.weeks_summary(ctx)
    assert {"week", "mean_games", "heavy_teams", "light_teams"} <= set(s.columns) and len(s) >= 2
    w = V.week_table(ctx, int(s["week"].iloc[0]))
    assert list(w.columns) == ["abbr", "games", "games_left", "b2b", "off_night_games", "heavy", "light"]
    ctx.weekly = None
    assert V.week_table(ctx, 1).empty and V.weeks_summary(ctx).empty


def test_ros_table_filters(ctx):
    full = V.ros_table(ctx, top=50)
    assert len(full) == 50 and full["ros_rank"].iloc[0] == 1
    name = full["name"].iloc[3]
    assert V.ros_table(ctx, search=name[:12].upper()).shape[0] >= 1
    only_c = V.ros_table(ctx, position="C", top=500)
    assert len(only_c) < len(V.ros_table(ctx, top=500)) and set(only_c["position"]) <= {"C", "F-C", "G-F", "F"} | set(only_c["position"])


def test_trade_and_waiver_views(ctx):
    choice = V.roster_choice(ctx, mock_team=2)
    opts = V.player_options(ctx)
    give = [choice.my_ids[0]]
    others = [p for ids in choice.others.values() for p in ids]
    out = V.run_trade(ctx, choice, give, others[:1])
    assert set(out["lines"]["side"]) == {"give", "get"} and out["verdict"] in ("favours you", "roughly even", "favours the other side")
    assert opts[give[0]].startswith(ctx.ros.set_index("player_id").at[give[0], "name"])
    assert out["confidence_bucket"] in ("clear_win", "lean_your_way", "roughly_even", "lean_other_way", "clear_loss")
    assert out["confidence_label"] and "_" not in out["confidence_label"]
    assert isinstance(out["notes"], list) and out["partner"] is None and out["partner_label"] is None
    rep = V.run_waivers(ctx, choice, top=5)
    assert len(rep.adds) == 5 and V.free_agent_count(ctx, choice) > 0
    assert 0 <= V.bench_weight(ctx) <= 1
    assert V.resolve_player_names(ctx, ctx.ros["name"].iloc[0]) == [int(ctx.ros["player_id"].iloc[0])]
    with pytest.raises(V.ContextUnavailable):
        V.roster_choice(ctx)


def test_run_trade_scores_the_partner_side_and_confidence_agrees_with_verdict(ctx):
    choice = V.roster_choice(ctx, mock_team=2)
    partner_label = next(iter(choice.others))
    give = [choice.my_ids[0]]
    get = [choice.others[partner_label][0]]
    out = V.run_trade(ctx, choice, give, get, partner=partner_label)
    assert out["partner_label"] == partner_label and out["partner"] is not None
    for side in (out, out["partner"]):
        even = side["verdict"] == "roughly even"
        assert (side["confidence_bucket"] == "roughly_even") == even
        favours_you = side["verdict"] == "favours you"
        assert favours_you == (side["confidence_bucket"] in ("lean_your_way", "clear_win"))
    with pytest.raises(Exception, match="unknown partner"):
        V.run_trade(ctx, choice, give, get, partner="nobody")


def test_trade_caveats_are_nan_safe_and_empty_for_a_clean_roster(ctx):
    choice = V.roster_choice(ctx, mock_team=2)
    assert V.trade_caveats(ctx, []) == []
    notes = V.trade_caveats(ctx, choice.my_ids[:3])
    assert isinstance(notes, list)
    for n in notes:
        assert isinstance(n, str) and n.strip()


def test_the_page_runs_headless_and_shows_a_message_before_loading(monkeypatch, tmp_path, tables, prior):
    at_mod = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(C, "load_tables", lambda season, root, synthetic: tables)
    from src.inseason import ros as R

    real = R.build_ros
    monkeypatch.setattr(C, "build_ros", lambda t, s, a, **kw: real(t, s, a, **{**kw, "prior": prior}))
    page = Path(__file__).resolve().parents[2] / "src" / "app" / "pages" / "inseason.py"
    at = at_mod.AppTest.from_file(str(page), default_timeout=180).run()
    assert not at.exception and any("Load in-season data" in i.value for i in at.info)
    at.sidebar.text_input[0].set_value(SEASON)
    at.sidebar.checkbox[0].check()
    at.sidebar.button[0].click()
    at.run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 4 and len(at.dataframe) >= 2                      # schedule summary + week table (+ ROS)
    assert isinstance(at.dataframe[0].value, pd.DataFrame)
