"""Context loading, roster resolution, the waiver finder and all three CLIs, on the synthetic league."""
import json

import numpy as np
import pandas as pd
import pytest
from ise_testkit import CFG13, SEASON, date_after_games, make_tables

from src.contracts import HISTORY_TABLES, History
from src.inseason import context as C
from src.inseason import ros as R
from src.inseason import trade as T
from src.inseason import waivers as W
from src.inseason.lineup import bench_weight_for, lineup_value, roster_frame
from src.models.registry import get_projector
from src.value.replacement import league_shape


@pytest.fixture(scope="module")
def tables():
    return make_tables()


@pytest.fixture(scope="module")
def prior(tables):
    return get_projector("baseline").project(History.until({k: tables[k] for k in HISTORY_TABLES}, SEASON))


@pytest.fixture(scope="module")
def as_of(tables):
    return date_after_games(tables, 22)


@pytest.fixture()
def ctx(tables, prior, as_of):
    return C.load_context(SEASON, as_of, synthetic=True, tables=tables, prior=prior, cfg=CFG13)


# ------------------------------------------------------------------ context

def test_load_context_builds_ros_schedule_calendar_and_weekly(ctx, as_of):
    assert ctx.season == SEASON and ctx.as_of == as_of and len(ctx.ros) > 50
    assert ctx.schedule is not None and ctx.calendar.source == "derived" and not ctx.weekly.empty
    assert any("derived" in n for n in ctx.notes)
    assert ctx.league is None and ctx.id_map == {} and ctx.injuries is None


def test_preseason_context_says_so(tables, prior):
    c = C.load_context(SEASON, "2023-09-01", synthetic=True, tables=tables, prior=prior, cfg=CFG13)
    assert any("preseason projection" in n for n in c.notes)


def test_bad_season_and_missing_data_are_context_unavailable(tmp_path):
    with pytest.raises(C.ContextUnavailable):
        C.load_context("nonsense", synthetic=True)
    with pytest.raises(C.ContextUnavailable, match="isn't available"):
        C.load_context(SEASON, "2023-12-01", data_dir=tmp_path / "empty")


def test_mock_snake_draft_is_a_snake_and_deterministic():
    vals = pd.Series({i: 100 - i for i in range(12)})
    d = C.mock_snake_draft(vals, 3, 4)
    assert d[0] == [0, 5, 6, 11] and d[1] == [1, 4, 7, 10] and d[2] == [2, 3, 8, 9]
    assert C.mock_snake_draft(vals, 3, 4) == d


LEAGUE = {"teams": [
    {"team_id": 1, "name": "Mine", "roster": [{"espn_player_id": 100, "lineup_slot": "PG"}, {"espn_player_id": 101, "lineup_slot": "IR"},
                                              {"espn_player_id": 999, "lineup_slot": "BE"}]},
    {"team_id": 2, "name": "Theirs", "roster": [{"espn_player_id": 102, "lineup_slot": "C"}]},
    {"team_id": 3, "name": "Empty", "roster": []},
]}


def test_mapped_teams_reports_unmapped_players_instead_of_guessing():
    teams, unmapped = C.mapped_teams(LEAGUE, {"100": 11, "101": 12, "102": 13})
    assert unmapped == 1 and teams[1]["players"] == [(11, "PG"), (12, "IR")] and teams[3]["players"] == []
    assert C.mapped_teams(None, {}) == ({}, 0)


def test_resolve_roster_from_espn_team_names_mock_and_errors(ctx):
    ids = list(ctx.ros["player_id"][:6])
    ctx.league = LEAGUE
    ctx.id_map = {"100": ids[0], "101": ids[1], "102": ids[2]}
    ch = C.resolve_roster(ctx, team_id=1)
    assert ch.my_ids == ids[:2] and ch.ir_ids == [ids[1]] and ch.rostered == set(ids[:3])
    assert "Mine" in ch.label and list(ch.others.values()) == [[ids[2]]]
    assert any("could not be mapped" in n for n in ctx.notes)
    with pytest.raises(C.ContextUnavailable, match="has no players yet"):
        C.resolve_roster(ctx, team_id=3)
    with pytest.raises(C.ContextUnavailable, match="say whose roster"):
        C.resolve_roster(ctx)
    ctx.league, ctx.id_map = None, {}
    names = ", ".join(ctx.ros.set_index("player_id").loc[ids[:3], "name"])
    assert C.resolve_roster(ctx, roster_names=names).my_ids == ids[:3]
    mock = C.resolve_roster(ctx, mock_team=2)
    assert len(mock.my_ids) == 13 and len(mock.rostered) == 13 * 13 and len(mock.others) == 12
    with pytest.raises(C.ContextUnavailable):
        C.resolve_roster(ctx, mock_team=99)


def test_free_agent_pool_excludes_rostered_and_inactive_players(ctx):
    mock = C.resolve_roster(ctx, mock_team=1)
    fa = C.free_agent_ids(ctx, mock.rostered)
    assert not set(fa) & mock.rostered and len(fa) > 0
    played = set(ctx.ros.loc[ctx.ros["gp"] > 0, "player_id"])
    assert set(fa) <= played                                          # only players active this season


def test_active_universe_ignores_a_roster_snapshot_that_shares_no_id_with_ros(tables, prior, monkeypatch):
    """A real NBA roster-snapshot file lives in the shared, cross-worktree NBA_DATA_DIR -- it can be sitting
    there from ordinary real-data use even while this context is built on the synthetic league (`--synthetic`
    / "Use synthetic demo data" in the app). Preseason (every row's gp == 0) plus a snapshot full of ids that
    don't exist in this ros frame used to leave active_universe() empty rather than falling back to the full
    synthetic universe, which silently emptied everything filtered through it: the mock-draft-team demo gave
    every team a 0-player roster (the app's trade analyzer/waiver finder then had no players to pick from),
    with no error anywhere to say why."""
    c = C.load_context(SEASON, "2023-09-01", synthetic=True, tables=tables, prior=prior, cfg=CFG13)
    assert (c.ros["gp"] == 0).all()  # preseason: the games-played path alone is empty, same as the real bug

    foreign_snapshot = pd.DataFrame({
        "player_id": [999999001, 999999002],
        "snapshot_date": [pd.Timestamp("2023-08-01")] * 2,
    })
    monkeypatch.setattr("src.ingest.nba_incoming.read_roster_snapshots", lambda base=None: foreign_snapshot)

    uni = C.active_universe(c)
    assert uni == set(c.ros["player_id"])          # falls back to the full synthetic universe, not emptied
    assert not uni & set(foreign_snapshot["player_id"])  # the foreign ids themselves are never let in

    mock = C.resolve_roster(c, mock_team=1)
    assert len(mock.my_ids) > 0                    # the mock draft actually gets a roster again


def test_active_universe_still_uses_a_snapshot_id_that_is_genuinely_in_ros(ctx):
    """The intersection-before-union fix must not stop a genuinely-relevant snapshot id (one that really is
    in this ros frame) from still extending the games-played universe beyond gp > 0, same as before the fix
    -- only a snapshot that shares *no* id with ros should be ignored (the two tests above/below)."""
    no_games_id = int(ctx.ros.loc[ctx.ros["gp"] == 0, "player_id"].iloc[0])
    assert no_games_id not in C.active_universe(ctx)  # not played, no snapshot yet: correctly excluded

    with pytest.MonkeyPatch.context() as mp:
        snapshot = pd.DataFrame({"player_id": [no_games_id], "snapshot_date": [ctx.as_of]})
        mp.setattr("src.ingest.nba_incoming.read_roster_snapshots", lambda base=None: snapshot)
        assert no_games_id in C.active_universe(ctx)  # a real, relevant snapshot id still gets included


def test_league_json_id_map_and_injury_cache_are_read_from_the_data_dir(tmp_path, tables):
    from src.ingest.http_cache import CachedHttpClient
    from src.store import write_table

    base = tmp_path / "d"
    (base / "processed" / "espn_league").mkdir(parents=True)
    (base / "processed" / "espn_league" / "77_2024.json").write_text(json.dumps(LEAGUE), encoding="utf-8")
    assert C.load_league_json(77, SEASON, base)["teams"][0]["name"] == "Mine"
    assert C.load_league_json(78, SEASON, base) is None and C.load_league_json(None, SEASON, base) is None
    (base / "processed" / "espn_league" / "79_2024.json").write_text("{not json", encoding="utf-8")
    assert C.load_league_json(79, SEASON, base) is None
    idm = pd.DataFrame({"player_id": [11], "source": ["espn"], "source_id": ["100"], "source_name": ["x"],
                        "match_method": ["exact"], "confidence": [1.0]})
    write_table(idm, "player_id_map", base)
    mp = C.load_id_map(base)
    assert mp == {"100": 11} and C.load_id_map(tmp_path / "none") == {}

    class Resp:
        status_code = 200
        headers = {}
        content = json.dumps([{"id": 100, "injuryStatus": "OUT"}, {"id": 101, "injuryStatus": "ACTIVE"},
                              {"id": 5, "injuryStatus": "OUT"}]).encode()

    class Sess:
        def get(self, *a, **k):
            return Resp()

    CachedHttpClient(base / "raw" / "espn", offline=False, session=Sess(), write_fetch_log=False).get_json(
        "players/2024", "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/2024/players",
        {"view": "kona_player_info", "limit": 600})
    inj = C.load_injuries(SEASON, base, mp)
    assert dict(inj) == {11: "OUT"}                                   # unmapped ESPN ids and ACTIVE players dropped
    assert C.load_injuries(SEASON, tmp_path / "none", mp) is None     # nothing cached: no network, no crash


# ------------------------------------------------------------------ waivers

def _mine(ctx):
    return C.resolve_roster(ctx, mock_team=3)


def test_find_waivers_matches_a_brute_force_add_drop(ctx):
    ch = _mine(ctx)
    rep = W.find_waivers(ctx, ch, top=10, candidates=25)
    assert rep.adds["gain_ros_fp"].is_monotonic_decreasing and len(rep.adds) == 10
    rostered = ch.rostered
    assert not set(rep.adds["player_id"]) & rostered
    shape = league_shape(ctx.cfg)
    idx = ctx.ros.drop_duplicates("player_id").set_index("player_id")
    w = bench_weight_for(ctx.ros, shape)
    base = lineup_value(roster_frame(idx, ch.my_ids), shape, w).value
    top = rep.adds.iloc[0]
    best = max(lineup_value(roster_frame(idx, [p for p in ch.my_ids if p != d] + [top["player_id"]]), shape, w).value - base
               for d in ch.my_ids)
    assert top["gain_ros_fp"] == pytest.approx(best)
    assert top["drop_id"] in ch.my_ids and top["drop"] == idx.at[top["drop_id"], "name"]


def test_an_open_roster_spot_needs_no_drop(ctx):
    ch = _mine(ctx)
    short = C.RosterChoice(ch.my_ids[:12], [], set(ch.my_ids[:12]), "short")   # everyone else is a free agent
    rep = W.find_waivers(ctx, short, top=5, candidates=20)
    assert set(rep.adds["drop"]) <= set(ctx.ros["name"]) | {"(open roster spot)"}
    assert len(rep.adds) == 5 and rep.adds["gain_ros_fp"].max() > 0   # measured against a replacement-level filler


def test_streaming_view_uses_games_left_in_the_week_and_excludes_injured_players(ctx):
    ch = _mine(ctx)
    rep = W.find_waivers(ctx, ch, top=8, candidates=30)
    assert rep.week is not None and "week" in rep.week_label and len(rep.streams)
    s = rep.streams
    assert s["week_gain_fp"].is_monotonic_decreasing
    np.testing.assert_allclose(s["week_exp_fp"], s["week_exp_games"] * s["ros_fppg"], rtol=1e-9)
    assert (s["week_exp_games"] <= s["week_team_games"] + 1e-9).all()
    victim = int(s["player_id"].iloc[0])
    ctx.injuries = pd.Series({victim: "OUT"})
    rep2 = W.find_waivers(ctx, ch, top=8, candidates=30)
    assert victim not in set(rep2.streams["player_id"])               # out for the week: not a streamer
    tab = rep2.adds.set_index("player_id")
    assert all(tab.loc[tab.index == victim, "injury"] == "OUT")
    nxt = W.find_waivers(ctx, ch, top=5, candidates=20, next_week=True)
    assert nxt.week == rep.week + 1


def test_without_a_schedule_the_streaming_view_is_unavailable_but_adds_work(ctx):
    ctx.schedule, ctx.calendar, ctx.weekly = None, None, None
    rep = W.find_waivers(ctx, _mine(ctx), top=5, candidates=15)
    assert len(rep.adds) == 5 and rep.streams.empty and any("streaming" in n for n in rep.notes)


def test_flags_and_absent_players_are_reported(ctx):
    rep = W.find_waivers(ctx, _mine(ctx), top=30, candidates=80)
    assert {"rising_minutes", "rising_usage", "inherits_from", "exp_gain_mpg", "min_delta"} <= set(rep.adds.columns)
    assert isinstance(rep.absent, pd.DataFrame) and {"player_id", "streak", "long_term"} <= set(rep.absent.columns)


# ------------------------------------------------------------------ CLIs

@pytest.fixture()
def cli_env(monkeypatch, tables, prior):
    monkeypatch.setattr(C, "load_tables", lambda season, root, synthetic: tables)
    monkeypatch.setattr(C, "load_league", lambda *a, **k: CFG13)
    monkeypatch.setattr(R, "build_ros", _cached_build(prior))
    monkeypatch.setattr(C, "build_ros", R.build_ros)
    return tables


def _cached_build(prior):
    real = R.build_ros

    def build(tables, season, as_of, **kw):
        kw["prior"] = prior
        return real(tables, season, as_of, **kw)

    return build


def _names(ctx, n):
    return list(ctx.ros.set_index("player_id").loc[list(ctx.ros["player_id"][:n]), "name"])


def test_ros_cli(cli_env, capsys, as_of, tmp_path):
    out = tmp_path / "ros.csv"
    rc = R.main(["--synthetic", "--season", SEASON, "--as-of", str(as_of.date()), "--top", "5", "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and "ROS projection" in text and out.exists()
    assert pd.read_csv(out)["ros_rank"].iloc[0] == 1
    assert R.main(["--synthetic", "--season", "bogus"]) == 2 and "error:" in capsys.readouterr().err


def test_trade_cli_end_to_end(cli_env, capsys, ctx, as_of):
    ch = _mine(ctx)
    idx = ctx.ros.drop_duplicates("player_id").set_index("player_id")["name"]
    give, get = idx[ch.my_ids[0]], idx[[p for p in C.free_agent_ids(ctx, ch.rostered)][0]]
    others = [p for ids in ch.others.values() for p in ids][:1]
    rc = T.main(["--synthetic", "--season", SEASON, "--as-of", str(as_of.date()), "--mock-draft-team", "3",
                 "--give", give, "--get", idx[others[0]]])
    out = capsys.readouterr().out
    assert rc == 0 and "Trade analysis" in out and "roster value over the rest of the season" in out
    rc = T.main(["--synthetic", "--season", SEASON, "--as-of", str(as_of.date()), "--mock-draft-team", "3",
                 "--give", "nobody at all", "--get", get])
    assert rc == 2 and "no player matches" in capsys.readouterr().err
    rc = T.main(["--synthetic", "--season", SEASON, "--as-of", str(as_of.date()), "--give", give, "--get", get])
    assert rc == 2 and "say whose roster" in capsys.readouterr().err


def test_waivers_cli_end_to_end(cli_env, capsys, as_of):
    rc = W.main(["--synthetic", "--season", SEASON, "--as-of", str(as_of.date()), "--mock-draft-team", "4", "--top", "5"])
    out = capsys.readouterr().out
    assert rc == 0 and "Waiver finder" in out and "Streaming candidates" in out
    rc = W.main(["--synthetic", "--season", SEASON, "--as-of", str(as_of.date())])
    assert rc == 2 and "error:" in capsys.readouterr().err
