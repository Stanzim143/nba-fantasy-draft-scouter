"""Leakage guards: structural (History), behavioural (future-invariance) and the cheater catalogue."""
import numpy as np
import pandas as pd
import pytest

from bt_helpers import NoiseProjector, OracleProjector
from src.backtest import leakage as L
from src.backtest.benchmarks import NaiveLastSeason
from src.backtest.errors import LeakageError
from src.backtest.harness import walk_forward
from src.contracts import History, season_start

S = "2017-18"


class Recorder:
    """Honest projector that records everything it was shown."""
    name = "recorder"

    def __init__(self):
        self.seen = []
        self.inner = NaiveLastSeason()

    def project(self, history):
        self.seen.append(history)
        return self.inner.project(history)


# ------------------------------------------------------------------ structural

def test_harness_never_hands_a_projector_any_season_ge_target(tables):
    rec = Recorder()
    seasons = ["2016-17", "2017-18", "2018-19"]
    walk_forward(tables, rec, seasons)
    assert [h.target_season for h in rec.seen] == seasons
    for h in rec.seen:
        cut = season_start(h.target_season)
        for name in ("game_logs", "team_games", "player_season_bio"):
            df = getattr(h, name)
            assert len(df) and (df["season"].map(season_start) < cut).all(), name
        h.assert_no_future()
        # The last season the projector can see is exactly the one before the target.
        assert season_start(h.last_season) == cut - 1


def test_extras_are_sliced_and_checked(tables):
    extra = pd.DataFrame({"season": ["2015-16", "2017-18", "2018-19"], "player_id": [1, 1, 1], "x": [1, 2, 3]})
    tables["injuries"] = extra
    h = L.build_history(tables, S)
    assert h.extras["injuries"]["season"].tolist() == ["2015-16"]
    # a hand-built History with a leaking extra is caught by the stricter check ...
    bad = History.until(tables, S)
    bad.extras["injuries"] = extra
    bad.assert_no_future()                      # ... which the contract's own check misses
    with pytest.raises(LeakageError, match="extras"):
        L.assert_history_clean(bad)


def test_sanitize_players_strips_future_derived_fields(tables):
    p = tables["players"]
    clean = L.sanitize_players(p, S)
    s = season_start(S)
    assert clean["to_year"].max() <= s - 1
    assert (clean["from_year"].dropna() < s).all()
    assert not (clean["draft_year"].dropna() > s).any()
    # undrafted players first seen at/after S are dropped (not knowable point-in-time)
    future_undrafted = p[p["draft_year"].isna() & (p["from_year"] >= s)]
    assert not clean["player_id"].isin(future_undrafted["player_id"]).any()
    # ... and it never mutates the input table
    assert p["to_year"].max() >= s
    # everyone who actually played before S is kept
    played_before = set(tables["game_logs"].query("season < @S")["player_id"])
    assert played_before <= set(clean["player_id"])


def test_build_history_flags_unsanitised_players(tables):
    # src/contracts.py's History.until now sanitises players unconditionally (this was itself
    # the leak this module first found and worked around locally; see ADR 0001 changelog), so
    # sanitize=False on build_history is a no-op today. The bypass below hand-builds a History
    # to simulate that guarantee being weakened again, to prove two independent checks still
    # catch it: contracts.py's own History.assert_no_future (the earliest, most general layer —
    # it fires first and raises plain AssertionError) and this module's assert_history_clean
    # (LeakageError), exercised directly against assert_no_future already having been satisfied.
    h = L.build_history(tables, S, sanitize=False)
    L.assert_history_clean(h, sanitized=True)  # nothing to flag: History.until already sanitised it

    leaky = History.until(tables, S)
    leaky.players = tables["players"]  # bypass History.until, restoring the raw (unsanitised) table
    with pytest.raises(AssertionError, match="players"):
        leaky.assert_no_future()

    # With that first, cheaper layer bypassed too, this module's own check still catches it.
    leaky.assert_no_future = lambda: None
    with pytest.raises(LeakageError, match="players"):
        L.assert_history_clean(leaky, sanitized=True)


def test_data_hash_is_content_based(tables):
    h1 = L.data_hash(tables)
    shuffled = {k: v.sample(frac=1.0, random_state=0).reset_index(drop=True) for k, v in tables.items()}
    assert L.data_hash(shuffled) == h1                       # row order does not matter
    tables["game_logs"].loc[0, "pts"] += 1
    assert L.data_hash(tables) != h1                          # any value change does


# ------------------------------------------------------------------ behavioural: honest passes

def test_honest_projector_passes_and_tables_are_restored(tables):
    before = L.data_hash(tables)
    frames = {k: v for k, v in tables.items()}
    L.assert_projector_ignores_future(NaiveLastSeason(), tables, S)
    assert L.data_hash(tables) == before
    assert all(tables[k] is frames[k] for k in tables)        # same objects, mutated-and-restored in place


def test_scramble_actually_changes_future_rows_and_only_them(tables):
    gl = tables["game_logs"]
    fut = gl["season"].map(season_start) >= season_start(S)
    past_before = gl[~fut].copy()
    fut_before = gl[fut].copy()
    with L.scrambled_future(tables, S):
        assert gl.loc[~fut].equals(past_before)
        assert not gl.loc[fut, "pts"].equals(fut_before["pts"])
        assert not gl.loc[fut, "player_id"].equals(fut_before["player_id"])
        assert tables["players"]["to_year"].max() >= season_start(S)
    pd.testing.assert_frame_equal(gl, pd.concat([past_before, fut_before]).sort_index())


def test_tables_restored_even_if_projector_raises(tables):
    before = L.data_hash(tables)

    class Boom:
        name = "boom"

        def project(self, history):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        with L.scrambled_future(tables, S):
            Boom().project(None)
    assert L.data_hash(tables) == before


# ------------------------------------------------------------------ behavioural: cheaters are caught

def test_oracle_cheater_via_captured_reference_is_caught(tables):
    oracle = OracleProjector(tables)
    with pytest.raises(LeakageError, match="reads future data"):
        L.assert_projector_ignores_future(oracle, tables, S)


def test_oracle_is_caught_in_every_season(tables):
    oracle = OracleProjector(tables)
    for s in ("2016-17", "2017-18", "2018-19"):
        with pytest.raises(LeakageError):
            L.assert_projector_ignores_future(oracle, tables, s)


def test_partial_cheater_using_only_next_season_gp_is_caught(tables):
    class GpPeeker:
        """Uses the past for per-game rates but peeks at the real games-played."""
        name = "gp_peeker"

        def __init__(self, tables):
            self._t = tables
            self._naive = NaiveLastSeason()

        def project(self, history):
            proj = self._naive.project(history)
            gl = self._t["game_logs"]
            gp = gl[gl["season"] == history.target_season].groupby("player_id")["game_id"].count()
            proj["proj_gp"] = proj["player_id"].map(gp).fillna(0.0).astype(float)
            proj["proj_total_fp"] = proj["proj_fppg"] * proj["proj_gp"]
            return proj

    with pytest.raises(LeakageError):
        L.assert_projector_ignores_future(GpPeeker(tables), tables, S)


class ToYearPeeker:
    """Reads the static players table: to_year reveals how long a player keeps playing."""
    name = "to_year_peeker"

    def __init__(self):
        self._naive = NaiveLastSeason()

    def project(self, history):
        proj = self._naive.project(history)
        ty = history.players.set_index("player_id")["to_year"]
        s = season_start(history.target_season)
        proj["proj_gp"] = (40.0 + 5.0 * (proj["player_id"].map(ty).fillna(s - 1) - s)).clip(0, 82)
        proj["proj_total_fp"] = proj["proj_fppg"] * proj["proj_gp"]
        return proj


def test_players_table_side_channel_needs_sanitising(tables):
    # History.until (src/contracts.py) now sanitises players unconditionally, so the channel is
    # closed by construction: the to_year peeker finds nothing to exploit regardless of the
    # sanitize= flag here, which is now a no-op kept only for backward compatibility.
    L.assert_projector_ignores_future(ToYearPeeker(), tables, S, sanitize=False)
    L.assert_projector_ignores_future(ToYearPeeker(), tables, S, sanitize=True)

    # The exploit is real in principle: prove it against a History whose players table bypasses
    # History.until's sanitisation (hand-edited, as no public API can produce this anymore).
    clean = L.build_history(tables, S)
    bypassed = History.until(tables, S)
    bypassed.players = tables["players"]
    honest = NaiveLastSeason().project(clean)
    exploited = ToYearPeeker().project(bypassed)
    assert not honest["proj_total_fp"].equals(exploited["proj_total_fp"])


def test_from_year_debut_peeker_sees_no_rookies_when_sanitised(tables):
    """A projector that looks for players with from_year == target would find every true rookie.

    This used to demonstrate a real leak in the contract's own History.until (see git history /
    ADR 0001 changelog): raw.players["from_year"] == s used to be non-empty. History.until now
    sanitises players unconditionally, so the assertion below is the regression test for that fix.
    """
    s = season_start(S)
    raw = History.until(tables, S)
    assert not (raw.players["from_year"] == s).any()           # the leak is closed in the contract itself
    clean = L.build_history(tables, S)
    assert not (clean.players["from_year"].dropna() >= s).any()
    debut_ids = set(tables["game_logs"].query("season == @S")["player_id"]) - set(
        tables["game_logs"].query("season < @S")["player_id"])
    drafted_debutants = debut_ids & set(clean.players["player_id"])
    # remaining debutants are only visible through draft_year (known before the season), never from_year
    for pid in drafted_debutants:
        row = clean.players[clean.players.player_id == pid].iloc[0]
        assert pd.notna(row["draft_year"]) and row["draft_year"] <= s


def test_stochastic_projector_is_rejected_not_misreported_as_leak(tables):
    class Dice:
        name = "dice"

        def __init__(self):
            self.inner = NaiveLastSeason()

        def project(self, history):
            p = self.inner.project(history)
            p["proj_fppg"] = p["proj_fppg"] * np.random.random(len(p))   # unseeded
            p["proj_total_fp"] = p["proj_fppg"] * p["proj_gp"]
            return p

    with pytest.raises(ValueError, match="not deterministic"):
        L.assert_projector_ignores_future(Dice(), tables, S)


def test_noise_projector_using_oracle_data_is_also_flagged(tables):
    with pytest.raises(LeakageError):
        L.assert_projector_ignores_future(NoiseProjector(tables, 0.2), tables, S)


def test_walk_forward_leak_check_hook(tables):
    walk_forward(tables, NaiveLastSeason(), ["2017-18", "2018-19"], leak_check=["2018-19"])
    with pytest.raises(LeakageError):
        walk_forward(tables, OracleProjector(tables), ["2017-18", "2018-19"], leak_check=["2018-19"])
    with pytest.raises(ValueError, match="not among"):
        walk_forward(tables, NaiveLastSeason(), ["2017-18"], leak_check=["2018-19"])
