"""Regression coverage for the "Load / rebuild board" cache-freshness bug: the button must always
get current data from disk, never a stale ``st.cache_data`` result from an earlier click (the
board cache has no ttl, unlike the watchlist's ttl=600 and the coach table's ttl=3600)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PAGE = Path(__file__).resolve().parents[2] / "src" / "app" / "draft_board.py"


def _board(name: str) -> pd.DataFrame:
    return pd.DataFrame({
        "rank": [1], "player_id": [1], "name": [name], "position": ["PG"], "tier": [1],
        "proj_fppg": [50.0], "proj_gp": [70.0], "proj_total_fp": [3500.0], "vorp": [900.0],
        "adp": [1.0], "fppg_p10": [30.0], "fppg_p50": [40.0], "fppg_p90": [50.0],
    })


def test_load_rebuild_board_button_reflects_data_changed_on_disk_between_clicks(monkeypatch):
    """Core regression test: click "Load / rebuild board" once, change what ``load_board`` returns
    (standing in for a daily_refresh run picking up a new trade/roster move/ADP change on disk),
    click it again, and confirm the app shows the *new* data rather than the cached first result."""
    at_mod = pytest.importorskip("streamlit.testing.v1")

    from src.app import loader

    calls = {"n": 0}

    def fake_load_board(*_args, **_kwargs):
        calls["n"] += 1
        return _board(f"Player{calls['n']}")

    monkeypatch.setattr(loader, "load_board", fake_load_board)

    at = at_mod.AppTest.from_file(str(PAGE), default_timeout=120)
    at.run()
    assert not at.exception

    load_button = [b for b in at.sidebar.button if b.label.startswith("Load")][0]

    load_button.click()
    at.run()
    assert not at.exception, at.exception
    assert calls["n"] == 1
    first = at.tabs[0].dataframe[0].value
    assert first["name"].tolist() == ["Player1"]

    # Click again with identical sidebar inputs: same cache-function arguments as before. Without
    # busting the cache first, Streamlit would hand back the "Player1" board from the first click.
    load_button = [b for b in at.sidebar.button if b.label.startswith("Load")][0]
    load_button.click()
    at.run()
    assert not at.exception, at.exception
    assert calls["n"] == 2, "load_board was not called again: the button did not bypass the cache"
    second = at.tabs[0].dataframe[0].value
    assert second["name"].tolist() == ["Player2"], "the board still shows the stale first result"

    # And the freshness caption is present, confirming the UI shows when the board was (re)loaded.
    caption_text = " ".join(c.value for c in at.caption)
    assert "Board loaded" in caption_text


def test_cached_load_board_reuses_result_for_identical_arguments(monkeypatch):
    """The other half of the cache-freshness story (item 3, draft-day-dry-run open items): the
    decorator must actually help when nothing changed -- two calls with identical arguments should
    hit ``load_board`` once, not twice, and return the identical board object's data. This is what
    makes a same-session tab switch / repeat rerun fast without needing the explicit
    ``.clear()`` the rebuild button always does (see the test above) -- without this, the
    decorator would be dead weight and the ~27s real-data cost would repeat on every rerun."""
    import src.app.draft_board as draft_board_mod

    calls = {"n": 0}

    def fake_load_board(*_args, **_kwargs):
        calls["n"] += 1
        return _board("Cached")

    # draft_board.py does `from src.app.loader import ... load_board ...`, binding the name
    # directly into its own module namespace -- patch that bound name, not src.app.loader's,
    # since _cached_load_board (already imported) closes over draft_board's own namespace.
    monkeypatch.setattr(draft_board_mod, "load_board", fake_load_board)
    _cached_load_board = draft_board_mod._cached_load_board
    _cached_load_board.clear()

    first = _cached_load_board("2026-27", "baseline", 13, False, None, "auto")
    second = _cached_load_board("2026-27", "baseline", 13, False, None, "auto")

    assert calls["n"] == 1, "identical arguments should not re-invoke load_board a second time"
    pd.testing.assert_frame_equal(first, second)
    _cached_load_board.clear()


def test_load_rebuild_board_clears_the_cache_before_calling_it():
    """Unit-level source check in case the full AppTest round-trip above is ever impractical to
    keep passing: the button handler must call ``_cached_load_board.clear()`` before invoking
    ``_cached_load_board`` (not after, and not only on a code path that skips the call), so a
    rebuild always bypasses whatever Streamlit had cached from an earlier click.

    AppTest executes the page as a fresh ``__main__`` module on every ``.run()`` (see
    ``streamlit.runtime.scriptrunner.script_runner``), rather than through this file's own
    ``src.app.draft_board`` import, so a mock/spy attached to the imported module's
    ``_cached_load_board.clear`` would silently watch the wrong function object. Reading the
    button handler's own source is the reliable way to pin this down at the unit level.
    """
    source = PAGE.read_text(encoding="utf-8")
    handler_start = source.index('if st.sidebar.button("Load / rebuild board"')
    handler = source[handler_start:handler_start + 800]
    clear_idx = handler.index("_cached_load_board.clear()")
    call_idx = handler.index("board = _cached_load_board(")
    assert clear_idx < call_idx, "_cached_load_board.clear() must run before the cached call"


def _watchlist_result(name: str):
    from src.value.breakouts import WatchlistResult

    wl = pd.DataFrame({
        "name": [name], "team": ["BOS"], "position": ["PG"], "age": [21],
        "is_rookie": [True], "draft_pick": [5], "board_rank": [200], "adp": [float("nan")],
        "adp_gap": [float("nan")], "base_fppg": [10.0], "layer_fppg": [14.0], "uplift": [4.0],
        "proj_gp": [60.0], "proj_total_fp": [840.0], "vorp": [-50.0], "useful_prob": [0.2],
        "breakout_prob": [0.3], "changed_team": [False], "risk_flags": [""], "contract_flag": [""],
        "sl_z": [1.0], "pre_z": [1.0], "evidence": ["SL 4g"], "player_id": [1],
        "young": [True], "under_radar": [True],
    })
    return WatchlistResult(watchlist=wl, model="baseline", notes=[])


def test_refresh_watchlist_button_reflects_data_changed_on_disk_between_clicks(monkeypatch):
    """The board's rebuild-cache-bust fix is mirrored for the watchlist: after the first load, the
    tab stashes the result in session_state and (before this fix) never called ``load_watchlist``
    again for the same (season, teams) key, ttl or not. The "Refresh watchlist" button must clear
    the cache and re-fetch, so a daily_refresh pulling new Summer League / preseason data reaches an
    already-open session."""
    at_mod = pytest.importorskip("streamlit.testing.v1")

    from src.app import loader

    calls = {"n": 0}

    def fake_load_board(*_args, **_kwargs):
        return _board("Anchor")

    def fake_load_watchlist(*_args, **_kwargs):
        calls["n"] += 1
        return _watchlist_result(f"Prospect{calls['n']}")

    monkeypatch.setattr(loader, "load_board", fake_load_board)
    monkeypatch.setattr(loader, "load_watchlist", fake_load_watchlist)

    at = at_mod.AppTest.from_file(str(PAGE), default_timeout=120)
    at.run()
    assert not at.exception

    # A board must be loaded first: the Breakouts tab only renders once main() gets past its
    # "load a board" early return.
    board_button = [b for b in at.sidebar.button if b.label.startswith("Load")][0]
    board_button.click()
    at.run()
    assert not at.exception, at.exception

    breakouts_tab = at.tabs[5]
    load_button = [b for b in breakouts_tab.button if b.label == "Load the breakout watchlist"][0]
    load_button.click()
    at.run()
    assert not at.exception, at.exception
    assert calls["n"] == 1
    breakouts_tab = at.tabs[5]
    first = breakouts_tab.dataframe[0].value
    assert first["name"].tolist() == ["Prospect1"]

    # The "Refresh watchlist" button lives in the branch taken once ``wl_key`` already matches the
    # current (season, teams) — that only becomes true starting with the *next* run after the load,
    # so a plain rerun (no click) is needed here before it appears. The dataframe stays "Prospect1"
    # and ``load_watchlist`` is not called again: this run reuses the stashed result, as intended.
    at.run()
    assert not at.exception, at.exception
    assert calls["n"] == 1, "a plain rerun must not re-fetch the watchlist on its own"
    breakouts_tab = at.tabs[5]
    still_first = breakouts_tab.dataframe[0].value
    assert still_first["name"].tolist() == ["Prospect1"]

    refresh_button = [b for b in breakouts_tab.button if b.label == "Refresh watchlist"]
    assert refresh_button, "the Refresh watchlist button did not appear after the first load"
    refresh_button[0].click()
    at.run()
    assert not at.exception, at.exception
    assert calls["n"] == 2, "load_watchlist was not called again: the refresh button did not bypass the cache"
    breakouts_tab = at.tabs[5]
    second = breakouts_tab.dataframe[0].value
    assert second["name"].tolist() == ["Prospect2"], "the watchlist still shows the stale first result"
