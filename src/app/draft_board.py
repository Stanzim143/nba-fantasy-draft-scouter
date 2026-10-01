"""Live draft board — Streamlit entrypoint.

Run it with::

    streamlit run src/app/draft_board.py

See ``docs/app.md`` for what it does and does not do, and
``docs/adr/0009-streamlit-app.md`` for the design decisions behind it.

This module is intentionally thin: it wires ``st.*`` calls to plain, unit-tested functions in
``src.app.loader`` (board building) and ``src.app.state`` (draft-in-progress state, filtering,
suggestions). Nothing that decides *what* the app does lives in here — only how it's rendered.
"""
from __future__ import annotations

# ruff: noqa: E402  (imports must follow the sys.path bootstrap below)
import sys
from pathlib import Path

# `streamlit run src/app/draft_board.py` puts this file's own directory on sys.path, not the
# repo root, so `import src...` would otherwise fail with "No module named 'src'" unless the app
# happens to be launched with the repo root already on PYTHONPATH. Put it there ourselves.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.app import coach_view, glossary, live_sync, txn_view
from src.app.loader import BoardUnavailable, load_board, load_watchlist, validate_data_dir_override
from src.app.state import (
    ME,
    OPPONENT,
    RANK_CAPTION,
    DraftError,
    DraftState,
    best_available,
    best_by_position,
    display_columns,
    draft_player,
    filter_board,
    json_to_state,
    load_flag_table,
    my_position_counts,
    need_board_view,
    rankings_disagreement_table,
    return_flag_table,
    state_to_dataframe,
    state_to_json,
    undraft_player,
)
from src.contracts import raw_dir, season_start
from src.ingest.espn_league import SOURCE as ESPN_SOURCE
from src.ingest.espn_league import ESPNLeagueClient
from src.models.registry import available_projectors
from src.value.breakouts import SORT_OPTIONS, select_watchlist
from src.value.league import load_league
from src.value.positions import ESPN_POSITIONS
from src.value.rankings_compare import DISAGREEMENT_THRESHOLD

# Pick up ESPN_LEAGUE_ID / ESPN_TEAM_ID from the repo's .env (never overrides real env vars).
live_sync.load_dotenv_file(_REPO_ROOT)

st.set_page_config(page_title="NBA Fantasy Draft Board", layout="wide")

WATCH_DISPLAY = ["watch_rank", "name", "team", "position", "age", "is_rookie", "draft_pick", "board_rank", "adp",
                 "useful_prob", "breakout_prob", "uplift", "layer_fppg", "proj_gp", "risk_flags", "contract_flag", "evidence"]
SORT_LABELS = {"useful_prob": "Chance of a useful breakout (calibrated)", "breakout_prob": "Chance of a breakout (calibrated)",
               "uplift": "Model uplift over the baseline (FPPG)", "sl_z": "Raw Summer League production (no model)",
               "pre_z": "Raw preseason production (no model)"}



@st.cache_data(ttl=600, show_spinner="Reading Summer League and preseason lines and projecting...")
def _cached_load_watchlist(season: str, model: str | None, teams: int | None, data_dir: str | None):
    return load_watchlist(season, model=model, teams=teams, data_dir=Path(data_dir) if data_dir else None)
POSITION_FILTER_OPTIONS = ["All", *ESPN_POSITIONS, "G", "F", "UTIL"]


def _column_config(columns) -> dict:
    """``column_config`` for a ``st.dataframe``/``st.data_editor`` call: a hover tooltip on every
    column that has a glossary entry (``src.app.glossary``), the single source of truth for this
    text (see the Glossary tab, ``_glossary_tab``). A column with no entry is left without a
    ``help=`` rather than raising, so an unexpected/ad-hoc column never breaks the app — but
    ``tests/app/test_glossary.py`` fails the build if that ever happens for a real board/table
    column, which is the point."""
    return {c: st.column_config.Column(help=glossary.tooltip(c)) for c in columns if glossary.tooltip(c)}


@st.cache_data(show_spinner="Building the board...")
def _cached_load_board(season: str, model: str, teams: int | None, synthetic: bool,
                       data_dir: str | None, positional: str) -> pd.DataFrame:
    # Streamlit's cache needs hashable, simple arguments; keep data_dir as a str/None here and
    # convert back to a Path just before calling the loader.
    return load_board(season, model, teams=teams, synthetic=synthetic,
                      data_dir=Path(data_dir) if data_dir else None, positional=positional)


def _init_state() -> None:
    if "draft_state" not in st.session_state:
        st.session_state.draft_state = DraftState()
    if "board" not in st.session_state:
        st.session_state.board = None
    if "board_meta" not in st.session_state:
        st.session_state.board_meta = {}
    if "live_sync_status" not in st.session_state:
        st.session_state.live_sync_status = live_sync.LiveSyncStatus()
    if "live_sync_enabled" not in st.session_state:
        st.session_state.live_sync_enabled = False
    if "live_sync_client" not in st.session_state:
        st.session_state.live_sync_client = None  # built lazily, keyed by league id (see below)
    if "live_sync_client_league_id" not in st.session_state:
        st.session_state.live_sync_client_league_id = None


# --------------------------------------------------------------------------- live draft sync (ESPN)

def _live_sync_client(league_id: int) -> ESPNLeagueClient:
    """Reuse one client per league id across reruns, so its own rate limit/backoff state (last
    request time, retry count) persists between polls instead of resetting on every click."""
    if (st.session_state.live_sync_client is None
            or st.session_state.live_sync_client_league_id != league_id):
        st.session_state.live_sync_client = ESPNLeagueClient(cache_dir=raw_dir(ESPN_SOURCE))
        st.session_state.live_sync_client_league_id = league_id
    return st.session_state.live_sync_client


def _board_name_position_maps(board: pd.DataFrame | None) -> tuple[dict, dict]:
    if board is None or "player_id" not in board.columns:
        return {}, {}
    names = dict(zip(board["player_id"], board["name"]))
    positions = dict(zip(board["player_id"], board.get("position", pd.Series(dtype=object))))
    return names, positions


def _run_live_sync(league_id: int, my_team_id: int | None, *, force: bool) -> None:
    """Poll ESPN once (subject to the poll-interval gate and the cross-session lock, unless
    ``force`` -- an explicit "Sync now" click always attempts the request) and apply any newly
    detected picks. Never raises: every failure path updates ``live_sync_status`` with an error
    message instead, and the manual "Mark drafted" form is untouched either way."""
    status: live_sync.LiveSyncStatus = st.session_state.live_sync_status
    now = datetime.now(timezone.utc)
    if not force and not live_sync.should_poll(status.last_synced_at, now):
        return

    lock_path = raw_dir(ESPN_SOURCE) / f"{league_id}_live_sync.lock"
    client = _live_sync_client(league_id)
    season = st.session_state.board_meta.get("season", "2026-27")
    try:
        season_id = season_start(season) + 1
    except ValueError:
        season_id = season_start("2026-27") + 1
    id_map = live_sync.load_id_map()
    board_names, board_positions = _board_name_position_maps(st.session_state.board)
    result = live_sync.sync_once(
        client, league_id, season_id,
        seen_overall_picks=status.seen_overall_picks, my_team_id=my_team_id,
        id_map=id_map, board_names=board_names, board_positions=board_positions, now=now,
        lock_path=lock_path,
    )

    status.last_synced_at = result.fetched_at
    if result.locked_out:
        status.last_skipped_locked = True
        return
    status.last_skipped_locked = False
    if not result.ok:
        status.last_error = result.error
        status.consecutive_failures += 1
        return
    status.last_error = None
    status.consecutive_failures = 0
    status.seen_overall_picks = result.seen_overall_picks
    if not result.detected:
        return
    new_state, applied, duplicate, unresolved = live_sync.apply_detected_picks(
        st.session_state.draft_state, list(result.detected))
    st.session_state.draft_state = new_state
    if applied:
        who = ", ".join(f"{d.name or d.espn_player_id} ({'me' if d.drafted_by == ME else 'opponent'})"
                        for d in applied)
        st.sidebar.success(f"Live sync: auto-marked {len(applied)} pick(s) drafted — {who}.")
    if duplicate:
        st.sidebar.caption(f"Live sync: {len(duplicate)} detected pick(s) were already marked "
                           "(manual click beat the sync) — not double-counted.")
    if unresolved:
        st.sidebar.warning(
            f"Live sync: {len(unresolved)} pick(s) could not be matched to a board player "
            "(no player_id_map entry yet — run `python -m src.ingest.espn_adp` to refresh it). "
            "Mark them manually below.")


DEFAULT_REAL_MODEL = "baseline_hurdle_adp_offseason_debut"


def _default_model() -> str:
    """The best measured model when real data is ingested, else the plain baseline (synthetic demo, fresh clone)."""
    try:
        from src.contracts import data_dir

        has_real = (data_dir() / "processed" / "game_logs.parquet").exists()
    except OSError:
        has_real = False
    return DEFAULT_REAL_MODEL if has_real and DEFAULT_REAL_MODEL in available_projectors() else "baseline"


def _sidebar() -> None:
    st.sidebar.header("Board")
    season = st.sidebar.text_input("Season", value=st.session_state.board_meta.get("season", "2026-27"),
                                   help="e.g. 2026-27")
    default_model = _default_model()
    model = st.sidebar.selectbox("Model", options=available_projectors(),
                                 index=available_projectors().index(
                                     st.session_state.board_meta.get("model", default_model))
                                 if st.session_state.board_meta.get("model", default_model) in available_projectors()
                                 else 0,
                                 help=f"`{DEFAULT_REAL_MODEL}` is the best measured model on real data (hurdle availability, "
                                      "ADP as a preseason signal, Summer League / preseason layer, debutants; ADR 0031/0032). "
                                      "`baseline` is the plain model and the one to use with the synthetic demo.")
    teams_override = st.sidebar.number_input("Teams (0 = use league config)", min_value=0, max_value=30,
                                             value=st.session_state.board_meta.get("teams") or 0)
    positional = st.sidebar.selectbox("Positional scarcity", options=["auto", "on", "off"], index=0)
    synthetic = st.sidebar.checkbox(
        "Use synthetic demo data", value=st.session_state.board_meta.get("synthetic", False),
        help="No real ingested data needed — try the app against a deterministic fake league.")
    data_dir = st.sidebar.text_input("Data directory override (optional)", value="",
                                     help="Leave blank to use NBA_DATA_DIR / ~/dev-data/nba-fantasy-2026")

    if st.sidebar.button("Load / rebuild board", type="primary"):
        # Bust the cache first: a plain call with the same arguments would otherwise hand back
        # whatever was cached from an earlier click, even if a daily_refresh run on disk has since
        # picked up a new trade/roster move/ADP change. The decorator stays (it still makes
        # ordinary reruns, like a tab switch, fast); only an explicit rebuild forces fresh data.
        _cached_load_board.clear()
        try:
            checked_dir = validate_data_dir_override(data_dir)
            board = _cached_load_board(season, model, teams_override or None, synthetic,
                                       str(checked_dir) if checked_dir else None, positional)
        except BoardUnavailable as e:
            st.session_state.board = None
            st.sidebar.error(str(e))
        else:
            st.session_state.board = board
            st.session_state.board_meta = {"season": season, "model": model,
                                           "teams": teams_override or None,
                                           "synthetic": synthetic, "positional": positional}
            st.session_state.board_loaded_at = pd.Timestamp.now(tz="UTC")
            st.sidebar.success(f"Loaded {len(board)} players for {season} ({model}).")

    st.sidebar.divider()
    st.sidebar.header("Draft state")
    if st.sidebar.button("Reset draft (clear all picks)"):
        st.session_state.confirm_reset_draft = True
    if st.session_state.get("confirm_reset_draft"):
        st.sidebar.warning(f"This clears all {len(st.session_state.draft_state)} picks and the live-sync "
                           "memory of which picks were seen. Export first if unsure.")
        c_yes, c_no = st.sidebar.columns(2)
        if c_yes.button("Yes, reset", type="primary"):
            st.session_state.draft_state = DraftState()
            # Also forget which ESPN picks were seen, or live sync would never re-apply them.
            st.session_state.live_sync_status = live_sync.LiveSyncStatus()
            st.session_state.confirm_reset_draft = False
            st.sidebar.info("Draft state cleared.")
        if c_no.button("Cancel"):
            st.session_state.confirm_reset_draft = False
            st.rerun()

    n_picks = len(st.session_state.draft_state)
    st.sidebar.download_button(
        "Export draft state (JSON)",
        data=state_to_json(st.session_state.draft_state, st.session_state.board_meta),
        file_name="draft_state.json", mime="application/json",
        disabled=n_picks == 0,
        help="Save the in-progress draft so it can be resumed after closing the browser.")

    uploaded = st.sidebar.file_uploader("Import draft state (JSON)", type=["json"])
    if uploaded is not None:
        try:
            new_state, meta = json_to_state(uploaded.getvalue().decode("utf-8"))
        except DraftError as e:
            st.sidebar.error(f"Could not import: {e}")
        else:
            st.session_state.draft_state = new_state
            if meta.get("season") and meta.get("season") != st.session_state.board_meta.get("season"):
                st.sidebar.warning(
                    f"Imported state was saved for season {meta['season']!r}; the current board is "
                    f"{st.session_state.board_meta.get('season')!r}. Picks were restored by player_id "
                    "regardless, but double-check they still make sense.")
            st.sidebar.success(f"Imported {len(new_state)} picks.")

    st.sidebar.divider()
    _live_sync_sidebar()


def _live_sync_sidebar() -> None:
    """Draft-day live sync (ADR 0025): auto-detect opponents' picks from the ESPN league sync
    instead of requiring a manual "Mark drafted" click for every one. Manual marking always stays
    available below regardless of whether this is enabled or working."""
    st.sidebar.header("Live draft sync (ESPN)", help=glossary.tooltip("live_sync_status"))
    enabled = st.sidebar.checkbox(
        "Auto-detect picks from ESPN", value=st.session_state.live_sync_enabled,
        help="Polls the ESPN league sync (read-only) for newly drafted players and marks them on "
            "this board automatically. \"Mark drafted\" below always keeps working, enabled or not.")
    st.session_state.live_sync_enabled = enabled

    default_league_id = live_sync.league_id_from_env()
    default_my_team_id = live_sync.default_team_id()
    league_id = st.sidebar.number_input(
        "ESPN league ID", min_value=0, value=default_league_id or 0, step=1, disabled=not enabled,
        help="Defaults to $ESPN_LEAGUE_ID (.env). 0 disables the sync.")
    my_team_id = st.sidebar.number_input(
        "My ESPN team ID (for attribution)", min_value=0,
        value=default_my_team_id or 0, step=1, disabled=not enabled,
        help="Defaults to $ESPN_TEAM_ID (.env), or whatever `python -m src.ops.nightly --set-team` "
            "last saved. Picks by this team are attributed to \"me\"; "
            "every other team is \"opponent\". Leave at 0 if unknown -- every detected pick is "
            "then attributed to the opponent side.")

    st.session_state.live_sync_league_id = int(league_id) or None
    st.session_state.live_sync_my_team_id = int(my_team_id) or None

    status: live_sync.LiveSyncStatus = st.session_state.live_sync_status
    if enabled and league_id:
        if st.sidebar.button("Sync now"):
            _run_live_sync(int(league_id), int(my_team_id) or None, force=True)
        _live_sync_status_caption(status)
    elif enabled:
        st.sidebar.caption("Enter an ESPN league ID (or set $ESPN_LEAGUE_ID) to start syncing.")


def _live_sync_status_caption(status: "live_sync.LiveSyncStatus") -> None:
    tip = glossary.tooltip("live_sync_status")
    if status.last_synced_at is None:
        st.sidebar.caption("Live sync: not yet run.", help=tip)
        return
    when = status.last_synced_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    if status.connected:
        st.sidebar.caption(f"Live sync: connected, last synced {when}.", help=tip)
    else:
        st.sidebar.caption(
            f"Live sync: ERROR ({status.consecutive_failures} in a row), last attempt {when} -- "
            f"{status.last_error}. Manual \"Mark drafted\" still works.", help=tip)
    if status.last_skipped_locked:
        st.sidebar.caption("(Last poll was skipped: another session is syncing this league.)")


def _mark_drafted_form(board: pd.DataFrame, key_prefix: str) -> None:
    avail = best_available(board, st.session_state.draft_state.drafted_ids)
    if avail.empty:
        st.info("Every projected player has been marked drafted.")
        return
    with st.form(f"{key_prefix}_draft_form", clear_on_submit=True):
        options = list(avail["player_id"])
        labels = {row.player_id: f"#{row.rank} {row.name} ({row.position or '?'})"
                 for row in avail.itertuples()}
        pid = st.selectbox("Player", options=options, format_func=lambda p: labels[p],
                           key=f"{key_prefix}_pid")
        drafted_by = st.radio("Drafted by", options=[ME, OPPONENT], horizontal=True,
                              format_func=lambda v: "Me" if v == ME else "Opponent",
                              key=f"{key_prefix}_by")
        submitted = st.form_submit_button("Mark drafted")
        if submitted:
            row = avail[avail["player_id"] == pid].iloc[0]
            try:
                st.session_state.draft_state = draft_player(
                    st.session_state.draft_state, pid, row["name"], row["position"], drafted_by)
            except DraftError as e:
                st.error(str(e))
            else:
                st.rerun()


def _returned_only_checkbox(board: pd.DataFrame, key: str) -> bool:
    """A filter for the advisory return flag (ADR 0023); shown only when the board carries the flag."""
    if "return_flag" not in board.columns:
        return False
    return st.checkbox("Returned-healthy only", value=False, key=key,
                       help="Missed 25%+ of last season's games at the start, then played 75%+ of the rest (advisory, not a projection).")


def _short_absences_checkbox(board: pd.DataFrame, key: str) -> bool:
    """A filter for the advisory short-absence flag (ADR 0024); shown only when the board carries the flag."""
    if "lm_flag" not in board.columns:
        return False
    return st.checkbox("Many short absences only", value=False, key=key,
                       help="Missed 6+ games in 1-2 game absences last season (cause unknown: rest or minor injury). Advisory, not a projection.")


def _best_available_tab(board: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        position = st.selectbox("Position", options=POSITION_FILTER_OPTIONS, key="ba_position")
    with c2:
        tiers = ["All"] + sorted(board["tier"].dropna().unique().tolist())
        tier = st.selectbox("Tier", options=tiers, key="ba_tier")
    with c3:
        search = st.text_input("Search by name", key="ba_search")
    returned = _returned_only_checkbox(board, "ba_returned")
    short_abs = _short_absences_checkbox(board, "ba_short")

    avail = best_available(board, st.session_state.draft_state.drafted_ids)
    filtered = filter_board(avail, position=position, tier=tier, search=search, returned_only=returned, short_absences_only=short_abs)
    st.caption(f"{len(filtered)} of {len(avail)} best-available players shown "
              f"({len(board) - len(avail)} already drafted). " + RANK_CAPTION)
    cols = display_columns(board)
    st.dataframe(filtered[cols], column_config=_column_config(cols), use_container_width=True, hide_index=True)

    st.subheader("Mark a player drafted")
    _mark_drafted_form(board, "ba")


def _suggestions_tab(board: pd.DataFrame) -> None:
    st.caption("Highest-VORP remaining player at each position — a quick scarcity read, not an "
              "auto-drafter.")
    my_ids = st.session_state.draft_state.my_ids
    counts = my_position_counts(board, my_ids)
    st.write("Your roster so far, by position eligibility: " +
            ", ".join(f"{pos} {n}" for pos, n in counts.items()))

    suggestions = best_by_position(board, st.session_state.draft_state.drafted_ids)
    cols = st.columns(len(ESPN_POSITIONS))
    for col, pos in zip(cols, ESPN_POSITIONS):
        with col:
            st.markdown(f"**{pos}**")
            df = suggestions[pos]
            if df.empty:
                st.caption("none left")
            else:
                sugg_cols = ["rank", "name", "proj_fppg", "vorp", "tier"]
                st.dataframe(df[sugg_cols], column_config=_column_config(sugg_cols),
                            hide_index=True, use_container_width=True)


@st.cache_data(show_spinner=False)
def _cached_league_config() -> dict:
    return load_league()


def _need_tab(board: pd.DataFrame) -> None:
    st.caption(
        "Crosses the best-available board against your own drafted roster's positional gaps "
        "(ADR 0026): each remaining specific position (PG/SG/SF/PF/C) gets a need score from 0 "
        "(fully covered, or your league carries no slot there) to 1 (nothing drafted there yet), "
        "spreading starter, flex (G/F) and bench slots fractionally across the positions that can "
        "fill them. A player's need score is the best of his eligible positions; `need_adj_vorp` "
        "adds a flat bonus (up to 40 FP) scaled by that score on top of `vorp`, which stays visible "
        "unchanged. This is a documented heuristic, not a validated projection input — see the ADR "
        "for its limitations.")
    try:
        league_cfg = _cached_league_config()
    except FileNotFoundError as e:
        st.error(f"Could not load league config: {e}")
        return

    my_ids = st.session_state.draft_state.my_ids
    if not my_ids:
        st.info("You haven't drafted anyone yet, so there is no roster need to weigh: this is "
                "showing plain best-available-by-VORP (need_adj_vorp is vorp plus the same "
                "constant for everyone, so the order is unchanged).")

    view = need_board_view(board, st.session_state.draft_state, league_cfg)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        position = st.selectbox("Position", options=POSITION_FILTER_OPTIONS, key="need_position")
    with c2:
        tiers = ["All"] + sorted(board["tier"].dropna().unique().tolist())
        tier = st.selectbox("Tier", options=tiers, key="need_tier")
    with c3:
        search = st.text_input("Search by name", key="need_search")
    filtered = filter_board(view.table, position=position, tier=tier, search=search)
    st.caption(f"{len(filtered)} of {len(view.table)} best-available players shown, ranked by "
              "need-adjusted VORP (need_rank), highest first.")
    # player_id (NEED_TABLE_COLUMNS, ADR 0026) is present in the data so this table can be joined
    # back onto the live board programmatically, but stays hidden from the rendered grid here --
    # column_config=None hides a column without dropping it from the underlying dataframe.
    st.dataframe(filtered, column_config={**_column_config(filtered.columns), "player_id": None},
                use_container_width=True, hide_index=True)

    st.subheader("Positional need")
    if view.categories_applicable:
        st.caption("This league is a categories/roto format; category need is not computed here "
                  "(see ADR 0026 — positional need only, explicitly scoped out).")
    st.dataframe(view.position_table, column_config=_column_config(view.position_table.columns),
                use_container_width=True, hide_index=True)


def _drafted_tab(board: pd.DataFrame) -> None:  # noqa: ARG001 - kept for a consistent tab signature
    state = st.session_state.draft_state
    df = state_to_dataframe(state)
    if df.empty:
        st.info("No players marked drafted yet.")
        return
    st.dataframe(df, column_config=_column_config(df.columns), use_container_width=True, hide_index=True)

    st.subheader("Undo a pick")
    options = list(df["player_id"])
    labels = {row.player_id: f"#{row.pick_no} {row.name} ({row.drafted_by})" for row in df.itertuples()}
    pid = st.selectbox("Player", options=options, format_func=lambda p: labels[p], key="undo_pid")
    if st.button("Undo this pick"):
        try:
            st.session_state.draft_state = undraft_player(st.session_state.draft_state, pid)
        except DraftError as e:
            st.error(str(e))
        else:
            st.rerun()


def _full_board_tab(board: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        position = st.selectbox("Position", options=POSITION_FILTER_OPTIONS, key="fb_position")
    with c2:
        tiers = ["All"] + sorted(board["tier"].dropna().unique().tolist())
        tier = st.selectbox("Tier", options=tiers, key="fb_tier")
    with c3:
        search = st.text_input("Search by name", key="fb_search")
    returned = _returned_only_checkbox(board, "fb_returned")
    short_abs = _short_absences_checkbox(board, "fb_short")

    st.caption(RANK_CAPTION)
    filtered = filter_board(board, position=position, tier=tier, search=search, returned_only=returned, short_absences_only=short_abs)
    drafted_ids = st.session_state.draft_state.drafted_ids
    by = {p.player_id: p.drafted_by for p in st.session_state.draft_state.picks}
    display = filtered[display_columns(board)].copy()
    display.insert(1, "drafted_by",
                   filtered["player_id"].map(lambda pid: by.get(pid, "") if pid in drafted_ids else ""))
    st.dataframe(display, column_config=_column_config(display.columns), use_container_width=True, hide_index=True)


def _breakouts_tab(board: pd.DataFrame) -> None:  # noqa: ARG001 - kept for a consistent tab signature
    meta = st.session_state.board_meta
    st.caption(
        "Young (23 or under) players the draft market is not paying for, ranked by how far their Summer League and preseason "
        "lines move the projection. The backtest found a real but modest signal: the top ten flagged players became useful "
        "(a breakout that also finished in the rostered pool) about **24%** of the time against a **10%** base rate, so treat "
        "this as an edge, not a lock. The preseason carries most of it; Summer League alone is close to noise.")
    # Streamlit runs every tab's body on every rerun, and building the watchlist projects ten seasons walk-forward (about a
    # minute the first time). Loading it on request keeps that off the critical path of a 90-second pick.
    key = (meta.get("season", "2026-27"), meta.get("teams"))
    if st.session_state.get("wl_key") != key:
        st.info("The watchlist projects ten seasons walk-forward: about a minute the first time, instant afterwards. It loads "
                "on request so it never slows a pick.")
        if not st.button("Load the breakout watchlist", key="wl_load"):
            return
        try:
            st.session_state.wl_result = _cached_load_watchlist(key[0], None, key[1], None)
        except BoardUnavailable as e:
            st.info(str(e))
            return
        st.session_state.wl_key = key
        st.session_state.wl_loaded_at = pd.Timestamp.now(tz="UTC")
    else:
        # Once loaded for this (season, teams) key, the watchlist stays in session_state and is
        # never re-fetched on its own — the ttl=600 cache only matters if something calls the
        # cached function again, which nothing here does automatically. Without this button a
        # refresh on disk (e.g. new Summer League/preseason data from daily_refresh) would never
        # reach an already-open session. Clearing the cache first mirrors the board's rebuild button
        # so this click always gets current data, not whatever is left in the 10-minute cache.
        loaded_at = st.session_state.get("wl_loaded_at")
        st.caption(f"Watchlist loaded {loaded_at:%Y-%m-%d %H:%M UTC}." if loaded_at is not None
                  else "Watchlist loaded.")
        if st.button("Refresh watchlist", key="wl_refresh"):
            _cached_load_watchlist.clear()
            try:
                st.session_state.wl_result = _cached_load_watchlist(key[0], None, key[1], None)
            except BoardUnavailable as e:
                st.info(str(e))
                return
            st.session_state.wl_loaded_at = pd.Timestamp.now(tz="UTC")
    result = st.session_state.wl_result
    for note in result.notes:
        st.caption(note)
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        sort_by = st.selectbox("Rank by", options=list(SORT_OPTIONS), format_func=SORT_LABELS.get, key="wl_sort")
    with c2:
        include_priced = st.checkbox("Include priced players", value=False, key="wl_priced",
                                     help="Players with an ADP inside the top 100.")
    with c3:
        all_ages = st.checkbox("Include older than 23", value=False, key="wl_ages")
    with c4:
        min_uplift = st.number_input("Min uplift (FPPG)", value=0.0, step=0.5, key="wl_uplift")
    hide_drafted = st.checkbox("Hide players already drafted", value=True, key="wl_hide")
    wl = select_watchlist(result.watchlist, young_only=not all_ages, under_radar_only=not include_priced,
                          min_uplift=min_uplift, sort_by=sort_by)
    if hide_drafted:
        wl = best_available(wl, st.session_state.draft_state.drafted_ids)
        wl["watch_rank"] = range(1, len(wl) + 1)
    st.caption(f"{len(wl)} players listed (model: {result.model}).")
    wl_cols = [c for c in WATCH_DISPLAY if c in wl.columns]
    st.dataframe(wl[wl_cols], column_config=_column_config(wl_cols), use_container_width=True, hide_index=True)


def _debutants_risk_tab(board: pd.DataFrame) -> None:
    st.caption(
        "**Debutants** are rostered players with no NBA game and no rookie draft slot this year: overseas stashes and earlier-year "
        "picks (Sorber, Marković, Toohey, Diop, Biberovic) and undrafted / two-way signees. They are projected from the draft-slot "
        "prior, always **low confidence**, and only in the `baseline_debut` / `baseline_offseason_debut` models. **Risk flags** "
        "(ESPN injury status, preseason absence, a new team, a star arriving or leaving) are advisory: they never change a "
        "projection, and only the preseason and roster parts could be checked against history (see docs/offseason.md).")
    notes = board.attrs.get("risk_notes") or []
    for n in notes:
        st.caption(n)
    drafted = st.session_state.draft_state.drafted_ids
    if "projection_class" in board.columns:
        d = board[board["projection_class"].isin(["stash", "undrafted"])]
        d = d[~d["player_id"].isin(drafted)]
        cls = st.radio("Class", ["stash", "undrafted"], horizontal=True, key="db_class")
        d = d[d["projection_class"] == cls]
        st.subheader(f"Debutants: {cls} ({len(d)})")
        cols = [c for c in ("rank", "name", "position", "age", "proj_fppg", "proj_gp", "proj_total_fp", "vorp", "p_play", "adp", "risk_flags") if c in d.columns]
        st.dataframe(d[cols], column_config=_column_config(cols), use_container_width=True, hide_index=True)
    else:
        st.info("This board has no debutants: choose the model `baseline_offseason_debut` (or `baseline_debut`) in the sidebar and rebuild.")
    if "risk_level" in board.columns:
        r = board[(board["risk_level"] != "") & ~board["player_id"].isin(drafted)]
        st.subheader(f"Draft-day risk flags ({len(r)} players)")
        cols = [c for c in ("rank", "name", "risk_level", "risk_flags", "proj_gp", "risk_gp", "proj_total_fp", "adp") if c in r.columns]
        st.dataframe(r[cols], column_config=_column_config(cols), use_container_width=True, hide_index=True)
    else:
        st.info("No risk overlay on this board (it appears for the live season once a roster snapshot exists: python -m src.ingest.preseason_refresh --with-extras).")
    if "return_flag" in board.columns:
        rt = return_flag_table(board, drafted)
        st.subheader(f"Returned from a long absence, healthy since ({len(rt)} players)")
        st.caption("Missed the first 25%+ of last season's team games, then played 75%+ of at least 15 remaining games (ADR 0021 cohort). "
                   "**Advisory judgement, not a projection**: the study found no significant absolute under-projection for this group "
                   "(+1.0 GP, 95% CI -4.1 to +5.8), so `proj_gp`, `proj_total_fp` and the rank are unchanged. `return_gp_upside_adv` is a fixed "
                   "rule (+5 GP for a block of 60%+, else 0) and `return_fp_upside_adv` is those games x `proj_fppg`, in total fantasy "
                   "points (ADR 0023). A player who is not on this list may still have missed time.")
        st.dataframe(rt, column_config=_column_config(rt.columns), use_container_width=True, hide_index=True)
    if "lm_flag" in board.columns:
        lt = load_flag_table(board, drafted)
        st.subheader(f"Many short absences last season ({len(lt)} players)")
        st.caption("Missed 6+ games in 1-2 game absences last season (rotation veterans, single team). **The data cannot say whether that was rest or a "
                   "minor injury**, and the back-to-back one-game absences on their own (`lm_rest_n`) showed no effect. Across ten seasons these players landed "
                   "about 2.6 GP [-5.1, -0.2] below the baseline projection, but the association did not improve out-of-sample forecasts, so "
                   "`proj_gp`, `proj_total_fp` and the rank are unchanged. `lm_gp_risk_adv` is a fixed advisory rule (-2 GP) and `lm_fp_risk_adv` those games x "
                   "`proj_fppg`, in total fantasy points (ADR 0024).")
        st.dataframe(lt, column_config=_column_config(lt.columns), use_container_width=True, hide_index=True)
    if "contract_flag" in board.columns:
        st.subheader("Contract flags (unvalidated)")
        st.caption("From dated Wikipedia contract events before opening night, plus the nominal rookie-scale clock. **Unvalidated and display only**: "
                   "a ten-season walk-forward found no reliable lift from contract year, so nothing here changes a projection or a rank (ADR 0019). "
                   "Coverage is partial and biased toward notable signings; a blank means *not known to be*, never *known not to be*.")
        c = board[(board["contract_flag"] != "") & ~board["player_id"].isin(drafted)]
        flag = st.selectbox("Flag", ["contract year", "new deal", "extension", "rookie final year (nominal)", "rookie option year (nominal)"],
                            key="cf_flag")
        c = c[c["contract_flag"] == flag]
        cols = [k for k in ("rank", "name", "position", "contract_flag", "contract_years_left", "contract_status", "contract_basis",
                            "proj_fppg", "proj_total_fp", "adp") if k in c.columns]
        st.dataframe(c[cols], column_config=_column_config(cols), use_container_width=True, hide_index=True)
    comparison = st.session_state.get("rankings_cmp", pd.DataFrame())
    if comparison is not None and not comparison.empty and "rankings_disagreement" in comparison.columns:
        dis = rankings_disagreement_table(board, comparison, drafted)
        st.subheader(f"External rankings disagreement ({len(dis)} players)")
        st.caption(
            f"Our board rank differs from Yahoo and/or FantasyPros by {DISAGREEMENT_THRESHOLD:.0f}+ rank spots, either "
            "direction (ADR 0029). **Advisory only**: `proj_fppg`, `proj_gp`, `proj_total_fp`, `vorp` and `rank` are "
            "unchanged -- this only says our board and the external consensus disagree sharply, not who is right. "
            "`rankings_disagreement_direction` is 'market_favors' (an external source ranks him earlier than we do) "
            "or 'we_favor' (we rank him earlier than that source).")
        st.dataframe(dis, column_config=_column_config(dis.columns), use_container_width=True, hide_index=True)
    else:
        st.info(
            "No external-rankings comparison loaded this session. Open the **External rankings** page (sidebar page "
            "picker), load Yahoo/FantasyPros, then come back here to see sharp disagreements against this board.")


def _transactions_tab(board: pd.DataFrame) -> None:
    st.caption(
        "Trades, signings, extensions, waivers and coach moves from ESPN's public transactions feed, joined to this board's ranks. "
        "**Descriptive only**: 'a player changed team' and 'a star arrived or left' were tested as projection inputs and did not "
        "help (ADR 0010, 0011), so nothing here changes a projection. Refresh: `python -m src.ops.txn_watch --refresh` "
        "(the daily refresh also does it). See ADR 0017.")
    try:
        ledger = txn_view.load_ledger()
    except txn_view.LedgerUnavailable as exc:
        st.info(str(exc))
        return
    st.caption(txn_view.freshness(ledger))
    c1, c2, c3, c4 = st.columns([1, 2, 2, 2])
    days = c1.number_input("Last N days", min_value=1, max_value=365, value=14)
    team_options = sorted(ledger["team_abbr"].unique())
    teams = c2.multiselect("Teams", team_options)
    kinds = c3.multiselect("Kinds", txn_view.KIND_CHOICES)
    player = c4.text_input("Player contains")
    everyone = st.checkbox("Include players outside the board's top 180 (camp signings, cuts)", value=False)
    roster = txn_view.load_roster()
    mv = txn_view.moves(ledger, board, roster, days=int(days), teams=teams or None, kinds=kinds or None, player=player or None,
                        relevant_only=not everyone)
    st.subheader(f"Player moves ({len(mv)})")
    cols = [c for c in txn_view.DISPLAY_COLUMNS if c in mv.columns]
    st.dataframe(mv[cols], column_config=_column_config(cols), use_container_width=True, hide_index=True)
    st.subheader("Coach and front-office moves")
    sf = txn_view.staff(ledger, days=max(int(days), 120), teams=teams or None)
    st.dataframe(sf, column_config=_column_config(sf.columns), use_container_width=True, hide_index=True)


@st.cache_data(ttl=3600, show_spinner="Building the coach table...")
def _cached_coach_table(season: str) -> pd.DataFrame:
    return coach_view.load_table(season)


def _coaches_tab(board: pd.DataFrame) -> None:
    st.caption(
        "Head coaches for the season, how each runs a team, and which ranked players sit under a new one. **Descriptive only**: "
        "coach style measurably follows a coach (top-five minutes, rotation depth, three-point share; star minutes probably; not pace) but the same "
        "information as a projection input showed no significant lift (ADR 0020), so no projection changes. Youth development is "
        "not a portable coach trait in this data. Refresh: `python -m src.ingest.wiki_coaches --current 2026-27`.")
    season = st.session_state.board_meta.get("season", "2026-27")
    try:
        table = _cached_coach_table(season)
    except coach_view.CoachesUnavailable as exc:
        st.info(str(exc))
        return
    for line in coach_view.summary_lines(table):
        st.write(f"- {line}")
    only_new = st.checkbox("Only teams with a new head coach", value=True, key="coach_only_new")
    show = table[table["new_to_team"]] if only_new else table
    coach_cols = [c for c in coach_view.TABLE_COLUMNS if c in show.columns]
    st.dataframe(show[coach_cols], column_config=_column_config(coach_cols), use_container_width=True, hide_index=True)
    aff = coach_view.affected(table, board)
    st.subheader(f"Ranked players (top 150) under a new head coach ({len(aff)})")
    if len(aff):
        aff_cols = ["rank", "name", "position", "team", "coach", "previous_coach", "first_time_head_coach"]
        st.dataframe(aff[aff_cols], column_config=_column_config(aff_cols),
                     use_container_width=True, hide_index=True)


def _glossary_tab() -> None:
    st.caption(
        "What every column on the board and its tables means, where its formula comes from, and how it was "
        "arrived at — the same facts as `docs/categories.md` (the deep reference, code-verified line by line), "
        "shorter and grouped for scanning. Hover any column header in the tables above for the short version.")
    for group_name, cols in glossary.GROUPS:
        with st.expander(f"{group_name} ({len(cols)} columns)"):
            for col, entry in cols.items():
                st.markdown(f"**`{col}`** — {entry.long}")
                if entry.formula:
                    st.code(entry.formula, language=None)
                st.caption(f"Source: {entry.source}")
                st.divider()


def main() -> None:
    _init_state()
    st.title("NBA Fantasy Draft Board")
    st.caption("Set a season and model in the sidebar, load the board, then mark picks as they "
              "happen. See docs/app.md for details and current limitations.")
    _sidebar()

    # Opportunistic auto-sync: since Streamlit has no background timer, "live" means "try again on
    # every rerun this session already causes (a click, a tab switch, a manual pick)", gated by the
    # poll-interval floor in _run_live_sync -- never a busy-poll, and the manual form is unaffected
    # either way (ADR 0025).
    if st.session_state.live_sync_enabled:
        auto_league_id = st.session_state.get("live_sync_league_id")
        if auto_league_id:
            _run_live_sync(auto_league_id, st.session_state.get("live_sync_my_team_id"), force=False)

    board = st.session_state.board
    if board is None:
        st.info("Load a board from the sidebar to get started. No real data ingested yet? Turn "
               "on **synthetic demo data** in the sidebar to try the app immediately.")
        return

    loaded_at = st.session_state.get("board_loaded_at")
    if loaded_at is not None:
        st.caption(f"Board loaded {loaded_at:%Y-%m-%d %H:%M UTC}. Click \"Load / rebuild board\" in "
                   "the sidebar any time to pick up changes on disk (a new trade, roster move, or "
                   "ADP refresh) — it always rebuilds from current data, never a stale cache.")

    (tab_best, tab_sugg, tab_need, tab_drafted, tab_full, tab_break, tab_risk, tab_txn, tab_coach,
     tab_glossary) = st.tabs(
        ["Best available", "Suggestions", "Best available by need", "Drafted", "Full board",
         "Breakouts", "Debutants & risk", "Transactions", "Coaches", "Glossary"])
    with tab_best:
        _best_available_tab(board)
    with tab_sugg:
        _suggestions_tab(board)
    with tab_need:
        _need_tab(board)
    with tab_drafted:
        _drafted_tab(board)
    with tab_full:
        _full_board_tab(board)
    with tab_break:
        _breakouts_tab(board)
    with tab_risk:
        _debutants_risk_tab(board)
    with tab_txn:
        _transactions_tab(board)
    with tab_coach:
        _coaches_tab(board)
    with tab_glossary:
        _glossary_tab()


if __name__ == "__main__":
    main()
