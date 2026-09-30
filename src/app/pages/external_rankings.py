"""External rankings comparison page (ADR 0028): our own draft board rank vs. Yahoo and FantasyPros
snapshot exports.

Streamlit lists it next to the draft board and the in-season page (``streamlit run
src/app/draft_board.py``, then pick "external rankings" in the sidebar page picker). A separate page
rather than another draft_board.py tab: this comparison is not draft-state-specific (nothing here
reads or writes drafted/undrafted status) and its own inputs (two file paths, refreshed manually by
the user) don't fit the draft board's season/model sidebar.

Nothing is parsed or built until **Load comparison** is pressed, so opening the app stays fast. All
join/parse logic lives in ``src.ingest.yahoo_rankings``, ``src.ingest.fantasypros_rankings`` and
``src.value.rankings_compare``; this file only wires widgets to it, following the same split as
``src/app/draft_board.py`` / ``src/app/pages/inseason.py``.

Yahoo and FantasyPros are one-off manual exports (see the ADR): there is no live API poll here, just
whatever file currently sits under ``<NBA_DATA_DIR>/external_rankings/``. Re-drop a fresher export
and reload to refresh.
"""
from __future__ import annotations

# ruff: noqa: E402  (imports must follow the sys.path bootstrap below)
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from src.app import glossary
from src.app.loader import BoardUnavailable, load_board
from src.contracts import data_dir as default_data_dir
from src.ingest.fantasypros_rankings import FantasyProsParseError, load_fantasypros_rankings
from src.ingest.yahoo_rankings import YahooRankingsError, load_yahoo_rankings
from src.models.registry import available_projectors
from src.value.rankings_compare import biggest_disagreements, build_comparison, unmatched_report

st.set_page_config(page_title="NBA Fantasy external rankings", layout="wide")

DEFAULT_YAHOO_NAME = "yahoo_latest.xlsx"
DEFAULT_FANTASYPROS_NAME = "fantasypros_latest.csv"


def _column_config(columns) -> dict:
    """Same convention as draft_board.py / inseason.py: a hover tooltip on every column with a
    glossary entry (src.app.glossary), the single source of truth for this text."""
    return {c: st.column_config.Column(help=glossary.tooltip(c)) for c in columns if glossary.tooltip(c)}


def _default_path(base: Path, filename: str) -> Path:
    p = base / "external_rankings" / filename
    return p


@st.cache_data(ttl=600, show_spinner="Loading and matching external rankings...")
def _cached_load(season: str, model: str, synthetic: bool, data_dir: str, teams: int,
                 yahoo_path: str, fp_path: str):
    from src.store import read_table

    base = Path(data_dir) if data_dir else None
    board = load_board(season, model=model, teams=teams or None, synthetic=synthetic, data_dir=base)
    players = read_table("players", base) if not synthetic else _synthetic_players(season, base)

    notes: list[str] = []
    resolved: dict[str, pd.DataFrame] = {}

    yp = Path(yahoo_path)
    if yp.exists():
        try:
            resolved["yahoo"] = load_yahoo_rankings(yp, players).frame
        except YahooRankingsError as exc:
            notes.append(f"Yahoo file could not be parsed ({exc}); excluded.")
    else:
        notes.append(f"Yahoo file not found at {yp}; excluded.")

    fpp = Path(fp_path)
    if fpp.exists():
        try:
            resolved["fantasypros"] = load_fantasypros_rankings(fpp, players).frame
        except FantasyProsParseError as exc:
            notes.append(f"FantasyPros file could not be parsed ({exc}); excluded.")
    else:
        notes.append(f"FantasyPros file not found at {fpp}; excluded.")

    comparison = build_comparison(board, resolved) if resolved else pd.DataFrame()
    unmatched = unmatched_report(resolved) if resolved else pd.DataFrame()
    return comparison, unmatched, resolved, notes


def _synthetic_players(season: str, data_dir: Path | None) -> pd.DataFrame:
    """The board's own synthetic-demo path has no real ``players`` table to match names against;
    build the same synthetic universe the demo board itself uses so a demo run still shows something
    (every name will legitimately be unmatched against Yahoo/FantasyPros' real player names -- that
    is expected and reported, not a bug)."""
    from src.contracts import season_start
    from src.synthetic import make_synthetic_tables

    start = season_start(season)
    tables = make_synthetic_tables(first_start=start - 6, last_start=start - 1, n_teams=14,
                                   games_per_team=60, seed=0)
    return tables["players"]


def _sidebar():
    st.sidebar.header("Board to compare")
    season = st.sidebar.text_input("Season", value="2026-27")
    models = available_projectors()
    model = st.sidebar.selectbox("Preseason model", models, index=models.index("baseline"))
    teams = st.sidebar.number_input("Teams override (0 = config)", min_value=0, value=0, step=1)
    synthetic = st.sidebar.checkbox("Use synthetic demo board", value=False,
                                    help="A deterministic fake league; names will not match Yahoo/FantasyPros.")
    data_dir = st.sidebar.text_input("Data directory override (optional)", value="")

    st.sidebar.header("External rankings files")
    base = Path(data_dir) if data_dir else default_data_dir()
    yahoo_path = st.sidebar.text_input(
        "Yahoo .xlsx path", value=str(_default_path(base, DEFAULT_YAHOO_NAME)),
        help="Manual export, 'Players' sheet. See docs/adr/0028-external-rankings-compare.md.")
    fp_path = st.sidebar.text_input(
        "FantasyPros .csv path", value=str(_default_path(base, DEFAULT_FANTASYPROS_NAME)),
        help="Manual export from the FantasyPros draft rankings page.")

    if st.sidebar.button("Load comparison", type="primary"):
        try:
            comparison, unmatched, resolved, notes = _cached_load(
                season, model, synthetic, data_dir, int(teams), yahoo_path, fp_path)
        except BoardUnavailable as exc:
            st.sidebar.error(str(exc))
            return
        st.session_state.rankings_cmp = comparison
        st.session_state.rankings_unmatched = unmatched
        st.session_state.rankings_resolved = resolved
        st.session_state.rankings_notes = notes
        st.sidebar.success(f"Loaded: {len(comparison)} players in the comparison.")


def _summary(resolved: dict[str, pd.DataFrame], notes: list[str]) -> None:
    cols = st.columns(len(resolved) + 1) if resolved else [st.container()]
    for i, (source, frame) in enumerate(resolved.items()):
        n = len(frame)
        matched = int(frame["matched"].sum())
        with cols[i]:
            st.metric(f"{source} matched", f"{matched}/{n}", help="Rows resolved onto a player_id; see the "
                      "Unmatched names tab for the rest.")
    for n in notes:
        st.info(n)


def _comparison_tab(comparison: pd.DataFrame) -> None:
    if comparison.empty:
        st.info("No comparison data loaded yet. Configure the sidebar and press **Load comparison**.")
        return
    st.caption(
        "rank_delta_<source> = our_rank - <source>_rank. **Positive** = that source ranks the player "
        "earlier (more favorably) than we do. **Negative** = we rank him earlier than that source. "
        "A player missing from FantasyPros (top ~300 only) or absent from our board is expected, not an "
        "anomaly -- see the on_our_board / on_yahoo / on_fantasypros flags."
    )
    name_filter = st.text_input("Filter by name (substring, case-insensitive)", value="")
    only_disagreements = st.checkbox("Only players on at least two sources (a rank_delta exists)", value=False)
    min_delta = st.slider("Minimum |rank_delta| (either source)", min_value=0, max_value=200, value=0, step=5)

    view = comparison
    if name_filter:
        view = view[view["name"].str.contains(name_filter, case=False, na=False)]
    if only_disagreements:
        view = view[view["max_abs_rank_delta"].notna()]
    if min_delta:
        view = view[view["max_abs_rank_delta"].fillna(0) >= min_delta]

    sort_choice = st.selectbox(
        "Sort by", ["Our rank", "Biggest disagreement (either direction)",
                    "Yahoo disagreement", "FantasyPros disagreement"])
    if sort_choice == "Biggest disagreement (either direction)":
        view = view.assign(_s=view["max_abs_rank_delta"].abs()).sort_values("_s", ascending=False, na_position="last").drop(columns="_s")
    elif sort_choice == "Yahoo disagreement":
        view = view.assign(_s=view["rank_delta_yahoo"].abs()).sort_values("_s", ascending=False, na_position="last").drop(columns="_s")
    elif sort_choice == "FantasyPros disagreement":
        view = view.assign(_s=view["rank_delta_fantasypros"].abs()).sort_values("_s", ascending=False, na_position="last").drop(columns="_s")
    # "Our rank" keeps build_comparison's own default order.

    st.dataframe(view, use_container_width=True, hide_index=True, column_config=_column_config(view.columns))


def _biggest_movers_tab(comparison: pd.DataFrame) -> None:
    if comparison.empty:
        st.info("No comparison data loaded yet.")
        return
    n = st.slider("How many to show", min_value=5, max_value=100, value=25, step=5, key="movers_n")
    source_choice = st.radio("Source", ["Either (max)", "Yahoo", "FantasyPros"], horizontal=True)
    source = {"Either (max)": None, "Yahoo": "yahoo", "FantasyPros": "fantasypros"}[source_choice]
    top = biggest_disagreements(comparison, n=n, source=source)
    st.dataframe(top, use_container_width=True, hide_index=True, column_config=_column_config(top.columns))


def _unmatched_tab(unmatched: pd.DataFrame) -> None:
    if unmatched.empty:
        st.success("Every row from every loaded source resolved onto a player_id.")
        return
    st.caption(
        "Names present in an external source that this project's own name-matching "
        "(src.ingest.id_map) could not resolve onto a player_id -- never silently dropped from the "
        "raw file, just excluded from the rank comparison above. 'ambiguous' means more than one "
        "candidate matched; 'unmatched' means none did."
    )
    source_filter = st.multiselect("Source", sorted(unmatched["source"].unique()),
                                   default=sorted(unmatched["source"].unique()))
    view = unmatched[unmatched["source"].isin(source_filter)] if source_filter else unmatched
    st.dataframe(view, use_container_width=True, hide_index=True, column_config=_column_config(view.columns))


def main() -> None:
    st.title("External rankings comparison")
    st.caption(
        "Compares this project's own draft board rank against manually-exported Yahoo and "
        "FantasyPros rankings snapshots. See docs/adr/0028-external-rankings-compare.md."
    )
    _sidebar()

    comparison = st.session_state.get("rankings_cmp", pd.DataFrame())
    unmatched = st.session_state.get("rankings_unmatched", pd.DataFrame())
    resolved = st.session_state.get("rankings_resolved", {})
    notes = st.session_state.get("rankings_notes", [])

    if resolved:
        _summary(resolved, notes)
    elif notes:
        for n in notes:
            st.info(n)

    tab_all, tab_movers, tab_unmatched = st.tabs(["All players", "Biggest disagreements", "Unmatched names"])
    with tab_all:
        _comparison_tab(comparison)
    with tab_movers:
        _biggest_movers_tab(comparison)
    with tab_unmatched:
        _unmatched_tab(unmatched)


main()
