"""In-season tools page (ADR 0015): schedule, rest-of-season projection, trade analyzer, waiver finder.

Streamlit lists it next to the draft board (``streamlit run src/app/draft_board.py``). Nothing is computed
until you press **Load in-season data** (building the rest-of-season projection takes a few seconds to a
minute), so opening the app or the draft board stays as fast as before. All logic lives in
``src/app/inseason_view.py`` and ``src/inseason``; this file only wires widgets to it. Offline-safe: it reads
the local data directory and the ESPN caches, and never makes a network request.
"""
from __future__ import annotations

# ruff: noqa: E402  (imports must follow the sys.path bootstrap below)
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import streamlit as st

from src.app import glossary, inseason_view as V, live_sync
from src.models.registry import available_projectors

live_sync.load_dotenv_file(_REPO_ROOT)

st.set_page_config(page_title="NBA Fantasy in-season tools", layout="wide")

#: Confidence bucket -> the Streamlit banner function that reads as that bucket (the same semantic
#: green/amber/red idiom ``st.metric``'s own delta coloring already uses elsewhere in this app; there
#: is no bespoke CSS/badge styling anywhere in the app to instead reuse, see draft_board.py).
_CONFIDENCE_BANNER = {"clear_win": st.success, "lean_your_way": st.info, "roughly_even": st.info,
                      "lean_other_way": st.warning, "clear_loss": st.error}


def _column_config(columns) -> dict:
    """``column_config`` for a ``st.dataframe`` call: a hover tooltip on every column that has a
    glossary entry (``src.app.glossary``), the single source of truth for this text — the same helper
    ``draft_board.py`` uses (``tests/app/test_glossary.py`` covers both)."""
    return {c: st.column_config.Column(help=glossary.tooltip(c)) for c in columns if glossary.tooltip(c)}


@st.cache_resource(ttl=600, show_spinner="Building the rest-of-season projection...")
def _cached_load(season: str, as_of: str, model: str, synthetic: bool, data_dir: str, league_id: int):
    return V.load(season, as_of, model=model, synthetic=synthetic, data_dir=data_dir or None,
                  league_id=league_id or None)


def _sidebar():
    st.sidebar.header("In-season data")
    season = st.sidebar.text_input("Season", value="2026-27")
    as_of = st.sidebar.date_input("As of (games on or before this date count)", value=date.today())
    models = available_projectors()
    model = st.sidebar.selectbox("Preseason model", models, index=models.index("baseline"))
    synthetic = st.sidebar.checkbox("Use synthetic demo data", value=False,
                                    help="A deterministic fake league; nothing real is needed.")
    data_dir = st.sidebar.text_input("Data directory override (optional)", value="")
    league_id = st.sidebar.number_input("ESPN league id (0 = $ESPN_LEAGUE_ID)", min_value=0, value=0, step=1)
    if st.sidebar.button("Load in-season data", type="primary"):
        try:
            st.session_state.ictx = _cached_load(season, str(as_of), model, synthetic, data_dir, int(league_id))
        except V.ContextUnavailable as exc:
            st.session_state.ictx = None
            st.sidebar.error(str(exc))
        else:
            st.sidebar.success(f"Loaded {len(st.session_state.ictx.ros)} players, as of {as_of}.")


def _roster_picker(ctx):
    mode = st.radio("Roster source", ["ESPN team", "Type names", "Mock draft team (demo)"], horizontal=True,
                    key="roster_mode", label_visibility="collapsed")
    kw = {}
    if mode == "ESPN team":
        kw["team_id"] = int(st.number_input("ESPN team id", min_value=0, value=0, step=1, key="team_id")) or None
    elif mode == "Type names":
        kw["names"] = st.text_area("Your players, comma separated", key="roster_names")
    else:
        kw["mock_team"] = int(st.number_input("Mock draft team (1-13)", min_value=1, max_value=30, value=1, key="mock_team"))
    try:
        return V.roster_choice(ctx, **kw)
    except (V.ContextUnavailable, V.NameError_) as exc:
        st.info(str(exc))
        return None


def _schedule_tab(ctx):
    if ctx.weekly is None:
        st.info("No schedule is stored for this season. Run `python -m src.inseason.schedule --season "
                f"{ctx.season} --ingest` once (one cached ESPN request).")
        return
    st.caption(f"Matchup calendar source: **{ctx.calendar.source}**. "
               "Heavy = 4+ games in 7 days, light = 2 or fewer. Off-night games are games on a light league-wide slate.")
    summary = V.weeks_summary(ctx)
    st.dataframe(summary, use_container_width=True, hide_index=True)
    cur = ctx.calendar.current_week(ctx.as_of)
    weeks = [w.week for w in ctx.calendar.weeks]
    week = st.selectbox("Week", weeks, index=weeks.index(cur.week) if cur else 0)
    st.dataframe(V.week_table(ctx, week), use_container_width=True, hide_index=True)


def _ros_tab(ctx):
    rep = ctx.ros.attrs.get("replacement", {})
    st.caption(f"Preseason projection shrunk toward {ctx.season} production through {ctx.as_of.date()}; "
               f"replacement level {rep.get('total', float('nan')):.0f} FP over the rest of the season.")
    c1, c2, c3 = st.columns([2, 1, 1])
    search = c1.text_input("Search by name", key="ros_search")
    pos = c2.selectbox("Position", ["All", "PG", "SG", "SF", "PF", "C", "G", "F"], key="ros_pos")
    top = int(c3.number_input("Rows", min_value=10, max_value=800, value=100, step=10, key="ros_top"))
    st.dataframe(V.ros_table(ctx, search=search, position=pos, top=top), use_container_width=True, hide_index=True)


def _trade_side_summary(label: str, side: dict) -> None:
    st.metric(f"{label}: rest-of-season FP change", f"{side['delta']:+.0f}",
             help=glossary.tooltip("trade_delta"))
    banner = _CONFIDENCE_BANNER.get(side["confidence_bucket"], st.info)
    banner(f"{side['confidence_label'].capitalize()} ({side['verdict']}, tolerance ±{side['tolerance']:.0f} FP)",
          icon=None)
    summary = pd.DataFrame([{
        "trade_verdict": side["verdict"], "trade_confidence": side["confidence_label"],
        "trade_before_value": side["before"], "trade_after_value": side["after"], "trade_tolerance": side["tolerance"],
        "trade_dropped": ", ".join(side["dropped"]) or "none",
        "trade_empty_slots": ", ".join(side["empty_slots"]) or "none",
    }])[V.TRADE_SUMMARY_COLUMNS]
    st.dataframe(summary, column_config=_column_config(summary.columns), use_container_width=True, hide_index=True)


def _trade_tab(ctx, choice):
    if choice is None:
        return
    opts = V.player_options(ctx)
    mine = [p for p in choice.my_ids if p in opts]
    give = st.multiselect("You give", mine, format_func=lambda p: opts[p], key="tr_give")
    others = [p for p in opts if p not in set(choice.my_ids)]
    get = st.multiselect("You get", others, format_func=lambda p: opts[p], key="tr_get")
    partner_options = ["(none)"] + sorted(choice.others)
    partner_pick = st.selectbox("Also score the other side of the trade (optional)", partner_options,
                                key="tr_partner", help=glossary.tooltip("trade_partner"))
    partner = None if partner_pick == "(none)" else partner_pick
    if not (give or get):
        st.caption("Pick players on at least one side.")
        return
    try:
        out = V.run_trade(ctx, choice, give, get, partner=partner)
    except ValueError as exc:
        st.error(str(exc))
        return
    except V.ContextUnavailable as exc:
        st.error(str(exc))
        return
    _trade_side_summary("You", out)
    lines = out["lines"].rename(columns={"side": "trade_side", "player": "trade_player",
                                         "ros_total_fp": "trade_ros_total_fp"})[V.TRADE_LINES_COLUMNS]
    st.dataframe(lines, column_config=_column_config(lines.columns), use_container_width=True, hide_index=True)
    if out["notes"]:
        st.caption("Advisory notes and caveats (never change the projection):")
        for n in out["notes"]:
            st.caption(f"- {n}")
    if out["partner"] is not None:
        st.divider()
        _trade_side_summary(out["partner_label"], out["partner"])


def _waivers_tab(ctx, choice):
    if choice is None:
        return
    next_week = st.checkbox("Stream for next week", value=False, key="wv_next")
    top = int(st.number_input("Rows", min_value=5, max_value=60, value=20, key="wv_top"))
    if not st.button("Find free agents", key="wv_go"):
        st.caption("Searching tries every add/drop pair against your lineup; it takes a few seconds.")
        return
    rep = V.run_waivers(ctx, choice, top=top, next_week=next_week)
    for n in rep.notes:
        st.caption(n)
    st.subheader("Best adds (rest-of-season value over the player replaced)")
    st.dataframe(rep.adds[[c for c in V.ADD_COLS if c in rep.adds.columns]], use_container_width=True, hide_index=True)
    st.subheader(f"Streaming candidates, {rep.week_label}")
    if rep.streams.empty:
        st.info("No schedule loaded, so no streaming view.")
    else:
        st.dataframe(rep.streams[[c for c in V.STREAM_COLS if c in rep.streams.columns]], use_container_width=True,
                     hide_index=True)


def main():
    st.title("NBA Fantasy in-season tools")
    st.caption("Schedule, rest-of-season projections, trade analyzer and waiver finder. See docs/inseason.md.")
    _sidebar()
    ctx = st.session_state.get("ictx")
    if ctx is None:
        st.info("Set the season and date in the sidebar and press **Load in-season data**. No real data? Tick "
                "**Use synthetic demo data**.")
        return
    for n in ctx.notes:
        st.caption(f"note: {n}")
    with st.expander("Whose roster? (used by the trade analyzer and the waiver finder)", expanded=False):
        choice = _roster_picker(ctx)
    t1, t2, t3, t4 = st.tabs(["Schedule", "Rest of season", "Trade analyzer", "Waivers"])
    with t1:
        _schedule_tab(ctx)
    with t2:
        _ros_tab(ctx)
    with t3:
        _trade_tab(ctx, choice)
    with t4:
        _waivers_tab(ctx, choice)


main()
