"""Turn the raw ``league_transactions`` ledger into fantasy-readable moves (ADR 0017).

Pure functions, frames in and frames out, so the CLI (``src.ops.txn_watch``), the daily report and the Streamlit tab all
show the same thing and it is tested offline.

* :func:`events` collapses the ledger's per-team rows into **one row per player move**: a trade appears in the ledger from
  both teams' points of view (``trade_in`` for the receiver, ``trade_out`` for the sender) and becomes one ``trade`` row
  ``from_abbr -> to_abbr``; ``signed`` / ``resigned`` / ``extended`` / ``converted`` / ``claimed`` are arrivals
  (``to_abbr`` only); ``waived`` is a departure (``from_abbr`` only).
* :func:`annotate` joins the draft board (rank, position, VORP, ADP) and the current roster snapshot, and adds a
  ``context`` string: the best-ranked teammates at the destination and whether the mover's position is crowded.
* :func:`digest` renders the lines a person reads.

**What this is not.** It is descriptive. ADR 0010 and 0011 tested "a player changed team" and "a star arrived / left" as
projection inputs against ten real seasons and found no lift (0011 measurably hurt), so nothing here changes a projection
or a rank: it tells you a move happened, how much the board cares about the player, and who else is in the room.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.ingest.espn_transactions import ESPN_TEAMS

RELEVANT_RANK = 180          # inside roughly the top 13 rounds of a 13-team draft: worth reading about
STAR_RANK = 60
CROWD_TOP = 8                # a position is "crowded" when this many of a team's best teammates share it
ARRIVAL_KINDS = ("signed", "resigned", "extended", "converted", "claimed")
EVENT_COLUMNS = ["txn_date", "kind", "person", "player_id", "from_abbr", "to_abbr", "detail", "first_seen", "description"]


def _abbr_lookup(_ledger: pd.DataFrame | None = None) -> dict[int, str]:
    """team_id -> NBA abbreviation for all 30 teams (a counterparty need not have a row of its own in the ledger)."""
    return {tid: abbr for tid, abbr in ESPN_TEAMS.values()}


def events(ledger: pd.DataFrame) -> pd.DataFrame:
    """One row per player move (see the module docstring). Staff rows and ``other`` rows are not included."""
    p = ledger[(ledger["subject_type"] == "player") & (ledger["kind"] != "other") & (ledger["person"] != "")].copy()
    if p.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    abbr = _abbr_lookup(ledger)
    p["other_abbr"] = p["other_team_id"].map(lambda v: abbr.get(int(v)) if pd.notna(v) else None)
    rows: list[dict] = []
    trades = p[p["kind"].isin(["trade_in", "trade_out"])]
    # one player-move per (date, person): the receiver's row names the destination, the sender's row the origin
    for (day, person), g in trades.groupby(["txn_date", "person"], sort=False):
        dest = g[g["kind"] == "trade_in"]
        orig = g[g["kind"] == "trade_out"]
        to_abbr = dest["team_abbr"].iloc[0] if len(dest) else (orig["other_abbr"].dropna().iloc[0] if len(orig) and orig["other_abbr"].notna().any() else None)
        from_abbr = orig["team_abbr"].iloc[0] if len(orig) else (dest["other_abbr"].dropna().iloc[0] if len(dest) and dest["other_abbr"].notna().any() else None)
        pid = g["player_id"].dropna()
        rows.append({"txn_date": day, "kind": "trade", "person": person, "player_id": pid.iloc[0] if len(pid) else pd.NA,
                     "from_abbr": from_abbr, "to_abbr": to_abbr, "detail": "", "first_seen": g["first_seen"].min(),
                     "description": g["description"].iloc[0]})
    for r in p[~p["kind"].isin(["trade_in", "trade_out"])].itertuples(index=False):
        arrival = r.kind in ARRIVAL_KINDS
        rows.append({"txn_date": r.txn_date, "kind": r.kind, "person": r.person, "player_id": r.player_id,
                     "from_abbr": None if arrival else r.team_abbr, "to_abbr": r.team_abbr if arrival else None,
                     "detail": r.detail, "first_seen": r.first_seen, "description": r.description})
    out = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    out["player_id"] = pd.array(pd.to_numeric(out["player_id"], errors="coerce"), dtype="Int64")
    return out.sort_values(["txn_date", "person"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def staff_events(ledger: pd.DataFrame) -> pd.DataFrame:
    """Coach and front-office moves: ``txn_date, team_abbr, kind, person, role, first_seen``."""
    s = ledger[ledger["subject_type"] == "staff"]
    out = s.rename(columns={"detail": "role"})[["txn_date", "team_abbr", "kind", "person", "role", "first_seen", "description"]]
    return out.sort_values(["txn_date", "team_abbr"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def _primary_pos(pos: object) -> str:
    return str(pos).split("-")[0] if isinstance(pos, str) and pos else ""


def annotate(ev: pd.DataFrame, board: pd.DataFrame | None, roster: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add ``rank``, ``position``, ``vorp``, ``adp``, ``relevant`` and ``context`` to ``events`` output.

    ``board``: the draft board (``player_id, rank, name, position, vorp, adp``). ``roster``: the newest roster snapshot
    (``player_id, team_abbr``); with it, ``context`` lists the destination's best-ranked teammates.
    """
    out = ev.copy()
    for c in ("rank", "vorp", "adp"):
        out[c] = float("nan")
    out["position"] = ""
    out["context"] = ""
    out["relevant"] = False
    if board is None or board.empty or out.empty:
        return out
    b = board.drop_duplicates("player_id").set_index("player_id")
    pid = out["player_id"].astype("float64")
    for c in ("rank", "vorp", "adp"):
        if c in b.columns:
            out[c] = pid.map(b[c].astype("float64"))
    if "position" in b.columns:
        out["position"] = pid.map(b["position"]).fillna("").astype(str)
    out["relevant"] = out["rank"].le(RELEVANT_RANK)

    if roster is not None and len(roster) and {"name", "position"} <= set(board.columns):
        ranked = roster[["player_id", "team_abbr"]].drop_duplicates("player_id").merge(
            board[["player_id", "rank", "name", "position"]].drop_duplicates("player_id"), on="player_id", how="inner")
        by_team = {t: g.sort_values("rank") for t, g in ranked.groupby("team_abbr")}
        ctx = []
        for r in out.itertuples(index=False):
            if not r.relevant:
                ctx.append("")
                continue
            parts = []
            dest = r.to_abbr if isinstance(r.to_abbr, str) else None
            orig = r.from_abbr if isinstance(r.from_abbr, str) else None
            if dest in by_team:
                mates = by_team[dest]
                mates = mates[mates["player_id"] != r.player_id]
                top = mates.head(CROWD_TOP)
                same = int((top["position"].map(_primary_pos) == _primary_pos(r.position)).sum()) if r.position else 0
                names = ", ".join(f"{m.name} (#{int(m.rank)})" for m in mates.head(3).itertuples(index=False))
                crowd = f"; {same} of the next {len(top)} best teammates share his position" if same >= 3 else ""
                if names:
                    parts.append(f"{dest} best teammates: {names}{crowd}")
            if orig in by_team:
                left = by_team[orig]
                left = left[left["player_id"] != r.player_id].head(3)
                if len(left):
                    parts.append(f"left behind at {orig}: " + ", ".join(f"{m.name} (#{int(m.rank)})" for m in left.itertuples(index=False)))
            ctx.append(" | ".join(parts))
        out["context"] = ctx
    return out


def _fmt_player(r) -> str:
    tags = [x for x in (r.position, "unranked" if pd.isna(r.rank) else f"#{int(r.rank)}") if x]
    return f"{r.person} ({', '.join(tags)})"


def line(r) -> str:
    """One human line for one annotated event row."""
    who = _fmt_player(r)
    d = f"{pd.Timestamp(r.txn_date):%b %d}"
    if r.kind == "trade":
        move = f"{r.from_abbr if isinstance(r.from_abbr, str) else '?'} -> {r.to_abbr if isinstance(r.to_abbr, str) else '?'}"
        return f"{d}  TRADE     {who}: {move}"
    label = {"signed": "SIGNED", "resigned": "RE-SIGNED", "extended": "EXTENDED", "converted": "CONVERTED",
             "claimed": "CLAIMED", "waived": "WAIVED"}.get(r.kind, r.kind.upper())
    team = f"to {r.to_abbr}" if isinstance(r.to_abbr, str) else f"by {r.from_abbr}"
    extra = f" [{r.detail.replace('_', ' ')}]" if r.detail else ""
    return f"{d}  {label:<9} {who}: {team}{extra}"


def digest(annotated: pd.DataFrame, *, relevant_only: bool = True, limit: int | None = 60,
           include_context: bool = True) -> list[str]:
    """Lines for the moves worth reading, best-ranked first within each day; unranked moves are counted, not listed,
    when ``relevant_only`` (the many camp signings and waivers of fringe players)."""
    if annotated.empty:
        return []
    frame = annotated[annotated["relevant"]] if relevant_only else annotated
    lines: list[str] = []
    for _day, g in frame.groupby("txn_date", sort=False):
        for r in g.sort_values(["rank", "person"], na_position="last").itertuples(index=False):
            lines.append(line(r))
            if include_context and r.context:
                lines.append(f"            {r.context}")
    hidden = len(annotated) - len(frame)
    if limit is not None and len(lines) > limit:
        lines = lines[:limit] + [f"... and {len(lines) - limit} more lines"]
    if relevant_only and hidden:
        lines.append(f"({hidden} other moves involve players outside the board's top {RELEVANT_RANK} or unranked)")
    return lines


def filter_events(annotated: pd.DataFrame, *, teams: Iterable[str] | None = None, kinds: Iterable[str] | None = None,
                  player: str | None = None) -> pd.DataFrame:
    out = annotated
    if teams:
        t = {x.upper() for x in teams}
        out = out[out["from_abbr"].isin(t) | out["to_abbr"].isin(t)]
    if kinds:
        out = out[out["kind"].isin(set(kinds))]
    if player:
        out = out[out["person"].str.contains(player, case=False, regex=False)]
    return out
