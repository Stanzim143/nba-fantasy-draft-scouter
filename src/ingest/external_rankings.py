"""Shared data contract for the two external-rankings snapshot parsers (Yahoo, FantasyPros) — see
``docs/adr/0028-external-rankings-compare.md``.

Unlike ``espn_adp.py``, these two sources are **one-off manual exports** the user downloads from
each site's UI, not a live HTTP API: there is no caching HTTP client here, just a file parser per
source. Both parsers (``src.ingest.yahoo_rankings``, ``src.ingest.fantasypros_rankings``) produce a
frame shaped exactly like :data:`RESOLVED_COLUMNS` below, built by calling :func:`resolve_players`
on their own source-specific parse step. That shared shape is what
``src.value.rankings_compare`` and the app page consume, so neither has to know which source it
came from beyond the ``source`` column.

Name resolution reuses ``src.ingest.id_map`` exactly as ``espn_adp.py`` does (exact -> alias ->
fuzzy, never a silent guess): nothing here reinvents name matching.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.ingest.id_map import MatchReport, match_players

#: Column order every resolved external-rankings frame has, regardless of source.
#:
#: ``adp`` and ``ecr_vs_adp`` are deliberately two different columns, not one generic "adp": Yahoo
#: exports a real average-draft-position value (an overall pick number), while FantasyPros' CSV
#: exports no raw ADP at all, only "ECR VS. ADP" -- a signed *delta* between FantasyPros' own expert
#: consensus rank and the ADP it was compared against. Conflating the two under one name would
#: silently imply they're the same quantity. Yahoo rows always have ``ecr_vs_adp`` null; FantasyPros
#: rows always have ``adp`` null.
RESOLVED_COLUMNS: tuple[str, ...] = (
    "source", "source_id", "source_name_raw", "name_parsed", "team", "positions",
    "status_tag", "ext_rank", "adp", "ecr_vs_adp", "player_id", "match_method", "confidence", "matched",
)

#: match_method values a resolved frame can carry (superset of id_map's: id_map never marks
#: "ambiguous" in its returned frame, since ambiguous rows are excluded and only reported -- we
#: fold them back in here as an explicit method value rather than silently lumping them in with
#: "unmatched", so the two failure modes stay distinguishable in the UI and the report).
MATCH_METHODS = ("exact", "normalized", "fuzzy", "ambiguous", "unmatched")


@dataclass
class ResolveResult:
    frame: pd.DataFrame           # RESOLVED_COLUMNS shape, one row per input row (never dropped)
    report: MatchReport           # id_map's own match report, for the CLI/UI to summarize


def resolve_players(parsed: pd.DataFrame, players: pd.DataFrame, *, source: str) -> ResolveResult:
    """Resolve ``parsed`` (one row per external-source player; see each parser's docstring for its
    required input columns) onto the canonical NBA ``player_id`` via ``id_map.match_players``, and
    return a frame in :data:`RESOLVED_COLUMNS` shape with exactly one output row per input row --
    unlike ``id_map.match_players`` itself (which returns only the matched subset), nothing here is
    ever silently dropped: an unmatched or ambiguous name still gets a row, with ``player_id`` null
    and ``matched`` false.

    ``parsed`` must have: ``source_id, source_name_raw, name_parsed, team, positions, status_tag,
    ext_rank, adp, ecr_vs_adp`` (the source that doesn't produce one of ``adp``/``ecr_vs_adp`` still
    supplies the column, filled with null -- see :data:`RESOLVED_COLUMNS`).
    """
    required = {"source_id", "source_name_raw", "name_parsed", "team", "positions", "status_tag",
                "ext_rank", "adp", "ecr_vs_adp"}
    missing = required - set(parsed.columns)
    if missing:
        raise ValueError(f"resolve_players: parsed frame missing columns {sorted(missing)}")

    match_input = parsed[["source_id", "name_parsed"]].rename(columns={"name_parsed": "name"})
    matched_frame, report = match_players(match_input, players, source=source,
                                          id_col="source_id", name_col="name", year_col=None)

    ambiguous_ids = {a["source_id"] for a in report.ambiguous}

    out = parsed.copy()
    out["source"] = source
    m = matched_frame.set_index("source_id") if len(matched_frame) else None
    if m is not None:
        out["player_id"] = out["source_id"].map(m["player_id"]).astype("Int64")
        out["match_method"] = out["source_id"].map(m["match_method"])
        out["confidence"] = out["source_id"].map(m["confidence"]).astype("float64")
    else:
        out["player_id"] = pd.array([pd.NA] * len(out), dtype="Int64")
        out["match_method"] = pd.NA
        out["confidence"] = pd.NA
    out["match_method"] = out["match_method"].where(
        out["match_method"].notna(),
        out["source_id"].isin(ambiguous_ids).map({True: "ambiguous", False: "unmatched"}),
    )
    out["matched"] = out["player_id"].notna()
    return ResolveResult(frame=out[list(RESOLVED_COLUMNS)].reset_index(drop=True), report=report)


def unresolved_rows(resolved: pd.DataFrame) -> pd.DataFrame:
    """Every row this source could not resolve (unmatched or ambiguous) -- never silently dropped,
    always available for an honest report."""
    return resolved[~resolved["matched"]].copy()
