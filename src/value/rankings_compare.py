"""Compare this project's own draft board rank against external rankings snapshots (Yahoo,
FantasyPros) — see ``docs/adr/0028-external-rankings-compare.md``.

Pure, unit-tested join logic; no Streamlit import here (the app page in
``src/app/pages/external_rankings.py`` wires this to widgets). Consumes:

* ``our_board``: this project's own board, at minimum ``player_id, name, rank`` (e.g. the output of
  ``src.value.board.build_board`` / ``src.app.loader.load_board``, or a saved board CSV read back in).
* ``resolved_by_source``: a mapping ``{"yahoo": <frame>, "fantasypros": <frame>}`` of frames already
  produced by ``src.ingest.yahoo_rankings.load_yahoo_rankings`` /
  ``src.ingest.fantasypros_rankings.load_fantasypros_rankings`` -- i.e. already in
  ``src.ingest.external_rankings.RESOLVED_COLUMNS`` shape, matched and unmatched rows both present.

Sign convention (the one fact every consumer of this module must get right)
-----------------------------------------------------------------------------
``rank_delta_<source> = our_rank - <source>_rank``.

**Positive** means the external source ranks the player *earlier* (a lower, more favorable rank
number) than we do -- i.e. the market/expert consensus is higher on this player than our board.
**Negative** means we rank the player earlier than the external source does -- our board is higher
on this player. This is the same direction ``src.value.board``'s own ``adp_gap`` column already
uses (``adp_gap = adp - rank``, positive = model ranks earlier than market drafts), kept consistent
here across both external sources so a reader doesn't have to learn two conventions.

Coverage is honestly asymmetric by design, not a data error
-------------------------------------------------------------
FantasyPros' CSV only carries its top ~300 players; Yahoo's carries ~686; this project's own board is
typically the full projected universe (often 400-600+). A player on our board absent from
FantasyPros is not an anomaly -- it just means that player fell outside FantasyPros' published
range. ``on_yahoo`` / ``on_fantasypros`` / ``on_our_board`` flags make this visible rather than
letting a null rank look like a bug.

Advisory disagreement flag (ADR 0029)
--------------------------------------
:func:`flag_disagreements` adds ``rankings_disagreement`` / ``rankings_disagreement_direction`` /
``rankings_disagreement_text`` -- a display-only flag for players where ``max_abs_rank_delta`` is
large, in *either* direction (our board well ahead of the external consensus, or well behind it).
``build_comparison`` calls it automatically with the default threshold, so every comparison frame
already carries these columns; nothing here writes ``our_rank`` or any board projection/VORP/rank
column, and the flag never changes what ``build_comparison`` or ``biggest_disagreements`` already
compute. See ``docs/adr/0029-rankings-disagreement-flag.md`` for how :data:`DISAGREEMENT_THRESHOLD`
was chosen and how the flag is surfaced (advisory only) on the draft board itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SOURCES: tuple[str, ...] = ("yahoo", "fantasypros")

#: Rank-spot threshold (either direction) for the advisory "sharp disagreement" flag (ADR 0029).
#: Chosen from the real 2026-09-28 Yahoo/FantasyPros snapshots restricted to the realistic draft
#: pool (our_rank <= 150, a 13-team/13-round league): median |rank_delta| there was ~37 spots and
#: the 75th percentile ~66; 50 sits above the typical disagreement without only firing on the
#: extreme tail (see the ADR for the full distribution and the caveat that this is not backtested
#: against any outcome -- it is a judgement about what counts as "sharp", not a fitted number).
DISAGREEMENT_THRESHOLD: float = 50.0

#: Columns the comparison table always has, before any sorting/filtering by the caller.
COMPARISON_COLUMNS: tuple[str, ...] = (
    "player_id", "name", "our_rank", "on_our_board",
    "yahoo_rank", "yahoo_team", "yahoo_positions", "yahoo_status_tag", "yahoo_adp", "on_yahoo",
    "rank_delta_yahoo",
    "fantasypros_rank", "fantasypros_team", "fantasypros_positions", "fantasypros_status_tag",
    "fantasypros_ecr_vs_adp", "on_fantasypros", "rank_delta_fantasypros",
    "max_abs_rank_delta",
    "rankings_disagreement", "rankings_disagreement_direction", "rankings_disagreement_text",
)

#: "ext_" prefixed (not plain "team"/"positions"/"status_tag") so these have their own glossary
#: entries distinct from the board's "team" (current NBA team) and any other column of the same
#: plain name elsewhere in the app -- src.app.glossary.COLUMN_GLOSSARY is a single flat namespace.
UNMATCHED_COLUMNS: tuple[str, ...] = (
    "source", "source_name_raw", "ext_team", "ext_positions", "ext_status_tag", "ext_rank", "match_method",
)


def _source_matched(resolved: pd.DataFrame, source: str) -> pd.DataFrame:
    """The matched subset of one source's resolved frame, one row per ``player_id`` (a source
    should never carry duplicate players, but a defensive drop keeps a malformed input from
    silently doubling rows in the merge)."""
    m = resolved[resolved["matched"]].copy()
    m["player_id"] = m["player_id"].astype("int64")
    return m.drop_duplicates("player_id", keep="first")


def build_comparison(our_board: pd.DataFrame, resolved_by_source: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join ``our_board`` against every source in ``resolved_by_source`` on ``player_id``.

    ``our_board`` needs ``player_id, name, rank`` (extra columns are ignored). Every source key must
    be one of :data:`SOURCES`. The result has one row per player who appears on **any** side (our
    board or a matched external row) -- nobody is dropped for appearing on only one side; see
    ``on_our_board``/``on_yahoo``/``on_fantasypros``.
    """
    unknown = set(resolved_by_source) - set(SOURCES)
    if unknown:
        raise ValueError(f"unknown source(s) in resolved_by_source: {sorted(unknown)}")
    for col in ("player_id", "name", "rank"):
        if col not in our_board.columns:
            raise ValueError(f"our_board is missing required column {col!r}")

    base = our_board[["player_id", "name", "rank"]].drop_duplicates("player_id", keep="first").copy()
    base["player_id"] = base["player_id"].astype("int64")
    base = base.rename(columns={"rank": "our_rank"})

    out = base
    for source in SOURCES:
        resolved = resolved_by_source.get(source)
        prefix = source
        if resolved is None or resolved.empty:
            out[f"{prefix}_rank"] = np.nan
            out[f"{prefix}_team"] = None
            out[f"{prefix}_positions"] = None
            out[f"{prefix}_status_tag"] = None
            if source == "yahoo":
                out["yahoo_adp"] = np.nan
            else:
                out["fantasypros_ecr_vs_adp"] = np.nan
            continue
        m = _source_matched(resolved, source)
        side = m[["player_id", "ext_rank", "team", "positions", "status_tag"]].rename(columns={
            "ext_rank": f"{prefix}_rank", "team": f"{prefix}_team",
            "positions": f"{prefix}_positions", "status_tag": f"{prefix}_status_tag",
        })
        if source == "yahoo":
            side["yahoo_adp"] = m["adp"].to_numpy()
        else:
            side["fantasypros_ecr_vs_adp"] = m["ecr_vs_adp"].to_numpy()
        out = out.merge(side, on="player_id", how="outer")

    # Names: prefer our_board's, fall back to whichever source has one for a player we don't board.
    if "name" not in out.columns:
        out["name"] = None
    for source in SOURCES:
        resolved = resolved_by_source.get(source)
        if resolved is None or resolved.empty:
            continue
        m = _source_matched(resolved, source).set_index("player_id")["name_parsed"]
        missing = out["name"].isna()
        if missing.any():
            out.loc[missing, "name"] = out.loc[missing, "player_id"].map(m)

    out["on_our_board"] = out["our_rank"].notna()
    out["on_yahoo"] = out["yahoo_rank"].notna()
    out["on_fantasypros"] = out["fantasypros_rank"].notna()
    out["rank_delta_yahoo"] = out["our_rank"] - out["yahoo_rank"]
    out["rank_delta_fantasypros"] = out["our_rank"] - out["fantasypros_rank"]
    out["max_abs_rank_delta"] = out[["rank_delta_yahoo", "rank_delta_fantasypros"]].abs().max(axis=1, skipna=True)
    out = flag_disagreements(out)

    out = out.sort_values(["our_rank", "player_id"], na_position="last", kind="mergesort").reset_index(drop=True)
    return out[list(COMPARISON_COLUMNS)]


def flag_disagreements(comparison: pd.DataFrame, threshold: float = DISAGREEMENT_THRESHOLD) -> pd.DataFrame:
    """Advisory "our board disagrees sharply with the external consensus" flag (ADR 0029).

    ``comparison`` needs ``our_rank``, ``yahoo_rank``, ``fantasypros_rank``, ``rank_delta_yahoo``,
    ``rank_delta_fantasypros`` and ``max_abs_rank_delta`` (i.e. it is, or came from,
    :func:`build_comparison`'s own output -- that function already calls this and includes the
    result in :data:`COMPARISON_COLUMNS`, so most callers never need to call this directly).

    Adds three columns, blank/``False`` for every player below ``threshold`` or with no external
    rank at all:

    ``rankings_disagreement``            ``True`` when ``max_abs_rank_delta >= threshold``.
    ``rankings_disagreement_direction``  ``"market_favors"`` (an external source ranks him earlier
                                          than we do) or ``"we_favor"`` (we rank him earlier than
                                          that source) -- named for whichever of
                                          ``rank_delta_yahoo`` / ``rank_delta_fantasypros`` has the
                                          larger magnitude (ties go to Yahoo, the wider-coverage
                                          source); ``""`` when not flagged.
    ``rankings_disagreement_text``       a one-line human-readable sentence naming the driving
                                          source and both ranks, e.g. ``"fantasypros ranks him 62
                                          spots earlier (#41) than our board (#103); external
                                          consensus is higher on him than we are"``.

    Never writes ``our_rank`` or any of the *_rank / rank_delta_* / max_abs_rank_delta columns
    themselves -- this only reads them to decide the flag. Not a projection or a rank: see the
    module docstring and ``docs/adr/0029-rankings-disagreement-flag.md``.
    """
    out = comparison.copy()
    delta_yahoo = out["rank_delta_yahoo"].to_numpy(float)
    delta_fp = out["rank_delta_fantasypros"].to_numpy(float)
    abs_yahoo = np.abs(delta_yahoo)
    abs_fp = np.abs(delta_fp)
    # Which source drives max_abs_rank_delta: the larger |delta|; NaN loses to any real number, and
    # a tie (both present and equal, or both absent) is broken to Yahoo, the wider-coverage source.
    use_fp = np.nan_to_num(abs_fp, nan=-1.0) > np.nan_to_num(abs_yahoo, nan=-1.0)
    driving_delta = np.where(use_fp, delta_fp, delta_yahoo)
    driving_source = np.where(use_fp, "fantasypros", "yahoo")
    driving_rank = np.where(use_fp, out["fantasypros_rank"].to_numpy(float), out["yahoo_rank"].to_numpy(float))

    max_abs = out["max_abs_rank_delta"].to_numpy(float)
    flagged = np.nan_to_num(max_abs, nan=-1.0) >= threshold

    direction = np.full(len(out), "", dtype=object)
    direction[flagged & (driving_delta > 0)] = "market_favors"
    direction[flagged & (driving_delta < 0)] = "we_favor"

    our_rank = out["our_rank"].to_numpy(float)
    text = np.full(len(out), "", dtype=object)
    for i in np.flatnonzero(flagged):
        src, ext_r, our_r, delta = driving_source[i], driving_rank[i], our_rank[i], driving_delta[i]
        if not np.isfinite(ext_r) or not np.isfinite(our_r):
            continue
        our_i, ext_i, mag = int(our_r), int(ext_r), abs(int(delta))
        if delta > 0:
            text[i] = (f"{src} ranks him {mag} spots earlier (#{ext_i}) than our board (#{our_i}); "
                       f"external consensus is higher on him than we are")
        else:
            text[i] = (f"our board ranks him {mag} spots earlier (#{our_i}) than {src} (#{ext_i}); "
                       f"we are higher on him than external consensus")

    out["rankings_disagreement"] = flagged
    out["rankings_disagreement_direction"] = direction
    out["rankings_disagreement_text"] = text
    return out


def unmatched_report(resolved_by_source: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every row across every source that could not be resolved onto a ``player_id`` (unmatched or
    ambiguous) -- never silently dropped, always surfaced for review. Sorted by source then
    ``ext_rank`` so the most fantasy-relevant misses show first."""
    frames = []
    for source, resolved in resolved_by_source.items():
        if resolved is None or resolved.empty:
            continue
        miss = resolved[~resolved["matched"]].copy()
        if miss.empty:
            continue
        miss["source"] = source
        miss = miss.rename(columns={"team": "ext_team", "positions": "ext_positions",
                                    "status_tag": "ext_status_tag"})
        frames.append(miss[list(UNMATCHED_COLUMNS)])
    if not frames:
        return pd.DataFrame(columns=UNMATCHED_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["source", "ext_rank"], kind="mergesort").reset_index(drop=True)


def biggest_disagreements(comparison: pd.DataFrame, *, n: int = 25, source: str | None = None) -> pd.DataFrame:
    """The ``n`` rows with the largest rank disagreement, either direction.

    ``source`` restricts to one source's ``rank_delta_<source>`` column; ``None`` (default) uses
    ``max_abs_rank_delta`` (the bigger of the two sources' disagreements, when both are present).
    """
    col = "max_abs_rank_delta" if source is None else f"rank_delta_{source}"
    if col not in comparison.columns:
        raise ValueError(f"unknown source {source!r}")
    sortable = comparison.assign(_abs=comparison[col].abs()) if source else comparison.assign(_abs=comparison[col])
    return sortable.dropna(subset=[col]).sort_values("_abs", ascending=False).drop(columns="_abs").head(n)
