"""Wikipedia team-season "Transactions" ingest: pull (cached, polite) each team-season page's raw
wikitext, parse its ``===Trades===`` / ``====Additions====`` / ``====Subtractions====`` tables into
dated, per-team arrival/departure rows, map player names onto the canonical NBA ``player_id``, and
write the ad-hoc ``team_transactions`` table plus ``"wikipedia_transactions"`` rows into
``player_id_map``.

    python -m src.ingest.wiki_transactions --seasons 2015-16:2025-26 [--offline] [--refresh]

See ``docs/adr/0011-transactions-layer.md`` for the full design and the reasoning behind every
decision below -- this module implements that ADR's D1-D3 exactly (D4/D5, the feature and model
layers that consume ``team_transactions``, live elsewhere). What it produces:

* ``<data_dir>/processed/team_transactions.parquet`` -- an ad-hoc table (NOT part of
  ``src/contracts.py``'s ``TABLES``, same precedent as ``adp.parquet``/``espn_adp.py``), columns
  ``season, team_id, player_id, direction, source_kind, txn_date``. Only rows whose player name
  resolved to a real ``player_id`` are kept; everything else (draft picks, cash considerations,
  unmatched names) is silently dropped, never guessed.
* ``<data_dir>/processed/player_id_map.parquet`` -- ``"wikipedia_transactions"`` rows added
  alongside whatever other sources (``"espn"``, ...) already live there, through
  ``src.ingest.id_map.match_players`` unchanged.

Parsing (ADR 0011 D2): row/cell-aware, not a flat "grab every [[...]] link" scrape. Each subsection's
wikitable is split into ``|-``-delimited rows, then into cells (lines starting with ``|`` within a
row, with the usual ``| <attrs> | <content>`` wiki-cell-attribute prefix stripped). ``Additions``/
``Subtractions`` rows are single-team perspective: the player is the *first* wikilink anywhere in
the row (robust to whichever column position it happens to sit in -- verified to vary across the
five real pages this was built against, see fixtures); the date comes *only* from that row's first
``<ref ...date=...>`` citation, never from a "Signed"/date-looking column, which real pages
sometimes fill with contract-dollar text instead (the whole reason this project doesn't trust
column headers here). ``Trades`` rows are two-sided (``To Team A<hr />...`` / ``To Team B<hr />...``);
the side whose team name (bold or not, linked or not) matches the page's own team is "in", every
other side present in that same row is "out" -- a row with no side matching the page's own team
(the label/other-leg rows of a rare three-team trade) is skipped rather than guessed at.

License/rate-limit posture: see ADR 0011 -- CC BY-SA 4.0, read-only ``action=raw``, descriptive UA,
>= 1s between requests (this project's own extra-conservative floor; Wikipedia's own policy is far
more permissive), nothing from the source is committed (only the small set of derived facts is
written to the gitignored local data directory).
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.contracts import ContractError, data_dir, season_start, seasons_between
from src.ingest.http_cache import CachedHttpClient, HttpCacheError, default_cache_dir
from src.ingest.id_map import MatchReport, match_players
from src.store import read_table, table_exists, write_table

SOURCE = "wikipedia_transactions"
CACHE_DIR_NAME = "wikipedia"
UA = "nba-fantasy-2026-research/1.0 (personal, non-commercial research project; contact via GitHub repo)"

WIKI_API_URL = "https://en.wikipedia.org/w/index.php"

TEAM_TRANSACTIONS_COLUMNS = ["season", "team_id", "player_id", "direction", "source_kind", "txn_date"]

REPORT_NAME = "wiki_transactions_report.json"


class WikiTransactionsError(RuntimeError):
    """The fetched/parsed/matched data is unusable in a way the CLI should stop and report."""


# --------------------------------------------------------------------------- team mapping

# Canonical NBA team_id -> the exact team name used in that team's Wikipedia season-page title.
# Stable for the whole 2015-16..2025-26 window this project covers (no relocations in that span).
TEAM_WIKI_NAMES: dict[int, str] = {
    1610612737: "Atlanta Hawks", 1610612751: "Brooklyn Nets", 1610612738: "Boston Celtics",
    1610612766: "Charlotte Hornets", 1610612741: "Chicago Bulls", 1610612739: "Cleveland Cavaliers",
    1610612742: "Dallas Mavericks", 1610612743: "Denver Nuggets", 1610612765: "Detroit Pistons",
    1610612744: "Golden State Warriors", 1610612745: "Houston Rockets", 1610612754: "Indiana Pacers",
    1610612746: "Los Angeles Clippers", 1610612747: "Los Angeles Lakers", 1610612763: "Memphis Grizzlies",
    1610612748: "Miami Heat", 1610612749: "Milwaukee Bucks", 1610612750: "Minnesota Timberwolves",
    1610612740: "New Orleans Pelicans", 1610612752: "New York Knicks", 1610612760: "Oklahoma City Thunder",
    1610612753: "Orlando Magic", 1610612755: "Philadelphia 76ers", 1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers", 1610612758: "Sacramento Kings", 1610612759: "San Antonio Spurs",
    1610612761: "Toronto Raptors", 1610612762: "Utah Jazz", 1610612764: "Washington Wizards",
}


def team_season_page_title(team_id: int, season: str) -> str:
    """'2016-17', GSW -> '2016–17 Golden State Warriors season' (EN DASH, per the real page titles)."""
    if team_id not in TEAM_WIKI_NAMES:
        raise WikiTransactionsError(f"unknown team_id {team_id!r}; not in TEAM_WIKI_NAMES")
    y = season_start(season)
    return f"{y}–{str(y + 1)[-2:]} {TEAM_WIKI_NAMES[team_id]} season"


# --------------------------------------------------------------------------- season parsing (same grammar as espn_adp)

_SEASON_TOKEN = re.compile(r"^\d{4}-\d{2}$")


def _one_season(token: str) -> str:
    token = token.strip()
    if not _SEASON_TOKEN.match(token):
        raise ValueError(f"malformed season {token!r}; expected like '2023-24'")
    season_start(token)
    return token


def parse_seasons(spec: str) -> list[str]:
    """'A:B' range, single season, or comma list -- same grammar as ``espn_adp.parse_seasons``."""
    if not spec or not spec.strip():
        raise ValueError("no seasons given; expected like '2015-16:2025-26'")
    out: set[str] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty item in season list {spec!r}")
        if ":" in part:
            a, _, b = part.partition(":")
            first, last = season_start(_one_season(a)), season_start(_one_season(b))
            if last < first:
                raise ValueError(f"reversed season range {part!r}")
            out.update(seasons_between(first, last))
        else:
            out.add(_one_season(part))
    return sorted(out, key=season_start)


# --------------------------------------------------------------------------- wikitext section helpers

_HEADING_RE = re.compile(r"(?m)^[ \t]*(={2,6})\s*(.+?)\s*\1[ \t]*$")


def _headings(text: str) -> list[tuple[int, str, int, int]]:
    """Every wiki heading line in ``text``: ``(level, title, content_start, heading_line_start)``."""
    out = []
    for m in _HEADING_RE.finditer(text):
        out.append((len(m.group(1)), m.group(2).strip(), m.end(), m.start()))
    return out


def _section_text(text: str, level: int | tuple[int, ...], title: str) -> str:
    """Text strictly between a heading ``level``/``title`` (case-insensitive, whitespace-tolerant --
    real pages spell the same heading both ``==Transactions==`` and ``== Transactions ==``) and the
    next heading at that level or shallower. ``level`` may be a tuple to accept more than one
    nesting depth: verified necessary in the wild -- some team-season pages put ``Additions``/
    ``Subtractions``/``Re-signed`` one level shallower (``===Re-signed===``, a sibling of
    ``===Free agency===``) instead of nested a level deeper (``====Re-signed====``) as every one of
    the five saved fixtures happens to do; searching by title directly within the whole
    ``Transactions`` section (not requiring a matched ``Free agency`` wrapper first) sidesteps that
    variance entirely rather than silently truncating the section to nothing. Returns '' if the
    heading is not present at all: a missing subsection is normal (not every page has every
    subsection) and must never crash parsing.
    """
    levels = (level,) if isinstance(level, int) else tuple(level)
    headings = _headings(text)
    title_norm = title.strip().lower()
    for i, (lvl, t, content_start, _line_start) in enumerate(headings):
        if lvl in levels and t.lower() == title_norm:
            end = len(text)
            for lvl2, _t2, _cs2, line_start2 in headings[i + 1:]:
                if lvl2 <= lvl:
                    end = line_start2
                    break
            return text[content_start:end]
    return ""


# --------------------------------------------------------------------------- wikitable row/cell splitting

_ROW_SEP_RE = re.compile(r"(?m)^\|-.*$")
_TABLE_END_RE = re.compile(r"(?m)^\|\}\s*$")
_ATTR_KEY_RE = re.compile(r"[A-Za-z][A-Za-z-]*\s*=")


def _split_rows(table_text: str) -> list[str]:
    """Split a section's wikitable text into per-``|-``-row raw blocks (dropping the preamble before
    the first row separator: the ``{|`` opener and any header ``!`` cells)."""
    parts = _ROW_SEP_RE.split(table_text)
    rows = []
    for chunk in parts[1:]:
        m = _TABLE_END_RE.search(chunk)
        rows.append(chunk[:m.start()] if m else chunk)
    return rows


def _split_cells(row_text: str) -> list[str]:
    """Split one row's raw text into cells: a line starting with ``|`` begins a new cell; any
    following line that does *not* start with ``|`` is a continuation of the current cell (wiki
    cells routinely span several physical lines, e.g. a multi-asset trade side)."""
    cells: list[str] = []
    cur: list[str] | None = None
    for line in row_text.split("\n"):
        if line.startswith("|"):
            if cur is not None:
                cells.append("\n".join(cur))
            cur = [line[1:]]
        elif cur is not None:
            cur.append(line)
    if cur is not None:
        cells.append("\n".join(cur))
    return cells


def _strip_cell_attrs(cell: str) -> str:
    """Strip a leading ``style="..." | ...`` / ``rowspan=2 | ...`` wiki-cell-attribute prefix, if
    present -- but never touch a genuine ``|`` inside content (e.g. a ``{{cite web|...}}`` template's
    own pipes), which is why this only fires when the text *before* the first ``|`` looks like wiki
    attributes (an ``attr=`` pattern, or nothing at all) and contains no wikilink."""
    if "|" not in cell:
        return cell
    before, _, after = cell.partition("|")
    if "[[" in before:
        return cell
    if before.strip() == "" or _ATTR_KEY_RE.search(before):
        return after
    return cell


# --------------------------------------------------------------------------- wikitext -> plain-text helpers

WIKILINK_RE = re.compile(r"\[\[([^\[\]|]+)(?:\|([^\[\]]+))?\]\]")


def _extract_links(text: str) -> list[str]:
    """Every ``[[Target]]``/``[[Target|Display]]`` wikilink in ``text``, in order, as its display
    text (the human-readable name -- preferred over the link target, which often carries a
    disambiguator like ``(basketball)``)."""
    out = []
    for m in WIKILINK_RE.finditer(text):
        disp = m.group(2) if m.group(2) is not None else m.group(1)
        out.append(disp.strip())
    return out


def _strip_markup(text: str) -> str:
    """Reduce wikitext to plain text: drop citations/templates, resolve wikilinks to display text,
    drop bold/italic markup and any remaining HTML-ish tags. Used only to *detect* things (a plain
    date, a "To Team" side header) -- never to extract assets, which always comes from the raw
    wikilinks so no asset is ever silently mangled."""
    text = re.sub(r"<ref\b[^>]*?/>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<ref\b[^>]*>.*?</ref>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = WIKILINK_RE.sub(lambda m: m.group(2) or m.group(1), text)
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_DATE_PATTERNS = (
    re.compile(r"^[A-Za-z]+\.?\s+\d{1,2},\s*\d{4}$"),   # "June 23, 2016"
    re.compile(r"^\d{1,2}\s+[A-Za-z]+\s+\d{4}$"),        # "5 September 2023"
)


def _looks_like_date(plain: str) -> bool:
    plain = plain.strip()
    return any(p.match(plain) for p in _DATE_PATTERNS)


_REF_RE = re.compile(r"<ref\b[^>]*?/>|<ref\b[^>]*>(.*?)</ref>", re.IGNORECASE | re.DOTALL)
_CITATION_DATE_RE = re.compile(r"(?<![A-Za-z-])date\s*=\s*([^|{}<\n]+)", re.IGNORECASE)


def _first_citation_date(text: str) -> pd.Timestamp:
    """The row's first inline ``<ref>...{{cite ...|date=Month Day, Year|...}}...</ref>`` date, or
    ``NaT`` if none is present/parseable -- the *only* date source trusted for Additions/Subtractions
    (never a "Signed"/date-looking column's raw text, which real pages sometimes fill with
    contract-dollar text instead; see the module docstring and ADR 0011 D-"do not trust column
    headers")."""
    for m in _REF_RE.finditer(text):
        body = m.group(1) or ""
        dm = _CITATION_DATE_RE.search(body)
        if not dm:
            continue
        ts = pd.to_datetime(dm.group(1).strip(), errors="coerce")
        if pd.notna(ts):
            return ts
    return pd.NaT


def _normalize_team_name(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


# --------------------------------------------------------------------------- section parsers (pure: wikitext -> rows)

def _parse_single_team_rows(section_text: str, *, direction: str, source_kind: str) -> tuple[list[dict], int]:
    """``Additions``/``Subtractions`` rows: single-team perspective. Player = first wikilink anywhere
    in the row (robust to the row's exact column layout, which varies across real pages -- see
    module docstring); date = the row's first citation date only, else NaT (row is kept, not
    dropped). A row with no wikilink at all cannot be interpreted as a player and is skipped."""
    rows_out: list[dict] = []
    n_no_player = 0
    for row_raw in _split_rows(section_text):
        if not _split_cells(row_raw):
            continue
        links = _extract_links(row_raw)
        if not links:
            n_no_player += 1
            continue
        rows_out.append({
            "wiki_name": links[0],
            "direction": direction,
            "source_kind": source_kind,
            "txn_date": _first_citation_date(row_raw),
        })
    return rows_out, n_no_player


def _parse_trades(trades_text: str, team_id: int) -> tuple[list[dict], dict]:
    """``Trades`` rows: two-sided (``To Team<hr />...``). The side matching the page's own team is
    "in"; every *other* side present in the SAME row is "out". A row with no side matching the
    page's own team (the connective/other-leg rows of a rare three-team trade, which would otherwise
    misattribute a leg that never touched this team) is skipped, not guessed at -- see module
    docstring and ADR 0011 D2."""
    rows_out: list[dict] = []
    counts = {"n_rows": 0, "n_no_own_side": 0, "n_rows_with_assets": 0}
    own_norm = _normalize_team_name(TEAM_WIKI_NAMES[team_id])

    for row_raw in _split_rows(trades_text):
        cells_raw = _split_cells(row_raw)
        if not cells_raw:
            continue
        counts["n_rows"] += 1
        cells = [_strip_cell_attrs(c) for c in cells_raw]

        sides: list[tuple[bool, str]] = []  # (is_own_side, body_text)
        for c in cells:
            if "<hr" not in c.lower():
                continue
            parts = re.split(r"<hr\s*/?>", c, maxsplit=1, flags=re.IGNORECASE)
            header, body = parts[0], (parts[1] if len(parts) > 1 else "")
            header_plain = _strip_markup(header)
            m = re.match(r"^to\s+(.+)$", header_plain, re.IGNORECASE | re.DOTALL)
            if not m:
                continue
            sides.append((_normalize_team_name(m.group(1)) == own_norm, body))

        own_bodies = [b for is_own, b in sides if is_own]
        other_bodies = [b for is_own, b in sides if not is_own]
        if not own_bodies:
            counts["n_no_own_side"] += 1
            continue

        txn_date = pd.NaT
        for c in cells:
            plain = _strip_markup(c)
            if _looks_like_date(plain):
                txn_date = pd.to_datetime(plain, errors="coerce")
                break
        if pd.isna(txn_date):
            txn_date = _first_citation_date(row_raw)

        got_assets = False
        for body in own_bodies:
            for name in _extract_links(body):
                rows_out.append({"wiki_name": name, "direction": "in", "source_kind": "trade", "txn_date": txn_date})
                got_assets = True
        for body in other_bodies:
            for name in _extract_links(body):
                rows_out.append({"wiki_name": name, "direction": "out", "source_kind": "trade", "txn_date": txn_date})
                got_assets = True
        if got_assets:
            counts["n_rows_with_assets"] += 1

    return rows_out, counts


@dataclass
class ParsedPage:
    team_id: int
    season: str
    rows: list[dict] = field(default_factory=list)
    trade_counts: dict = field(default_factory=dict)
    n_addition_no_player: int = 0
    n_subtraction_no_player: int = 0


def parse_team_season_wikitext(wikitext: str, team_id: int, season: str) -> ParsedPage:
    """Pure function: raw wikitext -> parsed transaction rows for one team-season. No network, no
    filesystem -- this is the piece the offline tests exercise directly against the real fixtures.
    A page missing the whole ``==Transactions==`` section, or any of its subsections, yields zero
    rows from that part rather than raising -- normal, not an error (see ``_section_text``)."""
    transactions_text = _section_text(wikitext, 2, "Transactions")
    trades_text = _section_text(transactions_text, 3, "Trades")
    # Additions/Subtractions/Re-signed are searched directly within the Transactions section (not
    # required to be found strictly inside a matched "Free agency" wrapper first) and at either
    # nesting depth (====, the depth of all five saved fixtures, or ===, verified present on other
    # real pages) -- see _section_text's docstring for why.
    additions_text = _section_text(transactions_text, (3, 4), "Additions")
    subtractions_text = _section_text(transactions_text, (3, 4), "Subtractions")
    # Note: "Re-signed" is deliberately never parsed -- a re-signed player never left, so it is
    # neither an arrival nor a departure (ADR 0011).

    trade_rows, trade_counts = _parse_trades(trades_text, team_id)
    addition_rows, n_add_no_player = _parse_single_team_rows(additions_text, direction="in", source_kind="addition")
    subtraction_rows, n_sub_no_player = _parse_single_team_rows(
        subtractions_text, direction="out", source_kind="subtraction")

    rows = trade_rows + addition_rows + subtraction_rows
    for r in rows:
        r["team_id"] = team_id
        r["season"] = season

    return ParsedPage(team_id=team_id, season=season, rows=rows, trade_counts=trade_counts,
                      n_addition_no_player=n_add_no_player, n_subtraction_no_player=n_sub_no_player)


# --------------------------------------------------------------------------- fetch

def fetch_team_season_wikitext(client: CachedHttpClient, team_id: int, season: str, *,
                               refresh: bool = False) -> str:
    title = team_season_page_title(team_id, season)
    return client.get_text("team_season", WIKI_API_URL, {"title": title, "action": "raw"},
                           headers={"User-Agent": UA}, refresh=refresh)


# --------------------------------------------------------------------------- pipeline

@dataclass
class TxnIngestResult:
    seasons: list[str]
    fetched: int = 0
    skipped_pages: list[dict] = field(default_factory=list)
    n_trade_rows: int = 0
    n_addition_rows: int = 0
    n_subtraction_rows: int = 0
    match_report: MatchReport | None = None
    written_rows: int = 0
    id_map_rows: int = 0
    network_requests: int = 0
    cache_hits: int = 0


def run_ingest(seasons: list[str], client: CachedHttpClient, base: Path | None = None, *,
              refresh: bool = False, log=print) -> TxnIngestResult:
    if not seasons:
        raise WikiTransactionsError("no seasons to ingest")
    base = base or data_dir()
    result = TxnIngestResult(seasons=list(seasons))

    all_rows: list[dict] = []
    for season in seasons:
        for team_id in sorted(TEAM_WIKI_NAMES):
            title = team_season_page_title(team_id, season)
            try:
                wikitext = fetch_team_season_wikitext(client, team_id, season, refresh=refresh)
            except HttpCacheError as exc:
                log(f"[{season}] {TEAM_WIKI_NAMES[team_id]}: fetch failed ({exc}); skipped")
                result.skipped_pages.append({"season": season, "team_id": team_id, "title": title,
                                             "reason": str(exc)})
                continue
            result.fetched += 1
            page = parse_team_season_wikitext(wikitext, team_id, season)
            all_rows.extend(page.rows)
            result.n_trade_rows += sum(1 for r in page.rows if r["source_kind"] == "trade")
            result.n_addition_rows += sum(1 for r in page.rows if r["source_kind"] == "addition")
            result.n_subtraction_rows += sum(1 for r in page.rows if r["source_kind"] == "subtraction")
            log(f"[{season}] {TEAM_WIKI_NAMES[team_id]}: {len(page.rows)} candidate rows "
                f"(trades: {page.trade_counts.get('n_rows_with_assets', 0)}/{page.trade_counts.get('n_rows', 0)} "
                f"rows w/ own-side assets; additions/subtractions rows w/o a player link: "
                f"{page.n_addition_no_player}/{page.n_subtraction_no_player})")

    universe_map: dict[tuple[str, int], dict] = {}
    for r in all_rows:
        yr = season_start(r["season"])
        key = (r["wiki_name"], yr)
        universe_map.setdefault(key, {"name": r["wiki_name"], "season_start": yr})
    universe = pd.DataFrame(
        [{"source_id": f"{name}__{yr}", "name": v["name"], "season_start": v["season_start"]}
         for (name, yr), v in universe_map.items()],
        columns=["source_id", "name", "season_start"])

    log(f"matching {len(universe)} distinct wiki transaction names onto NBA player_id")
    nba_players = read_table("players", base)
    id_map_new, report = match_players(universe, nba_players, source=SOURCE)
    result.match_report = report
    log(f"id matching: {report.summary()}")

    name_to_player_id: dict[tuple[str, int], int] = {}
    for r in id_map_new.itertuples(index=False):
        name, _, yr_s = r.source_id.rpartition("__")
        name_to_player_id[(name, int(yr_s))] = int(r.player_id)

    out_rows = []
    for r in all_rows:
        pid = name_to_player_id.get((r["wiki_name"], season_start(r["season"])))
        if pid is None:
            continue
        out_rows.append({"season": r["season"], "team_id": r["team_id"], "player_id": pid,
                         "direction": r["direction"], "source_kind": r["source_kind"],
                         "txn_date": r["txn_date"]})

    txns = pd.DataFrame(out_rows, columns=TEAM_TRANSACTIONS_COLUMNS)
    txns["team_id"] = txns["team_id"].astype("int64")
    txns["player_id"] = txns["player_id"].astype("int64")
    txns["txn_date"] = pd.to_datetime(txns["txn_date"])
    txns = txns.drop_duplicates(ignore_index=True)
    txns = txns.sort_values(["season", "team_id", "direction", "txn_date"], kind="stable").reset_index(drop=True)

    # Replace only the (season, team) pages fetched this run; every other season/team already in the
    # table (and every page that failed to fetch) keeps its previously ingested rows.
    fetched_pages = {(s, t) for s in seasons for t in TEAM_WIKI_NAMES} -         {(p["season"], p["team_id"]) for p in result.skipped_pages}
    existing_txns = read_team_transactions(base) if team_transactions_path(base).exists() else None
    txns = merge_team_transactions(existing_txns, txns, fetched_pages)
    written_path = write_team_transactions(txns, base)
    result.written_rows = len(txns)
    log(f"wrote {len(txns):,} team_transactions rows -> {written_path}")

    id_map = merge_id_map(_existing_id_map(base), id_map_new, source=SOURCE)
    id_map_path = write_table(id_map, "player_id_map", base)
    result.id_map_rows = len(id_map)
    log(f"wrote {len(id_map):,} player_id_map rows -> {id_map_path}")

    result.network_requests = client.stats.network_requests
    result.cache_hits = client.stats.cache_hits
    return result


def _existing_id_map(base: Path) -> pd.DataFrame | None:
    return read_table("player_id_map", base, validate=False) if table_exists("player_id_map", base) else None


def merge_team_transactions(existing: pd.DataFrame | None, new: pd.DataFrame,
                            fetched_pages: set[tuple[str, int]]) -> pd.DataFrame:
    """``new`` replaces the rows of exactly the (season, team_id) pages fetched this run; all other
    existing rows are kept, so a partial or single-season run never truncates the table."""
    if existing is None or existing.empty:
        return new.reset_index(drop=True)
    in_run = pd.Series([(s, int(t)) in fetched_pages for s, t in zip(existing["season"], existing["team_id"])],
                       index=existing.index)
    frames = [f for f in (existing[~in_run], new) if len(f)]
    if not frames:
        return new.reset_index(drop=True)
    merged = pd.concat(frames, ignore_index=True)
    merged["txn_date"] = pd.to_datetime(merged["txn_date"])
    return merged.sort_values(["season", "team_id", "direction", "txn_date"], kind="stable").reset_index(drop=True)


def merge_id_map(existing: pd.DataFrame | None, new: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Upsert this source's freshly matched rows by ``source_id`` (rows for names outside this run's
    seasons are kept); leave every other source (e.g. ``"espn"``) untouched."""
    if existing is None or existing.empty:
        return new.reset_index(drop=True)
    kept = existing[(existing["source"] != source)
                    | ~existing["source_id"].isin(set(new["source_id"]))]
    frames = [f for f in (kept, new) if len(f)]
    if not frames:
        return new.reset_index(drop=True)
    merged = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].copy()
    return merged.sort_values(["source", "source_id"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- ad-hoc table IO

def team_transactions_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / "processed" / "team_transactions.parquet"


def write_team_transactions(df: pd.DataFrame, base: Path | None = None) -> Path:
    """``team_transactions.parquet`` isn't a contract table (ADR 0011 D1, same precedent as
    ``adp.parquet``/``espn_adp.write_adp``), so this writes it directly rather than through
    ``store.write_table``."""
    import os
    import tempfile

    missing = [c for c in TEAM_TRANSACTIONS_COLUMNS if c not in df.columns]
    if missing:
        raise WikiTransactionsError(f"internal error: team_transactions frame missing {missing}")
    df = df[TEAM_TRANSACTIONS_COLUMNS].copy()
    path = team_transactions_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_team_transactions(base: Path | None = None) -> pd.DataFrame:
    path = team_transactions_path(base)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; has the wiki_transactions ingest run?")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.ingest.wiki_transactions",
        description="Ingest Wikipedia team-season Transactions sections into team_transactions.parquet, "
                    "mapped onto NBA player_id.",
    )
    p.add_argument("--seasons", required=True, help="e.g. 2015-16:2025-26, 2023-24, or 2015-16:2017-18,2020-21")
    p.add_argument("--offline", action="store_true",
                   help="never touch the network; fail with a clear error if a response is not cached "
                        "(also enabled by NBA_OFFLINE=1)")
    p.add_argument("--refresh", action="store_true", help="re-download pages even if cached")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="override the data root (default: NBA_DATA_DIR or ~/dev-data/nba-fantasy-2026)")
    p.add_argument("--min-interval", type=float, default=1.0,
                   help="minimum seconds between network requests (default 1.0 -- conservative; "
                        "Wikipedia's own posted politeness policy is far more permissive)")
    p.add_argument("--max-retries", type=int, default=5)
    return p


def main(argv: list[str] | None = None, *, client: CachedHttpClient | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        seasons = parse_seasons(args.seasons)
    except ValueError as exc:
        parser.error(str(exc))
    base = args.data_dir or data_dir()
    if client is None:
        client = CachedHttpClient(default_cache_dir(CACHE_DIR_NAME), offline=True if args.offline else None,
                                  min_interval=args.min_interval, max_retries=args.max_retries,
                                  headers={"User-Agent": UA})
    try:
        result = run_ingest(seasons, client, base, refresh=args.refresh)
    except (HttpCacheError, WikiTransactionsError, ContractError, FileNotFoundError) as exc:
        print(f"wiki transactions ingest failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"done: {len(seasons)} seasons x {len(TEAM_WIKI_NAMES)} teams, {result.fetched} pages fetched, "
          f"{len(result.skipped_pages)} skipped, {result.n_trade_rows} trade / {result.n_addition_rows} "
          f"addition / {result.n_subtraction_rows} subtraction candidate rows, "
          f"{result.written_rows} team_transactions rows written, "
          f"{result.network_requests} network requests, {result.cache_hits} cache hits")
    if result.match_report is not None:
        print(f"id matching: {result.match_report.summary()}")
    if result.skipped_pages:
        print(f"WARNING: {len(result.skipped_pages)} page(s) skipped; their existing rows were kept. "
              f"Re-run to retry.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
