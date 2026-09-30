"""Wikipedia team-season contract events: dated "N-year contract" phrases parsed into ``player_contracts``.

    python -m src.ingest.wiki_contracts --seasons 2015-16:2025-26 [--offline] [--refresh]

ADR 0013 recorded that the team-season wikitext already cached for ADR 0011 holds about 620 dated
"N-year contract" phrases which the transactions parser discards (it keeps only direction, kind and
date). ADR 0019 builds the structured parser for them. This module is an additive sibling of
``wiki_transactions`` (it reuses that module's fetch, section and player-name helpers unchanged and
does not touch ``team_transactions``): the ``Re-signed`` / ``Additions`` / ``Subtractions`` /
``Contract extensions`` tables of each page become one row per contract event.

Why a new parser rather than ``wiki_transactions``' row splitter: contract text lives in a "Signed" /
"Contract Terms" / "Reason" cell whose column position varies, and newer pages share one cell across
several rows with ``rowspan`` (a date, or a single ``Two-way contract`` cell). Here every table is
expanded into a full grid (rowspan carried down, ``||`` inline cells split) and the row is read by
content, not by header: the player is the first non-team, non-contract wikilink; the date is the first
date-looking cell text (else the row's first citation ``date=``); the contract terms are regexes over
the rest of the row text.

Output, ``<data_dir>/processed/player_contracts.parquet`` (an ad-hoc table, ADR 0011 D1 pattern, own
schema check in :func:`validate_player_contracts`; NOT in ``src/contracts.py``'s ``TABLES``). Columns
(``PLAYER_CONTRACTS_COLUMNS``):

``season``       leak-guard tag: the season the event FOLLOWS. An event dated 1 Oct of year Y or later
                 follows season Y; one dated Jan-Sep follows Y-1 (the season that just ended). So
                 ``History.until(T)`` (drops ``season >= T``) keeps exactly the events dated before
                 1 Oct of T's start year, the same cutoff as ADR 0011 D1, structurally.
``page_season``  the team-season page the row came from.
``start_year``   first season (start year) the contract covers: a deal dated 1 Jul or later covers the
                 season starting that year, a deal dated Jan-Jun covers the season already under way.
``team_id``      the team the contract is with (page team, or the "new team" of a ``Subtractions`` row).
``player_id``    canonical NBA id, from ``id_map.match_players``; unmatched rows are dropped, never guessed.
``event``        sign | resign | extension | claimed | waived | bought_out | expired | retired |
                 option_exercised | option_declined | traded | other.
``contract_type`` standard | two_way | ten_day | exhibit10 | minimum | rookie, or '' when no terms.
``years``, ``days``, ``amount_usd`` as stated (amount = the number in the text, usually the total) and
                 ``terms_source`` (cell: the table text; title: only the cited article's title said it).
``signed_date``  the event date (NaT when the row has none) and ``date_source`` (cell, citation,
                 access_date, cell_no_year, none).
flags            ``is_two_way, is_ten_day, is_exhibit10, is_minimum, is_extension, is_rookie,
                 non_guaranteed, has_option, multiyear_unspecified``.

No Wikipedia text is stored: only these derived facts (CC BY-SA 4.0, see ADR 0011 and ADR 0019).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts import ContractError, data_dir, season_start, season_str
from src.ingest.http_cache import CachedHttpClient, HttpCacheError, default_cache_dir
from src.ingest.id_map import match_players
from src.ingest.wiki_transactions import (
    CACHE_DIR_NAME, TEAM_WIKI_NAMES, UA, WikiTransactionsError, _headings, _section_text, _split_rows,
    _strip_markup, fetch_team_season_wikitext, parse_seasons, team_season_page_title,
)
from src.store import read_table

SOURCE = "wikipedia_contracts"
REPORT_NAME = "wiki_contracts_report.json"

PLAYER_CONTRACTS_COLUMNS = [
    "season", "page_season", "start_year", "team_id", "player_id", "event", "contract_type", "years", "days",
    "amount_usd", "signed_date", "date_source", "is_two_way", "is_ten_day", "is_exhibit10", "is_minimum",
    "is_extension", "is_rookie", "non_guaranteed", "has_option", "multiyear_unspecified", "section", "terms_source",
]
BOOL_COLUMNS = ["is_two_way", "is_ten_day", "is_exhibit10", "is_minimum", "is_extension", "is_rookie",
                "non_guaranteed", "has_option", "multiyear_unspecified"]
EVENTS = ("sign", "resign", "extension", "claimed", "waived", "bought_out", "expired", "retired",
          "option_exercised", "option_declined", "traded", "other")
CONTRACT_EVENTS = ("sign", "resign", "extension")     # events that create or extend a contract
END_EVENTS = ("waived", "bought_out", "expired", "retired")   # events that end the previous contract
CONTRACT_TYPES = ("", "standard", "two_way", "ten_day", "exhibit10", "minimum", "rookie")
TERMS_SOURCES = ("", "cell", "title")
DATE_SOURCES = ("cell", "citation", "access_date", "cell_no_year", "none")

_TEAM_BY_NAME = {n.lower(): tid for tid, n in TEAM_WIKI_NAMES.items()}


# --------------------------------------------------------------------------- season / date tags

def tag_start_year(d: pd.Timestamp) -> int:
    """Start year of the season an event dated ``d`` follows (Oct 1 rolls over: ADR 0011 D1's cutoff)."""
    return d.year if d.month >= 10 else d.year - 1


def coverage_start_year(d: pd.Timestamp) -> int:
    """Start year of the first season a contract signed on ``d`` covers (the league year turns on 1 Jul)."""
    return d.year if d.month >= 7 else d.year - 1


_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
     "November", "December"], 1)}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True)) + "|" + "|".join(m[:3] for m in _MONTHS) + "|Sept"
_MD_RE = re.compile(rf"\b(?P<mon>{_MONTH_ALT})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(?P<y>\d{{4}}))?\b", re.I)
_DM_RE = re.compile(rf"\b(?P<d>\d{{1,2}})\s+(?P<mon>{_MONTH_ALT})\.?,?\s+(?P<y>\d{{4}})\b", re.I)


def _month_num(tok: str) -> int | None:
    t = tok.lower().rstrip(".")
    if t == "sept":
        return 9
    if t in _MONTHS:
        return _MONTHS[t]
    for name, i in _MONTHS.items():
        if name[:3] == t[:3] and len(t) == 3:
            return i
    return None


def parse_cell_date(plain: str, page_season: str) -> tuple[pd.Timestamp, str]:
    """First calendar date in ``plain`` text. ``("Month D, YYYY" | "D Month YYYY")`` is exact; a yearless
    ``Month D`` takes the page's start year for Jun-Dec and the following year for Jan-May (a team-season
    page runs from the June draft to the end of that season). Returns ``(NaT, "none")`` if none."""
    y0 = season_start(page_season)
    cands: list[tuple[int, pd.Timestamp, str]] = []
    for rx in (_DM_RE, _MD_RE):
        for m in rx.finditer(plain):
            mon = _month_num(m.group("mon"))
            if mon is None:
                continue
            day = int(m.group("d"))
            ys = m.group("y")
            year = int(ys) if ys else (y0 if mon >= 6 else y0 + 1)
            try:
                ts = pd.Timestamp(year=year, month=mon, day=day)
            except ValueError:
                continue
            cands.append((m.start(), ts, "cell" if ys else "cell_no_year"))
    if not cands:
        return pd.NaT, "none"
    cands.sort(key=lambda c: c[0])
    return cands[0][1], cands[0][2]


_DTS_RE = re.compile(r"\{\{\s*dts\s*\|([^{}]*)\}\}", re.IGNORECASE)
_MONTH_NAMES = list(_MONTHS)


def expand_dts(raw: str) -> str:
    """Replace ``{{dts|...}}`` date templates by their date text so :func:`parse_cell_date` can read them.
    ``_strip_markup`` (shared with ADR 0011, untouched) deletes every template, which used to leave a row
    whose only date is a ``dts`` cell undated. Forms handled: ``{{dts|July 8, 2016}}``,
    ``{{dts|2026|June|23}}`` (also numeric month), ``{{dts|2016-07-08}}``; named arguments
    (``format=...``, ``link=...``) are ignored. An unrecognised shape is left as-is (still dropped)."""
    def sub(m: re.Match) -> str:
        args = [a.strip() for a in m.group(1).split("|") if a.strip() and "=" not in a]
        if len(args) == 1:
            iso = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", args[0])
            if iso:
                mon = int(iso.group(2))
                if 1 <= mon <= 12:
                    return f"{_MONTH_NAMES[mon - 1].title()} {int(iso.group(3))}, {iso.group(1)}"
                return m.group(0)
            return args[0]
        if len(args) >= 3 and re.fullmatch(r"\d{4}", args[0]):
            mon = args[1]
            if mon.isdigit() and 1 <= int(mon) <= 12:
                mon = _MONTH_NAMES[int(mon) - 1].title()
            return f"{mon} {args[2]}, {args[0]}"
        return m.group(0)
    return _DTS_RE.sub(sub, raw)


def _plausible(ts: pd.Timestamp, page_season: str, *, strict: bool = False) -> bool:
    """Inside the page window. Default: 1 Jun of the year BEFORE the page's start year to 30 Sep after the
    season. ``strict`` (used for citation / access dates, which can only be a fallback for the event date):
    1 Jun of the page's start year, so an article dated before the page's own draft-to-season window is
    never read as the event date (an earlier date = an earlier season tag = the leaking direction)."""
    y0 = season_start(page_season)
    return pd.Timestamp(y0 - 1 if not strict else y0, 6, 1) <= ts <= pd.Timestamp(y0 + 1, 9, 30)


_REF_RE = re.compile(r"<ref\b[^>]*?/>|<ref\b[^>]*>(.*?)</ref>", re.IGNORECASE | re.DOTALL)
_CITE_DATE_RE = re.compile(r"(?<![A-Za-z-])date\s*=\s*([^|{}<\n]+)", re.IGNORECASE)
_CITE_TITLE_RE = re.compile(r"(?<![A-Za-z-])title\s*=\s*([^|{}<\n]+)", re.IGNORECASE)


def _citation_date(raw: str) -> pd.Timestamp:
    for m in _REF_RE.finditer(raw):
        body = m.group(1) or ""
        dm = _CITE_DATE_RE.search(body)
        if dm:
            ts = pd.to_datetime(dm.group(1).strip(), errors="coerce")
            if pd.notna(ts):
                return ts
    return pd.NaT


_ACCESS_DATE_RE = re.compile(r"access-?date\s*=\s*([^|{}<\n]+)", re.IGNORECASE)


def _access_date(raw: str) -> pd.Timestamp:
    """First citation ``access-date``: never earlier than the event, so using it can only move an event's
    season tag later (safe for the leak guard), never earlier. Only a fallback for rows with no other date."""
    for m in _REF_RE.finditer(raw):
        dm = _ACCESS_DATE_RE.search(m.group(1) or "")
        if dm:
            ts = pd.to_datetime(dm.group(1).strip(), errors="coerce")
            if pd.notna(ts):
                return ts
    return pd.NaT


def _citation_titles(raw: str) -> str:
    out = []
    for m in _REF_RE.finditer(raw):
        tm = _CITE_TITLE_RE.search(m.group(1) or "")
        if tm:
            out.append(tm.group(1))
    return " ".join(out)


# --------------------------------------------------------------------------- table grid (rowspan aware)

_TOKEN = "\x00{}\x00"
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
_ROWSPAN_RE = re.compile(r"rowspan\s*=\s*[\"']?(\d+)", re.I)
_ATTR_KEY_RE = re.compile(r"[A-Za-z][A-Za-z-]*\s*=")


def _mask(text: str) -> tuple[str, list[str]]:
    """Replace refs and templates by placeholders so a ``||`` or ``|`` inside them cannot split a cell."""
    store: list[str] = []

    def keep(m: re.Match) -> str:
        store.append(m.group(0))
        return _TOKEN.format(len(store) - 1)

    text = re.sub(r"<ref\b[^>]*?/>|<ref\b[^>]*>.*?</ref>", keep, text, flags=re.I | re.S)
    for _ in range(3):
        text = _TEMPLATE_RE.sub(keep, text)
    return text, store


def _unmask(text: str, store: list[str]) -> str:
    for _ in range(4):
        new = re.sub(r"\x00(\d+)\x00", lambda m: store[int(m.group(1))], text)
        if new == text:
            break
        text = new
    return text


def _split_attr(cell: str) -> tuple[str, int]:
    """Drop a leading ``attrs |`` prefix (only when it looks like wiki attributes); return (content, rowspan)."""
    if "|" in cell:
        before, _, after = cell.partition("|")
        if "[[" not in before and "\x00" not in before and (before.strip() == "" or _ATTR_KEY_RE.search(before)):
            m = _ROWSPAN_RE.search(before)
            return after, int(m.group(1)) if m else 1
    return cell, 1


@dataclass
class Cell:
    raw: str
    inherited: bool = False


def _row_cells(row_text: str) -> list[tuple[str, int]]:
    """(cell text, rowspan) per physical cell of one ``|-`` row. A line starting with ``|`` or ``!`` opens
    a cell (``||`` / ``!!`` split several on one line); other lines continue the previous cell."""
    masked, store = _mask(row_text)
    cells: list[str] = []
    cur: str | None = None
    for line in masked.split("\n"):
        if line.startswith("|") or line.startswith("!"):
            if cur is not None:
                cells.append(cur)
            body = line[1:]
            if body.startswith("|") or body.startswith("!"):     # "|| a || b" style: first piece has an extra bar
                body = body[1:]
            pieces = re.split(r"\|\||!!", body)
            cells.extend(pieces[:-1])
            cur = pieces[-1]
        elif cur is not None:
            cur += "\n" + line
    if cur is not None:
        cells.append(cur)
    out = []
    for c in cells:
        content, span = _split_attr(c)
        out.append((_unmask(content, store), span))
    return out


def table_grid(section_body: str) -> list[list[Cell]]:
    """Every row of every wikitable in ``section_body`` as a list of cells, ``rowspan`` cells repeated
    (marked ``inherited``) in the rows they span."""
    rows_out: list[list[Cell]] = []
    carry: dict[int, list] = {}           # col -> [remaining, raw]
    for row_raw in _split_rows(section_body):
        phys = _row_cells(row_raw)
        if not phys:
            continue
        cells: list[Cell] = []
        col = 0
        for raw, span in phys:
            while col in carry and carry[col][0] > 0:
                cells.append(Cell(carry[col][1], True))
                carry[col][0] -= 1
                col += 1
            cells.append(Cell(raw, False))
            if span > 1:
                carry[col] = [span - 1, raw]
            else:
                carry.pop(col, None)
            col += 1
        while col in carry and carry[col][0] > 0:
            cells.append(Cell(carry[col][1], True))
            carry[col][0] -= 1
            col += 1
        rows_out.append(cells)
    return rows_out


# --------------------------------------------------------------------------- sections

SECTION_KIND = {
    "re-signed": "resign", "re-signings": "resign", "resigned": "resign", "re-signed players": "resign",
    "additions": "add", "signings": "add", "additions from non-nba players": "add",
    "subtractions": "sub", "waivings": "sub",
    "contract extensions": "extension",
}


def contract_sections(transactions_text: str) -> list[tuple[str, str]]:
    """``(kind, own body)`` for each contract-bearing subsection. The body is the text up to the next heading
    of ANY level, so a parent (``Free agency``, ``Contracts``) never re-counts its children."""
    hs = _headings(transactions_text)
    out = []
    for i, (_lvl, title, content_start, _line_start) in enumerate(hs):
        kind = SECTION_KIND.get(title.strip().lower())
        if kind is None:
            continue
        end = hs[i + 1][3] if i + 1 < len(hs) else len(transactions_text)
        out.append((kind, transactions_text[content_start:end]))
    return out


# --------------------------------------------------------------------------- row content

_LINK_RE = re.compile(r"\[\[([^\[\]|]+)(?:\|([^\[\]]+))?\]\]")
_NON_PLAYER_LINK = re.compile(
    r"contract|two-way|exhibit|waive|free agen|^nba\b|g league|d-league|draft|sign-and-trade|option|retire|"
    r"^list of|extension|training camp|men's basketball|women's|club$|^file:|^image:|"
    r"buyout|national basketball", re.I)


def _is_team_target(target: str) -> bool:
    t = target.strip().lower()
    return t in _TEAM_BY_NAME or bool(re.search(
        r"\b(hawks|celtics|nets|hornets|bulls|cavaliers|mavericks|nuggets|pistons|warriors|rockets|pacers|"
        r"clippers|lakers|grizzlies|heat|bucks|timberwolves|pelicans|knicks|thunder|magic|76ers|suns|"
        r"trail blazers|kings|spurs|raptors|jazz|wizards|swarm|legends|skyforce|mad ants|"
        r"charge|hustle|blue coats|vipers|stars|herd|clamps|bighorns|"
        r"g league|d-league)\b", t))


def _clean_name(display: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", display.strip())


_SORTNAME_RE = re.compile(r"\{\{\s*sortname\s*\|([^|{}]*)\|([^|{}]*)(?:\|[^{}]*)?\}\}", re.I)
_WORDS_NOT_NAME = re.compile(
    r"contract|waive|expire|trade|free agen|two-way|option|signed|retire|released|draft|claimed|"
    r"n/a|tba|team|unrestricted|restricted|buyout|extension|minimum|exhibit|day|deal|year|coach|"
    r"g league|reason|date|player|addition|subtraction|season|signing", re.I)
_NAME_TOKEN = re.compile(r"^(?:[A-Z][\w'.\-]*|de|van|von|del|la|le|da|di|dos|mc[A-Z]\w*)$")


def expand_sortname(raw: str) -> str:
    """``{{sortname|LeBron|James}}`` (newer pages' player cell; no wikilink) -> ``[[LeBron James]]``."""
    return _SORTNAME_RE.sub(lambda m: f"[[{m.group(1).strip()} {m.group(2).strip()}]]", raw)


def _plain_name(plain: str) -> str | None:
    """A player written as plain text (no link): 2 to 4 capitalised tokens, no digits or contract words."""
    t = re.sub(r"\s*\([^)]*\)", "", plain).strip()
    toks = t.split()
    if not 2 <= len(toks) <= 4 or re.search(r"\d", t) or _WORDS_NOT_NAME.search(t) or _is_team_target(t):
        return None
    if not all(_NAME_TOKEN.match(x) for x in toks) or _MD_RE.search(t):
        return None
    return t


def find_player(cells: list[Cell]) -> str | None:
    """Display name of the row's player: scanning the row's own cells left to right, the first cell that holds a
    wikilink (or ``{{sortname}}``) which is not a team, a contract-type page or a draft/league page, or -- when a
    cell is a plain-text name (no link at all) -- that name. Order matters: a plain-text player followed by a
    linked foreign club must give the player, not the club. Matching to ``players`` later is the real gate."""
    for c in cells:
        if c.inherited:
            continue
        raw = expand_sortname(re.sub(r"<ref[^>]*?/>|<ref[^>]*>.*?</ref>", "", c.raw, flags=re.I | re.S))
        found = False
        for m in _LINK_RE.finditer(raw):
            found = True
            target, disp = m.group(1).strip(), (m.group(2) or m.group(1)).strip()
            if _NON_PLAYER_LINK.search(target) or _is_team_target(target) or _is_team_target(disp):
                continue
            return _clean_name(disp)
        if not found:
            nm = _plain_name(_plain(raw))
            if nm:
                return nm
    return None


def find_team_id(cells: list[Cell]) -> int | None:
    """The first NBA team linked in the row (used for ``Subtractions``: the new team)."""
    for c in cells:
        for m in _LINK_RE.finditer(c.raw):
            tid = _TEAM_BY_NAME.get(m.group(1).strip().lower())
            if tid is not None:
                return tid
    return None


_NUM_WORD = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_YEARS_RE = re.compile(r"\b(\d{1,2}|one|two|three|four|five|six)\s*[-‐-―]?\s*(?:year|yr)s?\b(?!\s*[-‐-―]?\s*old)", re.I)
_DAYS_RE = re.compile(r"\b(\d{1,2})\s*[-‐-―]?\s*days?\b", re.I)
_AMT_RE = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(million|mil\b|m\b|thousand|k\b|billion|b\b)?", re.I)
_MULTIYEAR_RE = re.compile(r"multi[\s\-‐-―]*year", re.I)


def parse_years(text: str) -> int | None:
    m = _YEARS_RE.search(text)
    if not m:
        return None
    tok = m.group(1).lower()
    n = int(tok) if tok.isdigit() else _NUM_WORD[tok]
    return n if 1 <= n <= 6 else None


def parse_amount(text: str) -> float | None:
    """First dollar figure in ``text`` as US dollars (``$54.3 million`` -> 54,300,000). Implausible values
    (under $10,000 or over $1 billion) are treated as not stated."""
    m = _AMT_RE.search(text.replace(" ", " "))
    if not m:
        return None
    v = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    if unit in ("million", "mil", "m"):
        v *= 1e6
    elif unit in ("thousand", "k"):
        v *= 1e3
    elif unit in ("billion", "b"):
        v *= 1e9
    return v if 1e4 <= v <= 1e9 else None


@dataclass
class ParsedRow:
    wiki_name: str
    page_season: str
    page_team_id: int
    section: str
    event: str
    team_id: int | None
    contract_type: str = ""
    years: int | None = None
    days: int | None = None
    amount_usd: float | None = None
    date: pd.Timestamp = pd.NaT
    date_source: str = "none"
    cite_date: pd.Timestamp = pd.NaT       # first citation ``date=`` on the row (any year); cross-check for the yearless roll
    is_two_way: bool = False
    is_ten_day: bool = False
    is_exhibit10: bool = False
    is_minimum: bool = False
    is_extension: bool = False
    is_rookie: bool = False
    non_guaranteed: bool = False
    has_option: bool = False
    multiyear_unspecified: bool = False
    has_terms: bool = False
    terms_source: str = ""


def _plain(raw: str) -> str:
    t = _strip_markup(raw).replace("&nbsp;", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", t).strip()


_WAIVE = re.compile(r"waive|released?\b|cut\b", re.I)
_BUYOUT = re.compile(r"buy[\s-]?out|bought out|contract buyout", re.I)
_RETIRE = re.compile(r"retire", re.I)
_EXPIRED = re.compile(r"expire|free[\s-]?agen|\bufa\b|\brfa\b|unrestricted|restricted|contract ended|opted out|opt[\s-]?out", re.I)
_OPT_DECLINED = re.compile(r"declin|not exercise|did not pick|option (?:was )?not", re.I)
_OPT_EXERCISED = re.compile(r"exercis|picked up|option (?:was )?picked", re.I)
_TRADE = re.compile(r"\btrade[sd]?\b", re.I)
_CLAIM = re.compile(r"claimed|waiver claim|off waivers", re.I)
_CONTRACT_HINT = re.compile(r"\d\s*[-‐-―]?\s*(?:year|yr|day)|\$|two[\s-]*way|exhibit|extension|multi[\s-]*year|"
                            r"\b(?:minimum|non-?guaranteed|guaranteed|training camp|\bsigned\b)", re.I)


def classify_row(kind: str, cells: list[Cell], page_season: str, page_team_id: int, draft_names: frozenset[str]) -> ParsedRow | None:
    """One table row -> :class:`ParsedRow` (or ``None`` when no player can be identified)."""
    name = find_player(cells)
    if name is None:
        return None
    raw_all = "\n".join(c.raw for c in cells if not c.inherited)
    raw_inherited = "\n".join(c.raw for c in cells if c.inherited)
    # Text the terms are read from: the whole row as plain text, minus the player's own name.
    body_plain = _plain(" | ".join(expand_sortname(c.raw) for c in cells))
    terms = body_plain.replace(name, " ")
    tl = terms.lower()

    # ---- date: first date in a cell's own text (cells first, then citations)
    date, dsrc = pd.NaT, "none"
    for c in cells:
        d, s = parse_cell_date(_plain(expand_dts(c.raw)), page_season)
        if pd.notna(d) and _plausible(d, page_season):
            date, dsrc = d, s
            break
    if pd.isna(date):
        d = _citation_date(raw_all + "\n" + raw_inherited)
        if pd.notna(d) and _plausible(d, page_season, strict=True):
            date, dsrc = d, "citation"
    if pd.isna(date):
        d = _access_date(raw_all + "\n" + raw_inherited)
        if pd.notna(d) and _plausible(d, page_season, strict=True):
            date, dsrc = d, "access_date"

    # ---- event
    event = "other"
    if kind == "extension":
        event = "extension"
    elif kind == "resign":
        event = "resign"
    elif kind == "add":
        event = "sign"
    if kind == "sub" or kind == "resign":
        if _BUYOUT.search(tl):
            event = "bought_out"
        elif _WAIVE.search(tl):
            event = "waived"
        elif _RETIRE.search(tl):
            event = "retired"
        elif kind == "sub":
            # "Second 10-day contract expired" / "Two-way contract expired" end a deal; they do not sign one.
            has_contract = (parse_years(terms) is not None or parse_amount(terms) is not None
                            or (bool(re.search(r"two[\s-]*way|exhibit|\d\s*[-‐-―]?\s*day|extension", tl))
                                and not re.search(r"expire", tl)))
            if has_contract:
                event = "sign"                      # signed elsewhere: contract text on a departure row
            elif _OPT_DECLINED.search(tl):
                event = "option_declined"
            elif _EXPIRED.search(tl):
                event = "expired"
            elif _TRADE.search(tl):
                event = "traded"
            elif re.search(r"\bsigned\b", tl):
                event = "sign"
    elif kind == "add" and _CLAIM.search(tl):
        event = "claimed"
    if kind in ("add", "resign", "extension") and re.search(r"\bextension\b", tl):
        event = "extension"
    if _OPT_EXERCISED.search(tl) and event in ("other", "resign") and "year" not in tl:
        event = "option_exercised"

    team_id = page_team_id
    if kind == "sub":
        team_id = find_team_id([c for c in cells if not c.inherited])  # None when the new team is not an NBA club

    row = ParsedRow(wiki_name=name, page_season=page_season, page_team_id=page_team_id, section=kind,
                    event=event, team_id=team_id, date=date, date_source=dsrc,
                    cite_date=_citation_date(raw_all + "\n" + raw_inherited))
    if event not in CONTRACT_EVENTS:
        return row

    # ---- contract terms (only for events that create a contract). The table's own cells win; when they
    # state nothing (newer pages list only the date and the player) the cited article's title is the only
    # place the terms appear ("Lakers re-sign X to 4-year deal"), so it is read as a labelled fallback.
    cell_terms_present = (parse_years(terms) is not None or parse_amount(terms) is not None or bool(
        re.search(r"two[\s-]*way|exhibit|training camp|\bminimum\b|\d\s*[-‐-―]?\s*day|multi[\s-]*year", tl)))
    title_txt = "" if cell_terms_present else _plain(_citation_titles(raw_all))
    eff = (terms + " " + title_txt).strip()
    tl = eff.lower()
    years = parse_years(eff)
    amount = parse_amount(eff)
    dm = _DAYS_RE.search(eff)
    row.terms_source = "title" if (title_txt and (years or amount or dm or re.search(r"two[\s-]*way|exhibit|training camp", tl))) else        ("cell" if cell_terms_present else "")
    row.years = years
    row.amount_usd = amount
    # A cell listing 10-day contracts and then a longer deal ("Two 10-day contracts / 2-year contract ...") is the longer deal.
    row.is_ten_day = dm is not None and years is None
    row.days = int(dm.group(1)) if dm and years is None else None
    row.is_two_way = bool(re.search(r"two[\s-]*way", tl))
    row.is_exhibit10 = bool(re.search(r"exhibit\s*[-]?\s*(?:10|ten)|training camp", tl))
    row.is_minimum = bool(re.search(r"\bminimum\b", tl))
    row.is_extension = event == "extension"
    row.non_guaranteed = bool(re.search(r"non[\s-]*guaranteed|partially guaranteed", tl))
    row.has_option = bool(re.search(r"team option|player option|\boption\b", tl))
    row.multiyear_unspecified = bool(_MULTIYEAR_RE.search(tl)) and years is None
    row.is_rookie = bool(re.search(r"nba draft|draft pick|first[\s-]round pick|second[\s-]round pick|rookie|\bDP\b", body_plain)) \
        or _clean_name(name) in draft_names or bool(re.search(r"\[\[NBA draft", raw_all, re.I))
    if row.is_ten_day:
        row.years = None
        row.contract_type = "ten_day"
    elif row.is_two_way:
        row.contract_type = "two_way"
    elif row.is_exhibit10:
        row.contract_type = "exhibit10"
    elif row.is_rookie and event == "sign":
        row.contract_type = "rookie"
    elif row.is_minimum:
        row.contract_type = "minimum"
    else:
        row.contract_type = "standard"
    row.has_terms = bool(row.years or row.days or row.amount_usd or row.is_two_way or row.is_exhibit10
                         or row.is_minimum or row.multiyear_unspecified)
    return row


def _draft_names(wikitext: str) -> frozenset[str]:
    body = _section_text(wikitext, 2, "Draft picks")
    names = set()
    for m in _LINK_RE.finditer(body):
        names.add(_clean_name(m.group(2) or m.group(1)))
    return frozenset(names)


@dataclass
class PageParse:
    team_id: int
    season: str
    rows: list[ParsedRow] = field(default_factory=list)
    n_table_rows: int = 0
    n_no_player: int = 0
    n_year_phrases_page: int = 0         # "N-year" contract phrases anywhere in the page wikitext
    n_year_phrases_tables: int = 0       # ... inside the parsed transaction tables


_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_PHRASE_RE = re.compile(r"\b(\d{1,2}|one|two|three|four|five)[\s\-‐-―]*years?\b[\s,]*(?:non[\s-]*guaranteed\s+|partially guaranteed\s+)?(?:contract|deal|extension|,?\s*\$)", re.I)


ROLL_GAP_DAYS = 120
CITE_SLACK_DAYS = 30


def roll_yearless_dates(rows: list["ParsedRow"], page_season: str) -> None:
    """Fix the year of yearless ``Month D`` cells using the table's own chronological order (in place).

    :func:`parse_cell_date` reads a yearless date in the page's start year (Jun-Dec) or the next (Jan-May).
    A team-season page also lists the summer AFTER its season (contracts expiring 30 Jun, July signings
    added late), and there a yearless "June 30" / "July 9" would be read one year too EARLY, which is the
    leaking direction. Tables are (mostly) chronological, so a yearless date that falls more than
    ``ROLL_GAP_DAYS`` before the latest earlier dated cell in the same section is moved to the next year
    (when that stays inside the page window). A wrong roll of a name-sorted table can only make an event
    later (hidden longer), never earlier. Yearless dates with no such evidence are unchanged; the residual
    ambiguity (a yearless June date in the first rows of a table cannot be shown to belong to the next
    year) is documented in ADR 0019.

    Cross-check with the row's own citation (``cite_date``, dated on/after the event): a roll is skipped when
    the citation is already earlier than the rolled date (minus ``CITE_SLACK_DAYS``), and a yearless row whose
    page-year reading is later than its own citation (an earlier-season event listed at the top of the table,
    e.g. an April date on the next season's page) is not allowed to anchor ``run_max`` for later rows."""
    run_max = pd.NaT
    for r in rows:
        if pd.isna(r.date) or r.date_source not in ("cell", "cell_no_year"):
            continue
        if r.date_source == "cell_no_year":
            if pd.notna(run_max) and (run_max - r.date).days > ROLL_GAP_DAYS:
                rolled = r.date + pd.DateOffset(years=1)
                # The row's own citation is dated on/after the event: a citation already earlier than the
                # rolled date says the roll is wrong, so the row keeps its page-year reading.
                cite_ok = pd.isna(r.cite_date) or r.cite_date >= rolled - pd.Timedelta(days=CITE_SLACK_DAYS)
                if cite_ok and _plausible(rolled, page_season):
                    r.date = rolled
            elif pd.notna(r.cite_date) and r.cite_date < r.date - pd.Timedelta(days=CITE_SLACK_DAYS):
                continue   # dated after its own citation (year-late reading of an earlier-season event): cannot anchor the run
        if pd.isna(run_max) or r.date > run_max:
            run_max = r.date


def parse_team_season_contracts(wikitext: str, team_id: int, season: str) -> PageParse:
    """Pure: raw wikitext -> contract-event rows of one team-season page. A page with no ``Transactions``
    section (or none of the contract subsections) yields zero rows, never an error."""
    page = PageParse(team_id=team_id, season=season)
    page.n_year_phrases_page = len(_PHRASE_RE.findall(wikitext))
    tx = _section_text(wikitext, 2, "Transactions")
    draft = _draft_names(wikitext)
    for kind, body in contract_sections(tx):
        body = _COMMENT_RE.sub("", body)              # commented-out rows are not events
        page.n_year_phrases_tables += len(_PHRASE_RE.findall(body))
        sect_rows: list[ParsedRow] = []
        for cells in table_grid(body):
            if re.fullmatch(r"[}>!<\-\s|]*", _plain(" ".join(c.raw for c in cells))):
                continue                                # table debris ("|}-->"), not a row
            page.n_table_rows += 1
            row = classify_row(kind, cells, season, team_id, draft)
            if row is None:
                page.n_no_player += 1
                continue
            sect_rows.append(row)
        roll_yearless_dates(sect_rows, season)
        page.rows.extend(sect_rows)
    return page


# --------------------------------------------------------------------------- frame assembly

def _rows_to_frame(rows: list[ParsedRow]) -> pd.DataFrame:
    recs = []
    for r in rows:
        if pd.notna(r.date):
            tag, cov = tag_start_year(r.date), coverage_start_year(r.date)
        else:
            tag = cov = season_start(r.page_season)
        recs.append({
            "wiki_name": r.wiki_name, "season": season_str(tag), "page_season": r.page_season, "start_year": cov,
            "team_id": r.team_id if r.team_id is not None else pd.NA, "event": r.event,
            "contract_type": r.contract_type, "years": r.years if r.years is not None else pd.NA,
            "days": r.days if r.days is not None else pd.NA,
            "amount_usd": r.amount_usd if r.amount_usd is not None else np.nan,
            "signed_date": r.date, "date_source": r.date_source, "is_two_way": r.is_two_way,
            "is_ten_day": r.is_ten_day, "is_exhibit10": r.is_exhibit10, "is_minimum": r.is_minimum,
            "is_extension": r.is_extension, "is_rookie": r.is_rookie, "non_guaranteed": r.non_guaranteed,
            "has_option": r.has_option, "multiyear_unspecified": r.multiyear_unspecified, "section": r.section,
            "has_terms": r.has_terms, "terms_source": r.terms_source,
        })
    return pd.DataFrame(recs)


def dedupe_events(df: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """A departure row on team A ("4-year contract ... New team: B") and the arrival row on B's page are the
    same event. Keep one per (player, event, years, amount) within ``window_days``, preferring the
    arrival row (its team is the page team, its date usually a citation on the signing itself)."""
    if df.empty:
        return df
    df = df.copy()
    df["_pref"] = (df["section"] == "sub").astype(int)
    df = df.sort_values(["player_id", "event", "signed_date", "_pref"], kind="stable")
    keep = np.ones(len(df), dtype=bool)
    last: dict[tuple, tuple[pd.Timestamp, int]] = {}
    idx = list(df.index)
    for pos, i in enumerate(idx):
        r = df.loc[i]
        if r["event"] not in CONTRACT_EVENTS or pd.isna(r["signed_date"]):
            continue
        key = (int(r["player_id"]), r["event"], None if pd.isna(r["years"]) else int(r["years"]),
               None if pd.isna(r["amount_usd"]) else float(r["amount_usd"]), bool(r["is_ten_day"]))
        prev = last.get(key)
        if prev is not None and abs((r["signed_date"] - prev[0]).days) <= window_days:
            if df.loc[idx[prev[1]], "_pref"] > r["_pref"]:
                keep[prev[1]] = False
                last[key] = (r["signed_date"], pos)
            else:
                keep[pos] = False
            continue
        last[key] = (r["signed_date"], pos)
    return df[keep].drop(columns="_pref").reset_index(drop=True)


# --------------------------------------------------------------------------- schema

def validate_player_contracts(df: pd.DataFrame) -> pd.DataFrame:
    """Own schema check (ADR 0011 D1 pattern): columns, dtypes, enums, and the leak-guard tag invariants."""
    missing = [c for c in PLAYER_CONTRACTS_COLUMNS if c not in df.columns]
    if missing:
        raise WikiContractsError(f"player_contracts missing columns {missing}")
    if df["player_id"].isna().any():
        raise WikiContractsError("player_contracts.player_id has nulls (unmatched rows must be dropped)")
    bad = sorted(set(df["event"]) - set(EVENTS))
    if bad:
        raise WikiContractsError(f"player_contracts.event has unknown values {bad}")
    bad = sorted(set(df["contract_type"]) - set(CONTRACT_TYPES))
    if bad:
        raise WikiContractsError(f"player_contracts.contract_type has unknown values {bad}")
    bad = sorted(set(df["terms_source"]) - set(TERMS_SOURCES))
    if bad:
        raise WikiContractsError(f"player_contracts.terms_source has unknown values {bad}")
    bad = sorted(set(df["date_source"]) - set(DATE_SOURCES))
    if bad:
        raise WikiContractsError(f"player_contracts.date_source has unknown values {bad}")
    for c in ("season", "page_season"):
        for v in df[c].unique():
            try:
                season_start(v)
            except ValueError as exc:
                raise WikiContractsError(f"player_contracts.{c} has a malformed season: {exc}") from exc
    yrs = pd.to_numeric(df["years"], errors="coerce").dropna()
    if len(yrs) and not yrs.between(1, 6).all():
        raise WikiContractsError("player_contracts.years outside 1..6")
    dated = df[df["signed_date"].notna()]
    if len(dated):
        want = dated["signed_date"].map(tag_start_year)
        got = dated["season"].map(season_start)
        if not (want.to_numpy() == got.to_numpy()).all():
            raise WikiContractsError("player_contracts.season tag disagrees with signed_date (leak-guard tag)")
    if df.duplicated(PLAYER_CONTRACTS_COLUMNS).any():
        raise WikiContractsError("player_contracts has duplicate rows")
    return df


class WikiContractsError(WikiTransactionsError):
    """The parsed/matched contract data is unusable in a way the CLI should stop and report."""


def finalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["player_id"] = out["player_id"].astype("int64")
    out["start_year"] = out["start_year"].astype("int64")
    out["team_id"] = pd.array(out["team_id"], dtype="Int64")
    out["years"] = pd.array(out["years"], dtype="Int64")
    out["days"] = pd.array(out["days"], dtype="Int64")
    out["amount_usd"] = out["amount_usd"].astype("float64")
    out["signed_date"] = pd.to_datetime(out["signed_date"])
    for c in BOOL_COLUMNS:
        out[c] = out[c].astype(bool)
    out = out[PLAYER_CONTRACTS_COLUMNS].drop_duplicates(ignore_index=True)
    return out.sort_values(["player_id", "signed_date", "event"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- IO

def player_contracts_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / "processed" / "player_contracts.parquet"


def write_player_contracts(df: pd.DataFrame, base: Path | None = None) -> Path:
    import os
    import tempfile

    validate_player_contracts(df)
    path = player_contracts_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        df[PLAYER_CONTRACTS_COLUMNS].to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_player_contracts(base: Path | None = None) -> pd.DataFrame:
    path = player_contracts_path(base)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; has the wiki_contracts ingest run?")
    return validate_player_contracts(pd.read_parquet(path))


# --------------------------------------------------------------------------- pipeline

@dataclass
class ContractIngestResult:
    seasons: list[str]
    fetched: int = 0
    skipped_pages: list[dict] = field(default_factory=list)
    written_rows: int = 0
    report: dict = field(default_factory=dict)
    network_requests: int = 0
    cache_hits: int = 0


def run_ingest(seasons: list[str], client: CachedHttpClient, base: Path | None = None, *, refresh: bool = False,
               log=print, write: bool = True) -> ContractIngestResult:
    if not seasons:
        raise WikiContractsError("no seasons to ingest")
    base = base or data_dir()
    result = ContractIngestResult(seasons=list(seasons))
    all_rows: list[ParsedRow] = []
    page_stats: list[dict] = []
    for season in seasons:
        for team_id in sorted(TEAM_WIKI_NAMES):
            try:
                text = fetch_team_season_wikitext(client, team_id, season, refresh=refresh)
            except HttpCacheError as exc:
                log(f"[{season}] {TEAM_WIKI_NAMES[team_id]}: fetch failed ({exc}); skipped")
                result.skipped_pages.append({"season": season, "team_id": team_id,
                                             "title": team_season_page_title(team_id, season), "reason": str(exc)})
                continue
            result.fetched += 1
            pg = parse_team_season_contracts(text, team_id, season)
            all_rows.extend(pg.rows)
            page_stats.append({"season": season, "team_id": team_id, "n_table_rows": pg.n_table_rows,
                               "n_no_player": pg.n_no_player, "n_phrases_page": pg.n_year_phrases_page,
                               "n_phrases_tables": pg.n_year_phrases_tables, "n_rows": len(pg.rows)})
    raw = _rows_to_frame(all_rows)
    if raw.empty:
        raise WikiContractsError("no contract rows parsed from any page")

    universe = (raw.assign(yr=raw["season"].map(season_start).add(1))   # matching year = the season the name appears
                .drop_duplicates(["wiki_name", "yr"])[["wiki_name", "yr"]])
    uni = pd.DataFrame({"source_id": universe["wiki_name"] + "__" + universe["yr"].astype(str),
                        "name": universe["wiki_name"], "season_start": universe["yr"]})
    log(f"matching {len(uni)} distinct (name, season) contract names onto NBA player_id")
    nba_players = read_table("players", base)
    id_map, match_report = match_players(uni, nba_players, source=SOURCE)
    log(f"id matching: {match_report.summary()}")
    lookup = {}
    for r in id_map.itertuples(index=False):
        name, _, yr = r.source_id.rpartition("__")
        lookup[(name, int(yr))] = int(r.player_id)
    raw["player_id"] = [lookup.get((n, season_start(s) + 1)) for n, s in zip(raw["wiki_name"], raw["season"])]
    matched = raw[raw["player_id"].notna()].copy()
    deduped = dedupe_events(matched)
    frame = finalize_frame(deduped.drop(columns=["wiki_name", "has_terms"]))

    result.report = build_report(raw, matched, deduped, frame, page_stats, match_report, seasons)
    if write:
        path = write_player_contracts(frame, base)
        result.written_rows = len(frame)
        log(f"wrote {len(frame):,} player_contracts rows -> {path}")
        rp = path.parent / REPORT_NAME
        rp.write_text(json.dumps(result.report, indent=2, default=str), encoding="utf-8")
    else:
        result.written_rows = len(frame)
    result.network_requests = client.stats.network_requests
    result.cache_hits = client.stats.cache_hits
    return result


def build_report(raw: pd.DataFrame, matched: pd.DataFrame, deduped: pd.DataFrame, frame: pd.DataFrame,
                 page_stats: list[dict], match_report, seasons: list[str]) -> dict:
    ps = pd.DataFrame(page_stats)
    by_season = {}
    for s in seasons:
        r = raw[raw["page_season"] == s]
        m = matched[matched["page_season"] == s]
        p = ps[ps["season"] == s] if len(ps) else ps
        ce = r[r["event"].isin(CONTRACT_EVENTS)]
        me = m[m["event"].isin(CONTRACT_EVENTS)]
        by_season[s] = {
            "pages": int(len(p)), "table_rows": int(p["n_table_rows"].sum()) if len(p) else 0,
            "phrases_in_page": int(p["n_phrases_page"].sum()) if len(p) else 0,
            "phrases_in_tables": int(p["n_phrases_tables"].sum()) if len(p) else 0,
            "rows_parsed": int(len(r)), "contract_event_rows": int(len(ce)),
            "rows_with_years": int(ce["years"].notna().sum()), "rows_with_amount": int(ce["amount_usd"].notna().sum()),
            "rows_dated": int(r["signed_date"].notna().sum()),
            "matched_rows": int(len(m)), "matched_contract_event_rows": int(len(me)),
            "matched_with_years": int(me["years"].notna().sum()),
        }
    return {
        "source": SOURCE, "seasons": seasons,
        "totals": {
            "pages": int(len(ps)), "table_rows": int(ps["n_table_rows"].sum()), "rows_parsed": int(len(raw)),
            "phrases_in_page": int(ps["n_phrases_page"].sum()), "phrases_in_tables": int(ps["n_phrases_tables"].sum()),
            "contract_event_rows": int(raw["event"].isin(CONTRACT_EVENTS).sum()),
            "rows_with_years": int((raw["event"].isin(CONTRACT_EVENTS) & raw["years"].notna()).sum()),
            "rows_with_amount": int((raw["event"].isin(CONTRACT_EVENTS) & raw["amount_usd"].notna()).sum()),
            "rows_dated": int(raw["signed_date"].notna().sum()),
            "matched_rows": int(len(matched)), "after_dedupe": int(len(deduped)), "written": int(len(frame)),
            "events": raw["event"].value_counts().to_dict(),
        },
        "by_season": by_season,
        "id_matching": match_report.summary(),
    }


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.ingest.wiki_contracts",
                                description="Parse dated contract events from cached Wikipedia team-season pages "
                                            "into player_contracts.parquet (ADR 0019).")
    p.add_argument("--seasons", required=True, help="e.g. 2015-16:2025-26")
    p.add_argument("--offline", action="store_true", help="never touch the network (also NBA_OFFLINE=1)")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--min-interval", type=float, default=1.0)
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
        res = run_ingest(seasons, client, base, refresh=args.refresh)
    except (HttpCacheError, WikiTransactionsError, ContractError, FileNotFoundError) as exc:
        print(f"wiki contracts ingest failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    t = res.report["totals"]
    print(f"done: {res.fetched} pages, {len(res.skipped_pages)} skipped, {t['rows_parsed']} rows parsed, "
          f"{t['contract_event_rows']} contract-event rows ({t['rows_with_years']} with years), "
          f"{t['matched_rows']} matched, {res.written_rows} written; {res.network_requests} network requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
