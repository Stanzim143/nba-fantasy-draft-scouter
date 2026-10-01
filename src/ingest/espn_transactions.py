"""ESPN's public NBA transactions feed -> the dated, append-only ``league_transactions`` ledger (ADR 0017).

    python -m src.ingest.espn_transactions [--days 90 | --from 2026-06-01 --to 2026-09-25] [--offline]

Why this source. Every earlier team-context source in this repo was retrospective (Wikipedia team-season pages, ADR
0011) or a roster diff (``roster_snapshots``, ADR 0012), which sees *that* a player changed team but not *how*
(trade, signing, waiver claim, two-way conversion) and never sees staff moves. ESPN's site API
(``site.api.espn.com/apis/site/v2/sports/basketball/nba/transactions``) is the same free, no-auth family as the player
feed ``espn_adp`` already reads (ADR 0005 R2 stance: personal, non-commercial, low volume, local only). It returns one
row per team per day with a free-text description, filterable by ``dates=YYYYMMDD-YYYYMMDD`` and paged with ``page``;
it reaches back to at least 2016 (which the coach layer uses for hires, firings and assistant appointments).

What it produces: ``<data>/processed/league_transactions.parquet``, an ad-hoc table (same precedent as ``adp``), one row
per *person per action*, never rewritten:

``txn_key`` (stable hash), ``txn_date``, ``team_id`` / ``team_abbr`` (the team whose row it is), ``kind``, ``subject_type``
(``player`` | ``staff``), ``person``, ``player_id`` (nullable: unmatched or staff), ``other_team_id`` (a trade's other
side, nullable), ``detail`` (contract type or staff role), ``description`` (the full source row), ``first_seen`` /
``last_seen`` (UTC ingest timestamps -- ``first_seen`` is what "new since I last looked" means; ESPN rows carry a date,
not a time).

Player kinds: ``signed``, ``resigned``, ``extended``, ``converted`` (two-way to NBA), ``waived``, ``claimed`` (off
waivers), ``trade_in`` / ``trade_out`` (from the row's team's point of view; the counterparty's row mirrors it, and a
consumer that wants one row per player-move dedupes on ``(txn_date, player_id, kind)`` -- see
``src.features.txn_impact``). Staff kinds: ``hired``, ``fired``, ``resigned_staff``, ``extended_staff`` with ``detail`` = role.

Parsing is deliberately conservative (the ADR 0011 stance): a sentence it cannot read is counted in the report and its
row is still recorded as ``kind='other'`` with the whole description, never guessed into a player move; a name that
does not resolve to a canonical ``player_id`` keeps ``player_id`` null and is listed in the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.contracts import ContractError, data_dir
from src.ingest.http_cache import CachedHttpClient, HttpCacheError, default_cache_dir
from src.ingest.id_map import match_players, normalize_name

SOURCE = "espn_transactions"
CACHE_LABEL = "transactions"
API_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/transactions"
TABLE = "league_transactions"
REPORT_NAME = "espn_transactions_report.json"
PAGE_SIZE = 300
MAX_PAGES = 40                      # a runaway-pagination guard: 40 * 300 rows is far more than any window we ask for
CHUNK_DAYS = 31                     # historical backfills are fetched a month at a time (and cached that way)
HISTORICAL_DAYS = 60                # older than this at ingest time: history, not news (see ``_frame``)
LIVE_REFRESH_DAYS = 3               # windows ending within this many days of today are re-downloaded, older ones are final

COLUMNS = ["txn_key", "txn_date", "team_id", "team_abbr", "kind", "subject_type", "person", "player_id", "other_team_id",
           "detail", "description", "first_seen", "last_seen"]
PLAYER_KINDS = ("signed", "resigned", "extended", "converted", "waived", "claimed", "trade_in", "trade_out")
STAFF_KINDS = ("hired", "fired", "resigned_staff", "extended_staff")


class EspnTransactionsError(RuntimeError):
    """The feed's shape is not what this parser was built against, in a way the CLI should stop and report."""


# --------------------------------------------------------------------------- teams

# ESPN's abbreviation (its own scheme: GS, NO, NY, SA, UTAH, WSH) -> canonical NBA team_id and the NBA abbreviation.
ESPN_TEAMS: dict[str, tuple[int, str]] = {
    "ATL": (1610612737, "ATL"), "BKN": (1610612751, "BKN"), "BOS": (1610612738, "BOS"), "CHA": (1610612766, "CHA"),
    "CHI": (1610612741, "CHI"), "CLE": (1610612739, "CLE"), "DAL": (1610612742, "DAL"), "DEN": (1610612743, "DEN"),
    "DET": (1610612765, "DET"), "GS": (1610612744, "GSW"), "HOU": (1610612745, "HOU"), "IND": (1610612754, "IND"),
    "LAC": (1610612746, "LAC"), "LAL": (1610612747, "LAL"), "MEM": (1610612763, "MEM"), "MIA": (1610612748, "MIA"),
    "MIL": (1610612749, "MIL"), "MIN": (1610612750, "MIN"), "NO": (1610612740, "NOP"), "NY": (1610612752, "NYK"),
    "OKC": (1610612760, "OKC"), "ORL": (1610612753, "ORL"), "PHI": (1610612755, "PHI"), "PHX": (1610612756, "PHX"),
    "POR": (1610612757, "POR"), "SA": (1610612759, "SAS"), "SAC": (1610612758, "SAC"), "TOR": (1610612761, "TOR"),
    "UTAH": (1610612762, "UTA"), "WSH": (1610612764, "WAS"),
    # historical franchises ESPN used in older rows
    "NJ": (1610612751, "BKN"), "NOH": (1610612740, "NOP"), "NOK": (1610612740, "NOP"), "SEA": (1610612760, "OKC"),
    "CHO": (1610612766, "CHA"), "UTA": (1610612762, "UTA"), "GSW": (1610612744, "GSW"), "NYK": (1610612752, "NYK"),
    "SAS": (1610612759, "SAS"), "WAS": (1610612764, "WAS"), "NOP": (1610612740, "NOP"), "BRK": (1610612751, "BKN"),
}

# Counterparty names as they appear inside a description ("from Atlanta", "from the L.A. Clippers", "from Cleveland").
_LOCATIONS = {
    "atlanta": "ATL", "brooklyn": "BKN", "boston": "BOS", "charlotte": "CHA", "chicago": "CHI", "cleveland": "CLE",
    "dallas": "DAL", "denver": "DEN", "detroit": "DET", "golden state": "GS", "houston": "HOU", "indiana": "IND",
    "memphis": "MEM", "miami": "MIA", "milwaukee": "MIL", "minnesota": "MIN", "new orleans": "NO", "new york": "NY",
    "oklahoma city": "OKC", "orlando": "ORL", "philadelphia": "PHI", "phoenix": "PHX", "portland": "POR",
    "sacramento": "SAC", "san antonio": "SA", "toronto": "TOR", "utah": "UTAH", "washington": "WSH",
}
_FULL_NAMES = {
    "los angeles lakers": "LAL", "la lakers": "LAL", "lakers": "LAL", "los angeles clippers": "LAC",
    "la clippers": "LAC", "clippers": "LAC", "sixers": "PHI", "76ers": "PHI", "blazers": "POR", "trail blazers": "POR",
    "cavs": "CLE", "wolves": "MIN", "mavs": "DAL",
}


def team_from_text(text: str) -> tuple[int, str] | None:
    """Resolve a counterparty phrase to ``(team_id, nba_abbr)``, or None (never guess an ambiguous 'Los Angeles')."""
    s = re.sub(r"[.']", "", str(text)).lower().strip()
    s = re.sub(r"^(?:the|a)\s+", "", s)
    s = re.sub(r"\s+", " ", s)
    if not s:
        return None
    if s in _FULL_NAMES:
        return ESPN_TEAMS[_FULL_NAMES[s]]
    for loc in sorted(_LOCATIONS, key=len, reverse=True):
        if s == loc or s.startswith(loc + " "):
            return ESPN_TEAMS[_LOCATIONS[loc]]
    return None


def team_from_espn(abbr: str | None) -> tuple[int, str] | None:
    return ESPN_TEAMS.get(str(abbr or "").upper())


# --------------------------------------------------------------------------- sentence grammar

_VERBS = ("Signed", "Re-signed", "Resigned", "Waived", "Claimed", "Acquired", "Converted", "Hired", "Fired", "Announced",
          "Placed", "Named", "Promoted", "Traded", "Released", "Received", "Agreed", "Extended", "Sent", "Assigned",
          "Recalled", "Activated", "Reassigned", "Exercised", "Declined", "Waive", "Draft")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!])\s+(?=(?:" + "|".join(re.escape(v) for v in _VERBS) + r")\b)")

_POS = r"(?:PGs?|SGs?|SFs?|PFs?|G[-/]Fs?|F[-/]Cs?|G[-/]Cs?|C[-/]Fs?|F[-/]Gs?|C[-/]Gs?|Gs?|Fs?|Cs?|guards?|forwards?|centers?|centres?)"
_POS_PREFIX = re.compile(r"^" + _POS + r"\s+(?=[A-Z])")
_NON_PLAYER = re.compile(
    r"\b(?:cash|draft consideration|draft considerations|pick|picks|swap|rights|trade exception|exception|"
    r"considerations?|first-round|second-round|protected|future)\b", re.I)
_ITEM_SPLIT = re.compile(r",\s*(?:and\s+)?|\s+and\s+")

_CONTRACT_RE = re.compile(r"\b(two-way|rookie[- ]scale|exhibit\s*10|exhibit\s*9|10-day|hardship|training camp|"
                          r"minimum|non-guaranteed|multi-year|one-year|veteran|extension)\b", re.I)


def _split_items(text: str) -> list[str]:
    return [i.strip(" .") for i in _ITEM_SPLIT.split(text) if i.strip(" .")]


_NAME_TAIL = re.compile(r"\s+(?:for the remainder|for the rest|through\b|off waivers|after\b|from\b|in a\b|in an\b|in exchange|and sent\b|"
                       r"to a\b|on a\b|until\b|with\b|\()", re.I)
_TEAM_PREFIX = re.compile(r"^(?:[A-Z][A-Za-z.]*\s+){1,3}(?=" + _POS + r"\s+[A-Z])")
_PARTICLES = {"de", "da", "di", "van", "von", "der", "den", "la", "le", "del", "dos", "el", "bin", "al", "ben", "st.", "mc"}
_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}


def valid_name(name: str) -> bool:
    """A person's name: 2 to 5 tokens, each capitalised (or a particle or suffix). Anything with a stray lowercase word in it is
    text the grammar failed to cut, and is refused rather than recorded as a person."""
    toks = name.split()
    if not 2 <= len(toks) <= 5:
        return False
    return all(t[0].isupper() or t.lower() in _PARTICLES or t.lower() in _SUFFIXES for t in toks) and not any(t.isdigit() for t in toks)


def clean_names(text: str) -> list[str]:
    """``"Gs Buddy Hield, Ryan Nembhard and cash considerations"`` -> ``["Buddy Hield", "Ryan Nembhard"]``. Anything that does
    not read as a person's name is dropped (see :func:`valid_name`)."""
    out = []
    for item in _split_items(text):
        if _NON_PLAYER.search(item):
            continue
        item = re.sub(r"^" + _POS + r"['’]s\s+", "", item)
        item = _TEAM_PREFIX.sub("", item)
        name = _NAME_TAIL.split(_POS_PREFIX.sub("", item).strip())[0].strip(" ,.")
        if valid_name(name):
            out.append(name)
    return out


def contract_detail(text: str) -> str:
    m = _CONTRACT_RE.findall(text or "")
    if not m:
        return ""
    words = [re.sub(r"[- ]", "_", w.lower()) for w in m]
    # 'veteran' + 'extension' collapse to one label; keep first occurrence order, drop duplicates
    seen: list[str] = []
    for w in words:
        if w not in seen:
            seen.append(w)
    return "+".join(seen)


@dataclass
class Action:
    kind: str
    person: str
    subject_type: str = "player"
    other_team: tuple[int, str] | None = None
    detail: str = ""


def _staff_role(text: str) -> str:
    t = text.lower()
    if "head coach" in t:
        return "head_coach"
    if "assistant coach" in t or "associate head coach" in t:
        return "assistant_coach"
    if "general manager" in t:
        return "general_manager"
    if "president" in t or "basketball operations" in t:
        return "president_ops"
    if "coach" in t:
        return "head_coach" if t.strip() in ("coach", "the coach") else "coach_other"
    return "front_office"


_HIRE_AS = re.compile(r"^(?:Hired|Named|Promoted)\s+(?P<who>.+?)\s+(?:as|to)\s+(?:the\s+)?(?P<role>.+?)\.?$")
_NAMED_MANY = re.compile(r"^Named\s+(?P<body>.+?)\s+(?P<role>(?:associate |assistant |interim )?(?:head )?coach(?:es)?|assistant general managers?|"
                         r"general manager|president.*|.*basketball operations.*)\.?$", re.I)
_FIRED = re.compile(r"^Fired\s+(?P<body>.+?)\.?$")
_RESIGN = re.compile(r"^Announced\s+the\s+(?:resignation|retirement|departure)\s+of\s+(?P<body>.+?)\.?$")


def _staff_people(body: str) -> list[tuple[str, str]]:
    """'head coach Billy Donovan' / 'executive vice president of basketball operations A B and general manager C D' ->
    [(role, name)]. Splits on ' and <role words> ' boundaries, then peels the role prefix off each piece."""
    pieces = re.split(r"\s+and\s+(?=(?:interim |assistant |associate |head |general |executive |senior |vice |president))", body)
    out = []
    for piece in pieces:
        m = re.match(r"^(?P<role>(?:[a-z][a-z-]*\s+)+?)(?P<name>[A-Z][\w.'’\- ]+)$", piece.strip())
        if m and valid_name(m.group("name").strip()):
            out.append((_staff_role(m.group("role")), m.group("name").strip()))
    return out


def parse_staff_sentence(s: str) -> list[Action] | None:
    """Coach / front-office sentences. Returns None if ``s`` is not one (the caller then tries the player grammar)."""
    m = re.match(r"^(?:Signed|Re-signed|Extended)\s+(?P<body>(?:(?:interim|assistant|associate|head)\s+)*coach\s+[A-Z].+?)\s+to\s+(?:a|an)\s+.*(?:extension|contract).*?\.?$", s)
    if m:
        return [Action("extended_staff", n, "staff", detail=r) for r, n in _staff_people(m.group("body"))] or []
    m = _RESIGN.match(s)
    if m:
        return [Action("resigned_staff", n, "staff", detail=r) for r, n in _staff_people(m.group("body"))] or []
    m = _FIRED.match(s)
    if m and not _POS_PREFIX.match(m.group("body")):
        return [Action("fired", n, "staff", detail=r) for r, n in _staff_people(m.group("body"))] or []
    m = _NAMED_MANY.match(s)
    if m and re.search(r"coach|manager|president|operations", m.group("role"), re.I):
        if re.search(r"\s(?:as|and (?:[A-Z][\w.'-]+ ){1,3}(?:head|assistant|associate|interim))\s", " " + m.group("body") + " "):
            return None                                       # several roles in one sentence: refuse rather than guess who is which
        names = [n.strip() for n in re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group("body")) if valid_name(n.strip())]
        rt = m.group("role").lower().strip(" .")
        if "assistant" in rt:
            role = "assistant_coach" if "general manager" not in rt else "general_manager"
        elif rt in ("coach", "head coach"):
            role = "head_coach" if len(names) == 1 else "coach_other"      # two people cannot both be "the" head coach
        else:
            role = _staff_role(rt)
        return [Action("hired", n, "staff", detail=role) for n in names]
    m = _HIRE_AS.match(s)
    if m and re.search(r"coach|manager|president|operations|director|governor", m.group("role"), re.I):
        if re.search(r",|\bto\b", m.group("role")) or (s.startswith("Promoted") and re.search(r"\band\b", m.group("role"))):
            return None                                       # a multi-appointment sentence: not one hire
        who = m.group("who").strip()
        if s.startswith("Promoted"):
            who = re.sub(r"^(?:(?:assistant|associate|interim|head)\s+)*(?:coach|general manager|director\b[^A-Z]*)\s+", "", who)
        return [Action("hired", who, "staff", detail=_staff_role(m.group("role")))] if valid_name(who) else []
    return None


_TEAM_TAIL = r"(?:the\s+)?[A-Z][A-Za-z.'’]*(?:\s+[A-Z][A-Za-z.'’]*){0,2}"
_FROM = re.compile(r"\bfrom\s+(?P<team>" + _TEAM_TAIL + r")")


def _acquired(body: str) -> list[Action]:
    """``Acquired <in> [from Team] [in exchange for|for <out>] [and sent <out> to Team] [in a three-team trade ...]``."""
    acts: list[Action] = []
    main = re.split(r"\s+in a (?:three|four)[- ]team trade", body)[0]
    sent_out = re.search(r"(?:,\s*and|\s+and|,|\bthat)?\s*sent\s+(?P<out>.+?)\s+to\s+(?P<team>" + _TEAM_TAIL + r")", body)
    ex = re.search(r"\s+(?:in exchange(?:d)?(?:\s+for)?|for)\s+(?P<out>.+)$", main)
    incoming = main
    if ex:
        incoming = main[:ex.start()]
        from_teams = {t for mm in _FROM.finditer(incoming) if (t := team_from_text(mm.group("team")))}
        three = bool(re.search(r"(?:three|four)[- ]team", body))
        back = next(iter(from_teams)) if len(from_teams) == 1 and not three else None   # a guess only with exactly one candidate
        out_part = re.split(r"\s+and sent\s+", ex.group("out"))[0]
        for n in clean_names(out_part):
            acts.append(Action("trade_out", n, other_team=back))
        sent_after = re.search(r"\s+and sent\s+(?P<o>.+?)\s+to\s+(?P<t>" + _TEAM_TAIL + r")", ex.group("out"))
        if sent_after:
            for n in clean_names(sent_after.group("o")):
                acts.append(Action("trade_out", n, other_team=team_from_text(sent_after.group("t"))))
    sm = re.search(r"\s+and sent\s+.+$", incoming)
    if sm:
        incoming = incoming[:sm.start()]
    # "X from T1 and Y from T2": split where a position-tagged item follows an 'and'
    pieces = re.split(r"\s+and\s+(?=" + _POS + r"\s+[A-Z])", incoming)
    teams = {t for p in pieces for t in [team_from_text(mm.group("team")) for mm in _FROM.finditer(p)] if t}
    shared = next(iter(teams)) if len(teams) == 1 else None
    for p in pieces:
        fm = _FROM.search(p)
        other = team_from_text(fm.group("team")) if fm else shared
        names_part = p[:fm.start()] if fm else p
        for n in clean_names(names_part):
            acts.append(Action("trade_in", n, other_team=other))
    if sent_out and not ex:
        other = team_from_text(sent_out.group("team"))
        for n in clean_names(sent_out.group("out")):
            acts.append(Action("trade_out", n, other_team=other))
    return acts


def parse_sentence(s: str) -> list[Action] | None:
    """One sentence -> actions; ``[]`` if it is understood but has no person (picks/cash only); None if unrecognised."""
    s = s.strip()
    if not s:
        return []
    staff = parse_staff_sentence(s)
    if staff is not None:
        return staff
    verb = s.split()[0].rstrip(".")
    body = s[len(s.split()[0]):].strip().rstrip(".")
    if verb in ("Signed", "Re-signed", "Resigned"):
        out: list[Action] = []
        clauses = re.split(r",?\s+and\s+(?=" + _POS + r"\s+[A-Z])", body)          # each position-tagged clause has its own contract
        shared = ""
        if len(clauses) > 1 and " to " in clauses[-1] and re.search(r"contracts\b", clauses[-1]):
            shared = clauses[-1].partition(" to ")[2]                                # "F A and G B to rookie scale contracts"
        for clause in clauses:
            names_part, _, contract = clause.partition(" to ")
            own_terms = _NAME_TAIL.search(names_part) is not None                    # "for the remainder of the season", "through ..."
            detail = contract_detail(contract or ("" if own_terms else shared))
            kind = "resigned" if verb != "Signed" else "signed"
            if "extension" in detail:
                kind = "extended"
            out += [Action(kind, n, detail=detail) for n in clean_names(names_part)]
        return out
    if verb == "Extended":
        return [Action("extended", n, detail=contract_detail(body)) for n in clean_names(re.split(r"\s+(?:to|through)\s", body)[0])]
    if verb == "Waived":
        return [Action("waived", n) for n in clean_names(body)]
    if verb == "Placed":
        m = re.match(r"^(?P<who>.+?)\s+on\s+waivers$", body)
        return [Action("waived", n) for n in clean_names(m.group("who"))] if m else None
    if verb == "Claimed":
        return [Action("claimed", n) for n in clean_names(re.sub(r"\s+off\s+waivers.*$", "", body))]
    if verb == "Converted":
        m = re.match(r"^(?:the\s+contract\s+of\s+)?(?P<who>.+?)\s+to\s+(?:an?\s+)?(?P<to>.+)$", body)
        return [Action("converted", n, detail="to_nba" if "NBA" in m.group("to") else contract_detail(m.group("to")))
                for n in clean_names(m.group("who"))] if m else None
    if verb == "Acquired":
        return _acquired(body)
    if verb == "Traded":
        m = re.match(r"^(?P<out>.+?)\s+to\s+(?P<team>" + _TEAM_TAIL + r")(?:\s+(?:in exchange(?:\s+for)?|for)\s+(?P<in>.+))?$", body)
        if not m:
            return None
        other = team_from_text(m.group("team"))
        acts = [Action("trade_out", n, other_team=other) for n in clean_names(m.group("out"))]
        acts += [Action("trade_in", n, other_team=other) for n in clean_names(m.group("in") or "")]
        return acts
    if verb in ("Received", "Sent"):
        return []                                            # mirrors of an Acquired sentence, or picks/cash only
    return None


_COMPOUND = re.compile(r",?\s+and\s+(?=(?:waived|signed|released|claimed|re-signed)\s)", re.I)


def split_sentences(description: str) -> list[str]:
    out: list[str] = []
    for part in _SENTENCE_SPLIT.split(str(description).strip()):
        for piece in _COMPOUND.split(part.strip()):
            piece = piece.strip()
            if piece:
                out.append(piece[0].upper() + piece[1:])
    return out


def parse_description(description: str) -> tuple[list[Action], int]:
    """All actions in one feed row and the number of sentences that were not understood."""
    actions: list[Action] = []
    unparsed = 0
    for sent in split_sentences(description):
        parsed = parse_sentence(sent)
        if parsed is None:
            unparsed += 1
        else:
            actions.extend(parsed)
    return actions, unparsed


# --------------------------------------------------------------------------- feed rows -> ledger rows

def txn_key(txn_date: str, team_id: int, kind: str, person: str) -> str:
    """Stable across parser improvements: it uses only what the feed itself says (day, team, action, person), never
    anything the parser infers (counterparty), so a better parse updates a row instead of duplicating it."""
    return hashlib.sha1("|".join([txn_date, str(team_id), kind, normalize_name(person)]).encode()).hexdigest()[:16]


def feed_rows_to_records(feed: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Feed rows (``date``, ``description``, ``team``) -> one record per person-action (``player_id`` still unset)."""
    records: list[dict[str, Any]] = []
    stats = {"rows": 0, "unknown_team": 0, "unparsed_sentences": 0, "other_rows": 0}
    for row in feed:
        stats["rows"] += 1
        team = team_from_espn((row.get("team") or {}).get("abbreviation"))
        if team is None:
            stats["unknown_team"] += 1
            continue
        day = str(row.get("date", ""))[:10]
        desc = str(row.get("description", "")).strip()
        if not day or not desc:
            continue
        actions, unparsed = parse_description(desc)
        stats["unparsed_sentences"] += unparsed
        if not actions:
            actions = [Action("other", "")]
            stats["other_rows"] += 1
        seen: set[str] = set()
        for a in actions:
            key = txn_key(day, team[0], a.kind, a.person or desc)
            if key in seen:
                continue
            seen.add(key)
            records.append({"txn_key": key, "txn_date": pd.Timestamp(day), "team_id": team[0], "team_abbr": team[1],
                            "kind": a.kind, "subject_type": a.subject_type, "person": a.person, "player_id": None,
                            "other_team_id": a.other_team[0] if a.other_team else None, "detail": a.detail,
                            "description": desc})
    return records, stats


# --------------------------------------------------------------------------- fetching

def _yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def windows(start: date, end: date, chunk_days: int = CHUNK_DAYS) -> list[tuple[date, date]]:
    out, cur = [], start
    while cur <= end:
        nxt = min(cur + timedelta(days=chunk_days - 1), end)
        out.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    return out


def fetch_window(client: CachedHttpClient, start: date, end: date, *, today: date | None = None) -> list[dict[str, Any]]:
    """Every feed row in ``[start, end]`` (all pages). Windows that touch the last few days re-download (the feed is
    still filling in); older windows are final and served from the cache."""
    today = today or date.today()
    live = end >= today - timedelta(days=LIVE_REFRESH_DAYS)
    rows: list[dict[str, Any]] = []
    for page in range(1, MAX_PAGES + 1):
        params = {"dates": f"{_yyyymmdd(start)}-{_yyyymmdd(end)}", "limit": PAGE_SIZE, "page": page}
        payload = client.get_json(CACHE_LABEL, API_URL, params, refresh=live and not client.offline)
        if isinstance(payload, dict) and "transactions" not in payload and payload.get("count") == 0 and "pageIndex" in payload:
            break                                     # a window with no transactions omits the key entirely (seen for Oct 2018)
        if not isinstance(payload, dict) or "transactions" not in payload:
            raise EspnTransactionsError(f"unexpected transactions payload for {params}: keys {list(payload)[:6] if isinstance(payload, dict) else type(payload)}")
        rows.extend(payload["transactions"])
        if page >= int(payload.get("pageCount") or 1):
            break
    else:
        raise EspnTransactionsError(f"more than {MAX_PAGES} pages for {start}..{end}; narrow the window")
    return rows


# --------------------------------------------------------------------------- matching and the ledger

def _name_universe(base: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(active, everyone)``. ``active`` is who can plausibly appear in a current move: ``players`` rows still playing
    within a season of the newest data, plus everyone on a roster snapshot. ``everyone`` adds ``player_profiles`` (the
    all-time ``playerindex``: undrafted signees and two-way players, but also retired namesakes such as Tim Hardaway
    Sr.). Matching tries ``active`` first, so a father and son resolve to the one playing now."""
    from src.store import read_table, table_exists

    def norm(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in ("from_year", "to_year"):
            df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else float("nan")
        df["player_id"] = df["player_id"].astype("int64")
        return df[["player_id", "player_name", "from_year", "to_year"]].drop_duplicates("player_id")

    active: list[pd.DataFrame] = []
    extra: list[pd.DataFrame] = []
    if table_exists("players", base):
        p = read_table("players", base, validate=False)
        to = pd.to_numeric(p["to_year"], errors="coerce")
        active.append(norm(p[to.isna() | (to >= to.max() - 1)]))
        extra.append(norm(p))
    for name, bucket in (("roster_snapshots", active), ("player_profiles", extra)):
        path = base / "processed" / f"{name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if {"player_id", "player_name"} <= set(df.columns):
                bucket.append(norm(df))
    if not active and not extra:
        raise EspnTransactionsError("no player names available to match against; run the NBA ingest first")
    act = pd.concat(active, ignore_index=True).drop_duplicates("player_id") if active else norm(
        pd.DataFrame({"player_id": pd.Series(dtype="int64"), "player_name": pd.Series(dtype="object")}))
    every = pd.concat([act] + extra, ignore_index=True).drop_duplicates("player_id")
    return act, every


def attach_player_ids(records: list[dict[str, Any]], universe: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fill ``player_id`` on the player-subject records; returns (match summary, unresolved names)."""
    active, everyone = universe
    latest: dict[str, pd.Timestamp] = {}
    for r in records:
        if r["subject_type"] == "player" and r["person"]:
            latest[r["person"]] = max(latest.get(r["person"], r["txn_date"]), r["txn_date"])
    names = sorted(latest)
    if not names:
        return {"names": 0, "matched": 0}, []
    # the league year of the move breaks ties between a player and a same-named relative
    year = {n: t.year - (1 if t.month < 7 else 0) for n, t in latest.items()}

    def run(pool: list[str], uni: pd.DataFrame):
        src = pd.DataFrame({"source_id": pool, "name": pool, "season_start": [year[n] for n in pool]})
        return match_players(src, uni, source=SOURCE)

    lookup: dict[str, int] = {}
    report = None
    if len(active):
        found, report = run(names, active)
        lookup.update({str(r.source_id): int(r.player_id) for r in found.itertuples(index=False)})
    rest = [n for n in names if n not in lookup]
    if rest:
        found, report = run(rest, everyone)
        lookup.update({str(r.source_id): int(r.player_id) for r in found.itertuples(index=False)})
    for r in records:
        if r["subject_type"] == "player" and r["person"]:
            r["player_id"] = lookup.get(r["person"])
    unresolved = ([{"name": u["name"], "why": "unmatched"} for u in report.unmatched]
                  + [{"name": a["name"], "why": "ambiguous"} for a in report.ambiguous]) if rest and report else []
    n_amb = sum(1 for u in unresolved if u["why"] == "ambiguous")
    return {"names": len(names), "matched": len(lookup), "match_rate": round(len(lookup) / len(names), 4),
            "summary": f"{len(lookup)}/{len(names)} matched ({len(lookup) / len(names):.1%}); "
                       f"{n_amb} ambiguous, {len(unresolved) - n_amb} unmatched"}, unresolved


def ledger_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / "processed" / f"{TABLE}.parquet"


def read_ledger(base: Path | None = None) -> pd.DataFrame:
    path = ledger_path(base)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.ingest.espn_transactions`")
    return pd.read_parquet(path)


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _frame(records: list[dict[str, Any]], now: datetime) -> pd.DataFrame:
    """Records -> ledger rows. A record dated more than ``HISTORICAL_DAYS`` before ``now`` is history, not news (a
    backfill, or the first run's long lookback): its ``first_seen`` is its own date, so "new since I last looked" never
    floods with old moves."""
    df = pd.DataFrame(records, columns=COLUMNS[:-2])
    stamp = pd.Timestamp(now)
    stamp = (stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")).tz_localize(None)   # naive = UTC
    df["last_seen"] = stamp
    df["first_seen"] = stamp
    old = pd.to_datetime(df["txn_date"]) < stamp.normalize() - pd.Timedelta(days=HISTORICAL_DAYS)
    df.loc[old, "first_seen"] = pd.to_datetime(df.loc[old, "txn_date"])
    return _typed(df)


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["txn_date"] = pd.to_datetime(df["txn_date"]).astype("datetime64[ms]")
    df["team_id"] = df["team_id"].astype("int64")
    for c in ("player_id", "other_team_id"):
        df[c] = pd.array(pd.to_numeric(df[c], errors="coerce"), dtype="Int64")
    for c in ("first_seen", "last_seen"):
        df[c] = pd.to_datetime(df[c]).astype("datetime64[ms]")
    for c in ("txn_key", "team_abbr", "kind", "subject_type", "person", "detail", "description"):
        df[c] = df[c].fillna("").astype(str)
    return df[COLUMNS]


def merge_ledger(existing: pd.DataFrame | None, new: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Append-only merge: a key already in the ledger keeps its ``first_seen``, gets a fresh ``last_seen`` and any newly
    resolved ``player_id``; a new key is appended. Returns (ledger, number of new rows)."""
    new = new.drop_duplicates("txn_key")
    if existing is None or existing.empty:
        return new.sort_values(["txn_date", "team_id", "txn_key"], kind="mergesort").reset_index(drop=True), len(new)
    # A row of an earlier parse whose feed row (same day, team, text) is now read differently is superseded: dropped, and the
    # rows that replace it inherit the earliest ``first_seen`` so a better parser never re-announces old news.
    sig = ["txn_date", "team_id", "description"]
    seen_now = new[sig].drop_duplicates()
    tagged = existing.merge(seen_now.assign(_now=True), on=sig, how="left")
    stale = tagged["_now"].fillna(False).to_numpy(bool) & ~existing["txn_key"].isin(set(new["txn_key"])).to_numpy(bool)
    if stale.any():
        first = existing[stale].groupby(sig)["first_seen"].min().rename("_inherit").reset_index()
        new = new.merge(first, on=sig, how="left")
        new["first_seen"] = new["_inherit"].where(new["_inherit"].notna() & ~new["txn_key"].isin(set(existing["txn_key"])), new["first_seen"])
        new = new.drop(columns="_inherit")
        existing = existing[~stale]
    old = existing.set_index("txn_key")
    fresh = new.drop_duplicates("txn_key").set_index("txn_key")
    both = old.index.intersection(fresh.index)
    old.loc[both, "last_seen"] = fresh.loc[both, "last_seen"]
    for col in ("player_id", "other_team_id"):            # a better parse or a later id-map fills gaps, never overwrites
        fill = old.loc[both, col].isna() & fresh.loc[both, col].notna()
        old.loc[fill[fill].index, col] = fresh.loc[fill[fill].index, col]
    added = fresh.loc[fresh.index.difference(old.index)]
    merged = pd.concat([old.reset_index(), added.reset_index()], ignore_index=True)
    merged = _typed(merged).sort_values(["txn_date", "team_id", "txn_key"], kind="mergesort").reset_index(drop=True)
    return merged, len(added)


@dataclass
class IngestResult:
    start: date
    end: date
    feed_rows: int = 0
    records: int = 0
    new_rows: int = 0
    ledger_rows: int = 0
    stats: dict[str, int] = field(default_factory=dict)
    match: dict[str, Any] = field(default_factory=dict)
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    by_kind: dict[str, int] = field(default_factory=dict)
    network_requests: int = 0
    cache_hits: int = 0


def run_ingest(start: date, end: date, client: CachedHttpClient, base: Path | None = None, *, now: datetime | None = None,
               log=print) -> IngestResult:
    if end < start:
        raise EspnTransactionsError(f"reversed window {start}..{end}")
    base = base or data_dir()
    now = now or datetime.now(timezone.utc)
    res = IngestResult(start, end)
    feed: list[dict[str, Any]] = []
    for a, b in windows(start, end):
        chunk = fetch_window(client, a, b, today=now.date())
        log(f"  {a}..{b}: {len(chunk)} feed rows")
        feed.extend(chunk)
    res.feed_rows = len(feed)
    records, res.stats = feed_rows_to_records(feed)
    res.records = len(records)
    if records:
        res.match, res.unresolved = attach_player_ids(records, _name_universe(base))
    res.by_kind = pd.Series([r["kind"] for r in records], dtype="object").value_counts().to_dict() if records else {}
    if records:
        merged, res.new_rows = merge_ledger(read_ledger(base) if ledger_path(base).exists() else None, _frame(records, now))
        _atomic_parquet(merged, ledger_path(base))
        res.ledger_rows = len(merged)
    res.network_requests, res.cache_hits = client.stats.network_requests, client.stats.cache_hits
    report = {"window": [str(start), str(end)], "feed_rows": res.feed_rows, "records": res.records, "new_rows": res.new_rows,
              "ledger_rows": res.ledger_rows, "parse": res.stats, "matching": res.match, "by_kind": res.by_kind,
              "unresolved_names": res.unresolved[:200], "at": now.isoformat()}
    rp = base / "processed" / REPORT_NAME
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return res


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.ingest.espn_transactions", description=__doc__.split("\n\n")[0])
    p.add_argument("--days", type=int, default=90, help="ingest the last N days (default 90; ignored with --from)")
    p.add_argument("--from", dest="start", default=None, help="window start, YYYY-MM-DD (for a historical backfill)")
    p.add_argument("--to", dest="end", default=None, help="window end, YYYY-MM-DD (default today)")
    p.add_argument("--offline", action="store_true", help="never touch the network (also NBA_OFFLINE=1)")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--min-interval", type=float, default=2.0, help="minimum seconds between network requests (default 2)")
    return p


def main(argv: list[str] | None = None, *, client: CachedHttpClient | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    base = args.data_dir or data_dir()
    end = date.fromisoformat(args.end) if args.end else date.today()
    start = date.fromisoformat(args.start) if args.start else end - timedelta(days=args.days)
    if client is None:
        client = CachedHttpClient(default_cache_dir("espn") if args.data_dir is None else base / "raw" / "espn",
                                  offline=True if args.offline else None, min_interval=args.min_interval)
    try:
        res = run_ingest(start, end, client, base)
    except (EspnTransactionsError, HttpCacheError, ContractError, FileNotFoundError) as exc:
        print(f"espn transactions ingest failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"done: {res.feed_rows} feed rows -> {res.records} records ({res.new_rows} new; ledger {res.ledger_rows}); "
          f"{res.network_requests} network requests, {res.cache_hits} cache hits")
    print(f"parse: {res.stats}; kinds: {res.by_kind}")
    if res.match:
        print(f"players: {res.match.get('summary', res.match)}")
    if res.unresolved:
        print(f"unresolved names ({len(res.unresolved)}): {', '.join(u['name'] for u in res.unresolved[:15])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
