"""Head coach per team-season from Wikipedia -> the ad-hoc ``team_coaches`` table (ADR 0020).

    python -m src.ingest.wiki_coaches --seasons 2015-16:2025-26 [--offline] [--refresh]   # completed seasons (infoboxes)
    python -m src.ingest.wiki_coaches --current 2026-27                                    # the live season (one request)

**Completed seasons.** The 330 team-season pages ``wiki_transactions`` already caches (CC BY-SA 4.0, read-only ``action=raw``,
descriptive UA, >= 1 s between requests, ADR 0011) carry an infobox with a ``| coach = ...`` field: the head coach, and on a
mid-season change every coach in order with a tag, e.g. ``[[Jacque Vaughn]] (fired)<br>[[Kevin Ollie]] (interim)``. This module
reads that field and nothing else and re-uses the same cache, so a run after ``wiki_transactions`` makes no new requests.

**The live season.** A season that has not started has no team-season page. "List of current NBA head coaches" has every team's
head coach and start date, and it is the source that already reflects this offseason's hires (verified 2026-09-25: it names
Tiago Splitter at Chicago, Sean Sweeney at Orlando, Micah Nori at Portland and Jamahl Mosley at New Orleans, none of which
ESPN's transactions feed carries for Chicago or Orlando). It is re-downloaded on each ``--current`` run, and a run that does not
yield all 30 teams writes nothing.

One row per (season, team, coach in listed order): ``season, team_id, seq, coach_name, coach_key, status, is_opening,
n_coaches, source, start_date``. ``is_opening`` is the first-listed coach: the one the season began with, which is what a
preseason draft decision can know (and what the projection layer must use). ``status`` is ``""`` (no tag), ``fired``,
``resigned``, ``interim``, ``acting`` (an absence: leave, illness), ``left`` ("until <date>") or ``other``. Two hazards are
handled by *reporting*, not guessing: a season whose infobox has no coach field, and the honest ambiguity of a first-listed coach
who was absent for part of the season (2015-16 Golden State: Kerr, with Luke Walton 39-4 as interim). ``n_coaches > 1`` marks
every such season so an analysis can exclude them.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.contracts import ContractError, data_dir
from src.ingest.http_cache import CachedHttpClient, HttpCacheError, default_cache_dir
from src.ingest.id_map import normalize_name
from src.ingest.wiki_transactions import (CACHE_DIR_NAME, TEAM_WIKI_NAMES, UA, WIKI_API_URL, fetch_team_season_wikitext,
                                          parse_seasons)

TABLE = "team_coaches"
COLUMNS = ["season", "team_id", "seq", "coach_name", "coach_key", "status", "is_opening", "n_coaches", "source", "start_date"]
SOURCE_INFOBOX = "wikipedia_infobox"
SOURCE_CURRENT = "wikipedia_current_list"
CURRENT_LIST_TITLE = "List of current NBA head coaches"


class WikiCoachesError(RuntimeError):
    pass


# Seasons whose infobox has no coach field. Each entry is a fact checked against the team's own season-page prose, not a guess:
# 2015-16 Cleveland began with David Blatt, who was fired on 2016-01-22 and replaced by Tyronn Lue.
OVERRIDES: dict[tuple[str, int], list[tuple[str, str]]] = {
    ("2015-16", 1610612739): [("David Blatt", "fired"), ("Tyronn Lue", "")],
}

_COACH_FIELD = re.compile(r"(?ms)^\|\s*coach\s*=\s*(.*?)(?=^\|\s*[A-Za-z_]+\s*=|^\}\})")
_LINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
_REF = re.compile(r"<ref\b[^>]*?/>|<ref\b[^>]*>.*?</ref>", re.I | re.S)
_EFN = re.compile(r"\{\{efn\b(?:[^{}]|\{\{[^{}]*\}\})*\}\}", re.I | re.S)


@dataclass
class CoachEntry:
    name: str
    status: str


def _status(tag_text: str) -> str:
    t = tag_text.lower()
    if "interim" in t:
        return "interim"
    if "acting" in t:
        return "acting"
    if "fired" in t:
        return "fired"
    if "resign" in t:
        return "resigned"
    if "leave" in t or "surgery" in t or "illness" in t:
        return "acting"
    if "until" in t or "through" in t:
        return "left"
    return "other" if t.strip() else ""


def parse_coach_field(wikitext: str) -> list[CoachEntry]:
    """The ``coach`` infobox field -> coaches in listed order with a status; ``[]`` if there is no such field."""
    m = _COACH_FIELD.search(wikitext)
    if not m:
        return []
    body = _EFN.sub("", _REF.sub("", m.group(1)))
    body = re.sub(r"\{\{\s*ubl\s*\|", "", body, flags=re.I).replace("}}", "")
    links = list(_LINK.finditer(body))
    out: list[CoachEntry] = []
    for i, lk in enumerate(links):
        name = (lk.group(2) or lk.group(1)).strip()
        end = links[i + 1].start() if i + 1 < len(links) else len(body)
        tag = re.sub(r"<[^>]+>|\|", " ", body[lk.end():end])
        out.append(CoachEntry(name, _status(re.sub(r"\[\[.*", "", tag))))
    return out


def coach_key(name: str) -> str:
    """Stable identity across seasons: accent/punctuation/suffix-insensitive (``Wes Unseld Jr.`` -> ``wes unseld``)."""
    return normalize_name(name)


def page_rows(wikitext: str, team_id: int, season: str) -> tuple[list[dict], str]:
    """(rows, how): ``how`` is ``infobox``, ``override`` or ``missing``."""
    entries = parse_coach_field(wikitext)
    how = "infobox"
    if not entries and (season, team_id) in OVERRIDES:
        entries, how = [CoachEntry(n, s) for n, s in OVERRIDES[(season, team_id)]], "override"
    if not entries:
        return [], "missing"
    return [{"season": season, "team_id": team_id, "seq": i, "coach_name": e.name, "coach_key": coach_key(e.name),
             "status": e.status, "is_opening": i == 0, "n_coaches": len(entries), "source": SOURCE_INFOBOX,
             "start_date": pd.NaT} for i, e in enumerate(entries)], how


# --------------------------------------------------------------------------- the live season

_LIST_TEAM = re.compile(r"(?m)^\|\s*\[\[([^\]|]+)(?:\|[^\]]*)?\]\]\s*$")
_SORTNAME = re.compile(r"\{\{\s*sortname\s*\|\s*([^|}]+?)\s*\|\s*([^|}]+?)\s*(?:\|[^}]*)?\}\}", re.I)
_DTS = re.compile(r"\{\{\s*Dts\s*\|\s*(\d{4})\s*\|\s*([A-Za-z]+)\s*\|\s*(\d{1,2})", re.I)
_WIKI_TEAM_ALIASES = {"la clippers": "Los Angeles Clippers"}


def _team_id_from_wiki(name: str) -> int | None:
    n = _WIKI_TEAM_ALIASES.get(name.strip().lower(), name.strip())
    for tid, wiki in TEAM_WIKI_NAMES.items():
        if wiki.lower() == n.lower():
            return tid
    return None


def parse_current_list(wikitext: str) -> list[dict]:
    """Rows of the ``Coaches`` table: ``team_id, coach_name, start_date`` (one head coach per team). A row it cannot read
    completely (no team, coach or date) is skipped and left for the caller's completeness check, never guessed."""
    out = []
    for block in re.split(r"(?m)^\|-.*$", wikitext):
        tm, cm, dm = _LIST_TEAM.search(block), _SORTNAME.search(block), _DTS.search(block)
        if not (tm and cm and dm):
            continue
        tid = _team_id_from_wiki(tm.group(1))
        if tid is None:
            continue
        try:
            start = pd.Timestamp(f"{dm.group(2)} {int(dm.group(3))}, {dm.group(1)}")
        except ValueError:
            continue
        out.append({"team_id": tid, "coach_name": f"{cm.group(1)} {cm.group(2)}".strip(), "start_date": start})
    return out


def current_rows(wikitext: str, season: str) -> list[dict]:
    return [{"season": season, "team_id": r["team_id"], "seq": 0, "coach_name": r["coach_name"],
             "coach_key": coach_key(r["coach_name"]), "status": "", "is_opening": True, "n_coaches": 1,
             "source": SOURCE_CURRENT, "start_date": r["start_date"]} for r in parse_current_list(wikitext)]


def fetch_current_list(client: CachedHttpClient, *, refresh: bool = True) -> str:
    """The list changes whenever a team hires or fires, so it is re-downloaded by default (one request)."""
    return client.get_text("current_coaches", WIKI_API_URL, {"title": CURRENT_LIST_TITLE, "action": "raw"},
                           headers={"User-Agent": UA}, refresh=refresh)


# --------------------------------------------------------------------------- table io

def table_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / "processed" / f"{TABLE}.parquet"


def read_team_coaches(base: Path | None = None) -> pd.DataFrame:
    p = table_path(base)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run `python -m src.ingest.wiki_coaches --seasons 2015-16:2025-26`")
    return pd.read_parquet(p)


def _typed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["team_id"] = df["team_id"].astype("int64")
    df["seq"] = df["seq"].astype("int64")
    df["n_coaches"] = df["n_coaches"].astype("int64")
    df["is_opening"] = df["is_opening"].astype(bool)
    df["start_date"] = pd.to_datetime(df["start_date"]).astype("datetime64[ms]")
    return df[COLUMNS].sort_values(["season", "team_id", "seq"], kind="mergesort").reset_index(drop=True)


def _write(df: pd.DataFrame, base: Path) -> None:
    path = table_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _replace_seasons(new: pd.DataFrame, seasons: set[str], base: Path) -> pd.DataFrame:
    """Write ``new`` in place of ``seasons``; every other season already in the table is kept as it is."""
    p = table_path(base)
    old = pd.read_parquet(p) if p.exists() else None
    parts = [x for x in ((old[~old["season"].isin(seasons)] if old is not None else None), new) if x is not None and len(x)]
    merged = _typed(pd.concat(parts, ignore_index=True))
    _write(merged, base)
    return merged


def run_current(season: str, client: CachedHttpClient, base: Path | None = None, *, refresh: bool = True, log=print) -> dict:
    """Replace ``season``'s rows with the current list. Needs all 30 teams, else nothing is written."""
    base = base or data_dir()
    rows = current_rows(fetch_current_list(client, refresh=refresh and not client.offline), season)
    teams = {r["team_id"] for r in rows}
    if teams != set(TEAM_WIKI_NAMES):
        missing = sorted(TEAM_WIKI_NAMES[t] for t in set(TEAM_WIKI_NAMES) - teams)
        raise WikiCoachesError(f"the current-coaches list gave {len(teams)} of 30 teams (missing {missing}); nothing written")
    _replace_seasons(_typed(pd.DataFrame(rows, columns=COLUMNS)), {season}, base)
    log(f"  {season}: {len(rows)} head coaches from the current list")
    return {"season": season, "teams": len(rows)}


@dataclass
class IngestResult:
    seasons: list[str]
    pages: int = 0
    missing: list[dict] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    overrides: int = 0
    rows: int = 0
    multi_coach_seasons: int = 0
    network_requests: int = 0
    cache_hits: int = 0


def run_ingest(seasons: list[str], client: CachedHttpClient, base: Path | None = None, *, refresh: bool = False,
               log=print) -> IngestResult:
    if not seasons:
        raise WikiCoachesError("no seasons to ingest")
    base = base or data_dir()
    res = IngestResult(seasons=list(seasons))
    rows: list[dict] = []
    for season in seasons:
        for team_id in sorted(TEAM_WIKI_NAMES):
            try:
                text = fetch_team_season_wikitext(client, team_id, season, refresh=refresh)
            except HttpCacheError as exc:
                res.skipped.append({"season": season, "team_id": team_id, "reason": str(exc)})
                log(f"[{season}] {TEAM_WIKI_NAMES[team_id]}: fetch failed ({exc}); skipped")
                continue
            res.pages += 1
            page, how = page_rows(text, team_id, season)
            if how == "missing":
                res.missing.append({"season": season, "team_id": team_id, "team": TEAM_WIKI_NAMES[team_id]})
                log(f"[{season}] {TEAM_WIKI_NAMES[team_id]}: no coach in the infobox and no override")
                continue
            res.overrides += how == "override"
            rows.extend(page)
    if not rows:
        raise WikiCoachesError("no coach rows parsed")
    new = _typed(pd.DataFrame(rows, columns=COLUMNS))
    res.rows = len(new)
    res.multi_coach_seasons = int((new["is_opening"] & (new["n_coaches"] > 1)).sum())
    _replace_seasons(new, set(seasons), base)
    res.network_requests, res.cache_hits = client.stats.network_requests, client.stats.cache_hits
    return res


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.ingest.wiki_coaches", description=__doc__.split("\n\n")[0])
    p.add_argument("--seasons", default=None, help="e.g. 2015-16:2025-26 (infobox coaches of completed seasons)")
    p.add_argument("--current", default=None, metavar="SEASON",
                   help="refresh the live season (e.g. 2026-27) from Wikipedia's list of current head coaches (one request)")
    p.add_argument("--offline", action="store_true", help="never touch the network (also NBA_OFFLINE=1)")
    p.add_argument("--refresh", action="store_true", help="re-download season pages even if cached")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--min-interval", type=float, default=1.0)
    return p


def main(argv: list[str] | None = None, *, client: CachedHttpClient | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.seasons and not args.current:
        parser.error("give --seasons and/or --current")
    try:
        seasons = parse_seasons(args.seasons) if args.seasons else []
    except ValueError as exc:
        parser.error(str(exc))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    base = args.data_dir or data_dir()
    if client is None:
        client = CachedHttpClient(default_cache_dir(CACHE_DIR_NAME), offline=True if args.offline else None,
                                  min_interval=args.min_interval, headers={"User-Agent": UA})
    try:
        if seasons:
            res = run_ingest(seasons, client, base, refresh=args.refresh)
            print(f"done: {res.pages} pages, {res.rows} coach rows ({res.multi_coach_seasons} seasons with a mid-season change, "
                  f"{res.overrides} from an override), {len(res.missing)} without a coach, {len(res.skipped)} skipped; "
                  f"{res.network_requests} network requests, {res.cache_hits} cache hits")
        if args.current:
            run_current(args.current, client, base)
    except (HttpCacheError, WikiCoachesError, ContractError, FileNotFoundError) as exc:
        print(f"wiki coaches ingest failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
