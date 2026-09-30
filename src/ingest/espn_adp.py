"""ESPN historical ADP ingest: pull (cached, polite), detect wiped seasons, map ids, write the
``adp`` file the backtest's ``AdpBenchmark``/``load_adp`` expect, and populate ``player_id_map``.

    python -m src.ingest.espn_adp --seasons 2015-16:2026-27 [--offline] [--refresh]

What it produces (see ``src/backtest/benchmarks.py`` for the consumer and ``docs/adr/0007-adp-ingest.md``
for the design writeup):

* ``<data_dir>/processed/adp.parquet`` -- columns ``season, source, source_id, adp`` (plus a
  ``name`` and an informational ``adp_source`` column), one row per player per season, as
  ``load_adp`` expects. ``source`` is always ``"espn"`` -- see "The 2025-26 gap" below.
* ``<data_dir>/processed/player_id_map.parquet`` -- ``(source, source_id) -> player_id`` for every
  ESPN player id this run could resolve onto the canonical NBA ``player_id``, written through
  ``src.store.write_table`` (which validates the ``player_id_map`` contract).

Endpoint (per ``docs/research/espn-api-findings.md`` / ``docs/research/snippets/espn_players_adp.py``,
already verified there -- this module does not rediscover any of it):

    GET https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season_id}/players
        ?view=kona_player_info
        X-Fantasy-Filter: {"limit": N, "sortPercOwned": {...}, "filterStatsForTopScoringPeriodIds": {...}}

``season_id`` is the season's END year (2027 = 2026-27). The filter must be top-level, not wrapped in
``{"players": {...}}`` (that wrapper is a *league* endpoint convention and is silently ignored here).

The 2025-26 gap
----------------
ESPN's ADP for 2025-26 is wiped (every player reads exactly 140.0 -- the "no ADP" sentinel -- except one
stray real value). ``detect_wiped_season`` flags and excludes any season like this rather than ingesting
garbage. To still cover that season, this module optionally fills it from FantasyPros'
``adp/overall.php?year=<Y>`` "ESPN" column (ADR 0005 D3; verified in
``docs/research/snippets/fantasypros_adp.py``), matched onto the *same* ESPN player ids already resolved
from other seasons (so the id-map join in ``load_adp`` still works unmodified, with ``source`` staying
``"espn"``). The informational ``adp_source`` column records ``"espn"`` vs ``"fantasypros_fill"`` per row
so this is never silently blended -- see ADR 0007 for the reasoning.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.contracts import (
    ContractError, data_dir, season_start, season_str, seasons_between, validate_table,
)
from src.ingest.http_cache import CachedHttpClient, HttpCacheError, default_cache_dir
from src.ingest.id_map import MatchReport, match_players
from src.store import read_table, table_exists, write_table

SOURCE = "espn"
CACHE_DIR_NAME = "espn"
FP_CACHE_DIR_NAME = "fantasypros"
UA = "nba-fantasy-2026-research/0.1 (personal, non-commercial; contact via GitHub repo)"

PLAYERS_URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season_id}/players"
FANTASYPROS_URL = "https://www.fantasypros.com/nba/adp/overall.php"

# ESPN's "no ADP" sentinel. 139.99 also appears in the wild per the research doc.
NO_ADP_SENTINEL = 139.9
# A season is "wiped" (ESPN removed the real values) when fewer than this many players carry a real
# ADP. 2025-26 has exactly 1; every real season observed has 140+.
WIPED_MIN_REAL_ADP = 10

REPORT_NAME = "espn_adp_report.json"


class EspnAdpError(RuntimeError):
    """The pulled/matched data is unusable in a way the CLI should stop and report."""


# --------------------------------------------------------------------------- season <-> ESPN id

def espn_season_id(season: str) -> int:
    """'2026-27' -> 2027 (ESPN's season id is the END year)."""
    return season_start(season) + 1


def season_from_espn_id(season_id: int) -> str:
    return season_str(season_id - 1)


_SEASON_TOKEN = re.compile(r"^\d{4}-\d{2}$")


def _one_season(token: str) -> str:
    token = token.strip()
    if not _SEASON_TOKEN.match(token):
        raise ValueError(f"malformed season {token!r}; expected like '2023-24'")
    season_start(token)
    return token


def parse_seasons(spec: str) -> list[str]:
    """Same grammar as ``nba_stats.parse_seasons``: 'A:B' range, single season, or comma list."""
    if not spec or not spec.strip():
        raise ValueError("no seasons given; expected like '2015-16:2026-27'")
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


# --------------------------------------------------------------------------- ESPN fetch + flatten

def fantasy_filter(limit: int, season_id: int) -> dict:
    return {
        "limit": limit,
        "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
        "filterStatsForTopScoringPeriodIds": {"value": 1, "additionalValue": [f"00{season_id}", f"10{season_id}"]},
    }


def fetch_espn_players(client: CachedHttpClient, season_id: int, *, limit: int = 600,
                       refresh: bool = False) -> list[dict]:
    url = PLAYERS_URL.format(season_id=season_id)
    headers = {"X-Fantasy-Filter": json.dumps(fantasy_filter(limit, season_id)), "User-Agent": UA}
    payload = client.get_json(f"players/{season_id}", url, {"view": "kona_player_info", "limit": limit},
                              headers=headers, refresh=refresh)
    if not isinstance(payload, list):
        raise EspnAdpError(f"season {season_id}: expected a JSON list of players, got {type(payload).__name__}")
    return payload


def flatten_espn_player(p: dict, season: str) -> dict:
    own = p.get("ownership") or {}
    adp = own.get("averageDraftPosition")
    return {
        "season": season,
        "source_id": str(p["id"]),
        "name": p.get("fullName") or "",
        "pro_team_id": p.get("proTeamId"),
        "adp": float(adp) if adp is not None else None,
    }


def espn_players_frame(raw: list[dict], season: str) -> pd.DataFrame:
    rows = [flatten_espn_player(p, season) for p in raw if p.get("id") is not None and p.get("fullName")]
    return pd.DataFrame(rows, columns=["season", "source_id", "name", "pro_team_id", "adp"])


# --------------------------------------------------------------------------- wiped-season detection

@dataclass
class WipedCheck:
    season: str
    n_players: int
    n_real_adp: int
    best_adp: float | None
    wiped: bool


def detect_wiped_season(df: pd.DataFrame, season: str, *, min_real: int = WIPED_MIN_REAL_ADP) -> WipedCheck:
    """A season is "wiped" when almost no player carries a real (non-sentinel, non-null) ADP."""
    real = df["adp"].notna() & (df["adp"] > 0) & (df["adp"] < NO_ADP_SENTINEL)
    n_real = int(real.sum())
    best = float(df.loc[real, "adp"].min()) if n_real else None
    return WipedCheck(season=season, n_players=len(df), n_real_adp=n_real, best_adp=best,
                      wiped=n_real < min_real)


def real_adp_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop null/zero/sentinel rows (the ones a wiped or partly-wiped season still carries)."""
    real = df["adp"].notna() & (df["adp"] > 0) & (df["adp"] < NO_ADP_SENTINEL)
    return df[real].copy()


# --------------------------------------------------------------------------- FantasyPros gap fill

_FP_PLAYER = re.compile(r"^(?P<name>.*?)\s*\((?P<team>[A-Z]{2,3}) - (?P<pos>[A-Z,]+)\)\s*(?P<tag>.*)$")


def fetch_fantasypros_adp(client: CachedHttpClient, year: int, *, refresh: bool = False) -> pd.DataFrame:
    """FantasyPros 'ESPN' column ADP for the season starting ``year`` (their ``year=`` is our start year)."""
    from bs4 import BeautifulSoup  # local import: only needed for the fallback path

    html = client.get_text("fantasypros_adp", FANTASYPROS_URL, {"year": year},
                           headers={"User-Agent": UA}, refresh=refresh)
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if table is None:
        raise EspnAdpError(f"FantasyPros year={year}: no <table> found in the response")
    rows = table.find_all("tr")
    header = [c.get_text(" ", strip=True) for c in rows[0].find_all(["th", "td"])]
    out = []
    for r in rows[1:]:
        cells = [c.get_text(" ", strip=True) for c in r.find_all(["th", "td"])]
        if len(cells) != len(header):
            continue
        rec = dict(zip(header, cells))
        m = _FP_PLAYER.match(rec.get("Player", ""))
        name = m.group("name") if m else rec.get("Player", "")
        for k in ("ESPN", "AVG"):
            v = rec.get(k)
            rec[k] = float(v) if v and v.replace(".", "", 1).isdigit() else None
        out.append({"name": name, "espn_adp": rec.get("ESPN"), "avg_adp": rec.get("AVG")})
    return pd.DataFrame(out, columns=["name", "espn_adp", "avg_adp"])


def fill_gap_season(fp: pd.DataFrame, espn_id_map: pd.DataFrame, season: str) -> tuple[pd.DataFrame, dict]:
    """Match FantasyPros names onto *already-resolved* ESPN ids, so the gap season still keys on
    ``source_id`` values that exist in ``player_id_map`` -- no separate FantasyPros namespace needed."""
    from src.ingest.id_map import alias_key

    idx: dict[str, list[str]] = {}
    for sid, name in zip(espn_id_map["source_id"], espn_id_map["source_name"]):
        idx.setdefault(alias_key(name), []).append(sid)

    rows, matched, unmatched = [], 0, []
    for r in fp.itertuples(index=False):
        espn_val = r.espn_adp if r.espn_adp is not None and not pd.isna(r.espn_adp) else None
        avg_val = r.avg_adp if r.avg_adp is not None and not pd.isna(r.avg_adp) else None
        adp = espn_val if espn_val is not None else avg_val
        adp_source = "fantasypros_fill" if espn_val is not None else "fantasypros_fill_avg"
        if adp is None:
            continue
        candidates = idx.get(alias_key(r.name), [])
        if len(candidates) != 1:
            unmatched.append(r.name)
            continue
        rows.append({"season": season, "source": SOURCE, "source_id": candidates[0], "name": r.name,
                     "adp": float(adp), "adp_source": adp_source})
        matched += 1
    frame = pd.DataFrame(rows, columns=["season", "source", "source_id", "name", "adp", "adp_source"])
    return frame, {"n_rows": len(fp), "n_matched": matched, "n_unmatched": len(unmatched), "unmatched": unmatched}


# --------------------------------------------------------------------------- pipeline

@dataclass
class AdpIngestResult:
    seasons: list[str]
    wiped: list[str] = field(default_factory=list)
    gap_filled: list[str] = field(default_factory=list)
    season_counts: dict[str, dict] = field(default_factory=dict)
    match_report: MatchReport | None = None
    adp_rows: int = 0
    id_map_rows: int = 0
    network_requests: int = 0
    cache_hits: int = 0


def run_ingest(
    seasons: list[str],
    client: CachedHttpClient,
    base: Path | None = None,
    *,
    limit: int = 600,
    refresh: bool = False,
    fill_gap: bool = True,
    fp_client: CachedHttpClient | None = None,
    min_real_adp: int = WIPED_MIN_REAL_ADP,
    log=print,
) -> AdpIngestResult:
    if not seasons:
        raise EspnAdpError("no seasons to ingest")
    base = base or data_dir()
    result = AdpIngestResult(seasons=list(seasons))

    per_season_raw: dict[str, pd.DataFrame] = {}
    all_players: dict[str, dict] = {}  # source_id -> {name, season_start} of the earliest sighting
    for season in seasons:
        sid = espn_season_id(season)
        log(f"[{season}] pulling ESPN player universe (season_id={sid})")
        raw = fetch_espn_players(client, sid, limit=limit, refresh=refresh)
        df = espn_players_frame(raw, season)
        per_season_raw[season] = df
        check = detect_wiped_season(df, season, min_real=min_real_adp)
        result.season_counts[season] = {"n_players": check.n_players, "n_real_adp": check.n_real_adp,
                                        "best_adp": check.best_adp, "wiped": check.wiped}
        if check.wiped:
            result.wiped.append(season)
            log(f"[{season}] WIPED: only {check.n_real_adp} players carry a real ADP (< {min_real_adp} threshold); excluded")
        else:
            log(f"[{season}] {check.n_real_adp} players with a real ADP (best {check.best_adp})")
        for r in df.itertuples(index=False):
            if r.source_id not in all_players:
                all_players[r.source_id] = {"name": r.name, "season_start": season_start(season)}

    universe = pd.DataFrame(
        [{"source_id": sid, "name": v["name"], "season_start": v["season_start"]} for sid, v in all_players.items()])
    log(f"matching {len(universe)} distinct ESPN players onto NBA player_id")
    nba_players = read_table("players", base)
    id_map_new, report = match_players(universe, nba_players, source=SOURCE)
    result.match_report = report
    log(f"id matching: {report.summary()}")

    id_map = merge_id_map(_existing_id_map(base), id_map_new, source=SOURCE)
    validate_table(id_map, "player_id_map")

    adp_frames = []
    for season in seasons:
        if season in result.wiped:
            continue
        real = real_adp_rows(per_season_raw[season])
        if real.empty:
            continue
        f = real[["season", "source_id", "adp"]].copy()
        f["source"] = SOURCE
        f["name"] = real["name"]
        f["adp_source"] = SOURCE
        adp_frames.append(f)

    if fill_gap and result.wiped:
        fpc = fp_client or client
        for season in result.wiped:
            year = season_start(season)
            log(f"[{season}] filling from FantasyPros (year={year})")
            try:
                fp = fetch_fantasypros_adp(fpc, year, refresh=refresh)
            except (HttpCacheError, EspnAdpError) as exc:
                log(f"[{season}] FantasyPros fallback failed: {exc}; season stays excluded")
                continue
            filled, fp_report = fill_gap_season(fp, id_map[id_map["source"] == SOURCE], season)
            result.season_counts[season]["fantasypros_fill"] = fp_report
            if len(filled):
                adp_frames.append(filled[["season", "source_id", "adp", "source", "name", "adp_source"]])
                result.gap_filled.append(season)
                log(f"[{season}] filled {len(filled)}/{fp_report['n_rows']} players from FantasyPros")

    adp = (pd.concat(adp_frames, ignore_index=True) if adp_frames
           else pd.DataFrame(columns=["season", "source", "source_id", "adp", "name", "adp_source"]))
    adp = adp[["season", "source", "source_id", "adp", "name", "adp_source"]]
    adp["adp"] = adp["adp"].astype("float64")
    adp = adp.sort_values(["season", "adp"], kind="stable").reset_index(drop=True)

    written_adp_path = write_adp(adp, base)
    written_id_map_path = write_table(id_map, "player_id_map", base)
    result.adp_rows = len(adp)
    result.id_map_rows = len(id_map)
    log(f"wrote {len(adp):,} adp rows -> {written_adp_path}")
    log(f"wrote {len(id_map):,} player_id_map rows -> {written_id_map_path}")

    _write_report(base, result)
    result.network_requests = client.stats.network_requests
    result.cache_hits = client.stats.cache_hits
    return result


def adp_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / "processed" / "adp.parquet"


def write_adp(df: pd.DataFrame, base: Path | None = None) -> Path:
    """``adp.parquet`` isn't a contract table (its shape is defined ad hoc by ``benchmarks.ADP_REQUIRED``
    plus two informational columns), so this writes it directly rather than through ``store.write_table``."""
    import os
    import tempfile

    missing = [c for c in ("season", "source", "source_id", "adp") if c not in df.columns]
    if missing:
        raise EspnAdpError(f"internal error: adp frame missing {missing}")
    if df["adp"].isna().any():
        raise EspnAdpError("internal error: adp frame has null adp values after filtering")
    path = adp_path(base)
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


def _existing_id_map(base: Path) -> pd.DataFrame | None:
    return read_table("player_id_map", base, validate=False) if table_exists("player_id_map", base) else None


def merge_id_map(existing: pd.DataFrame | None, new: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Replace this source's rows with the freshly matched set; leave every other source untouched."""
    if existing is None or existing.empty:
        return new.reset_index(drop=True)
    kept = existing[existing["source"] != source]
    frames = [f for f in (kept, new) if len(f)]
    merged = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0].copy()
    return merged.sort_values(["source", "source_id"], kind="stable").reset_index(drop=True)


def _write_report(base: Path, result: AdpIngestResult) -> None:
    report = {
        "seasons": result.seasons, "wiped": result.wiped, "gap_filled": result.gap_filled,
        "season_counts": result.season_counts,
        "match_report": None if result.match_report is None else {
            "n_total": result.match_report.n_total, "n_matched": result.match_report.n_matched,
            "match_rate": result.match_report.match_rate, "n_exact": result.match_report.n_exact,
            "n_normalized": result.match_report.n_normalized, "n_fuzzy": result.match_report.n_fuzzy,
            "n_ambiguous": result.match_report.n_ambiguous, "n_unmatched": result.match_report.n_unmatched,
            "ambiguous": result.match_report.ambiguous, "unmatched": result.match_report.unmatched,
        },
        "adp_rows": result.adp_rows, "id_map_rows": result.id_map_rows,
    }
    path = (base or data_dir()) / "processed" / REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def report_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / "processed" / REPORT_NAME


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.ingest.espn_adp",
        description="Ingest ESPN historical ADP into the shared data store, mapped onto NBA player_id.",
    )
    p.add_argument("--seasons", required=True, help="e.g. 2015-16:2026-27, 2023-24, or 2015-16:2017-18,2020-21")
    p.add_argument("--offline", action="store_true",
                   help="never touch the network; fail with a clear error if a response is not cached "
                        "(also enabled by NBA_OFFLINE=1)")
    p.add_argument("--refresh", action="store_true", help="re-download season-level responses even if cached")
    p.add_argument("--limit", type=int, default=600, help="players to request per season (default 600)")
    p.add_argument("--no-gap-fill", action="store_true",
                   help="skip the FantasyPros fallback for a wiped season (e.g. 2025-26)")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="override the data root (default: NBA_DATA_DIR or ~/dev-data/nba-fantasy-2026)")
    p.add_argument("--min-interval", type=float, default=2.0,
                   help="minimum seconds between network requests (default 2.0; ADR 0005's politeness floor)")
    p.add_argument("--max-retries", type=int, default=5)
    return p


def main(argv: list[str] | None = None, *, client: CachedHttpClient | None = None,
         fp_client: CachedHttpClient | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        seasons = parse_seasons(args.seasons)
    except ValueError as exc:
        parser.error(str(exc))
    if args.min_interval < 2.0 and not args.offline:
        print(f"warning: --min-interval {args.min_interval} is below the 2.0 s politeness floor "
              f"(CLAUDE.md/ADR 0005); using 2.0", file=sys.stderr)
        args.min_interval = 2.0
    base = args.data_dir or data_dir()
    if client is None:
        client = CachedHttpClient(default_cache_dir(CACHE_DIR_NAME), offline=True if args.offline else None,
                                  min_interval=args.min_interval, max_retries=args.max_retries)
    if fp_client is None:
        fp_client = CachedHttpClient(default_cache_dir(FP_CACHE_DIR_NAME), offline=True if args.offline else None,
                                     min_interval=max(args.min_interval, 5.0), max_retries=args.max_retries)
    try:
        result = run_ingest(seasons, client, base, limit=args.limit, refresh=args.refresh,
                            fill_gap=not args.no_gap_fill, fp_client=fp_client)
    except (HttpCacheError, EspnAdpError, ContractError, FileNotFoundError) as exc:
        print(f"espn adp ingest failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"done: {len(seasons)} seasons ({len(result.wiped)} wiped, {len(result.gap_filled)} gap-filled), "
          f"{result.adp_rows} adp rows, {result.id_map_rows} id-map rows, "
          f"{result.network_requests} network requests, {result.cache_hits} cache hits")
    if result.match_report is not None:
        print(f"id matching: {result.match_report.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
