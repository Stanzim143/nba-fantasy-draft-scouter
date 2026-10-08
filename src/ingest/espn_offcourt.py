"""ESPN general NBA news feed -> append-only ``offcourt_news`` table (ADR 0037).

    python -m src.ingest.espn_offcourt                     # pull once
    python -m src.ingest.espn_offcourt --show [--days 30]  # print stored articles that match an off-court keyword

ESPN's public site API (``/apis/site/v2/sports/basketball/nba/news``, no cookies, one request) returns the ~50 newest NBA
articles, each tagged with the athletes it is about. Trade requests, holdouts and contract disputes are never injury-feed
items, so this is the only structured-ish place they surface. Every (article, athlete) is archived once with ``first_seen`` /
``last_seen``; the feed holds only the latest articles, so the history exists only because it is pulled repeatedly
(``python -m src.ingest.espn_news`` does this on every pull, so the scheduled news job covers it).

The table stores the article text, not the keyword verdict: ``src.features.offcourt`` tags at read time, so better keywords
also apply to the stored history. Attribution rule (``linked_athletes``): an article is attached to its tagged athletes only
when it names at most ``MAX_ATHLETES`` of them, or to a larger set's members only if their name is in the headline, so a
30-player "draft board" article never gives all 30 a trade-request flag.

Advisory only (ADR 0006/0016 stance; ADR 0037): the flag moves no projection.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.contracts import data_dir
from src.ingest import espn_adp
from src.ingest.espn_news import _fix_text, merge_news, write_parquet_atomic
from src.ingest.http_cache import CachedHttpClient, HttpCacheError

TABLE = "offcourt_news"
URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/news"
CACHE_LABEL = "offcourt_news"
LIMIT = 100
MAX_ATHLETES = 3
COLUMNS = ["article_id", "espn_id", "player_id", "player_name", "n_athletes", "headline", "description", "url", "published",
           "first_seen", "last_seen"]
KEY = ["article_id", "espn_id"]


def table_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / "processed" / f"{TABLE}.parquet"


def read_offcourt(base: Path | None = None) -> pd.DataFrame:
    path = table_path(base)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.ingest.espn_news` (or espn_offcourt)")
    return pd.read_parquet(path)


def linked_athletes(headline: str, athletes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The athletes an article is *about*: all of them when there are few, else those named in the headline."""
    if len(athletes) <= MAX_ATHLETES:
        return athletes
    low = headline.lower()
    return [a for a in athletes if a["name"] and a["name"].lower() in low]


def parse_feed(payload: dict[str, Any], seen_at: datetime, id_map: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for art in payload.get("articles") or []:
        aid, headline = art.get("id"), _fix_text(art.get("headline"))
        if aid is None or not headline:
            continue
        athletes = [{"id": str(c["athleteId"]), "name": _fix_text(c.get("description"))}
                    for c in art.get("categories") or [] if c.get("type") == "athlete" and c.get("athleteId") is not None]
        url = ((art.get("links") or {}).get("web") or {}).get("href") or ""
        for a in linked_athletes(headline, athletes):
            rows.append({"article_id": str(aid), "espn_id": a["id"], "player_name": a["name"], "n_athletes": len(athletes),
                         "headline": headline, "description": _fix_text(art.get("description")), "url": url,
                         "published": pd.to_datetime(art.get("published"), errors="coerce", utc=True),
                         "first_seen": pd.Timestamp(seen_at), "last_seen": pd.Timestamp(seen_at)})
    df = pd.DataFrame(rows, columns=[c for c in COLUMNS if c != "player_id"])
    if df.empty:
        df["player_id"] = pd.array([], dtype="Int64")
    else:
        df = df.drop_duplicates(KEY, keep="last")
        if id_map is not None and len(id_map):
            em = id_map[id_map["source"] == "espn"].drop_duplicates("source_id")
            m = pd.Series(em["player_id"].to_numpy(), index=em["source_id"].astype(str).to_numpy())
            df["player_id"] = pd.array(df["espn_id"].map(m), dtype="Int64")
        else:
            df["player_id"] = pd.array([pd.NA] * len(df), dtype="Int64")
    df["published"] = pd.to_datetime(df["published"], utc=True).dt.tz_convert(None)
    for c in ("published", "first_seen", "last_seen"):
        df[c] = pd.to_datetime(df[c]).astype("datetime64[ms]")
    return df[COLUMNS].reset_index(drop=True)


def fetch_feed(client: CachedHttpClient, *, refresh: bool = True) -> dict[str, Any]:
    payload = client.get_json(CACHE_LABEL, URL, {"limit": LIMIT}, headers={"User-Agent": espn_adp.UA}, refresh=refresh)
    if not isinstance(payload, dict) or "articles" not in payload:
        raise espn_adp.EspnAdpError("ESPN news feed has no 'articles' key; the endpoint shape may have changed")
    return payload


def run_pull(client: CachedHttpClient, base: Path | None = None, *, now: datetime | None = None,
             refresh: bool = True) -> dict[str, Any]:
    base = base or data_dir()
    payload = fetch_feed(client, refresh=refresh)
    id_map = None
    try:
        from src.store import read_table

        id_map = read_table("player_id_map", base, validate=False)
    except FileNotFoundError:
        pass
    df = parse_feed(payload, now or datetime.now(), id_map)
    path = table_path(base)
    old = pd.read_parquet(path) if path.exists() else None
    merged, fresh = merge_news(old, df, key=KEY, columns=COLUMNS)
    write_parquet_atomic(merged, path)
    return {"in_feed": len(df), "new": len(fresh), "total": len(merged), "unmapped": int(df["player_id"].isna().sum())}


def show(base: Path, days: float | None, now: datetime, out=print) -> int:
    from src.features import offcourt

    df = offcourt.tag_news(read_offcourt(base))
    df = df[df["kinds"].astype(bool)]
    if days is not None:
        df = df[df["published"] >= pd.Timestamp(now - timedelta(days=days))]
    for _, r in df.sort_values("published", ascending=False).iterrows():
        out(f"{r['published']:%b %d}  {r['player_name']}  [{r['kinds']}]  {r['headline']}")
    if df.empty:
        out("no matching off-court news")
    return len(df)


def main(argv=None, *, client: CachedHttpClient | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.ingest.espn_offcourt")
    p.add_argument("--show", action="store_true", help="print stored articles that match an off-court keyword")
    p.add_argument("--days", type=float, default=None)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--data-dir", type=Path, default=None)
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    base = args.data_dir or data_dir()
    if args.show:
        try:
            show(base, args.days, datetime.now())
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1
        return 0
    client = client or CachedHttpClient(base / "raw" / espn_adp.CACHE_DIR_NAME, offline=True if args.offline else None,
                                        min_interval=2.0)
    try:
        r = run_pull(client, base, refresh=not args.offline)
    except (espn_adp.EspnAdpError, HttpCacheError, OSError, ValueError) as exc:
        print(f"off-court news pull failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"{r['in_feed']} article-athlete rows in feed, {r['new']} new ({r['total']} stored, {r['unmapped']} unmapped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
