"""Live ESPN injury / player-news feed -> append-only ``injury_news`` table (ADR 0036).

    python -m src.ingest.espn_news                 # pull once, print what is new since the last pull
    python -m src.ingest.espn_news --watch 15      # pull every 15 minutes until interrupted
    python -m src.ingest.espn_news --show [--days 3] [--team MIA] [--player Giannis] [--all]

ESPN's public site API (``/apis/site/v2/sports/basketball/nba/injuries``, no cookies, one request returns all 30 teams)
carries, per injured player, the status (``Out`` / ``Day-To-Day``), the injury (type, body part, side), the expected
return date, and a short and a long news blurb written by RotoWire. The fantasy players payload used by
``espn_status`` has only a status code and a "last news" timestamp; this feed has the *reason*, which is what separates a
sore ankle from a season-ending tear.

Every distinct (note id, status) is kept once with ``first_seen`` / ``last_seen``, so the table is a history: ESPN
exposes only the current notes. The feed is **advisory** (ADR 0006/0016 stance): it feeds board flags and the news view
and changes no projection, because no history exists to back-test a news effect on.

Same accepted-risk stance as ADR 0005 R2 (personal, non-commercial, low volume, local only): one request per pull, with
``--watch`` refusing intervals under 5 minutes.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.contracts import data_dir
from src.ingest import espn_adp
from src.ingest.http_cache import CachedHttpClient

TABLE = "injury_news"
URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
CACHE_LABEL = "injury_news"
MIN_WATCH_MINUTES = 5
COLUMNS = ["note_id", "espn_id", "player_id", "player_name", "team_abbr", "status", "injury_type", "body_part", "detail",
           "side", "return_date", "fantasy_status", "headline", "text", "source", "published", "first_seen", "last_seen"]
KEY = ["note_id", "status"]
_ATHLETE_ID = re.compile(r"/id/(\d+)")


def table_path(base: Path | None = None) -> Path:
    return (base or data_dir()) / "processed" / f"{TABLE}.parquet"


def _fix_text(s: Any) -> str:
    """ESPN sometimes returns UTF-8 read as latin-1 (``AndrÃ©``); undo that when it is clearly the case."""
    s = "" if s is None else str(s)
    if "Ã" in s or "â€" in s:
        try:
            return s.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            try:
                return s.encode("latin-1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                return s
    return s


def _espn_id(athlete: dict[str, Any]) -> str | None:
    if athlete.get("id") is not None:
        return str(athlete["id"])
    for link in athlete.get("links") or []:
        m = _ATHLETE_ID.search(link.get("href") or "")
        if m:
            return m.group(1)
    return None


def parse_feed(payload: dict[str, Any], seen_at: datetime, id_map: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (note id, status) in the feed. ``player_id`` is null where ESPN's id is not in ``player_id_map``."""
    rows = []
    for team in payload.get("injuries") or []:
        for inj in team.get("injuries") or []:
            athlete = inj.get("athlete") or {}
            note_id = inj.get("id")
            if note_id is None or not athlete.get("displayName"):
                continue
            det = inj.get("details") or {}
            fantasy = (det.get("fantasyStatus") or {}).get("abbreviation") or ""
            status = (inj.get("status") or "").strip() or "Unknown"
            # the notes block repeats the same text; the top-level short/long comment is the canonical copy
            rows.append({
                "note_id": str(note_id), "espn_id": _espn_id(athlete), "player_name": athlete["displayName"],
                "team_abbr": (athlete.get("team") or {}).get("abbreviation") or "", "status": status,
                "injury_type": det.get("type") or "", "body_part": det.get("location") or "",
                "detail": det.get("detail") or "", "side": det.get("side") or "",
                "return_date": pd.to_datetime(det.get("returnDate"), errors="coerce"),
                "fantasy_status": fantasy, "headline": _fix_text(inj.get("shortComment")),
                "text": _fix_text(inj.get("longComment")), "source": inj.get("source") or "",
                "published": pd.to_datetime(inj.get("date"), errors="coerce", utc=True),
                "first_seen": pd.Timestamp(seen_at), "last_seen": pd.Timestamp(seen_at),
            })
    df = pd.DataFrame(rows, columns=[c for c in COLUMNS if c != "player_id"])
    if df.empty:
        df["player_id"] = pd.array([], dtype="Int64")
        return df[COLUMNS]
    df = df.drop_duplicates(KEY, keep="last")
    if id_map is not None and len(id_map):
        em = id_map[id_map["source"] == "espn"].drop_duplicates("source_id")
        m = pd.Series(em["player_id"].to_numpy(), index=em["source_id"].astype(str).to_numpy())
        df["player_id"] = pd.array(df["espn_id"].map(m), dtype="Int64")
    else:
        df["player_id"] = pd.array([pd.NA] * len(df), dtype="Int64")
    df["published"] = df["published"].dt.tz_convert(None)
    for c in ("return_date", "published", "first_seen", "last_seen"):
        df[c] = pd.to_datetime(df[c]).astype("datetime64[ms]")
    return df[COLUMNS]


def read_news(base: Path | None = None) -> pd.DataFrame:
    path = table_path(base)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m src.ingest.espn_news`")
    return pd.read_parquet(path)


def merge_news(old: pd.DataFrame | None, new: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(merged table, rows that are new). Existing keys keep their ``first_seen`` and get the newer ``last_seen``."""
    if old is None or old.empty:
        return new.reset_index(drop=True), new
    tagged = old[KEY].merge(new[KEY + ["last_seen"]].rename(columns={"last_seen": "_seen"}), on=KEY, how="left")
    old = old.copy()
    old["last_seen"] = tagged["_seen"].where(tagged["_seen"].notna(), old["last_seen"]).to_numpy()
    fresh = new.merge(old[KEY], on=KEY, how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    merged = pd.concat([old, fresh[COLUMNS]], ignore_index=True)
    return merged.sort_values(["published", "note_id"], kind="mergesort").reset_index(drop=True), fresh


def write_news(df: pd.DataFrame, base: Path | None = None) -> Path:
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
    return path


def fetch_feed(client: CachedHttpClient, *, refresh: bool = True) -> dict[str, Any]:
    payload = client.get_json(CACHE_LABEL, URL, headers={"User-Agent": espn_adp.UA}, refresh=refresh)
    if not isinstance(payload, dict) or "injuries" not in payload:
        raise espn_adp.EspnAdpError("ESPN injuries feed has no 'injuries' key; the endpoint shape may have changed")
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
    if df.empty and not payload.get("injuries"):
        raise espn_adp.EspnAdpError("ESPN injuries feed listed no teams; nothing archived")
    path = table_path(base)
    old = pd.read_parquet(path) if path.exists() else None
    merged, fresh = merge_news(old, df)
    write_news(merged, base)
    return {"in_feed": len(df), "new": len(fresh), "total": len(merged), "unmapped": int(df["player_id"].isna().sum()),
            "fresh": fresh}


def latest_per_player(news: pd.DataFrame) -> pd.DataFrame:
    """Newest note per mapped player that is still in the feed's latest pull (ESPN drops players when they return)."""
    if news.empty:
        return news
    current = news[news["last_seen"] == news["last_seen"].max()]
    cur = current.dropna(subset=["player_id"])
    return cur.sort_values(["published", "note_id"]).drop_duplicates("player_id", keep="last").reset_index(drop=True)


def format_item(r: pd.Series, *, long: bool = False) -> str:
    when = r["published"].strftime("%b %d %H:%M") if pd.notna(r["published"]) else "?"
    inj = " ".join(x for x in (r["injury_type"] or r["detail"], f"({r['body_part']})" if r["body_part"] else "") if x)
    back = f", est. return {r['return_date']:%b %d}" if pd.notna(r["return_date"]) else ""
    head = f"{when}  {r['player_name']} ({r['team_abbr']})  {r['status']}{(' - ' + inj) if inj else ''}{back}"
    body = r["headline"] + (f"\n      {r['text']}" if long and r["text"] else "")
    return f"{head}\n    {body}"


def show(news: pd.DataFrame, *, days: float | None, team: str | None, player: str | None, current_only: bool,
         now: datetime, out: Callable[[str], None] = print) -> int:
    df = news
    if current_only:
        df = df[df["last_seen"] == df["last_seen"].max()]
    if days is not None:
        df = df[df["published"] >= pd.Timestamp(now - timedelta(days=days))]
    if team:
        df = df[df["team_abbr"].str.upper() == team.upper()]
    if player:
        df = df[df["player_name"].str.contains(player, case=False, regex=False)]
    df = df.sort_values("published", ascending=False)
    for _, r in df.iterrows():
        out(format_item(r, long=bool(player)))
    if df.empty:
        out("no matching news")
    return len(df)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.ingest.espn_news", description=__doc__.split("\n\n")[0])
    p.add_argument("--watch", type=float, metavar="MINUTES", default=None, help=f"pull repeatedly (>= {MIN_WATCH_MINUTES})")
    p.add_argument("--show", action="store_true", help="print stored news instead of pulling")
    p.add_argument("--days", type=float, default=None, help="--show: only notes published in the last N days")
    p.add_argument("--team", default=None, help="--show: team abbreviation, e.g. MIA")
    p.add_argument("--player", default=None, help="--show: name substring (prints the long text)")
    p.add_argument("--all", action="store_true", help="--show: include players no longer in the feed")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--data-dir", type=Path, default=None)
    return p


def main(argv=None, *, client: CachedHttpClient | None = None, sleep: Callable[[float], None] = time.sleep,
         max_pulls: int | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    base = args.data_dir or data_dir()
    if args.show:
        try:
            news = read_news(base)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            return 1
        show(news, days=args.days, team=args.team, player=args.player, current_only=not args.all, now=datetime.now())
        return 0
    if args.watch is not None and args.watch < MIN_WATCH_MINUTES:
        print(f"--watch must be at least {MIN_WATCH_MINUTES} minutes (polite polling, ADR 0005 R2)", file=sys.stderr)
        return 2
    client = client or CachedHttpClient(base / "raw" / espn_adp.CACHE_DIR_NAME, offline=True if args.offline else None,
                                        min_interval=2.0)
    pulls = 0
    while True:
        try:
            r = run_pull(client, base, refresh=not args.offline)
        except (espn_adp.EspnAdpError, OSError, ValueError) as exc:
            print(f"news pull failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.watch is None:
                return 1
        else:
            stamp = datetime.now().strftime("%H:%M")
            print(f"[{stamp}] {r['in_feed']} notes in feed, {r['new']} new ({r['total']} stored, {r['unmapped']} unmapped)")
            for _, row in r["fresh"].sort_values("published").iterrows():
                print(format_item(row))
        pulls += 1
        if args.watch is None or (max_pulls is not None and pulls >= max_pulls):
            return 0
        sleep(args.watch * 60)


if __name__ == "__main__":
    raise SystemExit(main())
