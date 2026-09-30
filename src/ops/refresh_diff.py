"""Point-in-time ESPN snapshots and the run-over-run diffs behind the daily report (ADR 0014).

ADR 0005: our own dated snapshots of the ESPN player universe are the only real point-in-time ADP and injury source,
because ESPN wipes prior seasons and publishes no history. This module archives one compact snapshot per distinct pull
under ``<data>/archive/espn_players/<season>/`` (never committed) and diffs two of them, plus the roster map and the
breakout watchlist, so the report can say what changed since the previous run.

Everything here is pure (files in, dicts out) so it is tested offline with tiny fixtures.
"""
from __future__ import annotations

import gzip
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

# ESPN's "no ADP" sentinel (see src.ingest.espn_adp.NO_ADP_SENTINEL); anything at or above it is unpriced.
NO_ADP = 139.9
ARCHIVE_SUBDIR = Path("archive") / "espn_players"


# --------------------------------------------------------------------------- snapshots

def compact_players(raw: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The fields worth keeping from ESPN's ``kona_player_info`` rows (a few KB per hundred players when gzipped)."""
    out = []
    for p in raw:
        if p.get("id") is None or not p.get("fullName"):
            continue
        own = p.get("ownership") or {}
        adp = own.get("averageDraftPosition")
        out.append({
            "id": str(p["id"]), "name": p["fullName"], "team_id": p.get("proTeamId"),
            "status": p.get("injuryStatus"), "injured": bool(p.get("injured")), "active": p.get("active"),
            "adp": float(adp) if adp is not None else None,
            "pct_owned": own.get("percentOwned"), "pct_started": own.get("percentStarted"),
            "news_ms": p.get("lastNewsDate"),
        })
    return sorted(out, key=lambda r: r["id"])


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def snapshot_dir(base: Path, season: str) -> Path:
    return base / ARCHIVE_SUBDIR / season


def list_snapshots(base: Path, season: str) -> list[Path]:
    d = snapshot_dir(base, season)
    return sorted(d.glob("*.json.gz")) if d.exists() else []


def read_snapshot(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def write_snapshot(base: Path, season: str, raw: Iterable[Mapping[str, Any]], taken_at: datetime) -> tuple[Path, bool]:
    """Archive ``raw`` as a dated snapshot. Returns ``(path, written)``; ``written`` is False when the newest existing
    snapshot has identical content (an offline re-run or a re-run within ESPN's own update interval), so a twice-a-day
    schedule never fills the archive with duplicates. Snapshots are never overwritten or deleted."""
    players = compact_players(raw)
    if not players:
        raise ValueError("ESPN payload contained no players; refusing to archive an empty snapshot")
    existing = list_snapshots(base, season)
    if existing:
        try:
            if read_snapshot(existing[-1])["players"] == players:
                return existing[-1], False
        except (OSError, ValueError, KeyError):
            pass                                             # unreadable newest snapshot: write a fresh one
    taken = taken_at.astimezone(timezone.utc)
    path = snapshot_dir(base, season) / f"{taken:%Y%m%dT%H%M%SZ}.json.gz"
    if path.exists():
        return path, False
    body = json.dumps({"taken_at": taken.isoformat(), "season": season, "players": players},
                      separators=(",", ":")).encode("utf-8")
    _atomic_write(path, gzip.compress(body, mtime=0))
    return path, True


def previous_snapshot(base: Path, season: str, current: Path) -> Path | None:
    older = [p for p in list_snapshots(base, season) if p.name < current.name]
    return older[-1] if older else None


# --------------------------------------------------------------------------- diffs

def _real_adp(v: float | None) -> float | None:
    return v if v is not None and v < NO_ADP else None


def _norm_status(s: str | None) -> str:
    return "ACTIVE" if s in (None, "", "ACTIVE") else str(s)


def injury_changes(prev: Mapping[str, Any], cur: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Players whose ESPN injury status differs between two snapshots, most draft-relevant (best ADP) first."""
    before = {p["id"]: p for p in prev["players"]}
    rows = []
    for p in cur["players"]:
        old = before.get(p["id"])
        if old is None:
            continue
        a, b = _norm_status(old.get("status")), _norm_status(p.get("status"))
        if a != b:
            rows.append({"id": p["id"], "name": p["name"], "from": a, "to": b, "adp": _real_adp(p.get("adp")),
                         "pct_owned": p.get("pct_owned")})
    rows.sort(key=lambda r: (r["adp"] is None, r["adp"] if r["adp"] is not None else 0, -(r["pct_owned"] or 0)))
    return rows


def adp_moves(prev: Mapping[str, Any], cur: Mapping[str, Any], *, min_move: float = 2.0,
              top: int = 12) -> dict[str, list[dict[str, Any]]]:
    """ADP risers (drafted earlier now), fallers, and players that gained or lost an ADP entirely."""
    before = {p["id"]: p for p in prev["players"]}
    risers, fallers, priced, unpriced = [], [], [], []
    for p in cur["players"]:
        old = before.get(p["id"])
        if old is None:
            continue
        a, b = _real_adp(old.get("adp")), _real_adp(p.get("adp"))
        if a is not None and b is not None:
            d = b - a
            if d <= -min_move:
                risers.append({"id": p["id"], "name": p["name"], "from": a, "to": b, "delta": d})
            elif d >= min_move:
                fallers.append({"id": p["id"], "name": p["name"], "from": a, "to": b, "delta": d})
        elif a is None and b is not None:
            priced.append({"id": p["id"], "name": p["name"], "from": None, "to": b, "delta": None})
        elif a is not None and b is None:
            unpriced.append({"id": p["id"], "name": p["name"], "from": a, "to": None, "delta": None})
    risers.sort(key=lambda r: r["delta"])
    fallers.sort(key=lambda r: -r["delta"])
    return {"risers": risers[:top], "fallers": fallers[:top], "priced": priced[:top], "unpriced": unpriced[:top],
            "n_risers": len(risers), "n_fallers": len(fallers)}


def roster_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """``player_id -> [name, team_abbr]`` from roster snapshot rows."""
    return {str(r["player_id"]): [str(r["player_name"]), str(r["team_abbr"])] for r in rows}


def roster_diff(prev: Mapping[str, list[str]], cur: Mapping[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    arrived = [{"name": cur[k][0], "to": cur[k][1]} for k in sorted(set(cur) - set(prev), key=lambda k: cur[k][0])]
    departed = [{"name": prev[k][0], "from": prev[k][1]} for k in sorted(set(prev) - set(cur), key=lambda k: prev[k][0])]
    moved = [{"name": cur[k][0], "from": prev[k][1], "to": cur[k][1]}
             for k in sorted(set(cur) & set(prev), key=lambda k: cur[k][0]) if prev[k][1] != cur[k][1]]
    return {"arrived": arrived, "departed": departed, "moved": moved}


def watchlist_diff(prev: list[Mapping[str, Any]], cur: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Players newly on, dropped from, and moving 5+ places within the watchlist (rows carry ``player_id``, ``name``, ``rank``)."""
    before = {str(r["player_id"]): r for r in prev}
    now = {str(r["player_id"]): r for r in cur}
    new = [now[k] for k in now if k not in before]
    dropped = [before[k] for k in before if k not in now]
    moved = [{"name": now[k]["name"], "from": before[k]["rank"], "to": now[k]["rank"]}
             for k in now if k in before and abs(int(now[k]["rank"]) - int(before[k]["rank"])) >= 5]
    new.sort(key=lambda r: r["rank"])
    dropped.sort(key=lambda r: r["rank"])
    moved.sort(key=lambda r: r["to"])
    return {"new": new, "dropped": dropped, "moved": moved}


def games_delta(prev: Mapping[str, Any] | None, cur: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """New Summer League / preseason games ingested since the previous run: ``{context: {before, now, new, last_date}}``."""
    out: dict[str, dict[str, Any]] = {}
    for ctx, e in (cur or {}).items():
        before = ((prev or {}).get(ctx) or {}).get("games")
        now = e.get("games") or 0
        out[ctx] = {"before": before, "now": now, "new": (now - before) if before is not None else None,
                    "last_date": e.get("last_date")}
    return out
