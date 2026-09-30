"""ESPN league-agnostic player universe: ADP, ownership, auction value, ESPN projections.

RESEARCH PROTOTYPE - not imported by the codebase, not run by pytest.

What was verified (by running this script), 2026-09-22, no cookies, no login:
  * GET https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season_id}/players
    ?view=kona_player_info with an ``X-Fantasy-Filter`` header returns HTTP 200 JSON.  Full 500-player pulls
    succeeded for season_id 2016 ... 2027; limit-3 probes also succeeded for 2010 and 2015.
    season_id = END year: 2027 == the 2026-27 season.
  * The filter must be TOP-LEVEL (``{"limit": N, "sortPercOwned": {...}}``). The
    ``{"players": {...}}`` wrapper that espn-api uses on *league* URLs is silently
    ignored on this games-level URL (returns the full 3,184-player, 24 MB universe).
  * ``filterStatsForTopScoringPeriodIds`` with ``additionalValue`` ["00<season>", "10<season>"]
    shrinks the payload ~30x (about 2.5 KB per player instead of ~45 KB).
  * ``ownership.averageDraftPosition`` / ``percentOwned`` / ``auctionValueAverage`` are present
    for every season fetched.  For 2026-27 the ownership block is live (``date`` = fetch time).
    For past seasons ``date`` is null and the values are a season-level aggregate whose
    exact snapshot time is NOT documented (see docs/research/espn-api-findings.md).
  * ESPN projections live in ``stats[]`` entries with ``statSourceId == 1`` and
    ``statSplitTypeId == 0`` (id "10<season>").  Actuals are ``statSourceId == 0`` ("00<season>").

Usage (venv python; needs only ``requests``)::

    python espn_players_adp.py 2027            # prints top 15 by ADP
    python espn_players_adp.py 2025 --out adp_2025.csv --limit 400

Politeness: one request per season, descriptive User-Agent, no retries loop.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

import requests

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/{season_id}/players"
UA = "nba-fantasy-2026-research/0.1 (personal research; contact via GitHub repo)"

# ESPN slot ids used in eligibleSlots (see espn_position_map.py for the full table)
SLOT_NAMES = {0: "PG", 1: "SG", 2: "SF", 3: "PF", 4: "C", 5: "G", 6: "F", 7: "SG/SF", 8: "G/F",
              9: "PF/C", 10: "F/C", 11: "UTIL", 12: "BE", 13: "IR"}
# defaultPositionId is slot id + 1 for the five base positions (1=PG ... 5=C)
DEFAULT_POS = {1: "PG", 2: "SG", 3: "SF", 4: "PF", 5: "C"}


def fetch_players(season_id: int, limit: int = 400) -> list[dict]:
    flt = {
        "limit": limit,
        "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
        "filterStatsForTopScoringPeriodIds": {
            "value": 1, "additionalValue": [f"00{season_id}", f"10{season_id}"]},
    }
    r = requests.get(BASE.format(season_id=season_id), params={"view": "kona_player_info"},
                     headers={"X-Fantasy-Filter": json.dumps(flt), "User-Agent": UA}, timeout=60)
    r.raise_for_status()
    return r.json()


def flatten(p: dict, season_id: int) -> dict:
    own = p.get("ownership") or {}
    proj = next((s for s in p.get("stats", [])
                 if s.get("seasonId") == season_id and s.get("statSourceId") == 1
                 and s.get("statSplitTypeId") == 0), None)
    ps = (proj or {}).get("stats", {})  # stat ids: 0 PTS,1 BLK,2 STL,3 AST,6 REB,11 TO,13 FGM,14 FGA,15 FTM,16 FTA,17 3PM,28 MPG,42 GP
    ranks = p.get("draftRanksByRankType") or {}
    return {
        "season_id": season_id,
        "espn_id": p["id"],
        "name": p["fullName"],
        "pro_team_id": p.get("proTeamId"),
        "default_pos": DEFAULT_POS.get(p.get("defaultPositionId")),
        "eligible": "/".join(SLOT_NAMES[s] for s in p.get("eligibleSlots", [])
                             if s <= 6),  # base positions only; 7-10 are combo slots
        "injury_status": p.get("injuryStatus"),
        "adp": own.get("averageDraftPosition"),
        "pct_owned": own.get("percentOwned"),
        "pct_started": own.get("percentStarted"),
        "auction_avg": own.get("auctionValueAverage"),
        "own_date_ms": own.get("date"),
        "espn_rank_standard": (ranks.get("STANDARD") or {}).get("rank"),
        "espn_auction_standard": (ranks.get("STANDARD") or {}).get("auctionValue"),
        "proj_gp": ps.get("42"), "proj_mpg": ps.get("28"), "proj_pts": ps.get("0"),
        "proj_reb": ps.get("6"), "proj_ast": ps.get("3"), "proj_stl": ps.get("2"),
        "proj_blk": ps.get("1"), "proj_to": ps.get("11"),
        "proj_applied_total_espn_default": (proj or {}).get("appliedTotal"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("season_id", type=int, help="END year, e.g. 2027 for 2026-27")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--out", help="write CSV here (keep it OUTSIDE the repo / under data/raw)")
    a = ap.parse_args()
    rows = [flatten(p, a.season_id) for p in fetch_players(a.season_id, a.limit)]
    rows = [r for r in rows if r["adp"] is not None]
    rows.sort(key=lambda r: r["adp"])
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    for r in rows[:15]:
        print(f'{r["adp"]:7.2f}  {r["name"]:26s} {r["default_pos"]:>2} {r["eligible"]:12s} '
              f'own={r["pct_owned"]:.1f} auc={r["auction_avg"]:.1f} proj_gp={r["proj_gp"]}')
    print(f"{len(rows)} players with an ADP field", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
