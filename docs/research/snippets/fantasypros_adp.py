"""FantasyPros consensus ADP (Yahoo / ESPN / CBS / NFBC columns + AVG) for a chosen season, parsed from HTML.

RESEARCH PROTOTYPE - not imported by the codebase, not run by pytest.

Verified 2026-09-22 (this script's parser was run on saved copies of every page below):
  * https://www.fantasypros.com/nba/adp/overall.php?year=<Y> returns HTTP 200 HTML for Y in
    2015, 2016, 2018, 2020, 2022, 2024, 2025 (title says "2015-16" ... "2025-26"); the bare URL is 2026-27.
    (2017, 2019, 2021, 2023 were not fetched; expected to work, UNVERIFIED.)  year=Y  ->  season "Y-(Y+1)".
  * One <table>, 203-266 rows per season, columns: Rank | Player | Yahoo | ESPN | [CBS or NFBCK] | AVG.
    The site set differs per season (2015: Yahoo/ESPN/NFBCK; 2016-2020: Yahoo/ESPN/CBS; 2022+: Yahoo/ESPN).
  * The Player cell is "Name (TEAM - POS[,POS]) TAG": TEAM/TAG are TODAY's team/status (e.g. 2015-16 Anthony
    Davis shows "WAS ... OUT"), NOT the team at the time.  Never use them as point-in-time features.
  * Cross-check: FantasyPros' ESPN column vs the ESPN API ADP for the same season, Spearman rho on matched
    players: 2015-16 1.00, 2016-17 1.00, 2018-19 0.97, 2020-21 0.81, 2022-23 0.84, 2024-25 0.92.
  * robots.txt (2026-09-22): ``Crawl-delay: 5``; Disallow only /ajax/, /api/, /json/, /xml/, /nba/ranker/.
    Terms of Use (https://www.fantasypros.com/about/legal/): copies for personal use only, no republishing.
    => keep output LOCAL under data/raw (gitignored). Never commit it.
NOT verified: when the ADP snapshot was taken relative to the draft (the pages carry no snapshot date).
"""
from __future__ import annotations

import re
import sys
import time
import urllib.request

from bs4 import BeautifulSoup

UA = "Mozilla/5.0 (compatible; nba-fantasy-2026-research/0.1; personal research)"
URL = "https://www.fantasypros.com/nba/adp/overall.php?year={year}"
_PLAYER = re.compile(r"^(?P<name>.*?)\s*\((?P<team>[A-Z]{2,3}) - (?P<pos>[A-Z,]+)\)\s*(?P<tag>.*)$")


def fetch_html(year: int) -> str:
    req = urllib.request.Request(URL.format(year=year), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "ignore")


def parse_adp(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    rows = table.find_all("tr")
    header = [c.get_text(" ", strip=True) for c in rows[0].find_all(["th", "td"])]
    out = []
    for r in rows[1:]:
        cells = [c.get_text(" ", strip=True) for c in r.find_all(["th", "td"])]
        if len(cells) != len(header):
            continue
        rec = dict(zip(header, cells))
        m = _PLAYER.match(rec["Player"])
        rec.update(m.groupdict() if m else {"name": rec["Player"]})
        for k in ("Rank", "Yahoo", "ESPN", "CBS", "NFBCK", "AVG"):
            if k in rec:
                rec[k] = float(rec[k]) if rec[k].replace(".", "", 1).isdigit() else None
        out.append(rec)
    return out


if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    rows = parse_adp(fetch_html(year))
    print(f"{len(rows)} rows for {year}-{(year + 1) % 100:02d}; columns: {list(rows[0])}")
    for r in rows[:5]:
        print(r["Rank"], r["name"], r.get("Yahoo"), r.get("ESPN"), r["AVG"])
    time.sleep(5)  # honour Crawl-delay if you loop over seasons
