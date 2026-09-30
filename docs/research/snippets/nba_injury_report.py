"""Fetch + parse one official NBA injury-report PDF (fixed-column text via poppler/xpdf ``pdftotext -table``).

RESEARCH PROTOTYPE - not imported by the codebase, not run by pytest.

What was verified (by running this on real files), 2026-09-22:
  * Public PDFs live at
      https://ak-static.cms.nba.com/referee/injury/Injury-Report_<YYYY-MM-DD>_<HH>[_<MM>]<AM|PM>.pdf
    e.g. Injury-Report_2025-12-15_05PM.pdf (a 5:30 PM ET report) and, since the 2025-26 season,
    quarter-hour names such as Injury-Report_2026-04-26_06_15PM.pdf.  Every report carries its own
    timestamp in the page header, so a row is naturally "as of" a datetime -> leakage-safe.
  * Existence probes (HTTP HEAD 200 vs 403 for absent keys, S3 behaviour): reports exist from
    ~2018-12-19 (2018-12-12 absent, 2018-12-19 present) through today; NOTHING for 2015-16 ... mid 2018-19.
  * The host resets the TCP connection for a User-Agent that contains an e-mail address in
    parentheses.  A ``Mozilla/5.0 (compatible; <project>/<ver>; <purpose>)`` UA is accepted.
  * Three layouts exist:  (a) 2018-19..2020-21: Category | Reason | Current Status | Previous Status,
    (b) 2021-22..2024-25: Current Status | Reason,  (c) 2025-26: same columns as (b) but Reason is
    "Injury/Illness - Right Knee; Sprain" / "G League - Two-Way" / "Not With Team".
    Rows for a player are 1 line; long reasons wrap onto a continuation line with no name.
  * Parser check: run on Injury-Report 2019-01-10_05PM (layout a), 2021-12-01_05PM and 2023-12-01_05PM (b),
    2025-12-15_05PM (c).  Parsed row counts equalled an independent count of status tokens (55/55 and
    100/100 on the two hand-checked files); teams == 2 x matchups; no unknown status strings.
    KNOWN LIMITS: in layout (a) the 'reason' string concatenates Category + Reason + Previous Status
    (e.g. 'Injury/Illness Left Fourth Metacarpal - Fracture'); wrapped reasons are re-joined with a space;
    not yet run on every one of the ~thousands of archived files (parse-rate over the whole archive is UNVERIFIED).

Requirements: ``pdftotext`` (xpdf/poppler) on PATH. No pip installs.  The ``-table`` mode gives clean rows;
``-layout`` interleaves wrapped lines and is NOT reliable.

NBA.com Terms of Use forbid redistribution; keep the raw PDFs under data/raw (gitignored) and commit nothing.
Be polite: the archive is one file per (date, time-slot); fetch only slots you need, sleep between calls.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

UA = "Mozilla/5.0 (compatible; nba-fantasy-2026-research/0.1; personal research)"
URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{slot}.pdf"
STATUSES = ("Out", "Doubtful", "Questionable", "Probable", "Available")


def fetch_pdf(date: str, slot: str, dest: Path) -> Path | None:
    """date 'YYYY-MM-DD', slot like '05PM' or '06_15PM'. Returns None when absent (S3 answers 403)."""
    req = urllib.request.Request(URL.format(date=date, slot=slot), headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            dest.write_bytes(r.read())
    except urllib.error.HTTPError as e:  # 403 == key does not exist
        if e.code in (403, 404):
            return None
        raise
    return dest


def pdf_to_text(pdf: Path) -> str:
    out = subprocess.run(["pdftotext", "-table", str(pdf), "-"], capture_output=True, text=True, check=True)
    return out.stdout


_STAMP = re.compile(r"Injury\s+Report:\s+(\d\d)/(\d\d)/(\d\d)\s+(\d\d:\d\d)\s+([AP]M)")
_DATE = re.compile(r"^\d\d/\d\d/\d{4}$")
_TIME = re.compile(r"^\d\d:\d\d \(ET\)$")
_MATCHUP = re.compile(r"^[A-Z]{2,3}@[A-Z]{2,3}$")
_STATUS = set(STATUSES) | {"NOT YET SUBMITTED"}


def parse(text: str) -> list[dict]:
    """One dict per player-status row.

    Content-based, not position-based: ``-table`` output repeats the column header only on page 1 and the
    column offsets drift between pages, so fields are recognised by shape after splitting on 2+ spaces:
    date (mm/dd/yyyy), time ("07:00 (ET)"), matchup (AAA@BBB), player ("Last, First"), team (no comma),
    status (closed vocabulary).  Tokens between player and status = category/reason (2018-2021 layout puts
    Reason BEFORE Current Status), tokens after status = reason (2021+) or previous status (2018-2021).
    A line with no player and no status is the wrapped tail of the previous reason.
    """
    rows: list[dict] = []
    report_ts = game_date = game_time = matchup = team = None
    for raw in text.splitlines():
        m = _STAMP.search(raw)
        if m:
            report_ts = f"20{m.group(3)}-{m.group(1)}-{m.group(2)} {m.group(4)} {m.group(5)}"
            continue
        line = raw.strip()
        if not line or line.startswith("Page ") or line.startswith("Game Date") or "Injury Report" in line:
            continue
        toks = re.split(r"\s{2,}", line)
        i_status = next((i for i, t in enumerate(toks) if t in _STATUS), None)
        # peel leading context tokens
        j = 0
        while j < len(toks) and (i_status is None or j < i_status):
            t = toks[j]
            if _DATE.match(t):
                game_date = t
            elif _TIME.match(t):
                game_time = t
            elif _MATCHUP.match(t):
                matchup = t
            elif "," not in t and t not in _STATUS and (i_status is not None and j + 1 <= i_status - 1
                                                          or i_status is None and False):
                # a team name: no comma, followed by a player token before the status
                team = t
            else:
                break
            j += 1
        rest = toks[j:]
        if i_status is None:
            # wrapped continuation of the previous row's reason
            if rows and rest:
                rows[-1]["reason"] = (rows[-1]["reason"] + " " + " ".join(rest)).strip()
            continue
        k = i_status - j                      # index of status within rest
        before, status, after = rest[:k], rest[k], rest[k + 1:]
        if status == "NOT YET SUBMITTED":     # team-level row: the team token is the only thing before
            team = before[-1] if before else team
            player, mid = "", []
        else:
            player, mid = (before[0], before[1:]) if before else ("", [])
        rows.append({
            "report_ts": report_ts, "game_date": game_date, "game_time": game_time, "matchup": matchup,
            "team": team, "player": player, "status": status,
            "reason": " ".join(mid + after).strip(),
        })
    return rows


def _selftest(pdf: Path) -> None:
    rows = parse(pdf_to_text(pdf))
    bad = [r for r in rows if r["status"] not in _STATUS or not r["player"] and r["status"] != "NOT YET SUBMITTED"]
    print(f"{pdf.name}: {len(rows)} rows, unknown statuses: {len(bad)}; first 3:")
    for r in rows[:3]:
        print("   ", r)


if __name__ == "__main__":
    date, slot = (sys.argv[1], sys.argv[2]) if len(sys.argv) == 3 else ("2025-12-15", "05PM")
    with tempfile.TemporaryDirectory() as d:
        p = fetch_pdf(date, slot, Path(d) / "r.pdf")
        if p is None:
            print("no such report")
        else:
            _selftest(p)
        time.sleep(2)
