"""Advisory off-court flags and hand-written player notes for the board (ADR 0037).

Three inputs, none of which touches a projection, ``risk_level`` or the games haircut (no history exists to back-test them;
ADR 0033 tested the injury-report reason labels as model features and found ~1% gain, not adopted):

* ``injury_report_flags``: games the player was listed Out last season for an off-court reason in the official NBA injury
  reports (personal reasons, not with team, trade pending, suspension): ``"7 games out for personal reasons in 2025-26"``.
* ``news_flags``: ESPN general-news articles (``offcourt_news``, ``src.ingest.espn_offcourt``) whose headline or blurb matches a
  trade-request / holdout / contract-dispute / leave-of-absence / suspension keyword, newest per kind within ``NEWS_DAYS``:
  ``'off-court news: trade request, Sep 12 ("Giannis ...")'``.
* ``read_notes`` / ``note_flags``: ``config/player_notes.yaml``, the user's own judgment, shown verbatim as ``note: ...``.

Keyword matching is a coarse screen of a headline, not an understanding of it; the flag shows the headline so the reader can
judge. ``tag_news`` runs at read time so improved keywords apply to the stored history.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.contracts import season_start

NEWS_DAYS = 60            # an article older than this no longer flags
MIN_REPORT_GAMES = 3      # fewer off-court games than this in the injury reports is routine, not a flag
HEADLINE_CHARS = 60
NOTES_PATH = Path(__file__).resolve().parents[2] / "config" / "player_notes.yaml"

NEWS_KINDS: dict[str, re.Pattern] = {
    "trade request": re.compile(
        r"trade request|request(?:s|ed|ing)? (?:a )?trade|ask(?:s|ed|ing)? (?:for )?(?:a )?trade|ask(?:s|ed|ing)? to be traded|"
        r"want(?:s|ed)? (?:to be traded|a trade|out of (?:town|the team|his team))|demand(?:s|ed)? (?:a )?trade|"
        r"request(?:s|ed)? (?:to be )?traded", re.I),
    "holdout": re.compile(
        r"hold-?outs?\b|hold(?:s|ing)? out of (?:camp|training)|skip(?:s|ped|ping)? (?:training )?camp|"
        r"(?:will not|won'?t|not|didn'?t|did not|fail(?:s|ed)? to) report(?:ing)? to", re.I),
    "contract dispute": re.compile(
        r"(?:contract|extension) (?:dispute|standoff|impasse|disagreement|stalemate)|unhappy with (?:his |the )?contract|"
        r"reject(?:s|ed)? (?:a |the )?(?:contract )?extension|(?:contract|extension) (?:talks|negotiations) (?:stall|break|collaps)|"
        r"growing cloud", re.I),
    "leave of absence": re.compile(
        r"leave of absence|personal reasons|personal matter|away from the team|excused absence", re.I),
    "suspension": re.compile(r"\bsuspen(?:d|ds|ded|sion|sions)\b", re.I),
}

# injury-report off-court categories, matched on category + detail text. "Not With Team" lines whose detail is really a
# G League reconditioning stint or a covid isolation are not off-court.
_REPORT_KINDS = (
    ("trade pending", re.compile(r"trade pending", re.I)),
    ("suspension", re.compile(r"suspension", re.I)),
    ("personal reasons", re.compile(r"personal", re.I)),
    ("not with team", re.compile(r"not with team", re.I)),
)
_NOT_OFFCOURT = re.compile(r"return to competition|reconditioning|self.?isolat|health and safety|g league", re.I)


def report_kind(category: str | None, detail: str | None) -> str:
    """Off-court kind of one injury-report line, or '' (injury, rest, G League ... are not off-court)."""
    text = f"{category or ''} {detail or ''}"
    if _NOT_OFFCOURT.search(text):
        return ""
    for name, pat in _REPORT_KINDS:
        if pat.search(text):
            return name
    return ""


_DENIAL = re.compile(r"denie[sd]|denying|(?:has not|hasn'?t|never|not) request|no trade request|rumou?rs?", re.I)


def tag_news(news: pd.DataFrame) -> pd.DataFrame:
    """``news`` plus ``kinds``: the matched keyword kinds, comma-separated ('' when none)."""
    out = news.copy()
    text = (out["headline"].fillna("") + ". " + out["description"].fillna("")).tolist()
    out["kinds"] = [", ".join(k for k, pat in NEWS_KINDS.items() if pat.search(t) and not (k == "trade request" and _DENIAL.search(t)))
                    for t in text]
    return out


def injury_report_flags(board_ids, reports: pd.DataFrame | None, season: str) -> pd.DataFrame:
    """``report_flag`` per player from the newest season in ``reports`` that precedes ``season``."""
    ids = np.asarray(board_ids, dtype="int64")
    out = pd.DataFrame({"player_id": ids, "report_flag": ""})
    if reports is None or reports.empty:
        return out
    prior = reports[reports["season"].map(season_start) < season_start(season)]
    if prior.empty:
        return out
    last = max(prior["season"], key=season_start)
    r = prior[(prior["season"] == last) & prior["player_id"].notna() & (prior["status"] == "Out")].copy()
    r["kind"] = [report_kind(c, d) for c, d in zip(r["category"], r["detail"])]
    r = r[r["kind"] != ""]
    if r.empty:
        return out
    r["player_id"] = r["player_id"].astype("int64")
    n = r.drop_duplicates(["player_id", "game_date", "kind"]).groupby(["player_id", "kind"]).size()
    text: dict[int, str] = {}
    for pid, g in n.groupby(level=0):
        parts = [f"{int(v)} game{'s' if v != 1 else ''} out for {k}" for (_, k), v in g.sort_values(ascending=False).items()
                 if v >= MIN_REPORT_GAMES]
        if parts:
            text[int(pid)] = f"{'; '.join(parts)} in {last}"
    out["report_flag"] = [text.get(int(i), "") for i in ids]
    return out


def news_flags(board_ids, news: pd.DataFrame | None, *, today, days: int = NEWS_DAYS) -> pd.DataFrame:
    """``news_flag`` per player: newest matching article per kind within ``days``, with its date and a headline excerpt."""
    ids = np.asarray(board_ids, dtype="int64")
    out = pd.DataFrame({"player_id": ids, "offcourt_news_flag": ""})
    if news is None or news.empty:
        return out
    t = tag_news(news.dropna(subset=["player_id"]))
    t = t[(t["kinds"] != "") & (t["published"] >= pd.Timestamp(today) - pd.Timedelta(days=days))]
    if t.empty:
        return out
    t["player_id"] = t["player_id"].astype("int64")
    text: dict[int, str] = {}
    for pid, g in t.sort_values("published", ascending=False).groupby("player_id"):
        seen: dict[str, str] = {}
        for _, r in g.iterrows():
            for k in r["kinds"].split(", "):
                if k not in seen:
                    head = r["headline"] if len(r["headline"]) <= HEADLINE_CHARS else r["headline"][:HEADLINE_CHARS - 1].rstrip() + "…"
                    seen[k] = f'{k}, {r["published"]:%b %d} ("{head}")'
        text[int(pid)] = "off-court news: " + "; ".join(seen.values())
    out["offcourt_news_flag"] = [text.get(int(i), "") for i in ids]
    return out


def offcourt_flags(board_ids, reports: pd.DataFrame | None, news: pd.DataFrame | None, season: str, *, today) -> pd.DataFrame:
    """One ``offcourt_flag`` per player joining the injury-report and the news parts."""
    a, b = injury_report_flags(board_ids, reports, season), news_flags(board_ids, news, today=today)
    flag = ["; ".join(p for p in pair if p) for pair in zip(a["report_flag"], b["offcourt_news_flag"])]
    return pd.DataFrame({"player_id": a["player_id"], "offcourt_flag": flag})


# --------------------------------------------------------------------------- hand-written notes

def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace(".", "")).strip()


def read_notes(path: Path | None = None) -> list[dict]:
    """Entries of the notes file: ``[{player | player_id, note, date?}]``. A missing or empty file is no notes."""
    path = path or NOTES_PATH
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("notes") if isinstance(data, dict) else data
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a list under 'notes:'")
    return [e for e in entries if isinstance(e, dict)]


def note_flags(board_ids, names: pd.Series, entries: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """``(frame with note_flag, problems)``. ``names`` maps player_id -> name; an entry matches by ``player_id`` or a
    case/accent-insensitive full name. Unmatched, ambiguous or note-less entries are reported, never guessed."""
    ids = np.asarray(board_ids, dtype="int64")
    on_board = set(ids.tolist())
    by_name: dict[str, list[int]] = {}
    for pid, nm in names.items():
        by_name.setdefault(_norm(nm), []).append(int(pid))
    text: dict[int, list[str]] = {}
    problems: list[str] = []
    for e in entries:
        note = str(e.get("note") or "").strip()
        who = e.get("player_id", e.get("player"))
        if not note or who is None:
            problems.append(f"player notes: skipped an entry without player and note ({e})")
            continue
        if "player_id" in e:
            hits = [int(e["player_id"])]
        else:
            hits = by_name.get(_norm(who), [])
        hits = [h for h in hits if h in on_board]
        if len(hits) != 1:
            problems.append(f"player notes: '{who}' {'matches no board player' if not hits else 'is ambiguous'}; note skipped")
            continue
        date = e.get("date")
        text.setdefault(hits[0], []).append(f"note: {note}" + (f" ({date})" if date else ""))
    frame = pd.DataFrame({"player_id": ids, "note_flag": ["; ".join(text.get(int(i), [])) for i in ids]})
    return frame, problems
