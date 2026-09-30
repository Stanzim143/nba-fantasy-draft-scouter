"""Match external-source player identities onto the canonical NBA ``player_id``.

Used by ``espn_adp.py`` to populate the ``player_id_map`` contract table (see
``src/contracts.py``). The approach follows the feasibility test in
``docs/research/espn-api-findings.md`` section 8: normalized-name matching against the local
``players`` table, with a small manual alias table for known name-form collisions (nicknames,
legal-name changes) and a year-window tie-break for ambiguous names, falling back to fuzzy
matching only when nothing else resolves. Anything that stays ambiguous or unmatched is reported,
never silently guessed into the table.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Known name-form collisions between ESPN's player names and nba_api's canonical names: legal-name
# changes, common nickname/full-name mismatches, and similar. Each inner set holds every normalized
# spelling that must resolve to the same person. Found by cross-referencing the miss list in
# docs/research/espn-api-findings.md section 8 plus well-known NBA name changes; extend this table
# as real ingest runs surface new collisions (see the "unmatched"/"ambiguous" report).
ALIAS_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"enes kanter", "enes freedom"}),
    frozenset({"lou williams", "louis williams"}),
    frozenset({"cam reddish", "cameron reddish"}),
    frozenset({"cam thomas", "cameron thomas"}),
    frozenset({"cam johnson", "cameron johnson"}),
    frozenset({"nic claxton", "nicolas claxton"}),
    frozenset({"og anunoby", "ogugua anunoby"}),
    frozenset({"metu", "chimezie metu"}),
    frozenset({"nene", "nene hilario", "maybyner hilario"}),
    frozenset({"kevin knox", "kevin knox ii"}),
    frozenset({"marcus morris", "marcus morris sr"}),
    frozenset({"otto porter", "otto porter jr"}),
    frozenset({"gary trent", "gary trent jr"}),
    frozenset({"kelly oubre", "kelly oubre jr"}),
    frozenset({"troy brown", "troy brown jr"}),
    frozenset({"wendell carter", "wendell carter jr"}),
    frozenset({"vernon carey", "vernon carey jr"}),
    frozenset({"derrick jones", "derrick jones jr"}),
    frozenset({"jaren jackson", "jaren jackson jr"}),
    frozenset({"kevin porter", "kevin porter jr"}),
    frozenset({"tim hardaway", "tim hardaway jr"}),
    frozenset({"larry nance", "larry nance jr"}),
    frozenset({"jabari smith", "jabari smith jr"}),
    frozenset({"aj dybantsa", "anicet dybantsa"}),      # ESPN's transactions feed uses the long form (2026 rookie, unique surname)
    frozenset({"scotty pippen", "scotty pippen jr"}),
    frozenset({"gary payton", "gary payton ii"}),
    frozenset({"glenn robinson", "glenn robinson iii"}),
    frozenset({"kira lewis", "kira lewis jr"}),
    frozenset({"harry giles", "harry giles iii"}),
    frozenset({"jaime jaquez", "jaime jaquez jr"}),
    frozenset({"ron holland", "ronald holland", "ron holland ii"}),
    frozenset({"bub carrington", "carlton carrington"}),
)

_ALIAS_KEY: dict[str, str] = {}
for _group in ALIAS_GROUPS:
    _canonical = min(_group)  # deterministic representative
    for _spelling in _group:
        _ALIAS_KEY[_spelling] = _canonical


def normalize_name(name: str) -> str:
    """Accent/punctuation/suffix-insensitive comparison key.

    ``"Dončić"`` -> ``"doncic"``, ``"Jimmy Butler III"`` -> ``"jimmy butler"``,
    ``"Gary Trent Jr."`` -> ``"gary trent"``.
    """
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[.'\-]", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    tokens = [t for t in s.split() if t]
    if len(tokens) > 1 and tokens[-1] in SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens)


def alias_key(name: str) -> str:
    """``normalize_name`` plus the manual alias table: alternate spellings collapse to one key."""
    n = normalize_name(name)
    return _ALIAS_KEY.get(n, n)


@dataclass
class MatchReport:
    n_total: int = 0
    n_exact: int = 0
    n_normalized: int = 0
    n_fuzzy: int = 0
    n_ambiguous: int = 0
    n_unmatched: int = 0
    ambiguous: list[dict] = field(default_factory=list)
    unmatched: list[dict] = field(default_factory=list)

    @property
    def n_matched(self) -> int:
        return self.n_exact + self.n_normalized + self.n_fuzzy

    @property
    def match_rate(self) -> float:
        return self.n_matched / self.n_total if self.n_total else float("nan")

    def summary(self) -> str:
        return (
            f"{self.n_matched}/{self.n_total} matched ({self.match_rate:.1%}): "
            f"{self.n_exact} exact, {self.n_normalized} normalized/alias, {self.n_fuzzy} fuzzy; "
            f"{self.n_ambiguous} ambiguous, {self.n_unmatched} unmatched"
        )


FUZZY_CUTOFF = 0.90


def match_players(
    source_players: pd.DataFrame,
    nba_players: pd.DataFrame,
    *,
    source: str,
    id_col: str = "source_id",
    name_col: str = "name",
    year_col: str | None = "season_start",
) -> tuple[pd.DataFrame, MatchReport]:
    """Match ``source_players`` (one row per distinct external player) onto NBA ``player_id``.

    ``source_players`` needs ``id_col`` (external id, will become ``source_id``) and ``name_col``.
    If ``year_col`` is present, it is used to prefer NBA candidates whose ``[from_year, to_year]``
    (inclusive, ``to_year`` null treated as open-ended) covers that year when a name is otherwise
    ambiguous. Returns a ``player_id_map``-shaped frame (no ``player_id`` rows for anything
    unresolved) and a :class:`MatchReport` covering every input row, matched or not.
    """
    nba = nba_players[["player_id", "player_name", "from_year", "to_year"]].copy()
    nba["norm"] = nba["player_name"].map(normalize_name)
    nba["alias"] = nba["player_name"].map(alias_key)

    by_norm: dict[str, list[int]] = {}
    by_alias: dict[str, list[int]] = {}
    for row in nba.itertuples(index=False):
        by_norm.setdefault(row.norm, []).append(row.player_id)
        by_alias.setdefault(row.alias, []).append(row.player_id)
    all_norms = nba["norm"].tolist()

    def year_window_filter(candidates: list[int], year: float | None) -> list[int]:
        if year is None or pd.isna(year) or len(candidates) <= 1:
            return candidates
        sub = nba[nba["player_id"].isin(candidates)]
        y = int(year)
        ok = sub[(sub["from_year"].isna() | (sub["from_year"] <= y + 1))
                 & (sub["to_year"].isna() | (sub["to_year"] >= y - 1))]
        picked = ok["player_id"].tolist()
        return picked if len(picked) == 1 else candidates

    report = MatchReport()
    out_rows: list[dict] = []
    for row in source_players.itertuples(index=False):
        d = row._asdict()
        report.n_total += 1
        src_id = d[id_col]
        name = d[name_col]
        year = d.get(year_col) if year_col else None
        norm = normalize_name(name)
        alias = alias_key(name)

        candidates = by_norm.get(norm, [])
        method = "exact"
        if not candidates:
            candidates = by_alias.get(alias, [])
            method = "normalized"

        resolved = year_window_filter(candidates, year)
        if len(resolved) == 1:
            confidence = 1.0 if method == "exact" else 0.9
            out_rows.append({"player_id": resolved[0], "source": source, "source_id": str(src_id),
                             "source_name": name, "match_method": method, "confidence": confidence})
            if method == "exact":
                report.n_exact += 1
            else:
                report.n_normalized += 1
            continue

        if len(candidates) > 1:
            report.n_ambiguous += 1
            report.ambiguous.append({"source_id": str(src_id), "name": name,
                                     "candidates": candidates, "year": year})
            continue

        # nothing exact/aliased: try fuzzy, restricted to a plausible year window if we have one
        pool = all_norms
        close = difflib.get_close_matches(norm, pool, n=3, cutoff=FUZZY_CUTOFF)
        fuzzy_candidates: list[int] = []
        for c in close:
            fuzzy_candidates.extend(by_norm[c])
        fuzzy_candidates = year_window_filter(sorted(set(fuzzy_candidates)), year)
        if len(fuzzy_candidates) == 1:
            ratio = difflib.SequenceMatcher(None, norm, close[0]).ratio()
            out_rows.append({"player_id": fuzzy_candidates[0], "source": source, "source_id": str(src_id),
                             "source_name": name, "match_method": "fuzzy", "confidence": round(ratio, 3)})
            report.n_fuzzy += 1
            continue
        if len(fuzzy_candidates) > 1:
            report.n_ambiguous += 1
            report.ambiguous.append({"source_id": str(src_id), "name": name,
                                     "candidates": fuzzy_candidates, "year": year})
            continue

        report.n_unmatched += 1
        report.unmatched.append({"source_id": str(src_id), "name": name, "year": year})

    cols = ["player_id", "source", "source_id", "source_name", "match_method", "confidence"]
    frame = pd.DataFrame(out_rows, columns=cols)
    if len(frame):
        frame["player_id"] = frame["player_id"].astype("int64")
        frame["confidence"] = frame["confidence"].astype("float64")
    return frame, report
