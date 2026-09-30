"""Parse a manually-downloaded FantasyPros "Draft ALL Rankings" CSV export and resolve it onto the
canonical NBA ``player_id``. See ``docs/adr/0028-external-rankings-compare.md`` for the design
writeup and ``src.ingest.external_rankings`` for the shared contract this feeds.

Like ``src.ingest.yahoo_rankings`` (its sibling), this is a **one-off manual export** the user
downloads from FantasyPros' UI -- there is no live HTTP API here, just a file parser.

The real file's ``TEAM`` column is always empty; team, positions, and an optional trailing
injury/status tag are instead packed into the ``PLAYER NAME`` field itself, e.g.::

    "Nikola Jokic (DEN - C)"
    "Anthony Davis (WAS - PF,C) OUT"
    "Russell Westbrook III (FA - PG,SG) RET"

``_FP_PLAYER_RE`` (below) parses that field. Name suffixes ("Jr.", "III", ...) are deliberately
left attached to the captured name -- ``src.ingest.id_map.normalize_name`` strips them downstream,
during ``resolve_players``, so stripping them here too would be redundant (and risks diverging from
that single source of truth).

The ``ECR VS. ADP`` column uses the literal string ``"-"`` to mean "no ADP data available"; that
must become a real null, never ``0`` (a real ``0`` delta is a valid signed value: FantasyPros' ECR
exactly matches ADP for that player).

Following the "never silently guess" discipline in ``docs/adr/0007-adp-ingest.md`` D4 and
``id_map.py``: a ``PLAYER NAME`` value that fails to match the expected pattern raises
:class:`FantasyProsParseError` naming the offending row, rather than producing a partially-wrong
name/team/position silently.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from src.contracts import data_dir
from src.ingest.external_rankings import ResolveResult, resolve_players
from src.store import read_table

SOURCE = "fantasypros"

#: Default location for the manually-downloaded CSV, if the user drops it in the shared data dir
#: under this name rather than passing --path explicitly (see CLI below).
DEFAULT_RELATIVE_PATH = Path("external_rankings") / "fantasypros_latest.csv"

#: Parses FantasyPros' packed "PLAYER NAME" field: "<name> (<team> - <pos,pos,...>) [<tag>]".
#: - ``name``: the raw player name, suffix (Jr., III, ...) still attached -- left for id_map to
#:   strip via normalize_name, never duplicated/stripped here.
#: - ``team``: 2-3 letter abbreviation, or the literal "FA" for a free agent.
#: - ``pos``: a comma-joined list of 1-2 letter position codes (e.g. "PG,SG").
#: - ``tag``: whatever (if anything) trails the closing paren, e.g. "OUT", "DTD", "TWO-WAY", "RET",
#:   or a future/rare tag like "G-League" -- captured generically rather than as a closed enum, so a
#:   legitimate new tag string doesn't crash or get silently dropped.
_FP_PLAYER_RE = re.compile(
    r"^(?P<name>.+?)\s*\((?P<team>[A-Z]{2,3})\s*-\s*(?P<pos>[A-Z]{1,2}(?:,[A-Z]{1,2})*)\)\s*(?P<tag>.*)$"
)

#: Columns ``parse_fantasypros_csv`` returns -- the "parsed" shape ``resolve_players`` requires
#: (see ``external_rankings.resolve_players`` docstring).
PARSED_COLUMNS: tuple[str, ...] = (
    "source_id", "source_name_raw", "name_parsed", "team", "positions",
    "status_tag", "ext_rank", "adp", "ecr_vs_adp",
)


class FantasyProsParseError(RuntimeError):
    """A row's ``PLAYER NAME`` field (or another required column) could not be parsed cleanly."""


def _parse_ecr_vs_adp(raw: object, *, rk: object) -> int | None:
    s = str(raw).strip()
    if s == "-":
        return None
    try:
        return int(s)
    except (TypeError, ValueError) as exc:
        raise FantasyProsParseError(
            f"row RK={rk!r}: could not parse 'ECR VS. ADP' value {raw!r} as an int or '-' sentinel"
        ) from exc


def parse_fantasypros_csv(path: Path) -> pd.DataFrame:
    """Parse a FantasyPros "Draft ALL Rankings" CSV export into :data:`PARSED_COLUMNS` shape.

    Raises :class:`FantasyProsParseError` if a required column is missing, or if any row's
    ``PLAYER NAME`` field or ``ECR VS. ADP`` value doesn't match the expected format -- never
    silently guesses a partial parse.
    """
    df = pd.read_csv(path, dtype=str)
    required = {"RK", "PLAYER NAME", "ECR VS. ADP"}
    missing = required - set(df.columns)
    if missing:
        raise FantasyProsParseError(f"{path}: missing expected column(s) {sorted(missing)}")

    rows: list[dict] = []
    for _, d in df.iterrows():
        rk_raw = d["RK"]
        try:
            rk = int(str(rk_raw).strip())
        except (TypeError, ValueError) as exc:
            raise FantasyProsParseError(f"could not parse RK value {rk_raw!r} as an int") from exc

        raw_name = d["PLAYER NAME"]
        m = _FP_PLAYER_RE.match(str(raw_name).strip())
        if not m:
            raise FantasyProsParseError(
                f"row RK={rk}: could not parse PLAYER NAME field {raw_name!r} "
                f"(expected '<name> (<team> - <pos,pos>) [<tag>]')"
            )
        tag = m.group("tag").strip() or None

        rows.append({
            "source_id": str(rk),
            "source_name_raw": raw_name,
            "name_parsed": m.group("name").strip(),
            "team": m.group("team"),
            "positions": m.group("pos"),
            "status_tag": tag,
            "ext_rank": rk,
            "adp": None,
            "ecr_vs_adp": _parse_ecr_vs_adp(d["ECR VS. ADP"], rk=rk),
        })

    return pd.DataFrame(rows, columns=list(PARSED_COLUMNS))


def load_fantasypros_rankings(path: Path, players: pd.DataFrame) -> ResolveResult:
    """Parse ``path`` and resolve every row onto the canonical NBA ``player_id``."""
    return resolve_players(parse_fantasypros_csv(path), players, source=SOURCE)


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.ingest.fantasypros_rankings",
        description="Parse a manually-downloaded FantasyPros rankings CSV and resolve it onto NBA player_id.",
    )
    p.add_argument("--path", type=Path, default=None,
                   help=f"path to the FantasyPros CSV export (default: <data-dir>/{DEFAULT_RELATIVE_PATH})")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="override the shared data root (default: NBA_DATA_DIR or ~/dev-data/nba-fantasy-2026)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    base = args.data_dir or data_dir()
    path = args.path or (base / DEFAULT_RELATIVE_PATH)

    try:
        players = read_table("players", base)
        result = load_fantasypros_rankings(path, players)
    except (FantasyProsParseError, FileNotFoundError) as exc:
        print(f"fantasypros rankings load failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(result.report.summary())
    unresolved = result.frame[~result.frame["matched"]]
    print(f"{len(unresolved)} unmatched/ambiguous name(s) out of {len(result.frame)} total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
