"""Map dataset position strings onto ESPN lineup slots.

The dataset carries one coarse position string per player (``'G'``, ``'F-C'``, ``'G-F'``; ingest
may also deliver the long forms ``'Guard-Forward'`` or the specific ``'PG'``/``'SF'``). ESPN
lineups have five specific positions plus flexible slots::

    PG  SG  SF  PF  C      specific positions
    G                      PG or SG
    F                      SF or PF
    UTIL                   anyone

ESPN decides eligibility from games played at each position, which we do not have. The mapping
below is therefore a documented approximation:

=========================  =======================  ==========================================
dataset position           eligible positions       comment
=========================  =======================  ==========================================
``G``                      PG, SG                   full guard group
``F``                      SF, PF                   full forward group
``C``                      C
``PG`` / ``SG`` / ...      itself                   specific tokens are taken literally
``G-F`` / ``F-G``          SG, SF                   the guard/forward seam
``F-C`` / ``C-F``          PF, C                    the forward/center seam
any other combination      union of the tokens      e.g. ``PG-SG`` -> PG, SG
missing / unparseable      none (UTIL only)         never invented; flagged by ``is_known``
=========================  =======================  ==========================================

Flex slots follow: a player fills ``G`` if eligible at PG or SG, ``F`` if eligible at SF or PF,
and every player fills ``UTIL``.
"""
from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

ESPN_POSITIONS = ("PG", "SG", "SF", "PF", "C")
FLEX_SLOTS = {"G": ("PG", "SG"), "F": ("SF", "PF")}
UTIL = "UTIL"
ALL_SLOTS = ESPN_POSITIONS + tuple(FLEX_SLOTS) + (UTIL,)

_LONG = {"GUARD": "G", "FORWARD": "F", "CENTER": "C", "CENTRE": "C"}
_TOKEN_SETS = {
    "PG": ("PG",), "SG": ("SG",), "SF": ("SF",), "PF": ("PF",), "C": ("C",),
    "G": ("PG", "SG"), "F": ("SF", "PF"),
}
_SEAMS = {frozenset({"G", "F"}): ("SG", "SF"), frozenset({"F", "C"}): ("PF", "C")}


def _tokens(position: object) -> list[str]:
    if position is None or (not isinstance(position, str) and pd.isna(position)):
        return []
    raw = re.split(r"[-/,\s]+", str(position).strip().upper())
    toks = [_LONG.get(t, t) for t in raw if t]
    return [t for t in toks if t in _TOKEN_SETS]


def is_known(position: object) -> bool:
    """True when the position string could be parsed into at least one ESPN position."""
    return bool(_tokens(position))


def eligible_positions(position: object) -> tuple[str, ...]:
    """ESPN specific positions (PG/SG/SF/PF/C) a player with this dataset position can fill."""
    toks = _tokens(position)
    if not toks:
        return ()
    uniq = list(dict.fromkeys(toks))
    if len(uniq) == 1:
        return _TOKEN_SETS[uniq[0]]
    if frozenset(uniq) in _SEAMS:
        return _SEAMS[frozenset(uniq)]
    out: set[str] = set()
    for t in uniq:
        out.update(_TOKEN_SETS[t])
    return tuple(p for p in ESPN_POSITIONS if p in out)


def eligible_slots(position: object) -> tuple[str, ...]:
    """All ESPN lineup slots (specific, flex and UTIL) the player can occupy."""
    pos = eligible_positions(position)
    slots = list(pos)
    for flex, members in FLEX_SLOTS.items():
        if any(m in pos for m in members):
            slots.append(flex)
    slots.append(UTIL)
    return tuple(slots)


def position_group(position: object) -> str:
    """Coarse group used as a modelling covariate: ``'G'``, ``'F'``, ``'C'`` or ``'U'`` (unknown).

    Taken from the first token, so ``'F-C'`` is a forward and ``'C-F'`` a center.
    """
    toks = _tokens(position)
    if not toks:
        return "U"
    return {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C"}[toks[0]]


def position_groups(positions: Iterable[object]) -> list[str]:
    return [position_group(p) for p in positions]
