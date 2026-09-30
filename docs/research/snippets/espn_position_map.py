"""ESPN basketball position ids, lineup-slot ids and eligibility rules for OUR league's slots.

RESEARCH PROTOTYPE - not imported by the codebase, not run by pytest.

Verified 2026-09-22 against the live ESPN player universe (season_id 2027, 2,694 players that carry
``eligibleSlots``; see espn_players_adp.py):
  * ``defaultPositionId``: 1=PG 2=SG 3=SF 4=PF 5=C  (== base slot id + 1; equals espn-api's
    ``POSITION_MAP[defaultPositionId - 1]``).  -1 / 0 appear for inactive or unassigned players.
  * ``eligibleSlots`` uses lineup-slot ids:
        0 PG, 1 SG, 2 SF, 3 PF, 4 C, 5 G, 6 F, 7 SG/SF, 8 G/F, 9 PF/C, 10 F/C, 11 UTIL, 12 BE, 13 IR
    and every ESPN combo slot is DERIVED from the base positions - these rules held for 2,694 / 2,694 players:
        G (5)      <=> PG or SG          F (6)      <=> SF or PF
        SG/SF (7)  <=> SG or SF          G/F (8)    <=> PG/SG/SF/PF (any non-C)
        PF/C (9)   <=> PF or C           F/C (10)   <=> SF, PF or C
        UTIL, BE, IR: always present.
  * So a C-only player (e.g. Jokic, slots [4, 9, 10, 11, 12, 13]) is NOT eligible for our F slot; a PG/SG is
    eligible for G.  ~10% of players carry two base positions.
  * Our league (config/league.yaml): PG SG SF PF C G F UTIL x3 -> slot ids 0,1,2,3,4,5,6,11.

Not verified: ESPN's rule for GAINING a new position eligibility (games-played threshold) - it is not in the
API payload for 2027; ``gamesPlayedByPosition`` appears only in pre-2018 season payloads.
"""
from __future__ import annotations

SLOT = {0: "PG", 1: "SG", 2: "SF", 3: "PF", 4: "C", 5: "G", 6: "F", 7: "SG/SF", 8: "G/F",
        9: "PF/C", 10: "F/C", 11: "UTIL", 12: "BE", 13: "IR"}
SLOT_ID = {v: k for k, v in SLOT.items()}
DEFAULT_POSITION = {1: "PG", 2: "SG", 3: "SF", 4: "PF", 5: "C"}     # defaultPositionId -> base position
OUR_STARTING_SLOTS = {"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1, "G": 1, "F": 1, "UTIL": 3}


def base_positions(eligible_slots: list[int]) -> set[str]:
    return {SLOT[s] for s in eligible_slots if s <= 4}


def eligible_for_our_slots(eligible_slots: list[int]) -> set[str]:
    """Which of OUR starting slots (PG SG SF PF C G F UTIL) this player can occupy."""
    return {SLOT[s] for s in eligible_slots if SLOT[s] in OUR_STARTING_SLOTS}


def derive_slots(base: set[str]) -> set[int]:
    """Inverse: derive the full eligibleSlots list from base positions (used to self-check the rules)."""
    s = {SLOT_ID[b] for b in base}
    if base & {"PG", "SG"}:
        s.add(5)
    if base & {"SF", "PF"}:
        s.add(6)
    if base & {"SG", "SF"}:
        s.add(7)
    if base & {"PG", "SG", "SF", "PF"}:
        s.add(8)
    if base & {"PF", "C"}:
        s.add(9)
    if base & {"SF", "PF", "C"}:
        s.add(10)
    return s | {11, 12, 13}


if __name__ == "__main__":
    # Real payload fragments copied from the 2027 pull (Jokic, Gilgeous-Alexander, Edwards, Tatum)
    fixtures = {
        "Nikola Jokic": [4, 9, 10, 11, 12, 13],
        "Shai Gilgeous-Alexander": [0, 5, 8, 11, 12, 13],
        "Anthony Edwards": [1, 2, 5, 6, 7, 8, 10, 11, 12, 13],
    }
    for name, slots in fixtures.items():
        assert derive_slots(base_positions(slots)) == set(slots), name
        print(f"{name:26s} base={sorted(base_positions(slots))} our_slots={sorted(eligible_for_our_slots(slots))}")
