"""In-season tools (roadmap phase 5, ADR 0015): schedule awareness, rest-of-season projection, trade
analyzer and waiver / free-agent finder.

Modules (each with a CLI, ``python -m src.inseason.<name> --help``)::

    weeks       the league's matchup weeks (ESPN ``matchupPeriods``, or a documented derived fallback)
    schedule    the ``schedule_games`` table, games per team per matchup week, back-to-backs, off nights
    ros         rest-of-season projection: preseason projection shrunk toward season-to-date production
    ros_eval    walk-forward evidence that the blend beats preseason-only (and season-to-date-only)
    lineup      slot-aware roster value (max-weight matching of players to the league's lineup slots)
    trade       trade analyzer: rest-of-season FP in vs out, with the freed / filled roster spots
    signals     minutes / usage trends and injury-replacement beneficiaries from game logs
    waivers     free-agent finder and streaming candidates
    context     shared loading (tables, ROS frame, schedule, league rosters)

Every function that reads season-to-date data takes an ``as_of`` date: games dated after it never
influence the output (leakage tests in ``tests/inseason``).
"""
