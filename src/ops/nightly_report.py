"""Render the nightly in-season report (``reports/nightly/<date>.md``) from a run record (ADR 0018).

Like ``daily_report`` this only formats a plain dict built by ``src.ops.nightly``, so it is tested offline against hand-made
records. Order: header and alerts, steps, what changed since the previous night, the week ahead, recommendations, alerts on
players, lineup hints, data freshness, files.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

MAX_LINES = 25


def _cap(items: list[str], n: int = MAX_LINES) -> list[str]:
    return items[:n] + ([f"- ... and {len(items) - n} more"] if len(items) > n else [])


def _section(title: str, lines: list[str], empty: str) -> list[str]:
    return ["", f"## {title}", ""] + (lines if lines else [empty])


def _cell(v: Any, nd: int = 1) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    if isinstance(v, bool):
        return "yes" if v else ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v).replace("|", "/").replace("\n", " ")


def table(rows: Sequence[Mapping[str, Any]], cols: Sequence[tuple[str, str]], limit: int) -> list[str]:
    """Markdown table of ``rows`` with ``(key, header)`` columns; columns that are empty everywhere are dropped."""
    rows = list(rows)[:limit]
    keep = [(k, h) for k, h in cols if any(_cell(r.get(k)) for r in rows)]
    if not rows or not keep:
        return []
    out = ["| " + " | ".join(h for _, h in keep) + " |", "|" + "---|" * len(keep)]
    out += ["| " + " | ".join(_cell(r.get(k)) for k, _ in keep) + " |" for r in rows]
    return out


def render_report(run: Mapping[str, Any]) -> str:
    d = run.get("diffs") or {}
    an = run.get("analysis") or {}
    meta = an.get("meta") or {}
    tables = an.get("tables") or {}
    out = [f"# Nightly {run['date']}: {run['outcome'].upper()}", ""]
    if run.get("team_setup"):
        out += [f"**ACTION NEEDED: {run['team_setup']}.**", ""]
    out.append(f"Run `{run['run_id']}` | season {run['season']} | games completed through {run['completed_through']}, next slate "
               f"{run['slate']} | {run.get('seconds', 0):.0f}s | {run.get('days_left', '?')} day(s) left in the window "
               f"({run['window'][0]} .. {run['window'][1]})")
    if run.get("replay"):
        out.append("\nREPLAY: this run was made with `--as-of`; alerts and freshness checks are off and the data may be historical.")
    if run.get("prev_run_id"):
        out.append(f"\nChanges below are since the previous run `{run['prev_run_id']}` ({run.get('prev_time', '?')}).")
    else:
        out.append("\nFirst archived nightly run: nothing to diff against yet; the next run reports changes.")
    if (run.get("alert") or {}).get("reasons"):
        out += ["", "## ALERT", ""] + [f"- {r}" for r in run["alert"]["reasons"]]
    if meta.get("team_skipped"):
        out += ["", f"Setup needed: team-specific sections are off. {meta['team_skipped']}."]
    if meta.get("mock"):
        out += ["", "MOCK roster: the team below is team N of a mock snake draft, not your real roster."]

    steps = run.get("steps") or []
    out += ["", "## Steps", "", "| step | status | seconds | note |", "|---|---|---|---|"]
    for s in steps:
        note = (s.get("error") or s.get("note") or "").replace("|", "/").replace("\n", " ")[:160]
        out.append(f"| {s['name']} | {s['status']} | {s.get('seconds', 0):.0f} | {note} |")
    fails = [s for s in steps if s["status"] in ("failed", "timeout", "degraded")]
    if fails:
        out += ["", "Failures:"] + [f"- **{s['name']}** ({s['status']}): {(s.get('error') or '').strip()[:400]}" for s in fails]
        out.append("\nSteps that did run are current; sections built from data of a failed step say so or use what is on disk.")
    for name, err in (an.get("errors") or {}).items():
        out.append(f"- artifact `{name}` failed: {err[:300]}")

    # ---- games
    g = run.get("games")
    lines = []
    if g:
        if g.get("skipped"):
            lines.append(f"- {g['skipped']}")
        else:
            lines.append(f"- {g.get('new_games', 0)} new game(s), {g.get('new_player_rows', 0)} new player rows"
                         + (f", {g['new_players']} new player(s) added to the players table" if g.get("new_players") else "")
                         + f"; games stored through {g.get('last_game_date')} (window pulled from {g.get('date_from') or 'season start'}; "
                         f"{g.get('network_requests', 0)} request(s))")
    wk = meta.get("week") or {}
    if wk.get("slate_games") == 0:
        lines.append(f"- no games on the next slate ({run['slate']})"
                     + (f"; next game day {wk['next_game_day']} (an off day, the All-Star break, or the season has not started)" if wk.get("next_game_day") else "; no later games scheduled"))
    elif wk:
        lines.append(f"- next slate {run['slate']}: {wk.get('slate_games')} game(s)")
    out += _section("Games and schedule", lines, "games step did not run")
    sc = run.get("schedule")
    if sc:
        out.append(f"- schedule table: {sc['games']} games; " + ("unchanged" if not sc["changed"] else
                   f"CHANGED: {sc['moved']} moved, {sc['added']} added, {sc['removed']} removed"))
    if (d.get("calendar") or None):
        out.append(f"- MATCHUP CALENDAR SOURCE CHANGED: {d['calendar']['from']} -> {d['calendar']['to']} (ESPN published the real periods)"
                   if d["calendar"]["to"] == "espn" else f"- matchup calendar source changed: {d['calendar']['from']} -> {d['calendar']['to']}")

    # ---- what changed
    out += ["", "# What changed"]
    inj = d.get("injuries")
    lines = [f"- {'**MINE** ' if x.get('mine') else ''}{x['name']}: {x['from']} -> {x['to']} ({(x.get('pct_owned') or 0):.0f}% owned)" for x in (inj or [])]
    out += _section("Injury status changes (ESPN)", _cap(lines), "none (or no previous snapshot to compare)")
    lg = d.get("league") or {}
    lines = []
    for m in lg.get("moves", []):
        lines.append(f"- {'**MINE** ' if m['mine'] else ''}{m['team']}: added {', '.join(m['added']) or 'nobody'}; dropped {', '.join(m['dropped']) or 'nobody'}")
    lines += [f"- transaction ({x['status']}) {x['team']}: {', '.join(x['items'])}" for x in lg.get("new_transactions", [])]
    out += _section("League roster moves and transactions", _cap(lines), "none (or no previous league sync)")
    r = d.get("roster")
    lines = []
    if r:
        lines += [f"- signed/arrived: {x['name']} ({x['to']})" for x in r["arrived"]]
        lines += [f"- left NBA rosters: {x['name']} (was {x['from']})" for x in r["departed"]]
        lines += [f"- traded/moved: {x['name']} {x['from']} -> {x['to']}" for x in r["moved"]]
    out += _section("NBA roster moves", _cap(lines), "none (or no previous snapshot)")
    lines = []
    for key, label in (("rising", "now rising in minutes/usage"), ("beneficiaries", "newly expected to gain minutes")):
        x = d.get(key)
        if x:
            lines += [f"- {label}: {y['name']}" + (f" (+{y['min_delta']} min)" if 'min_delta' in y else f" (+{y.get('gain_mpg')} min, {y.get('out')} is out)") for y in x["new"]]
            if key == "rising":
                lines += [f"- no longer flagged: {y['name']}" for y in x["gone"]]
    out += _section("Minutes trends", _cap(lines), "no new flags")
    lines = []
    ad = d.get("adds")
    if ad:
        lines += [f"- NEW top add: {y['name']} (+{y['gain']} FP over {y['drop']})" for y in ad["new"]]
        lines += [f"- no longer a top add: {y['name']}" for y in ad["gone"]]
    mine = d.get("my_roster")
    if mine:
        lines.append(f"- your roster changed since the last run (player ids added {mine['added']}, dropped {mine['dropped']})")
    out += _section("Waiver board changes", _cap(lines), "top adds unchanged (or no previous run)")
    ro = d.get("ros")
    lines = []
    if ro:
        lines += [f"- riser {y['name']}: #{y['from']} -> #{y['to']}" for y in ro["risers"]]
        lines += [f"- faller {y['name']}: #{y['from']} -> #{y['to']}" for y in ro["fallers"]]
        lines += [f"- entered the top 60: {n}" for n in ro["entered"]] + [f"- left the top 60: {n}" for n in ro["left_top"]]
    out += _section("Rest-of-season rank movers (top 60)", _cap(lines), "no moves of 8+ places")

    # ---- week ahead
    out += ["", "# The week ahead"]
    for label, key in (("This week", "this_week"), ("Next week", "next_week")):
        w = wk.get(key)
        rows = tables.get(f"week_{key}")
        if not w or not rows:
            continue
        col = "games_left" if key == "this_week" else "games"
        out += ["", f"## {label}: matchup week {w['week']} ({w['start']} .. {w['end']}, {w['kind']}; league mean {w['mean_games']:.2f} games/team; calendar {wk.get('calendar_source')})", ""]
        best = sorted(rows, key=lambda x: (-x[col], -x["off_night_games"]))[:8]
        out += ["Most games" + (" left" if key == "this_week" else "") + ": " + ", ".join(f"{x['abbr']} {x[col]:.0f}" + (f" ({x['b2b']:.0f} b2b)" if x["b2b"] else "") for x in best)]
        light = [x["abbr"] for x in rows if x.get("light")]
        heavy = [x["abbr"] for x in rows if x.get("heavy")]
        if heavy:
            out.append(f"\nHeavy (4+ games): {', '.join(heavy)}")
        if light:
            out.append(f"\nLight (2 or fewer): {', '.join(light)}")
        mine_rows = [x for x in rows if x.get("mine")]
        if mine_rows:
            out += ["", "Your teams: " + ", ".join(f"{x['abbr']} {x[col]:.0f}" for x in mine_rows)]
    if not wk:
        out += ["", "No week data this run (the schedule table is missing or the analysis step failed)."]

    # ---- recommendations
    out += ["", "# Recommendations"]
    adds = tables.get("waivers_adds") or []
    out += ["", f"## Waiver adds ({meta.get('team_label') or 'no team configured'})", ""]
    pos = [a for a in adds if (a.get("gain_ros_fp") or 0) > 0]
    if pos:
        out += table(pos, [("name", "player"), ("position", "pos"), ("team", "team"), ("gain_ros_fp", "ROS gain"), ("drop", "drop"),
                           ("ros_fppg", "FPPG"), ("ros_games", "ROS games"), ("min_delta", "min trend"), ("inherits_from", "inherits from"),
                           ("injury", "injury"), ("in_espn_fa_list", "on ESPN FA list")], 15)
    elif adds:
        out.append("No free agent improves your roster over the rest of the season (every add-drop is 0 or negative).")
    elif tables.get("free_agents_top"):
        out.append("No team configured; the best free agents by rest-of-season value:")
        out += [""] + table(tables["free_agents_top"], [("name", "player"), ("position", "pos"), ("team", "team"), ("ros_fppg", "FPPG"), ("ros_total_fp", "ROS FP")], 15)
    else:
        out.append("not available this run")
    for label, key, wlabel in (("Streaming, this week", "waivers_streams", meta.get("stream_week")), ("Streaming, next week", "waivers_streams_next", meta.get("stream_week_next"))):
        rows = [x for x in (tables.get(key) or []) if (x.get("week_gain_fp") or 0) > 0]
        if rows:
            out += ["", f"## {label} ({wlabel})", ""]
            out += table(rows, [("name", "player"), ("position", "pos"), ("team", "team"), ("week_exp_games", "exp games"), ("week_b2b", "b2b"),
                                ("week_off_night", "off-night"), ("week_exp_fp", "exp FP"), ("week_gain_fp", "gain vs drop"), ("drop", "drop"), ("injury", "injury")], 10)

    # ---- alerts
    out += ["", "# Player alerts"]
    for title, key, cols in (
            ("Rising minutes / usage", "alerts_rising", [("name", "player"), ("team", "team"), ("status", "whose"), ("base_mpg", "earlier mpg"), ("recent_mpg", "last 5 mpg"), ("min_delta", "delta"), ("recent_fppg", "recent FPPG"), ("injury", "injury")]),
            ("Injury beneficiaries (a rotation teammate is out)", "alerts_beneficiaries", [("name", "player"), ("team", "team"), ("status", "whose"), ("out_name", "because"), ("out_streak", "games out"), ("gain_mpg", "exp +min"), ("gain_fppg", "exp +FPPG"), ("injury", "injury")])):
        rows = tables.get(key) or []
        out += ["", f"## {title}", ""]
        out += table(rows, cols, 12) or ["none flagged"]
    absent = tables.get("alerts_absent") or []
    if absent:
        out += ["", "## Rotation players currently out", ""]
        out += table(absent, [("name", "player"), ("team", "team"), ("status", "whose"), ("mpg", "mpg"), ("streak", "games missed"), ("source", "source")], 10)

    # ---- lineup
    lu = meta.get("lineup")
    out += ["", "# Lineup hints", ""]
    if lu and tables.get("lineup"):
        out.append(f"Week {lu['week']}: best lineup expects {lu['week_expected_fp']:.0f} FP from your starters this week (games left x availability x FPPG; "
                   f"daily lineups lock at tip-off, this is a weekly view). Next slate {lu['slate']}: {lu['today_players']} of your players have a game "
                   f"({lu['today_expected_fp']:.0f} expected FP).")
        if lu.get("empty_slots"):
            out.append(f"\nEmpty starting slots: {', '.join(lu['empty_slots'])}")
        out += [""] + table(tables["lineup"], [("name", "player"), ("position", "pos"), ("team", "team"), ("slot_this_week", "slot"), ("games_left_week", "games left"),
                                                ("plays_today", "plays"), ("today_slot", "slot on the slate"), ("injury", "injury"), ("week_exp_fp", "wk FP"), ("hint", "hint")], 20)
        for s in lu.get("swaps", []):
            out.append(f"\nConsider: add {s['add']} for {s['drop']} ({s['games']:.1f} games, +{s['week_gain_fp']:.0f} FP this week)")
    else:
        out.append("not available (needs a configured team, a synced league with a drafted roster, and the schedule)")
    for n in an.get("notes") or []:
        out.append(f"\nNote: {n}")

    f = run.get("freshness") or {}
    if f:
        out += ["", "## Data freshness", ""] + [f"- {k}: {v}" for k, v in f.items()]
    if run.get("files"):
        out += ["", "## Files", ""] + [f"- {k}: `{v}`" for k, v in run["files"].items()]
    return "\n".join(out) + "\n"

