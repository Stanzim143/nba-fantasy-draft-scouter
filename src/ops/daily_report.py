"""Render the human-readable daily run report (``reports/daily/<date>.md``) from a run record (ADR 0014).

The run record is a plain JSON-able dict built by ``src.ops.daily_refresh``; this module only formats it, so the report
is tested offline against hand-made records.
"""
from __future__ import annotations

from typing import Any, Mapping

MAX_LINES = 25


def _cap(items: list[str], n: int = MAX_LINES) -> list[str]:
    return items[:n] + ([f"- ... and {len(items) - n} more"] if len(items) > n else [])


def _adp(v: float | None) -> str:
    return "unpriced" if v is None else f"{v:.1f}"


def _section(title: str, lines: list[str], empty: str) -> list[str]:
    return ["", f"## {title}", ""] + (lines if lines else [empty])


def render_report(run: Mapping[str, Any]) -> str:
    d = run.get("diffs") or {}
    out = [f"# Daily refresh {run['date']}: {run['outcome'].upper()}", ""]
    days = run.get("days_to_draft")
    draft = run.get("draft_date")
    countdown = ("" if days is None else
                 f" | draft in {days} day(s)" if days > 0 else " | DRAFT DAY" if days == 0 else f" | draft was {-days} day(s) ago")
    out.append(f"Run `{run['run_id']}` | season {run['season']} | finished {run.get('finished_local', run.get('finished_at', '?'))}"
               f" | {run.get('seconds', 0):.0f}s{countdown}")
    hours = run.get("hours_to_draft")
    if run.get("draft_start") and hours is not None:
        out.append(f"\nDraft start: {run['draft_start']}; " + (
            f"{hours:.1f} h from the start of this run." if hours > 0 else "it has already started."))
    if run.get("draft_date_source") == "fallback":
        out.append(f"\nNote: no draft date is set, assuming the latest plausible one ({draft}). Set `draft.date` in `config/league.yaml`, or use "
                   "`--draft-date`, `NBA_DRAFT_DATE` or `draft_date` in `daily_refresh.json` in the data directory.")
    if run.get("prev_run_id"):
        out.append(f"\nChanges below are since the previous run `{run['prev_run_id']}` ({run.get('prev_time', '?')}).")
    else:
        out.append("\nFirst archived run: there is nothing to diff against yet; the next run reports changes.")

    alert = run.get("alert") or {}
    if alert.get("reasons"):
        out += ["", "## ALERT", ""] + [f"- {r}" for r in alert["reasons"]]

    steps = run.get("steps") or []
    fails = [s for s in steps if s["status"] in ("failed", "timeout")]
    out += ["", "## Steps", "", "| step | status | seconds | note |", "|---|---|---|---|"]
    for s in steps:
        note = (s.get("error") or s.get("note") or "").replace("|", "/").replace("\n", " ")[:160]
        out.append(f"| {s['name']} | {s['status']} | {s.get('seconds', 0):.0f} | {note} |")
    if fails:
        out += ["", "Failures:"] + [f"- **{s['name']}** ({s['status']}): {(s.get('error') or '').strip()[:400]}" for s in fails]
        out.append("\nThe steps that did run are still current; the report uses whatever data is on disk for the rest.")

    # ---- roster moves
    r = d.get("roster")
    lines = []
    if r:
        lines += [f"- arrived on a roster: {x['name']} ({x['to']})" for x in r["arrived"]]
        lines += [f"- left the rosters: {x['name']} (was {x['from']})" for x in r["departed"]]
        lines += [f"- moved: {x['name']} {x['from']} -> {x['to']}" for x in r["moved"]]
    out += _section("Roster moves", _cap(lines), "none (or no previous snapshot to compare)")

    # ---- league transactions (ADR 0017)
    tx = d.get("transactions")
    if tx is not None:
        lines = [f"- {x.strip()}" if not x.startswith(" ") else f"  {x.strip()}" for x in tx.get("lines", [])]
        lines += [f"- staff: {x}" for x in tx.get("staff", [])]
        head = ("" if tx.get("ranked") else "No board was available this run, so moves are listed unranked. ")
        out += _section(f"League transactions ({tx.get('new_rows', 0)} new since the previous run; feed through {tx.get('ledger_through', '?')})",
                        ([head] if head else []) + _cap(lines, 40) if lines else [], "no new transactions")

    # ---- head coach changes (ADR 0020)
    cc = d.get("coaches")
    if cc is not None:
        out += _section("Head coach changes (Wikipedia list of current head coaches)",
                        [f"- {x.get('team', x['team_id'])}: {x['from'] or '?'} -> {x['to'] or '?'}" for x in cc], "none")

    # ---- injuries
    inj = d.get("injuries")
    lines = [f"- {x['name']}: {x['from']} -> {x['to']} (ADP {_adp(x['adp'])})" for x in (inj or [])]
    out += _section("Injury status changes (ESPN)", _cap(lines), "none")

    # ---- ADP
    a = d.get("adp")
    lines = []
    if a:
        lines += [f"- riser {x['name']}: {x['from']:.1f} -> {x['to']:.1f} ({x['delta']:+.1f})" for x in a["risers"]]
        lines += [f"- faller {x['name']}: {x['from']:.1f} -> {x['to']:.1f} ({x['delta']:+.1f})" for x in a["fallers"]]
        lines += [f"- now priced {x['name']}: ADP {x['to']:.1f}" for x in a["priced"]]
        lines += [f"- lost ADP {x['name']} (was {x['from']:.1f})" for x in a["unpriced"]]
    out += _section("ADP movers (moves of 2+ picks)", _cap(lines, 40), "no movers")

    # ---- profiles and the status archive (ADR 0016)
    lines = []
    pr = (run.get("extras") or {}).get("profiles")
    if pr:
        lines.append(f"- profiles: {pr.get('profiles')} players ({pr.get('with_birthdate')} with a birthdate), {pr.get('combine_rows')} draft-combine rows"
                     + (f"; combine years failed: {pr['combine_failed_years']}" if pr.get("combine_failed_years") else ""))
    sa = d.get("status_archive")
    if sa:
        ch = sa.get("changed")
        lines.append(f"- status archive: {sa.get('players')} players archived today ({sa.get('by_status')})"
                     + ("" if ch is None else "; counts vs the previous archive: "
                        + (", ".join(f"{k} {a}->{b}" for k, (a, b) in ch.items()) if ch else "unchanged")))
    if lines:
        out += ["", "## Player profiles and status archive", ""] + lines

    # ---- games
    g = d.get("games")
    lines = []
    for ctx, e in (g or {}).items():
        new = e["new"]
        lines.append(f"- {ctx}: {e['now']} games total"
                     + ("" if new is None else f" (+{new} new)") + (f", through {e['last_date']}" if e.get("last_date") else ""))
    out += _section("Summer League / preseason games", lines, "not ingested this run")

    # ---- watchlist
    w = d.get("watchlist")
    lines = []
    if w:
        lines += [f"- NEW on the list: #{x['rank']} {x['name']} ({x.get('team') or '?'}, uplift {x.get('uplift', 0):+.1f})" for x in w["new"]]
        lines += [f"- DROPPED off the list: {x['name']} (was #{x['rank']})" for x in w["dropped"]]
        lines += [f"- moved: {x['name']} #{x['from']} -> #{x['to']}" for x in w["moved"]]
    model = run.get("model")
    out += _section(f"Breakout watchlist changes{f' (model {model})' if model else ''}", _cap(lines),
                    "no changes" if run.get("watchlist_top") else "watchlist not available this run")
    top = run.get("watchlist_top") or []
    if top:
        out += ["", "Top of the watchlist:", "", "| # | player | team | uplift | useful_prob | ADP | evidence |", "|---|---|---|---|---|---|---|"]
        for x in top[:10]:
            p = x.get("useful_prob")
            out.append(f"| {x['rank']} | {x['name']} | {x.get('team') or ''} | {x.get('uplift', 0):+.1f} | "
                       f"{'' if p is None else f'{p:.0%}'} | {_adp(x.get('adp'))} | {x.get('evidence', '')} |")

    # ---- league
    lg = run.get("league")
    if lg:
        out += ["", "## ESPN league", ""]
        out.append(f"- {lg.get('name', 'league')}: {lg.get('teams')} teams, "
                   + ("DRAFT COMPLETED" if lg.get("draft_completed") else "draft not completed yet")
                   + (", draft in progress" if lg.get("draft_in_progress") else "")
                   + f"; {lg.get('n_transactions', 0)} transactions; config discrepancies: {lg.get('discrepancies', 0)}")

    # ---- board
    b = run.get("boards") or {}
    if b:
        out += ["", "## Draft board", ""]
        for model_name, info in b.items():
            note = "; includes the debutants and undrafted signees" if "debut" in model_name else ""
            out.append(f"- `{info['path']}` ({model_name}, {info['rows']} players{note}). Top: "
                       + ", ".join(f"{t['rank']}. {t['name']}" for t in info["top"][:10]))

    # ---- freshness
    f = run.get("freshness") or {}
    if f:
        out += ["", "## Data freshness", ""]
        out += [f"- {k}: {v}" for k, v in f.items()]
    if run.get("files"):
        out += ["", "## Files", ""] + [f"- {k}: `{v}`" for k, v in run["files"].items()]
    return "\n".join(out) + "\n"
