"""Single source of truth for what every board/table column means, for the app's hover tooltips
and its Glossary tab.

No Streamlit import here on purpose — this is a plain dict, unit-tested without a running app
(``tests/app/test_glossary.py``). Every fact below is derived from ``docs/categories.md`` (the
authoritative, code-verified reference for every board category, formula and constant), not
re-derived from scratch: each entry cites the ``categories.md`` section it comes from, and the
ADR/module behind it where ``categories.md`` names one. Where a column has no ``categories.md``
coverage, the entry says so and cites the code directly.

Where ``categories.md`` section 17 flags an unresolved doc/code discrepancy, the entry here
reflects that honestly instead of picking a side (see ``RETURN_TAIL`` below, discrepancy 4).
"""
from __future__ import annotations

from dataclasses import dataclass

CATEGORIES = "docs/categories.md"


@dataclass(frozen=True)
class Entry:
    """One glossary entry. ``short`` is the hover tooltip (1-2 sentences); ``long`` is the fuller
    explanation shown in the Glossary tab; ``formula`` (optional) is the formula in words or
    symbols; ``source`` cites the doc section and/or ADR this was derived from."""

    short: str
    long: str
    source: str
    formula: str | None = None


def _e(short: str, long: str, source: str, formula: str | None = None) -> Entry:
    return Entry(short=short, long=long, source=source, formula=formula)


# --------------------------------------------------------------------------------------------
# Identity and projection columns (categories.md section 4, section 20 "Board: identity and
# projection")
# --------------------------------------------------------------------------------------------

_IDENTITY_PROJECTION: dict[str, Entry] = {
    "rank": _e(
        "Position in the VORP ordering, 1 = best. Sorted by vorp descending, ties broken by "
        "proj_total_fp then player_id (never by FPPG alone).",
        "Every projected player gets a rank. Ties are broken by proj_total_fp descending, then "
        "player_id ascending, so the order is fully deterministic.",
        f"{CATEGORIES}#8-value-replacement-level-vorp-rank",
        formula="sort by vorp desc, then proj_total_fp desc, then player_id asc",
    ),
    "player_id": _e(
        "The NBA's own player id, used to join every table in the app.",
        "Internal join key from the NBA stats data; not meaningful on its own beyond identifying "
        "the player uniquely.",
        f"{CATEGORIES}#3-positions-and-eligibility",
    ),
    "name": _e("Player name.", "Player name, as carried through from the projection.", CATEGORIES),
    "position": _e(
        "Coarse dataset position (G, F, C, G-F, F-C, ...), not ESPN's per-position eligibility.",
        "One coarse position string per player. It is mapped to ESPN slots (eligible_positions, "
        "eligible_slots) elsewhere in the app; the mapping is an approximation, not ESPN's real "
        "games-played eligibility.",
        f"{CATEGORIES}#3-positions-and-eligibility",
    ),
    "age": _e(
        "Player age on 1 October of the season start.",
        "Used as a modelling covariate (age curve, rookie prior) as well as for display.",
        f"{CATEGORIES}#3-positions-and-eligibility",
    ),
    "proj_fppg": _e(
        "Projected fantasy points per game played (a mean), from the league's own scoring "
        "weights.",
        "The per-minute rate model's stat block, scored by the league's fantasy-point weights "
        "(config/league.yaml). This is what proj_total_fp and the floor/median/ceiling band are "
        "built from.",
        f"{CATEGORIES}#4-projection-columns",
        formula="FP = 1*PTS + 1*REB + 2*AST + 4*STL + 4*BLK - 2*TO + 2*FGM - 1*FGA + 1*FTM - 1*FTA + 1*3PM",
    ),
    "proj_gp": _e(
        "Expected games played: the mean of the player's availability distribution, not a "
        "guarantee.",
        "A fractional-logistic model of the share of the schedule a player is expected to play, "
        "given recent games-played history, last season's share, worst recent season, age and "
        "minutes. Conditional on the player being active in the target season (a retirement or "
        "season-ending injury before opening night is not reflected here — see the risk overlay).",
        f"{CATEGORIES}#42-games-played-proj_gp",
        formula="proj_gp = mu * L  (mu = logistic(...), L = season length)",
    ),
    "proj_total_fp": _e(
        "Projected total fantasy points for the season = proj_fppg x proj_gp. This is what rank "
        "and VORP are based on, not FPPG alone.",
        "The rank and VORP are built on total production, not rate: a high-FPPG player who is "
        "expected to miss a lot of games can rank below a lower-FPPG player who plays a full "
        "season.",
        f"{CATEGORIES}#43-total-proj_total_fp",
        formula="proj_total_fp = proj_fppg * proj_gp",
    ),
    "vorp": _e(
        "Value over replacement player: proj_total_fp minus the replacement player's total. What "
        "you actually draft on.",
        "The replacement total is the (R+1)-th best player's proj_total_fp, where R is the "
        "rostered-pool size (teams x starters, plus a bench share that behaves like starters when "
        "the league's own starters are often hurt or resting). Negative values (below replacement) "
        "are kept, not clipped.",
        f"{CATEGORIES}#8-value-replacement-level-vorp-rank",
        formula="vorp = proj_total_fp - repl_total",
    ),
    "vorp_per_game": _e(
        "Rate value: proj_fppg minus the replacement player's FPPG. Ignores health; always "
        "league-wide.",
        "Uses the FPPG of the same replacement player used for vorp's total (interpolated the same "
        "way). Unlike vorp, it is a per-game rate and says nothing about how many games a player "
        "is expected to play.",
        f"{CATEGORIES}#8-value-replacement-level-vorp-rank",
        formula="vorp_per_game = proj_fppg - repl.per_game",
    ),
    "fppg_p10": _e(
        "10th percentile of a single game's fantasy points (not of a season average).",
        "The floor of the per-game distribution, built from the projection's own dispersion model. "
        "Because the residual is taken around the projection (not the realised season mean), the "
        "band includes projection error as well as game-to-game noise.",
        f"{CATEGORIES}#5-the-floor--median--ceiling-band",
        formula="p10 = proj_fppg + q10 * sd",
    ),
    "fppg_p50": _e(
        "Median of a single game's fantasy points. Usually below proj_fppg because game scores "
        "are right-skewed.",
        "The per-game median; sits below the mean (proj_fppg) because a player's individual games "
        "are right-skewed (a few huge games pull the mean up more than the median).",
        f"{CATEGORIES}#5-the-floor--median--ceiling-band",
        formula="p50 = proj_fppg + q50 * sd",
    ),
    "fppg_p90": _e(
        "90th percentile of a single game's fantasy points (a ceiling game, not a ceiling "
        "season).",
        "The top of the per-game distribution. A wide p10..p90 band marks a player whose individual "
        "games swing a lot, which matters more for H2H weekly matchups than for the season total.",
        f"{CATEGORIES}#5-the-floor--median--ceiling-band",
        formula="p90 = proj_fppg + q90 * sd",
    ),
    "tier": _e(
        "Value-cliff group: 1 is best, and tiers never decrease with rank. The last tier is every "
        "player at or below replacement.",
        "Players are grouped by cliffs in VORP (a big drop relative to the local gap size), not by "
        "rank buckets, so tier width varies. At most max_tiers (default 12) tiers above "
        "replacement, plus one final 'replaceable' tier for vorp <= 0 on top of that cap.",
        f"{CATEGORIES}#9-tiers",
    ),
    "is_rookie": _e(
        "True only for the current draft class (not stashes or undrafted debutants from earlier "
        "years).",
        "Marks players drafted in the target year with no NBA game history. Stashes and undrafted "
        "signees are separate projection_class values, not is_rookie.",
        f"{CATEGORIES}#6-who-gets-which-projection",
    ),
    "confidence": _e(
        "low / medium / high, from how much real shot-attempt history backs the projection.",
        "Set from the FGA data weight (how much the empirical-Bayes shrinkage trusts the player's "
        "own history over the role-appropriate prior). Rookies and debutants always show low.",
        f"{CATEGORIES}#6-who-gets-which-projection",
        formula="low: w < 0.5, medium: w < 0.8, high: w >= 0.8 (w = n_eff / (n_eff + kappa_fga))",
    ),
    "projection_class": _e(
        "veteran / rookie / stash / undrafted: how the player is being projected.",
        "veteran = own history; rookie = this year's draft-slot prior; stash = an earlier-year "
        "pick now arriving (draft-slot prior at a discounted pick); undrafted = never drafted, in "
        "the NBA picture now (slot prior at pick 61). Rookie and debutant lines are priors, not "
        "evidence of who the player is.",
        f"{CATEGORIES}#6-who-gets-which-projection",
    ),
    "p_play": _e(
        "Chance a debutant (stash/undrafted) plays at all this season. Stashes use a fixed 0.90.",
        "For a stash, a constant STASH_P_PLAY = 0.90 (history cannot identify it: every historical "
        "stash that could be recognised played). For an undrafted signee, a small logistic fit on "
        "preseason games share, preseason minutes and Summer League minutes.",
        f"{CATEGORIES}#6-who-gets-which-projection",
    ),
}

# --------------------------------------------------------------------------------------------
# Market and overlay columns (categories.md sections 10-13, section 20 "Board: market and
# overlays")
# --------------------------------------------------------------------------------------------

_VALUE_MARKET: dict[str, Entry] = {
    "adp": _e(
        "ESPN's average draft position (overall pick number; lower = drafted earlier). Optional: "
        "blank without an ingested ADP file.",
        "From the ESPN players feed, mapped to player_id (earliest ADP wins on a duplicate). Only "
        "ingests the top ~600 by ownership, so the tail (near the 140.0 'no ADP' sentinel) is not "
        "comparable to rank.",
        f"{CATEGORIES}#10-adp-and-adp_gap",
    ),
    "adp_gap": _e(
        "adp - rank. Positive = the model ranks him earlier than the market drafts him (a value "
        "pick); negative = the market pays more than the model does.",
        "Read it only where both adp and rank are inside the first ~140 picks; rank is a "
        "league-wide VORP order over all players while adp is an average pick number, so the "
        "two scales are not directly interchangeable outside that range.",
        f"{CATEGORIES}#10-adp-and-adp_gap",
        formula="adp_gap = adp - rank",
    ),
}

_RISK_OVERLAY: dict[str, Entry] = {
    "risk_level": _e(
        "'', watch, or high. Advisory only — never changes proj_gp, proj_total_fp, vorp or rank.",
        "The highest level triggered by any risk source (ESPN injury status, preseason absence, "
        "team/star context). Shown only in the live season window (August of the season's start "
        "year through the end of that year) and only on a real (non-synthetic) board.",
        "docs/adr/0016-risk-overlay.md",
    ),
    "risk_flags": _e(
        "Readable text for every triggered risk source (ESPN status, preseason absence, new team, "
        "star arrived/left; the return-flag sentence is appended here too).",
        "Free text built from whichever risk sources triggered. It never changes a projection; "
        "it's shown beside proj_gp so you can judge for yourself.",
        f"{CATEGORIES}#11-risk-overlay",
    ),
    "risk_gp_haircut": _e(
        "Advisory share of games at risk, 0 to 0.20 (the max across triggered sources, never a "
        "sum). Assumption, not a measured estimate.",
        "ESPN keeps no status history and injury-report PDFs don't exist before opening week, so "
        "the haircut sizes (OUT 0.20, DAY_TO_DAY 0.03, SUSPENDED 0.10) are stated assumptions in "
        "ADR 0016 D4, not fitted values. The preseason-absence flag was measured but could not be "
        "separated from 'not under contract', so it warns without a haircut (0.0).",
        f"{CATEGORIES}#11-risk-overlay",
    ),
    "risk_gp": _e(
        "proj_gp after the advisory haircut. Display only — proj_gp itself is never modified.",
        "A what-if number for the draft-day risk overlay; the board's actual proj_gp, proj_total_fp "
        "and vorp are computed without it.",
        f"{CATEGORIES}#11-risk-overlay",
        formula="risk_gp = proj_gp * (1 - risk_gp_haircut)",
    ),
}

_RETURN_OVERLAY: dict[str, Entry] = {
    "return_flag": _e(
        "'returned-healthy' or blank: missed the first 25%+ of last season's team games, then "
        "played 75%+ of at least 15 remaining games. Advisory, not a projection.",
        "Cohort membership only (ADR 0023, ADR 0021's primary cohort): a veteran on one team last "
        "season, mpg >= 15, whose absence and recovery pattern matches. A player not on this list "
        "may still have missed time.",
        f"{CATEGORIES}#12-returned-healthy-flag",
    ),
    "return_block_pct": _e(
        "The missed lead block as a percent of last season's team games.",
        "How much of last season's start the player missed before returning, as a percentage of "
        "his team's games.",
        f"{CATEGORIES}#12-returned-healthy-flag",
        formula="return_block_pct = 100 * first_idx / L",
    ),
    "return_tail": _e(
        "Games played / team games since his first appearance last season (e.g. '16/20').",
        # ADR/docs discrepancy (categories.md section 17 item 4): flagged, not resolved, so this
        # entry states both the code condition and the gap rather than picking a side.
        "Requires he then played at least 75% of a remaining block of >= 15 team games "
        "(min_tail = 15 in the code and ADR 0023). Note: categories.md flags that the in-app "
        "checkbox help text for this cohort omits that 15-remaining-games condition — an "
        "unresolved documentation gap, not a code bug (see categories.md section 17 item 4).",
        f"{CATEGORIES}#17-code-versus-documentation-discrepancies-found-while-writing-this",
        formula="return_tail = gp / (L - first_idx), shown as 'gp/(L-first_idx)'",
    ),
    "return_gp_upside_adv": _e(
        "Advisory 0 or 5 extra games. A fixed judgement call, not a projection change: proj_gp is "
        "untouched.",
        "5 if the missed lead block was >= 60% of team games, else 0 (capped so proj_gp + upside "
        "never exceeds the season length). ADR 0021's pre-registered study found no significant "
        "absolute under-projection for the full cohort; the >= 60% block was the one cell with a "
        "raw hint, so the +5 is 'sized by judgement, not derived'.",
        f"{CATEGORIES}#12-returned-healthy-flag",
    ),
    "return_fp_upside_adv": _e(
        "The advisory games (return_gp_upside_adv) valued in total fantasy points.",
        "A display convenience: the same advisory games expressed in the board's own scoring, "
        "using this player's proj_fppg.",
        f"{CATEGORIES}#12-returned-healthy-flag",
        formula="return_fp_upside_adv = return_gp_upside_adv * proj_fppg",
    ),
    "return_validation": _e(
        "Constant label: advisory_judgement_not_projection. Present so a CSV consumer without the "
        "app's captions still sees the caveat.",
        "Always the same string on every row; a machine-readable reminder that this overlay never "
        "changes a projection or rank.",
        f"{CATEGORIES}#12-returned-healthy-flag",
    ),
}

_LOAD_MANAGEMENT_OVERLAY: dict[str, Entry] = {
    "lm_flag": _e(
        "'short-absences' or blank: 6+ games missed last season in isolated 1-2 game absences "
        "(cause unknown: rest or minor injury). Advisory, not a projection.",
        "A veteran on a single team, >= 20 mpg, >= 41 games the season before, with 6 or more "
        "games missed in interior runs of 1-2 consecutive games. No data labels why a game was "
        "missed, so this can only describe the shape of last season's absences.",
        f"{CATEGORIES}#18-load-management-where-it-fits",
    ),
    "lm_iso_n": _e(
        "Count of last season's isolated (1-2 game) absences behind lm_flag. NaN outside the "
        "study's universe.",
        "The primary proxy measured in ADR 0024's walk-forward study: more short absences went "
        "with a lower season than projected (-1.4 GP per +1 SD), the opposite of a 'rested star "
        "bounces back' story.",
        f"{CATEGORIES}#18-load-management-where-it-fits",
    ),
    "lm_rest_n": _e(
        "Of lm_iso_n, how many were single-game absences on the second night of a back-to-back. "
        "Information only.",
        "The rest-specific proxy on its own showed no measurable effect in the study "
        "(-3 total FP [-36, +30]), so it is shown for context but is not part of the advisory "
        "haircut's trigger.",
        f"{CATEGORIES}#18-load-management-where-it-fits",
    ),
    "lm_gp_risk_adv": _e(
        "Advisory 0 or -2 games. A fixed judgement call, not a projection change: proj_gp is "
        "untouched.",
        "-2 games when lm_flag is set. Post-hoc association was about -2.6 GP [-5.1, -0.2], but "
        "correcting proj_gp with the proxy did not improve out-of-sample accuracy, so the verdict "
        "is a fixed advisory rule, not a model input.",
        f"{CATEGORIES}#18-load-management-where-it-fits",
    ),
    "lm_fp_risk_adv": _e(
        "The advisory games (lm_gp_risk_adv) valued in total fantasy points.",
        "A display convenience: the advisory -2 games expressed in the board's own scoring, using "
        "this player's proj_fppg.",
        f"{CATEGORIES}#18-load-management-where-it-fits",
        formula="lm_fp_risk_adv = lm_gp_risk_adv * proj_fppg",
    ),
    "lm_validation": _e(
        "Constant label: advisory_descriptive_not_projection.",
        "Always the same string on every row; a machine-readable reminder that this overlay never "
        "changes a projection or rank.",
        f"{CATEGORIES}#18-load-management-where-it-fits",
    ),
}

_CONTRACT_OVERLAY: dict[str, Entry] = {
    "contract_flag": _e(
        "contract year / new deal / extension / rookie final year (nominal) / rookie option year "
        "(nominal) / blank. Unvalidated and display only.",
        "A ten-season walk-forward found no reliable lift from contract-year status, so nothing "
        "here changes a projection or rank. A blank means 'not known to be', never 'known not to "
        "be' — Wikipedia coverage is partial and biased toward notable signings.",
        f"{CATEGORIES}#13-contract-flags",
    ),
    "contract_years_left": _e(
        "Seasons left including this one, for known-length deals only (NaN otherwise).",
        "Only populated when contract_status is 'known' (an active deal with a stated length).",
        f"{CATEGORIES}#13-contract-flags",
        formula="contract_years_left = end - season_start + 1",
    ),
    "contract_status": _e(
        "known / no_length / lapsed / unknown.",
        "known = an active deal with a stated length; no_length = a signing whose length wasn't "
        "reported; lapsed = the latest known deal ended or was bought out; unknown = no contract "
        "event found.",
        f"{CATEGORIES}#13-contract-flags",
    ),
    "contract_basis": _e(
        "wiki / rookie_scale / blank.",
        "wiki = derived from a dated Wikipedia contract event; rookie_scale = the nominal "
        "first-round rookie-scale clock (only used when no known deal exists); blank = neither.",
        f"{CATEGORIES}#13-contract-flags",
    ),
    "contract_validation": _e(
        "Constant label: unvalidated_display_only.",
        "Always the same string on every row; a machine-readable reminder that this overlay never "
        "changes a projection or rank.",
        f"{CATEGORIES}#13-contract-flags",
    ),
}

# --------------------------------------------------------------------------------------------
# Draft-state / UI columns (not on the projection board; categories.md section 19's tab table)
# --------------------------------------------------------------------------------------------

_DRAFT_STATE: dict[str, Entry] = {
    "drafted_by": _e(
        "'Me' or the opponent label, blank if still available.",
        "Which side of the draft-state took this player, from the app's own in-session tracking "
        "(not part of the projection).",
        f"{CATEGORIES}#19-tabs-and-pages",
    ),
    "pick_no": _e(
        "The order picks were marked in this session (1, 2, 3, ...).",
        "Assigned when a pick is logged in the app; not the real draft's actual pick number unless "
        "picks were marked in real time.",
        f"{CATEGORIES}#19-tabs-and-pages",
    ),
    "live_sync_status": _e(
        "Whether draft-day auto-detect (ESPN league sync) is connected and when it last polled. "
        "Advisory: \"Mark drafted\" always keeps working, synced or not.",
        "Polls ESPN's read-only draft-detail endpoint for newly filled picks, maps them onto this "
        "board's player_id, and auto-marks them drafted, attributed to \"me\" or \"opponent\" by "
        "ESPN team id. A network/auth error, an un-ingested id map, or another session already "
        "syncing this league all degrade to an error state here rather than stopping the app — "
        "the manual form is the fallback in every case (ADR 0025).",
        "docs/adr/0025-draft-live-sync.md",
    ),
}

# --------------------------------------------------------------------------------------------
# Breakouts tab (categories.md section 14)
# --------------------------------------------------------------------------------------------

_BREAKOUTS: dict[str, Entry] = {
    "watch_rank": _e(
        "Position in the watchlist's current sort order (not the board rank).",
        "Re-numbered whenever the sort option or filters change; see board_rank for the player's "
        "actual draft-board rank.",
        f"{CATEGORIES}#14-breakouts-tab-the-watchlist",
    ),
    "team": _e("Current NBA team (abbreviation).", "Current NBA team, from the roster snapshot.", CATEGORIES),
    "draft_pick": _e(
        "Overall NBA draft pick number for this player (rookies) or his original pick (stashes).",
        "Feeds the draft-slot prior for rookies/stashes; see projection_class.",
        f"{CATEGORIES}#6-who-gets-which-projection",
    ),
    "board_rank": _e(
        "This player's rank on the main draft board (VORP order), for comparison against "
        "watch_rank.",
        "Lets you see where the watchlist's pick sits relative to the full board's own ranking.",
        f"{CATEGORIES}#14-breakouts-tab-the-watchlist",
    ),
    "base_fppg": _e(
        "proj_fppg before the offseason (Summer League/preseason) adjustment layer.",
        "The baseline model's FPPG, i.e. proj_fppg with offseason_adj backed out; compare to "
        "layer_fppg to see the size of the adjustment.",
        f"{CATEGORIES}#7-the-offseason-layer",
    ),
    "layer_fppg": _e(
        "proj_fppg after the offseason (Summer League/preseason) adjustment layer.",
        "The layered model's FPPG; layer_fppg - base_fppg = uplift (offseason_adj).",
        f"{CATEGORIES}#7-the-offseason-layer",
    ),
    "uplift": _e(
        "offseason_adj: how much Summer League/preseason evidence moved this player's projected "
        "FPPG above the baseline.",
        "A ridge-regression adjustment fit on how SL/preseason production explained the baseline's "
        "past misses, gated off unless it measurably improves cross-validated accuracy. The "
        "preseason carries almost all of the real signal; Summer League alone is close to noise.",
        f"{CATEGORIES}#7-the-offseason-layer",
        formula="uplift = layer_fppg - base_fppg (= offseason_adj)",
    ),
    "useful_prob": _e(
        "Calibrated chance this player becomes a 'useful' breakout: a breakout that also finishes "
        "in the rostered pool.",
        "A logistic calibration fit on the historical backtest's out-of-sample scores, using "
        "uplift and is_rookie. The top-ten flagged players became useful about 24% of the time "
        "against a 10% base rate in the ten-season backtest.",
        f"{CATEGORIES}#14-breakouts-tab-the-watchlist",
        formula="P(useful) = sigmoid(intercept + coef_score*uplift + coef_rookie*is_rookie)",
    ),
    "breakout_prob": _e(
        "Calibrated chance this player becomes a breakout: actual FPPG beats the base projection "
        "by 4.0+ FPPG and 25%+, in 20+ games.",
        "The same calibration approach as useful_prob, against the plainer breakout definition "
        "(not gated on finishing in the rostered pool).",
        f"{CATEGORIES}#14-breakouts-tab-the-watchlist",
        formula="P(breakout) = sigmoid(intercept + coef_score*uplift + coef_rookie*is_rookie)",
    ),
    "changed_team": _e(
        "Whether the roster-snapshot team differs from the team he finished last season with.",
        "A descriptive flag; team/roster context was tested as a projection input and found no "
        "lift (ADR 0010, 0011), so it does not affect uplift or the probabilities.",
        f"{CATEGORIES}#15-transactions-and-coaches-tabs",
    ),
    "sl_z": _e(
        "Raw Summer League per-36 production, as a z-score against that event's cohort. No model "
        "involved.",
        "Available as a sort option that ignores the calibrated model entirely, for judging the "
        "raw evidence yourself.",
        f"{CATEGORIES}#7-the-offseason-layer",
    ),
    "pre_z": _e(
        "Raw preseason per-36 production, as a z-score against that event's cohort. No model "
        "involved.",
        "Available as a sort option that ignores the calibrated model entirely. The preseason "
        "carries most of the real breakout signal (rookies AUC 0.66) versus Summer League alone "
        "(AUC 0.52-0.57).",
        f"{CATEGORIES}#7-the-offseason-layer",
    ),
    "evidence": _e(
        "A short readable summary of the games/minutes/production behind the uplift, e.g. "
        "'SL 4g 27mpg z+2.8'.",
        "Text built from the same games/minutes/z-score inputs as the offseason adjustment, for a "
        "quick sanity check without opening the raw columns.",
        f"{CATEGORIES}#14-breakouts-tab-the-watchlist",
    ),
}

# --------------------------------------------------------------------------------------------
# Transactions and Coaches tabs (categories.md section 15)
# --------------------------------------------------------------------------------------------

_TRANSACTIONS: dict[str, Entry] = {
    "txn_date": _e(
        "Date of the transaction, from ESPN's public transactions feed.",
        "The feed is known to be incomplete; 99% of per-player latest moves agree with the NBA "
        "roster snapshot (ADR 0017).",
        f"{CATEGORIES}#15-transactions-and-coaches-tabs",
    ),
    "kind": _e(
        "trade / signed / resigned / extended / converted / claimed / waived (player moves), or "
        "hired / fired / resigned_staff / extended_staff (staff moves).",
        "A trade collapses into one row per team ('from_abbr -> to_abbr'); unparsed feed text is "
        "stored as 'other' and never guessed.",
        f"{CATEGORIES}#15-transactions-and-coaches-tabs",
    ),
    "person": _e("The player or staff member's name.", "Name as parsed from the ESPN feed text.", CATEGORIES),
    "from_abbr": _e(
        "Origin team abbreviation (trades and departures).",
        "Populated for trade and waived rows; blank for a pure arrival.",
        f"{CATEGORIES}#15-transactions-and-coaches-tabs",
    ),
    "to_abbr": _e(
        "Destination team abbreviation (trades and arrivals).",
        "Populated for trade and arrival rows (signed, resigned, extended, converted, claimed); "
        "blank for a pure departure.",
        f"{CATEGORIES}#15-transactions-and-coaches-tabs",
    ),
    "detail": _e(
        "Extra parsed detail, e.g. 'two_way' for a two-way contract signing.",
        "Free-form detail captured alongside the transaction kind.",
        f"{CATEGORIES}#15-transactions-and-coaches-tabs",
    ),
    "context": _e(
        "For relevant rows: the destination team's best-ranked teammates and any positional "
        "crowding, plus who is left behind at the origin team.",
        "Shows the new team's three best-ranked teammates and flags when 3+ of the next 8 best "
        "teammates share this player's position.",
        f"{CATEGORIES}#15-transactions-and-coaches-tabs",
    ),
    "relevant": _e(
        "Whether this move involves a board player inside the top 180 ranks (roughly 13 rounds).",
        "Used to hide low-relevance camp signings/cuts by default; toggle 'Include players outside "
        "the board's top 180' to see everything.",
        f"{CATEGORIES}#15-transactions-and-coaches-tabs",
        formula="relevant = rank <= 180",
    ),
    "team_abbr": _e("Team abbreviation for a staff move.", "The team a coach/front-office move applies to.", CATEGORIES),
    "role": _e(
        "Staff role, e.g. head_coach.",
        "Parsed from the ESPN feed text for hired/fired/resigned_staff/extended_staff rows.",
        f"{CATEGORIES}#15-transactions-and-coaches-tabs",
    ),
}

_COACHES: dict[str, Entry] = {
    "coach": _e("Head coach's name.", "Current head coach for the team-season.", CATEGORIES),
    "since": _e("Date this coach's tenure with the team started.", "From the team_coaches table.", CATEGORIES),
    "new_to_team": _e(
        "True if the season-opening coach differs from the one the team finished last season "
        "with.",
        "Drives which teams show in the 'Only teams with a new head coach' filter and the "
        "affected-players table below it.",
        f"{CATEGORIES}#15-transactions-and-coaches-tabs",
    ),
    "previous_coach": _e(
        "The coach this team finished last season with, if different (marked '(interim)' / "
        "'(acting)' where applicable).",
        "Blank when the head coach is unchanged.",
        f"{CATEGORIES}#15-transactions-and-coaches-tabs",
    ),
    "prior_seasons": _e(
        "How many earlier head-coaching seasons this coach has in the table.",
        "0 for a true first-time head coach, whose 'expected shift' columns are also 0 (no history "
        "to import).",
        f"{CATEGORIES}#15-transactions-and-coaches-tabs",
    ),
    "career_star": _e(
        "This coach's earlier teams' league-relative minutes for their single top minute-getter, "
        "averaged over his prior head-coaching seasons.",
        "One of six style axes (star, top5, depth10, pace, three, young/old) computed per "
        "team-season and stored league-relative (value minus that season's league mean).",
        f"{CATEGORIES}#15-transactions-and-coaches-tabs",
    ),
    "career_top5": _e(
        "This coach's earlier teams' league-relative mean minutes of their five biggest "
        "minute-getters.",
        "Style measurably follows a coach for this axis (+0.43 [+0.02, +0.70] in ADR 0020's data).",
        f"{CATEGORIES}#15-transactions-and-coaches-tabs",
    ),
    "career_depth10": _e(
        "This coach's earlier teams' league-relative rotation depth (players per game with 10+ "
        "minutes).",
        "Style measurably follows a coach for this axis (+0.42 [+0.03, +0.78] in ADR 0020's data).",
        f"{CATEGORIES}#15-transactions-and-coaches-tabs",
    ),
    "career_pace": _e(
        "This coach's earlier teams' league-relative pace (possessions per game).",
        "Displayed for context only: ADR 0020 found pace does NOT measurably follow a coach in "
        "this data.",
        f"{CATEGORIES}#15-transactions-and-coaches-tabs",
        formula="pace = (FGA + 0.44*FTA - OREB + TOV) / games",
    ),
    "career_old": _e(
        "This coach's earlier teams' league-relative share of minutes by players aged 31+.",
        "ADR 0020 found veteran-minutes share does NOT measurably follow a coach in this data.",
        f"{CATEGORIES}#15-transactions-and-coaches-tabs",
    ),
    "career_young": _e(
        "This coach's earlier teams' league-relative share of minutes by players aged 23 or "
        "under.",
        "ADR 0020 found youth-minutes share does NOT measurably follow a coach in this data, so "
        "'develops young players' is not shown as a portable coach trait here.",
        f"{CATEGORIES}#15-transactions-and-coaches-tabs",
    ),
    "first_time_head_coach": _e(
        "True if this coach has no prior head-coaching season in the table.",
        "A first-time head coach gets no expected-shift estimate ('no head-coaching history to "
        "import').",
        f"{CATEGORIES}#15-transactions-and-coaches-tabs",
    ),
}

# --------------------------------------------------------------------------------------------
# Best available by need (ADR 0026)
# --------------------------------------------------------------------------------------------

_NEED_OVERLAY: dict[str, Entry] = {
    "need_score": _e(
        "How open this player's best-fit position is on your own roster, 0 (fully covered, or "
        "your league carries no slot there) to 1 (nothing drafted there yet).",
        "The best (highest) need score among the specific ESPN positions (PG/SG/SF/PF/C) a player "
        "is eligible at; 0 for a player with no parseable position. Each specific position's need "
        "score is how much of its fractional roster capacity (starter slot, plus an even share of "
        "flex G/F, UTIL and bench) is not yet filled by your drafted players. With no players "
        "drafted yet every position scores 1.0, so the bonus below becomes the same constant for "
        "everyone (see need_adj_vorp).",
        "docs/adr/0026-need-adjusted-board.md",
        formula="need_score(pos) = clamp((capacity(pos) - filled(pos)) / capacity(pos), 0, 1); "
                "player's need_score = max over eligible positions",
    ),
    "need_adj_vorp": _e(
        "vorp plus a bonus (up to 40 FP) scaled by need_score. vorp itself is unchanged and still "
        "shown alongside it.",
        "A heuristic nudge for roster construction, not a validated projection input: no backtest "
        "sizes the bonus. With an empty roster it is vorp plus the same constant for every player "
        "(need_score is 1.0 everywhere), so the ranking is unchanged from plain VORP.",
        "docs/adr/0026-need-adjusted-board.md",
        formula="need_adj_vorp = vorp + 40 * need_score",
    ),
    "need_rank": _e(
        "Position in the need_adj_vorp ordering on the Best available by need tab, 1 = best.",
        "Reassigned 1..N over the undrafted board sorted by need_adj_vorp descending (ties by vorp "
        "then player_id); not the same ordering as the main board's rank unless every need_score "
        "is equal.",
        "docs/adr/0026-need-adjusted-board.md",
    ),
    "need_position": _e(
        "One of the five specific ESPN positions (PG/SG/SF/PF/C) — a row label on the positional "
        "need breakdown table, not the board's coarse dataset position column.",
        "The positional-need table has one row per specific ESPN position, distinct from the "
        "board's `position` column (a coarse dataset string like 'G' or 'F-C').",
        "docs/adr/0026-need-adjusted-board.md",
    ),
    "need_capacity": _e(
        "How many roster slots (fractional) your league carries at this specific position, "
        "counting a share of flex/UTIL/bench slots.",
        "Starter slot at that position, plus flex G/F split evenly across its two positions, plus "
        "UTIL and bench split evenly across all five specific positions. From config/league.yaml.",
        "docs/adr/0026-need-adjusted-board.md",
    ),
    "need_filled": _e(
        "How many of your drafted players are eligible at this specific position.",
        "Counts every eligible drafted player at each position he qualifies for (a player counts "
        "toward more than one position when multi-position eligible) — a demand-side "
        "approximation, not an exact lineup assignment.",
        "docs/adr/0026-need-adjusted-board.md",
    ),
}

# --------------------------------------------------------------------------------------------
# Trade analyzer tab (categories.md section 16 "Trade analyzer" / section 20 "Trade analyzer app
# columns", ADR 0027)
# --------------------------------------------------------------------------------------------

_TRADE: dict[str, Entry] = {
    "trade_side": _e(
        "'give' or 'get': which side of the trade this row is.",
        "One row per player named in the trade; give = leaving your roster, get = arriving.",
        f"{CATEGORIES}#20-glossary-of-every-column",
    ),
    "trade_player": _e(
        "Player name on that side of the trade.",
        "Resolved from the rest-of-season frame the same way every other in-season tool names "
        "players.",
        f"{CATEGORIES}#20-glossary-of-every-column",
    ),
    "trade_ros_total_fp": _e(
        "That player's rest-of-season total fantasy points (same quantity as ros_total_fp).",
        "The same rest-of-season projection total used everywhere else in the in-season tools, "
        "shown per traded player so the trade's raw inputs are visible alongside its verdict.",
        f"{CATEGORIES}#16-in-season-page",
    ),
    "trade_verdict": _e(
        "'roughly even' / 'favours you' / 'favours the other side', from delta versus tolerance.",
        "TradeResult.verdict: 'roughly even' when |delta| is inside the tolerance band, otherwise "
        "whichever side gained rest-of-season value.",
        f"{CATEGORIES}#16-in-season-page",
    ),
    "trade_confidence": _e(
        "clear win / lean your way / roughly even / lean other way / clear loss: how far delta "
        "sits outside the tolerance band.",
        "A magnitude read on the same tolerance band verdict already uses, not a new statistic: "
        "inside the band is 'roughly even'; up to 3x the band is a 'lean'; beyond that is a "
        "'clear' win or loss. Computed by the pure, unit-tested confidence_bucket() in "
        "src/inseason/trade.py.",
        f"{CATEGORIES}#16-in-season-page",
        formula="bucket by |delta|/tolerance: <=1 even, <=3 lean, >3 clear; sign gives the direction",
    ),
    "trade_delta": _e(
        "Rest-of-season fantasy points gained (+) or lost (-) by the trade, all roster effects "
        "included.",
        "value(roster after the trade) - value(roster before), where both rosters are first made "
        "a full roster (freed spots filled by the best free agent, an overfull roster drops its "
        "least valuable player) so the number reflects the whole trade, not a phantom empty slot.",
        f"{CATEGORIES}#16-in-season-page",
        formula="trade_delta = after.value - before.value",
    ),
    "trade_tolerance": _e(
        "The +-1.5% of roster value band that trade_verdict and trade_confidence are read "
        "against.",
        "ROS projections are not precise enough to call a small delta a real edge; a trade inside "
        "this band is 'roughly even' rather than a false-precision favours-you/them call.",
        f"{CATEGORIES}#16-in-season-page",
        formula="trade_tolerance = DEFAULT_TOLERANCE (0.015) * before.value",
    ),
    "trade_before_value": _e(
        "Slot-aware rest-of-season roster value before the trade.",
        "The same lineup value used throughout the in-season tools: the best assignment of "
        "players to starting slots plus a weighted bench, before the trade is applied.",
        f"{CATEGORIES}#16-in-season-page",
    ),
    "trade_after_value": _e(
        "Slot-aware rest-of-season roster value after the trade (and after filling/trimming the "
        "roster to a legal size).",
        "Same as trade_before_value, computed on the post-trade roster once it has been made a "
        "full roster again (fit_to_capacity).",
        f"{CATEGORIES}#16-in-season-page",
    ),
    "trade_dropped": _e(
        "Players force-dropped because the trade leaves the roster overfull, or 'none'.",
        "Chosen by marginal lineup value (the player whose removal costs the least), never a "
        "player named in the trade itself.",
        f"{CATEGORIES}#16-in-season-page",
    ),
    "trade_empty_slots": _e(
        "Starting lineup slots left empty after the trade, or 'none'.",
        "A slot with no eligible player assigned to it in the best post-trade assignment (e.g. "
        "trading away your only centre with no other C-eligible player).",
        f"{CATEGORIES}#16-in-season-page",
    ),
    "trade_partner": _e(
        "The other team's label, when scoring their side of the same trade too (mirrors the "
        "CLI's --partner).",
        "Runs evaluate_trade a second time with give/get swapped for the named partner roster "
        "(RosterChoice.others); '(none)' skips it.",
        f"{CATEGORIES}#16-in-season-page",
    ),
}

# --------------------------------------------------------------------------------------------
# External rankings comparison (ADR 0028)
# --------------------------------------------------------------------------------------------

_EXTERNAL_RANKINGS: dict[str, Entry] = {
    "our_rank": _e(
        "This project's own board rank for the player (same as the main board's `rank`).",
        "Carried straight from the board used for the comparison (whatever season/model was loaded "
        "in the sidebar); null when the player is not on that board at all.",
        "docs/adr/0028-external-rankings-compare.md",
    ),
    "on_our_board": _e(
        "Whether this player appears on our own loaded board.",
        "False for a player an external source ranks but our board does not project at all.",
        "docs/adr/0028-external-rankings-compare.md",
    ),
    "yahoo_rank": _e(
        "Yahoo's own 'Rank' column from its draft-analysis export, for players Yahoo could be "
        "matched onto our player_id for.",
        "From the manually-exported Yahoo Fantasy Basketball Draft Analysis workbook ('Players' "
        "sheet). Null when Yahoo does not carry this player or the name could not be resolved (see "
        "the Unmatched names tab).",
        "docs/adr/0028-external-rankings-compare.md",
    ),
    "yahoo_team": _e("Team abbreviation as Yahoo's export shows it.", "Verbatim from Yahoo's 'Team' column.",
                     "docs/adr/0028-external-rankings-compare.md"),
    "yahoo_positions": _e("Eligible positions as Yahoo's export shows them.",
                          "Verbatim comma string from Yahoo's 'Eligible Positions' column.",
                          "docs/adr/0028-external-rankings-compare.md"),
    "yahoo_status_tag": _e(
        "Yahoo's own status marker (P/Q/O/NA/OFS), verbatim.",
        "Yahoo's visible status letters, passed through as-is; no definitions inferred here (see "
        "Yahoo's own 'Source & Notes' sheet for what Yahoo says they mean).",
        "docs/adr/0028-external-rankings-compare.md",
    ),
    "yahoo_adp": _e(
        "Yahoo's average draft position: 'All Drafts ADP' if present, else 'Preseason ADP'.",
        "A real overall-pick-number average, distinct from `ecr_vs_adp` below (FantasyPros does not "
        "export a raw ADP number at all, only a delta) -- see the module docstring in "
        "src/value/rankings_compare.py for why they are two separate columns.",
        "docs/adr/0028-external-rankings-compare.md",
    ),
    "on_yahoo": _e("Whether this player was resolved in the loaded Yahoo export.",
                  "False if Yahoo doesn't carry the player, or the name couldn't be matched onto player_id.",
                  "docs/adr/0028-external-rankings-compare.md"),
    "rank_delta_yahoo": _e(
        "our_rank - yahoo_rank. Positive: Yahoo ranks him earlier (more favorably) than we do. "
        "Negative: we rank him earlier than Yahoo.",
        "The sign convention matches the board's own adp_gap (adp - rank, positive = model likes him "
        "earlier than the market): here positive means the external source likes the player earlier "
        "than our board does.",
        "docs/adr/0028-external-rankings-compare.md",
        formula="rank_delta_yahoo = our_rank - yahoo_rank",
    ),
    "fantasypros_rank": _e(
        "FantasyPros' own consensus rank ('RK' column) from its draft rankings export, for players "
        "resolved onto our player_id.",
        "FantasyPros' CSV only covers its own top ~300ish players; a null here for a player who is "
        "on our board and/or Yahoo's export is expected coverage, not a data error.",
        "docs/adr/0028-external-rankings-compare.md",
    ),
    "fantasypros_team": _e(
        "Team parsed out of FantasyPros' 'PLAYER NAME' field ('FA' for a free agent).",
        "FantasyPros' own 'TEAM' column is always empty in this export; team is instead parsed from "
        "the combined name string, e.g. 'Anthony Davis (WAS - PF,C) OUT'.",
        "docs/adr/0028-external-rankings-compare.md",
    ),
    "fantasypros_positions": _e(
        "Eligible positions parsed out of FantasyPros' 'PLAYER NAME' field.",
        "Comma string parsed from the same combined name field as fantasypros_team.",
        "docs/adr/0028-external-rankings-compare.md",
    ),
    "fantasypros_status_tag": _e(
        "The optional trailing status word parsed out of FantasyPros' 'PLAYER NAME' field (e.g. OUT, "
        "DTD, TWO-WAY, RET, G-League), or blank.",
        "Captured generically (whatever trails the closing parenthesis), not from a fixed list, so a "
        "tag FantasyPros hasn't used before still parses instead of failing.",
        "docs/adr/0028-external-rankings-compare.md",
    ),
    "fantasypros_ecr_vs_adp": _e(
        "FantasyPros' own 'ECR VS. ADP' column: how far their expert consensus rank differs from the "
        "ADP they compared it to. Not the same quantity as yahoo_adp.",
        "A signed delta, not an absolute ADP value; FantasyPros' CSV export has no raw ADP column at "
        "all. The literal '-' FantasyPros shows for 'no ADP data' is stored as a real null here, "
        "never as 0.",
        "docs/adr/0028-external-rankings-compare.md",
    ),
    "on_fantasypros": _e("Whether this player was resolved in the loaded FantasyPros export.",
                        "False if FantasyPros doesn't carry the player (common -- its list is much "
                        "shorter than ours or Yahoo's), or the name couldn't be matched.",
                        "docs/adr/0028-external-rankings-compare.md"),
    "rank_delta_fantasypros": _e(
        "our_rank - fantasypros_rank. Positive: FantasyPros ranks him earlier than we do. Negative: "
        "we rank him earlier than FantasyPros.",
        "Same sign convention as rank_delta_yahoo, for a direct read across both sources.",
        "docs/adr/0028-external-rankings-compare.md",
        formula="rank_delta_fantasypros = our_rank - fantasypros_rank",
    ),
    "max_abs_rank_delta": _e(
        "The larger of |rank_delta_yahoo| and |rank_delta_fantasypros| (whichever sources are "
        "present); used to sort by 'biggest disagreement, either direction'.",
        "Ignores a missing source rather than treating it as zero disagreement; null only when "
        "neither external source resolved this player.",
        "docs/adr/0028-external-rankings-compare.md",
    ),
    "source": _e(
        "Which external source a row in the unmatched-names report came from ('yahoo' or "
        "'fantasypros').",
        "Distinguishes the two sources' unmatched lists when shown together.",
        "docs/adr/0028-external-rankings-compare.md",
    ),
    "source_name_raw": _e(
        "The player name exactly as that external source wrote it, before any parsing or matching.",
        "For FantasyPros this is the whole combined 'Name (TEAM - POS) TAG' string; for Yahoo it is "
        "just the 'Player' cell. Kept verbatim so a failed match can be diagnosed by eye.",
        "docs/adr/0028-external-rankings-compare.md",
    ),
    "ext_team": _e(
        "Team as parsed from the unmatched row's external source (may be 'FA' for FantasyPros).",
        "Same value as yahoo_team/fantasypros_team would have held, shown under one name in the "
        "combined unmatched-names report across both sources.",
        "docs/adr/0028-external-rankings-compare.md",
    ),
    "ext_positions": _e(
        "Eligible positions as parsed from the unmatched row's external source.",
        "Same value as yahoo_positions/fantasypros_positions would have held, in the combined "
        "unmatched-names report.",
        "docs/adr/0028-external-rankings-compare.md",
    ),
    "ext_status_tag": _e(
        "Status marker as parsed from the unmatched row's external source.",
        "Same value as yahoo_status_tag/fantasypros_status_tag would have held, in the combined "
        "unmatched-names report.",
        "docs/adr/0028-external-rankings-compare.md",
    ),
    "ext_rank": _e(
        "That external source's own rank/order number for this (unmatched) player.",
        "Lets you judge how fantasy-relevant an unmatched name is -- a miss at rank 5 is a real "
        "problem; a miss at rank 680 rarely matters.",
        "docs/adr/0028-external-rankings-compare.md",
    ),
    "match_method": _e(
        "How src.ingest.id_map resolved (or failed to resolve) this name: exact / normalized / "
        "fuzzy / ambiguous / unmatched.",
        "'ambiguous' means more than one canonical player matched and neither could be preferred; "
        "'unmatched' means none did. Neither is ever silently guessed into a player_id.",
        "docs/adr/0028-external-rankings-compare.md",
    ),
    "rankings_disagreement": _e(
        "Advisory flag: our board and the external consensus disagree by 50+ rank spots (either "
        "direction).",
        "True when max_abs_rank_delta >= DISAGREEMENT_THRESHOLD (50, grounded in the real 2026-09-28 "
        "snapshot's draft-pool distribution -- see the ADR). Never changes our_rank, vorp, "
        "proj_total_fp or any board projection; a judgement about what counts as 'sharp', not a "
        "fitted number.",
        "docs/adr/0029-rankings-disagreement-flag.md",
    ),
    "rankings_disagreement_direction": _e(
        "'market_favors' (an external source ranks him earlier than we do) or 'we_favor' (we rank "
        "him earlier than that source); blank when not flagged.",
        "Named for whichever of rank_delta_yahoo / rank_delta_fantasypros has the larger magnitude "
        "(ties go to Yahoo, the wider-coverage source).",
        "docs/adr/0029-rankings-disagreement-flag.md",
    ),
    "rankings_disagreement_text": _e(
        "One-line explanation of the flagged disagreement, naming the driving source and both "
        "ranks.",
        "E.g. \"fantasypros ranks him 62 spots earlier (#41) than our board (#103); external "
        "consensus is higher on him than we are\". Blank when not flagged.",
        "docs/adr/0029-rankings-disagreement-flag.md",
    ),
}

# --------------------------------------------------------------------------------------------
# Assemble the registry and groups
# --------------------------------------------------------------------------------------------

#: (group title, columns-in-that-group) in the same order as categories.md section 20, for the
#: app's Glossary tab.
GROUPS: list[tuple[str, dict[str, Entry]]] = [
    ("Identity and projection", _IDENTITY_PROJECTION),
    ("Value and VORP / ADP", _VALUE_MARKET),
    ("Risk overlay (ADR 0016)", _RISK_OVERLAY),
    ("Returned-healthy overlay (ADR 0023)", _RETURN_OVERLAY),
    ("Load-management overlay (ADR 0024)", _LOAD_MANAGEMENT_OVERLAY),
    ("Contract overlay (ADR 0019)", _CONTRACT_OVERLAY),
    ("Best available by need (ADR 0026)", _NEED_OVERLAY),
    ("Draft-state / UI", _DRAFT_STATE),
    ("Breakouts / watchlist (ADR 0012)", _BREAKOUTS),
    ("Transactions (ADR 0017)", _TRANSACTIONS),
    ("Coaches (ADR 0020)", _COACHES),
    ("Trade analyzer (ADR 0027)", _TRADE),
    ("External rankings comparison (ADR 0028)", _EXTERNAL_RANKINGS),
]

#: Every column, name -> Entry, flattened for lookup.
COLUMN_GLOSSARY: dict[str, Entry] = {}
for _group_name, _cols in GROUPS:
    for _col, _entry in _cols.items():
        if _col in COLUMN_GLOSSARY:
            raise ValueError(f"duplicate glossary entry for column {_col!r}")
        COLUMN_GLOSSARY[_col] = _entry
del _group_name, _cols, _col, _entry


def tooltip(column: str) -> str | None:
    """The short hover-tooltip text for ``column``, or ``None`` if it has no glossary entry."""
    entry = COLUMN_GLOSSARY.get(column)
    return entry.short if entry else None


def entry(column: str) -> Entry | None:
    """The full :class:`Entry` for ``column``, or ``None`` if it has no glossary entry."""
    return COLUMN_GLOSSARY.get(column)
