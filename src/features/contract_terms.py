"""Veteran contract-terms features from the dated Wikipedia contract events (ADR 0019).

Input is ``player_contracts`` (``src.ingest.wiki_contracts``): one row per dated contract event, each
tagged with ``season`` = the season the event FOLLOWS, so ``History.until(T)`` keeps exactly the events
dated before 1 Oct of T's start year. Everything below is computed *as of the start of the target season
T* from those events only, plus (for first-rounders) the static rookie-scale clock of ADR 0013. Nothing
after the start of T can enter: the events are sliced again here by tag (defence in depth, so a caller
that passes the unsliced table cannot leak), and the target season's own games are never read.

**Unknown is unknown.** Wikipedia's coverage is partial and biased toward notable signings (ADR 0019
quantifies it). A player with no usable event has ``status == "unknown"`` and every indicator 0; he is
never labelled "not in a contract year". The four statuses:

``known``     an active contract with a stated length: ``years_remaining >= 1`` seasons including T.
``no_length`` a contract event whose length the page does not state (only the signing is known).
``lapsed``    the latest known contract ended before T (expired, or a waive / buyout / retirement
              followed it): he must be on a deal we never saw. Treated as unknown for contract year.
``unknown``   no event at all.

Contract length arithmetic (each line an assumption, labelled): a deal dated 1 Jul or later covers the
season starting that year, one dated Jan-Jun covers the season already under way (``start_year``); a
contract of ``N`` years covers ``start_year .. start_year + N - 1``. An **extension** adds ``N`` seasons
after the running contract's end: the previous known end if there is one, else the season in which it was
signed (the common case: extended in or just before the final year; a year-early extension is
under-counted by a season). Ten-day contracts never set the status. A ``claimed`` event hands the old
contract back after a waive. The rookie-scale clock (ADR 0013) is added only for first-rounders whose
status is not ``known``: a Wikipedia-stated deal always beats the nominal scale.

Design columns (all 0/1, unknown = all zero, so the fitted contrast is "this state vs unknown"):
``cy`` (final year of a known deal), ``yr2``, ``yr3p``, ``ext`` (latest deal is an extension signed last
season or this offseason), ``new_deal`` (a deal signed 1 Jul - 30 Sep of T's start year), ``two_way``,
``minimum`` (latest deal is a two-way / minimum deal still running), ``lapsed``, ``rookie_cy`` (nominal
rookie-scale final year, no known wiki deal), ``rookie_opt`` (nominal option year, likewise).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.contracts import season_start
from src.features.contract import ContractClock, ContractFit, contract_clock, fit_adjustment
from src.ingest.wiki_contracts import CONTRACT_EVENTS, END_EVENTS

TERMS_FEATURE_NAMES = ("cy", "yr2", "yr3p", "ext", "new_deal", "two_way", "minimum", "lapsed", "rookie_cy", "rookie_opt")
MIN_MARKED_ROWS = 40

_NEEDED = ["season", "start_year", "player_id", "event", "years", "signed_date", "is_two_way", "is_ten_day",
           "is_minimum", "is_extension"]


def slice_contracts(contracts: pd.DataFrame | None, target_season: str) -> pd.DataFrame:
    """Rows whose tag is strictly before ``target_season`` (the same rule as ``History.until``)."""
    if contracts is None or len(contracts) == 0:
        return pd.DataFrame(columns=_NEEDED)
    cutoff = season_start(target_season)
    return contracts[contracts["season"].map(season_start) < cutoff].reset_index(drop=True)


def assert_contracts_no_future(contracts: pd.DataFrame, target_season: str) -> None:
    if len(contracts) and (contracts["season"].map(season_start) >= season_start(target_season)).any():
        raise AssertionError(f"leakage: player_contracts contains seasons >= {target_season}")


@dataclass(frozen=True)
class TermsState:
    """One player's contract state at the start of a target season."""

    status: str                      # known | no_length | lapsed | unknown
    years_remaining: float           # seasons left including the target season; NaN unless status == "known"
    new_deal: bool
    extension: bool
    two_way: bool
    minimum: bool
    end_season: float                # start year of the last covered season (NaN when unknown)


_UNKNOWN = TermsState("unknown", np.nan, False, False, False, False, np.nan)


def _order_key(g: pd.DataFrame) -> pd.Series:
    fallback = pd.to_datetime(g["start_year"].astype(int).astype(str) + "-08-01")
    return g["signed_date"].fillna(fallback)


def _resolve(events: pd.DataFrame, s: int) -> TermsState:
    """Walk one player's events (chronological) and return the state at the start of season ``s``."""
    end = None            # start year of the last season covered, None = length unknown
    have_deal = False
    active = False
    cov = None
    ext = two_way = minimum = False
    for e in events.itertuples(index=False):
        ev = e.event
        yrs = None if pd.isna(e.years) else int(e.years)
        if ev in CONTRACT_EVENTS and not bool(e.is_ten_day):
            start = int(e.start_year)
            if ev == "extension" or bool(e.is_extension):
                base = max(end, start) if (active and end is not None) else start
                end = base + yrs if yrs else None
            else:
                end = start + yrs - 1 if yrs else None
            have_deal, active, cov = True, True, start
            ext, two_way, minimum = ev == "extension" or bool(e.is_extension), bool(e.is_two_way), bool(e.is_minimum)
        elif ev in END_EVENTS and have_deal:
            active = False
        elif ev == "claimed" and have_deal:
            active = True
    if not have_deal:
        return _UNKNOWN
    if not active or (end is not None and end < s):
        return TermsState("lapsed", np.nan, False, False, False, False, np.nan if end is None else float(end))
    new = cov == s
    if end is None:
        return TermsState("no_length", np.nan, new, ext and cov >= s - 1, two_way and new, minimum and new, np.nan)
    rem = end - s + 1
    return TermsState("known", float(rem), new, ext and cov >= s - 1,
                      two_way, minimum, float(end))


@dataclass(frozen=True)
class TermsFeatures:
    """Per-player contract-terms features for one target season (arrays share the ``player_id`` order)."""

    player_id: np.ndarray
    status: np.ndarray               # object: known | no_length | lapsed | unknown
    years_remaining: np.ndarray      # float, NaN unless known
    contract_year: np.ndarray        # bool: known, final year
    new_deal: np.ndarray
    extension: np.ndarray
    two_way: np.ndarray
    minimum: np.ndarray
    rookie_cy: np.ndarray
    rookie_opt: np.ndarray

    @property
    def known(self) -> np.ndarray:
        return self.status == "known"

    def design(self) -> np.ndarray:
        rem = self.years_remaining
        known = self.known
        cols = [
            self.contract_year, known & (rem == 2), known & (rem >= 3), self.extension, self.new_deal,
            self.two_way, self.minimum, self.status == "lapsed", self.rookie_cy, self.rookie_opt,
        ]
        return np.column_stack([np.asarray(c, dtype=float) for c in cols])

    def flag(self) -> np.ndarray:
        """Display label per player; '' when nothing is known that is worth showing."""
        out = np.full(len(self.player_id), "", dtype=object)
        out[self.rookie_opt] = "rookie_option_year"
        out[self.new_deal] = "new_deal"
        out[self.extension] = "extension"
        out[self.rookie_cy] = "rookie_contract_year"
        out[self.contract_year] = "contract_year"
        return out

    def basis(self) -> np.ndarray:
        out = np.full(len(self.player_id), "", dtype=object)
        out[self.status != "unknown"] = "wiki"
        out[(self.status != "known") & (self.rookie_cy | self.rookie_opt)] = "rookie_scale"
        return out


def terms_features(contracts: pd.DataFrame | None, target_season: str, player_ids: np.ndarray,
                   clock: ContractClock | None = None) -> TermsFeatures:
    """Features of ``player_ids`` at the start of ``target_season``. ``contracts`` may be the whole table or
    a ``History``-sliced one; rows tagged at or after the target are ignored either way."""
    pids = np.asarray(player_ids, dtype="int64")
    s = season_start(target_season)
    c = slice_contracts(contracts, target_season)
    states: dict[int, TermsState] = {}
    if len(c):
        c = c[c["player_id"].isin(set(pids.tolist()))]
        for pid, g in c.groupby("player_id", sort=False):
            g = g.assign(_k=_order_key(g)).sort_values("_k", kind="stable")
            states[int(pid)] = _resolve(g, s)
    st = [states.get(int(p), _UNKNOWN) for p in pids]
    status = np.array([x.status for x in st], dtype=object)
    rem = np.array([x.years_remaining for x in st], dtype=float)
    known = status == "known"
    if clock is not None:
        rcy = np.asarray(clock.is_contract_year, bool) & ~known
        ropt = np.asarray(clock.is_option_year, bool) & ~known
    else:
        rcy = ropt = np.zeros(len(pids), dtype=bool)
    return TermsFeatures(
        player_id=pids, status=status, years_remaining=rem,
        contract_year=known & (rem == 1),
        new_deal=np.array([x.new_deal and x.status != "lapsed" for x in st], dtype=bool),
        extension=np.array([x.extension for x in st], dtype=bool),
        two_way=np.array([x.two_way for x in st], dtype=bool),
        minimum=np.array([x.minimum for x in st], dtype=bool),
        rookie_cy=rcy, rookie_opt=ropt)


# --------------------------------------------------------------------------- fit

@dataclass
class TermsTrainingSet:
    design: np.ndarray
    resid: np.ndarray
    weight: np.ndarray
    season: np.ndarray
    known: np.ndarray                # per row: status == known (for coverage diagnostics)


MIN_TRAIN_GP = 10


def build_terms_training_set(players: pd.DataFrame, contracts: pd.DataFrame | None, seasons: list[str],
                             project: Callable[[str], pd.DataFrame | None], actual: pd.DataFrame) -> TermsTrainingSet | None:
    """Stack the walk-forward residuals of ``seasons`` against the as-of-season-start terms features.

    ``project(s)`` is the base projection of ``s`` from data before ``s``; the terms of a historical row
    use only the events tagged before ``s``, exactly the information available at the start of ``s``.
    """
    Xs, ys, ws, ss, ks = [], [], [], [], []
    for s in seasons:
        sy = season_start(s)
        proj = project(s)
        if proj is None or proj.empty:
            continue
        act = actual[(actual["s"] == sy) & (actual["gp"] >= MIN_TRAIN_GP)][["player_id", "gp", "fppg"]]
        m = proj[["player_id", "proj_fppg"]].merge(act, on="player_id", how="inner")
        if m.empty:
            continue
        pids = m["player_id"].to_numpy("int64")
        tf = terms_features(contracts, s, pids, contract_clock(players, s, pids))
        Xs.append(tf.design())
        ys.append((m["fppg"] - m["proj_fppg"]).to_numpy(float))
        ws.append(m["gp"].to_numpy(float))
        ss.append(np.full(len(m), sy))
        ks.append(tf.known)
    if not Xs:
        return None
    return TermsTrainingSet(np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), np.concatenate(ss), np.concatenate(ks))


def fit_terms_adjustment(ts: TermsTrainingSet, *, min_rows: int | None = None, min_gain: float | None = None) -> ContractFit:
    """Same ridge + leave-one-season-out gate as ADR 0013, over :data:`TERMS_FEATURE_NAMES`."""
    kw = {} if min_rows is None else {"min_rows": min_rows}
    if min_gain is not None:
        kw["min_gain"] = min_gain
    return fit_adjustment(ts.design, ts.resid, ts.weight, ts.season, names=TERMS_FEATURE_NAMES, **kw)


# --------------------------------------------------------------------------- coverage (reporting)

def coverage_by_season(contracts: pd.DataFrame, game_logs: pd.DataFrame, seasons: list[str], min_gp: int = 10) -> pd.DataFrame:
    """For each target season, the veterans it will score (>= ``min_gp`` games in the season and at least one
    earlier season played) and how many have each terms status at the start of the season."""
    gl = game_logs.assign(s=game_logs["season"].map(season_start))
    gp = gl.groupby(["player_id", "s"]).size().rename("gp").reset_index()
    first = gp.groupby("player_id")["s"].min().rename("first_s")
    rows = []
    for season in seasons:
        s = season_start(season)
        cur = gp[(gp["s"] == s) & (gp["gp"] >= min_gp)].merge(first, on="player_id")
        vets = cur[cur["first_s"] < s]["player_id"].to_numpy("int64")
        tf = terms_features(contracts, season, vets)
        rows.append({
            "season": season, "veterans": len(vets),
            "known": int((tf.status == "known").sum()), "no_length": int((tf.status == "no_length").sum()),
            "lapsed": int((tf.status == "lapsed").sum()), "unknown": int((tf.status == "unknown").sum()),
            "any_event": int((tf.status != "unknown").sum()),
            "contract_year": int(tf.contract_year.sum()), "new_deal": int(tf.new_deal.sum()),
            "extension": int(tf.extension.sum()), "two_way": int(tf.two_way.sum()), "minimum": int(tf.minimum.sum()),
        })
    out = pd.DataFrame(rows)
    out["known_pct"] = out["known"] / out["veterans"]
    out["any_event_pct"] = out["any_event"] / out["veterans"]
    return out
