"""Availability features from *labelled* absences: the NBA injury reports' reason text (ADR 0033).

The games-missed proxy cannot say why a game was missed. The official injury report can (``Injury/Illness - Right Knee; Sprain``
vs ``Injury/Illness - Left Ankle; Injury Management`` vs ``G League - Two-Way`` vs ``Personal Reasons``), and because it is
published before each game it also gives a spell's length in games, which the report itself never states.

``classify_reason`` buckets one report line into a *kind* (injury, illness, recovery, rest, g_league, personal, suspension,
covid, not_with_team, other) and a *body part*. ``build_label_panel`` turns the table into one row per (player, season) over
the team games that have a report:

    inj_out      share of covered team games the player was listed Out for injury, illness or recovery
    rest_out     share listed Out for rest / injury management (the load-management label ADR 0024 lacked)
    other_out    share listed Out for G League, personal, suspension, covid protocol or not-with-team reasons
    longest      longest run of consecutive covered team games listed Out for injury/illness/recovery, as a share (duration)
    n_spells     number of such runs (frequency)

``InjuryLabelFeatures.build`` returns a ``[covered, inj_out, rest_out, other_out, longest, n_spells]`` matrix for
``AvailabilityModel`` / ``AppearanceModel``: recency-weighted over the last ``n_lags`` seasons *that have reports*; a season
without reports (before 2018-19, or a gap) contributes nothing and a row with none of them is all zeros with ``covered = 0``,
so those rows still train the base model. Only seasons strictly before the target are read (``History.until`` slices the
table like every other extra; ``assert_labels_no_future`` checks).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.contracts import season_start
from src.models.panel import lag_matrix, season_starts

KINDS = ("injury", "illness", "recovery", "rest", "g_league", "personal", "suspension", "covid", "not_with_team", "other")
INJURY_KINDS = ("injury", "illness", "recovery")
REST_KINDS = ("rest",)
OTHER_KINDS = ("g_league", "personal", "suspension", "covid", "not_with_team", "other")
LABEL_COLS = ("inj_out", "rest_out", "other_out", "longest", "n_spells")
MIN_SEASON_COVERAGE = 0.4          # a season counts as covered when reports exist for at least this share of its team games

_ILLNESS = re.compile(r"illness|flu\b|virus|infection|covid|sick|food poisoning|strep|pneumonia|dehydrat|bronch", re.I)
_REST = re.compile(r"\brest\b|injury management|load management|maintenance|managed", re.I)
_RECOVERY = re.compile(r"recovery|rehab|reconditioning|return to competition|conditioning|post.?surgical|surgery", re.I)
_PARTS = (
    ("foot_ankle", re.compile(r"ankle|foot|toe|achilles|heel|plantar|navicular|metatars|arch", re.I)),
    ("knee", re.compile(r"knee|acl|mcl|lcl|pcl|menisc|patell", re.I)),
    ("leg_hip", re.compile(r"hamstring|quad|thigh|calf|groin|hip|glute|adductor|\bleg\b|shin|tibia|fibula|femur|abductor|pelvi|soleus|gastroc|tendinitis.*leg", re.I)),
    ("back", re.compile(r"\bback\b|spine|spinal|lumbar|sacr|rib|oblique|abdom|core|trunk|flank", re.I)),
    ("upper_limb", re.compile(r"shoulder|elbow|wrist|hand|finger|thumb|\barm\b|bicep|tricep|forearm|metacarpal|pinky|pectoral|clavicle", re.I)),
    ("head_neck", re.compile(r"concussion|head|neck|eye|face|nose|facial|jaw|orbital|ear\b|dental|tooth|cervical", re.I)),
)


def classify_reason(category: str | None, detail: str | None) -> tuple[str, str]:
    """``(kind, body_part)`` of one report line. ``body_part`` is '' unless the kind is an injury-like one."""
    cat = (category or "").strip().lower()
    det = (detail or "").strip()
    text = f"{cat} {det}".lower()
    if cat.startswith("g league") or "two-way" in text or "on assignment" in text:
        return "g_league", ""
    if "suspen" in text:
        return "suspension", ""
    if "health and safety" in text or "covid-19" in text:
        return "covid", ""
    if "personal" in text or "bereavement" in text or "family" in text or "paternity" in text or "birth" in text:
        return "personal", ""
    if "not with team" in text:
        return "not_with_team", ""
    if cat in ("rest", "injury management", "load management") or (_REST.search(det) and not _RECOVERY.search(det)):
        return "rest", ""
    if cat.startswith("injury") or cat.startswith("illness") or cat == "" and det:
        if _ILLNESS.search(det) and not any(p.search(det) for _, p in _PARTS):
            return "illness", ""
        kind = "recovery" if _RECOVERY.search(det) else "injury"
        for name, pat in _PARTS:
            if pat.search(det):
                return kind, name
        return kind, "other"
    if "return to competition" in text or "reconditioning" in text or "rehab" in text:
        return "recovery", ""
    return "other", ""


def assert_labels_no_future(reports: pd.DataFrame, target_season: str) -> None:
    if len(reports) and (reports["season"].map(season_start) >= season_start(target_season)).any():
        raise AssertionError(f"leakage: injury reports contain seasons >= {target_season}")


def _runs(flags: np.ndarray) -> tuple[int, int]:
    """(longest run of True, number of runs)."""
    if not flags.any():
        return 0, 0
    padded = np.concatenate(([False], flags, [False])).astype(np.int8)
    d = np.diff(padded)
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return int((ends - starts).max()), int(len(starts))


def build_label_panel(reports: pd.DataFrame, team_games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(panel, coverage)``.

    ``coverage``: one row per season with ``s`` and ``share`` = covered team games / all team games (covered = the game date
    has a fetched report). ``panel``: one row per (player_id, s) that was ever listed Out, with ``LABEL_COLS``.
    """
    cols = ["player_id", "s", *LABEL_COLS]
    if reports.empty or team_games.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=["s", "share"])
    tg = team_games[["season", "team_abbr", "game_date"]].copy()
    tg["s"] = season_starts(tg["season"])
    tg["date"] = pd.to_datetime(tg["game_date"]).dt.normalize()
    rep_dates = reports[["game_date"]].copy()
    rep_dates["date"] = pd.to_datetime(rep_dates["game_date"]).dt.normalize()
    have = set(rep_dates["date"])
    tg["covered"] = tg["date"].isin(have)
    cov = tg.groupby("s")["covered"].mean().rename("share").reset_index()

    r = reports[reports["player_id"].notna() & (reports["status"] == "Out")].copy()
    if r.empty:
        return pd.DataFrame(columns=cols), cov
    r["player_id"] = r["player_id"].astype("int64")
    r["s"] = season_starts(r["season"])
    r["date"] = pd.to_datetime(r["game_date"]).dt.normalize()
    kinds = [classify_reason(c, d)[0] for c, d in zip(r["category"], r["detail"])]
    r["kind"] = kinds
    # the player's team that season: the team on most of his Out rows (a traded player's later rows use the later team's schedule)
    covered_games = tg[tg["covered"]]
    sched = {k: g.sort_values("date")["date"].to_numpy() for k, g in covered_games.groupby(["team_abbr", "s"])}
    rows = []
    for (pid, s), g in r.groupby(["player_id", "s"]):
        team = g["team_abbr"].mode()
        dates = sched.get((team.iloc[0] if len(team) else None, s))
        if dates is None or not len(dates):
            continue
        by_date = g.drop_duplicates("date").set_index("date")["kind"]
        kind = pd.Series(by_date.reindex(dates).to_numpy(), index=dates)
        n = len(dates)
        inj = kind.isin(INJURY_KINDS).to_numpy()
        longest, spells = _runs(inj)
        rows.append((int(pid), int(s), inj.sum() / n, kind.isin(REST_KINDS).sum() / n, kind.isin(OTHER_KINDS).sum() / n,
                     longest / n, spells))
    return pd.DataFrame(rows, columns=cols), cov


def _recency(lags: np.ndarray, decay: float) -> np.ndarray:
    present = np.isfinite(lags)
    v0 = np.where(present, lags, 0.0)
    w = decay ** np.arange(lags.shape[1]) * present
    tot = w.sum(axis=1)
    return np.where(tot > 0, (w * v0).sum(axis=1) / np.maximum(tot, 1e-12), np.nan)


@dataclass
class InjuryLabelFeatures:
    panel: pd.DataFrame                 # player_id, s, LABEL_COLS
    covered_seasons: frozenset          # season starts with enough reports to trust an absent row as "healthy"
    n_lags: int
    decay: float
    n_cols: int = field(default=6, init=False)

    @classmethod
    def from_reports(cls, reports: pd.DataFrame, team_games: pd.DataFrame, target_season: str, *, n_lags: int,
                     decay: float) -> "InjuryLabelFeatures":
        assert_labels_no_future(reports, target_season)
        panel, cov = build_label_panel(reports, team_games)
        covered = frozenset(int(s) for s, sh in zip(cov["s"], cov["share"]) if sh >= MIN_SEASON_COVERAGE)
        return cls(panel, covered, n_lags, decay)

    def build(self, pids, target_s, age=None) -> np.ndarray:
        """``(n, 6)``: ``[covered, inj_out, rest_out, other_out, longest, n_spells]`` (``age`` accepted for the hook signature)."""
        pids, ts = np.asarray(pids, "int64"), np.asarray(target_s, "int64")
        n = len(pids)
        out = np.zeros((n, self.n_cols))
        if not self.covered_seasons:
            return out
        pidx = self.panel.set_index(["player_id", "s"]).sort_index() if len(self.panel) else None
        if pidx is not None:
            lags = lag_matrix(pidx, pids, ts, self.n_lags, list(LABEL_COLS))
        else:
            lags = {c: np.full((n, self.n_lags), np.nan) for c in LABEL_COLS}
        cov_mask = np.zeros((n, self.n_lags), bool)
        for k in range(1, self.n_lags + 1):
            cov_mask[:, k - 1] = np.isin(ts - k, list(self.covered_seasons))
        for c in LABEL_COLS:        # a covered season with no row for the player means "never listed Out": zero, not missing
            lags[c] = np.where(cov_mask, np.nan_to_num(lags[c], nan=0.0), np.nan)
        any_cov = cov_mask.any(axis=1)
        out[:, 0] = any_cov
        for j, c in enumerate(LABEL_COLS, start=1):
            out[:, j] = np.nan_to_num(_recency(lags[c], self.decay), nan=0.0)
        return out
