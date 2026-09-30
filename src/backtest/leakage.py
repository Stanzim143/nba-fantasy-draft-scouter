"""Leakage guards.

Three independent layers protect a backtest:

1. **Structural.** ``History.until`` (``src/contracts.py``) slices every season-indexed table to
   seasons < target *and* sanitises the static ``players`` table (see ``_sanitize_players`` there)
   so ``from_year``/``to_year``/``draft_year`` cannot reveal who debuts, retires, or gets drafted
   in the target season or later. ``build_history`` here is now a thin wrapper adding the
   ``extras``-aware ``assert_history_clean`` check on top; it no longer re-applies this module's
   own ``sanitize_players`` (see the fixed-bug note on that function below — it duplicated logic
   that ``History.until`` already does correctly, and the duplicate copy was never exercised
   against real pandas nullable dtypes until it broke the CLI). The ``sanitize=`` flags on
   ``build_history`` and ``assert_projector_ignores_future`` are kept only for backward
   compatibility and are no-ops today: there is no public way to obtain an *unsanitised* History
   any more, by design. (Originally this module discovered and worked around the leak locally,
   before the fix moved into the shared contract — see the History.until docstring and ADR 0001.)
2. **Behavioural.** ``assert_projector_ignores_future`` re-runs a projector after every row of
   season >= target has been scrambled *in place* in the caller's own table objects.  A projector
   that holds a reference to the full tables (or otherwise reaches the future by a side channel)
   changes its output and is flagged.
3. **Statistical (elsewhere).** Implausibly perfect scores are a smell worth investigating;
   the oracle test in tests/backtest shows what "too good" looks like.

Known limits of the behavioural check (documented in docs/backtest.md): it cannot see a
projector that reads data from disk on its own, keeps a *copy* of the tables made before the
check, or captured a single column object rather than the frame.  Those are caught only by code
review; the sanitised ``History`` remains the primary defence.
"""
from __future__ import annotations

import contextlib
import hashlib
from typing import Iterator, Mapping

import numpy as np
import pandas as pd

from src.backtest.errors import LeakageError
from src.contracts import TABLES, History, Projector, season_start

# --------------------------------------------------------------------------- data hash


def data_hash(tables: Mapping[str, pd.DataFrame]) -> str:
    """Order-insensitive content hash of a table dict (sha256 hex).

    Rows are sorted by the table key when the table is a contract table, columns by name.  Stable
    within one pandas/numpy install (recorded alongside the versions in run metadata).
    """
    h = hashlib.sha256()
    for name in sorted(tables):
        df = tables[name]
        d = df[sorted(df.columns)]
        key = list(TABLES[name].key) if name in TABLES else []
        if key and all(k in d.columns for k in key):
            d = d.sort_values(key, kind="stable")
        h.update(name.encode())
        h.update("|".join(f"{c}:{d[c].dtype}" for c in d.columns).encode())
        h.update(str(len(d)).encode())
        if len(d):
            h.update(pd.util.hash_pandas_object(d, index=False).to_numpy().tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- structural guards

def sanitize_players(players: pd.DataFrame, target_season: str) -> pd.DataFrame:
    """Return ``players`` as it could have been known before ``target_season`` began.

    * drop players drafted after the target season's draft (``draft_year > S``);
    * drop undrafted players whose first NBA season is >= S (not knowable point-in-time);
    * ``from_year >= S`` -> NaN; ``to_year`` capped at S-1.
    Never mutates the input.

    Superseded by ``src.contracts._sanitize_players``, which ``History.until`` now applies
    unconditionally and ``build_history`` below relies on exclusively — prefer that. Kept here,
    fixed and still tested, only because other code may still call it directly.

    Bug fixed 2026-09-22: on real ingested data, ``draft_year``/``from_year`` are pandas nullable
    ``Int64`` columns, so ``drop`` above ends up nullable-``boolean`` dtype. Its ``.to_numpy()``
    (with no NAs, since ``|`` with a non-null comparison resolves them) degraded to an **object**
    array of Python ``bool``s, and ``~`` on *that* performs Python's bitwise complement on each
    ``bool``-as-``int`` (``~True == -2``, not ``False``) instead of logical negation --
    ``p[~drop.to_numpy()]`` then tried to select columns named ``-1``/``-2`` and raised ``KeyError``.
    This crashed ``python -m src.backtest`` on every real-data run (it was never triggered by the
    synthetic fixtures used elsewhere in this test suite, whose ``draft_year`` is plain ``int64``
    with no nulls). Fixed by negating while still a pandas boolean Series (whose own ``~``
    operator is a correct logical NOT) and only converting to a plain numpy ``bool`` array, with
    no remaining nulls, at the point of use.
    """
    s = season_start(target_season)
    p = players.copy()
    draft = pd.to_numeric(p["draft_year"], errors="coerce")
    frm = pd.to_numeric(p["from_year"], errors="coerce")
    to = pd.to_numeric(p["to_year"], errors="coerce")
    drop = (draft > s) | (draft.isna() & (frm >= s))
    keep = (~drop).fillna(True).to_numpy(dtype=bool)
    p, frm, to = p[keep].copy(), frm[keep], to[keep]
    p["from_year"] = frm.where(~(frm >= s)).astype("Int64")
    p["to_year"] = to.clip(upper=s - 1).astype("Int64")
    return p.reset_index(drop=True)


def build_history(tables: Mapping[str, pd.DataFrame], season: str, *, sanitize: bool = True) -> History:
    """The one way the harness (and the leakage check) constructs what a projector may see.

    ``sanitize`` is kept only for backward compatibility and no longer does anything:
    ``History.until`` already sanitises ``players`` unconditionally (see its docstring), so this
    used to just redundantly re-run this module's own (now-fixed, but still risk-prone to
    duplicate) ``sanitize_players`` on top of already-clean data. Relying on a single
    implementation removes that duplication entirely.
    """
    h = History.until(tables, season)
    h.assert_no_future()
    assert_history_clean(h, sanitized=True)
    return h


def assert_history_clean(history: History, *, sanitized: bool = True) -> None:
    """Stricter than ``History.assert_no_future``: checks ``extras`` and ``players`` too."""
    history.assert_no_future()
    cutoff = season_start(history.target_season)
    for name, df in history.extras.items():
        if "season" in df.columns and len(df) and (df["season"].map(season_start) >= cutoff).any():
            raise LeakageError(f"leakage: extras[{name!r}] contains seasons >= {history.target_season}")
    if sanitized and len(history.players):
        for col, bad in (("from_year", lambda x: x >= cutoff), ("to_year", lambda x: x >= cutoff),
                         ("draft_year", lambda x: x > cutoff)):
            v = pd.to_numeric(history.players[col], errors="coerce")
            if bad(v).any():
                raise LeakageError(
                    f"leakage: players.{col} reveals seasons >= {history.target_season} "
                    f"(players table was not sanitised)")


# --------------------------------------------------------------------------- perturbation

def _future_mask(df: pd.DataFrame, cutoff: int) -> np.ndarray:
    return (df["season"].map(season_start) >= cutoff).to_numpy()


def _scramble_season_frame(df: pd.DataFrame, cutoff: int, rng: np.random.Generator) -> None:
    """Scramble, in place, every row of ``df`` whose season >= cutoff."""
    m = _future_mask(df, cutoff)
    n = int(m.sum())
    if not n:
        return
    for col in list(df.columns):
        if col == "season":
            continue
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            arr = s.to_numpy().copy()
            arr[m] = rng.random(n) < 0.5
            df[col] = pd.Series(arr, index=df.index, dtype=s.dtype)
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            arr = s.to_numpy(dtype="float64", na_value=np.nan).copy()
            if col.endswith("_id"):
                arr[m] = arr[m][rng.permutation(n)]           # break identity <-> stats links
            else:
                arr[m] = np.round(arr[m] * rng.uniform(0.3, 2.0, n), 2) + rng.integers(1, 5, n)
            if pd.api.types.is_integer_dtype(s):
                arr = np.rint(arr)
            df[col] = pd.Series(arr, index=df.index).astype(s.dtype)


def _scramble_players(df: pd.DataFrame, cutoff: int, rng: np.random.Generator) -> None:
    """Scramble the future-derived parts of the static players table, in place.

    Bug fixed 2026-09-22 (found by a third round of adversarial verification, same class as the
    ``sanitize_players`` fix above): on real data ``draft_year``/``from_year``/``to_year`` are
    pandas nullable ``Int64``, so a boolean expression built from them (e.g. ``ser >= cutoff``)
    is nullable-``boolean`` dtype and can genuinely contain ``pd.NA`` (not just True/False) --
    e.g. a player with unknown ``draft_year`` *and* unknown ``from_year`` makes
    ``draft.isna() & (frm >= cutoff)`` itself ``pd.NA`` (Kleene logic: unknown AND unknown is
    unknown). Converting that straight to numpy left real ``pd.NA`` values sitting in the
    resulting array; summing an object array containing ``pd.NA`` returns ``pd.NA``, and
    ``if pd.NA > 1:`` raises ``TypeError: boolean value of NA is ambiguous``. This crashed
    ``python -m src.backtest ... --leak-check`` on real data every time. Fixed the same way as
    ``sanitize_players``: resolve every remaining null to False (unknown => not scrambled, same
    as the old plain-float64 behaviour these nullable dtypes replaced) before the single
    ``.to_numpy(dtype=bool)`` call.
    """
    frm = pd.to_numeric(df["from_year"], errors="coerce")
    to = pd.to_numeric(df["to_year"], errors="coerce")
    draft = pd.to_numeric(df["draft_year"], errors="coerce")
    for col, ser in (("from_year", frm), ("to_year", to)):
        m = (ser >= cutoff).fillna(False).to_numpy(dtype=bool)
        if m.any():
            arr = ser.to_numpy(dtype="float64", na_value=np.nan).copy()
            arr[m] = cutoff + rng.integers(0, 7, int(m.sum()))
            df[col] = pd.Series(arr, index=df.index).astype(df[col].dtype)
    future = ((draft > cutoff) | (draft.isna() & (frm >= cutoff))).fillna(False).to_numpy(dtype=bool)
    if future.sum() > 1:
        ids = df["player_id"].to_numpy().copy()
        ids[future] = ids[future][rng.permutation(int(future.sum()))]
        df["player_id"] = pd.Series(ids, index=df.index).astype(df["player_id"].dtype)
        for col in ("birthdate", "height_in", "weight_lb", "player_name"):
            vals = df[col].to_numpy().copy()
            vals[future] = vals[future][rng.permutation(int(future.sum()))]
            df[col] = pd.Series(vals, index=df.index).astype(df[col].dtype)


@contextlib.contextmanager
def scrambled_future(tables: Mapping[str, pd.DataFrame], season: str, seed: int = 0) -> Iterator[None]:
    """Temporarily scramble, IN PLACE, every future row of the caller's own DataFrame objects.

    In-place matters: a cheating projector that captured a reference to ``tables['game_logs']``
    sees the scrambled values, whereas swapping the dict entries would leave its reference
    pointing at the pristine data.  Everything is restored on exit and verified by content hash.
    """
    cutoff = season_start(season)
    frames = {n: df for n, df in tables.items() if isinstance(df, pd.DataFrame)}
    before = data_hash(frames)
    snapshots = {n: df.copy(deep=True) for n, df in frames.items()}
    rng = np.random.default_rng(seed)
    try:
        for name, df in frames.items():
            if name == "players":
                _scramble_players(df, cutoff, rng)
            elif "season" in df.columns:
                _scramble_season_frame(df, cutoff, rng)
        yield
    finally:
        for name, df in frames.items():
            snap = snapshots[name]
            for col in snap.columns:
                df[col] = snap[col]
        if data_hash(frames) != before:  # pragma: no cover - would be a bug in this module
            raise RuntimeError("scrambled_future failed to restore the tables; do not trust this session")


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("player_id", kind="stable").reset_index(drop=True)


def _diff_summary(a: pd.DataFrame, b: pd.DataFrame) -> str:
    if len(a) != len(b) or set(a["player_id"]) != set(b["player_id"]):
        return f"different player sets ({len(a)} vs {len(b)} rows)"
    a, b = _canonical(a), _canonical(b)
    cols = []
    for c in a.columns:
        if c not in b.columns:
            cols.append(f"{c} (missing in perturbed run)")
        elif pd.api.types.is_float_dtype(a[c]):
            if not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=1e-9, atol=1e-9, equal_nan=True):
                n = int((~np.isclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=1e-9, atol=1e-9, equal_nan=True)).sum())
                cols.append(f"{c} ({n} rows)")
        elif not a[c].equals(b[c]):
            cols.append(c)
    return "columns differ: " + ", ".join(cols[:8]) if cols else "no difference"


def frames_equivalent(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return _diff_summary(a, b) == "no difference"


def assert_projector_ignores_future(
    projector: Projector,
    tables: Mapping[str, pd.DataFrame],
    season: str,
    *,
    sanitize: bool = True,
    seed: int = 0,
    check_determinism: bool = True,
) -> None:
    """Raise ``LeakageError`` unless ``projector``'s output for ``season`` is invariant to the future.

    Runs the projector on ``build_history(tables, season)`` then again while every row of season
    >= ``season`` (and the future-derived parts of ``players``) is scrambled in place in
    ``tables``.  Outputs must be identical.  With ``check_determinism`` a preliminary second
    clean run rejects non-deterministic projectors, whose noise would otherwise be
    indistinguishable from leakage (raises ``ValueError``: fix the projector's seeding).
    """
    baseline = projector.project(build_history(tables, season, sanitize=sanitize))
    if check_determinism:
        again = projector.project(build_history(tables, season, sanitize=sanitize))
        if not frames_equivalent(baseline, again):
            raise ValueError(
                f"projector {getattr(projector, 'name', projector)!r} is not deterministic "
                f"({_diff_summary(baseline, again)}); the future-invariance check needs a fixed seed")
    with scrambled_future(tables, season, seed=seed):
        perturbed = projector.project(build_history(tables, season, sanitize=sanitize))
    if not frames_equivalent(baseline, perturbed):
        raise LeakageError(
            f"projector {getattr(projector, 'name', projector)!r} output for {season} changed when "
            f"seasons >= {season} were scrambled ({_diff_summary(baseline, perturbed)}): "
            f"it reads future data")
