"""Opt-in: independently re-verify ADR 0029's D3 real-data statistics and the
``DISAGREEMENT_THRESHOLD`` sanity against the real Yahoo/FantasyPros snapshot, when both the real
history tables and the real snapshot files are present (skipped otherwise, following the same
"opt-in on real ingested data" pattern as ``tests/models/test_model_real_data.py``).

This is not a re-statement of the ADR's own numbers -- it rebuilds the real 2026-27 baseline board,
re-parses the real snapshot files, and rebuilds the comparison from scratch, then checks the
distribution shape a fresh snapshot could plausibly break (e.g. a matching regression that suddenly
resolves far fewer/more players, or shifts the disagreement rate wildly) without pinning the exact
2026-09-28 numbers as a brittle golden file.
"""
from __future__ import annotations

import pytest

from src.contracts import HISTORY_TABLES, data_dir, table_path
from src.ingest.fantasypros_rankings import DEFAULT_RELATIVE_PATH as FP_DEFAULT_RELATIVE_PATH
from src.ingest.fantasypros_rankings import load_fantasypros_rankings
from src.ingest.yahoo_rankings import default_path as yahoo_default_path
from src.ingest.yahoo_rankings import load_yahoo_rankings
from src.value.rankings_compare import DISAGREEMENT_THRESHOLD, build_comparison

_MISSING_HISTORY = [t for t in HISTORY_TABLES if not table_path(t).exists()]
_YAHOO_PATH = yahoo_default_path()
_FP_PATH = data_dir() / FP_DEFAULT_RELATIVE_PATH
_MISSING_SNAPSHOTS = [p for p in (_YAHOO_PATH, _FP_PATH) if not p.exists()]

pytestmark = pytest.mark.skipif(
    bool(_MISSING_HISTORY) or bool(_MISSING_SNAPSHOTS),
    reason=(f"real data not available (missing history tables: {_MISSING_HISTORY}, "
           f"missing snapshot files: {_MISSING_SNAPSHOTS})"),
)


@pytest.fixture(scope="module")
def real_comparison():
    """Rebuild the real 2026-27 baseline board and the real Yahoo/FantasyPros comparison from
    scratch -- the same pipeline the app and ADR 0029's own D3 analysis used, not a fixture."""
    from src.app.loader import load_board
    from src.store import read_table

    board = load_board("2026-27", "baseline", teams=13)
    players = read_table("players", data_dir())
    resolved = {
        "yahoo": load_yahoo_rankings(_YAHOO_PATH, players).frame,
        "fantasypros": load_fantasypros_rankings(_FP_PATH, players).frame,
    }
    return build_comparison(board, resolved)


def test_disagreement_threshold_flags_a_plausible_band_of_the_realistic_draft_pool(real_comparison):
    """Sanity band, not a golden-file pin: at DISAGREEMENT_THRESHOLD=50, the flagged fraction of the
    realistic draft pool (our_rank <= 150, this league's 13x13) should be a real, informative
    minority -- not near-0% (the threshold would be pointless) and not near-100% (the threshold
    would be meaningless/too loose). ADR 0029's own D3 distribution (n=150, median 36.5, p75 66.0,
    p90 99.3) implies something in the neighborhood of a third to a half of that pool clears 50; a
    future snapshot that silently broke matching (e.g. resolved far fewer players, or shifted ranks
    systematically) would very likely push this fraction outside a generous [0.10, 0.75] band."""
    pool = real_comparison[real_comparison["our_rank"] <= 150]
    assert len(pool) >= 100, "the realistic draft pool should have close to 150 rows with real data"

    flagged_frac = float(pool["rankings_disagreement"].mean())
    assert 0.10 <= flagged_frac <= 0.75, (
        f"flagged fraction of the top-150 pool at threshold={DISAGREEMENT_THRESHOLD} is "
        f"{flagged_frac:.3f}, outside the plausible [0.10, 0.75] band -- re-check the real "
        "snapshot/matching, or reconsider the threshold (see ADR 0029 D3)."
    )


def test_top_150_pool_has_computable_deltas_for_most_rows(real_comparison):
    """A silently-broken match (e.g. an id_map regression that stops resolving most players onto
    an external rank) would show up here first: most of the top-150 pool should have at least one
    external rank, so max_abs_rank_delta is computable rather than null."""
    pool = real_comparison[real_comparison["our_rank"] <= 150]
    with_delta = pool["max_abs_rank_delta"].notna().sum()
    assert with_delta / len(pool) >= 0.90, (
        f"only {with_delta}/{len(pool)} of the top-150 pool have a computable max_abs_rank_delta; "
        "expected the large majority to resolve onto at least one external source's top ranks."
    )


def test_unrestricted_distribution_is_looser_than_the_realistic_pool(real_comparison):
    """D3's own qualitative claim: the unrestricted (deep-bench-included) distribution should have a
    materially higher median |rank_delta| than the realistic top-150 pool, since coverage thins out
    and gets noisier past the players anyone would actually draft."""
    unrestricted_median = real_comparison["max_abs_rank_delta"].dropna().median()
    pool_median = real_comparison.loc[real_comparison["our_rank"] <= 150, "max_abs_rank_delta"].dropna().median()
    assert unrestricted_median > pool_median, (
        "expected the unrestricted median |rank_delta| to exceed the realistic top-150 pool's "
        f"median (got unrestricted={unrestricted_median:.1f}, pool={pool_median:.1f})"
    )
