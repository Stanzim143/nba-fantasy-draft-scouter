"""Display-only contract flags for a board or watchlist (ADR 0019). UNVALIDATED: projections and ranks never change.

    from src.value.contract_flags import compute_contract_flags, attach_contract_flags
    overlay, notes = compute_contract_flags(board, history, season, data_dir)
    board = attach_contract_flags(board, overlay)

The flags come from ``player_contracts.parquet`` (dated Wikipedia contract events, ``python -m src.ingest.wiki_contracts``)
sliced to events dated before opening night of ``season``, plus the nominal rookie-scale clock for first-rounders with no
known deal. The walk-forward (ADR 0019) found no reliable lift from these features, so nothing here feeds a projection.
They are shown because a contract year is information a drafter may want, not because it was shown to predict anything.
**A blank flag means "not known to be", never "known not to be":** coverage of Wikipedia is partial and biased toward
notable signings, and ``contract_status`` says exactly what is known for each player.

Columns added by :func:`attach_contract_flags`:

``contract_flag``        contract year | new deal | extension | rookie final year (nominal) | rookie option year (nominal) | ''
``contract_years_left``  seasons left on a known deal including ``season`` (NaN unless the deal and its length are known)
``contract_status``      known | no_length | lapsed | unknown
``contract_basis``       wiki | rookie_scale | ''
``contract_validation``  constant ``unvalidated_display_only`` on every row, so a consumer of the board CSV (which has no
                         console) sees that the four columns above did not survive the walk-forward evaluation
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.contracts import History
from src.features.contract import contract_clock
from src.features.contract_terms import terms_features

FLAG_COLUMNS = ["contract_flag", "contract_years_left", "contract_status", "contract_basis"]
VALIDATION_COLUMN = "contract_validation"
VALIDATION_LABEL = "unvalidated_display_only"
_LABELS = {"contract_year": "contract year", "new_deal": "new deal", "extension": "extension",
           "rookie_contract_year": "rookie final year (nominal)", "rookie_option_year": "rookie option year (nominal)"}


def compute_contract_flags(board: pd.DataFrame, history: History, season: str, data_dir: Path | None = None
                           ) -> tuple[pd.DataFrame, list[str]]:
    """The overlay (``player_id`` + :data:`FLAG_COLUMNS`) for every player of ``board`` and notes on what was missing."""
    notes: list[str] = []
    ids = board["player_id"].to_numpy("int64")
    contracts = None
    try:
        from src.ingest.wiki_contracts import read_player_contracts

        contracts = read_player_contracts(data_dir)
    except FileNotFoundError:
        notes.append("no player_contracts table: only the nominal rookie-scale flags are shown "
                     "(run python -m src.ingest.wiki_contracts)")
    tf = terms_features(contracts, season, ids, contract_clock(history.players, season, ids))
    flag = np.array([_LABELS.get(f, f) for f in tf.flag()], dtype=object)
    if contracts is not None:
        n_known = int((tf.status == "known").sum())
        n_any = int((tf.status != "unknown").sum())
        notes.append(f"contract flags (unvalidated, display only): {n_known}/{len(ids)} players have a known deal length and "
                     f"{n_any}/{len(ids)} any Wikipedia contract event dated before opening night; blank = not known to be, "
                     f"not known not to be")
    out = pd.DataFrame({"player_id": ids, "contract_flag": flag, "contract_years_left": tf.years_remaining,
                        "contract_status": tf.status, "contract_basis": tf.basis()})
    return out, notes


def attach_contract_flags(board: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    """``board`` with the overlay merged in (row order and every existing column unchanged)."""
    out = board.merge(overlay, on="player_id", how="left")
    out["contract_flag"] = out["contract_flag"].fillna("")
    out["contract_status"] = out["contract_status"].fillna("unknown")
    out["contract_basis"] = out["contract_basis"].fillna("")
    out[VALIDATION_COLUMN] = VALIDATION_LABEL
    out.attrs = dict(board.attrs)
    return out
