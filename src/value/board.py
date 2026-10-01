"""The ranked draft board: projections + VORP + tiers (+ ADP gap).

Library use::

    board = build_board(projections, players, teams=13, adp=adp_frame)

CLI (reads the parquet store, see ``src.store``; ``NBA_DATA_DIR`` selects the data root)::

    python -m src.value.board --season 2026-27 --model baseline --out board.csv
    python -m src.value.board --season 2019-20 --synthetic --out board.csv     # demo, no real data needed

Columns of the board (sorted by ``rank``; complete: every projected player appears)::

    rank            1 = best VORP (ties broken by projected total, then player_id)
    player_id, name, position
    proj_fppg, proj_gp, proj_total_fp
    vorp            proj_total_fp - replacement total
    vorp_per_game   proj_fppg - replacement FPPG
    fppg_p10 / fppg_p50 / fppg_p90   floor / median / ceiling of game-level FP
    tier            1 = best; the last tier is "at or below replacement"
    adp, adp_gap    only when an ADP frame is supplied; adp_gap = adp - rank
                    (positive: the model ranks him earlier than the market drafts him = value)

The CLI (real data only) then adds display-only overlays that never change a projection or a rank: the draft-day risk flags (ADR 0016),
the advisory return flag ``return_flag`` / ``return_block_pct`` / ``return_tail`` / ``return_gp_upside_adv`` / ``return_fp_upside_adv``
(ADR 0023), the advisory short-absence flag ``lm_flag`` / ``lm_iso_n`` / ``lm_rest_n`` / ``lm_gp_risk_adv`` / ``lm_fp_risk_adv`` (ADR 0024)
and the unvalidated contract flags (ADR 0019).

The CLI's ``--adp`` accepts either shape: a plain ``player_id, adp`` CSV, or the raw ingested
ADP table (``season, source, source_id, adp``, e.g. ``adp.parquet`` from
``src.ingest.espn_adp``) alongside ``player_id_map`` in the same data dir — the CLI maps
``source_id`` to ``player_id`` and filters to ``--season`` automatically (see ``load_adp`` in
``src.backtest.benchmarks``).

Extra projection columns that exist (``is_rookie``, ``confidence``, ``age``, and from the debutant models ``projection_class`` and
``p_play``) are carried along.
The replacement-level and positional-scarcity details are in ``board.attrs``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.contracts import HISTORY_TABLES, History, validate_table
from src.value.league import load_league
from src.value.replacement import DEFAULT_SCARCITY_THRESHOLD, league_shape
from src.value.tiers import assign_tiers
from src.value.vorp import POSITIONAL_MODES, compute_vorp

CARRIED = ("age", "is_rookie", "confidence", "projection_class", "p_play",
           "proj_p_appear", "proj_total_fp_p10", "proj_total_fp_p50", "proj_total_fp_p90")
BASE_COLUMNS = ["rank", "player_id", "name", "position", "proj_fppg", "proj_gp", "proj_total_fp",
                "vorp", "vorp_per_game", "fppg_p10", "fppg_p50", "fppg_p90", "tier"]


def build_board(projections: pd.DataFrame, players: pd.DataFrame | None = None, cfg: dict | None = None, *,
                teams: int | None = None, adp: pd.DataFrame | None = None, bench_weight: float | None = None,
                season_games: float = 82.0, positional: str = "auto",
                scarcity_threshold: float = DEFAULT_SCARCITY_THRESHOLD, blend=None, **tier_kwargs) -> pd.DataFrame:
    """Rank ``projections`` (one model, one season) into a draft board.

    ``teams`` overrides ``league.teams`` from the config (13, confirmed live against the real
    ESPN league; see ADR 0008). ``adp`` needs
    ``player_id`` and ``adp`` columns (overall pick number, 1 = first pick). ``blend`` (an
    ``src.value.adp_blend.AdpBlend``, needs ``adp``) adds ``blend_total_fp / blend_vorp / blend_rank /
    blend_tier``, an ADP-anchored second ordering (ADR 0032); every other column is unchanged. Extra
    keyword arguments go to ``assign_tiers``.
    """
    validate_table(projections, "projections")
    if projections["model"].nunique() > 1 or projections["season"].nunique() > 1:
        raise ValueError("build_board needs projections from a single model and season")
    proj = projections.reset_index(drop=True)
    if players is not None and "position" in players.columns:
        pos = players.drop_duplicates("player_id").set_index("player_id")["position"]
        mapped = pos.reindex(proj["player_id"]).to_numpy()
        if "position" in proj.columns:          # debutant rows carry the position from their profile (ADR 0016)
            mapped = pd.Series(mapped, dtype="object").where(pd.Series(mapped).notna(), proj["position"].reset_index(drop=True)).to_numpy()
        proj = proj.assign(position=mapped)
    elif "position" not in proj.columns:
        proj = proj.assign(position=None)
    proj["position"] = proj["position"].astype("object")

    shape = league_shape(cfg or load_league(), teams=teams)
    has_pos = proj["position"].notna().any()
    res = compute_vorp(proj if has_pos else proj.drop(columns="position"), shape, bench_weight=bench_weight,
                       season_games=season_games, positional=positional if has_pos else "off",
                       scarcity_threshold=scarcity_threshold)
    b = proj.join(res.frame)
    b = b.sort_values(["vorp", "proj_total_fp", "player_id"], ascending=[False, False, True],
                      kind="mergesort").reset_index(drop=True)
    b["rank"] = range(1, len(b) + 1)
    b["tier"] = assign_tiers(b["vorp"].to_numpy(), floor=0.0, **tier_kwargs)
    b = b.rename(columns={"player_name": "name"})
    cols = BASE_COLUMNS + [c for c in CARRIED if c in b.columns]
    if adp is not None:
        if not {"player_id", "adp"} <= set(adp.columns):
            raise ValueError("adp frame needs columns player_id and adp")
        a = adp.drop_duplicates("player_id").set_index("player_id")["adp"].astype(float)
        b["adp"] = a.reindex(b["player_id"]).to_numpy()
        b["adp_gap"] = b["adp"] - b["rank"]
        cols += ["adp", "adp_gap"]
    out = b[cols].copy()
    out.attrs["replacement"] = res.replacement.as_dict()
    out.attrs["positional"] = res.positional.as_dict() if res.positional else None
    out.attrs["positional_used"] = res.positional_used
    if blend is not None and adp is not None:
        from src.value.adp_blend import add_blend_columns

        attrs = dict(out.attrs)
        out = add_blend_columns(out, proj, adp, blend, cfg=cfg, teams=teams, bench_weight=bench_weight,
                                season_games=season_games, positional=positional,
                                scarcity_threshold=scarcity_threshold, **tier_kwargs)
        out.attrs.update(attrs)
    return out


# --------------------------------------------------------------------------- CLI

def _load_history(season: str, synthetic: bool, data_dir: Path | None) -> History:
    from src.contracts import season_start

    if synthetic:
        from src.synthetic import make_synthetic_tables

        start = season_start(season)
        tables = make_synthetic_tables(first_start=start - 6, last_start=start - 1, n_teams=14,
                                       games_per_team=60, seed=0)
    else:
        from src.store import load_tables

        tables = load_tables(HISTORY_TABLES, base=data_dir)
    return History.until(tables, season)


def _safe_print(text: str, *, file=None) -> None:
    """``print`` that never crashes on a non-UTF-8 console (e.g. Windows cp1252) when a player
    name contains a non-ASCII character such as an accent (Jokic, Doncic, ...): encoding a name
    like that is exactly what killed this CLI on a plain Windows terminal before this fix, even
    though nothing else was wrong. Falls back to replacing the unencodable characters."""
    stream = file or sys.stdout
    try:
        print(text, file=stream)
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding), file=stream)


def load_adp_for_board(path: Path, season: str, data_dir: Path | None, *, source: str = "espn",
                        min_confidence: float = 0.0) -> tuple[pd.DataFrame, dict | None]:
    """Read ``--adp`` in either accepted shape.

    A plain ``player_id, adp`` CSV is used as-is. The raw ingested ADP table (``season, source,
    source_id, adp``, e.g. ``adp.parquet``) is mapped to ``player_id`` via ``player_id_map`` from
    the same data dir and filtered to ``season``; returns ``(frame, stats)`` where ``stats`` is
    ``None`` for the plain shape and a mapping-coverage dict for the mapped shape.
    """
    raw = pd.read_parquet(path) if path.suffix.lower() in (".parquet", ".pq") else pd.read_csv(path)
    if {"player_id", "adp"} <= set(raw.columns) and "source_id" not in raw.columns:
        return raw[["player_id", "adp"]], None

    import tempfile

    from src.backtest.benchmarks import BacktestError, load_adp
    from src.store import load_tables

    id_map = load_tables(("player_id_map",), base=data_dir)["player_id_map"]
    # Scope the raw rows to ``season`` before handing them to load_adp, so the mapping-coverage
    # stats we report (n_rows, n_unmapped, ...) describe this season, not every season in the
    # file -- load_adp itself only filters its *output* frame by season, not its input stats.
    season_raw = raw[raw["season"] == season] if "season" in raw.columns else raw
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / f"adp_{season}{path.suffix or '.parquet'}"
        if tmp_path.suffix.lower() in (".parquet", ".pq"):
            season_raw.to_parquet(tmp_path)
        else:
            season_raw.to_csv(tmp_path, index=False)
        try:
            mapped = load_adp(tmp_path, id_map, source=source, min_confidence=min_confidence)
        except BacktestError as e:
            raise BacktestError(str(e)) from e
    frame = mapped.frame[mapped.frame["season"] == season]
    stats = {"n_rows": mapped.n_rows, "n_unmapped": mapped.n_unmapped,
             "n_low_confidence": mapped.n_low_confidence, "n_duplicates": mapped.n_duplicates,
             "n_for_season": len(frame)}
    return frame[["player_id", "adp"]], stats


def load_blend(spec: str | Path | None, data_dir: Path | None):
    """The ADP blend named by ``spec`` ('off' / None -> none; 'auto' -> the data dir's file if present; else a path).

    A missing or unreadable 'auto' file is simply no blend (the board then has no ``blend_*`` columns); an explicit path
    that cannot be read is an error, so a typo is not silently ignored.
    """
    from src.contracts import data_dir as default_dir
    from src.value.adp_blend import BLEND_FILE, AdpBlend

    if spec is None or str(spec) == "off":
        return None
    if str(spec) == "auto":
        path = (data_dir or default_dir()) / "processed" / BLEND_FILE
        if not path.exists():
            return None
        try:
            return AdpBlend.load(path)
        except (ValueError, TypeError, KeyError, OSError):
            return None
    return AdpBlend.load(Path(spec))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.value.board", description=__doc__.split("\n\n")[0])
    ap.add_argument("--season", required=True, help="target season, e.g. 2026-27")
    ap.add_argument("--model", default="baseline", help="projector name (see src.models.registry)")
    ap.add_argument("--out", type=Path, required=True, help="CSV path to write")
    ap.add_argument("--teams", type=int, default=None, help="override league.teams (config value is 13, confirmed live)")
    ap.add_argument("--adp", type=Path, default=None,
                    help="CSV with player_id, adp columns, OR the raw ingested ADP table "
                         "(season, source, source_id, adp) alongside player_id_map -- see module docstring")
    ap.add_argument("--adp-source", default="espn", help="source namespace when --adp is the raw ingested table")
    ap.add_argument("--adp-min-confidence", type=float, default=0.0,
                    help="drop player_id_map matches below this confidence when --adp is the raw ingested table")
    ap.add_argument("--adp-blend", default="auto",
                    help="ADP + model blend coefficients (adp_blend.json from `python -m src.value.adp_blend fit`): a path, "
                         "'auto' (the data dir's file when it exists and --adp is given) or 'off'")
    ap.add_argument("--positional", choices=POSITIONAL_MODES, default="auto")
    ap.add_argument("--bench-weight", type=float, default=None, help="override the derived bench weight in [0, 1]")
    ap.add_argument("--data-dir", type=Path, default=None, help="data root (default: NBA_DATA_DIR / ~/dev-data)")
    ap.add_argument("--synthetic", action="store_true", help="use the synthetic league (demo/testing only)")
    ap.add_argument("--top", type=int, default=15, help="rows to print")
    args = ap.parse_args(argv)

    from src.models.panel import season_lengths, target_season_games
    from src.models.registry import get_projector

    try:
        history = _load_history(args.season, args.synthetic, args.data_dir)
    except FileNotFoundError as e:
        _safe_print(f"error: {e}", file=sys.stderr)
        return 2
    projector = get_projector(args.model)
    proj = projector.project(history)

    adp = None
    if args.adp:
        from src.backtest.benchmarks import BacktestError

        try:
            adp, adp_stats = load_adp_for_board(args.adp, args.season, args.data_dir,
                                                 source=args.adp_source,
                                                 min_confidence=args.adp_min_confidence)
        except (BacktestError, FileNotFoundError) as e:
            _safe_print(f"error: {e}", file=sys.stderr)
            return 2
        if adp_stats is not None:
            _safe_print(f"adp: {adp_stats['n_for_season']} players for {args.season} "
                        f"({adp_stats['n_unmapped']} unmapped, "
                        f"{adp_stats['n_low_confidence']} below confidence threshold, "
                        f"{adp_stats['n_duplicates']} duplicate picks, out of {adp_stats['n_rows']} raw rows)")
    games = target_season_games(season_lengths(history, None)) if len(history.team_games) else 82
    blend = load_blend(args.adp_blend, args.data_dir) if adp is not None else None
    board = build_board(proj, history.players, teams=args.teams, adp=adp, bench_weight=args.bench_weight,
                        season_games=games, positional=args.positional, blend=blend)
    if not args.synthetic:
        try:
            from src.value.risk import attach_risk, compute_risk, season_is_live

            if season_is_live(args.season):
                overlay, risk_notes = compute_risk(proj, board, history, args.season, args.data_dir)
                board = attach_risk(board, overlay)
                for n in risk_notes:
                    _safe_print(f"note: {n}")
        except (FileNotFoundError, KeyError, ValueError, OSError) as e:
            _safe_print(f"note: draft-day risk flags unavailable ({e})")
        try:   # ADR 0023: advisory "returned from a long absence, healthy since" flag; never changes a projection or a rank
            from src.value.return_flag import attach_return_flag, compute_return_flag

            rf, rf_notes = compute_return_flag(board, history, season_games=float(games))
            board = attach_return_flag(board, rf)
            for n in rf_notes:
                _safe_print(f"note: {n}")
        except (FileNotFoundError, KeyError, ValueError, OSError) as e:
            _safe_print(f"note: return flag unavailable ({e})")
        try:   # ADR 0024: advisory descriptive "many short absences" flag; never changes a projection or a rank
            from src.value.load_flag import attach_load_flag, compute_load_flag

            lf, lf_notes = compute_load_flag(board, history, season_games=float(games))
            board = attach_load_flag(board, lf)
            for n in lf_notes:
                _safe_print(f"note: {n}")
        except (FileNotFoundError, KeyError, ValueError, OSError) as e:
            _safe_print(f"note: short-absence flag unavailable ({e})")
        try:   # ADR 0019: display-only, unvalidated contract flags; never changes a projection or a rank
            from src.value.contract_flags import attach_contract_flags, compute_contract_flags

            cf, cf_notes = compute_contract_flags(board, history, args.season, args.data_dir)
            board = attach_contract_flags(board, cf)
            for n in cf_notes:
                _safe_print(f"note: {n}")
        except (FileNotFoundError, KeyError, ValueError, OSError) as e:
            _safe_print(f"note: contract flags unavailable ({e})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(args.out, index=False, encoding="utf-8")

    rep = board.attrs["replacement"]
    _safe_print(f"{args.model} board for {args.season}: {len(board)} players -> {args.out}")
    _safe_print(f"league: {rep['teams']} teams, replacement rank {rep['rank']:.1f} "
                f"(bench weight {rep['bench_weight']:.2f}), replacement season total {rep['total']:.0f} FP")
    pos = board.attrs["positional"]
    if pos:
        _safe_print(f"positional scarcity index {pos['scarcity_index']:.3f} "
                    f"({'material' if pos['material'] else 'weak'}; applied: {board.attrs['positional_used']})")
    _safe_print(board.head(args.top).to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
