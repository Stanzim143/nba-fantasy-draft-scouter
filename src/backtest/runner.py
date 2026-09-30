"""Command-line runner: ``python -m src.backtest``.

Examples::

    python -m src.backtest --model baseline --seasons 2016-17:2025-26 --out reports/
    python -m src.backtest --model baseline --benchmark naive_last_season
    python -m src.backtest --ablate baseline,baseline_injury,baseline_roster --n-boot 1000
    python -m src.backtest --synthetic --model naive_last_season --seasons 2017-18:2018-19   # smoke test

Model names resolve through ``src.models.registry.get_projector`` (imported lazily so this
package works before the model track lands).  ``naive_last_season`` falls back to the local
implementation in ``benchmarks.py`` when the registry has no such entry.  One command reruns any
season: ``--seasons 2021-22``.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import shlex
import sys
from typing import Sequence

import pandas as pd

from src.backtest.ablation import ablation_from_results
from src.backtest.benchmarks import AdpBenchmark, NaiveLastSeason, load_adp
from src.backtest.errors import BacktestError, LeakageError, ProjectorNotAvailable
from src.backtest.harness import BacktestResult, expand_seasons, walk_forward
from src.backtest.metrics import DEFAULT_KS, DEFAULT_MIN_GP
from src.backtest.misses import analyze_misses
from src.backtest.report import write_report
from src.contracts import HISTORY_TABLES, season_start
from src.value.league import load_league

log = logging.getLogger("src.backtest")

LOCAL_FALLBACKS = {"naive_last_season": NaiveLastSeason}


# --------------------------------------------------------------------------- projector resolution

def _import_registry():
    """Import the model registry, or return None if the model track has not landed yet."""
    try:
        return importlib.import_module("src.models.registry")
    except ModuleNotFoundError as exc:
        # Only "the registry itself is absent" is tolerated; a missing dependency *inside* an
        # existing registry must surface, so it is re-raised.
        if exc.name and "src.models.registry".startswith(exc.name):
            return None
        raise


def resolve_projector(name: str):
    """Look ``name`` up in the model registry, falling back to local benchmark implementations."""
    reg = _import_registry()
    if reg is not None:
        try:
            p = reg.get_projector(name)
        except (KeyError, LookupError, ValueError):
            p = None
        if p is not None:
            return p() if isinstance(p, type) else p
    if name in LOCAL_FALLBACKS:
        if reg is None:
            log.info("model registry not available; using the local %s implementation", name)
        else:
            log.info("%r not in the model registry; using the local implementation", name)
        return LOCAL_FALLBACKS[name]()
    if reg is None:
        raise ProjectorNotAvailable(
            f"cannot resolve projector {name!r}: the model registry (src.models.registry) is not "
            f"available yet. Until the model track lands, only {sorted(LOCAL_FALLBACKS)} run from the CLI.")
    avail = []
    try:
        avail = list(reg.available_projectors())
    except Exception:  # noqa: BLE001 - a hint only
        pass
    raise ProjectorNotAvailable(f"unknown projector {name!r}; available: {sorted(avail)}")


# --------------------------------------------------------------------------- data

def load_run_tables(args) -> dict[str, pd.DataFrame]:
    if args.synthetic:
        from src.synthetic import make_synthetic_tables
        seasons = expand_seasons(args.seasons)
        first = season_start(seasons[0]) - args.synthetic_history
        last = season_start(seasons[-1])
        return make_synthetic_tables(first_start=first, last_start=last, n_teams=args.synthetic_teams,
                                     games_per_team=args.synthetic_games, seed=args.synthetic_seed)
    from src.store import load_tables
    try:
        tables = dict(load_tables(HISTORY_TABLES))
    except FileNotFoundError as exc:
        raise BacktestError(f"{exc}. Run the ingest first, or pass --synthetic for a pipeline smoke test.") from exc
    # Optional feature tables travel as History.extras, so History.until slices them and the leakage checks
    # cover them. Absent files are simply absent: models that need them fall back to the base projection.
    try:
        from src.ingest.nba_offseason import read_offseason_logs, read_offseason_team_games

        tables["offseason_logs"] = read_offseason_logs()
        tables["offseason_team_games"] = read_offseason_team_games()
    except (ImportError, FileNotFoundError):
        pass
    try:   # ADR 0019: dated contract events, tagged with the season they follow so History.until slices them
        from src.ingest.wiki_contracts import read_player_contracts

        tables["player_contracts"] = read_player_contracts()
    except (ImportError, FileNotFoundError):
        pass
    return tables


# --------------------------------------------------------------------------- CLI

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.backtest", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="projector name to backtest (model registry)")
    p.add_argument("--benchmark", help="comma-separated benchmark projector names, e.g. naive_last_season")
    p.add_argument("--ablate", help="comma-separated ordered variants, e.g. baseline,injury,roster,contract")
    p.add_argument("--adp-file", help="ADP CSV/Parquet (see src/backtest/benchmarks.py) for an ADP benchmark")
    p.add_argument("--adp-source", default="espn", help="player_id_map source for the ADP file (default espn)")
    p.add_argument("--method-checks", action="store_true",
                   help="append the ADR 0030 checks: games-played calibration by risk group, leave-one-season-out "
                        "sensitivity and, with --adp-file, the ADP + model arms and top-N tier comparison")
    p.add_argument("--seasons", default="2016-17:2025-26", help="e.g. 2016-17:2025-26, or 2021-22, or a,b list")
    p.add_argument("--out", default="reports", help="output root; the run goes in <out>/<run-id>/")
    p.add_argument("--run-id", help="override the run directory name")
    p.add_argument("--ks", default=",".join(map(str, DEFAULT_KS)), help="top-K cutoffs (default 12,50,100)")
    p.add_argument("--min-gp", type=int, default=DEFAULT_MIN_GP, help="min games for FPPG metrics")
    p.add_argument("--n-boot", type=int, default=500, help="bootstrap replicates")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.add_argument("--sig-metrics", help="comma-separated metrics for the paired ablation lifts (default: "
                                         "spearman_total_fp,top50_hit,mae_total_fp); e.g. add mae_gp")
    p.add_argument("--top-misses", type=int, default=10, help="misses listed per season (0 = skip)")
    p.add_argument("--leak-check", action="store_true",
                   help="also run the future-invariance check on the last season (3 extra runs per projector)")
    p.add_argument("--cache-dir", help="cache projections (needs projector.fingerprint)")
    p.add_argument("--synthetic", action="store_true", help="run on a synthetic league (smoke test; NOT a result)")
    p.add_argument("--synthetic-teams", type=int, default=30)
    p.add_argument("--synthetic-games", type=int, default=82)
    p.add_argument("--synthetic-history", type=int, default=3, help="synthetic seasons before the first target")
    p.add_argument("--synthetic-seed", type=int, default=0)
    return p


def _names(csv: str | None) -> list[str]:
    return [x.strip() for x in csv.split(",") if x.strip()] if csv else []


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)

    model, benchmarks, ablate = (args.model, _names(args.benchmark), _names(args.ablate))
    if not (model or benchmarks or ablate or args.adp_file):
        parser.error("nothing to run: give --model, --benchmark, --ablate and/or --adp-file")
    try:
        return _run(args, model, benchmarks, ablate, shlex.join(["python", "-m", "src.backtest", *argv]))
    except LeakageError as exc:
        print(f"LEAKAGE: {exc}", file=sys.stderr)
        return 3
    except (BacktestError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(args, model: str | None, benchmarks: list[str], ablate: list[str], command: str) -> int:
    ks = tuple(int(k) for k in args.ks.split(","))
    seasons = expand_seasons(args.seasons)
    tables = load_run_tables(args)
    scoring = load_league()["scoring"]
    kw = dict(scoring=scoring, ks=ks, min_gp=args.min_gp, synthetic=args.synthetic,
              cache_dir=args.cache_dir, leak_check=[seasons[-1]] if args.leak_check else (),
              extra_metadata={"command": command})

    results: dict[str, BacktestResult] = {}

    def run(name: str) -> BacktestResult:
        if name not in results:
            log.info("== %s", name)
            results[name] = walk_forward(tables, resolve_projector(name), seasons, **kw)
        return results[name]

    variants = [run(n) for n in ablate]
    primary = run(model) if model else (variants[-1] if variants else None)
    bench = [run(n) for n in benchmarks if n != (primary.projector if primary else None)]
    adp = None
    if args.adp_file:
        adp = load_adp(args.adp_file, tables_id_map(args), source=args.adp_source)
        log.info("ADP: %d rows, %.1f%% unmapped", adp.n_rows, 100 * adp.unmapped_share)
        bench.append(walk_forward(tables, AdpBenchmark(adp), seasons, **kw))
    if primary is None:                       # only benchmarks requested: report the first
        primary, bench = bench[0], bench[1:]

    ablation = (ablation_from_results([(r.projector, r) for r in variants], n_boot=args.n_boot, seed=args.seed,
                                      **({"sig_metrics": [m.strip() for m in args.sig_metrics.split(",") if m.strip()]}
                                         if getattr(args, "sig_metrics", None) else {}))
                if len(variants) >= 2 else None)
    misses = None
    if args.top_misses and not primary.config["rank_only"]:
        misses = analyze_misses(primary, tables, top_n=args.top_misses)
    extra_md = None
    if args.method_checks and not primary.config["rank_only"]:
        from src.backtest.method_checks import method_checks_markdown
        extra_md = method_checks_markdown(primary, tables, adp.frame if adp is not None else None)
    out = write_report(primary, args.out, benchmarks=bench, ablation=ablation, misses=misses,
                       run_id=args.run_id, n_boot=args.n_boot, seed=args.seed, command=command, extra_md=extra_md)

    s = primary.summary()
    print(f"{primary.projector}: spearman_total_fp={s['spearman_total_fp']:.3f} "
          f"top{max(ks)}_hit={s[f'top{max(ks)}_hit']:.3f} mae_total_fp={s['mae_total_fp']:.1f} "
          f"({len(seasons)} seasons){'  [SYNTHETIC DATA]' if args.synthetic else ''}")
    print(f"report: {out / 'report.md'}")
    return 0


def tables_id_map(args) -> pd.DataFrame:
    """player_id_map for the ADP benchmark (synthetic runs have none, so the ADP hook needs real data)."""
    if args.synthetic:
        raise BacktestError("--adp-file needs the ingested player_id_map; it cannot be combined with --synthetic")
    from src.store import read_table
    try:
        return read_table("player_id_map")
    except FileNotFoundError as exc:
        raise BacktestError(f"{exc} (the ADP benchmark maps ids through player_id_map)") from exc
