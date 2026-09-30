import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from bt_helpers import NoiseProjector
from src.backtest import runner
from src.backtest.ablation import ablation_from_results
from src.backtest.benchmarks import AdpBenchmark, NaiveLastSeason
from src.backtest.errors import LeakageError, ProjectorNotAvailable
from src.backtest.harness import BacktestResult, walk_forward
from src.backtest.misses import analyze_misses
from src.backtest.report import default_run_id, write_report

REPO = Path(__file__).resolve().parents[2]
SEASONS = ["2016-17", "2017-18", "2018-19"]
PNGS = ["metrics_by_season.png", "scatter_total_fp.png", "calibration.png", "ablation.png"]
PNG_MAGIC = bytes([0x89]) + b"PNG\r\n" + bytes([0x1A]) + b"\n"


class Variant(NaiveLastSeason):
    """Naive with games-played shrunk; a distinguishable registry entry."""

    def __init__(self, name, shrink):
        super().__init__()
        self.name, self._shrink = name, shrink
        self.fingerprint = f"{name}/{shrink}"

    def project(self, history):
        p = super().project(history)
        p["proj_gp"] = p["proj_gp"] * self._shrink
        p["proj_total_fp"] = p["proj_fppg"] * p["proj_gp"]
        p["model"] = self.name
        return p


def _is_png(path: Path) -> bool:
    return path.read_bytes()[:8] == PNG_MAGIC and path.stat().st_size > 2000


# ------------------------------------------------------------------ report

@pytest.fixture(scope="module")
def full_run(_tables_master):
    tables = _tables_master          # read-only use: shared across this module
    base = walk_forward(tables, NaiveLastSeason(), SEASONS, synthetic=True)
    v1 = walk_forward(tables, Variant("v1", 0.9), SEASONS, synthetic=True)
    v2 = walk_forward(tables, NoiseProjector(tables, 0.3), SEASONS, synthetic=True)
    abl = ablation_from_results([("naive", base), ("v1", v1), ("noisy_oracle", v2)], n_boot=40, seed=1)
    return tables, base, v1, abl


def test_write_report_writes_every_expected_file(full_run, tmp_path):
    tables, base, v1, abl = full_run
    misses = analyze_misses(base, tables, top_n=5, gp_gap=10)
    out = write_report(base, tmp_path, benchmarks=[v1], ablation=abl, misses=misses, n_boot=40,
                       command="python -m src.backtest --demo")
    assert out == tmp_path / default_run_id(base)
    for name in ["report.md", "metrics_by_season.csv", "players.parquet", "run.json", "misses.csv", "ablation.csv",
                 *PNGS]:
        assert (out / name).exists(), name
    for png in PNGS:
        assert _is_png(out / png), png
    md = (out / "report.md").read_text(encoding="utf-8")
    for token in ["# Backtest report: naive_last_season", "SYNTHETIC DATA", "## Headline metrics", "## By season",
                  "## Evaluation universe and coverage", "## Benchmarks", "## Predicted vs actual",
                  "## Floor and ceiling calibration", "## Ablation", "## Biggest misses", "## Notes",
                  "Spearman, total FP", "95% CI", "python -m src.backtest --demo", "![ablation](ablation.png)",
                  base.metadata["data_hash"][:16]]:
        assert token in md, token
    assert "n_coverage_miss" in md and "verdict" in md
    # Regression: the "## Run" table's single-column DataFrame used to be named "", colliding
    # with df_to_markdown's own reset-index column name and corrupting every cell in it into a
    # pandas Series repr. Caught for real on a real backtest run's report.md, not by a test,
    # because no prior test asserted on the "## Run" section's actual cell content.
    run_section = md.split("## Run", 1)[1].split("##", 1)[0]
    assert "Name:" not in run_section
    assert "dtype:" not in run_section
    assert "| run id | " + default_run_id(base) + " |" in run_section
    assert "| projector | naive_last_season |" in run_section


def test_report_omits_optional_sections_and_charts(full_run, tmp_path):
    tables, base, *_ = full_run
    out = write_report(base, tmp_path, run_id="plain", n_boot=30)
    md = (out / "report.md").read_text(encoding="utf-8")
    assert out.name == "plain"
    assert "## Ablation" not in md and "## Benchmarks" not in md and "## Biggest misses" not in md
    assert not (out / "ablation.png").exists() and not (out / "misses.csv").exists()
    assert _is_png(out / "metrics_by_season.png") and _is_png(out / "scatter_total_fp.png")


def test_real_data_run_has_no_synthetic_banner(tables, tmp_path):
    res = walk_forward(tables, NaiveLastSeason(), ["2017-18"], synthetic=False)
    md = (write_report(res, tmp_path, n_boot=20) / "report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC DATA" not in md and "used real ingested data" in md


def test_report_is_reproducible(full_run, tmp_path):
    _, base, *_ = full_run
    a = write_report(base, tmp_path / "a", run_id="r", n_boot=30, seed=3)
    b = write_report(base, tmp_path / "b", run_id="r", n_boot=30, seed=3)
    assert (a / "metrics_by_season.csv").read_text() == (b / "metrics_by_season.csv").read_text()
    assert (a / "report.md").read_text() == (b / "report.md").read_text()


def test_rank_only_report_has_no_error_or_calibration_sections(tables, tmp_path):
    naive = walk_forward(tables, NaiveLastSeason(), SEASONS)
    f = naive.players[naive.players["projected"]]
    adp = walk_forward(tables, AdpBenchmark(pd.DataFrame({
        "season": f["season"], "player_id": f["player_id"], "player_name": f["player_name"],
        "adp": -f["proj_total_fp"]})), SEASONS)
    out = write_report(adp, tmp_path, n_boot=20)
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "Floor and ceiling" not in md and not (out / "calibration.png").exists()
    assert not (out / "scatter_total_fp.png").exists()
    assert "Top-50 hit rate" in md and _is_png(out / "metrics_by_season.png")


def test_saved_report_result_can_be_reloaded(full_run, tmp_path):
    _, base, *_ = full_run
    out = write_report(base, tmp_path, n_boot=20)
    back = BacktestResult.load(out)
    pd.testing.assert_frame_equal(back.players, base.players)


# ------------------------------------------------------------------ registry resolution

def _registry(**projectors):
    mod = types.ModuleType("src.models.registry")

    def get_projector(name):
        if name not in projectors:
            raise KeyError(name)
        return projectors[name]

    mod.get_projector = get_projector
    mod.available_projectors = lambda: sorted(projectors)
    return mod


def test_absent_registry_gives_a_helpful_error_but_local_benchmark_still_works(monkeypatch):
    monkeypatch.setattr(runner, "_import_registry", lambda: None)
    with pytest.raises(ProjectorNotAvailable, match=r"model registry.*not.*available.*naive_last_season"):
        runner.resolve_projector("baseline")
    assert isinstance(runner.resolve_projector("naive_last_season"), NaiveLastSeason)


def test_real_import_of_a_missing_registry_returns_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "src.models.registry", None)      # import raises ModuleNotFoundError
    assert runner._import_registry() is None


def test_registry_is_used_when_present_and_classes_are_instantiated(monkeypatch):
    inst = Variant("baseline", 1.0)
    monkeypatch.setattr(runner, "_import_registry", lambda: _registry(baseline=inst, cls=NaiveLastSeason))
    assert runner.resolve_projector("baseline") is inst
    assert isinstance(runner.resolve_projector("cls"), NaiveLastSeason)


def test_registry_entry_overrides_the_local_fallback(monkeypatch):
    theirs = Variant("naive_last_season", 1.0)
    monkeypatch.setattr(runner, "_import_registry", lambda: _registry(naive_last_season=theirs))
    assert runner.resolve_projector("naive_last_season") is theirs


def test_unknown_name_lists_available_projectors(monkeypatch):
    monkeypatch.setattr(runner, "_import_registry", lambda: _registry(baseline=Variant("baseline", 1.0)))
    with pytest.raises(ProjectorNotAvailable, match=r"unknown projector 'nope'.*baseline"):
        runner.resolve_projector("nope")
    assert isinstance(runner.resolve_projector("naive_last_season"), NaiveLastSeason)   # falls back locally


def test_broken_registry_import_is_not_swallowed(monkeypatch):
    def boom(name, *a, **k):
        raise ModuleNotFoundError("No module named 'lightgbm_x'", name="lightgbm_x")

    monkeypatch.setattr("importlib.import_module", boom)
    with pytest.raises(ModuleNotFoundError, match="lightgbm_x"):
        runner._import_registry()


# ------------------------------------------------------------------ CLI

ARGS = ["--synthetic", "--synthetic-teams", "8", "--synthetic-games", "30", "--synthetic-history", "2",
        "--n-boot", "25", "--seasons", "2016-17:2017-18", "--top-misses", "3"]


def _with_seasons(seasons):
    args = list(ARGS)
    args[args.index("--seasons") + 1] = seasons
    return args


def test_cli_smoke_on_synthetic_data(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(runner, "_import_registry", lambda: None)
    rc = runner.main(["--model", "naive_last_season", "--out", str(tmp_path), "--run-id", "smoke", *ARGS])
    assert rc == 0
    out = tmp_path / "smoke"
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC DATA" in md and "python -m src.backtest --model naive_last_season" in md
    assert _is_png(out / "metrics_by_season.png") and (out / "misses.csv").exists()
    stdout = capsys.readouterr().out
    assert "spearman_total_fp=" in stdout and "[SYNTHETIC DATA]" in stdout and "smoke" in stdout
    assert pd.read_csv(out / "metrics_by_season.csv", index_col=0).shape[0] == 2


def test_cli_single_season_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_import_registry", lambda: None)
    args = _with_seasons("2017-18")
    assert runner.main(["--model", "naive_last_season", "--out", str(tmp_path), "--run-id", "one", *args]) == 0
    assert pd.read_csv(tmp_path / "one" / "metrics_by_season.csv", index_col=0).index.tolist() == ["2017-18"]


def test_cli_ablation_benchmark_and_leak_check(tmp_path, monkeypatch):
    reg = _registry(a=Variant("a", 1.0), b=Variant("b", 0.95), c=Variant("c", 0.9),
                    naive_last_season=NaiveLastSeason())
    monkeypatch.setattr(runner, "_import_registry", lambda: reg)
    rc = runner.main(["--ablate", "a,b,c", "--benchmark", "naive_last_season", "--leak-check", "--out",
                      str(tmp_path), "--run-id", "abl", *ARGS])
    assert rc == 0
    out = tmp_path / "abl"
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "## Ablation" in md and "## Benchmarks" in md and "'2017-18': 'passed'" in md
    ab = pd.read_csv(out / "ablation.csv", index_col=0)
    assert ab.index.tolist() == ["a", "b", "c"]
    assert _is_png(out / "ablation.png")
    assert "# Backtest report: c" in md                    # ablation-only: the last variant is the headline


def test_cli_sig_metrics_selects_the_ablation_lifts(tmp_path, monkeypatch):
    reg = _registry(a=Variant("a", 1.0), b=Variant("b", 0.95))
    monkeypatch.setattr(runner, "_import_registry", lambda: reg)
    assert runner.main(["--ablate", "a,b", "--sig-metrics", "mae_gp,mae_total_fp", "--out", str(tmp_path), "--run-id", "sig", *ARGS]) == 0
    ab = pd.read_csv(tmp_path / "sig" / "ablation.csv", index_col=0)
    assert "lift_mae_gp" in ab.columns and "lift_mae_total_fp_lo" in ab.columns and "lift_top50_hit" not in ab.columns


def test_cli_model_plus_benchmark_dedupes_and_compares(tmp_path, monkeypatch):
    reg = _registry(base=Variant("base", 1.0), naive_last_season=NaiveLastSeason())
    monkeypatch.setattr(runner, "_import_registry", lambda: reg)
    assert runner.main(["--model", "base", "--benchmark", "naive_last_season,base", "--out", str(tmp_path),
                        "--run-id", "mb", *ARGS]) == 0
    md = (tmp_path / "mb" / "report.md").read_text(encoding="utf-8")
    assert "Paired lift of the model over each benchmark" in md and "naive_last_season" in md


def test_cli_error_exits(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(runner, "_import_registry", lambda: None)
    with pytest.raises(SystemExit) as e:
        runner.main([])
    assert e.value.code == 2
    assert runner.main(["--model", "baseline", "--out", str(tmp_path), *ARGS]) == 2
    err = capsys.readouterr().err
    assert "model registry" in err and "baseline" in err and not any(tmp_path.iterdir())
    assert runner.main(["--model", "naive_last_season", "--adp-file", "x.csv", "--out", str(tmp_path), *ARGS]) == 2
    assert "cannot be combined with --synthetic" in capsys.readouterr().err
    no_history = _with_seasons("2019-20") + ["--synthetic-history", "0"]      # later flag wins in argparse
    assert runner.main(["--model", "naive_last_season", "--out", str(tmp_path), *no_history]) == 2
    assert "no earlier history" in capsys.readouterr().err


def test_cli_missing_store_tables_point_at_ingest(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(runner, "_import_registry", lambda: None)
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path / "empty"))
    rc = runner.main(["--model", "naive_last_season", "--out", str(tmp_path), "--seasons", "2020-21"])
    assert rc == 2
    assert "ingest" in capsys.readouterr().err


def test_cli_runs_against_the_parquet_store(tmp_path, monkeypatch, tables):
    from src.store import write_table
    for name, df in tables.items():
        write_table(df, name, tmp_path / "data")
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(runner, "_import_registry", lambda: None)
    rc = runner.main(["--model", "naive_last_season", "--seasons", "2017-18:2018-19", "--n-boot", "20",
                      "--out", str(tmp_path / "out"), "--run-id", "real"])
    assert rc == 0
    md = (tmp_path / "out" / "real" / "report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC" not in md


def test_cli_adp_flag_with_store(tmp_path, monkeypatch, tables):
    from src.store import write_table
    naive = walk_forward(tables, NaiveLastSeason(), ["2017-18", "2018-19"])
    f = naive.players[naive.players["projected"]]
    idmap = pd.DataFrame({"player_id": f["player_id"].unique(), "source": "espn"})
    idmap["source_id"] = idmap["player_id"].astype(str)
    idmap["source_name"], idmap["match_method"], idmap["confidence"] = "n", "exact", 1.0
    for name, df in {**tables, "player_id_map": idmap}.items():
        write_table(df, name, tmp_path / "data")
    adp = pd.DataFrame({"season": f["season"], "source": "espn", "source_id": f["player_id"].astype(str),
                        "adp": f.groupby("season")["proj_total_fp"].rank(ascending=False)})
    adp.to_csv(tmp_path / "adp.csv", index=False)
    monkeypatch.setenv("NBA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(runner, "_import_registry", lambda: None)
    rc = runner.main(["--model", "naive_last_season", "--adp-file", str(tmp_path / "adp.csv"), "--seasons",
                      "2017-18:2018-19", "--n-boot", "20", "--out", str(tmp_path / "out"), "--run-id", "adp"])
    assert rc == 0
    md = (tmp_path / "out" / "adp" / "report.md").read_text(encoding="utf-8")
    assert "## Benchmarks" in md and "| adp" in md
    # ADP built from the naive ranking must tie the naive model on rank metrics
    assert "Paired lift of the model over each benchmark" in md


def test_cli_leakage_exit_code(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(runner, "_import_registry", lambda: None)

    def leaky(*a, **k):
        raise LeakageError("boom")

    monkeypatch.setattr(runner, "walk_forward", leaky)
    assert runner.main(["--model", "naive_last_season", "--out", str(tmp_path), *ARGS]) == 3
    assert "LEAKAGE" in capsys.readouterr().err


def test_module_entry_point_help():
    r = subprocess.run([sys.executable, "-m", "src.backtest", "--help"], cwd=REPO, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0
    assert "--model" in r.stdout and "--ablate" in r.stdout and "--synthetic" in r.stdout


def test_reports_directory_ignores_generated_output():
    assert (REPO / "reports" / ".gitkeep").exists()
    ignore = (REPO / "reports" / ".gitignore").read_text()
    assert "*" in ignore and "!.gitkeep" in ignore
