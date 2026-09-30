from src import cli


def test_no_args_prints_usage_and_fails(capsys):
    assert cli.main([]) == 2
    assert "nba-fantasy bootstrap" in capsys.readouterr().err


def test_help_succeeds():
    assert cli.main(["--help"]) == 0


def test_bootstrap_dispatches_to_dry_run(capsys):
    assert cli.main(["bootstrap", "--dry-run"]) == 0
    assert "data dir:" in capsys.readouterr().out


def test_entry_point_is_declared():
    import tomllib
    from pathlib import Path

    cfg = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    assert cfg["project"]["scripts"]["nba-fantasy"] == "src.cli:main"
