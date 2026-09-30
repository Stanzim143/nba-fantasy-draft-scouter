import io
import sys

import pytest

from src.inseason._console import use_utf8_console


def test_utf8_console_keeps_accented_names(monkeypatch):
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252", errors="strict"))
    use_utf8_console()
    print("Jokić Marković")
    sys.stdout.flush()
    assert raw.getvalue().decode("utf-8").strip() == "Jokić Marković"


def test_utf8_console_ignores_streams_that_cannot_be_reconfigured(monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", None)
    use_utf8_console()


@pytest.mark.parametrize("mod", ["ros", "ros_eval", "schedule", "signals", "trade", "waivers"])
def test_every_inseason_cli_sets_it_up(mod):
    import importlib
    import inspect

    assert "use_utf8_console()" in inspect.getsource(importlib.import_module(f"src.inseason.{mod}").main)
