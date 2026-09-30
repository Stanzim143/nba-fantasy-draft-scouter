"""Data-pack round trip: private / licensed / backup files never enter the pack."""
import zipfile

import pytest

from src.ops import share


def _tree(root):
    proc = root / "processed"
    (proc / "espn_league").mkdir(parents=True)
    (proc / "players.parquet").write_bytes(b"p")
    (proc / "report.json").write_text("{}")
    (proc / "players.parquet.bak_pre_rookies").write_bytes(b"b")
    (proc / "espn_league" / "123_2027.json").write_text("{}")
    (proc / "notes.txt").write_text("x")
    (root / "external_rankings").mkdir()
    (root / "external_rankings" / "yahoo.xlsx").write_bytes(b"y")


def test_pack_keeps_only_shareable_tables(tmp_path):
    src, out = tmp_path / "src", tmp_path / "pack.zip"
    _tree(src)
    assert sorted(share.pack(out, root=src)) == ["players.parquet", "report.json"]
    assert sorted(zipfile.ZipFile(out).namelist()) == ["players.parquet", "report.json"]


def test_unpack_round_trip_refuses_overwrite_without_force(tmp_path):
    src, dst, out = tmp_path / "src", tmp_path / "dst", tmp_path / "pack.zip"
    _tree(src)
    share.pack(out, root=src)
    assert share.unpack(out, root=dst) == 2
    assert (dst / "processed" / "players.parquet").read_bytes() == b"p"
    with pytest.raises(SystemExit):
        share.unpack(out, root=dst)
    assert share.unpack(out, root=dst, force=True) == 2


def test_unpack_rejects_path_traversal_and_foreign_files(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil.parquet", "x")
    with pytest.raises(SystemExit):
        share.unpack(bad, root=tmp_path / "dst")
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("run.py", "x")
    with pytest.raises(SystemExit):
        share.unpack(bad, root=tmp_path / "dst")
