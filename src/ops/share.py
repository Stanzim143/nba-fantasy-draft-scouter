"""Hand the project to someone else: bundle / restore the processed data, set up a virtualenv.

`python -m src.ops.share pack [out.zip]`   zip `<data_dir>/processed` (parquet + json only)
`python -m src.ops.share unpack <zip>`     restore it into `<data_dir>/processed`

The pack deliberately leaves out anything private or licensed: the ESPN league snapshot
(`espn_league/`, which names the league), `*.bak*` files, and `external_rankings/` (manually exported
Yahoo / FantasyPros files) and `raw/` are not under `processed/` at all. It is meant to be sent
privately, never committed (`*.zip` is git-ignored).
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from src.contracts import data_dir

KEEP_SUFFIXES = {".parquet", ".json"}
SKIP_DIRS = {"espn_league"}


def _packable(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    return (
        path.is_file()
        and path.suffix in KEEP_SUFFIXES
        and ".bak" not in path.name
        and not (set(rel.parts[:-1]) & SKIP_DIRS)
    )


def pack(out: Path, root: Path | None = None) -> list[str]:
    root = (root or data_dir()) / "processed"
    if not root.is_dir():
        raise SystemExit(f"no processed data at {root}; run the ingest first")
    files = sorted(p for p in root.rglob("*") if _packable(p, root))
    if not files:
        raise SystemExit(f"nothing to pack under {root}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, p.relative_to(root).as_posix())
    return [p.relative_to(root).as_posix() for p in files]


def unpack(archive: Path, root: Path | None = None, force: bool = False) -> int:
    dest = (root or data_dir()) / "processed"
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        for name in names:  # refuse path traversal and non-data files
            target = (dest / name).resolve()
            if dest.resolve() not in target.parents or Path(name).suffix not in KEEP_SUFFIXES:
                raise SystemExit(f"refusing unexpected entry in archive: {name}")
        clashes = [n for n in names if (dest / n).exists()]
        if clashes and not force:
            raise SystemExit(f"{len(clashes)} file(s) already exist in {dest} (e.g. {clashes[0]}); use --force")
        zf.extractall(dest)
    return len(names)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.ops.share", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack", help="zip the processed data for sending privately")
    p.add_argument("out", nargs="?", default="nba-data-pack.zip", type=Path)
    u = sub.add_parser("unpack", help="restore a data pack into the data directory")
    u.add_argument("archive", type=Path)
    u.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args(argv)
    if args.cmd == "pack":
        names = pack(args.out)
        print(f"packed {len(names)} files -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"restored {unpack(args.archive, force=args.force)} files into {data_dir() / 'processed'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
