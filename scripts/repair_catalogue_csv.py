#!/usr/bin/env python3
"""Truncate catalogue CSV to rows_written from progress.json (drop partial snap)."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path


def repair(csv: Path, backup: bool = True) -> None:
    prog = csv.with_suffix(csv.suffix + ".progress.json")
    if not prog.exists():
        raise SystemExit(f"no progress file: {prog}")
    meta = json.loads(prog.read_text())
    target = int(meta["rows_written"])
    tmp = csv.with_suffix(csv.suffix + ".repair.tmp")

    data_rows = 0
    kept = 0
    with csv.open() as fin, tmp.open("w") as fout:
        for line in fin:
            if line.startswith("#") or line.startswith("snap,"):
                fout.write(line)
                continue
            data_rows += 1
            if data_rows <= target:
                fout.write(line)
                kept += 1

    if data_rows == target:
        print(f"{csv}: OK ({target:,} data rows, nothing to trim)")
        tmp.unlink(missing_ok=True)
        return

    if backup:
        bak = csv.with_suffix(csv.suffix + ".pre_repair.bak")
        if not bak.exists():
            print(f"backup -> {bak}")
            shutil.copy2(csv, bak)

    tmp.replace(csv)
    print(f"{csv}: trimmed {data_rows:,} -> {kept:,} data rows (progress rows_written={target:,})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path)
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args()
    repair(args.csv, backup=not args.no_backup)
