"""Convert catalogue CSVs to Parquet next to the originals.

Usage::

    python scripts/convert_parquet.py [--index data/index.json] [--only SUBSTRING]

The API automatically prefers ``<name>.parquet`` when it exists, which
makes queries on the 9.9 GB L2p8 catalogues interactive. DuckDB streams
the conversion, so memory stays modest regardless of file size. Parquet
files are written to the data tree (not the repo).
"""

import argparse
import json
from pathlib import Path

import duckdb

WEBSITE_DIR = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=WEBSITE_DIR / "data" / "index.json")
    ap.add_argument("--only", default="", help="only catalogues whose relpath contains this")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    with open(args.index) as fh:
        index = json.load(fh)
    data_root = Path(index["data_root"])

    cats = [
        f
        for f in index["files"]
        if f["kind"] == "catalogue"
        and f["filename"].endswith(".csv")
        and args.only in f["relpath"]
    ]
    for f in cats:
        csv_path = data_root / f["relpath"]
        pq_path = csv_path.with_suffix(".parquet")
        if pq_path.exists() and not args.force:
            print(f"exists: {pq_path}")
            continue
        print(f"converting {csv_path} ({f['size_bytes'] / 1e9:.1f} GB)")
        target = str(pq_path).replace("'", "''")
        duckdb.connect().execute(
            "COPY (SELECT * FROM read_csv(?, header=true, comment='#')) "
            f"TO '{target}' (FORMAT parquet, COMPRESSION zstd)",
            [str(csv_path)],
        )
        print(f"  -> {pq_path} ({pq_path.stat().st_size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
