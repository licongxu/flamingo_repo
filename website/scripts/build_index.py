"""Scan the FLAMINGO data tree and write the ``index.json`` manifest.

Usage::

    python scripts/build_index.py [--root /rds/rds-lxu/flamingo] [--out data/index.json]

Build artifacts (``*.progress.json``, ``*.pre_repair.bak``) are skipped.
Re-running is idempotent: file ids are a hash of the relative path.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
from pathlib import Path

DEFAULT_ROOT = "/rds/rds-lxu/flamingo"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "index.json"

SKIP_SUFFIXES = (".progress.json", ".pre_repair.bak")

# halo_catalogue_M500c_<cut>_zlt3_<variant>_yang26rot[_qfrommz].csv
CAT_RE = re.compile(
    r"^halo_catalogue_M500c_(?P<cut>[0-9e]+)_zlt(?P<zmax>[0-9p.]+)_"
    r"(?P<variant>.+?)_yang26rot(?P<qfrommz>_qfrommz)?\.(csv|parquet)$"
)
# y_unlensed_<variant>_lc<N>[_nside<n>].fits
YMAP_RE = re.compile(
    r"^y_unlensed_(?P<variant>.+?)_lc(?P<lc>\d+)(_nside(?P<nside>\d+))?\.fits$"
)
# comptonY_L2p8_m9_lc<N>_nside<n>_(shell<idx>|rotgroup<g>_shells<a>-<b>).fits
SHELL_RE = re.compile(
    r"^comptonY_(?P<variant>.+?)_lc(?P<lc>\d+)_nside(?P<nside>\d+)_"
    r"(shell(?P<shell>\d+)|rotgroup(?P<group>\d+)_shells(?P<sh0>\d+)-(?P<sh1>\d+))\.fits$"
)
LIGHTCONE_DIR_RE = re.compile(r"^lightcone(\d+)$")


def file_id(relpath: str) -> str:
    return hashlib.sha1(relpath.encode()).hexdigest()[:12]


def classify(relpath: Path) -> dict | None:
    """Parse one relative path into a manifest entry (metadata part only)."""
    name = relpath.name
    if any(name.endswith(s) for s in SKIP_SUFFIXES):
        return None
    if len(relpath.parts) < 2:
        return None
    simulation = relpath.parts[0]
    lightcone = None
    for part in relpath.parts[1:-1]:
        m = LIGHTCONE_DIR_RE.match(part)
        if m:
            lightcone = int(m.group(1))

    base = {
        "simulation": simulation,
        "lightcone": lightcone,
        "variant": "fiducial",
        "nside": None,
        "mass_cut": None,
        "shell": None,
        "shell_range": None,
        "rotation_group": None,
    }

    m = CAT_RE.match(name)
    if m:
        variant = m.group("variant")
        return {
            **base,
            "kind": "catalogue",
            "variant": "fiducial" if variant == simulation else variant,
            "mass_cut": m.group("cut"),
            "has_q": bool(m.group("qfrommz")),
        }
    m = YMAP_RE.match(name)
    if m:
        variant = m.group("variant")
        return {
            **base,
            "kind": "map",
            "variant": "fiducial" if variant == simulation else variant,
            "lightcone": int(m.group("lc")),
            # The y_unlensed maps without _nside in the name are nside 4096
            # (12 * 4096^2 float64 pixels = 1.6 GB, matching the file size).
            "nside": int(m.group("nside")) if m.group("nside") else 4096,
        }
    m = SHELL_RE.match(name)
    if m:
        entry = {
            **base,
            "kind": "map",
            "lightcone": int(m.group("lc")),
            "nside": int(m.group("nside")),
        }
        if m.group("shell") is not None:
            entry["shell"] = int(m.group("shell"))
        else:
            entry["rotation_group"] = int(m.group("group"))
            entry["shell_range"] = f"{m.group('sh0')}-{m.group('sh1')}"
        return entry
    if name.endswith(".json"):
        return {**base, "kind": "aux"}
    if name.endswith((".fits", ".csv", ".parquet", ".hdf5", ".h5", ".npz")):
        return {**base, "kind": "other"}
    return None


def build_index(root: Path) -> dict:
    root = Path(root)
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # Parquet conversions of an existing CSV are an internal acceleration
        # artifact, not a second catalogue.
        if path.suffix == ".parquet" and path.with_suffix(".csv").is_file():
            continue
        rel = path.relative_to(root)
        entry = classify(rel)
        if entry is None:
            continue
        st = path.stat()
        entry.update(
            id=file_id(str(rel)),
            relpath=str(rel),
            filename=path.name,
            size_bytes=st.st_size,
            mtime_utc=datetime.datetime.fromtimestamp(
                st.st_mtime, tz=datetime.timezone.utc
            ).isoformat(timespec="seconds"),
        )
        files.append(entry)
    return {
        "generated_utc": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "data_root": str(root),
        "n_files": len(files),
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    if not args.root.is_dir():
        print(f"error: data root {args.root} not found", file=sys.stderr)
        return 1
    index = build_index(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(index, fh, indent=1)
    os.replace(tmp, args.out)  # atomic: a running server never sees a partial file
    total_gb = sum(f["size_bytes"] for f in index["files"]) / 1e9
    print(f"wrote {args.out}: {index['n_files']} files, {total_gb:.1f} GB indexed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
