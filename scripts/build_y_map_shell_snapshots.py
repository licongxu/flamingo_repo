"""Stream and store per-shell unlensed Compton-y maps (NSIDE=4096) from FLAMINGO.

Writes ``comptonY_{run}_lc{observer}_nside{nside}_shell{idx}.fits`` under the
output directory and a JSON sidecar mapping each shell to overlapping simulation
snapshots (see ``flamingo_shells``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import healpy as hp
import hdfstream
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flamingo_shells import shell_snapshot_table  # noqa: E402


def _stream(root, path: str, retries: int = 3) -> np.ndarray:
    for attempt in range(1, retries + 1):
        try:
            return root[path]["ComptonY"][...]
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                raise
            wait = 30 * attempt
            print(f"    stream error ({exc!r}); retry {attempt} in {wait}s", flush=True)
            time.sleep(wait)


def _healpix_base(run: str, nside: int, observer: int) -> str:
    return (
        f"FLAMINGO/{run}/{run}/healpix_maps/nside_{nside}/"
        f"lightcone{observer}_shells"
    )


def build_shells(
    run: str,
    observer: int,
    shell_indices: list[int],
    out_dir: Path,
    *,
    nside: int = 4096,
    skip_existing: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = hdfstream.open("cosma", "/")
    base = _healpix_base(run, nside, observer)

    for i in shell_indices:
        out_path = out_dir / f"comptonY_{run}_lc{observer}_nside{nside}_shell{i}.fits"
        if skip_existing and out_path.exists():
            print(f"shell_{i}: {out_path.name} exists, skip", flush=True)
            continue

        path = f"{base}/shell_{i}/lightcone{observer}.shell_{i}.0.hdf5"
        t0 = time.time()
        print(f"shell_{i}: streaming {path} …", flush=True)
        m = _stream(root, path)
        hp.write_map(str(out_path), m, nest=False, overwrite=True, dtype=np.float64)
        print(
            f"shell_{i}: wrote {out_path.name}  sum={m.sum():.4f}  "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        del m

    meta_path = out_dir / f"shell_snapshots_{run}_lc{observer}.json"
    meta = {
        "run": run,
        "observer": observer,
        "nside": nside,
        "layout": "L2P8",
        "shells": shell_snapshot_table(shell_indices),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {meta_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="L2p8_m9")
    p.add_argument("--observer", type=int, default=0)
    p.add_argument("--nside", type=int, default=4096)
    p.add_argument(
        "--shells",
        type=int,
        nargs="+",
        required=True,
        help="Shell indices to download (e.g. 17 18 19 20 21).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/rds/rds-lxu/flamingo/L2p8_m9/lightcone0/healpix_map/snapshots"),
    )
    p.add_argument("--force", action="store_true", help="Re-download even if FITS exists.")
    args = p.parse_args()
    build_shells(
        args.run,
        args.observer,
        args.shells,
        args.out_dir,
        nside=args.nside,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
