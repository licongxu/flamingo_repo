"""Build the full L2p8_m9 lc0 Compton-y map with NO yang26 shell rotation.

Same source and shell coverage as ``y_unlensed_L2p8_m9_lc0.fits`` (shells 0..59,
z < 3, NSIDE=4096) but each shell is summed directly without alm-space rotation.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import healpy as hp
import hdfstream
import numpy as np

DEFAULT_SHELL_MAX = 60  # shells 0..59, z < 3 (matches rotated lc0 product)


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


def build(run: str, observer: int, nside: int, shell_max: int, out_path: Path) -> None:
    npix = 12 * nside * nside
    root = hdfstream.open("cosma", "/")
    base = f"FLAMINGO/{run}/{run}/healpix_maps/nside_{nside}/lightcone{observer}_shells"
    y = np.zeros(npix, dtype=np.float64)

    print(
        f"Building unrotated y-map: run={run} lc{observer} "
        f"shells=0..{shell_max - 1} NSIDE={nside}",
        flush=True,
    )
    t_all = time.time()
    for i in range(shell_max):
        path = f"{base}/shell_{i}/lightcone{observer}.shell_{i}.0.hdf5"
        t0 = time.time()
        m = _stream(root, path)
        y += m
        print(
            f"shell_{i}: sum={m.sum():.4f}  cumulative={y.sum():.4f}  "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    hp.write_map(str(out_path), y, nest=False, overwrite=True, dtype=np.float64)
    print(f"\nWrote {out_path}", flush=True)
    print(
        f"  NSIDE={nside} RING  mean(y)={y.mean():.6e}  "
        f"min={y.min():.3e} max={y.max():.3e}",
        flush=True,
    )
    print(
        f"  sum(y)={y.sum():.6f}  finite={np.all(np.isfinite(y))}  "
        f"total_time={time.time() - t_all:.0f}s",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="L2p8_m9")
    p.add_argument("--observer", type=int, default=0)
    p.add_argument("--nside", type=int, default=4096)
    p.add_argument("--shell-max", type=int, default=DEFAULT_SHELL_MAX)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/scratch/scratch-lxu/flamingo_repo/data/hydro_L2p8m9/map/"
            "y_unlensed_L2p8_m9_lc0_unrotated.fits"
        ),
    )
    args = p.parse_args()
    build(args.run, args.observer, args.nside, args.shell_max, args.out)


if __name__ == "__main__":
    main()
