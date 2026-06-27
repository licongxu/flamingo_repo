"""Build the un-lensed Compton-y map from ONLY the unrotated lightcone shells.

For the L2p8 boxes the FLAMINGO yang26 rotation frame is the identity for the
first seven lightcone shells (shells 0..6, theta=phi=0 in ANGLES_L2P8); rotation
first turns on at shell 7. Those seven shells tile comoving radius 3..1412 Mpc,
i.e. redshift z = 0 .. 0.35 in equal dz=0.05 bins. Because every shell here is
identity, the integrated map is a plain per-shell sum: no SHT, no alm rotation.

Streams per-shell ``ComptonY`` (RING, dimensionless) from the FLAMINGO portal via
hdfstream and writes the summed map. Default NSIDE=4096.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import healpy as hp
import hdfstream
import numpy as np

# Shells with identity (theta=phi=0) yang26 rotation for L2p8: 0..6 inclusive.
UNROTATED_SHELLS = range(0, 7)


def _stream(root, path, retries=3):
    for attempt in range(1, retries + 1):
        try:
            return root[path]
        except Exception as exc:  # noqa: BLE001 - broad network errors
            if attempt == retries:
                raise
            wait = 30 * attempt
            print(f"    stream error ({exc!r}); retry {attempt} in {wait}s", flush=True)
            time.sleep(wait)


def build(run, observer, nside, out_path):
    npix = 12 * nside * nside
    root = hdfstream.open("cosma", "/")
    base = f"FLAMINGO/{run}/{run}/healpix_maps/nside_{nside}/lightcone{observer}_shells"
    y = np.zeros(npix, dtype=np.float64)
    for i in UNROTATED_SHELLS:
        f = _stream(root, f"{base}/shell_{i}/lightcone{observer}.shell_{i}.0.hdf5")
        cy = f["ComptonY"]
        attrs = dict(cy.attrs)
        r_in = float(attrs["comoving_inner_radius"][0])
        r_out = float(attrs["comoving_outer_radius"][0])
        exp = float(attrs["expected_sum"][0])
        t0 = time.time()
        m = cy[...]
        y += m
        print(f"shell_{i}: r=[{r_in:.1f},{r_out:.1f}] Mpc  sum={m.sum():.3f} "
              f"(expected {exp:.3f}, ratio {m.sum()/exp:.5f})  {time.time()-t0:.0f}s",
              flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hp.write_map(str(out_path), y, nest=False, overwrite=True, dtype=np.float64)
    print(f"\nWrote {out_path}\n  NSIDE={nside} RING  mean(y)={y.mean():.6e}  "
          f"min={y.min():.3e} max={y.max():.3e}  finite={np.all(np.isfinite(y))}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="L2p8_m9")
    p.add_argument("--observer", type=int, default=0)
    p.add_argument("--nside", type=int, default=4096)
    p.add_argument("--out", type=Path, default=Path(
        "/scratch/scratch-lxu/flamingo_repo/data/map/"
        "y_unlensed_L2p8_m9_lc0_unrotated_z0_0p35_nside4096.fits"))
    a = p.parse_args()
    build(a.run, a.observer, a.nside, a.out)


if __name__ == "__main__":
    main()
