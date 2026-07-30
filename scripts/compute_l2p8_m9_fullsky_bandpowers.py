#!/usr/bin/env python3
"""Compute the fiducial L2p8_m9 full-sky 18-bin tSZ bandpowers."""

from pathlib import Path

import healpy as hp
import numpy as np


MAP_FILE = Path("data/hydro_L2p8m9/map/y_unlensed_L2p8_m9_lc0.fits")
OUTPUT = Path(
    "data_paper/binned_bandpowers/Dl_yy_L2p8_m9_fullsky_binned_18.txt"
)
ELL_MIN = np.array(
    [9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835]
)
ELL_MAX = np.array(
    [12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835, 1085]
)
ELL_EFF = np.array(
    [
        10.0, 13.5, 18.0, 23.5, 30.5, 40.0, 52.5, 68.5, 89.5,
        117.0, 152.5, 198.0, 257.5, 335.5, 436.5, 567.5, 738.0, 959.5,
    ]
)


def main() -> None:
    sky_map = hp.read_map(MAP_FILE, dtype=np.float64)
    nside = hp.npix2nside(sky_map.size)
    cl = hp.anafast(sky_map - np.mean(sky_map), lmax=int(ELL_MAX[-1]))
    cl /= hp.pixwin(nside, lmax=int(ELL_MAX[-1])) ** 2
    ell = np.arange(cl.size)
    dl = ell * (ell + 1.0) * cl / (2.0 * np.pi)
    binned = np.array(
        [
            np.mean(dl[(ell >= ell_min) & (ell <= ell_max)])
            for ell_min, ell_max in zip(ELL_MIN, ELL_MAX, strict=True)
        ]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        OUTPUT,
        np.column_stack([ELL_EFF, 1e12 * binned]),
        header="L2p8_m9 fullsky; uniform mean over inclusive Planck bins\n"
        "ell_eff  1e12_D_ell_yy",
    )
    print(f"nside={nside}; wrote {OUTPUT}")
    print(np.column_stack([ELL_EFF, 1e12 * binned]))


if __name__ == "__main__":
    main()
