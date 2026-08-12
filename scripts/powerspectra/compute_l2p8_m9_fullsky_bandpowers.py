#!/usr/bin/env python3
"""Compute the fiducial L2p8_m9 full-sky 18-bin tSZ bandpowers."""

from pathlib import Path

import healpy as hp
import numpy as np

from flamingo.powerspectra.bandpowers import (
    PLANCK_ELL_EFF,
    PLANCK_ELL_MAX,
    bin_planck_dl,
)

MAP_FILE = Path("data/hydro_L2p8m9/map/y_unlensed_L2p8_m9_lc0.fits")
OUTPUT = Path(
    "data_paper/binned_bandpowers/Dl_yy_L2p8_m9_fullsky_binned_18.txt"
)
def main() -> None:
    sky_map = hp.read_map(MAP_FILE, dtype=np.float64)
    nside = hp.npix2nside(sky_map.size)
    cl = hp.anafast(sky_map - np.mean(sky_map), lmax=int(PLANCK_ELL_MAX[-1]))
    cl /= hp.pixwin(nside, lmax=int(PLANCK_ELL_MAX[-1])) ** 2
    ell = np.arange(cl.size)
    binned = bin_planck_dl(ell, cl)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        OUTPUT,
        np.column_stack([PLANCK_ELL_EFF, 1e12 * binned]),
        header="L2p8_m9 fullsky; uniform mean over inclusive Planck bins\n"
        "ell_eff  1e12_D_ell_yy",
    )
    print(f"nside={nside}; wrote {OUTPUT}")
    print(np.column_stack([PLANCK_ELL_EFF, 1e12 * binned]))


if __name__ == "__main__":
    main()
