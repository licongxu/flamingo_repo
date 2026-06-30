"""Bin cluster number counts N(M_500c, z) for eight L2p8_m9 lightcones.

Reads precomputed per-lightcone (z, M) histograms from the scratch cache
(produced by streaming the full catalogues) and writes a reusable product to
``data/cnc/`` for notebooks and plotting scripts.

Selection: M_500c >= 1e13 Msun, 0 <= z <= 3.
Default grid: 15 redshift bins, 12 log-spaced M_500c bins.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
SCRATCH_CACHE = Path("/scratch/scratch-lxu/flamingo_map_build/L2p8_Mz_hist_cache")
OUT = _REPO / "data/cnc/L2p8_m9_M500c_binned_multilc.npz"

N_OBS = 8
Z_MIN, Z_MAX = 0.0, 3.0
M_MIN = 1.0e13
N_Z_BINS = 15
N_M_BINS = 12
M_LOG_MIN, M_LOG_MAX = 13.0, 15.2

Z_EDGES = np.linspace(Z_MIN, Z_MAX, N_Z_BINS + 1)
M_EDGES = np.logspace(M_LOG_MIN, M_LOG_MAX, N_M_BINS + 1)
Z_CENT = 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])
M_CENT = np.sqrt(M_EDGES[:-1] * M_EDGES[1:])


def cache_path(obs: int) -> Path:
    return SCRATCH_CACHE / f"counts_lc{obs}_Mz_z{Z_MIN}-{Z_MAX}.npy"


def load_counts_lc() -> np.ndarray:
    arrays = []
    for obs in range(N_OBS):
        path = cache_path(obs)
        if not path.exists():
            raise FileNotFoundError(
                f"missing cache {path}; run the L2p8 streaming scan first "
                "(see notebooks/33_l2p8_multilc_poisson_Mz_counts.ipynb)"
            )
        arrays.append(np.load(path))
    counts = np.stack(arrays, axis=0)
    if counts.shape != (N_OBS, N_Z_BINS, N_M_BINS):
        raise ValueError(f"unexpected shape {counts.shape}")
    return counts


def main() -> None:
    counts_lc = load_counts_lc()
    mean_2d = counts_lc.mean(axis=0)
    std_2d = counts_lc.std(axis=0, ddof=1)
    n_z_lc = counts_lc.sum(axis=2)
    n_m_lc = counts_lc.sum(axis=1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        bins_edges_z=Z_EDGES,
        bins_edges_M=M_EDGES,
        z_centres=Z_CENT,
        M_centres=M_CENT,
        counts_lc=counts_lc,
        mean_2d=mean_2d,
        std_2d=std_2d,
        Nz_lc=n_z_lc,
        Nm_lc=n_m_lc,
        Nz_mean=n_z_lc.mean(axis=0),
        Nz_std=n_z_lc.std(axis=0, ddof=1),
        Nm_mean=n_m_lc.mean(axis=0),
        Nm_std=n_m_lc.std(axis=0, ddof=1),
        n_lightcones=N_OBS,
        M_min=M_MIN,
        z_min=Z_MIN,
        z_max=Z_MAX,
        catalogue=(
            "halo_catalogue_M500c_1e13_zlt3_L2p8_m9_yang26rot.csv "
            "from /rds/rds-lxu/flamingo/L2p8_m9/lightcone{0..7}/catalogues/"
        ),
    )

    print(f"wrote {OUT}")
    print(f"  shape counts_lc = {counts_lc.shape}")
    print(f"  total per lc    = {counts_lc.sum(axis=(1, 2))}")
    print(f"  z bins: {N_Z_BINS} over [{Z_MIN}, {Z_MAX}]")
    print(f"  M bins: {N_M_BINS} log-spaced [{M_EDGES[0]:.2e}, {M_EDGES[-1]:.2e}] Msun")


if __name__ == "__main__":
    main()
