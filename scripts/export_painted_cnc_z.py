#!/usr/bin/env python3
"""Build the per-realization cluster number-count ensemble N_cl(z) for the
JXPaint benchmark painted-map suite (1000 realizations), for the nb43 covariance
notebook.

For each realization i we read the matching mock catalogue, select clusters with
``snr_true > 5`` (the same detection cut used to build the masked tSZ power
spectrum in ``compute_ps_from_painted_maps.py``), and histogram them in redshift
using the nb40 ``z_edges`` (10 bins over 0.005 < z < 1.0). The result is a
(1000, 10) array whose sample covariance across realizations pairs with the
already-cached tSZ power-spectrum ensembles.

Only the ``z`` and ``snr_true`` columns are read to keep the 1000-file pass fast.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

CAT_DIR = "/rds/rds-lxu/tsz_project/tsz_catalogue_benchmark"
CAT_TMPL = "catalogue_bench_snr_{i}.csv"
NMAPS = 1000
SNR_CUT = 5.0

# nb40 CNC convention: 10 uniform z bins over 0.005 < z < 1.0.
Z_EDGES = np.linspace(0.005, 1.0, 11)

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "nb43_painted_cnc_z.npz",
)


def main() -> None:
    n_z = len(Z_EDGES) - 1
    N_z = np.zeros((NMAPS, n_z), dtype=np.float64)
    n_qgt5 = np.zeros(NMAPS, dtype=np.int64)
    n_qgt5_zlt1 = np.zeros(NMAPS, dtype=np.int64)

    t0 = time.time()
    for i in range(NMAPS):
        path = os.path.join(CAT_DIR, CAT_TMPL.format(i=i))
        df = pd.read_csv(path, usecols=["z", "snr_true"])
        det = df["snr_true"].to_numpy() > SNR_CUT
        z_det = df["z"].to_numpy()[det]
        n_qgt5[i] = det.sum()
        counts, _ = np.histogram(z_det, bins=Z_EDGES)
        N_z[i] = counts
        n_qgt5_zlt1[i] = int(counts.sum())
        if i % 100 == 0 or i == NMAPS - 1:
            print(f"[{i:4d}] N(q>5)={n_qgt5[i]:5d} N(q>5,z<1)={n_qgt5_zlt1[i]:5d} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    z_centres = 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])
    np.savez(
        OUT,
        N_z=N_z,
        z_edges=Z_EDGES,
        z_centres=z_centres,
        n_qgt5=n_qgt5,
        n_qgt5_zlt1=n_qgt5_zlt1,
        snr_cut=SNR_CUT,
        nmaps=NMAPS,
        cat_dir=CAT_DIR,
    )
    print(f"[done] N_z shape={N_z.shape}, mean N(q>5,z<1)={n_qgt5_zlt1.mean():.1f}, "
          f"mean per bin={N_z.mean(axis=0).round(1)}")
    print(f"[done] wrote {OUT}")


if __name__ == "__main__":
    main()
