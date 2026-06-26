"""Binned N(z, q) for B=1.35 Y_R500c catalogue restricted to z < 0.35 (unrotated map volume).

Uses 7 equal-width z bins from 0.005 to 0.35 (matching the unrotated lightcone shells).
q bins match the standard CNC grid (5–40, log-spaced, 5 bins).

Run:
    python scripts/bin_cnc_arnaudB135_Y500c_unrot_z0p35.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
B = 1.35
B_TAG = "135"
Z_MIN, Z_MAX = 0.005, 0.35
N_Z_BINS = 7

CAT = _REPO / f"data/hydro_L2p8m9/catalogue/halo_catalogue_M500c_5e13_zlt3_y0q_arnaudB{B_TAG}_Y500c.csv"
REF = _REPO / "data/cnc/observed_cnc_binned_qgt5.npz"
OUT_NPZ = _REPO / f"data/cnc/observed_cnc_binned_qgt5_arnaudB{B_TAG}_Y500c_unrot_z0p35.npz"
OUT_TXT = _REPO / f"data/cnc/N2d_z_q_bin_arnaudB{B_TAG}_Y500c_unrot_z0p35.txt"

ref = np.load(REF, allow_pickle=True)
q_edges = ref["bins_edges_q"]
q_centres = ref["q_centres"]
z_edges = np.linspace(Z_MIN, Z_MAX, N_Z_BINS + 1)
z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])

df = pd.read_csv(CAT, comment="#")
z = df["z"].to_numpy(float)
q = df["q"].to_numpy(float)
good = np.isfinite(q) & (z < Z_MAX)
z, q = z[good], q[good]

QCUT = 5.0
n_total_qgt5 = int(np.sum(q > QCUT))

N2d, _, _ = np.histogram2d(z, q, bins=[z_edges, q_edges])
Nz = N2d.sum(axis=1)
Nq = N2d.sum(axis=0)
n_in_grid = int(N2d.sum())
n_qgt40 = int(np.sum(q > q_edges[-1]))

np.savez(
    OUT_NPZ,
    z_edges=z_edges,
    q_edges=q_edges,
    z_centres=z_centres,
    q_centres=q_centres,
    N2d=N2d,
    Nz=Nz,
    Nq=Nq,
    n_total_qgt5=n_total_qgt5,
    B=B,
    aperture="Y500c",
    r_out=1.0,
    z_max=Z_MAX,
    map_tag="unrot_z0p35",
)
np.savetxt(OUT_TXT, N2d, fmt="%d")

print(f"catalogue: {CAT.name}  (z < {Z_MAX})")
print(f"TOTAL N(q>5)  = {n_total_qgt5}   (z < {Z_MAX}, all q>5 including q>40)")
print(f"  in 5<q<40 grid: {n_in_grid};  q>40: {n_qgt40}")
print()
print("N(z,q) 2-D table (rows=z bins, cols=q bins):")
hdr = "z\\q  " + " ".join(f"{q_centres[j]:7.1f}" for j in range(len(q_centres)))
print(hdr)
for i in range(len(z_centres)):
    row = " ".join(f"{N2d[i, j]:7.0f}" for j in range(len(q_centres)))
    print(f"{z_centres[i]:.3f} {row}")
print(f"\nwrote {OUT_NPZ}")
print(f"wrote {OUT_TXT}")
