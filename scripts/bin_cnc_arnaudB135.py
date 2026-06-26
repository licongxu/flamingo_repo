"""Binned cluster number counts N(q, z) for the B=1.35 derived catalogue.

Uses the same (z, q) bin edges as the existing CNC products in data/cnc/ so the
B=1.35 counts are directly comparable to the B=1 / B=1.25 versions. Reports the
total number of clusters above q=5 (the matched-filter detection threshold).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys

_REPO = Path(__file__).resolve().parents[1]
B = float(sys.argv[1]) if len(sys.argv) > 1 else 1.35
B_TAG = f"{B:.2f}".replace(".", "")
CAT = _REPO / f"data/hydro_L2p8m9/catalogue/halo_catalogue_M500c_5e13_zlt3_y0q_arnaudB{B_TAG}.csv"
REF = _REPO / "data/cnc/observed_cnc_binned_qgt5.npz"
OUT = _REPO / f"data/cnc/observed_cnc_binned_qgt5_arnaudB{B_TAG}.npz"

ref = np.load(REF, allow_pickle=True)
z_edges = ref["bins_edges_z"]
q_edges = ref["bins_edges_q"]
z_centres = ref["z_centres"]
q_centres = ref["q_centres"]

df = pd.read_csv(CAT, comment="#")
z = df["z"].to_numpy(float)
q = df["q"].to_numpy(float)
good = np.isfinite(q)
z, q = z[good], q[good]

QCUT = 5.0
n_total_qgt5 = int(np.sum(q > QCUT))

# 2-D binned counts on the shared grid (note: top q edge is 40, so the 2-D grid
# excludes the rare q>40 clusters; the total above q=5 below counts ALL q>5).
N2d, _, _ = np.histogram2d(z, q, bins=[z_edges, q_edges])
Nz = N2d.sum(axis=1)
Nq = N2d.sum(axis=0)
n_in_grid = int(N2d.sum())
n_qgt40 = int(np.sum(q > q_edges[-1]))

np.savez(
    OUT,
    z_edges=z_edges,
    q_edges=q_edges,
    z_centres=z_centres,
    q_centres=q_centres,
    N2d=N2d,
    Nz=Nz,
    Nq=Nq,
    n_total_qgt5=n_total_qgt5,
    B=B,
)

print(f"catalogue: {CAT.name}")
print(f"TOTAL N(q>5)  = {n_total_qgt5}   (all z, all q>5 including q>40)")
print(f"  in 5<q<40 grid: {n_in_grid};  q>40: {n_qgt40}")
print()
print("N(z) per redshift bin (q>5):")
for i in range(len(z_centres)):
    print(f"  z[{z_edges[i]:.3f},{z_edges[i+1]:.3f})  N={Nz[i]:6.0f}")
print()
print("N(q) per SNR bin:")
for j in range(len(q_centres)):
    print(f"  q[{q_edges[j]:6.2f},{q_edges[j+1]:6.2f})  N={Nq[j]:6.0f}")
print()
print("N(z,q) 2-D table (rows=z bins, cols=q bins):")
hdr = "z\\q  " + " ".join(f"{q_centres[j]:7.1f}" for j in range(len(q_centres)))
print(hdr)
for i in range(len(z_centres)):
    row = " ".join(f"{N2d[i,j]:7.0f}" for j in range(len(q_centres)))
    print(f"{z_centres[i]:.3f} {row}")
print(f"\nwrote {OUT}")
