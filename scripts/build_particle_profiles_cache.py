"""Cache particle-based 3-D gas profiles (pressure, temperature, density) of FLAMINGO
clusters to a local CSV.

Literature-standard method: read the gas particles around each halo and bin them in
3-D radius. We use the FLAMINGO **full** snapshot (snap 0078, z=0) streamed via
hdfstream, with SWIFT cell metadata for fast spatial selection (no swiftsimio needed);
halo centres / R500c / M500c come from SOAP. Each halo is fully resolved, so a few
hundred clusters give a well-sampled stacked profile.

For each halo with M500c > 5e13 Msun we record, in log radial bins x = r/R500c:
  - electron-pressure proxy  P_e(x) = <n_e T>  (volume-weighted)
  - temperature              T(x)   = <T>      (mass-weighted)
  - gas mass per bin         (-> density rho(x) = sum m / shell volume)
  - particle count.

Run:
    python scripts/build_particle_profiles_cache.py
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
import hdfstream

BASE = "FLAMINGO/L2p8_m9/L2p8_m9"
SOAP = f"{BASE}/SOAP-HBT/halo_properties_0078.hdf5"
SNAP = f"{BASE}/snapshots/flamingo_0078/flamingo_0078.hdf5"
M500_MIN = 5.0e13       # Msun (matches the catalogue cut)
N_SAMPLE = 500
SEED = 0
X_EDGES = np.logspace(np.log10(0.05), np.log10(5.0), 13)   # 12 log radial bins
NB = len(X_EDGES) - 1
OUT = ("/scratch/scratch-lxu/flamingo_repo/data/hydro_L2p8m9/catalogue/"
       "particle_gas_profiles_snap0078.csv")


def cells_within(ctr, rmax, cellsz, dim, box):
    """(ix,iy,iz) of cells whose AABB lies within rmax of ctr (periodic)."""
    ci = [int(np.floor((ctr[d] % box) / cellsz)) % dim for d in range(3)]
    span = int(np.ceil(rmax / cellsz))
    out = []
    for o in itertools.product(range(-span, span + 1), repeat=3):
        c = [(ci[d] + o[d]) % dim for d in range(3)]
        dmin2 = 0.0
        for d in range(3):
            lo = c[d] * cellsz
            delta = ((ctr[d] - lo) + box / 2) % box - box / 2
            if delta < 0:
                dmin2 += delta ** 2
            elif delta > cellsz:
                dmin2 += (delta - cellsz) ** 2
        if dmin2 < rmax ** 2:
            out.append((c[0], c[1], c[2]))
    return out


def main() -> None:
    root = hdfstream.open("cosma", "/")
    soap = root[SOAP]
    M500 = np.asarray(soap["SO/500_crit/TotalMass"][:], dtype=np.float64) * 1e10  # full (selection)
    idx = np.where(M500 > M500_MIN)[0]
    rng = np.random.default_rng(SEED)
    samp = np.sort(rng.choice(idx, size=min(N_SAMPLE, idx.size), replace=False))
    # point-fetch only the sampled halos' centres / radii (avoid full 9 GB reads)
    ctrs = np.asarray(soap["InputHalos/HaloCentre"][samp], dtype=np.float64)
    R500s = np.asarray(soap["SO/500_crit/SORadius"][samp], dtype=np.float64)
    M500s = M500[samp]
    print(f"{idx.size} halos with M500c>{M500_MIN:.0e}; profiling N={samp.size}", flush=True)

    snap = root[SNAP]
    md = dict(snap["Cells/Meta-data"].attrs)
    cellsz = float(md["size"][0]); dim = int(md["dimension"][0])
    box = float(np.asarray(snap["Header"].attrs["BoxSize"])[0])
    cen = np.asarray(snap["Cells/Centres"][:], dtype=np.float64)
    off = np.asarray(snap["Cells/OffsetsInFile/PartType0"][:]).astype(np.int64)
    cnt = np.asarray(snap["Cells/Counts/PartType0"][:]).astype(np.int64)
    files = np.asarray(snap["Cells/Files/PartType0"][:]).astype(np.int64)
    # global offset into the (virtual) concatenated dataset: file_start + offset_in_file
    nfile = files.max() + 1
    file_nparts = np.array([(off[files == f] + cnt[files == f]).max() if (files == f).any() else 0
                            for f in range(nfile)], dtype=np.int64)
    file_start = np.concatenate([[0], np.cumsum(file_nparts)[:-1]])
    goff = file_start[files] + off
    # (ix,iy,iz) -> cell array index, from the cell centres
    ijk = np.round(cen / cellsz - 0.5).astype(int) % dim
    lookup = {(int(a), int(b), int(c)): k for k, (a, b, c) in enumerate(ijk)}

    CO = snap["PartType0/Coordinates"]; NE = snap["PartType0/ElectronNumberDensities"]
    TE = snap["PartType0/Temperatures"]; MA = snap["PartType0/Masses"]
    DE = snap["PartType0/Densities"]

    rows = []
    t0 = time.time()
    for j in range(samp.size):
        ctr, R500, M = ctrs[j], float(R500s[j]), float(M500s[j])
        rmax = 5.0 * R500
        rr, neT, TT, mm, vv = [], [], [], [], []
        for ix, iy, iz in cells_within(ctr, rmax, cellsz, dim, box):
            k = lookup.get((ix, iy, iz))
            if k is None or cnt[k] == 0:
                continue
            o, n = int(goff[k]), int(cnt[k])
            co = np.asarray(CO[o:o + n], dtype=np.float64)
            d = co - ctr; d -= box * np.round(d / box)
            r = np.sqrt((d ** 2).sum(1))
            sel = r < rmax
            if not sel.any():
                continue
            ne = np.asarray(NE[o:o + n])[sel]; tt = np.asarray(TE[o:o + n])[sel]
            m = np.asarray(MA[o:o + n])[sel]; dd = np.asarray(DE[o:o + n])[sel]
            rr.append(r[sel] / R500); neT.append(ne * tt); TT.append(tt)
            mm.append(m); vv.append(m / dd)
        if not rr:
            continue
        x = np.concatenate(rr); P = np.concatenate(neT); T = np.concatenate(TT)
        Mp = np.concatenate(mm); V = np.concatenate(vv)
        ib = np.digitize(x, X_EDGES) - 1
        row = {"M500c_Msun": M, "R500c_Mpc": R500}
        for b in range(NB):
            s = ib == b
            sv = float(np.sum(V[s])); sm = float(np.sum(Mp[s]))
            row[f"Pe_{b}"] = float(np.sum(P[s] * V[s]) / sv) if sv > 0 else np.nan
            row[f"T_{b}"] = float(np.sum(T[s] * Mp[s]) / sm) if sm > 0 else np.nan
            row[f"Msum_{b}"] = sm
            row[f"N_{b}"] = int(s.sum())
        rows.append(row)
        if (j + 1) % 50 == 0:
            print(f"  {j + 1}/{samp.size}  ({time.time() - t0:.0f}s, {len(rows)} kept)")

    df = pd.DataFrame(rows)
    header = (
        "# Particle-based 3-D gas profiles, FLAMINGO L2p8_m9 full snapshot 0078 (z=0).\n"
        f"# M500c>{M500_MIN:.0e} Msun, random sample N={len(rows)} (seed={SEED}).\n"
        f"# Radial bins x=r/R500c, log edges {X_EDGES[0]:.3f}..{X_EDGES[-1]:.1f} ({NB} bins).\n"
        "# Pe_b=<n_e T> (vol-weighted, arb units); T_b=<T> (mass-weighted, K);\n"
        "# Msum_b=gas mass in bin (1e10 Msun units); N_b=particle count.\n"
    )
    with open(OUT, "w") as f:
        f.write(header)
        df.to_csv(f, index=False)
    print(f"wrote {OUT}  ({len(rows)} halos, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
