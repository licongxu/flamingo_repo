#!/usr/bin/env python3
"""Per-patch joint ensemble [N_z(q>5), D_ell(full sky), D_ell(masked q>5)] on the
L1_m9 fiducial y-map, for the nb43 CNC x tSZ-PS covariance via patch resampling.

The sky is split into the 192 HEALPix ``NSIDE_PATCH=4`` superpixels (~215 deg^2
each).  For each patch we measure

* ``N_z``      -- number counts of q_from_mz > 5, z < 1 clusters in the nb40
                  redshift bins (10 bins over 0.005 < z < 1),
* ``N_qz``     -- the same counts in the nb40 (z, q) 2-D bins,
* ``dl_full``  -- binned D_ell of the unmasked y-map restricted to the patch,
* ``dl_mask``  -- binned D_ell of the q>5-masked y-map restricted to the patch,

using the fsky-corrected pseudo-C_ell estimator
``anafast((y - <y>_w) * w) / mean(w^2)`` with a C1-apodized patch footprint
(and, for the masked case, the same 5 x theta_500 cluster-hole mask as
``compute_masked_tsz_ps_L1_m9_feedback.py``).  Spectra are pixel-window
deconvolved and averaged into the last 9 bins of the project-standard 18-bin
log-spaced scheme (all with ell_eff > 100).

The sample covariance of the joint vector across the 192 patches is the
patch-resampling estimate of the CNC x PS covariance used in nb43.

Output: ``data/nb43_L1_m9_patch_cov/patch_ensemble.npz`` (+ ``.json`` manifest),
checkpointed every ``CHECKPOINT_EVERY`` patches so the run is resumable.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from flamingo.geometry import ARCMIN_PER_RAD, query_disc_separation
from flamingo.powerspectra import apodize
from hmfast.cosmology import Cosmology

MAP_PATH = Path("/rds/rds-lxu/flamingo/L1_m9/maps/y_unlensed_L1_m9_lc0_nside4096.fits")
CAT_PATH = Path(
    "/rds/rds-lxu/flamingo/L1_m9/catalogues/"
    "halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommz.csv"
)
OUT_DIR = _REPO / "data/nb43_L1_m9_patch_cov"
OUT_NPZ = OUT_DIR / "patch_ensemble.npz"

NSIDE_WORK = 1024
NSIDE_PATCH = 4
N_PATCH = 12 * NSIDE_PATCH**2
Q_CUT = 5.0
Z_CUT = 1.0
R_MASK = 5.0
APOD_DEG = 0.5
CHECKPOINT_EVERY = 24

# nb40 CNC binning.
Z_EDGES = np.linspace(0.005, 1.0, 11)
Q_EDGES = np.geomspace(5.0, 40.0, 6)

# Last 9 bins of the project-standard 18-bin log scheme (all ell_eff > 100).
ELL_LO = np.array([102, 133, 173, 224, 292, 380, 494, 642, 835])
ELL_HI = np.array([133, 173, 224, 292, 380, 494, 642, 835, 1085])
ELL_EFF = 0.5 * (ELL_LO + ELL_HI - 1)  # matches the project-standard scheme (117 .. 959.5)
LMAX = int(ELL_HI.max())

A_S_D3A = 2.099e-9
D3A = dict(
    H0=68.1,
    omega_b=0.022539,
    omega_cdm=0.118729,
    n_s=0.967,
    tau_reio=0.0544,
    ln1e10A_s=float(np.log(1e10 * A_S_D3A)),
)


def bin_dl(cl: np.ndarray, pwf: np.ndarray) -> np.ndarray:
    """Average pixel-window-deconvolved D_ell into the 9 log bins."""
    ell = np.arange(cl.size)
    dl = ell * (ell + 1.0) / (2.0 * np.pi) * cl / pwf[: cl.size] ** 2
    out = np.empty(ELL_LO.size)
    for k, (lo, hi) in enumerate(zip(ELL_LO, ELL_HI)):
        sel = (ell >= lo) & (ell < hi)
        out[k] = dl[sel].mean()
    return out


def patch_dl(ymap: np.ndarray, weight: np.ndarray, pwf: np.ndarray) -> np.ndarray:
    """fsky-corrected pseudo-C_ell D_ell of ``ymap`` under ``weight``."""
    w2 = float(np.mean(weight**2))
    y0 = float(np.sum(weight * ymap) / np.sum(weight))
    cl = hp.anafast((ymap - y0) * weight, lmax=LMAX) / w2
    return bin_dl(cl, pwf)


def main() -> None:
    n_first = int(sys.argv[sys.argv.index("--patches") + 1]) if "--patches" in sys.argv else N_PATCH
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[load] map {MAP_PATH.name}", flush=True)
    ymap = hp.read_map(MAP_PATH)
    ymap = hp.ud_grade(ymap, NSIDE_WORK).astype(np.float64)
    pwf = np.asarray(hp.pixwin(NSIDE_WORK, lmax=LMAX), dtype=np.float64)
    print(f"[load] degraded to nside {NSIDE_WORK} ({time.time()-t0:.0f}s)", flush=True)

    cat = pd.read_csv(
        CAT_PATH,
        comment="#",
        usecols=["theta_rot_rad", "phi_rot_rad", "q_from_mz", "R_500c_Mpc", "z"],
    )
    qv = cat["q_from_mz"].to_numpy(np.float64)
    zv = cat["z"].to_numpy(np.float64)
    th = cat["theta_rot_rad"].to_numpy(np.float64)
    ph = cat["phi_rot_rad"].to_numpy(np.float64)
    det = np.isfinite(qv) & (qv > Q_CUT)
    print(f"[cat] {len(cat):,} rows, N(q>{Q_CUT:g}) = {det.sum():,}", flush=True)

    cosmo = Cosmology(**D3A)
    dA = np.asarray(cosmo.angular_diameter_distance(zv[det]), dtype=np.float64)
    theta500_arcmin = cat["R_500c_Mpc"].to_numpy(np.float64)[det] / dA * ARCMIN_PER_RAD

    print("[mask] building q>5 hole mask ...", flush=True)
    holes = np.ones(hp.nside2npix(NSIDE_WORK))
    for t, p, t5 in zip(th[det], ph[det], theta500_arcmin):
        pix, _ = query_disc_separation(NSIDE_WORK, t, p, R_MASK * t5 / ARCMIN_PER_RAD)
        holes[pix] = 0.0
    holes_apo = apodize(holes, aperture_deg=APOD_DEG)
    print(f"[mask] f_sky(binary) = {holes.mean():.4f} ({time.time()-t0:.0f}s)", flush=True)

    # Patch index of every working pixel and of every detected cluster (RING at NSIDE_PATCH).
    pix_theta, pix_phi = hp.pix2ang(NSIDE_WORK, np.arange(hp.nside2npix(NSIDE_WORK)))
    patch_of_pix = hp.ang2pix(NSIDE_PATCH, pix_theta, pix_phi)
    del pix_theta, pix_phi
    patch_of_cl = hp.ang2pix(NSIDE_PATCH, th[det], ph[det])

    # Per-patch number counts (q>5, z<1) in nb40 bins.
    zin = (zv[det] < Z_CUT)
    N_qz = np.zeros((N_PATCH, Z_EDGES.size - 1, Q_EDGES.size - 1))
    for p in range(N_PATCH):
        sel = zin & (patch_of_cl == p)
        N_qz[p], _, _ = np.histogram2d(
            zv[det][sel], qv[det][sel], bins=[Z_EDGES, Q_EDGES]
        )
    N_z = N_qz.sum(axis=2)
    print(f"[cnc] total N(q>5, z<1) = {N_z.sum():.0f}", flush=True)

    dl_full = np.full((N_PATCH, ELL_EFF.size), np.nan)
    dl_mask = np.full((N_PATCH, ELL_EFF.size), np.nan)
    fsky_patch = np.zeros(N_PATCH)
    start = 0
    if OUT_NPZ.exists():
        old = np.load(OUT_NPZ)
        if old["dl_full"].shape == dl_full.shape:
            dl_full, dl_mask = old["dl_full"], old["dl_mask"]
            fsky_patch = old["fsky_patch"]
            start = int(old["n_done"])
            print(f"[resume] {start} patches already done", flush=True)

    def save(n_done: int) -> None:
        np.savez(
            OUT_NPZ,
            N_z=N_z, N_qz=N_qz, z_edges=Z_EDGES, q_edges=Q_EDGES,
            ell_eff=ELL_EFF, ell_lo=ELL_LO, ell_hi=ELL_HI,
            dl_full=dl_full, dl_mask=dl_mask, fsky_patch=fsky_patch,
            n_done=n_done, nside_work=NSIDE_WORK, nside_patch=NSIDE_PATCH,
            q_cut=Q_CUT, z_cut=Z_CUT, r_mask=R_MASK, apod_deg=APOD_DEG,
        )

    for p in range(start, n_first):
        tp = time.time()
        w = (patch_of_pix == p).astype(np.float64)
        w_apo = apodize(w, aperture_deg=APOD_DEG)
        dl_full[p] = patch_dl(ymap, w_apo, pwf)
        wm = w_apo * holes_apo
        dl_mask[p] = patch_dl(ymap, wm, pwf)
        fsky_patch[p] = float(w.mean())
        if p % CHECKPOINT_EVERY == CHECKPOINT_EVERY - 1 or p == n_first - 1:
            save(p + 1)
        print(
            f"[{p:3d}] N(q>5,z<1)={int(N_z[p].sum()):3d} "
            f"D_full(ell~336)={dl_full[p][4]:.3e} D_mask={dl_mask[p][4]:.3e} "
            f"({time.time()-tp:.1f}s, total {time.time()-t0:.0f}s)",
            flush=True,
        )

    save(n_first)
    git_hash = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO, capture_output=True, text=True
    ).stdout.strip()
    manifest = dict(
        script="scripts/export_L1_m9_patch_cnc_ps.py",
        git_hash=git_hash,
        map=str(MAP_PATH),
        catalogue=str(CAT_PATH),
        nside_work=NSIDE_WORK,
        nside_patch=NSIDE_PATCH,
        n_patch=N_PATCH,
        q_cut=Q_CUT,
        z_cut=Z_CUT,
        r_mask=R_MASK,
        apod_deg=APOD_DEG,
        ell_lo=ELL_LO.tolist(),
        ell_hi=ELL_HI.tolist(),
        cosmology="D3A (FLAMINGO)",
        runtime_s=round(time.time() - t0, 1),
        n_done=n_first,
    )
    (OUT_DIR / "patch_ensemble.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] wrote {OUT_NPZ} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
