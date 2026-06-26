"""NaMaster mask-decoupled tSZ bandpowers for B=1, Y_R500c (yang26-rot map).

Modes:
  qgt5    — mask 5×θ500 discs around q>5 clusters, C1 0.5° apodization
  fullsky — no cluster masking (f_sky=1), monopole subtract only

Outputs (under data/bandpowers_arnaudB1_Y500c/):
  Dl_yy_qgt5_binned_18_Y500c.txt
  Dl_yy_fullsky_binned_18_Y500c.txt

Run:
    python scripts/compute_bandpowers_arnaudB1_Y500c.py --mode qgt5
    python scripts/compute_bandpowers_arnaudB1_Y500c.py --mode fullsky
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from flamingo.catalogue import load_catalogue
from flamingo.geometry import ARCMIN_PER_RAD, query_disc_separation
from flamingo.powerspectra import apodize

Y_MAP = _REPO / "data/hydro_L2p8m9/map/y_unlensed_L2p8_m9_lc0.fits"
Y0Q = _REPO / "data/hydro_L2p8m9/catalogue/halo_catalogue_M500c_5e13_zlt3_y0q_arnaudB1_Y500c.csv"
YANG26 = _REPO / "data/hydro_L2p8m9/catalogue/halo_catalogue_M500c_5e13_zlt3_soap_snr_d3a_yang26rot.csv"
OUT_DIR = _REPO / "data/bandpowers_arnaudB1_Y500c"

Q_CUT = 5.0
MASK_RADIUS_FACTOR = 5.0
APOD_DEG = 0.5
LMAX = 2048

BINS = np.array([
    [9, 12, 10.0], [12, 16, 13.5], [16, 21, 18.0], [21, 27, 23.5],
    [27, 35, 30.5], [35, 46, 40.0], [46, 60, 52.5], [60, 78, 68.5],
    [78, 102, 89.5], [102, 133, 117.0], [133, 173, 152.5], [173, 224, 198.0],
    [224, 292, 257.5], [292, 380, 335.5], [380, 494, 436.5], [494, 642, 567.5],
    [642, 835, 738.0], [835, 1085, 959.5],
], dtype=float)


def _build_mask(ymap, nside, log) -> tuple[np.ndarray, int, float]:
    y0q = load_catalogue(Y0Q)
    yang = load_catalogue(YANG26)
    if len(y0q) != len(yang):
        raise RuntimeError(f"row mismatch y0q={len(y0q)} yang26={len(yang)}")

    q = y0q["q"].to_numpy(float)
    sel = np.isfinite(q) & (q > Q_CUT)
    n_cl = int(sel.sum())
    theta500_arcmin = y0q["theta500_arcmin"].to_numpy(float)
    theta_rot = yang["theta_rot_rad"].to_numpy(float)
    phi_rot = yang["phi_rot_rad"].to_numpy(float)

    mask = np.ones(ymap.size, dtype=np.float64)
    t0 = time.time()
    for i in np.where(sel)[0]:
        r_rad = MASK_RADIUS_FACTOR * theta500_arcmin[i] / ARCMIN_PER_RAD
        pix, _ = query_disc_separation(nside, theta_rot[i], phi_rot[i], r_rad)
        mask[pix] = 0.0
    fsky_bin = float(mask.mean())
    log(
        f"  yang26-rot positions: N(q>{Q_CUT})={n_cl:,}  "
        f"5×θ500 discs in {time.time()-t0:.1f}s  f_sky_binary={fsky_bin:.4f}"
    )
    log(f"  apodizing mask (C1, {APOD_DEG}°)...")
    t1 = time.time()
    mask_apod = apodize(mask, aperture_deg=APOD_DEG, apotype="C1")
    fsky_eff = float((mask_apod ** 2).mean())
    log(f"  apodization done in {time.time()-t1:.1f}s  f_sky_eff=<w^2>={fsky_eff:.4f}")
    return mask_apod, n_cl, fsky_eff


def _build_fullsky_mask(ymap, log) -> tuple[np.ndarray, float]:
    mask = np.ones(ymap.size, dtype=np.float64)
    fsky_eff = float((mask ** 2).mean())
    log(f"  full sky (no cluster holes)  f_sky_eff={fsky_eff:.4f}")
    return mask, fsky_eff


def _binned_dl_namaster(ymap, mask_apod, log) -> np.ndarray:
    import pymaster as nmt

    monopole = float(np.average(ymap, weights=mask_apod))
    log(f"  subtracting monopole = {monopole:.4e}")
    ydelta = ymap - monopole

    ell_min_i = BINS[:, 0].astype(int)
    ell_max_i = BINS[:, 1].astype(int)
    ells = np.arange(LMAX + 1, dtype=int)
    bpws = np.full(LMAX + 1, -1, dtype=int)
    weights = np.zeros(LMAX + 1)
    for b, (lo, hi) in enumerate(zip(ell_min_i, ell_max_i)):
        sel = np.arange(lo, hi)
        bpws[sel] = b
        weights[sel] = 1.0 / len(sel)
    f_ell = ells * (ells + 1.0) / (2.0 * np.pi)
    bins = nmt.NmtBin(bpws=bpws, ells=ells, weights=weights, lmax=LMAX, f_ell=f_ell)

    t0 = time.time()
    fld = nmt.NmtField(mask_apod, [ydelta], lmax=LMAX)
    wsp = nmt.NmtWorkspace()
    wsp.compute_coupling_matrix(fld, fld, bins)
    dl_b = wsp.decouple_cell(nmt.compute_coupled_cell(fld, fld))[0]
    log(f"  NaMaster decoupled pseudo-Cl in {time.time()-t0:.1f}s")
    return np.asarray(dl_b, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("qgt5", "fullsky"), default="qgt5",
        help="qgt5: masked q>5 catalogue; fullsky: unmasked map",
    )
    args = parser.parse_args()

    def log(msg: str) -> None:
        print(msg, flush=True)

    if args.mode == "qgt5":
        out_txt = OUT_DIR / "Dl_yy_qgt5_binned_18_Y500c.txt"
        out_meta = OUT_DIR / "meta_qgt5_Y500c.npz"
    else:
        out_txt = OUT_DIR / "Dl_yy_fullsky_binned_18_Y500c.txt"
        out_meta = OUT_DIR / "meta_fullsky_Y500c.npz"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"mode={args.mode}")
    log(f"loading map {Y_MAP}")
    ymap = hp.read_map(str(Y_MAP))
    nside = hp.npix2nside(ymap.size)
    log(f"NSIDE={nside}")

    if args.mode == "qgt5":
        mask_apod, n_cl, fsky_eff = _build_mask(ymap, nside, log)
        meta_extra = dict(n_masked=n_cl, q_cut=Q_CUT)
    else:
        mask_apod, fsky_eff = _build_fullsky_mask(ymap, log)
        meta_extra = dict(n_masked=0, q_cut=None)

    dl_b = _binned_dl_namaster(ymap, mask_apod, log)

    np.savetxt(out_txt, np.column_stack([BINS[:, 2], dl_b]), fmt="%.6e")
    np.savez(out_meta, bins=BINS, Dl=dl_b, fsky_eff=fsky_eff, mode=args.mode, **meta_extra)
    log(f"wrote {out_txt}")
    for i in range(len(BINS)):
        log(f"  bin {i:2d} ell_eff={BINS[i,2]:7.1f}  D_ell={dl_b[i]:.4e}")


if __name__ == "__main__":
    main()
