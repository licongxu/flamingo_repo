"""NaMaster masked tSZ bandpowers for L2p8_m9 lightcones 0..7 (q cuts).

Same pipeline as ``scripts/compute_masked_tsz_ps_L1_m9_feedback.py`` and
``notebooks/09_cluster_number_counts.ipynb``.

Maps: ``/rds/.../L2p8_m9/lightcone{N}/healpix_map/y_unlensed_L2p8_m9_lc{N}.fits``
Catalogues: ``.../lightcone{N}/catalogues/halo_catalogue_M500c_5e13_..._qfrommz.csv``

Outputs:
  data/bandpowers_L2p8_m9_multilc/masked_tsz_ps.npz
  data/bandpowers_L2p8_m9_multilc/Dl_yy_lc{N}_{tag}.txt

Run:
    python scripts/compute_masked_tsz_ps_L2p8_m9_multilc.py
    python scripts/compute_masked_tsz_ps_L2p8_m9_multilc.py --lightcone 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from flamingo.geometry import ARCMIN_PER_RAD, query_disc_separation
from flamingo.powerspectra import apodize, decoupled_dl
from hmfast.cosmology import Cosmology

BASE = Path("/rds/rds-lxu/flamingo/L2p8_m9")
OUT_DIR = _REPO / "data/bandpowers_L2p8_m9_multilc"
OUT_NPZ = OUT_DIR / "masked_tsz_ps.npz"

N_LIGHTCONES = 8
LMAX = 6000
DELL = 30
R_MASK = 5.0
APOD_DEG = 0.5
Q_CUTS = [50.0, 20.0, 10.0, 5.0]
CUT_TAGS = ["qgt50", "qgt20", "qgt10", "qgt5"]

A_S_D3A = 2.099e-9
D3A = dict(
    H0=68.1,
    omega_b=0.022539,
    omega_cdm=0.118729,
    n_s=0.967,
    tau_reio=0.0544,
    ln1e10A_s=float(np.log(1e10 * A_S_D3A)),
)

CAT_COLS = [
    "theta_rot_rad",
    "phi_rot_rad",
    "q_from_mz",
    "R_500c_Mpc",
    "z",
]


def map_path(lightcone: int) -> Path:
    return BASE / f"lightcone{lightcone}" / "healpix_map" / f"y_unlensed_L2p8_m9_lc{lightcone}.fits"


def cat_path(lightcone: int) -> Path:
    return (
        BASE
        / f"lightcone{lightcone}"
        / "catalogues"
        / "halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_qfrommz.csv"
    )


def theta500_arcmin(r500_mpc: np.ndarray, z: np.ndarray, cosmo: Cosmology) -> np.ndarray:
    dA = np.asarray(cosmo.angular_diameter_distance(z), dtype=np.float64)
    return r500_mpc / dA * ARCMIN_PER_RAD


def dl_nmt(
    ymap: np.ndarray,
    weight: np.ndarray | None,
    *,
    nside: int,
    pwf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ell_eff, _, cl = decoupled_dl(ymap, weight, delta_ell=DELL, lmax=LMAX)
    pw = np.interp(ell_eff, np.arange(pwf.size), pwf)
    dl = ell_eff * (ell_eff + 1.0) / (2.0 * np.pi) * cl / pw**2
    return ell_eff, dl


def build_mask(
    nside: int,
    qv: np.ndarray,
    qcut: float,
    theta_rot: np.ndarray,
    phi_rot: np.ndarray,
    theta500: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    sel = np.isfinite(qv) & (qv > qcut)
    n_cl = int(sel.sum())
    mask = np.ones(hp.nside2npix(nside), dtype=np.float64)
    for i in np.where(sel)[0]:
        r_rad = R_MASK * theta500[i] / ARCMIN_PER_RAD
        pix, _ = query_disc_separation(nside, theta_rot[i], phi_rot[i], r_rad)
        mask[pix] = 0.0
    return mask, n_cl, float(mask.mean())


def process_lightcone(lightcone: int, cosmo: Cosmology) -> dict:
    mpath = map_path(lightcone)
    cpath = cat_path(lightcone)
    label = f"lc{lightcone}"
    if not mpath.exists():
        raise FileNotFoundError(f"missing map {mpath}")
    if not cpath.exists():
        raise FileNotFoundError(f"missing catalogue {cpath}")

    t0 = time.time()
    print(f"=== {label} ===", flush=True)
    print(f"  loading map {mpath.name}", flush=True)
    ymap = hp.read_map(mpath)
    nside = hp.npix2nside(ymap.size)
    pwf = hp.pixwin(nside, lmax=LMAX)

    print(f"  loading catalogue {cpath.name}", flush=True)
    cat = pd.read_csv(cpath, comment="#", usecols=CAT_COLS)
    theta_rot = cat["theta_rot_rad"].to_numpy(dtype=np.float64)
    phi_rot = cat["phi_rot_rad"].to_numpy(dtype=np.float64)
    qv = cat["q_from_mz"].to_numpy(dtype=np.float64)
    theta500 = theta500_arcmin(
        cat["R_500c_Mpc"].to_numpy(dtype=np.float64),
        cat["z"].to_numpy(dtype=np.float64),
        cosmo,
    )

    print("  full sky (NaMaster)...", flush=True)
    ellb, dl_full = dl_nmt(ymap, None, nside=nside, pwf=pwf)

    dl_cuts: list[np.ndarray] = []
    n_det: list[int] = []
    fsky_bin: list[float] = []
    for qcut, tag in zip(Q_CUTS, CUT_TAGS):
        print(f"  q>{qcut:g} mask + NaMaster...", flush=True)
        mask, n_cl, f_bin = build_mask(nside, qv, qcut, theta_rot, phi_rot, theta500)
        mask_apo = apodize(mask, aperture_deg=APOD_DEG)
        _, dl = dl_nmt(ymap, mask_apo, nside=nside, pwf=pwf)
        dl_cuts.append(dl)
        n_det.append(n_cl)
        fsky_bin.append(f_bin)
        print(f"    N(q>{qcut:g})={n_cl:,}  f_sky={f_bin:.4f}", flush=True)

        out_txt = OUT_DIR / f"Dl_yy_{label}_{tag}.txt"
        np.savetxt(out_txt, np.column_stack([ellb, dl]), fmt="%.6e")
        np.savez(
            OUT_DIR / f"meta_{label}_{tag}.npz",
            lightcone=lightcone,
            tag=tag,
            q_cut=qcut,
            ellb=ellb,
            dl=dl,
            n_detected=n_cl,
            fsky_binary=f_bin,
            lmax=LMAX,
            delta_ell=DELL,
            r_mask=R_MASK,
            apod_deg=APOD_DEG,
        )

    np.savetxt(
        OUT_DIR / f"Dl_yy_{label}_fullsky.txt",
        np.column_stack([ellb, dl_full]),
        fmt="%.6e",
    )
    np.savez(
        OUT_DIR / f"meta_{label}_fullsky.npz",
        lightcone=lightcone,
        tag="fullsky",
        q_cut=None,
        ellb=ellb,
        dl=dl_full,
        n_detected=0,
        fsky_binary=1.0,
        lmax=LMAX,
        delta_ell=DELL,
    )

    print(f"  done in {time.time()-t0:.0f}s", flush=True)
    return dict(
        ellb=ellb,
        dl_full=dl_full,
        dl_cuts=np.stack(dl_cuts, axis=0),
        n_detected=np.array(n_det, dtype=int),
        fsky_binary=np.array(fsky_bin, dtype=float),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lightcone", type=int, action="append")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    lightcones = args.lightcone if args.lightcone is not None else list(range(N_LIGHTCONES))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_NPZ.exists() and not args.force and args.lightcone is None:
        print(f"exists {OUT_NPZ} (use --force to rebuild all)")
        return

    cosmo = Cosmology(**D3A)

    existing: dict[int, dict] = {}
    ellb_ref: np.ndarray | None = None
    if OUT_NPZ.exists() and args.lightcone is not None:
        old = np.load(OUT_NPZ, allow_pickle=True)
        ellb_ref = old["ellb"]
        for i, lc in enumerate(old["lightcones"]):
            lc = int(lc)
            if lc not in lightcones:
                existing[lc] = dict(
                    dl_full=old["dl_fullsky"][i],
                    dl_cuts=old["dl_masked"][i],
                    n_detected=old["n_detected"][i],
                    fsky_binary=old["fsky_binary"][i],
                )

    rows: dict[int, dict] = dict(existing)
    for lc in lightcones:
        row = process_lightcone(lc, cosmo)
        if ellb_ref is None:
            ellb_ref = row["ellb"]
        elif not np.allclose(row["ellb"], ellb_ref):
            raise ValueError(f"ell bin mismatch for lc{lc}")
        rows[lc] = row

    done_lcs = list(range(N_LIGHTCONES)) if args.lightcone is None else sorted(rows)
    dl_fullsky = np.stack([rows[lc]["dl_full"] for lc in done_lcs], axis=0)
    dl_masked = np.stack([rows[lc]["dl_cuts"] for lc in done_lcs], axis=0)
    n_detected = np.stack([rows[lc]["n_detected"] for lc in done_lcs], axis=0)
    fsky_binary = np.stack([rows[lc]["fsky_binary"] for lc in done_lcs], axis=0)

    np.savez(
        OUT_NPZ,
        lightcones=np.array(done_lcs, dtype=int),
        ellb=ellb_ref,
        q_cuts=np.array(Q_CUTS),
        cut_tags=np.array(CUT_TAGS),
        dl_fullsky=dl_fullsky,
        dl_masked=dl_masked,
        n_detected=n_detected,
        fsky_binary=fsky_binary,
        lmax=LMAX,
        delta_ell=DELL,
        r_mask=R_MASK,
        apod_deg=APOD_DEG,
        base=str(BASE),
    )
    print(f"\nwrote {OUT_NPZ}  shape dl_masked={dl_masked.shape}", flush=True)
    print("\n=== summary N(q>cut) per lightcone ===")
    for il, lc in enumerate(done_lcs):
        parts = "  ".join(
            f"q>{int(q)}={n_detected[il, ic]:4d}" for ic, q in enumerate(Q_CUTS)
        )
        print(f"  lc{lc}  {parts}")


if __name__ == "__main__":
    main()
