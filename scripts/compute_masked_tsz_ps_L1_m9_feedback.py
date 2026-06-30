"""NaMaster masked tSZ bandpowers for L1_m9 feedback y-maps (q cuts).

Same map-side pipeline as ``notebooks/09_cluster_number_counts.ipynb`` (L2p8):
  * mask 5×θ500 discs around clusters with q_from_mz > q_cut
  * C1 0.5° apodization
  * mask-weighted monopole subtracted
  * NaMaster decoupled pseudo-Cl, linear Δℓ=30 bins to ℓ_max=6000
  * HEALPix pixel-window deconvolution

Catalogues: ``halo_catalogue_M500c_5e13_zlt3_{variant}_yang26rot_qfrommz.csv``
Maps: ``y_unlensed_{variant}_lc0_nside4096.fits``

Outputs:
  data/bandpowers_L1_m9_feedback/masked_tsz_ps.npz
  data/bandpowers_L1_m9_feedback/Dl_yy_{variant}_{tag}.txt  (per variant × cut)

Run:
    python scripts/compute_masked_tsz_ps_L1_m9_feedback.py
    python scripts/compute_masked_tsz_ps_L1_m9_feedback.py --variant L1_m9
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

MAP_DIR = Path("/rds/rds-lxu/flamingo/L1_m9/maps")
CAT_DIR = Path("/rds/rds-lxu/flamingo/L1_m9/catalogues")
OUT_DIR = _REPO / "data/bandpowers_L1_m9_feedback"
OUT_NPZ = OUT_DIR / "masked_tsz_ps.npz"

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

VARIANTS = [
    "L1_m9",
    "fgas+2sigma",
    "fgas-2sigma",
    "fgas-4sigma",
    "fgas-8sigma",
    "Mstar-1sigma",
    "Mstar-1sigma_fgas-4sigma",
    "Jet",
    "Jet_fgas-4sigma",
]

CAT_COLS = [
    "theta_rot_rad",
    "phi_rot_rad",
    "q_from_mz",
    "R_500c_Mpc",
    "z",
]


def map_path(variant: str) -> Path:
    return MAP_DIR / f"y_unlensed_{variant}_lc0_nside4096.fits"


def cat_path(variant: str) -> Path:
    return CAT_DIR / f"halo_catalogue_M500c_5e13_zlt3_{variant}_yang26rot_qfrommz.csv"


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
    idx = np.where(sel)[0]
    for i in idx:
        r_rad = R_MASK * theta500[i] / ARCMIN_PER_RAD
        pix, _ = query_disc_separation(nside, theta_rot[i], phi_rot[i], r_rad)
        mask[pix] = 0.0
    return mask, n_cl, float(mask.mean())


def process_variant(variant: str, cosmo: Cosmology, *, force: bool = False) -> dict:
    mpath = map_path(variant)
    cpath = cat_path(variant)
    if not mpath.exists():
        raise FileNotFoundError(f"missing map {mpath}")
    if not cpath.exists():
        raise FileNotFoundError(f"missing catalogue {cpath}")

    t0 = time.time()
    print(f"=== {variant} ===", flush=True)
    print(f"  loading map {mpath.name}", flush=True)
    ymap = hp.read_map(mpath, verbose=False)
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

        out_txt = OUT_DIR / f"Dl_yy_{variant}_{tag}.txt"
        np.savetxt(out_txt, np.column_stack([ellb, dl]), fmt="%.6e")
        np.savez(
            OUT_DIR / f"meta_{variant}_{tag}.npz",
            variant=variant,
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

    out_full = OUT_DIR / f"Dl_yy_{variant}_fullsky.txt"
    np.savetxt(out_full, np.column_stack([ellb, dl_full]), fmt="%.6e")
    np.savez(
        OUT_DIR / f"meta_{variant}_fullsky.npz",
        variant=variant,
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
    p.add_argument("--variant", action="append")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    variants = args.variant or VARIANTS
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_NPZ.exists() and not args.force and args.variant is None:
        print(f"exists {OUT_NPZ} (use --force to rebuild all)")
        return

    cosmo = Cosmology(**D3A)

    existing: dict[str, dict] = {}
    ellb_ref: np.ndarray | None = None
    if OUT_NPZ.exists() and args.variant is not None:
        old = np.load(OUT_NPZ, allow_pickle=True)
        ellb_ref = old["ellb"]
        for i, v in enumerate(old["variants"]):
            if v not in variants:
                existing[str(v)] = dict(
                    dl_full=old["dl_fullsky"][i],
                    dl_cuts=old["dl_masked"][i],
                    n_detected=old["n_detected"][i],
                    fsky_binary=old["fsky_binary"][i],
                )

    rows: dict[str, dict] = dict(existing)
    for variant in variants:
        row = process_variant(variant, cosmo, force=args.force)
        if ellb_ref is None:
            ellb_ref = row["ellb"]
        elif not np.allclose(row["ellb"], ellb_ref):
            raise ValueError(f"ell bin mismatch for {variant}")
        rows[variant] = row

    done_variants = variants if args.variant else VARIANTS
    if args.variant:
        done_variants = [v for v in VARIANTS if v in rows]

    dl_fullsky = np.stack([rows[v]["dl_full"] for v in done_variants], axis=0)
    dl_masked = np.stack([rows[v]["dl_cuts"] for v in done_variants], axis=0)
    n_detected = np.stack([rows[v]["n_detected"] for v in done_variants], axis=0)
    fsky_binary = np.stack([rows[v]["fsky_binary"] for v in done_variants], axis=0)

    np.savez(
        OUT_NPZ,
        variants=np.array(done_variants),
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
        map_dir=str(MAP_DIR),
        cat_dir=str(CAT_DIR),
    )
    print(f"\nwrote {OUT_NPZ}  shape dl_masked={dl_masked.shape}", flush=True)
    print("\n=== summary N(q>cut) per variant ===")
    for iv, variant in enumerate(done_variants):
        parts = "  ".join(
            f"q>{int(q)}={n_detected[iv, ic]:4d}" for ic, q in enumerate(Q_CUTS)
        )
        print(f"  {variant:28s}  {parts}")


if __name__ == "__main__":
    main()
