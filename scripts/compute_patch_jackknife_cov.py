"""Patch-jackknife covariance for masked tSZ bandpowers (L1_m9 fiducial).

Uses pixell flat-sky FFT to compute per-patch power spectra, avoiding
NaMaster mode-coupling / mask-decoupling issues. Each patch is projected
to a tangential enmap, masked, and FFT'd.

Full-sky covariance = patch covariance / N_patches.

Usage::

    python scripts/compute_patch_jackknife_cov.py --q-cut 5 --n-patches 12
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
from pixell import enmap, reproject, curvedsky

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from flamingo.catalogue import theta_500  # noqa: E402

MAP_DIR = Path("/rds/rds-lxu/flamingo/L1_m9/maps")
CAT_DIR = Path("/rds/rds-lxu/flamingo/L1_m9/catalogues")
OUT_DIR = REPO / "data_paper" / "patch_jackknife_cov"
TAG = "qfrommz_alpha_fixed_1p12"

FWHM_ARCMIN = 10.0
R_MULT = 4.0
LMAX = 10000

ELL_MIN = np.array(
    [9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835]
)
ELL_MAX = np.array(
    [12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835, 1085]
)
ELL_EFF = np.array(
    [10.0, 13.5, 18.0, 23.5, 30.5, 40.0, 52.5, 68.5, 89.5, 117.0, 152.5, 198.0,
     257.5, 335.5, 436.5, 567.5, 738.0, 959.5]
)

CHUNK = 1_000_000
COLUMNS = ["z", "R_500c_Mpc", "theta_rot_rad", "phi_rot_rad", "q_from_mz"]


def load_catalogue(path: Path, q_cut: float):
    cols: dict[str, list[np.ndarray]] = {"theta": [], "phi": [], "t500": [], "q": []}
    n_rows = 0
    for chunk in pd.read_csv(path, comment="#", usecols=COLUMNS, chunksize=CHUNK):
        cols["theta"].append(chunk["theta_rot_rad"].to_numpy(np.float64))
        cols["phi"].append(chunk["phi_rot_rad"].to_numpy(np.float64))
        cols["q"].append(chunk["q_from_mz"].to_numpy(np.float64))
        cols["t500"].append(
            theta_500(chunk["R_500c_Mpc"].to_numpy(np.float64), chunk["z"].to_numpy(np.float64))
        )
        n_rows += len(chunk)
    out = {k: np.concatenate(v) for k, v in cols.items()}
    keep = out["q"] > q_cut
    print(f"  catalogue: {n_rows:,} halos, {int(keep.sum()):,} with q>{q_cut}", flush=True)
    return {k: v[keep] for k, v in out.items()}


def build_healpix_mask(nside, theta, phi, radius):
    """Binary disc mask: 1 outside discs, 0 inside."""
    mask = np.ones(hp.nside2npix(nside), dtype=np.float64)
    for th, ph, rr in zip(theta, phi, radius):
        mask[hp.query_disc(nside, hp.ang2vec(th, ph), float(rr))] = 0.0
    return mask


def patch_centers(n_patches):
    """Return (dec, ra) centers and (width_deg, height_deg) for N equal-area patches.

    Uses a declination x RA grid.  Patch dimensions are chosen to be
    large enough to cover ell ~ 6 (the lowest Planck bin starts at ell=9,
    which needs ~40 deg across).
    """
    # Grid options: more RA splits than dec to keep patches wider in dec
    grids = {12: (3, 4), 24: (4, 6), 6: (2, 3), 8: (2, 4), 16: (4, 4)}
    n_dec, n_ra = grids.get(n_patches, (3, 4))
    dec_edges = np.linspace(-90, 90, n_dec + 1)
    dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])
    dec_width = 180.0 / n_dec  # degrees
    ra_edges = np.linspace(0, 360, n_ra + 1)
    ra_centers = 0.5 * (ra_edges[:-1] + ra_edges[1:])
    ra_width = 360.0 / n_ra  # degrees
    centers = []
    for dec in dec_centers:
        for ra in ra_centers:
            centers.append((np.deg2rad(dec), np.deg2rad(ra)))
    patch_size = max(dec_width, ra_width)  # use the larger dimension for the enmap
    return centers, (n_dec, n_ra), patch_size


def patch_bandpowers_pixell(ymap_healpix, mask_healpix, nside, center, width_deg, pixwin_cl):
    """Compute 18-bin D_ell for a patch using pixell flat-sky FFT.

    1. Project HEALPix map + mask to a tangential enmap
    2. Subtract mask-weighted monopole
    3. FFT power spectrum
    4. Bin into 18 Planck bins
    5. Deconvolve pixel window
    """
    dec, ra = center
    width = np.deg2rad(width_deg)
    shape, wcs = enmap.geometry(pos=(dec, ra), shape=(800, 800),
                                res=width / 800, proj="tan")
    stamp = reproject.healpix2map(ymap_healpix, shape, wcs, method="spline", order=1)
    mask_stamp = reproject.healpix2map(mask_healpix, shape, wcs, method="spline", order=1)

    # Apply mask
    f_sky = float(mask_stamp.mean())
    if f_sky < 0.01:
        return None, 0.0

    stamp_masked = stamp * mask_stamp
    monopole = float(np.sum(mask_stamp * stamp) / np.sum(mask_stamp))
    stamp_masked = stamp_masked - monopole * mask_stamp

    # FFT power spectrum
    fft = enmap.fft(stamp_masked)
    ps2d = np.abs(fft) ** 2 / float(mask_stamp.sum())

    # Bin in ell using modlmap — use width=1 so every Planck bin has coverage
    modl = enmap.modlmap(shape, wcs)
    ell_fine = np.arange(0.5, LMAX + 1, 1.0)
    ps1d = np.zeros(len(ell_fine))
    flat_ell = modl.ravel()
    flat_ps = ps2d.ravel()
    # Sort by ell for fast binning
    order = np.argsort(flat_ell)
    flat_ell = flat_ell[order]
    flat_ps = flat_ps[order]
    for i, el in enumerate(ell_fine):
        # Assign each mode to nearest integer ell
        pass
    # Actually use np.histogram for speed
    ell_int = np.round(modl).astype(int).ravel()
    ps_sum = np.bincount(ell_int, weights=ps2d.ravel(), minlength=LMAX + 2)
    n_modes = np.bincount(ell_int, minlength=LMAX + 2)
    ps1d = np.divide(ps_sum, n_modes, out=np.zeros_like(ps_sum, dtype=float), where=n_modes > 0)
    ell_arr = np.arange(LMAX + 2, dtype=float)

    # Deconvolve pixel window
    pixwin = hp.pixwin(nside, lmax=LMAX)
    ps1d_corrected = ps1d / (pixwin[np.clip(ell_arr.astype(int), 0, LMAX)] ** 2)

    # Bin into 18 Planck bins
    dl = ell_arr * (ell_arr + 1.0) * ps1d_corrected / (2.0 * np.pi)
    dl_binned = np.array(
        [np.nanmean(dl[(ell_arr >= lo) & (ell_arr <= hi)])
         for lo, hi in zip(ELL_MIN, ELL_MAX)]
    )
    return dl_binned, f_sky


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-cut", type=float, default=5.0)
    parser.add_argument("--n-patches", type=int, default=12)
    parser.add_argument("--variant", default="L1_m9")
    parser.add_argument("--patch-width-deg", type=float, default=30.0,
                        help="patch width in degrees")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    q_cut = args.q_cut
    n_patches = args.n_patches
    if np.isinf(q_cut):
        cut_tag = "fullsky"
    else:
        cut_tag = f"qgt{int(q_cut)}"

    t0 = time.time()
    print(f"Loading map: {args.variant}", flush=True)
    ymap = hp.read_map(MAP_DIR / f"y_unlensed_{args.variant}_lc0_nside4096.fits", dtype=np.float64)
    nside = hp.npix2nside(ymap.size)
    print(f"  nside={nside}, npix={ymap.size} ({time.time()-t0:.0f}s)", flush=True)

    print(f"Loading catalogue and building q>{q_cut} mask ...", flush=True)
    cat_path = CAT_DIR / f"halo_catalogue_M500c_5e13_zlt3_{args.variant}_yang26rot_{TAG}.csv"
    if np.isinf(q_cut):
        # Full sky: no masking
        mask_q = np.ones(hp.nside2npix(nside), dtype=np.float64)
        f_sky_q = 1.0
        print(f"  full sky: f_sky=1.0000 (no mask)", flush=True)
    else:
        cat = load_catalogue(cat_path, q_cut)
        mask_floor = np.deg2rad(2.0 * FWHM_ARCMIN / 60.0)
        radius = np.maximum(R_MULT * cat["t500"], mask_floor)
        mask_q = build_healpix_mask(nside, cat["theta"], cat["phi"], radius)
        f_sky_q = float(mask_q.mean())
        print(f"  q>{q_cut} mask: f_sky={f_sky_q:.4f} ({time.time()-t0:.0f}s)", flush=True)

    centers, grid, patch_size_deg = patch_centers(n_patches)
    n_dec, n_ra = grid
    print(f"Patch grid: {n_dec} dec x {n_ra} ra = {n_patches} patches", flush=True)
    print(f"Patch size: {patch_size_deg:.0f} deg", flush=True)

    pixwin_cl = hp.pixwin(nside, lmax=LMAX) ** 2

    print(f"Computing bandpowers per patch (pixell flat-sky FFT) ...", flush=True)
    bandpowers = []
    f_sky_patches = []
    for i, center in enumerate(centers):
        dl, f_sky_p = patch_bandpowers_pixell(
            ymap, mask_q, nside, center, patch_size_deg, pixwin_cl
        )
        if dl is None:
            print(f"  patch {i}: f_sky too small, skipping", flush=True)
            continue
        bandpowers.append(dl)
        f_sky_patches.append(f_sky_p)
        print(f"  patch {i}: f_sky={f_sky_p:.4f} Dl[0]={dl[0]:.4e} Dl[-1]={dl[-1]:.4e} ({time.time()-t0:.0f}s)", flush=True)

    bandpowers = np.array(bandpowers)
    n_used = len(bandpowers)
    print(f"\nComputed {n_used} patch bandpowers ({time.time()-t0:.0f}s)", flush=True)

    # Patch-jackknife covariance
    d_mean = bandpowers.mean(axis=0)
    d_centered = bandpowers - d_mean
    cov_patch = (d_centered.T @ d_centered) / (n_used - 1)
    cov_fullsky = cov_patch / n_used

    # Patch-jackknife bandpowers are already in physical D_ell units
    # (matching the data after data_scale=1e-12 in the likelihood).
    # No additional 1e12 scaling needed.
    cov_fullsky_scaled = cov_fullsky

    out_path = OUT_DIR / f"cov_patch_jackknife_L1_m9_{cut_tag}_{TAG}_binned_18.npy"
    np.save(out_path, cov_fullsky_scaled)
    print(f"\nSaved: {out_path}", flush=True)

    # Compare with existing covariance
    from scripts.run_masked_ps_chains import data_files, load_converged_artifacts
    artifacts = load_converged_artifacts()
    if np.isinf(q_cut):
        _, cov_existing_path = data_files("fullsky", artifacts["covariance_paths"])
        cov_label = "fullsky"
    else:
        _, cov_existing_path = data_files(f"qgt{int(q_cut)}", artifacts["covariance_paths"])
        cov_label = f"q>{int(q_cut)}"
    cov_existing = np.load(cov_existing_path)

    print(f"\n=== Comparison ({cov_label}) ===")
    err_new = np.sqrt(np.diag(cov_fullsky_scaled))
    err_old = np.sqrt(np.diag(cov_existing))
    print(f"  {'ell':>6s} {'err_new':>12s} {'err_old':>12s} {'ratio':>8s}")
    for i in range(18):
        r = err_new[i] / err_old[i] if err_old[i] > 0 else 0
        print(f"  {ELL_EFF[i]:6.0f} {err_new[i]:12.4e} {err_old[i]:12.4e} {r:8.1f}")
    print(f"\n  mean ratio (new/old): {np.mean(err_new/err_old):.1f}")

    # Plot: 2D covariance matrices + 1D error comparison
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 2D: patch-jackknife cov
    ax = axes[0]
    im = ax.imshow(cov_fullsky_scaled, origin="upper", cmap="viridis",
                  aspect="auto", norm=matplotlib.colors.LogNorm())
    ax.set_title("Patch-jackknife cov")
    ax.set_xlabel("bin"); ax.set_ylabel("bin")
    ax.set_xticks(range(0, 18, 3)); ax.set_yticks(range(0, 18, 3))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 2D: analytical cov
    ax = axes[1]
    im = ax.imshow(cov_existing, origin="upper", cmap="viridis",
                  aspect="auto", norm=matplotlib.colors.LogNorm())
    ax.set_title("Analytical cov")
    ax.set_xlabel("bin"); ax.set_ylabel("bin")
    ax.set_xticks(range(0, 18, 3)); ax.set_yticks(range(0, 18, 3))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 1D: error comparison
    ax = axes[2]
    ax.semilogy(ELL_EFF, err_old, 'o-', label="Analytical cov", color="#1f4e79")
    ax.semilogy(ELL_EFF, err_new, 's-', label="Patch-jackknife cov", color="#e67e22")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\sigma(D_\ell)$")
    ax.set_title(f"Diagonal (L1_m9 {cov_label})")
    ax.legend(frameon=False)

    fig.suptitle(f"Covariance comparison (L1_m9 {cov_label})", y=1.02)
    fig.tight_layout()
    plot_path = OUT_DIR / f"cov_comparison_{cut_tag}.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}")

    # Plot: 2D covariance matrices + 1D error comparison
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 2D: patch-jackknife cov
    ax = axes[0]
    im = ax.imshow(cov_fullsky_scaled, origin="upper", cmap="viridis",
                  aspect="auto", norm=matplotlib.colors.LogNorm())
    ax.set_title("Patch-jackknife cov")
    ax.set_xlabel("bin"); ax.set_ylabel("bin")
    ax.set_xticks(range(0, 18, 3)); ax.set_yticks(range(0, 18, 3))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 2D: sim-realization cov
    ax = axes[1]
    im = ax.imshow(cov_existing, origin="upper", cmap="viridis",
                  aspect="auto", norm=matplotlib.colors.LogNorm())
    ax.set_title("Sim-realization cov")
    ax.set_xlabel("bin"); ax.set_ylabel("bin")
    ax.set_xticks(range(0, 18, 3)); ax.set_yticks(range(0, 18, 3))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 1D: error comparison
    ax = axes[2]
    ax.semilogy(ELL_EFF, err_old, 'o-', label="Sim-realization cov", color="#1f4e79")
    ax.semilogy(ELL_EFF, err_new, 's-', label="Patch-jackknife cov", color="#e67e22")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\sigma(D_\ell)$")
    ax.set_title(f"Diagonal (L1_m9 q>{int(q_cut)})")
    ax.legend(frameon=False)

    fig.suptitle(f"Covariance comparison (L1_m9 q>{int(q_cut)})", y=1.02)
    fig.tight_layout()
    plot_path = OUT_DIR / f"cov_comparison_{cut_tag}.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
