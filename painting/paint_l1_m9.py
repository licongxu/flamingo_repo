"""Paint L1_m9 SOAP catalogues with JXPaint and measure 18-bin masked D_ell.

One map: D3A cosmology, custom GNFW, full-sky best-fit A_SZ.
Default: M>=1e13 lightcone (writes *_m1e13*); 5e13 CNC products are left in place.
SNR holes still come from the 5e13 qfrommap catalogue. No pixwin.

    python painting/paint_l1_m9.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("MPLBACKEND", "Agg")

import healpy as hp
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from flamingo.catalogue import theta_500  # noqa: E402
from flamingo.inference.bandpowers import ELL_EFF, ELL_MAX, ELL_MIN  # noqa: E402
from flamingo.inference.masked_ps import B_HYDROSTATIC, GNFW_SHAPE  # noqa: E402
from scripts.iterate_l1_m9_asz_covariance import FIXED  # noqa: E402

JXPAINT_SRC = Path("/scratch/scratch-lxu/agent_dev/auto_research_agent/JXPaint/src")
CAT_DIR = Path("/rds/rds-lxu/flamingo/.hbt_join_fix_staging/20260731/L1_m9/catalogues")
CAT_FILE = CAT_DIR / "halo_catalogue_M500c_1e13_zlt3_L1_m9_yang26rot.csv"
CAT_MASK = CAT_DIR / "halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommap.csv"
OUT_TAG = "m1e13"
DATA = REPO / "data_paper" / "binned_bandpowers"
HYDRO_MAP = Path("/rds/rds-lxu/flamingo/L1_m9/maps/y_unlensed_L1_m9_lc0_nside4096.fits")
HERE = Path(__file__).resolve().parent

NSIDE = 1024
# Hydro-map inference point: D3A + custom GNFW, A_SZ-only full-sky best fit.
A_SZ = -4.1075214
ALPHA_SZ = float(FIXED["alpha_SZ"])
B = float(FIXED["B"])
H0 = float(FIXED["H0"])
OMEGA_M = 0.306
FWHM_ARCMIN = 10.0
R_MULT = 4.0
APOSIZE_DEG = 0.25
LMAX = int(ELL_MAX.max()) + 500
Q_CUTS = (("qgt50", 50.0), ("qgt20", 20.0), ("qgt10", 10.0), ("qgt5", 5.0))
BATCH = 5000
N_PER_HALO = 30_000
E_PER_HALO = 64


def healpy_to_jxpaint(theta, phi):
    """JXPaint lon/lat from HEALPix colatitude/longitude (radians)."""
    return np.asarray(phi, dtype=float), np.pi - np.asarray(theta, dtype=float)


def y0_for_painter(y0_param, F0, B=B):
    """y0_true so that (y0_true/B^{1/3}) * F(0) = y0_param.

    JXPaint multiplies by the unnormalized projected GNFW ``F`` (F(0)~7.8),
    not F/F(0). Passing y0_param raw overpaints C_ell by ~F(0)^2.
    """
    return np.asarray(y0_param, dtype=float) * (B ** (1.0 / 3.0)) / float(F0)


def bin_dl_18(ell, cl):
    """Inclusive Planck bins of D_ell, returned as 1e12 D_ell. No pixwin."""
    dl = ell * (ell + 1.0) * cl / (2.0 * np.pi) * 1e12
    dl = np.asarray(dl, dtype=float)
    dl[:2] = np.nan
    out = np.full(ELL_EFF.size, np.nan)
    for i, (lo, hi) in enumerate(zip(ELL_MIN, ELL_MAX, strict=True)):
        out[i] = np.nanmean(dl[(ell >= lo) & (ell <= hi)])
    return out


def binary_disc_mask(nside, theta, phi, radius):
    mask = np.ones(hp.nside2npix(nside), dtype=np.float64)
    for th, ph, rr in zip(theta, phi, radius):
        mask[hp.query_disc(nside, hp.ang2vec(th, ph), float(rr))] = 0.0
    return mask


def _jxpaint():
    if str(JXPAINT_SRC) not in sys.path:
        sys.path.insert(0, str(JXPAINT_SRC))
    from jxpaint.cosmology import FlatLCDM
    from jxpaint.painting.gpu_native import paint_catalogue_gpu_native
    from jxpaint.profiles.custom_gnfw import CustomGNFWPressureProfile
    from jxpaint.profiles.shape_table import load_beamed_table

    return FlatLCDM, CustomGNFWPressureProfile, load_beamed_table, paint_catalogue_gpu_native


def load_catalogue(path=CAT_FILE, need_q=False):
    cols = ["z", "M_500c_Msun", "theta_rot_rad", "phi_rot_rad", "R_500c_Mpc"]
    if need_q:
        cols.append("q_from_aperture")
    df = pd.read_csv(path, comment="#", usecols=cols)
    theta = df["theta_rot_rad"].to_numpy(np.float64)
    phi = df["phi_rot_rad"].to_numpy(np.float64)
    lon, lat = healpy_to_jxpaint(theta, phi)
    out = {
        "z": df["z"].to_numpy(np.float64),
        "M": df["M_500c_Msun"].to_numpy(np.float64) / 1e14,
        "lon": lon,
        "lat": lat,
        "theta": theta,
        "phi": phi,
        "t500": theta_500(
            df["R_500c_Mpc"].to_numpy(np.float64), df["z"].to_numpy(np.float64)
        ),
    }
    if need_q:
        out["q"] = df["q_from_aperture"].to_numpy(np.float64)
    return out


def paint_map(cat):
    FlatLCDM, CustomGNFWPressureProfile, load_beamed_table, paint = _jxpaint()
    cosmo = FlatLCDM(h=H0 / 100.0, Omega_m=OMEGA_M)
    profile = CustomGNFWPressureProfile(
        A_SZ=A_SZ, alpha_SZ=ALPHA_SZ, B=B, cosmo=cosmo, **GNFW_SHAPE
    )
    F0 = float(profile.F(0.0))
    y0 = y0_for_painter(np.asarray(profile.y0_param(cat["M"], cat["z"])), F0)
    print(f"  F(0)={F0:.4f}  y0_true = y0_param * B^(1/3) / F(0)  BATCH={BATCH}", flush=True)
    st = load_beamed_table()
    n = cat["M"].size
    ymap = np.zeros(hp.nside2npix(NSIDE), dtype=np.float64)
    t0 = time.time()

    def paint_slice(i0, i1, batch):
        # ponytail: pad so a short slice of nearby clusters cannot collapse
        # e_max = N_h * e_per_halo below the disc need.
        def pad(a, fill=0.0):
            out = np.full(batch, fill, dtype=np.float64)
            out[: i1 - i0] = a[i0:i1]
            return out

        return paint(
            pad(cat["z"], 0.1),
            pad(cat["M"], 1.0),
            pad(cat["lon"]),
            pad(cat["lat"]),
            pad(y0),
            st,
            nside=NSIDE,
            cosmo=cosmo,
            bias_B=B,
            e_per_halo=E_PER_HALO,
            n_per_halo=N_PER_HALO,
        )

    for i0 in range(0, n, BATCH):
        i1 = min(i0 + BATCH, n)
        try:
            ymap += paint_slice(i0, i1, BATCH)
        except Exception as exc:
            print(
                f"  FAIL {i0:,}:{i1:,} z=[{cat['z'][i0:i1].min():.3f},"
                f"{cat['z'][i0:i1].max():.3f}] {type(exc).__name__}: {exc}; "
                "retry 2000",
                flush=True,
            )
            for j0 in range(i0, i1, 2000):
                j1 = min(j0 + 2000, i1)
                ymap += paint_slice(j0, j1, 2000)
        if i0 == 0 or (i0 // BATCH) % 50 == 0:
            print(
                f"  painted {i1:,}/{n:,} ({time.time() - t0:.0f}s) "
                f"z=[{cat['z'][i0:i1].min():.3f},{cat['z'][i0:i1].max():.3f}]",
                flush=True,
            )
    print(f"  done in {time.time() - t0:.0f}s  max={ymap.max():.4e}  mean={ymap.mean():.4e}", flush=True)
    return ymap


def fullsky_dl(ymap, beam):
    cl = hp.anafast(ymap - ymap.mean(), lmax=LMAX)
    cl = cl / np.maximum(beam**2, 1e-30)
    return bin_dl_18(np.arange(cl.size, dtype=float), cl)


def masked_dl(ymap, mask_apo, beam):
    import pymaster as nmt

    w = mask_apo
    m = ymap - float(np.sum(w * ymap) / np.sum(w))
    field = nmt.NmtField(w, [m], beam=beam, lmax=LMAX)
    bins = nmt.NmtBin.from_lmax_linear(LMAX, nlb=1)
    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(field, field, bins)
    cl = workspace.decouple_cell(nmt.compute_coupled_cell(field, field))[0]
    cl_full = np.full(LMAX + 1, np.nan)
    cl_full[bins.get_effective_ells().astype(int)] = cl
    return bin_dl_18(np.arange(LMAX + 1, dtype=float), cl_full)


def hydro_file(tag):
    if tag == "fullsky":
        return DATA / "Dl_yy_L1_m9_fullsky_binned_18.txt"
    return DATA / f"Dl_yy_L1_m9_masked_{tag}_qfrommap_binned_18.txt"


def plot_vs_hydro(results):
    import matplotlib.pyplot as plt

    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    labels = [("fullsky", "full sky")] + [(tag, rf"$q>{cut:g}$") for tag, cut in Q_CUTS]
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    for (tag, label), color in zip(labels, colors):
        hydro = np.loadtxt(hydro_file(tag))
        ax.plot(hydro[:, 0], hydro[:, 1], "o", color=color, ms=3.8, zorder=3)
        ax.plot(ELL_EFF, results[tag], color=color, lw=2.0, zorder=2, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(ELL_EFF[0]), float(ELL_EFF[-1]))
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$10^{12}D_\ell^{yy}$")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        out = HERE / f"painted_vs_hydro_qfrommap_{OUT_TAG}.{suffix}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out, flush=True)
    plt.close(fig)


def deconvolve_beam(m, fwhm_arcmin=FWHM_ARCMIN, lmax=LMAX):
    """Harmonic-space 10' Gaussian deconvolution, same beam as the PS."""
    bl = hp.gauss_beam(fwhm=np.deg2rad(fwhm_arcmin / 60.0), lmax=lmax)
    alm = hp.map2alm(np.asarray(m, dtype=np.float64), lmax=lmax)
    return hp.alm2map(hp.almxfl(alm, 1.0 / np.maximum(bl, 1e-2)), nside=hp.get_nside(m), lmax=lmax)


def plot_one_point_pdf(ymap):
    """Pixel-y PDF: beam-deconvolved painted map vs unbeamed hydro, same nside."""
    import matplotlib.pyplot as plt

    painted = deconvolve_beam(ymap)
    hydro = hp.ud_grade(hp.read_map(str(HYDRO_MAP), dtype=np.float64), nside_out=NSIDE)
    lo = float(min(painted.min(), hydro.min(), 0.0))
    hi = float(np.percentile(np.concatenate([painted, hydro]), 99.9))
    bins = np.linspace(lo, hi, 80)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for arr, label, ls in (
        (painted, r"painted (beam-deconvolved)", "-"),
        (hydro, r"L1_m9 hydro ($N_{\mathrm{side}}=1024$)", "--"),
    ):
        ax.hist(arr, bins=bins, density=True, histtype="step", lw=1.8, ls=ls, label=label)
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$P(y)$")
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        out = HERE / f"painted_vs_hydro_onepoint_pdf_{OUT_TAG}.{suffix}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out, flush=True)
    plt.close(fig)


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    map_path = HERE / f"l1_m9_{OUT_TAG}_d3a_asz_fullsky_nside1024.fits"
    print("=== catalogue ===", flush=True)
    cat = load_catalogue(CAT_FILE, need_q=False)
    mask_cat = load_catalogue(CAT_MASK, need_q=True)
    print(
        f"  paint {cat['M'].size:,} halos  M_1e14=[{cat['M'].min():.3f}, {cat['M'].max():.3f}]",
        flush=True,
    )
    print(f"  mask catalogue {mask_cat['M'].size:,} (5e13 qfrommap)", flush=True)
    print(
        f"  inference: A_SZ={A_SZ}  alpha_SZ={ALPHA_SZ}  B={B}  "
        f"H0={H0}  Omega_m={OMEGA_M}  GNFW={GNFW_SHAPE}",
        flush=True,
    )
    if B != B_HYDROSTATIC:
        raise RuntimeError(f"FIXED B={B} != B_HYDROSTATIC={B_HYDROSTATIC}")

    print("=== paint ===", flush=True)
    ymap = paint_map(cat)
    hp.write_map(str(map_path), ymap, dtype=np.float64, overwrite=True)
    print("wrote", map_path, flush=True)

    import pymaster as nmt

    beam = hp.gauss_beam(fwhm=np.deg2rad(FWHM_ARCMIN / 60.0), lmax=LMAX)
    floor = np.deg2rad(2.0 * FWHM_ARCMIN / 60.0)
    results = {"fullsky": fullsky_dl(ymap, beam)}
    np.savetxt(
        HERE / f"Dl_yy_painted_l1_m9_{OUT_TAG}_fullsky_binned_18.txt",
        np.column_stack([ELL_EFF, results["fullsky"]]),
        fmt="%.6e",
        header="ell_eff  1e12_D_ell_yy  (beam-deconvolved, no pixwin)",
    )
    print("fullsky Dl[0], Dl[-1] =", results["fullsky"][0], results["fullsky"][-1], flush=True)

    for tag, cut in Q_CUTS:
        keep = mask_cat["q"] > cut
        radius = np.maximum(R_MULT * mask_cat["t500"][keep], floor)
        print(f"=== {tag}: {int(keep.sum()):,} holes ===", flush=True)
        mask_bin = binary_disc_mask(
            NSIDE, mask_cat["theta"][keep], mask_cat["phi"][keep], radius
        )
        mask_apo = nmt.mask_apodization(mask_bin, APOSIZE_DEG, apotype="C2")
        print(f"  f_sky raw={mask_bin.mean():.4f}  eff={np.mean(mask_apo**2):.4f}", flush=True)
        results[tag] = masked_dl(ymap, mask_apo, beam)
        np.savetxt(
            HERE / f"Dl_yy_painted_l1_m9_{OUT_TAG}_masked_{tag}_binned_18.txt",
            np.column_stack([ELL_EFF, results[tag]]),
            fmt="%.6e",
            header=(
                f"ell_eff  1e12_D_ell_yy  q>{cut:g} qfrommap; "
                "r=max(4*theta500, 2x10arcmin); C2 0.25 deg; beam-deconvolved, no pixwin"
            ),
        )

    plot_vs_hydro(results)
    print("=== 1-point PDF ===", flush=True)
    plot_one_point_pdf(ymap)
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
