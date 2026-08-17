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
from functools import lru_cache
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
MAP_DIR = HYDRO_MAP.parent
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
Q_PDF_CUT = 5.0
BATCH = 5000
N_PER_HALO = 30_000
E_PER_HALO = 64
# 5e13 has no separate paint catalogue; same halos as the qfrommap mask file.
PAINT_JOBS = (
    ("5e13", CAT_MASK, r"painted $M>5\times 10^{13}$", "#1f77b4"),
    ("m1e13", CAT_FILE, r"painted $M>10^{13}$", "#d95f02"),
)


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


def paint_map(cat, ckpt=None):
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
    i_start = 0
    ymap = np.zeros(hp.nside2npix(NSIDE), dtype=np.float64)
    if ckpt is not None and Path(ckpt).exists():
        data = np.load(ckpt)
        ymap = np.asarray(data["ymap"], dtype=np.float64)
        i_start = int(data["i1"])
        print(f"  resume {ckpt} from {i_start:,}/{n:,}", flush=True)
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

    def save_ckpt(i1):
        if ckpt is None:
            return
        # ponytail: GPU abort is uncatchable; dump ymap so a death only costs one print cadence.
        np.savez(ckpt, ymap=ymap, i1=np.int64(i1))
        print(f"  ckpt {ckpt} i1={i1:,}", flush=True)

    for i0 in range(i_start, n, BATCH):
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
        if i0 == i_start or (i0 // BATCH) % 50 == 0:
            print(
                f"  painted {i1:,}/{n:,} ({time.time() - t0:.0f}s) "
                f"z=[{cat['z'][i0:i1].min():.3f},{cat['z'][i0:i1].max():.3f}]",
                flush=True,
            )
            save_ckpt(i1)
    save_ckpt(n)
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


def painted_map_path(tag):
    """Durable Nside=1024 FITS, next to the hydro y map. Repo copy is extra."""
    return MAP_DIR / f"y_painted_L1_m9_{tag}_d3a_asz_nside1024.fits"


def painted_map_repo_path(tag):
    return HERE / f"l1_m9_{tag}_d3a_asz_fullsky_nside1024.fits"


def write_painted_map(tag, ymap):
    ymap = np.asarray(ymap, dtype=np.float64)
    nside = hp.get_nside(ymap)
    if nside != NSIDE:
        raise RuntimeError(f"painted nside={nside} != {NSIDE}")
    paths = (painted_map_path(tag), painted_map_repo_path(tag))
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        hp.write_map(str(path), ymap, dtype=np.float64, overwrite=True)
        print("wrote", path, f"nside={nside}", flush=True)
    return paths[0]


@lru_cache(maxsize=1)
def load_hydro_nside():
    return hp.ud_grade(hp.read_map(str(HYDRO_MAP), dtype=np.float64), nside_out=NSIDE)


@lru_cache(maxsize=4)
def qfrommap_binary_mask(cut=Q_PDF_CUT):
    """Binary qfrommap holes, same discs as the painted PS (no apodization)."""
    cat = load_catalogue(CAT_MASK, need_q=True)
    keep = cat["q"] > float(cut)
    floor = np.deg2rad(2.0 * FWHM_ARCMIN / 60.0)
    radius = np.maximum(R_MULT * cat["t500"][keep], floor)
    print(f"  qfrommap q>{cut:g}: {int(keep.sum()):,} holes", flush=True)
    return binary_disc_mask(NSIDE, cat["theta"][keep], cat["phi"][keep], radius)


def _pdf_series(painted):
    hydro = load_hydro_nside()
    mask = qfrommap_binary_mask(Q_PDF_CUT)
    keep = mask > 0.5
    qlab = rf"$q>{Q_PDF_CUT:g}$ qfrommap"
    series = [
        (hydro, r"L1_m9 hydro", "k", "-"),
        (hydro[keep], rf"L1_m9 hydro {qlab}", "k", "--"),
    ]
    labels = {tag: lab for tag, _, lab, _ in PAINT_JOBS}
    colors = {tag: color for tag, _, _, color in PAINT_JOBS}
    for tag, ymap in painted.items():
        arr = deconvolve_beam(ymap)
        lab = labels.get(tag, f"painted {tag}")
        color = colors.get(tag, "#1b9e77")
        series.append((arr, lab, color, "-"))
        series.append((arr[keep], rf"{lab} {qlab}", color, "--"))
    return series


def plot_one_point_pdf(ymap):
    """Pixel-y PDF before/after qfrommap masking: painted vs unbeamed hydro."""
    import matplotlib.pyplot as plt

    painted = ymap if isinstance(ymap, dict) else {OUT_TAG: ymap}
    series = _pdf_series(painted)
    stacked = np.concatenate([np.asarray(a, dtype=np.float64) for a, *_ in series])
    lo = float(min(stacked.min(), 0.0))
    hi = float(np.percentile(stacked, 99.9))
    bins = np.linspace(lo, hi, 80)
    pos = stacked[stacked > 0.0]
    log_lo = float(max(np.percentile(pos, 0.1), 1e-9)) if pos.size else 1e-9
    log_hi = float(np.percentile(pos, 99.99)) if pos.size else 1e-5
    log_bins = np.logspace(np.log10(log_lo), np.log10(log_hi), 80)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    ax, axlog = axes
    for arr, label, color, ls in series:
        ax.hist(arr, bins=bins, density=True, histtype="step", lw=1.6, ls=ls, color=color, label=label)
        axlog.hist(
            np.asarray(arr)[np.asarray(arr) > 0.0],
            bins=log_bins,
            density=True,
            histtype="step",
            lw=1.6,
            ls=ls,
            color=color,
        )
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$P(y)$")
    ax.legend(frameon=False, fontsize=8)
    axlog.set_xlabel(r"$y$")
    axlog.set_xscale("log")
    axlog.set_yscale("log")
    axlog.set_ylabel(r"$P(y)$")
    fig.tight_layout()
    tag = "qfrommap" if len(painted) != 1 else f"{next(iter(painted))}_qfrommap"
    for suffix in ("png", "pdf"):
        out = HERE / f"painted_vs_hydro_onepoint_pdf_{tag}.{suffix}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print("wrote", out, flush=True)
    plt.close(fig)


def ensure_painted(tag, cat_file):
    for path in (painted_map_path(tag), painted_map_repo_path(tag)):
        if path.exists():
            print(f"  load {path}", flush=True)
            return hp.read_map(str(path), dtype=np.float64)
    print(f"=== paint {tag} ===", flush=True)
    ckpt = painted_map_repo_path(tag).with_name(f"l1_m9_{tag}_paint_ckpt.npz")
    ymap = paint_map(load_catalogue(cat_file, need_q=False), ckpt=ckpt)
    write_painted_map(tag, ymap)
    Path(ckpt).unlink(missing_ok=True)
    return ymap


def pdf_main():
    """Hydro PDF immediately; paint missing 5e13 then 1e13 maps and replot."""
    HERE.mkdir(parents=True, exist_ok=True)
    painted = {}
    print("=== 1-point PDF (hydro, qfrommap) ===", flush=True)
    plot_one_point_pdf(painted)
    for tag, cat_file, *_ in PAINT_JOBS:
        painted[tag] = ensure_painted(tag, cat_file)
        print(f"=== 1-point PDF (+{tag}) ===", flush=True)
        plot_one_point_pdf(painted)
    print("=== done ===", flush=True)


def main():
    HERE.mkdir(parents=True, exist_ok=True)
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
    write_painted_map(OUT_TAG, ymap)

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
    if "--pdf" in sys.argv:
        pdf_main()
    else:
        main()
