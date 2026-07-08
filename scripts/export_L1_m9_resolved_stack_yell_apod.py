"""Per-cluster y_ell of resolved (q>5) clusters with apodized apertures and analytic background.

Successor to ``scripts/export_L1_m9_resolved_stack_yell.py`` (kept untouched) for
``notebooks/44_resolved_tsz_ps_analytic_bg.ipynb``. Two changes to the estimator:

1. **Apodized aperture.** The hard disc is replaced by W(r) = 1 for r <= X_AP*theta_500,
   a cos^2 (Hann) taper down to 0 at (X_AP + X_TAPER)*theta_500. Every masked stamp is
   apodized before the FFT.
2. **Analytic background (MASTER forward mode).** Instead of noisy per-cluster random
   apertures, the expected background pseudo-power under each cluster's window is
   predicted deterministically from the masked-sky power spectrum:

       B_i(ell) = int d^2 ell' / (2 pi)^2  |W_i(ell - ell')|^2  C_bg(ell'),

   evaluated exactly on the stamp FFT grid by circular convolution:
   ``B2D = ifft2(fft2(|W_ell|^2) * fft2(C_grid)).real / (MGRID * res_rad)^2``.
   C_bg is the decoupled NaMaster q>5 masked bandpower (Dl_yy_L1_m9_qgt5.txt),
   power-law extrapolated beyond its lmax and re-multiplied by the healpy pixel
   window, so it describes the pixelized map the stamps actually sample.
   The DC offset term ``delta^2 |W_i(ell)|^2`` (masked-sky mean minus global
   monopole) is stored separately.

Modes:
    python scripts/export_L1_m9_resolved_stack_yell_apod.py --selftest
        Gaussian-field normalization check of the analytic prediction (no map needed).
    python scripts/export_L1_m9_resolved_stack_yell_apod.py [--nmax 20]
        Per-cluster spectra + analytic backgrounds ->
        data/nb44_resolved_stack_yell_apod/L1_m9_qgt5_order{ORDER}_apod.npz
    python scripts/export_L1_m9_resolved_stack_yell_apod.py --ensemble 50
        Validation ensemble: N random apertures at each of N_THETA theta_500
        quantiles, same apodized windows -> ..._ensemble50.npz
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.fft

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import healpy as hp  # noqa: E402
from pixell import enmap, reproject  # noqa: E402

from flamingo.geometry import ARCMIN_PER_RAD, query_disc_separation  # noqa: E402
from hmfast.cosmology import Cosmology  # noqa: E402

MAP_PATH = Path("/rds/rds-lxu/flamingo/L1_m9/maps/y_unlensed_L1_m9_lc0_nside4096.fits")
CAT_PATH = Path(
    "/rds/rds-lxu/flamingo/L1_m9/catalogues/"
    "halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommz.csv"
)
BG_DL_PATH = _REPO / "data/bandpowers_L1_m9_feedback/Dl_yy_L1_m9_qgt5.txt"
OUT_DIR = _REPO / "data/nb44_resolved_stack_yell_apod"

Q_CUT = 5.0
X_AP = 5.0  # W = 1 inside X_AP * theta_500 (matches the 5 theta_500 mask discs)
X_TAPER = 1.0  # cos^2 taper over [X_AP, X_AP + X_TAPER] * theta_500
RES_ARCMIN = 0.45
MGRID = 2048
DELL = 25.0
LMAX_BIN = 6200.0
CHECKPOINT = 200
NSIDE = 4096
LMAX_EXT = 12288  # 3 * NSIDE: C_bg power-law extrapolated to here, zero beyond
N_THETA = 12  # theta_500 quantiles for the --ensemble validation grid

A_S_D3A = 2.099e-9
D3A = dict(
    H0=68.1,
    omega_b=0.022539,
    omega_cdm=0.118729,
    n_s=0.967,
    tau_reio=0.0544,
    ln1e10A_s=float(np.log(1e10 * A_S_D3A)),
)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_sample() -> pd.DataFrame:
    cat = pd.read_csv(CAT_PATH, comment="#")
    sel = np.isfinite(cat["q_from_mz"]) & (cat["q_from_mz"] > Q_CUT)
    cat = cat[sel].reset_index(drop=True)
    cosmo = Cosmology(**D3A)
    dA = np.asarray(cosmo.angular_diameter_distance(cat["z"].to_numpy()), dtype=np.float64)
    cat["dA_Mpc"] = dA
    cat["theta500_arcmin"] = cat["R_500c_Mpc"].to_numpy() / dA * ARCMIN_PER_RAD
    return cat.sort_values("theta500_arcmin", ascending=False).reset_index(drop=True)


def background_cl_grid(res_rad: float, apply_pixwin: bool = True) -> np.ndarray:
    """C_bg(|ell|) of the pixelized masked sky on the MGRID x MGRID FFT grid."""
    dl_tab = np.loadtxt(BG_DL_PATH)
    ell_t, dl_t = dl_tab[:, 0], dl_tab[:, 1]
    cl_t = 2.0 * np.pi * dl_t / (ell_t * (ell_t + 1.0))
    # power-law extrapolation from the last decade of measured bandpowers
    tail = ell_t > ell_t[-1] / 2.0
    slope = np.polyfit(np.log(ell_t[tail]), np.log(cl_t[tail]), 1)[0]
    ell_fine = np.arange(2.0, LMAX_EXT + 1.0)
    cl_fine = np.exp(np.interp(np.log(ell_fine), np.log(ell_t), np.log(cl_t)))
    ext = ell_fine > ell_t[-1]
    cl_fine[ext] = cl_t[-1] * (ell_fine[ext] / ell_t[-1]) ** slope
    if apply_pixwin:
        pw = hp.pixwin(NSIDE, lmax=LMAX_EXT)
        cl_fine *= np.interp(ell_fine, np.arange(pw.size), pw) ** 2
    lfreq = 2.0 * np.pi * np.fft.fftfreq(MGRID, d=res_rad)
    lmap = np.hypot(*np.meshgrid(lfreq, lfreq, indexing="ij"))
    cgrid = np.interp(lmap, ell_fine, cl_fine, left=cl_fine[0], right=0.0)
    cgrid[lmap > LMAX_EXT] = 0.0
    cgrid[0, 0] = 0.0  # monopole handled separately via the DC-offset term
    return cgrid


def taper_window(r: np.ndarray, theta500_arcmin: float) -> np.ndarray:
    """W(r): 1 inside X_AP*theta_500, cos^2 taper to 0 at (X_AP + X_TAPER)*theta_500."""
    r_in = X_AP * theta500_arcmin / ARCMIN_PER_RAD
    r_out = (X_AP + X_TAPER) * theta500_arcmin / ARCMIN_PER_RAD
    w = np.zeros_like(r)
    w[r <= r_in] = 1.0
    sel = (r > r_in) & (r < r_out)
    w[sel] = np.cos(0.5 * np.pi * (r[sel] - r_in) / (r_out - r_in)) ** 2
    return w


def cluster_yell2_apod(
    ymap: np.ndarray | None,
    monopole: float,
    lon_deg: float,
    lat_deg: float,
    theta500_arcmin: float,
    *,
    order: int,
    res_rad: float,
    bin_idx: np.ndarray,
    nbin: int,
    counts: np.ndarray,
    fc_grid: np.ndarray,
    stamp_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Azimuthal |y_ell|^2, analytic bg, |W_ell|^2 (all sr^2), plus y0 and window area.

    Returns ``(abs_y2, bg_pred, w2, y0, area_w)`` where ``area_w = int W d^2x`` (sr).
    ``stamp_override`` (selftest) bypasses the healpix reprojection.
    """
    r_out = (X_AP + X_TAPER) * theta500_arcmin / ARCMIN_PER_RAD
    npix = int(np.ceil(2.0 * r_out / res_rad))
    npix = min(npix + npix % 2, MGRID)
    dec, ra = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    shape, wcs = enmap.geometry(pos=(dec, ra), shape=(npix, npix), res=res_rad, proj="tan")
    r = np.asarray(enmap.modrmap(shape, wcs))
    win = taper_window(r, theta500_arcmin)

    if stamp_override is not None:
        stamp = stamp_override[:npix, :npix].astype(np.float64)
    else:
        stamp = reproject.healpix2map(ymap, shape, wcs, rot=None, method="spline", order=order)
        stamp = np.asarray(stamp, dtype=np.float64) - monopole
    stamp = stamp * win

    omega_pix = res_rad * res_rad
    padded = np.zeros((MGRID, MGRID), dtype=np.float64)
    padded[:npix, :npix] = stamp
    fmap = scipy.fft.fft2(padded, workers=8)
    y0 = float(np.abs(fmap[0, 0])) * omega_pix
    amp2 = (np.abs(fmap) * omega_pix) ** 2

    padded[:] = 0.0
    padded[:npix, :npix] = win
    fwin = scipy.fft.fft2(padded, workers=8)
    w2_grid = (np.abs(fwin) * omega_pix) ** 2
    area_w = float(np.abs(fwin[0, 0])) * omega_pix

    # analytic background: circular convolution of |W_ell|^2 with C_bg on the grid
    box = MGRID * res_rad
    bg2d = scipy.fft.ifft2(
        scipy.fft.fft2(w2_grid, workers=8) * fc_grid, workers=8
    ).real / (box * box)

    abs_y2 = np.bincount(bin_idx, weights=amp2.ravel(), minlength=nbin)[:nbin] / counts
    bg_pred = np.bincount(bin_idx, weights=bg2d.ravel(), minlength=nbin)[:nbin] / counts
    w2 = np.bincount(bin_idx, weights=w2_grid.ravel(), minlength=nbin)[:nbin] / counts
    return abs_y2, bg_pred, w2, y0, area_w


def random_centres(
    cat: pd.DataFrame, n_tot: int, nside: int, *, seed: int = 44
) -> tuple[np.ndarray, np.ndarray]:
    """n_tot random lon/lat (deg) outside the q>5 mask discs (radius X_AP*theta_500)."""
    print("building q>5 binary mask for random-centre rejection ...", flush=True)
    mask = np.ones(hp.nside2npix(nside), dtype=bool)
    theta_rot = np.pi / 2.0 - np.deg2rad(cat["lat_rot_deg"].to_numpy())
    phi_rot = np.deg2rad(cat["lon_rot_deg"].to_numpy())
    for i in range(len(cat)):
        r_rad = X_AP * cat["theta500_arcmin"].iloc[i] / ARCMIN_PER_RAD
        pix, _ = query_disc_separation(nside, theta_rot[i], phi_rot[i], r_rad)
        mask[pix] = False
    print(f"  mask f_sky = {mask.mean():.4f}", flush=True)

    rng = np.random.default_rng(seed)
    lon, lat = [], []
    while len(lon) < n_tot:
        n_draw = 2 * (n_tot - len(lon)) + 64
        lon_d = rng.uniform(0.0, 360.0, n_draw)
        lat_d = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n_draw)))
        pix = hp.ang2pix(nside, lon_d, lat_d, lonlat=True)
        keep = mask[pix]
        lon.extend(lon_d[keep])
        lat.extend(lat_d[keep])
    return np.array(lon[:n_tot]), np.array(lat[:n_tot])


def make_bins(res_rad: float) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    lfreq = 2.0 * np.pi * np.fft.fftfreq(MGRID, d=res_rad)
    lmap = np.hypot(*np.meshgrid(lfreq, lfreq, indexing="ij")).ravel()
    edges = np.arange(0.0, LMAX_BIN + DELL, DELL)
    nbin = edges.size - 1
    ell_b = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip((lmap / DELL).astype(np.int64), 0, nbin)
    counts = np.bincount(bin_idx, minlength=nbin + 1)[:nbin].astype(np.float64)
    counts[counts == 0] = 1.0
    return ell_b, bin_idx, nbin, counts


def selftest(order: int) -> None:
    """GRF check: simulate fields with C_bg, window them, compare to the prediction."""
    res_rad = RES_ARCMIN / ARCMIN_PER_RAD
    ell_b, bin_idx, nbin, counts = make_bins(res_rad)
    cgrid = background_cl_grid(res_rad, apply_pixwin=False)
    fc_grid = scipy.fft.fft2(cgrid, workers=8)

    rng = np.random.default_rng(2026)
    theta500 = 8.0  # arcmin; r_out = 48' well inside the 2048 * 0.45' stamp
    n_sim = 40
    acc = np.zeros(nbin)
    bg_pred = None
    for _ in range(n_sim):
        # GRF on the periodic box normalized so that <|Omega_pix FFT(grf)|^2> = C(l) * A_box:
        # fft2 of unit white noise has <|.|^2> = MGRID^2, so scale by sqrt(C) / res_rad.
        white = scipy.fft.fft2(rng.standard_normal((MGRID, MGRID)), workers=8)
        grf = scipy.fft.ifft2(white * np.sqrt(cgrid) / res_rad, workers=8).real
        abs_y2, pred, _, _, _ = cluster_yell2_apod(
            None, 0.0, 0.0, 0.0, theta500,
            order=order, res_rad=res_rad, bin_idx=bin_idx, nbin=nbin, counts=counts,
            fc_grid=fc_grid, stamp_override=grf,
        )
        acc += abs_y2
        bg_pred = pred
    acc /= n_sim
    sel = (ell_b > 300) & (ell_b < 6000)
    ratio = acc[sel] / bg_pred[sel]
    print(f"selftest ({n_sim} GRFs, theta500 = {theta500}'):")
    print(f"  mean ratio sim/pred (300 < ell < 6000): {ratio.mean():.4f}")
    print(f"  median: {np.median(ratio):.4f},  16-84%: {np.percentile(ratio, [16, 84])}")
    ok = abs(np.median(ratio) - 1.0) < 0.02
    print("  PASS" if ok else "  FAIL")
    sys.exit(0 if ok else 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--order", type=int, default=0, help="spline interpolation order")
    p.add_argument("--nmax", type=int, default=None)
    p.add_argument("--ensemble", type=int, default=0, help="random apertures per theta_500 node")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.selftest:
        selftest(args.order)

    suffix = f"_ensemble{args.ensemble}" if args.ensemble else "_apod"
    out = args.out or OUT_DIR / f"L1_m9_qgt5_order{args.order}{suffix}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial.npz")

    t0 = time.time()
    res_rad = RES_ARCMIN / ARCMIN_PER_RAD
    ell_b, bin_idx, nbin, counts = make_bins(res_rad)
    cgrid = background_cl_grid(res_rad)
    fc_grid = scipy.fft.fft2(cgrid, workers=8)

    print(f"loading {MAP_PATH.name} ...", flush=True)
    ymap = hp.read_map(MAP_PATH)
    monopole = float(ymap.mean())
    print(f"  monopole = {monopole:.4e}", flush=True)

    cat = load_sample()
    if args.nmax is not None and not args.ensemble:
        cat = cat.iloc[: args.nmax]

    if args.ensemble:
        # validation grid: N_THETA theta_500 quantiles x args.ensemble random centres
        qs = np.linspace(0.02, 0.98, N_THETA)
        theta_nodes = np.quantile(cat["theta500_arcmin"].to_numpy(), qs)
        n_cl = N_THETA * args.ensemble
        theta_list = np.repeat(theta_nodes, args.ensemble)
        nside = hp.npix2nside(ymap.size)
        lon_list, lat_list = random_centres(cat, n_cl, nside)
        # masked-sky mean offset for the DC term (same rejection region as the centres)
        print(f"ensemble: {N_THETA} theta nodes x {args.ensemble} randoms", flush=True)
    else:
        n_cl = len(cat)
        theta_list = cat["theta500_arcmin"].to_numpy()
        lon_list = cat["lon_rot_deg"].to_numpy()
        lat_list = cat["lat_rot_deg"].to_numpy()
    print(f"stamps: {n_cl} (q_from_mz > {Q_CUT:g})", flush=True)

    abs_y2 = np.zeros((n_cl, nbin))
    bg_pred = np.zeros((n_cl, nbin))
    w2 = np.zeros((n_cl, nbin))
    y0 = np.zeros(n_cl)
    area_w = np.zeros(n_cl)
    i_start = 0
    if partial.exists():
        prev = np.load(partial)
        if prev["abs_y2"].shape == abs_y2.shape and int(prev["n_done"]) < n_cl:
            i_start = int(prev["n_done"])
            for name, arr in (("abs_y2", abs_y2), ("bg_pred", bg_pred), ("w2", w2),
                              ("y0", y0), ("area_w", area_w)):
                arr[:i_start] = prev[name][:i_start]
            print(f"resuming from {partial.name} at stamp {i_start}", flush=True)

    t1 = time.time()
    for i in range(i_start, n_cl):
        abs_y2[i], bg_pred[i], w2[i], y0[i], area_w[i] = cluster_yell2_apod(
            ymap,
            monopole,
            float(lon_list[i]),
            float(lat_list[i]),
            float(theta_list[i]),
            order=args.order,
            res_rad=res_rad,
            bin_idx=bin_idx,
            nbin=nbin,
            counts=counts,
            fc_grid=fc_grid,
        )
        if (i + 1) % CHECKPOINT == 0 or i + 1 == n_cl:
            np.savez(partial, abs_y2=abs_y2, bg_pred=bg_pred, w2=w2, y0=y0,
                     area_w=area_w, n_done=i + 1)
            rate = (time.time() - t1) / (i + 1 - i_start)
            print(f"  {i + 1}/{n_cl}  ({rate:.2f} s/stamp)", flush=True)

    payload = dict(
        ell_b=ell_b,
        abs_y2=abs_y2,
        bg_pred=bg_pred,
        w2=w2,
        y0=y0,
        area_w=area_w,
        counts=counts,
        theta500_arcmin=theta_list,
        monopole=monopole,
        q_cut=Q_CUT,
        x_ap=X_AP,
        x_taper=X_TAPER,
        res_arcmin=RES_ARCMIN,
        mgrid=MGRID,
        dell=DELL,
        order=args.order,
        bg_dl_path=str(BG_DL_PATH),
        map_path=str(MAP_PATH),
        cat_path=str(CAT_PATH),
    )
    if not args.ensemble:
        payload.update(
            z=cat["z"].to_numpy(),
            M_500c_Msun=cat["M_500c_Msun"].to_numpy(),
            q_from_mz=cat["q_from_mz"].to_numpy(),
            Y_5R500c_Mpc2=cat["Y_5R500c_Mpc2"].to_numpy(),
            dA_Mpc=cat["dA_Mpc"].to_numpy(),
        )
    np.savez(out, **payload)
    partial.unlink(missing_ok=True)
    manifest = dict(
        git_hash=git_hash(),
        mode="ensemble" if args.ensemble else "clusters",
        n_stamps=n_cl,
        n_per_theta=args.ensemble,
        order=args.order,
        q_cut=Q_CUT,
        x_ap=X_AP,
        x_taper=X_TAPER,
        res_arcmin=RES_ARCMIN,
        mgrid=MGRID,
        dell=DELL,
        monopole=monopole,
        bg_dl_path=str(BG_DL_PATH),
        map_path=str(MAP_PATH),
        cat_path=str(CAT_PATH),
        runtime_s=round(time.time() - t0, 1),
        output=str(out),
    )
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}  ({manifest['runtime_s']} s total)", flush=True)


if __name__ == "__main__":
    main()
