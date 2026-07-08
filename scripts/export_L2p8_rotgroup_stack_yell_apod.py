"""nb44 redo of the nb42 rotation-group stamps: apodized windows + analytic background.

Successor to ``scripts/export_L2p8_rotgroup_stack_yell.py`` (kept untouched). Reuses the
nb42 stratified subsamples (``sample_group{g}.npz``) and measured group spectra
(``measured_dl_group{g}.npz``) so results are directly comparable. Changes:

1. apodized aperture: W(r) = 1 inside X_AP*theta_500, cos^2 taper to 0 at
   (X_AP + X_TAPER)*theta_500; stamps zero-padded to MGRID^2 (no hard edges);
2. deterministic per-cluster background B_i(ell) = |W_i|^2 (x) C_bg on the FFT grid,
   with C_bg = the group map's measured full-sky C_ell re-multiplied by the HEALPix
   pixel window (stamps sample the pixelized map), power-law extrapolated past lmax;
3. coherent phase-centred Re[y_ell] kept (nb42 convention) for the faint strata;
4. ``--stage ensemble``: random apertures at 12 theta_500 quantile nodes (uniform
   sphere, nb42 randoms convention) for the per-group transfer T-hat_g(ell).

The group maps subtract their own monopole and randoms are uniform, so the nb44-L1_m9
DC-offset term is zero here by construction.

Run (per group g = 0..4):
    python scripts/export_L2p8_rotgroup_stack_yell_apod.py --stage stamp --group g
    python scripts/export_L2p8_rotgroup_stack_yell_apod.py --stage ensemble --group g --n 100
Outputs: data/nb44_rotgroup_stack_yell_apod/group{g}_apod.npz, group{g}_ensemble{N}.npz.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.fft

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import healpy as hp  # noqa: E402
from pixell import enmap, reproject  # noqa: E402

from flamingo.geometry import ARCMIN_PER_RAD  # noqa: E402

MAP_DIR = Path("/rds/rds-lxu/flamingo/L2p8_m9/lightcone0/healpix_map/rotation_groups")
META_PATH = MAP_DIR / "rotation_groups_L2p8_m9_lc0.json"
NB42_DIR = _REPO / "data/nb42_rotgroup_stack_yell"
OUT_DIR = _REPO / "data/nb44_rotgroup_stack_yell_apod"

X_AP = 5.0
X_TAPER = 1.0
RES_ARCMIN = 0.45
MGRID = 2048
DELL = 25.0
LMAX_BIN = 6200.0
NSIDE = 4096
LMAX_EXT = 12288
N_THETA = 12
CHECKPOINT = 400
SEED = 44


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


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


def background_cl_grid(group: int, res_rad: float) -> np.ndarray:
    """C_bg(|ell|) of the pixelized group map on the FFT grid.

    The measured D_ell cache is pixwin- and fsky-corrected (nb39 convention), so the
    pixel window is re-applied to describe the pixelized map the stamps sample.
    """
    md = np.load(NB42_DIR / f"measured_dl_group{group}.npz")
    ell_t, dl_t = md["ell"], md["dl"]
    keep = ell_t >= 2
    ell_t, dl_t = ell_t[keep], dl_t[keep]
    cl_t = 2.0 * np.pi * dl_t / (ell_t * (ell_t + 1.0))
    tail = ell_t > ell_t[-1] / 2.0
    slope = np.polyfit(np.log(ell_t[tail]), np.log(cl_t[tail]), 1)[0]
    ell_fine = np.arange(2.0, LMAX_EXT + 1.0)
    cl_fine = np.exp(np.interp(np.log(ell_fine), np.log(ell_t), np.log(cl_t)))
    ext = ell_fine > ell_t[-1]
    cl_fine[ext] = cl_t[-1] * (ell_fine[ext] / ell_t[-1]) ** slope
    pw = hp.pixwin(NSIDE, lmax=LMAX_EXT)
    cl_fine *= np.interp(ell_fine, np.arange(pw.size), pw) ** 2
    lfreq = 2.0 * np.pi * np.fft.fftfreq(MGRID, d=res_rad)
    lmap = np.hypot(*np.meshgrid(lfreq, lfreq, indexing="ij"))
    cgrid = np.interp(lmap, ell_fine, cl_fine, left=cl_fine[0], right=0.0)
    cgrid[lmap > LMAX_EXT] = 0.0
    cgrid[0, 0] = 0.0
    return cgrid


def taper_window(r: np.ndarray, theta500_arcmin: float) -> np.ndarray:
    r_in = X_AP * theta500_arcmin / ARCMIN_PER_RAD
    r_out = (X_AP + X_TAPER) * theta500_arcmin / ARCMIN_PER_RAD
    w = np.zeros_like(r)
    w[r <= r_in] = 1.0
    sel = (r > r_in) & (r < r_out)
    w[sel] = np.cos(0.5 * np.pi * (r[sel] - r_in) / (r_out - r_in)) ** 2
    return w


def cluster_yell2_apod(
    ymap: np.ndarray,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Azimuthal |y_ell|^2, coherent Re[y_ell], analytic bg, |W_ell|^2, and y0."""
    r_out = (X_AP + X_TAPER) * theta500_arcmin / ARCMIN_PER_RAD
    npix = int(np.ceil(2.0 * r_out / res_rad))
    npix = min(max(npix + npix % 2, 8), MGRID)
    dec, ra = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    shape, wcs = enmap.geometry(pos=(dec, ra), shape=(npix, npix), res=res_rad, proj="tan")
    r = np.asarray(enmap.modrmap(shape, wcs))
    win = taper_window(r, theta500_arcmin)
    stamp = reproject.healpix2map(ymap, shape, wcs, rot=None, method="spline", order=order)
    stamp = (np.asarray(stamp, dtype=np.float64) - monopole) * win

    omega_pix = res_rad * res_rad
    padded = np.zeros((MGRID, MGRID), dtype=np.float64)
    padded[:npix, :npix] = stamp
    fmap = scipy.fft.fft2(padded, workers=8)
    y0 = float(np.abs(fmap[0, 0])) * omega_pix
    amp2 = (np.abs(fmap) * omega_pix) ** 2

    py, px = enmap.sky2pix(shape, wcs, np.array([dec, ra]))
    f1 = np.fft.fftfreq(MGRID)
    phase = np.exp(2j * np.pi * f1 * float(py))[:, None] * np.exp(
        2j * np.pi * f1 * float(px)
    )[None, :]
    re = (fmap * phase).real.ravel() * omega_pix

    padded[:] = 0.0
    padded[:npix, :npix] = win
    fwin = scipy.fft.fft2(padded, workers=8)
    w2_grid = (np.abs(fwin) * omega_pix) ** 2
    box = MGRID * res_rad
    bg2d = scipy.fft.ifft2(
        scipy.fft.fft2(w2_grid, workers=8) * fc_grid, workers=8
    ).real / (box * box)

    abs_y2 = np.bincount(bin_idx, weights=amp2.ravel(), minlength=nbin)[:nbin] / counts
    re_y = np.bincount(bin_idx, weights=re, minlength=nbin)[:nbin] / counts
    bg_pred = np.bincount(bin_idx, weights=bg2d.ravel(), minlength=nbin)[:nbin] / counts
    w2 = np.bincount(bin_idx, weights=w2_grid.ravel(), minlength=nbin)[:nbin] / counts
    return abs_y2, re_y, bg_pred, w2, y0


def run(group: int, order: int, nmax: int | None, ensemble: int) -> None:
    meta = json.loads(META_PATH.read_text())
    ginfo = meta["groups"][group]
    assert ginfo["group"] == group
    map_path = MAP_DIR / ginfo["fits"]
    suffix = f"_ensemble{ensemble}" if ensemble else "_apod"
    out = OUT_DIR / f"group{group}{suffix}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial.npz")

    t0 = time.time()
    res_rad = RES_ARCMIN / ARCMIN_PER_RAD
    ell_b, bin_idx, nbin, counts = make_bins(res_rad)
    fc_grid = scipy.fft.fft2(background_cl_grid(group, res_rad))

    print(f"loading {map_path.name} ...", flush=True)
    ymap = hp.read_map(str(map_path), dtype=np.float64)
    monopole = float(ymap.mean())
    print(f"  monopole = {monopole:.4e}", flush=True)

    s = np.load(NB42_DIR / f"sample_group{group}.npz")
    if ensemble:
        qs = np.linspace(0.02, 0.98, N_THETA)
        theta_list = np.repeat(np.quantile(s["theta500_arcmin"], qs), ensemble)
        rng = np.random.default_rng(SEED + group)
        n_cl = theta_list.size
        lon_list = rng.uniform(0.0, 360.0, n_cl)
        lat_list = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n_cl)))
    else:
        theta_list = s["theta500_arcmin"]
        lon_list = s["lon_nat_deg"]
        lat_list = s["lat_nat_deg"]
        n_cl = len(theta_list) if nmax is None else min(nmax, len(theta_list))
    print(f"group {group}: {n_cl} stamps ({'ensemble' if ensemble else 'clusters'})", flush=True)

    abs_y2 = np.zeros((n_cl, nbin))
    re_y = np.zeros((n_cl, nbin))
    bg_pred = np.zeros((n_cl, nbin))
    w2 = np.zeros((n_cl, nbin))
    y0 = np.zeros(n_cl)
    i_start = 0
    if partial.exists():
        prev = np.load(partial)
        if prev["abs_y2"].shape == abs_y2.shape and int(prev["n_done"]) < n_cl:
            i_start = int(prev["n_done"])
            for name, arr in (("abs_y2", abs_y2), ("re_y", re_y), ("bg_pred", bg_pred),
                              ("w2", w2), ("y0", y0)):
                arr[:i_start] = prev[name][:i_start]
            print(f"resuming at stamp {i_start}", flush=True)

    t1 = time.time()
    for i in range(i_start, n_cl):
        abs_y2[i], re_y[i], bg_pred[i], w2[i], y0[i] = cluster_yell2_apod(
            ymap, monopole, float(lon_list[i]), float(lat_list[i]), float(theta_list[i]),
            order=order, res_rad=res_rad, bin_idx=bin_idx, nbin=nbin, counts=counts,
            fc_grid=fc_grid,
        )
        if (i + 1) % CHECKPOINT == 0 or i + 1 == n_cl:
            np.savez(partial, abs_y2=abs_y2, re_y=re_y, bg_pred=bg_pred, w2=w2, y0=y0,
                     n_done=i + 1)
            rate = (time.time() - t1) / (i + 1 - i_start)
            print(f"  {i + 1}/{n_cl}  ({rate:.2f} s/stamp)", flush=True)

    payload = dict(
        ell_b=ell_b, abs_y2=abs_y2, re_y=re_y, bg_pred=bg_pred, w2=w2, y0=y0,
        counts=counts, theta500_arcmin=theta_list[:n_cl], monopole=monopole,
        x_ap=X_AP, x_taper=X_TAPER, res_arcmin=RES_ARCMIN, mgrid=MGRID, dell=DELL,
        order=order, map_path=str(map_path),
    )
    if not ensemble:
        payload.update(
            logm_edges=s["logm_edges"], N_bin=s["N_bin"], n_bin=s["n_bin"],
            mbin=s["mbin"][:n_cl], z=s["z"][:n_cl],
            M_500c_Msun=s["M_500c_Msun"][:n_cl], dA_Mpc=s["dA_Mpc"][:n_cl],
            z_inner=ginfo["z_inner"], z_outer=ginfo["z_outer"],
        )
    np.savez(out, **payload)
    partial.unlink(missing_ok=True)
    manifest = dict(
        group=group, mode="ensemble" if ensemble else "clusters", n_stamps=int(n_cl),
        n_per_theta=ensemble, order=order, x_ap=X_AP, x_taper=X_TAPER,
        res_arcmin=RES_ARCMIN, mgrid=MGRID, dell=DELL, monopole=monopole,
        git_hash=git_hash(), map_path=str(map_path),
        runtime_s=round(time.time() - t0, 1), output=str(out),
    )
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}  ({manifest['runtime_s']} s total)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["stamp", "ensemble"], required=True)
    p.add_argument("--group", type=int, required=True)
    p.add_argument("--order", type=int, default=0)
    p.add_argument("--nmax", type=int, default=None)
    p.add_argument("--n", type=int, default=100, help="ensemble randoms per theta node")
    args = p.parse_args()
    run(args.group, args.order, args.nmax, args.n if args.stage == "ensemble" else 0)


if __name__ == "__main__":
    main()
