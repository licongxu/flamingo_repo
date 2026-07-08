"""Per-cluster Fourier transforms y_ell of resolved (q>5) clusters on the L1_m9 y-map.

For every cluster with ``q_from_mz > Q_CUT`` in the fiducial L1_m9 catalogue:

1. cut a gnomonic (tangent-plane) stamp centred on the cluster from the
   nside=4096 Compton-y map, oversampled at ``RES_ARCMIN``;
2. subtract the global map monopole and apply a hard circular aperture of
   radius ``X_AP * theta_500`` (matching the 5 x theta_500 mask discs used by
   ``scripts/compute_masked_tsz_ps_L1_m9_feedback.py``);
3. zero-pad onto a common ``MGRID x MGRID`` grid and FFT, so every cluster's
   ``y_ell = Omega_pix * FFT`` (units sr) lives on the same flat-sky ell grid;
4. azimuthally bin ``|y_ell|`` and ``|y_ell|^2`` in linear ell bins.

The direct-sum "empirical theory" for the resolved tSZ power spectrum is then
``C_ell = sum_i <|y_ell,i|^2> / (4 pi)`` (notebook 14 convention), evaluated in
``notebooks/41_resolved_tsz_ps_fourier_stack.ipynb``.

Output: ``data/nb41_resolved_stack_yell/L1_m9_qgt5_order{ORDER}.npz``
(plus a ``.json`` manifest). Checkpointed every ``CHECKPOINT`` clusters; rerun
resumes from the partial file.

The stamps contain the background sky inside each aperture, whose summed power
does not cancel in the direct sum. ``--randoms N`` computes the standard
random-stamp background: for every cluster, N apertures of the same size at
random centres outside the q>5 binary mask discs, saved as
``..._randoms{N}.npz`` with the per-cluster mean ``<|y_ell|^2>``.

Run:
    python scripts/export_L1_m9_resolved_stack_yell.py            # full run
    python scripts/export_L1_m9_resolved_stack_yell.py --randoms 3
    python scripts/export_L1_m9_resolved_stack_yell.py --nmax 20  # quick test
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
OUT_DIR = _REPO / "data/nb41_resolved_stack_yell"

Q_CUT = 5.0
X_AP = 5.0  # aperture radius in units of theta_500 (= R_MASK of the masked PS)
RES_ARCMIN = 0.45  # stamp resolution; native nside4096 pixel is 0.859 arcmin
MGRID = 2048  # common zero-padded grid (2048 * 0.45' = 15.36 deg box)
DELL = 25.0  # linear ell bin width
LMAX_BIN = 6200.0
CHECKPOINT = 200

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
    # process large stamps first so early checkpoints capture the slow clusters
    return cat.sort_values("theta500_arcmin", ascending=False).reset_index(drop=True)


def cluster_yell2(
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
) -> tuple[np.ndarray, np.ndarray, float]:
    """Azimuthally binned |y_ell|, |y_ell|^2 (sr, sr^2) and DC mode |y(0)| (sr)."""
    r_ap = X_AP * theta500_arcmin / ARCMIN_PER_RAD
    npix = int(np.ceil(2.0 * r_ap / res_rad))
    npix = min(npix + npix % 2, MGRID)
    dec, ra = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    shape, wcs = enmap.geometry(pos=(dec, ra), shape=(npix, npix), res=res_rad, proj="tan")
    stamp = reproject.healpix2map(ymap, shape, wcs, rot=None, method="spline", order=order)
    stamp = np.asarray(stamp, dtype=np.float64) - monopole
    r = np.asarray(enmap.modrmap(shape, wcs))
    stamp[r > r_ap] = 0.0

    padded = np.zeros((MGRID, MGRID), dtype=np.float64)
    padded[:npix, :npix] = stamp
    fmap = scipy.fft.fft2(padded, workers=8)
    omega_pix = res_rad * res_rad
    y0 = float(np.abs(fmap[0, 0])) * omega_pix
    amp = np.abs(fmap).ravel() * omega_pix

    abs_y = np.bincount(bin_idx, weights=amp, minlength=nbin)[:nbin] / counts
    abs_y2 = np.bincount(bin_idx, weights=amp * amp, minlength=nbin)[:nbin] / counts
    return abs_y, abs_y2, y0


def random_centres(
    cat: pd.DataFrame, n_rand: int, nside: int, *, seed: int = 41
) -> tuple[np.ndarray, np.ndarray]:
    """(n_cl, n_rand) random lon/lat (deg) with centres outside the q>5 mask discs."""
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
    n_tot = len(cat) * n_rand
    lon, lat = [], []
    while len(lon) < n_tot:
        n_draw = 2 * (n_tot - len(lon)) + 64
        lon_d = rng.uniform(0.0, 360.0, n_draw)
        lat_d = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n_draw)))
        pix = hp.ang2pix(nside, lon_d, lat_d, lonlat=True)
        keep = mask[pix]
        lon.extend(lon_d[keep])
        lat.extend(lat_d[keep])
    lon = np.array(lon[:n_tot]).reshape(len(cat), n_rand)
    lat = np.array(lat[:n_tot]).reshape(len(cat), n_rand)
    return lon, lat


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--order", type=int, default=1, help="spline interpolation order")
    p.add_argument("--nmax", type=int, default=None, help="process only the first N clusters")
    p.add_argument("--randoms", type=int, default=0, help="random apertures per cluster")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    suffix = f"_randoms{args.randoms}" if args.randoms else ""
    out = args.out or OUT_DIR / f"L1_m9_qgt5_order{args.order}{suffix}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".partial.npz")

    t0 = time.time()
    print(f"loading {MAP_PATH.name} ...", flush=True)
    ymap = hp.read_map(MAP_PATH)
    monopole = float(ymap.mean())
    print(f"  monopole = {monopole:.4e}", flush=True)

    cat = load_sample()
    if args.nmax is not None:
        cat = cat.iloc[: args.nmax]
    n_cl = len(cat)
    print(f"clusters: {n_cl} (q_from_mz > {Q_CUT:g})", flush=True)

    res_rad = RES_ARCMIN / ARCMIN_PER_RAD
    lfreq = 2.0 * np.pi * np.fft.fftfreq(MGRID, d=res_rad)
    lmap = np.hypot(*np.meshgrid(lfreq, lfreq, indexing="ij")).ravel()
    edges = np.arange(0.0, LMAX_BIN + DELL, DELL)
    nbin = edges.size - 1
    ell_b = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip((lmap / DELL).astype(np.int64), 0, nbin)  # nbin = overflow bin
    counts = np.bincount(bin_idx, minlength=nbin + 1)[:nbin].astype(np.float64)
    counts[counts == 0] = 1.0

    abs_y = np.zeros((n_cl, nbin))
    abs_y2 = np.zeros((n_cl, nbin))
    y0 = np.zeros(n_cl)
    i_start = 0
    if partial.exists():
        prev = np.load(partial)
        if (
            prev["abs_y"].shape == abs_y.shape
            and int(prev["order"]) == args.order
            and int(prev["n_done"]) < n_cl
        ):
            i_start = int(prev["n_done"])
            abs_y[:i_start] = prev["abs_y"][:i_start]
            abs_y2[:i_start] = prev["abs_y2"][:i_start]
            y0[:i_start] = prev["y0"][:i_start]
            print(f"resuming from {partial.name} at cluster {i_start}", flush=True)

    if args.randoms:
        nside = hp.npix2nside(ymap.size)
        lon_r, lat_r = random_centres(cat, args.randoms, nside)

    t1 = time.time()
    for i in range(i_start, n_cl):
        row = cat.iloc[i]
        if args.randoms:
            acc = [
                cluster_yell2(
                    ymap,
                    monopole,
                    lon_r[i, j],
                    lat_r[i, j],
                    row["theta500_arcmin"],
                    order=args.order,
                    res_rad=res_rad,
                    bin_idx=bin_idx,
                    nbin=nbin,
                    counts=counts,
                )
                for j in range(args.randoms)
            ]
            abs_y[i] = np.mean([a[0] for a in acc], axis=0)
            abs_y2[i] = np.mean([a[1] for a in acc], axis=0)
            y0[i] = np.mean([a[2] for a in acc])
        else:
            abs_y[i], abs_y2[i], y0[i] = cluster_yell2(
                ymap,
                monopole,
                row["lon_rot_deg"],
                row["lat_rot_deg"],
                row["theta500_arcmin"],
                order=args.order,
                res_rad=res_rad,
                bin_idx=bin_idx,
                nbin=nbin,
                counts=counts,
            )
        if (i + 1) % CHECKPOINT == 0 or i + 1 == n_cl:
            np.savez(partial, abs_y=abs_y, abs_y2=abs_y2, y0=y0, n_done=i + 1, order=args.order)
            rate = (time.time() - t1) / (i + 1 - i_start)
            print(f"  {i + 1}/{n_cl}  ({rate:.2f} s/cluster)", flush=True)

    np.savez(
        out,
        ell_b=ell_b,
        abs_y=abs_y,
        abs_y2=abs_y2,
        y0=y0,
        counts=counts,
        theta500_arcmin=cat["theta500_arcmin"].to_numpy(),
        z=cat["z"].to_numpy(),
        M_500c_Msun=cat["M_500c_Msun"].to_numpy(),
        q_from_mz=cat["q_from_mz"].to_numpy(),
        Y_5R500c_Mpc2=cat["Y_5R500c_Mpc2"].to_numpy(),
        dA_Mpc=cat["dA_Mpc"].to_numpy(),
        monopole=monopole,
        q_cut=Q_CUT,
        x_ap=X_AP,
        res_arcmin=RES_ARCMIN,
        mgrid=MGRID,
        dell=DELL,
        order=args.order,
        n_randoms=args.randoms,
        map_path=str(MAP_PATH),
        cat_path=str(CAT_PATH),
    )
    partial.unlink(missing_ok=True)
    manifest = dict(
        n_randoms=args.randoms,
        git_hash=git_hash(),
        map_path=str(MAP_PATH),
        cat_path=str(CAT_PATH),
        q_cut=Q_CUT,
        x_ap=X_AP,
        res_arcmin=RES_ARCMIN,
        mgrid=MGRID,
        dell=DELL,
        order=args.order,
        n_clusters=n_cl,
        monopole=monopole,
        runtime_s=round(time.time() - t0, 1),
        output=str(out),
    )
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}  ({manifest['runtime_s']} s total)", flush=True)


if __name__ == "__main__":
    main()
