"""Per-cluster Fourier transforms y_ell for L2p8_m9 lc0 rotation-group maps (nb42).

Empirical "stacked profile" theory for the per-group tSZ power spectra of
notebook 39, with no halo-model input.  The five rotation-group maps are raw
(unrotated) shell sums, so cluster positions use the catalogue *native*
lightcone coordinates (verified: map/mean = 300-1000 at native positions of
the most massive group-1 clusters, ~1 at yang26-rotated positions).

Stage ``sample``:
    Stream the deep M_500c > 1e13 lc0 catalogue (24.8M rows) and draw a
    mass-stratified random subsample per rotation group: log10(M_500c) bins of
    0.25 dex over [13.0, 15.75], up to ``N_SAMP`` clusters per bin (bins with
    fewer are fully enumerated).  Exact bin counts ``N_b`` are stored so each
    sampled cluster carries the stratum weight ``N_b / n_b``.

Stage ``stamp`` (per group):
    1. read the group map, cache its measured full-sky D_ell (anafast, iter=0,
       pixwin- and fsky-corrected, the nb39 datapoint convention);
    2. for every sampled cluster cut a gnomonic stamp at ``RES_ARCMIN``,
       subtract the map monopole, apply a hard aperture of radius
       ``X_AP * theta_500``, zero-pad to ``MGRID``^2 and FFT
       (``y_ell = Omega_pix * FFT``, units sr), azimuthally bin;
    3. repeat on ``N_RAND`` random same-size apertures (uniform sphere) for
       the per-cluster background power ``<|y_ell|^2>`` (nb41 correction).

Besides ``|y_ell|`` and ``|y_ell|^2``, the azimuthally binned *coherent* profile
``re_y = <Re[y_ell * exp(i l.x_c)]>`` (phase-centred on the exact source pixel)
is stored.  For the low-mass strata the per-cluster power is far below the
background fluctuation and the incoherent stratified sum is noise-dominated
(error ~ N_b/sqrt(n_b)); stacking the coherent profiles first averages the
background down as 1/n_b and the squared stacked profile (sample-variance
debiased) recovers the stratum power.

Outputs (data/nb42_rotgroup_stack_yell/):
    sample_group{g}.npz            stratified subsample + bin counts
    group{g}_order{o}.npz (+.json) y_ell bins, backgrounds, measured D_ell

Run:
    python scripts/export_L2p8_rotgroup_stack_yell.py --stage sample
    python scripts/export_L2p8_rotgroup_stack_yell.py --stage stamp --group 0
    python scripts/export_L2p8_rotgroup_stack_yell.py --stage stamp --group 0 --nmax 40
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd
import scipy.fft

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import healpy as hp  # noqa: E402
from pixell import enmap, reproject  # noqa: E402

from flamingo.geometry import ARCMIN_PER_RAD  # noqa: E402
from hmfast.cosmology import Cosmology  # noqa: E402

MAP_DIR = Path("/rds/rds-lxu/flamingo/L2p8_m9/lightcone0/healpix_map/rotation_groups")
META_PATH = MAP_DIR / "rotation_groups_L2p8_m9_lc0.json"
CAT_PATH = Path(
    "/rds/rds-lxu/flamingo/L2p8_m9/lightcone0/catalogues/"
    "halo_catalogue_M500c_1e13_zlt3_L2p8_m9_yang26rot.csv"
)
OUT_DIR = _REPO / "data/nb42_rotgroup_stack_yell"

X_AP = 5.0  # aperture radius in units of theta_500 (nb41 convention)
RES_ARCMIN = 0.45  # stamp resolution; native nside4096 pixel is 0.859 arcmin
MGRID = 2048  # common zero-padded grid (2048 * 0.45' = 15.36 deg box)
DELL = 25.0  # linear ell bin width
LMAX_BIN = 6200.0
LMAX_MAP = 6000  # anafast lmax for the measured group D_ell (nb39 convention)
CHECKPOINT = 200
N_SAMP = 500  # cap per (group, mass bin) stratum
N_RAND = 2  # random apertures per cluster
SEED = 42
LOGM_EDGES = np.arange(13.0, 15.75 + 1e-9, 0.25)  # 11 bins
CAT_COLS = ["shell_idx", "z", "lon_nat_deg", "lat_nat_deg", "M_500c_Msun", "R_500c_Mpc"]

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


def load_meta() -> dict:
    return json.loads(META_PATH.read_text())


def sample_path(group: int) -> Path:
    return OUT_DIR / f"sample_group{group}.npz"


def stage_sample() -> None:
    """Stratified per-group subsample of the deep M>1e13 catalogue."""
    meta = load_meta()
    n_groups = meta["n_groups"]
    shell_to_group = {}
    for g in meta["groups"]:
        for s in g["shells"]:
            shell_to_group[s] = g["group"]

    nbin_m = LOGM_EDGES.size - 1
    counts = np.zeros((n_groups, nbin_m), dtype=np.int64)
    kept: list[pd.DataFrame] = []
    rng = np.random.default_rng(SEED)

    t0 = time.time()
    reader = pd.read_csv(CAT_PATH, comment="#", usecols=CAT_COLS, chunksize=2_000_000)
    n_rows = 0
    for chunk in reader:
        n_rows += len(chunk)
        grp = chunk["shell_idx"].map(shell_to_group).to_numpy()
        mbin = np.digitize(np.log10(chunk["M_500c_Msun"].to_numpy()), LOGM_EDGES) - 1
        ok = (mbin >= 0) & (mbin < nbin_m)
        np.add.at(counts, (grp[ok], mbin[ok]), 1)
        chunk = chunk.assign(group=grp, mbin=mbin, u=rng.random(len(chunk)))
        kept.append(chunk[ok])
        print(f"  {n_rows} rows ({time.time() - t0:.0f}s)", flush=True)
    cat = pd.concat(kept, ignore_index=True)
    del kept
    print(f"catalogue: {n_rows} rows, {len(cat)} in mass bins ({time.time() - t0:.0f}s)", flush=True)

    # exact per-stratum sampling: keep the N_SAMP smallest u in each (group, mbin)
    cat["rank"] = cat.groupby(["group", "mbin"])["u"].rank(method="first")
    samp = cat[cat["rank"] <= N_SAMP].drop(columns=["u", "rank"])
    del cat

    cosmo = Cosmology(**D3A)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for g in range(n_groups):
        sg = samp[samp["group"] == g].sort_values("M_500c_Msun", ascending=False)
        dA = np.asarray(cosmo.angular_diameter_distance(sg["z"].to_numpy()), dtype=np.float64)
        theta500 = sg["R_500c_Mpc"].to_numpy() / dA * ARCMIN_PER_RAD
        n_b = np.bincount(sg["mbin"].to_numpy(), minlength=nbin_m)
        np.savez(
            sample_path(g),
            group=g,
            logm_edges=LOGM_EDGES,
            N_bin=counts[g],
            n_bin=n_b,
            mbin=sg["mbin"].to_numpy(),
            z=sg["z"].to_numpy(),
            lon_nat_deg=sg["lon_nat_deg"].to_numpy(),
            lat_nat_deg=sg["lat_nat_deg"].to_numpy(),
            M_500c_Msun=sg["M_500c_Msun"].to_numpy(),
            R_500c_Mpc=sg["R_500c_Mpc"].to_numpy(),
            dA_Mpc=dA,
            theta500_arcmin=theta500,
            n_samp_cap=N_SAMP,
            seed=SEED,
        )
        print(
            f"group {g}: N={counts[g].sum()}  sampled={len(sg)}  "
            f"N_bin={counts[g].tolist()}",
            flush=True,
        )
    print(f"stage sample done ({time.time() - t0:.0f}s)", flush=True)


def measure_group_dl(ymap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full-sky D_ell of the group map (nb39 convention: iter=0, pixwin, fsky)."""
    nside = hp.npix2nside(ymap.size)
    fsky = float(np.mean(ymap != 0.0))
    cl = hp.anafast(ymap - ymap.mean(), lmax=LMAX_MAP, iter=0)
    ell = np.arange(cl.size, dtype=np.float64)
    cl = cl / (hp.pixwin(nside, lmax=LMAX_MAP) ** 2 * fsky)
    return ell, ell * (ell + 1.0) / (2.0 * np.pi) * cl


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Azimuthally binned |y_ell|, |y_ell|^2, coherent Re[y_ell] and DC |y(0)| (sr)."""
    r_ap = X_AP * theta500_arcmin / ARCMIN_PER_RAD
    npix = int(np.ceil(2.0 * r_ap / res_rad))
    npix = min(max(npix + npix % 2, 8), MGRID)
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

    # coherent profile: re-reference FFT phases to the exact source pixel so that
    # Re[y_ell] is the centred (Hankel-like) transform and background averages to 0
    py, px = enmap.sky2pix(shape, wcs, np.array([dec, ra]))
    f1 = np.fft.fftfreq(MGRID)
    phase = np.exp(2j * np.pi * f1 * float(py))[:, None] * np.exp(
        2j * np.pi * f1 * float(px)
    )[None, :]
    re = (fmap * phase).real.ravel() * omega_pix
    re_y = np.bincount(bin_idx, weights=re, minlength=nbin)[:nbin] / counts
    return abs_y, abs_y2, re_y, y0


def stage_stamp(group: int, order: int, nmax: int | None) -> None:
    meta = load_meta()
    ginfo = meta["groups"][group]
    assert ginfo["group"] == group
    map_path = MAP_DIR / ginfo["fits"]
    out = OUT_DIR / f"group{group}_order{order}.npz"
    partial = out.with_suffix(".partial.npz")

    s = np.load(sample_path(group))
    n_cl = len(s["z"]) if nmax is None else min(nmax, len(s["z"]))

    t0 = time.time()
    print(f"loading {map_path.name} ...", flush=True)
    ymap = hp.read_map(str(map_path), dtype=np.float64)
    monopole = float(ymap.mean())
    print(f"  monopole = {monopole:.4e}", flush=True)

    dl_cache = OUT_DIR / f"measured_dl_group{group}.npz"
    if dl_cache.exists():
        print(f"measured D_ell cached: {dl_cache.name}", flush=True)
    else:
        t1 = time.time()
        ell_map, dl_map = measure_group_dl(ymap)
        np.savez(dl_cache, ell=ell_map, dl=dl_map, lmax=LMAX_MAP, map_path=str(map_path))
        print(f"measured D_ell in {time.time() - t1:.0f}s -> {dl_cache.name}", flush=True)

    res_rad = RES_ARCMIN / ARCMIN_PER_RAD
    lfreq = 2.0 * np.pi * np.fft.fftfreq(MGRID, d=res_rad)
    lmap = np.hypot(*np.meshgrid(lfreq, lfreq, indexing="ij")).ravel()
    edges = np.arange(0.0, LMAX_BIN + DELL, DELL)
    nbin = edges.size - 1
    ell_b = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip((lmap / DELL).astype(np.int64), 0, nbin)  # nbin = overflow bin
    counts = np.bincount(bin_idx, minlength=nbin + 1)[:nbin].astype(np.float64)
    counts[counts == 0] = 1.0

    rng = np.random.default_rng(SEED + 100 * (group + 1))
    lon_r = rng.uniform(0.0, 360.0, (n_cl, N_RAND))
    lat_r = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, (n_cl, N_RAND))))

    abs_y = np.zeros((n_cl, nbin))
    abs_y2 = np.zeros((n_cl, nbin))
    re_y = np.zeros((n_cl, nbin))
    bg_y = np.zeros((n_cl, nbin))
    bg_y2 = np.zeros((n_cl, nbin))
    bg_re = np.zeros((n_cl, nbin))
    bg_re2 = np.zeros((n_cl, nbin))
    y0 = np.zeros(n_cl)
    i_start = 0
    if partial.exists():
        prev = np.load(partial)
        if prev["abs_y2"].shape == abs_y2.shape and int(prev["n_done"]) < n_cl:
            i_start = int(prev["n_done"])
            for name, arr in [
                ("abs_y", abs_y),
                ("abs_y2", abs_y2),
                ("re_y", re_y),
                ("bg_y", bg_y),
                ("bg_y2", bg_y2),
                ("bg_re", bg_re),
                ("bg_re2", bg_re2),
                ("y0", y0),
            ]:
                arr[:i_start] = prev[name][:i_start]
            print(f"resuming from {partial.name} at cluster {i_start}", flush=True)

    t1 = time.time()
    for i in range(i_start, n_cl):
        theta500 = float(s["theta500_arcmin"][i])
        kw = dict(order=order, res_rad=res_rad, bin_idx=bin_idx, nbin=nbin, counts=counts)
        abs_y[i], abs_y2[i], re_y[i], y0[i] = cluster_yell2(
            ymap, monopole, float(s["lon_nat_deg"][i]), float(s["lat_nat_deg"][i]), theta500, **kw
        )
        acc = [
            cluster_yell2(ymap, monopole, lon_r[i, j], lat_r[i, j], theta500, **kw)
            for j in range(N_RAND)
        ]
        bg_y[i] = np.mean([a[0] for a in acc], axis=0)
        bg_y2[i] = np.mean([a[1] for a in acc], axis=0)
        bg_re[i] = np.mean([a[2] for a in acc], axis=0)
        bg_re2[i] = np.mean([np.square(a[2]) for a in acc], axis=0)
        if (i + 1) % CHECKPOINT == 0 or i + 1 == n_cl:
            np.savez(
                partial,
                abs_y=abs_y,
                abs_y2=abs_y2,
                re_y=re_y,
                bg_y=bg_y,
                bg_y2=bg_y2,
                bg_re=bg_re,
                bg_re2=bg_re2,
                y0=y0,
                n_done=i + 1,
            )
            rate = (time.time() - t1) / (i + 1 - i_start)
            print(f"  {i + 1}/{n_cl}  ({rate:.2f} s/cluster)", flush=True)

    np.savez(
        out,
        ell_b=ell_b,
        abs_y=abs_y,
        abs_y2=abs_y2,
        re_y=re_y,
        bg_y=bg_y,
        bg_y2=bg_y2,
        bg_re=bg_re,
        bg_re2=bg_re2,
        y0=y0,
        counts=counts,
        logm_edges=s["logm_edges"],
        N_bin=s["N_bin"],
        n_bin=s["n_bin"],
        mbin=s["mbin"][:n_cl],
        z=s["z"][:n_cl],
        M_500c_Msun=s["M_500c_Msun"][:n_cl],
        theta500_arcmin=s["theta500_arcmin"][:n_cl],
        dA_Mpc=s["dA_Mpc"][:n_cl],
        z_inner=ginfo["z_inner"],
        z_outer=ginfo["z_outer"],
        monopole=monopole,
        x_ap=X_AP,
        res_arcmin=RES_ARCMIN,
        mgrid=MGRID,
        dell=DELL,
        order=order,
        n_randoms=N_RAND,
        map_path=str(map_path),
        cat_path=str(CAT_PATH),
    )
    partial.unlink(missing_ok=True)
    manifest = dict(
        group=group,
        z_range=[ginfo["z_inner"], ginfo["z_outer"]],
        shells=[ginfo["shell_first"], ginfo["shell_last"]],
        git_hash=git_hash(),
        map_path=str(map_path),
        cat_path=str(CAT_PATH),
        x_ap=X_AP,
        res_arcmin=RES_ARCMIN,
        mgrid=MGRID,
        dell=DELL,
        order=order,
        n_clusters=n_cl,
        n_randoms=N_RAND,
        n_samp_cap=N_SAMP,
        logm_edges=LOGM_EDGES.tolist(),
        N_bin=s["N_bin"].tolist(),
        n_bin=s["n_bin"].tolist(),
        seed=SEED,
        monopole=monopole,
        runtime_s=round(time.time() - t0, 1),
        output=str(out),
    )
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}  ({manifest['runtime_s']} s total)", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["sample", "stamp"], required=True)
    p.add_argument("--group", type=int, default=None, help="rotation group index (stage stamp)")
    p.add_argument("--order", type=int, default=0, help="spline interpolation order")
    p.add_argument("--nmax", type=int, default=None, help="process only the first N clusters")
    args = p.parse_args()

    if args.stage == "sample":
        stage_sample()
    else:
        if args.group is None:
            raise SystemExit("--group is required for stage stamp")
        stage_stamp(args.group, args.order, args.nmax)


if __name__ == "__main__":
    main()
