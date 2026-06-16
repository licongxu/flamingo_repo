"""Build the full-resolution (NSIDE=16384) un-lensed Compton-y map for L2p8_m9.

Streams per-shell ``ComptonY`` HEALPix maps (NSIDE=16384, RING, dimensionless)
from the FLAMINGO portal via hdfstream, applies the per-shell yang26 rotation
in alm space, and accumulates the integrated y-map. This is the full-resolution
counterpart of the NSIDE=4096 product; the algorithm is identical, only NSIDE
and the I/O bookkeeping differ.

Because a single NSIDE=16384 shell is ~25.8 GiB and ~60 shells (~1.5 TiB) are
streamed, the build checkpoints the accumulator after every rotation group and
can resume: re-running skips groups already folded into the checkpoint.

Reference: https://dataweb.cosma.dur.ac.uk:8443/flamingo/lightcones/integrated_lightcones.html

Notes
-----
No astropy (project rule): the deg<->rad conversion is plain numpy.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import ducc0
import healpy as hp
import hdfstream
import numpy as np
from scipy.spatial.transform import Rotation

# Per-shell rotation angles (theta, phi) in radians for L2p8 boxes, copied
# verbatim from the FLAMINGO portal (integrated_lightcones.html). Identical for
# every NSIDE; the rotation frame is a property of the lightcone, not the map.
ANGLES_L2P8 = np.array(
    [
        [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.11833333, 2.11833333, 2.11833333,
            2.11833333, 2.11833333, 2.11833333, 2.11833333, 2.11833333, 2.11833333,
            1.29070838, 1.29070838, 1.29070838, 1.29070838, 1.29070838, 1.29070838,
            1.29070838, 1.29070838, 1.29070838, 1.29070838, 1.29070838, 5.69217656,
            5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656,
            5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656, 5.69217656,
            5.69217656, 5.69217656, 5.69217656, 5.69217656, 3.79736641, 3.79736641,
            3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641,
            3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641, 3.79736641,
            3.79736641, 3.79736641, 3.79736641, 3.79736641, 1.32878635, 1.32878635,
            1.32878635, 1.32878635, 1.32878635, 1.32878635,
        ],
        [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.96440001, 0.96440001, 0.96440001,
            0.96440001, 0.96440001, 0.96440001, 0.96440001, 0.96440001, 0.96440001,
            1.74841793, 1.74841793, 1.74841793, 1.74841793, 1.74841793, 1.74841793,
            1.74841793, 1.74841793, 1.74841793, 1.74841793, 1.74841793, 0.56258515,
            0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515,
            0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515, 0.56258515,
            0.56258515, 0.56258515, 0.56258515, 0.56258515, 1.45462313, 1.45462313,
            1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313,
            1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313, 1.45462313,
            1.45462313, 1.45462313, 1.45462313, 1.45462313, 2.48706614, 2.48706614,
            2.48706614, 2.48706614, 2.48706614, 2.48706614,
        ],
    ]
)

NSIDE = 16384
NPIX = 12 * NSIDE * NSIDE
LMAX = 3 * NSIDE - 1             # healpy default band-limit for an NSIDE map
N_ANGLES = ANGLES_L2P8.shape[1]  # 65 (portal-documented)
DEFAULT_SHELL_MAX = 60           # shells 0..59 cover z < 3.0 (matches the 4096 map)
# Threads for the ducc SHT. Kept modest on purpose: this is a shared node and we
# prefer a slow, safe run over saturating all cores. Override with FLAMINGO_NTHREADS.
NTHREADS = int(os.environ.get("FLAMINGO_NTHREADS", "32"))


def _ducc_job(nthreads: int = NTHREADS):
    """A ducc sharp job configured for NSIDE/LMAX HEALPix transforms (64-bit)."""
    job = ducc0.sht.sharpjob_d()
    job.set_nthreads(nthreads)
    job.set_healpix_geometry(NSIDE)
    job.set_triangular_alm_info(LMAX, LMAX)
    return job


def _map2alm_ducc(m: np.ndarray, nthreads: int = NTHREADS, niter: int = 3) -> np.ndarray:
    """map2alm via ducc (adjoint synthesis + Jacobi iteration for quadrature).

    Reproduces healpy's iterative ``map2alm`` (the use_pixel_weights=False path),
    which is needed because healpy's own SHT is int32-limited at NSIDE=16384 and
    pixel-weight files do not exist above NSIDE=8192. Validated against
    ``healpy.rotate_map_alms`` to ~1e-13 at NSIDE<=1024.
    """
    job = _ducc_job(nthreads)
    pix_area = 4.0 * np.pi / NPIX
    alm = job.alm2map_adjoint(m) * pix_area
    for _ in range(niter):
        alm = alm + job.alm2map_adjoint(m - job.alm2map(alm)) * pix_area
    return alm


def rotate_map(hmap: np.ndarray, rot_theta: float, rot_phi: float) -> np.ndarray:
    """Rotate in alm space, matching the FLAMINGO yang26 convention (official method).

    inv=True, lon=phi[deg], lat=theta[deg]. Both the spherical-harmonic transforms
    AND the alm rotation use ducc (64-bit): healpy is int32-limited at NSIDE=16384,
    both in its SHT (NPIX > 2**31) and in ``rotate_alm`` ((lmax+1)*(lmax+2) > 2**31
    at lmax=49151). We take the rotation matrix from healpy's ``Rotator`` (the exact
    NSIDE=4096 convention), decompose it into ZYZ Euler angles, and feed those to
    ``ducc0.sht.rotate_alm``. Validated to reproduce ``healpy.rotate_map_alms`` to
    ~1e-13 at low NSIDE for every shell-group angle. Pixel-space rotation is
    deliberately NOT used: it suppresses power at high multipoles.
    """
    longitude = rot_phi * 180.0 / np.pi
    latitude = rot_theta * 180.0 / np.pi
    rot = hp.Rotator(rot=[longitude, latitude], inv=True)
    psi, theta, phi = Rotation.from_matrix(np.asarray(rot.mat)).as_euler("zyz")
    alm = _map2alm_ducc(hmap)
    alm = ducc0.sht.rotate_alm(alm, LMAX, psi, theta, phi, NTHREADS)
    return _ducc_job().alm2map(alm)


def group_consecutive_same_angles(angles: np.ndarray, shell_max: int):
    """Yield (i_start, i_stop_exclusive, theta, phi) for runs of identical
    (theta, phi). Rotating each group's summed map once (instead of per shell)
    cuts the number of expensive SHTs to a handful."""
    i = 0
    while i < shell_max:
        j = i + 1
        while j < shell_max and angles[0, j] == angles[0, i] and angles[1, j] == angles[1, i]:
            j += 1
        yield i, j, float(angles[0, i]), float(angles[1, i])
        i = j


def _stream_shell(root, path: str, retries: int = 3) -> np.ndarray:
    """Stream one shell's ComptonY array, retrying on transient network errors."""
    for attempt in range(1, retries + 1):
        try:
            return root[path]["ComptonY"][...]
        except Exception as exc:  # noqa: BLE001 - network layer raises broad errors
            if attempt == retries:
                raise
            wait = 30 * attempt
            print(f"    stream error ({exc!r}); retry {attempt}/{retries - 1} in {wait}s",
                  flush=True)
            time.sleep(wait)


def build_y_map(run: str, observer: int, shell_max: int, out_path: Path,
                ckpt_dir: Path) -> None:
    assert 0 <= observer <= 7, "L2p8 boxes have observers 0..7"
    if shell_max > N_ANGLES:
        raise ValueError(
            f"shell_max={shell_max} exceeds available rotation angles "
            f"(N_ANGLES={N_ANGLES})."
        )

    root = hdfstream.open("cosma", "/")
    base = f"FLAMINGO/{run}/{run}/healpix_maps/nside_{NSIDE}/lightcone{observer}_shells"

    groups = list(group_consecutive_same_angles(ANGLES_L2P8, shell_max))

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_y = ckpt_dir / f"ckpt_y_{run}_lc{observer}_nside{NSIDE}.npy"
    ckpt_meta = ckpt_dir / f"ckpt_meta_{run}_lc{observer}_nside{NSIDE}.json"

    # Resume if a checkpoint for the same configuration exists.
    done: set[int] = set()
    if ckpt_y.exists() and ckpt_meta.exists():
        meta = json.loads(ckpt_meta.read_text())
        if meta.get("shell_max") == shell_max and meta.get("nside") == NSIDE:
            y = np.load(ckpt_y)
            done = set(meta.get("completed_groups", []))
            print(f"resuming: {len(done)} groups already folded in "
                  f"(mean so far {y.mean():.3e})", flush=True)
        else:
            y = np.zeros(NPIX, dtype=np.float64)
    else:
        y = np.zeros(NPIX, dtype=np.float64)

    print(f"shell_max={shell_max} -> {len(groups)} angle-groups; "
          f"NSIDE={NSIDE}, NPIX={NPIX:,}", flush=True)

    for gi, (i0, i1, theta, phi) in enumerate(groups):
        if gi in done:
            print(f"group {gi+1}/{len(groups)} (shells {i0}..{i1-1}) already done, skip",
                  flush=True)
            continue

        t_group = time.time()
        group_sum = np.zeros(NPIX, dtype=np.float64)
        for i in range(i0, i1):
            path = f"{base}/shell_{i}/lightcone{observer}.shell_{i}.0.hdf5"
            t0 = time.time()
            print(f"[{run} obs{observer}] group {gi+1}/{len(groups)} "
                  f"(theta={theta:.5f}, phi={phi:.5f}) stream shell_{i}…", flush=True)
            group_sum += _stream_shell(root, path)
            print(f"    shell_{i} folded in ({time.time()-t0:.0f}s)", flush=True)

        if theta == 0.0 and phi == 0.0:
            y += group_sum
            print(f"  group {gi+1}: no rotation (identity)", flush=True)
        else:
            t0 = time.time()
            print(f"  group {gi+1}: rotating summed map once (alm, lmax={3*NSIDE-1})…",
                  flush=True)
            y += rotate_map(group_sum, theta, phi)
            print(f"  group {gi+1}: rotated ({time.time()-t0:.0f}s)", flush=True)
        del group_sum

        # Checkpoint the accumulator after each group.
        done.add(gi)
        np.save(ckpt_y, y)
        ckpt_meta.write_text(json.dumps(
            {"run": run, "observer": observer, "nside": NSIDE,
             "shell_max": shell_max, "completed_groups": sorted(done)}))
        print(f"  checkpoint saved after group {gi+1} "
              f"({time.time()-t_group:.0f}s, mean(y)={y.mean():.3e})", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    hp.write_map(str(out_path), y, nest=False, overwrite=True, dtype=np.float64)
    print(f"\nWrote {out_path}  mean(y)={y.mean():.3e}  NSIDE={NSIDE} (RING)", flush=True)
    print("Build complete. Checkpoint files can be deleted:", ckpt_y, ckpt_meta, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="L2p8_m9")
    p.add_argument("--observer", type=int, default=0)
    p.add_argument("--shell-max", type=int, default=DEFAULT_SHELL_MAX)
    p.add_argument("--out", type=Path,
                   default=Path("/rds/rds-lxu/tsz_project/flamingo_highres_maps/"
                                "y_unlensed_L2p8_m9_lc0_nside16384.fits"))
    p.add_argument("--ckpt-dir", type=Path,
                   default=Path("/scratch/scratch-lxu/flamingo_highres_build"))
    args = p.parse_args()
    build_y_map(args.run, args.observer, args.shell_max, args.out, args.ckpt_dir)


if __name__ == "__main__":
    main()
