"""Build a yang26-rotated Compton-y map (NSIDE=4096) from FLAMINGO healpix shells.

Streams per-shell ``ComptonY`` HEALPix maps (NSIDE=4096, RING, dimensionless)
from the FLAMINGO portal via hdfstream, applies the per-shell yang26 rotation
in alm space, and accumulates the integrated y-map.

L2p8 boxes use ``FLAMINGO/{run}/{run}/healpix_maps/...``; L1_m9 variants use
``FLAMINGO/L1_m9/{variant}/healpix_maps/...`` (no duplicated folder name).

Checkpoints after every rotation group so a long build can resume.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import healpy as hp
import hdfstream
import numpy as np

from build_y_map_highres import ANGLES_L2P8

# Per-shell rotation angles (theta, phi) in radians for L1 boxes.
ANGLES_L1 = np.array(
    [
        [
            0.0, 0.0, 3.26757547, 3.26757547, 3.26757547, 1.51289711, 1.51289711,
            3.13885639, 3.13885639, 3.13885639, 2.17061318, 2.17061318, 2.17061318,
            2.17061318, 4.59420579, 4.59420579, 4.59420579, 1.14273623, 1.14273623,
            1.14273623, 1.14273623, 2.02717201, 2.02717201, 2.02717201, 2.02717201,
            2.77675054, 2.77675054, 2.77675054, 2.77675054, 2.77675054, 0.83245259,
            0.83245259, 0.83245259, 0.83245259, 0.83245259, 0.83245259, 4.95779263,
            4.95779263, 4.95779263, 4.95779263, 4.95779263, 4.95779263, 4.95779263,
            2.52359739, 2.52359739, 2.52359739, 2.52359739, 2.52359739, 2.52359739,
            2.52359739, 2.52359739, 2.69301628, 2.69301628, 2.69301628, 2.69301628,
            2.69301628, 2.69301628, 2.69301628, 2.69301628, 2.69301628,
        ],
        [
            0.0, 0.0, 1.41518902, 1.41518902, 1.41518902, 0.80580058, 0.80580058,
            0.71830831, 0.71830831, 0.71830831, 1.77536892, 1.77536892, 1.77536892,
            1.77536892, 0.62434822, 0.62434822, 0.62434822, 2.14076603, 2.14076603,
            2.14076603, 2.14076603, 0.49840908, 0.49840908, 0.49840908, 0.49840908,
            2.0136344, 2.0136344, 2.0136344, 2.0136344, 2.0136344, 2.25356928,
            2.25356928, 2.25356928, 2.25356928, 2.25356928, 2.25356928, 1.85187078,
            1.85187078, 1.85187078, 1.85187078, 1.85187078, 1.85187078, 1.85187078,
            1.36014098, 1.36014098, 1.36014098, 1.36014098, 1.36014098, 1.36014098,
            1.36014098, 1.36014098, 2.35895331, 2.35895331, 2.35895331, 2.35895331,
            2.35895331, 2.35895331, 2.35895331, 2.35895331, 2.35895331,
        ],
    ]
)

DEFAULT_NSIDE = 4096
DEFAULT_SHELL_MAX = 60  # shells 0..59, z < 3


def _healpix_base(parent: str | None, variant: str, nside: int, observer: int) -> str:
    if parent is None:
        return (
            f"FLAMINGO/{variant}/{variant}/healpix_maps/nside_{nside}/"
            f"lightcone{observer}_shells"
        )
    return (
        f"FLAMINGO/{parent}/{variant}/healpix_maps/nside_{nside}/"
        f"lightcone{observer}_shells"
    )


def _angles_for(parent: str | None) -> np.ndarray:
    return ANGLES_L1 if parent == "L1_m9" else ANGLES_L2P8


CKPT_METHOD = "alm_accum_v1"


def _rotator_for(rot_theta: float, rot_phi: float) -> hp.Rotator:
    """yang26 rotation, identical convention to the catalogue position rotation."""
    longitude = rot_phi * 180.0 / np.pi
    latitude = rot_theta * 180.0 / np.pi
    return hp.Rotator(rot=[longitude, latitude], inv=True)


def rotate_map(hmap: np.ndarray, rot_theta: float, rot_phi: float) -> np.ndarray:
    """Rotate a map in alm space (FLAMINGO yang26 convention).

    Kept for reference / verification: equivalent to
    ``alm2map(rotate_alm(map2alm(hmap)))``.
    """
    return _rotator_for(rot_theta, rot_phi).rotate_map_alms(hmap)


def rotated_alm(hmap: np.ndarray, rot_theta: float, rot_phi: float, lmax: int) -> np.ndarray:
    """map2alm + yang26 rotate_alm, the per-group term of the integrated map.

    Summing these terms over all angle-groups and calling ``alm2map`` once is
    exactly equal to summing ``rotate_map_alms`` per group (alm2map is linear),
    but avoids one inverse transform per group.
    """
    alm = hp.map2alm(hmap, lmax=lmax, use_pixel_weights=True)
    return _rotator_for(rot_theta, rot_phi).rotate_alm(alm, lmax=lmax)


def group_consecutive_same_angles(angles: np.ndarray, shell_max: int):
    i = 0
    while i < shell_max:
        j = i + 1
        while (
            j < shell_max
            and angles[0, j] == angles[0, i]
            and angles[1, j] == angles[1, i]
        ):
            j += 1
        yield i, j, float(angles[0, i]), float(angles[1, i])
        i = j


def _stream_shell(root, path: str, retries: int = 3) -> np.ndarray:
    for attempt in range(1, retries + 1):
        try:
            return root[path]["ComptonY"][...]
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                raise
            wait = 30 * attempt
            print(
                f"    stream error ({exc!r}); retry {attempt}/{retries - 1} in {wait}s",
                flush=True,
            )
            time.sleep(wait)


def build_y_map(
    variant: str,
    observer: int,
    shell_max: int,
    out_path: Path,
    ckpt_dir: Path,
    *,
    parent: str | None = None,
    nside: int = DEFAULT_NSIDE,
) -> None:
    angles = _angles_for(parent)
    if shell_max > angles.shape[1]:
        raise ValueError(
            f"shell_max={shell_max} exceeds available rotation angles "
            f"(N_ANGLES={angles.shape[1]})."
        )

    npix = 12 * nside * nside
    lmax = 3 * nside - 1  # matches healpy rotate_map_alms / map2alm default
    root = hdfstream.open("cosma", "/")
    base = _healpix_base(parent, variant, nside, observer)
    groups = list(group_consecutive_same_angles(angles, shell_max))

    tag = f"{parent}_{variant}" if parent else variant
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_alm = ckpt_dir / f"ckpt_alm_{tag}_lc{observer}_nside{nside}.npy"
    ckpt_yident = ckpt_dir / f"ckpt_yident_{tag}_lc{observer}_nside{nside}.npy"
    ckpt_meta = ckpt_dir / f"ckpt_meta_{tag}_lc{observer}_nside{nside}.json"

    n_alm = hp.Alm.getsize(lmax)
    done: set[int] = set()
    alm_total = np.zeros(n_alm, dtype=np.complex128)
    y_ident = np.zeros(npix, dtype=np.float64)
    if ckpt_alm.exists() and ckpt_yident.exists() and ckpt_meta.exists():
        meta = json.loads(ckpt_meta.read_text())
        if (
            meta.get("method") == CKPT_METHOD
            and meta.get("shell_max") == shell_max
            and meta.get("nside") == nside
            and meta.get("variant") == variant
            and meta.get("parent") == parent
        ):
            alm_total = np.load(ckpt_alm)
            y_ident = np.load(ckpt_yident)
            done = set(meta.get("completed_groups", []))
            print(f"resuming: {len(done)} groups already folded in", flush=True)
        else:
            print("existing checkpoint incompatible (method/params), starting fresh", flush=True)

    print(
        f"variant={variant} parent={parent} lc{observer} shell_max={shell_max} "
        f"-> {len(groups)} angle-groups; NSIDE={nside}, NPIX={npix:,}, lmax={lmax} "
        f"(alm-accumulation: one alm2map at the end)",
        flush=True,
    )

    for gi, (i0, i1, theta, phi) in enumerate(groups):
        if gi in done:
            print(
                f"group {gi + 1}/{len(groups)} (shells {i0}..{i1 - 1}) already done, skip",
                flush=True,
            )
            continue

        t_group = time.time()
        group_sum = np.zeros(npix, dtype=np.float64)
        for i in range(i0, i1):
            path = f"{base}/shell_{i}/lightcone{observer}.shell_{i}.0.hdf5"
            t0 = time.time()
            print(
                f"[{tag} obs{observer}] group {gi + 1}/{len(groups)} "
                f"(theta={theta:.5f}, phi={phi:.5f}) stream shell_{i}…",
                flush=True,
            )
            group_sum += _stream_shell(root, path)
            print(f"    shell_{i} folded in ({time.time() - t0:.0f}s)", flush=True)

        if theta == 0.0 and phi == 0.0:
            y_ident += group_sum
            print(f"  group {gi + 1}: no rotation (identity)", flush=True)
        else:
            t0 = time.time()
            print(f"  group {gi + 1}: map2alm + rotate_alm (accumulate)…", flush=True)
            alm_total += rotated_alm(group_sum, theta, phi, lmax)
            print(f"  group {gi + 1}: alm folded ({time.time() - t0:.0f}s)", flush=True)
        del group_sum

        done.add(gi)
        np.save(ckpt_alm, alm_total)
        np.save(ckpt_yident, y_ident)
        ckpt_meta.write_text(
            json.dumps(
                {
                    "method": CKPT_METHOD,
                    "variant": variant,
                    "parent": parent,
                    "observer": observer,
                    "nside": nside,
                    "shell_max": shell_max,
                    "lmax": lmax,
                    "completed_groups": sorted(done),
                }
            )
        )
        print(
            f"  checkpoint saved after group {gi + 1} ({time.time() - t_group:.0f}s)",
            flush=True,
        )

    print("synthesising final map (single alm2map)…", flush=True)
    t0 = time.time()
    y = y_ident + hp.alm2map(alm_total, nside=nside, lmax=lmax)
    print(f"  alm2map done ({time.time() - t0:.0f}s)", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    hp.write_map(str(out_path), y, nest=False, overwrite=True, dtype=np.float64)
    print(
        f"\nWrote {out_path}  mean(y)={y.mean():.3e}  NSIDE={nside} (RING)",
        flush=True,
    )
    print(
        "Build complete. Checkpoint files can be deleted:",
        ckpt_alm, ckpt_yident, ckpt_meta, flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant",
        default="L2p8_m9",
        help="Simulation folder name (e.g. L1_m9, fgas+2sigma, L2p8_m9).",
    )
    p.add_argument(
        "--parent",
        default=None,
        help="Parent FLAMINGO folder for L1_m9 variants (use L1_m9). "
        "Omit for L2p8-style duplicated paths.",
    )
    p.add_argument("--observer", type=int, default=0)
    p.add_argument("--nside", type=int, default=DEFAULT_NSIDE)
    p.add_argument("--shell-max", type=int, default=DEFAULT_SHELL_MAX)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("/scratch/scratch-lxu/flamingo_map_build"),
    )
    args = p.parse_args()
    build_y_map(
        args.variant,
        args.observer,
        args.shell_max,
        args.out,
        args.ckpt_dir,
        parent=args.parent,
        nside=args.nside,
    )


if __name__ == "__main__":
    main()
