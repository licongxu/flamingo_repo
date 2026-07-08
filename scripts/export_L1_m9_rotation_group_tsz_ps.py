"""Cache tSZ D_ell for L1_m9 rotation-group maps (notebook 40).

For each feedback variant, measures D_ell for every rotation-group shell sum
(nb39 conventions: monopole subtracted, ``anafast`` with ``iter=0``, pixel
window deconvolved, f_sky corrected) plus two combined sums (all groups;
groups >= 1, i.e. excluding the identity group 0), and writes one ``.npz``
per variant under ``data/nb40_l1_m9_rotation_group_tsz_ps/``.

Existing per-variant ``.npz`` files are skipped unless ``--force``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import healpy as hp
import numpy as np

REPO = Path("/scratch/scratch-lxu/flamingo_repo")
MAP_DIR = Path("/rds/rds-lxu/flamingo/L1_m9/maps/rotation_groups")
OUT_DIR = REPO / "data/nb40_l1_m9_rotation_group_tsz_ps"
LMAX = 6000

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


def measure_dl_map(y: np.ndarray, lmax: int = LMAX) -> np.ndarray:
    """D_ell with monopole subtracted, pixwin and f_sky corrected (nb39)."""
    nside = hp.npix2nside(y.size)
    fsky = float(np.mean(y != 0.0))
    cl = hp.anafast(y - y.mean(), lmax=lmax, iter=0)
    ell = np.arange(cl.size)
    cl = cl / (hp.pixwin(nside, lmax=lmax) ** 2 * fsky)
    return ell * (ell + 1) / (2 * np.pi) * cl


def export_variant(variant: str, *, force: bool = False) -> None:
    out_npz = OUT_DIR / f"{variant}.npz"
    if out_npz.exists() and not force:
        print(f"{variant}: {out_npz.name} exists, skip", flush=True)
        return

    meta = json.loads((MAP_DIR / f"rotation_groups_{variant}_lc0.json").read_text())
    groups = meta["groups"]
    dl_groups = []
    y_all = None
    y_g1p = None
    for g in groups:
        t0 = time.time()
        y = hp.read_map(str(MAP_DIR / g["fits"]), dtype=np.float64)
        dl_groups.append(measure_dl_map(y))
        y_all = y if y_all is None else y_all + y
        if g["group"] >= 1:
            y_g1p = y if y_g1p is None else y_g1p + y
        print(
            f"{variant}: group {g['group']} shells {g['shell_first']}-{g['shell_last']} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
    dl_all = measure_dl_map(y_all)
    dl_g1p = measure_dl_map(y_g1p)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        variant=variant,
        ell=np.arange(LMAX + 1),
        dl=np.stack(dl_groups, axis=0),
        dl_all=dl_all,
        dl_g1plus=dl_g1p,
        group=np.array([g["group"] for g in groups]),
        shell_first=np.array([g["shell_first"] for g in groups]),
        shell_last=np.array([g["shell_last"] for g in groups]),
        z_inner=np.array([g["z_inner"] for g in groups]),
        z_outer=np.array([g["z_outer"] for g in groups]),
        rot_theta_rad=np.array([g["rot_theta_rad"] for g in groups]),
        rot_phi_rad=np.array([g["rot_phi_rad"] for g in groups]),
        sum_y=np.array([g["sum_y"] for g in groups]),
        lmax=LMAX,
        map_dir=str(MAP_DIR),
    )
    print(f"{variant}: wrote {out_npz}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", nargs="+", default=VARIANTS)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    for variant in args.variants:
        export_variant(variant, force=args.force)


if __name__ == "__main__":
    main()
