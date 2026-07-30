"""Random-position control for the masking-radius null test.

The masked tSZ spectrum keeps falling as the masking radius grows, but two very
different things drive that fall:

1. real cluster signal being removed -- the effect we want, which should
   saturate once the disc encloses the cluster;
2. sky simply being removed. MASTER decoupling is built to undo this, so the
   residual response to area loss should be small -- but it is an assumption,
   and at the aggressive end of the sweep (``q > 1`` at ``8 theta_500`` keeps
   only half the sky) it is worth measuring rather than trusting.

This script measures (2) by rebuilding each mask with the *same number of discs
and the same radius distribution* but with the disc centres scattered uniformly
over the sphere, so they no longer sit on clusters. Comparing this control to
the real sweep says how much of the measured decrement is physical.

Positions are drawn once per ``(q_cut, r_mult)`` from a fixed seed, so the
control is reproducible. Everything downstream -- apodization, monopole
subtraction, estimator, binning -- is imported unchanged from
:mod:`masking_radius_null_test`.

Run::

    python scripts/masking_radius_random_control.py --variant L1_m9 --q-cut 5
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

_spec = importlib.util.spec_from_file_location(
    "masking_radius_null_test", Path(__file__).parent / "masking_radius_null_test.py"
)
nt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nt)

SEED = 20260729


def random_positions(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``n`` directions drawn uniformly on the sphere (colatitude, longitude)."""
    rng = np.random.default_rng(seed)
    theta = np.arccos(rng.uniform(-1.0, 1.0, size=n))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return theta, phi


def control_path(variant: str, cut: float, r_mult: float) -> Path:
    radius_tag = f"r{r_mult:g}".replace(".", "p")
    return nt.OUT_DIR / f"{variant}_random_qgt{cut:g}_{radius_tag}.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", required=True, choices=sorted(nt.VARIANTS))
    parser.add_argument("--q-cut", type=float, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    variant, cut = args.variant, args.q_cut
    spec = nt.VARIANTS[variant]
    t0 = time.time()
    nt.OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== {variant} random control, q>{cut:g} ===", flush=True)
    ymap = hp.read_map(spec["map"], dtype=np.float64)
    nside = hp.npix2nside(ymap.size)
    cat = nt.load_catalogue(spec["catalogue"])
    keep = cat["q"] > cut
    t500 = cat["t500"][keep]
    print(f"  nside={nside}, {t500.size:,} discs to scatter", flush=True)

    pixwin2 = hp.pixwin(nside, lmax=nt.LMAX) ** 2
    ell = np.arange(nt.LMAX + 1, dtype=float)

    for i, r_mult in enumerate(nt.R_MULTS):
        if r_mult == 0.0:
            continue
        out = control_path(variant, cut, r_mult)
        if out.exists() and not args.force:
            print(f"  cached {out.name}", flush=True)
            continue

        t1 = time.time()
        theta_r, phi_r = random_positions(t500.size, SEED + i)
        mask_bin = nt.binary_disc_mask(nside, theta_r, phi_r, r_mult * t500)
        f_sky_raw = float(mask_bin.mean())
        mask_apo = nmt.mask_apodization(mask_bin, nt.APOSIZE_DEG, apotype=nt.APOTYPE)
        f_sky_eff = float(np.mean(mask_apo**2))
        del mask_bin
        print(
            f"  r={r_mult:g}: f_sky raw={f_sky_raw:.4f} eff={f_sky_eff:.4f} "
            f"({time.time() - t1:.0f}s)",
            flush=True,
        )

        cl = nt.decoupled_cl_per_ell(ymap, mask_apo, pixwin2)
        del mask_apo
        ell_log, dl_12 = nt.bin_cl_log(ell, cl)
        np.savez(
            out,
            variant=variant,
            q_cut=cut,
            r_mult=r_mult,
            randomised=True,
            n_masked=int(t500.size),
            f_sky_raw=f_sky_raw,
            f_sky_eff=f_sky_eff,
            ell_18=nt.ELL_EFF,
            dl_18=nt.bin_dl_18(ell, cl) * 1e12,
            ell_log=ell_log,
            dl_12=dl_12 * 1e12,
            nside=nside,
        )
        print(f"  wrote {out.name} ({time.time() - t0:.0f}s elapsed)", flush=True)

    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
