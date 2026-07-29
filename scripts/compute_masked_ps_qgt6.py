"""Compute masked tSZ bandpowers at ``q > 6`` for L1_m9 and L2p8_m9.

The ``q > 6`` threshold matches the Planck uRC analysis in TopoSZ (fig12 data).
Reuses the masking pipeline from the per-resolution scripts but runs only this cut.

Run::

    python scripts/compute_masked_ps_qgt6.py L1_m9
    python scripts/compute_masked_ps_qgt6.py L2p8_m9
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_variant(variant: str) -> None:
    if variant == "L1_m9":
        mod = _load_module("l1_masked", REPO / "scripts/compute_l1_m9_masked_ps_alpha_fixed_1p12.py")
    elif variant == "L2p8_m9":
        mod = _load_module("l2p8_masked", REPO / "scripts/compute_l2p8_m9_masked_ps_alpha_fixed_1p12.py")
    else:
        raise ValueError(f"unknown variant {variant!r}")

    import healpy as hp
    import json
    import numpy as np
    import pymaster as nmt
    import time

    t0 = time.time()
    mod.OUT_DIR.mkdir(parents=True, exist_ok=True)

    cut, tag = 6.0, "qgt6"
    print(f"=== {variant}: q>{cut:g} only ===", flush=True)

    ymap = hp.read_map(mod.MAP_FILE, dtype=np.float64)
    nside = hp.npix2nside(ymap.size)
    cat = mod.load_catalogue()
    q, theta, phi, t500 = cat["q"], cat["theta"], cat["phi"], cat["t500"]
    print(f"  nside={nside}, {q.size:,} halos", flush=True)

    pixwin2 = hp.pixwin(nside, lmax=mod.LMAX) ** 2
    ell = np.arange(mod.LMAX + 1, dtype=float)
    mask_floor = np.deg2rad(2.0 * mod.FWHM_ARCMIN / 60.0)

    keep = q > cut
    radius = np.maximum(mod.R_MULT * t500[keep], mask_floor)
    print(f"  masking {keep.sum():,} halos", flush=True)

    mask_bin = mod.binary_disc_mask(nside, theta[keep], phi[keep], radius)
    f_sky_raw = float(mask_bin.mean())
    mask_apo = nmt.mask_apodization(mask_bin, mod.APOSIZE_DEG, apotype=mod.APOTYPE)
    f_sky_eff = float(np.mean(mask_apo**2))
    del mask_bin
    print(f"  f_sky raw={f_sky_raw:.4f} eff={f_sky_eff:.4f}", flush=True)

    cl_masked = mod.decoupled_cl_per_ell(ymap, mask_apo, pixwin2)
    del mask_apo
    dl18 = mod.bin_dl_18(ell, cl_masked)
    ell_log, dl12 = mod.bin_cl_log(ell, cl_masked)

    header18 = (
        f"{variant} fiducial masked tSZ, q>{cut:g} ({mod.TAG}); synthetic-data masking: "
        f"r=max(4*theta500, 2x10arcmin), {mod.APOTYPE} apodization {mod.APOSIZE_DEG} deg, "
        "masked monopole subtracted, NaMaster MASTER per-ell, pixwin deconvolved; "
        "uniform mean over inclusive Planck bins\nell_eff  1e12_D_ell_yy"
    )
    header12 = (
        f"{variant} fiducial masked tSZ, q>{cut:g} ({mod.TAG}); same masking as the binned_18 "
        f"file; Delta ln ell = {mod.DLN_ELL}; ell_max = {mod.LMAX}; "
        f"HEALPix Nside={nside} pixel window deconvolved\nell_eff  1e12_D_ell_yy"
    )
    mod.write_bandpowers(
        mod.OUT_DIR / f"Dl_yy_{variant}_masked_{tag}_{mod.TAG}_binned_18.txt",
        mod.ELL_EFF,
        dl18 * 1e12,
        header18,
    )
    mod.write_bandpowers(
        mod.OUT_DIR / f"Dl_yy_{variant}_masked_{tag}_{mod.TAG}_logbins_dln0p4_lmax10000.txt",
        ell_log,
        dl12 * 1e12,
        header12,
    )

    meta_path = mod.OUT_DIR / f"{variant}_masked_{mod.TAG}_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {
            "map": str(mod.MAP_FILE),
            "catalogue": str(mod.CAT_FILE),
            "masking": {
                "radius": "max(4*theta500, 2*FWHM), FWHM=10 arcmin",
                "theta500": "R_500c/D_A(z), D3A cosmology",
                "positions": "theta_rot_rad/phi_rot_rad (yang26-rotated frame)",
                "apodization": f"{mod.APOTYPE} {mod.APOSIZE_DEG} deg",
                "monopole": "mask-weighted, subtracted",
                "estimator": f"NaMaster MASTER, nlb=1, lmax={mod.LMAX}, no beam, pixwin deconvolved",
            },
            "cuts": {},
        }
    meta["cuts"][tag] = {
        "q_cut": cut,
        "n_masked": int(keep.sum()),
        "f_sky_raw": f_sky_raw,
        "f_sky_eff": f_sky_eff,
        "runtime_seconds": time.time() - t0,
    }
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"  wrote {tag} ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=["L1_m9", "L2p8_m9"])
    args = parser.parse_args()
    run_variant(args.variant)


if __name__ == "__main__":
    main()
