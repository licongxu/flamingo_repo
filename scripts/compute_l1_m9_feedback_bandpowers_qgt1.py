"""Compute q>1 masked tSZ bandpowers for all L1_m9 feedback variants.

Full-sky bandpowers already exist in ``data_paper/feedback_bandpower`` from
``compute_l1_m9_feedback_bandpowers.py`` (q>5 run). This script adds only the
``qgt1`` masked products for every feedback prescription.

Run::

    python scripts/compute_l1_m9_feedback_bandpowers_qgt1.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt

REPO = Path(__file__).resolve().parents[1]
FEEDBACK = REPO / "scripts" / "compute_l1_m9_feedback_bandpowers.py"

spec = importlib.util.spec_from_file_location("feedback_bp", FEEDBACK)
mod = importlib.util.module_from_spec(spec)
sys.modules["feedback_bp"] = mod
spec.loader.exec_module(mod)

Q_CUT = 1.0
CUT_TAG = "qgt1"


def main() -> None:
    t0 = time.time()
    mod.OUT_DIR.mkdir(parents=True, exist_ok=True)
    pixwin2 = hp.pixwin(4096, lmax=mod.LMAX) ** 2
    ell = np.arange(mod.LMAX + 1, dtype=float)
    mask_floor = np.deg2rad(2.0 * mod.FWHM_ARCMIN / 60.0)

    metadata: dict[str, dict] = {}
    for variant, map_stem, cat_stem in mod.VARIANTS:
        print(f"\n=== {variant}: q>{Q_CUT:g} masked only ===", flush=True)
        ymap = hp.read_map(mod.map_file(map_stem), dtype=np.float64)
        nside = hp.npix2nside(ymap.size)
        cat = mod.load_catalogue(mod.cat_file(cat_stem))
        q, theta, phi, t500 = cat["q"], cat["theta"], cat["phi"], cat["t500"]

        keep = q > Q_CUT
        radius = np.maximum(mod.R_MULT * t500[keep], mask_floor)
        mask_bin = mod.binary_disc_mask(nside, theta[keep], phi[keep], radius)
        f_sky_raw = float(mask_bin.mean())
        mask_apo = nmt.mask_apodization(mask_bin, mod.APOSIZE_DEG, apotype=mod.APOTYPE)
        f_sky_eff = float(np.mean(mask_apo**2))
        del mask_bin
        print(
            f"  q>{Q_CUT:g}: {int(keep.sum()):,} halos, f_sky raw={f_sky_raw:.4f} "
            f"eff={f_sky_eff:.4f}",
            flush=True,
        )

        cl_masked = mod.decoupled_cl_per_ell(ymap, mask_apo, pixwin2)
        del mask_apo, ymap
        dl18_masked = mod.bin_dl_18(ell, cl_masked)
        dl12_masked = mod.bin_cl_log(ell, cl_masked)

        common = f"L1_m9 {variant} lightcone-0 tSZ; HEALPix Nside=4096 pixel window deconvolved"
        mask_note = (
            f"masked q>{Q_CUT:g} ({mod.TAG}); r=max({mod.R_MULT:g}*theta500, 2x{mod.FWHM_ARCMIN:g}arcmin), "
            f"{mod.APOTYPE} apodization {mod.APOSIZE_DEG} deg, masked monopole subtracted, "
            f"NaMaster MASTER per-ell (nlb=1, lmax={mod.LMAX})"
        )
        bin18_note = "uniform mean of D_ell over inclusive Planck bins"
        bin12_note = f"uniform mean of C_ell in log bins, Delta ln ell = {mod.DLN_ELL}, ell_max = {mod.LMAX}"
        cols = "ell_eff  1e12_D_ell_yy"

        mod.write_bandpowers(
            mod.OUT_DIR / f"Dl_yy_L1_m9_{variant}_masked_{CUT_TAG}_{mod.TAG}_binned_18.txt",
            mod.ELL_EFF,
            dl18_masked * 1e12,
            f"{common}; {mask_note}; {bin18_note}\n{cols}",
        )
        mod.write_bandpowers(
            mod.OUT_DIR
            / f"Dl_yy_L1_m9_{variant}_masked_{CUT_TAG}_{mod.TAG}_logbins_dln0p4_lmax10000.txt",
            mod.LOG_CENTRES,
            dl12_masked * 1e12,
            f"{common}; {mask_note}; {bin12_note}\n{cols}",
        )
        metadata[variant] = {
            "n_masked": int(keep.sum()),
            "f_sky_raw": f_sky_raw,
            "f_sky_eff": f_sky_eff,
            "runtime_seconds": time.time() - t0,
        }
        print(f"  wrote {variant} ({time.time() - t0:.0f}s)", flush=True)

    meta_path = mod.OUT_DIR / f"L1_m9_feedback_bandpowers_{CUT_TAG}_metadata.json"
    with open(meta_path, "w") as handle:
        json.dump(
            {
                "q_cut": Q_CUT,
                "cut_tag": CUT_TAG,
                "variants": metadata,
                "runtime_seconds": time.time() - t0,
            },
            handle,
            indent=2,
        )
    print(f"\ndone ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
