"""Masked tSZ power spectra of the fiducial L1_m9 map, best-fit-alpha-fixed q.

Masks the ``nside=4096`` fiducial lightcone-0 Compton-y map with clusters from
``halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommz_alpha_fixed_1p12.csv``
(``q_from_mz`` from the custom-GNFW alpha_SZ=1.12 best-fit scaling relation),
above each threshold in ``Q_CUTS``.

The masking prescription is the one used for the synthetic (painted-map) data
of the paper, i.e. ``compute_ps_from_painted_maps.py`` / fig-1 schematic:

* disc radius per cluster ``max(4 * theta_500, 2 * FWHM)`` with
  ``FWHM = 10 arcmin`` -- the painting truncation radius ``theta_max``;
  ``theta_500 = R_500c / D_A(z)`` evaluated in the D3A cosmology;
* binary disc mask (``healpy.query_disc``), C2-apodized at 0.25 deg;
* mask-weighted monopole subtracted before the MASTER estimate;
* NaMaster per-ell (``nlb=1``) decoupled pseudo-Cl, no beam (the FLAMINGO map
  is unsmoothed), HEALPix pixel window deconvolved per ell.

Bandpowers are stored in the two conventions already used for the full-sky
data products in ``data_paper/binned_bandpowers``:

* the 18 Planck bins -- uniform mean of ``D_ell`` over the inclusive integer
  bins, stored at the standard effective multipoles (matches
  ``Dl_yy_L1_m9_fullsky_binned_18.txt``);
* 12 logarithmic bins of width ``Delta ln ell = 0.4`` up to ``lmax=10000`` --
  uniform mean of ``C_ell`` per bin times ``ell(ell+1)/2pi`` at the geometric
  bin centre (matches ``..._logbins_dln0p4_lmax10000_pixwin_deconvolved.txt``).

Cluster positions are the yang26-rotated columns: the rotation-sanity check
below shows the rotated positions land on tSZ peaks while the natural ones do
not. A unit-mask NaMaster run validates the estimator against the stored
full-sky 18-bin bandpowers.

Run::

    python scripts/powerspectra/compute_l1_m9_masked_ps_alpha_fixed_1p12.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np
import pymaster as nmt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from flamingo.catalogue import load_masking_catalogue  # noqa: E402
from flamingo.catalogue.frame import rotation_sanity  # noqa: E402
from flamingo.masking import disc_mask  # noqa: E402
from flamingo.powerspectra.bandpowers import (  # noqa: E402
    PLANCK_ELL_EFF,
    bin_log_dl,
    bin_planck_dl,
    write_bandpowers,
)
from flamingo.powerspectra.namaster import decoupled_cl_per_ell  # noqa: E402
from flamingo.powerspectra.q_selection import cut_tags  # noqa: E402

MAP_FILE = Path("/rds/rds-lxu/flamingo/L1_m9/maps/y_unlensed_L1_m9_lc0_nside4096.fits")
CAT_FILE = Path(
    "/rds/rds-lxu/flamingo/L1_m9/catalogues/"
    "halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommz_alpha_fixed_1p12.csv"
)
OUT_DIR = REPO / "data_paper" / "binned_bandpowers"
FULLSKY_18 = OUT_DIR / "Dl_yy_L1_m9_fullsky_binned_18.txt"
TAG = "qfrommz_alpha_fixed_1p12"

Q_CUTS = [50.0, 20.0, 10.0, 5.0, 1.0]

# Synthetic-data masking prescription (painted-map benchmark).
FWHM_ARCMIN = 10.0
R_MULT = 4.0
APOSIZE_DEG = 0.25
APOTYPE = "C2"

LMAX = 10000
DLN_ELL = 0.4

def main(q_cuts: list[float] | None = None) -> None:
    q_cuts = Q_CUTS if q_cuts is None else q_cuts
    tags = cut_tags(q_cuts)
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== map + catalogue ===", flush=True)
    ymap = hp.read_map(MAP_FILE, dtype=np.float64)
    nside = hp.npix2nside(ymap.size)
    cat = load_masking_catalogue(CAT_FILE)
    q, theta, phi, t500 = cat["q"], cat["theta"], cat["phi"], cat["t500"]
    print(f"  nside={nside}, {q.size:,} halos, mean y={ymap.mean():.3e}", flush=True)

    sample = cat["nat_sample"]
    r_nat = rotation_sanity(ymap, sample[:, 0], sample[:, 1])
    r_rot = rotation_sanity(ymap, sample[:, 2], sample[:, 3])
    print(
        f"  rotation sanity: rot ratio={r_rot['ratio']:.2f}, nat ratio={r_nat['ratio']:.2f}",
        flush=True,
    )
    if r_rot["ratio"] <= r_nat["ratio"]:
        raise RuntimeError("rotated positions do not trace tSZ peaks better than natural ones")

    pixwin2 = hp.pixwin(nside, lmax=LMAX) ** 2
    ell = np.arange(LMAX + 1, dtype=float)
    mask_floor = np.deg2rad(2.0 * FWHM_ARCMIN / 60.0)

    print("=== unit-mask validation vs stored full-sky 18-bin ===", flush=True)
    cl_unit = decoupled_cl_per_ell(ymap, np.ones_like(ymap), pixwin2, lmax=LMAX)
    dl18_unit = bin_planck_dl(ell, cl_unit)
    stored = np.loadtxt(FULLSKY_18)[:, 1]
    max_dev = float(np.max(np.abs(dl18_unit * 1e12 / stored - 1.0)))
    print(f"  NaMaster unit mask vs stored anafast: max|frac diff| = {max_dev:.3e}", flush=True)

    metadata: dict[str, dict] = {}
    dl18_all, dl12_all = [], []
    for cut, tag in zip(q_cuts, tags, strict=True):
        keep = q > cut
        radius = np.maximum(R_MULT * t500[keep], mask_floor)
        print(
            f"=== q>{cut:g}: masking {keep.sum():,} halos "
            f"(radius floor {np.rad2deg(mask_floor) * 60:.0f} arcmin) ===",
            flush=True,
        )
        mask_bin = disc_mask(nside, theta[keep], phi[keep], radius, inclusive=False)
        f_sky_raw = float(mask_bin.mean())
        mask_apo = nmt.mask_apodization(mask_bin, APOSIZE_DEG, apotype=APOTYPE)
        f_sky_eff = float(np.mean(mask_apo**2))
        del mask_bin
        print(
          f"  f_sky raw={f_sky_raw:.4f} eff={f_sky_eff:.4f} ({time.time() - t0:.0f}s)",
          flush=True,
        )

        cl_masked = decoupled_cl_per_ell(ymap, mask_apo, pixwin2, lmax=LMAX)
        del mask_apo
        dl18 = bin_planck_dl(ell, cl_masked)
        ell_log, dl12 = bin_log_dl(ell, cl_masked)
        dl18_all.append(dl18)
        dl12_all.append(dl12)

        header18 = (
            f"L1_m9 fiducial masked tSZ, q>{cut:g} ({TAG}); synthetic-data masking: "
            f"r=max(4*theta500, 2x10arcmin), {APOTYPE} apodization {APOSIZE_DEG} deg, "
            "masked monopole subtracted, NaMaster MASTER per-ell, pixwin deconvolved; "
            "uniform mean over inclusive Planck bins\nell_eff  1e12_D_ell_yy"
        )
        header12 = (
            f"L1_m9 fiducial masked tSZ, q>{cut:g} ({TAG}); same masking as the binned_18 "
            f"file; Delta ln ell = {DLN_ELL}; ell_max = {LMAX}; "
            "HEALPix Nside=4096 pixel window deconvolved\nell_eff  1e12_D_ell_yy"
        )
        write_bandpowers(
            OUT_DIR / f"Dl_yy_L1_m9_masked_{tag}_{TAG}_binned_18.txt",
            PLANCK_ELL_EFF,
            dl18 * 1e12,
            header18,
        )
        write_bandpowers(
            OUT_DIR / f"Dl_yy_L1_m9_masked_{tag}_{TAG}_logbins_dln0p4_lmax10000.txt",
            ell_log,
            dl12 * 1e12,
            header12,
        )
        metadata[tag] = {
            "q_cut": cut,
            "n_masked": int(keep.sum()),
            "f_sky_raw": f_sky_raw,
            "f_sky_eff": f_sky_eff,
        }
        print(f"  wrote {tag} bandpowers ({time.time() - t0:.0f}s)", flush=True)

    np.savez(
        OUT_DIR / f"L1_m9_masked_{TAG}.npz",
        ell_eff_18=PLANCK_ELL_EFF,
        dl_18=np.stack(dl18_all),
        ell_log=ell_log,
        dl_12=np.stack(dl12_all),
        q_cuts=np.array(q_cuts),
        cut_tags=np.array(tags),
    )
    meta = {
        "map": str(MAP_FILE),
        "catalogue": str(CAT_FILE),
        "masking": {
            "radius": "max(4*theta500, 2*FWHM), FWHM=10 arcmin",
            "theta500": "R_500c/D_A(z), D3A cosmology",
            "positions": "theta_rot_rad/phi_rot_rad (yang26-rotated frame)",
            "apodization": f"{APOTYPE} {APOSIZE_DEG} deg",
            "monopole": "mask-weighted, subtracted",
            "estimator": f"NaMaster MASTER, nlb=1, lmax={LMAX}, no beam, pixwin deconvolved",
        },
        "unit_mask_vs_stored_fullsky_max_frac_diff": max_dev,
        "cuts": metadata,
        "runtime_seconds": time.time() - t0,
    }
    with open(OUT_DIR / f"L1_m9_masked_{TAG}_metadata.json", "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--q-cuts", nargs="+", type=float, default=Q_CUTS)
    main(parser.parse_args().q_cuts)
