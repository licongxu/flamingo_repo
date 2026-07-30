"""Full-sky and q>5 masked tSZ bandpowers for every L1_m9 feedback prescription.

Same maps, catalogues, masking and binning as the fiducial-only pipeline in
``compute_l1_m9_masked_ps_alpha_fixed_1p12.py``, applied to the nine
``nside=4096`` lightcone-0 Compton-y maps (fiducial + eight feedback variants).
Cluster catalogues all carry the identical
``_yang26rot_qfrommz_alpha_fixed_1p12.csv`` suffix, i.e. ``q_from_mz`` from the
custom-GNFW alpha_SZ=1.12 best-fit scaling relation.

Masking (only the q>5 cut is computed here):

* disc radius per cluster ``max(4 * theta_500, 2 * FWHM)`` with
  ``FWHM = 10 arcmin``; ``theta_500 = R_500c / D_A(z)`` in the D3A cosmology;
* binary disc mask (``healpy.query_disc``), C2-apodized at 0.25 deg;
* mask-weighted monopole subtracted before the MASTER estimate;
* NaMaster per-ell (``nlb=1``) decoupled pseudo-Cl, no beam, HEALPix pixel
  window deconvolved per ell.

Full sky uses ``healpy.anafast`` on the mean-subtracted map with the same pixel
window deconvolution -- verified against the NaMaster unit-mask run and against
both stored fiducial files to <1e-7 (see ``_check_fullsky_binning.py``).

Both binning conventions of ``data_paper/binned_bandpowers`` are written:

* the 18 Planck bins -- uniform mean of ``D_ell`` over the inclusive integer
  bins, at the standard effective multipoles;
* 12 logarithmic bins of width ``Delta ln ell = 0.4`` up to ``lmax=10000`` --
  uniform mean of ``C_ell`` per bin times ``ell(ell+1)/2pi`` at the geometric
  bin centre.

Run::

    python scripts/compute_l1_m9_feedback_bandpowers.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd
import pymaster as nmt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from flamingo.catalogue import theta_500  # noqa: E402
from flamingo.catalogue.frame import rotation_sanity  # noqa: E402

MAP_DIR = Path("/rds/rds-lxu/flamingo/L1_m9/maps")
CAT_DIR = Path("/rds/rds-lxu/flamingo/L1_m9/catalogues")
OUT_DIR = REPO / "data_paper" / "feedback_bandpower"
REF_DIR = REPO / "data_paper" / "binned_bandpowers"
TAG = "qfrommz_alpha_fixed_1p12"

# (variant tag used in output filenames, map stem, catalogue stem)
VARIANTS = [
    ("fiducial", "L1_m9", "L1_m9"),
    ("fgas+2sigma", "fgas+2sigma", "fgas+2sigma"),
    ("fgas-2sigma", "fgas-2sigma", "fgas-2sigma"),
    ("fgas-4sigma", "fgas-4sigma", "fgas-4sigma"),
    ("fgas-8sigma", "fgas-8sigma", "fgas-8sigma"),
    ("Jet", "Jet", "Jet"),
    ("Jet_fgas-4sigma", "Jet_fgas-4sigma", "Jet_fgas-4sigma"),
    ("Mstar-1sigma", "Mstar-1sigma", "Mstar-1sigma"),
    ("Mstar-1sigma_fgas-4sigma", "Mstar-1sigma_fgas-4sigma", "Mstar-1sigma_fgas-4sigma"),
]

Q_CUT = 5.0
CUT_TAG = "qgt5"

# Synthetic-data masking prescription (painted-map benchmark).
FWHM_ARCMIN = 10.0
R_MULT = 4.0
APOSIZE_DEG = 0.25
APOTYPE = "C2"

LMAX = 10000
ELL_MIN = np.array(
    [9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835]
)
ELL_MAX = np.array(
    [12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835, 1085]
)
ELL_EFF = np.array(
    [10.0, 13.5, 18.0, 23.5, 30.5, 40.0, 52.5, 68.5, 89.5, 117.0, 152.5, 198.0,
     257.5, 335.5, 436.5, 567.5, 738.0, 959.5]
)
N_LOG_BINS = 12
DLN_ELL = 0.4
LOG_EDGES = LMAX * np.exp(-N_LOG_BINS * DLN_ELL) * np.exp(DLN_ELL * np.arange(N_LOG_BINS + 1))
LOG_CENTRES = np.sqrt(LOG_EDGES[:-1] * LOG_EDGES[1:])

CHUNK = 1_000_000
COLUMNS = ["z", "R_500c_Mpc", "theta_rot_rad", "phi_rot_rad", "q_from_mz"]
SAMPLE_COLUMNS = ["theta_nat_rad", "phi_nat_rad"]


def map_file(stem: str) -> Path:
    return MAP_DIR / f"y_unlensed_{stem}_lc0_nside4096.fits"


def cat_file(stem: str) -> Path:
    return CAT_DIR / f"halo_catalogue_M500c_5e13_zlt3_{stem}_yang26rot_qfrommz_alpha_fixed_1p12.csv"


def load_catalogue(path: Path) -> dict[str, np.ndarray]:
    """Stream the catalogue; keep only what the masking step needs."""
    cols: dict[str, list[np.ndarray]] = {"theta": [], "phi": [], "t500": [], "q": [], "nat": []}
    n_rows = 0
    rng = np.random.default_rng(0)
    for chunk in pd.read_csv(
        path, comment="#", usecols=COLUMNS + SAMPLE_COLUMNS, chunksize=CHUNK
    ):
        cols["theta"].append(chunk["theta_rot_rad"].to_numpy(np.float64))
        cols["phi"].append(chunk["phi_rot_rad"].to_numpy(np.float64))
        cols["q"].append(chunk["q_from_mz"].to_numpy(np.float64))
        cols["t500"].append(
            theta_500(chunk["R_500c_Mpc"].to_numpy(np.float64), chunk["z"].to_numpy(np.float64))
        )
        take = rng.random(len(chunk)) < (30000 / 1.53e6)
        if take.any():
            cols["nat"].append(
                chunk.loc[
                    take, ["theta_nat_rad", "phi_nat_rad", "theta_rot_rad", "phi_rot_rad"]
                ].to_numpy()
            )
        n_rows += len(chunk)
    out = {k: np.concatenate(v) for k, v in cols.items() if k != "nat"}
    out["nat_sample"] = np.concatenate(cols["nat"])
    print(f"  catalogue: {n_rows:,} halos", flush=True)
    return out


def binary_disc_mask(nside: int, theta: np.ndarray, phi: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Binary mask, 0 inside discs -- as the synthetic benchmark (query_disc defaults)."""
    mask = np.ones(hp.nside2npix(nside), dtype=np.float64)
    for th, ph, rr in zip(theta, phi, radius):
        mask[hp.query_disc(nside, hp.ang2vec(th, ph), float(rr))] = 0.0
    return mask


def decoupled_cl_per_ell(ymap: np.ndarray, mask_apo: np.ndarray, pixwin2: np.ndarray) -> np.ndarray:
    """Pixwin-corrected decoupled C_ell at every ell=2..LMAX (NaMaster MASTER)."""
    w = mask_apo
    m = ymap - float(np.sum(w * ymap) / np.sum(w))
    field = nmt.NmtField(w, [m], lmax=LMAX)
    bins = nmt.NmtBin.from_lmax_linear(LMAX, nlb=1)
    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(field, field, bins)
    cl = workspace.decouple_cell(nmt.compute_coupled_cell(field, field))[0]

    ell_eff = bins.get_effective_ells().astype(int)
    cl_full = np.full(LMAX + 1, np.nan)
    cl_full[ell_eff] = cl
    return cl_full / pixwin2


def fullsky_cl_per_ell(ymap: np.ndarray, pixwin2: np.ndarray) -> np.ndarray:
    """Pixwin-corrected full-sky C_ell from anafast on the mean-subtracted map."""
    return hp.anafast(ymap - ymap.mean(), lmax=LMAX) / pixwin2


def bin_dl_18(ell: np.ndarray, cl: np.ndarray) -> np.ndarray:
    """Uniform mean of D_ell over the inclusive Planck bins (synthetic convention)."""
    dl = ell * (ell + 1.0) * cl / (2.0 * np.pi)
    return np.array(
        [np.nanmean(dl[(ell >= lo) & (ell <= hi)]) for lo, hi in zip(ELL_MIN, ELL_MAX)]
    )


def bin_cl_log(ell: np.ndarray, cl: np.ndarray) -> np.ndarray:
    """Uniform mean of C_ell in left-inclusive log bins, D_ell at geometric centres."""
    out = np.empty(N_LOG_BINS)
    for i, (lo, hi) in enumerate(zip(LOG_EDGES[:-1], LOG_EDGES[1:])):
        inside = (ell >= lo) & (ell <= hi) if i == N_LOG_BINS - 1 else (ell >= lo) & (ell < hi)
        if not np.any(inside):
            raise ValueError(f"log bin [{lo}, {hi}] is empty")
        out[i] = np.nanmean(cl[inside])
    return LOG_CENTRES * (LOG_CENTRES + 1.0) * out / (2.0 * np.pi)


def write_bandpowers(path: Path, ell: np.ndarray, dl_1e12: np.ndarray, header: str) -> None:
    np.savetxt(path, np.column_stack([ell, dl_1e12]), fmt="%.6e", header=header)


def max_frac_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a / b - 1.0)))


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pixwin2 = hp.pixwin(4096, lmax=LMAX) ** 2
    ell = np.arange(LMAX + 1, dtype=float)
    mask_floor = np.deg2rad(2.0 * FWHM_ARCMIN / 60.0)

    metadata: dict[str, dict] = {}
    for variant, map_stem, cat_stem in VARIANTS:
        print(f"\n=== {variant} ({time.time() - t0:.0f}s) ===", flush=True)
        ymap = hp.read_map(map_file(map_stem), dtype=np.float64)
        nside = hp.npix2nside(ymap.size)
        if nside != 4096:
            raise RuntimeError(f"{variant}: expected nside=4096, got {nside}")
        cat = load_catalogue(cat_file(cat_stem))
        q, theta, phi, t500 = cat["q"], cat["theta"], cat["phi"], cat["t500"]

        sample = cat["nat_sample"]
        r_nat = rotation_sanity(ymap, sample[:, 0], sample[:, 1])
        r_rot = rotation_sanity(ymap, sample[:, 2], sample[:, 3])
        print(f"  rotation sanity: rot={r_rot['ratio']:.2f}, nat={r_nat['ratio']:.2f}", flush=True)
        if r_rot["ratio"] <= r_nat["ratio"]:
            raise RuntimeError(f"{variant}: rotated positions do not trace tSZ peaks")

        cl_full = fullsky_cl_per_ell(ymap, pixwin2)
        dl18_full = bin_dl_18(ell, cl_full)
        dl12_full = bin_cl_log(ell, cl_full)
        print(f"  full sky done ({time.time() - t0:.0f}s)", flush=True)

        keep = q > Q_CUT
        radius = np.maximum(R_MULT * t500[keep], mask_floor)
        mask_bin = binary_disc_mask(nside, theta[keep], phi[keep], radius)
        f_sky_raw = float(mask_bin.mean())
        mask_apo = nmt.mask_apodization(mask_bin, APOSIZE_DEG, apotype=APOTYPE)
        f_sky_eff = float(np.mean(mask_apo**2))
        del mask_bin
        print(
            f"  q>{Q_CUT:g}: {int(keep.sum()):,} halos, f_sky raw={f_sky_raw:.4f} "
            f"eff={f_sky_eff:.4f} ({time.time() - t0:.0f}s)",
            flush=True,
        )

        cl_masked = decoupled_cl_per_ell(ymap, mask_apo, pixwin2)
        del mask_apo, ymap
        dl18_masked = bin_dl_18(ell, cl_masked)
        dl12_masked = bin_cl_log(ell, cl_masked)
        print(f"  masked done ({time.time() - t0:.0f}s)", flush=True)

        common = (
            f"L1_m9 {variant} lightcone-0 tSZ; HEALPix Nside=4096 pixel window deconvolved"
        )
        mask_note = (
            f"masked q>{Q_CUT:g} ({TAG}); r=max({R_MULT:g}*theta500, 2x{FWHM_ARCMIN:g}arcmin), "
            f"{APOTYPE} apodization {APOSIZE_DEG} deg, masked monopole subtracted, "
            f"NaMaster MASTER per-ell (nlb=1, lmax={LMAX})"
        )
        bin18_note = "uniform mean of D_ell over inclusive Planck bins"
        bin12_note = f"uniform mean of C_ell in log bins, Delta ln ell = {DLN_ELL}, ell_max = {LMAX}"
        cols = "ell_eff  1e12_D_ell_yy"

        write_bandpowers(
            OUT_DIR / f"Dl_yy_L1_m9_{variant}_fullsky_binned_18.txt",
            ELL_EFF,
            dl18_full * 1e12,
            f"{common}; full sky (anafast, mean subtracted); {bin18_note}\n{cols}",
        )
        write_bandpowers(
            OUT_DIR / f"Dl_yy_L1_m9_{variant}_fullsky_logbins_dln0p4_lmax10000.txt",
            LOG_CENTRES,
            dl12_full * 1e12,
            f"{common}; full sky (anafast, mean subtracted); {bin12_note}\n{cols}",
        )
        write_bandpowers(
            OUT_DIR / f"Dl_yy_L1_m9_{variant}_masked_{CUT_TAG}_{TAG}_binned_18.txt",
            ELL_EFF,
            dl18_masked * 1e12,
            f"{common}; {mask_note}; {bin18_note}\n{cols}",
        )
        write_bandpowers(
            OUT_DIR / f"Dl_yy_L1_m9_{variant}_masked_{CUT_TAG}_{TAG}_logbins_dln0p4_lmax10000.txt",
            LOG_CENTRES,
            dl12_masked * 1e12,
            f"{common}; {mask_note}; {bin12_note}\n{cols}",
        )

        entry = {
            "map": str(map_file(map_stem)),
            "catalogue": str(cat_file(cat_stem)),
            "n_halos": int(q.size),
            "n_masked": int(keep.sum()),
            "f_sky_raw": f_sky_raw,
            "f_sky_eff": f_sky_eff,
            "rotation_ratio_rot": r_rot["ratio"],
            "rotation_ratio_nat": r_nat["ratio"],
        }
        if variant == "fiducial":
            # Binning-consistency check against the already-published fiducial files.
            entry["check_vs_stored"] = {
                "fullsky_binned_18": max_frac_diff(
                    dl18_full * 1e12, np.loadtxt(REF_DIR / "Dl_yy_L1_m9_fullsky_binned_18.txt")[:, 1]
                ),
                "fullsky_logbins": max_frac_diff(
                    dl12_full * 1e12,
                    np.loadtxt(
                        REF_DIR
                        / "Dl_yy_L1_m9_fiducial_fullsky_logbins_dln0p4_lmax10000_pixwin_deconvolved.txt"
                    )[:, 1],
                ),
                "masked_qgt5_binned_18": max_frac_diff(
                    dl18_masked * 1e12,
                    np.loadtxt(REF_DIR / f"Dl_yy_L1_m9_masked_{CUT_TAG}_{TAG}_binned_18.txt")[:, 1],
                ),
                "masked_qgt5_logbins": max_frac_diff(
                    dl12_masked * 1e12,
                    np.loadtxt(
                        REF_DIR
                        / f"Dl_yy_L1_m9_masked_{CUT_TAG}_{TAG}_logbins_dln0p4_lmax10000.txt"
                    )[:, 1],
                ),
            }
            print(f"  vs stored fiducial: {entry['check_vs_stored']}", flush=True)
        metadata[variant] = entry

        with open(OUT_DIR / "L1_m9_feedback_bandpowers_metadata.json", "w") as handle:
            json.dump(
                {
                    "q_cut": Q_CUT,
                    "catalogue_suffix": "_yang26rot_qfrommz_alpha_fixed_1p12.csv",
                    "masking": {
                        "radius": f"max({R_MULT:g}*theta500, 2*FWHM), FWHM={FWHM_ARCMIN:g} arcmin",
                        "theta500": "R_500c/D_A(z), D3A cosmology",
                        "positions": "theta_rot_rad/phi_rot_rad (yang26-rotated frame)",
                        "apodization": f"{APOTYPE} {APOSIZE_DEG} deg",
                        "monopole": "mask-weighted, subtracted",
                        "estimator": f"NaMaster MASTER, nlb=1, lmax={LMAX}, no beam, pixwin deconvolved",
                    },
                    "fullsky_estimator": f"healpy.anafast, mean subtracted, lmax={LMAX}, pixwin deconvolved",
                    "binning_18": "uniform mean of D_ell over inclusive Planck bins",
                    "binning_12": f"uniform mean of C_ell, Delta ln ell = {DLN_ELL}, ell_max = {LMAX}",
                    "variants": metadata,
                    "runtime_seconds": time.time() - t0,
                },
                handle,
                indent=2,
            )
        print(f"  wrote {variant} ({time.time() - t0:.0f}s)", flush=True)

    print(f"\ndone ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
