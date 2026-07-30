"""Custom-GNFW best-fit theory covariance for L1_m9 binned bandpowers.

Evaluates the 18-bin / 12-log-bin Gaussian + one-halo-trispectrum covariance
with hmfast's ``ParametricGNFWPressureProfile`` at fixed ``alpha_SZ = 1.12``,
``B = 1.41``, and the best-fit ``A_SZ`` from
``chains/l1_m9_customgnfw_asz_alpha_fixed_1p12/best_fit.json``.

Products written:

* full sky (``f_sky = 1``);
* masked at each ``q`` in ``Q_CUTS`` (default 50, 20, 10, 5, 3, 1) using
  ``f_sky_eff`` from the fiducial paper masks and hmfast masked theory
  (``build_snr_grid`` + ``conditional_An_undetected``, ``sigma_lnY = 0.173``,
  ``n_power=2/1/4`` for 1h / 2h / trispectrum).

Run::

    python scripts/compute_l1_m9_customgnfw_bestfit_covariance.py
    ALLOW_CPU=1 python scripts/compute_l1_m9_customgnfw_bestfit_covariance.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Prefer GPU but allow CPU-only theory runs without pre-allocating all VRAM.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if os.environ.get("ALLOW_CPU") == "1":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
else:
    os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from hmfast.halos import HaloModel
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.profiles import ParametricGNFWPressureProfile
from hmfast.tracers import tSZTracer
from hmfast.tracers.tsz_completeness import (
    build_snr_grid,
    conditional_An_undetected,
    load_sigma_y0_curve,
)

from flamingo.catalogue.frame import D3A_COSMOLOGY
from flamingo.cnc import FILTER_NAME, SIGMA_Y0_FILE, SKYFRACS_FILE

from compute_l1_m9_simplegnfw_covariance import (
    ELL_EFF,
    ELL_SMOOTH,
    MASS_GRID,
    REDSHIFT_GRID,
    assemble_covariance,
    bin_dl,
    bin_trispectrum,
    gaussian_covariance,
    validate_covariance,
)

REPO = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO / "data_paper" / "covariance"
BEST_FIT_FILE = (
    REPO / "chains" / "l1_m9_customgnfw_asz_alpha_fixed_1p12" / "best_fit.json"
)
FEEDBACK_METADATA = (
    REPO
    / "data_paper"
    / "feedback_bandpower"
    / "L1_m9_feedback_bandpowers_metadata.json"
)
FEEDBACK_METADATA_QGT1 = (
    REPO
    / "data_paper"
    / "feedback_bandpower"
    / "L1_m9_feedback_bandpowers_qgt1_metadata.json"
)
FIDUCIAL_MASK_METADATA = (
    REPO
    / "data_paper"
    / "binned_bandpowers"
    / "L1_m9_masked_qfrommz_alpha_fixed_1p12_metadata.json"
)
MULTI_Q_METADATA = (
    REPO
    / "data_paper"
    / "binned_bandpowers"
    / "L1_m9_feedback_multi_q_bandpowers_metadata.json"
)

PROFILE_PARAMETERS = {
    "alpha_SZ": 1.12,
    "P0": 8.13,
    "c500": 1.156,
    "alpha": 1.062,
    "beta": 5.4807,
    "gamma": 0.3292,
    "B": 1.41,
}

# Match multi-q feedback ratio plot (strict → mild).
Q_CUTS = [50.0, 20.0, 10.0, 5.0, 3.0, 1.0]
SIGMA_LNY = 0.173


def cut_label(q: float) -> str:
    if float(q).is_integer():
        return f"masked_qgt{int(q)}"
    return f"masked_qgt{str(q).replace('.', 'p')}"


def load_f_sky_map() -> dict[str, float]:
    """Fiducial effective sky fraction per mask label (paper products)."""
    f_sky: dict[str, float] = {"fullsky": 1.0}
    if FIDUCIAL_MASK_METADATA.is_file():
        cuts = json.loads(FIDUCIAL_MASK_METADATA.read_text())["cuts"]
        for tag, entry in cuts.items():
            f_sky[f"masked_{tag}"] = float(entry["f_sky_eff"])
    # q>5 / q>1 from feedback metadata (same fiducial mask).
    if FEEDBACK_METADATA.is_file():
        f_sky["masked_qgt5"] = float(
            json.loads(FEEDBACK_METADATA.read_text())["variants"]["fiducial"][
                "f_sky_eff"
            ]
        )
    if FEEDBACK_METADATA_QGT1.is_file():
        f_sky["masked_qgt1"] = float(
            json.loads(FEEDBACK_METADATA_QGT1.read_text())["variants"]["fiducial"][
                "f_sky_eff"
            ]
        )
    # q>3 only in multi-q recompute metadata.
    if MULTI_Q_METADATA.is_file():
        multi = json.loads(MULTI_Q_METADATA.read_text())
        fid_cuts = multi.get("variants", {}).get("fiducial", {}).get("cuts", {})
        if "qgt3" in fid_cuts:
            f_sky["masked_qgt3"] = float(fid_cuts["qgt3"]["f_sky_eff"])
    return f_sky

LMAX_LOG = 10000
N_LOG_BINS = 12
DLN_ELL = 0.4
LOG_EDGES = LMAX_LOG * np.exp(-N_LOG_BINS * DLN_ELL) * np.exp(
    DLN_ELL * np.arange(N_LOG_BINS + 1)
)
LOG_CENTRES = np.sqrt(LOG_EDGES[:-1] * LOG_EDGES[1:])
ELL_SMOOTH_LOG = np.geomspace(9.0, float(LMAX_LOG), 80)
ELL_TRI_LOG = np.geomspace(9.0, float(LMAX_LOG), 40)


def _integer_ell_grid(edges: np.ndarray) -> np.ndarray:
    return np.arange(2, int(edges[-1]) + 1, dtype=float)


def bin_cl_log(ell: np.ndarray, cl: np.ndarray) -> np.ndarray:
    """Uniform mean of C_ell in log bins, D_ell at geometric centres."""
    ell = np.asarray(ell, dtype=float)
    cl = np.asarray(cl, dtype=float)
    ell_int = _integer_ell_grid(LOG_EDGES)
    cl_int = np.interp(np.log(ell_int), np.log(ell), cl)
    means = np.empty(N_LOG_BINS, dtype=float)
    for index, (lower, upper) in enumerate(zip(LOG_EDGES[:-1], LOG_EDGES[1:])):
        if index == N_LOG_BINS - 1:
            inside = (ell_int >= lower) & (ell_int <= upper)
        else:
            inside = (ell_int >= lower) & (ell_int < upper)
        means[index] = np.mean(cl_int[inside])
    return LOG_CENTRES * (LOG_CENTRES + 1.0) * means / (2.0 * np.pi)


def _bandpower_operator_log(ell_in: np.ndarray) -> np.ndarray:
    ell_in = np.asarray(ell_in, dtype=float)
    operator = np.empty((N_LOG_BINS, ell_in.size), dtype=float)
    for index in range(ell_in.size):
        basis = np.zeros(ell_in.size, dtype=float)
        basis[index] = 1.0
        operator[:, index] = bin_cl_log(ell_in, basis)
    return operator


def bin_trispectrum_log(ell: np.ndarray, trispectrum_cl: np.ndarray) -> np.ndarray:
    """Project a C_ell-basis trispectrum into log-bin D_ell space."""
    operator = _bandpower_operator_log(np.asarray(ell, dtype=float))
    return operator @ np.asarray(trispectrum_cl, dtype=float) @ operator.T


def gaussian_covariance_logbins(
    ell_cl: np.ndarray,
    cl: np.ndarray,
    f_sky: float,
) -> np.ndarray:
    """Gaussian bandpower covariance for the feedback log-bin convention."""
    ell_cl = np.asarray(ell_cl, dtype=float)
    cl = np.asarray(cl, dtype=float)
    ell_int = _integer_ell_grid(LOG_EDGES)
    cl_int = np.interp(np.log(ell_int), np.log(ell_cl), cl)
    covariance = np.zeros((N_LOG_BINS, N_LOG_BINS), dtype=float)
    bin_masks = []
    bin_counts = []
    for index, (lower, upper) in enumerate(zip(LOG_EDGES[:-1], LOG_EDGES[1:])):
        if index == N_LOG_BINS - 1:
            inside = (ell_int >= lower) & (ell_int <= upper)
        else:
            inside = (ell_int >= lower) & (ell_int < upper)
        bin_masks.append(inside)
        bin_counts.append(int(np.sum(inside)))
    for row in range(N_LOG_BINS):
        weight_row = (
            LOG_CENTRES[row]
            * (LOG_CENTRES[row] + 1.0)
            / (2.0 * np.pi * bin_counts[row])
        )
        for col in range(N_LOG_BINS):
            overlap = bin_masks[row] & bin_masks[col]
            if not np.any(overlap):
                continue
            ell = ell_int[overlap]
            weight_col = (
                LOG_CENTRES[col]
                * (LOG_CENTRES[col] + 1.0)
                / (2.0 * np.pi * bin_counts[col])
            )
            covariance[row, col] = np.sum(
                weight_row
                * weight_col
                * 2.0
                * cl_int[overlap] ** 2
                / ((2.0 * ell + 1.0) * f_sky)
            )
    return covariance


def validate_covariance_n(
    cov_gaussian: np.ndarray,
    trispectrum_binned: np.ndarray,
    cov_full: np.ndarray,
) -> dict[str, float]:
    """Validate shapes, finiteness, assembly, symmetry, and definiteness."""
    n_bin = cov_gaussian.shape[0]
    arrays = {
        "cov_gaussian": np.asarray(cov_gaussian, dtype=float),
        "trispectrum_binned": np.asarray(trispectrum_binned, dtype=float),
        "cov_full": np.asarray(cov_full, dtype=float),
    }
    for name, array in arrays.items():
        if array.shape != (n_bin, n_bin):
            raise ValueError(f"{name} has shape {array.shape}, expected ({n_bin}, {n_bin})")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite entries")
    expected = assemble_covariance(arrays["cov_gaussian"], arrays["trispectrum_binned"])
    component_residual = float(np.max(np.abs(arrays["cov_full"] - expected)))
    scale = float(np.max(np.abs(expected)))
    if component_residual > 1e-12 * max(scale, np.finfo(float).tiny):
        raise ValueError("full covariance does not equal Gaussian + T/(4 pi f_sky)")
    asymmetry = float(np.max(np.abs(arrays["cov_full"] - arrays["cov_full"].T)))
    if asymmetry > 1e-12 * max(scale, np.finfo(float).tiny):
        raise ValueError("full covariance is not symmetric")
    eigenvalues = np.linalg.eigvalsh(0.5 * (arrays["cov_full"] + arrays["cov_full"].T))
    min_eigenvalue = float(eigenvalues.min())
    if min_eigenvalue <= 0.0:
        raise ValueError("full covariance is not positive definite")
    return {
        "max_component_residual": component_residual,
        "max_asymmetry": asymmetry,
        "min_eigenvalue": min_eigenvalue,
        "max_eigenvalue": float(eigenvalues.max()),
        "condition_number": float(eigenvalues.max() / min_eigenvalue),
    }


def compute_fullsky(halo_model, tracer, ell, ell_tri, mass, redshift):
    """Raw full-sky 1h/2h Cl and one-halo trispectrum on the supplied ell grids."""
    cl_1h = np.asarray(halo_model.cl_1h(tracer, None, ell, mass, redshift))
    cl_2h = np.asarray(halo_model.cl_2h(tracer, None, ell, mass, redshift))
    cl_tri = np.asarray(
        halo_model.trispectrum_1h(tracer, None, ell_tri, ell_tri, mass, redshift)
    )
    return {
        "ell": np.asarray(ell, dtype=float),
        "ell_tri": np.asarray(ell_tri, dtype=float),
        "cl_1h": cl_1h,
        "cl_2h": cl_2h,
        "cl_tri": cl_tri,
    }


def compute_masked(halo_model, tracer, ell, ell_tri, mass, redshift, a_sz, *, q_cat: float):
    """Raw q-masked 1h/2h Cl and one-halo trispectrum on the supplied ell grids."""
    coeff, _ = load_sigma_y0_curve(
        sigma_obj_file=str(SIGMA_Y0_FILE),
        skyfr_file=str(SKYFRACS_FILE),
        filter_name=FILTER_NAME,
    )
    snr = build_snr_grid(
        halo_model,
        mass,
        redshift,
        a_sz,
        PROFILE_PARAMETERS["alpha_SZ"],
        PROFILE_PARAMETERS["B"],
        coeff=jnp.asarray(coeff),
    )
    masks = {
        n_power: conditional_An_undetected(
            snr,
            sigma_lnY=SIGMA_LNY,
            q_cat=q_cat,
            n_power=n_power,
            n_grid=512,
            nsig=8.0,
        )
        for n_power in (1, 2, 4)
    }
    cl_1h = np.asarray(
        halo_model.cl_1h_masked(tracer, None, ell, mass, redshift, masks[2], k_damp=0.0)
    )
    cl_2h = np.asarray(
        halo_model.cl_2h_masked(tracer, None, ell, mass, redshift, masks[1])
    )
    trispectrum_cl = np.asarray(
        halo_model.trispectrum_1h_masked(
            tracer, None, ell_tri, ell_tri, mass, redshift, masks[4], k_damp=0.0
        )
    )
    return {
        "ell": np.asarray(ell, dtype=float),
        "ell_tri": np.asarray(ell_tri, dtype=float),
        "cl_1h": cl_1h,
        "cl_2h": cl_2h,
        "cl_tri": trispectrum_cl,
    }


def bin_spectrum_18(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Bin raw theory spectra into the 18 Planck bins."""
    prefactor = raw["ell"] * (raw["ell"] + 1.0) / (2.0 * np.pi)
    dl_1h = bin_dl(raw["ell"], prefactor * raw["cl_1h"])
    dl_2h = bin_dl(raw["ell"], prefactor * raw["cl_2h"])
    return {
        "ell": ELL_EFF.copy(),
        "cl_total": raw["cl_1h"] + raw["cl_2h"],
        "dl_1h": dl_1h,
        "dl_2h": dl_2h,
        "dl_total": dl_1h + dl_2h,
        "trispectrum_binned": bin_trispectrum(raw["ell_tri"], raw["cl_tri"]),
    }


def bin_spectrum_log(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Bin raw theory spectra into the 12 feedback log bins."""
    return {
        "ell": LOG_CENTRES.copy(),
        "ell_cl": raw["ell"].copy(),
        "cl_total": raw["cl_1h"] + raw["cl_2h"],
        "dl_1h": bin_cl_log(raw["ell"], raw["cl_1h"]),
        "dl_2h": bin_cl_log(raw["ell"], raw["cl_2h"]),
        "dl_total": bin_cl_log(raw["ell"], raw["cl_1h"] + raw["cl_2h"]),
        "trispectrum_binned": bin_trispectrum_log(raw["ell_tri"], raw["cl_tri"]),
    }


def build_covariance_18(spectrum, f_sky):
    """Gaussian + trispectrum covariance and diagnostics at a sky fraction."""
    cov_gaussian = gaussian_covariance(spectrum["dl_total"], f_sky)
    cov_full = assemble_covariance(
        cov_gaussian, spectrum["trispectrum_binned"], f_sky
    )
    diagnostics = validate_covariance(
        cov_gaussian, spectrum["trispectrum_binned"] / f_sky, cov_full
    )
    return {
        "f_sky": f_sky,
        "cov_gaussian": cov_gaussian,
        "cov_full": cov_full,
        "diagnostics": diagnostics,
    }


def build_covariance_log(spectrum, f_sky):
    """Gaussian + trispectrum covariance for the 12 log bins."""
    cov_gaussian = gaussian_covariance_logbins(
        spectrum["ell_cl"], spectrum["cl_total"], f_sky
    )
    cov_full = assemble_covariance(
        cov_gaussian, spectrum["trispectrum_binned"], f_sky
    )
    diagnostics = validate_covariance_n(
        cov_gaussian, spectrum["trispectrum_binned"] / f_sky, cov_full
    )
    return {
        "f_sky": f_sky,
        "cov_gaussian": cov_gaussian,
        "cov_full": cov_full,
        "diagnostics": diagnostics,
    }


def main() -> None:
    """Run the covariance calculation and write artifacts for all q cuts."""
    started = time.perf_counter()
    devices = jax.devices()
    if (
        os.environ.get("ALLOW_CPU") != "1"
        and (not devices or devices[0].platform != "gpu")
    ):
        raise RuntimeError(f"CUDA device required, got {devices}")

    best_fit = json.loads(BEST_FIT_FILE.read_text())
    a_sz = float(best_fit["best_fit"]["A_SZ"])
    profile_parameters = {"A_SZ": a_sz, **PROFILE_PARAMETERS}
    f_sky_map = load_f_sky_map()
    for q in Q_CUTS:
        label = cut_label(q)
        if label not in f_sky_map:
            raise KeyError(f"missing f_sky_eff for {label}; have {sorted(f_sky_map)}")
        print(f"  {label}: f_sky_eff={f_sky_map[label]:.6f}", flush=True)

    halo_model = HaloModel(
        cosmology=D3A_COSMOLOGY,
        mass_definition=MassDefinition(500, "critical"),
        convert_masses=True,
        hm_consistency=False,
    )
    tracer = tSZTracer(profile=ParametricGNFWPressureProfile(**profile_parameters))
    mass = jnp.asarray(MASS_GRID)
    redshift = jnp.asarray(REDSHIFT_GRID)
    ell_18 = jnp.asarray(ELL_SMOOTH)
    ell_log = jnp.asarray(ELL_SMOOTH_LOG)
    ell_tri_log = jnp.asarray(ELL_TRI_LOG)

    raw_18: dict[str, dict[str, np.ndarray]] = {
        "fullsky": compute_fullsky(halo_model, tracer, ell_18, ell_18, mass, redshift)
    }
    raw_log: dict[str, dict[str, np.ndarray]] = {
        "fullsky": compute_fullsky(
            halo_model, tracer, ell_log, ell_tri_log, mass, redshift
        )
    }
    for q in Q_CUTS:
        label = cut_label(q)
        print(f"masked theory q>{q:g} ...", flush=True)
        raw_18[label] = compute_masked(
            halo_model, tracer, ell_18, ell_18, mass, redshift, a_sz, q_cat=float(q)
        )
        raw_log[label] = compute_masked(
            halo_model,
            tracer,
            ell_log,
            ell_tri_log,
            mass,
            redshift,
            a_sz,
            q_cat=float(q),
        )

    spectra_18 = {label: bin_spectrum_18(raw) for label, raw in raw_18.items()}
    spectra_log = {label: bin_spectrum_log(raw) for label, raw in raw_log.items()}
    cov_18 = {
        "fullsky": build_covariance_18(spectra_18["fullsky"], 1.0),
    }
    cov_log = {
        "fullsky": build_covariance_log(spectra_log["fullsky"], 1.0),
    }
    for q in Q_CUTS:
        label = cut_label(q)
        f_sky = f_sky_map[label]
        cov_18[label] = build_covariance_18(spectra_18[label], f_sky)
        cov_log[label] = build_covariance_log(spectra_log[label], f_sky)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, arrays in cov_18.items():
        np.save(
            OUTPUT_DIR / f"cov_full_L1_m9_customgnfw_bestfit_{label}_Dl_yy_binned_18.npy",
            arrays["cov_full"],
        )
    for label, arrays in cov_log.items():
        np.save(
            OUTPUT_DIR
            / f"cov_full_L1_m9_customgnfw_bestfit_{label}_Dl_yy_logbins_dln0p4_lmax10000.npy",
            arrays["cov_full"],
        )

    # Compact sigma table: ell + fullsky sigma + sigma per q cut (1e0 physical Dl).
    sigma_cols_18 = [np.sqrt(np.diag(cov_18["fullsky"]["cov_full"]))]
    sigma_cols_log = [np.sqrt(np.diag(cov_log["fullsky"]["cov_full"]))]
    sigma_header = ["sigma_fullsky"]
    for q in Q_CUTS:
        label = cut_label(q)
        sigma_cols_18.append(np.sqrt(np.diag(cov_18[label]["cov_full"])))
        sigma_cols_log.append(np.sqrt(np.diag(cov_log[label]["cov_full"])))
        sigma_header.append(f"sigma_{label}")

    # Keep legacy columns for qgt5/qgt1 theory spectra so existing plot scripts work.
    np.savetxt(
        OUTPUT_DIR / "Dl_yy_customgnfw_bestfit_theory_binned_18.txt",
        np.column_stack(
            [
                ELL_EFF,
                spectra_18["fullsky"]["dl_1h"],
                spectra_18["fullsky"]["dl_2h"],
                spectra_18["fullsky"]["dl_total"],
                spectra_18["masked_qgt5"]["dl_1h"],
                spectra_18["masked_qgt5"]["dl_2h"],
                spectra_18["masked_qgt5"]["dl_total"],
                np.sqrt(np.diag(cov_18["fullsky"]["cov_full"])),
                np.sqrt(np.diag(cov_18["masked_qgt5"]["cov_full"])),
                spectra_18["masked_qgt1"]["dl_1h"],
                spectra_18["masked_qgt1"]["dl_2h"],
                spectra_18["masked_qgt1"]["dl_total"],
                np.sqrt(np.diag(cov_18["masked_qgt1"]["cov_full"])),
            ]
        ),
        header=(
            "ell_eff  D_ell_1h  D_ell_2h  D_ell_total  "
            "D_ell_1h_masked_qgt5  D_ell_2h_masked_qgt5  D_ell_total_masked_qgt5  "
            "sigma_fullsky  sigma_masked_qgt5  "
            "D_ell_1h_masked_qgt1  D_ell_2h_masked_qgt1  D_ell_total_masked_qgt1  "
            "sigma_masked_qgt1"
        ),
        fmt="%.16e",
    )
    np.savetxt(
        OUTPUT_DIR / "Dl_yy_customgnfw_bestfit_theory_logbins_dln0p4_lmax10000.txt",
        np.column_stack(
            [
                LOG_CENTRES,
                spectra_log["fullsky"]["dl_1h"],
                spectra_log["fullsky"]["dl_2h"],
                spectra_log["fullsky"]["dl_total"],
                spectra_log["masked_qgt5"]["dl_1h"],
                spectra_log["masked_qgt5"]["dl_2h"],
                spectra_log["masked_qgt5"]["dl_total"],
                np.sqrt(np.diag(cov_log["fullsky"]["cov_full"])),
                np.sqrt(np.diag(cov_log["masked_qgt5"]["cov_full"])),
                spectra_log["masked_qgt1"]["dl_1h"],
                spectra_log["masked_qgt1"]["dl_2h"],
                spectra_log["masked_qgt1"]["dl_total"],
                np.sqrt(np.diag(cov_log["masked_qgt1"]["cov_full"])),
            ]
        ),
        header=(
            "ell_eff  D_ell_1h  D_ell_2h  D_ell_total  "
            "D_ell_1h_masked_qgt5  D_ell_2h_masked_qgt5  D_ell_total_masked_qgt5  "
            "sigma_fullsky  sigma_masked_qgt5  "
            "D_ell_1h_masked_qgt1  D_ell_2h_masked_qgt1  D_ell_total_masked_qgt1  "
            "sigma_masked_qgt1"
        ),
        fmt="%.16e",
    )
    # Multi-q sigma table (physical Dl, not *1e12).
    np.savetxt(
        OUTPUT_DIR / "Dl_yy_customgnfw_bestfit_sigma_all_q_binned_18.txt",
        np.column_stack([ELL_EFF, *sigma_cols_18]),
        header="ell_eff  " + "  ".join(sigma_header),
        fmt="%.16e",
    )
    np.savetxt(
        OUTPUT_DIR / "Dl_yy_customgnfw_bestfit_sigma_all_q_logbins_dln0p4_lmax10000.txt",
        np.column_stack([LOG_CENTRES, *sigma_cols_log]),
        header="ell_eff  " + "  ".join(sigma_header),
        fmt="%.16e",
    )

    metadata = {
        "profile_parameters": profile_parameters,
        "best_fit_file": str(BEST_FIT_FILE),
        "feedback_metadata_file": str(FEEDBACK_METADATA),
        "feedback_metadata_qgt1_file": str(FEEDBACK_METADATA_QGT1),
        "fiducial_mask_metadata_file": str(FIDUCIAL_MASK_METADATA),
        "q_cuts": list(Q_CUTS),
        "f_sky": {label: float(f_sky_map[label]) for label in ["fullsky"] + [cut_label(q) for q in Q_CUTS]},
        "masked_theory": {
            "sigma_lnY": SIGMA_LNY,
            "snr_grid": "build_snr_grid(A_SZ, alpha_SZ=1.12, B=1.41, immf6)",
            "q_cat": {cut_label(q): float(q) for q in Q_CUTS},
            "n_power": {
                "cl_1h_masked": 2,
                "cl_2h_masked": 1,
                "trispectrum_1h_masked": 4,
            },
            "sigma_y0_file": str(SIGMA_Y0_FILE),
            "skyfracs_file": str(SKYFRACS_FILE),
        },
        "diagnostics_binned_18": {
            label: arrays["diagnostics"] for label, arrays in cov_18.items()
        },
        "diagnostics_logbins": {
            label: arrays["diagnostics"] for label, arrays in cov_log.items()
        },
        "jax_devices": [str(device) for device in jax.devices()],
        "runtime_seconds": float(time.perf_counter() - started),
    }
    with (
        OUTPUT_DIR / "covariance_L1_m9_customgnfw_bestfit_metadata.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(metadata, indent=2, sort_keys=True))
    for label, arrays in cov_18.items():
        print(f"binned_18 {label} sigma:", np.sqrt(np.diag(arrays["cov_full"])))
    for label, arrays in cov_log.items():
        print(f"logbins {label} sigma:", np.sqrt(np.diag(arrays["cov_full"])))


if __name__ == "__main__":
    main()
