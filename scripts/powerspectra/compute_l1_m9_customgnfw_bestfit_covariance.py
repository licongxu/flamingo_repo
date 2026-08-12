"""Exact 18-bin qfrommap covariances for the fiducial L1_m9 theory."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if os.environ.get("ALLOW_CPU") == "1":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
else:
    os.environ.setdefault("JAX_PLATFORMS", "cuda")

import jax
import jax.numpy as jnp
import numpy as np
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
from flamingo.inference.bandpowers import (
    ELL_EFF,
    bin_dl_uniform,
    gaussian_bandpower_covariance,
    trispectrum_bandpower_covariance,
    validate_covariance,
)

jax.config.update("jax_enable_x64", True)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO / "data_paper" / "covariance" / "l1_m9_qfrommap"

CASES = ("fullsky", "qgt50", "qgt20", "qgt10", "qgt5")
Q_CUTS = {"qgt50": 50.0, "qgt20": 20.0, "qgt10": 10.0, "qgt5": 5.0}
QFROMMAP_F_SKY = {
    "fullsky": 1.0,
    "qgt50": 0.9972647840959187,
    "qgt20": 0.9858059150434103,
    "qgt10": 0.9435688947539681,
    "qgt5": 0.8595492784996496,
}

MASS_GRID = np.geomspace(1e11, 1e16, 64)
REDSHIFT_GRID = np.geomspace(0.005, 3.0, 96)
ELL_SMOOTH = np.geomspace(9.0, 1085.0, 50)
SIGMA_LNY = 0.173
SCATTER_GRID = 512
SCATTER_NSIG = 8.0
PROFILE_PARAMETERS = {
    "alpha_SZ": 1.12,
    "P0": 8.13,
    "c500": 1.156,
    "alpha": 1.062,
    "beta": 5.4807,
    "gamma": 0.3292,
    "B": 1.41,
}


def data_file(case: str) -> Path:
    """Return the observed 18-bin spectrum associated with a covariance case."""
    directory = REPO / "data_paper" / "binned_bandpowers"
    if case == "fullsky":
        return directory / "Dl_yy_L1_m9_fullsky_binned_18.txt"
    if case not in Q_CUTS:
        raise ValueError(f"unknown case {case!r}; expected one of {CASES}")
    return directory / f"Dl_yy_L1_m9_masked_{case}_qfrommap_binned_18.txt"


def build_covariance_18(
    raw: dict[str, np.ndarray],
    f_sky: float,
) -> dict[str, np.ndarray | dict[str, float] | float]:
    """Bin raw hmfast spectra and assemble exact synthetic-style covariance."""
    ell = np.asarray(raw["ell"], dtype=float)
    ell_tri = np.asarray(raw["ell_tri"], dtype=float)
    cl_1h = np.asarray(raw["cl_1h"], dtype=float)
    cl_2h = np.asarray(raw["cl_2h"], dtype=float)
    cl_total = cl_1h + cl_2h
    prefactor = ell * (ell + 1.0) / (2.0 * np.pi)
    dl_1h = bin_dl_uniform(ell, prefactor * cl_1h)
    dl_2h = bin_dl_uniform(ell, prefactor * cl_2h)
    covariance_g = gaussian_bandpower_covariance(ell, cl_total, f_sky)
    covariance_t = trispectrum_bandpower_covariance(
        ell_tri,
        np.asarray(raw["cl_tri"], dtype=float),
        f_sky,
    )
    covariance = covariance_g + covariance_t
    diagnostics = validate_covariance(covariance_g, covariance_t, covariance)
    return {
        "f_sky": float(f_sky),
        "ell": ELL_EFF.copy(),
        "dl_1h": dl_1h,
        "dl_2h": dl_2h,
        "dl_total": dl_1h + dl_2h,
        "cov_gaussian": covariance_g,
        "cov_trispectrum": covariance_t,
        "cov_full": covariance,
        "diagnostics": diagnostics,
    }


def _theory_context(a_sz: float):
    halo_model = HaloModel(
        cosmology=D3A_COSMOLOGY,
        mass_definition=MassDefinition(500, "critical"),
        convert_masses=True,
        hm_consistency=False,
    )
    profile = ParametricGNFWPressureProfile(A_SZ=a_sz, **PROFILE_PARAMETERS)
    tracer = tSZTracer(profile=profile)
    coefficients, _ = load_sigma_y0_curve(
        sigma_obj_file=str(SIGMA_Y0_FILE),
        skyfr_file=str(SKYFRACS_FILE),
        filter_name=FILTER_NAME,
    )
    return halo_model, tracer, jnp.asarray(coefficients)


def _raw_theory(
    halo_model: HaloModel,
    tracer: tSZTracer,
    noise_coeff: jax.Array,
    a_sz: float,
    q_cat: float,
) -> dict[str, np.ndarray]:
    ell = jnp.asarray(ELL_SMOOTH)
    mass = jnp.asarray(MASS_GRID)
    redshift = jnp.asarray(REDSHIFT_GRID)
    snr = build_snr_grid(
        halo_model,
        mass,
        redshift,
        a_sz,
        PROFILE_PARAMETERS["alpha_SZ"],
        PROFILE_PARAMETERS["B"],
        coeff=noise_coeff,
    )
    moments = {
        power: conditional_An_undetected(
            snr,
            sigma_lnY=SIGMA_LNY,
            q_cat=q_cat,
            n_power=power,
            n_grid=SCATTER_GRID,
            nsig=SCATTER_NSIG,
        )
        for power in (1, 2, 4)
    }
    cl_1h = halo_model.cl_1h_masked(
        tracer,
        None,
        ell,
        mass,
        redshift,
        moments[2],
        k_damp=0.0,
    )
    cl_2h = halo_model.cl_2h_masked(
        tracer,
        None,
        ell,
        mass,
        redshift,
        moments[1],
    )
    cl_tri = halo_model.trispectrum_1h_masked(
        tracer,
        None,
        ell,
        ell,
        mass,
        redshift,
        moments[4],
        k_damp=0.0,
    )
    return {
        "ell": np.asarray(ELL_SMOOTH),
        "ell_tri": np.asarray(ELL_SMOOTH),
        "cl_1h": np.asarray(cl_1h),
        "cl_2h": np.asarray(cl_2h),
        "cl_tri": np.asarray(cl_tri),
    }


def _artifact_paths(output_dir: Path, case: str) -> dict[str, Path]:
    stem = f"L1_m9_customgnfw_qfrommap_{case}_Dl_yy_binned_18"
    return {
        "cov_gaussian": output_dir / f"cov_gaussian_{stem}.npy",
        "cov_trispectrum": output_dir / f"cov_trispectrum_{stem}.npy",
        "cov_full": output_dir / f"cov_full_{stem}.npy",
        "theory": output_dir / f"theory_{stem}.txt",
    }


def compute_covariances(
    a_sz: float,
    output_dir: Path,
    cases: tuple[str, ...] = CASES,
) -> dict:
    """Compute and save exact covariance components at an explicit ``A_SZ``."""
    unknown = set(cases) - set(CASES)
    if unknown:
        raise ValueError(f"unknown cases: {sorted(unknown)}")
    if len(cases) != len(set(cases)):
        raise ValueError("cases must not contain duplicates")
    devices = jax.devices()
    if os.environ.get("ALLOW_CPU") != "1" and (
        not devices or devices[0].platform != "gpu"
    ):
        raise RuntimeError(f"CUDA device required, got {devices}")

    started = time.perf_counter()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    halo_model, tracer, noise_coeff = _theory_context(float(a_sz))
    case_results: dict[str, dict] = {}
    for case in cases:
        q_cat = np.inf if case == "fullsky" else Q_CUTS[case]
        raw = _raw_theory(halo_model, tracer, noise_coeff, float(a_sz), q_cat)
        result = build_covariance_18(raw, QFROMMAP_F_SKY[case])
        paths = _artifact_paths(output_dir, case)
        for component in ("cov_gaussian", "cov_trispectrum", "cov_full"):
            np.save(paths[component], result[component])
        np.savetxt(
            paths["theory"],
            np.column_stack(
                [result["ell"], result["dl_1h"], result["dl_2h"], result["dl_total"]]
            ),
            header="ell_eff D_ell_1h D_ell_2h D_ell_total",
            fmt="%.16e",
        )
        case_results[case] = {
            "q_cat": None if np.isinf(q_cat) else float(q_cat),
            "q_cat_theory": "infinity" if np.isinf(q_cat) else float(q_cat),
            "f_sky": QFROMMAP_F_SKY[case],
            "data_file": str(data_file(case).resolve()),
            "paths": {name: str(path) for name, path in paths.items()},
            "diagnostics": result["diagnostics"],
        }

    metadata = {
        "A_SZ": float(a_sz),
        "profile_parameters": {"A_SZ": float(a_sz), **PROFILE_PARAMETERS},
        "sigma_lnY": SIGMA_LNY,
        "mass_units": "M_sun",
        "binning": {
            "count": 18,
            "edges": "inclusive at both ends",
            "interpolation": "log-log D_ell to integer ell; uniform D_ell mean",
        },
        "covariance": {
            "gaussian": "shared-edge full Gaussian bandpower covariance",
            "trispectrum": "W4 one-halo trispectrum divided by 4*pi*f_sky",
            "assembly": "cov_full = cov_gaussian + cov_trispectrum",
        },
        "scatter_moments": {
            "cl_1h": 2,
            "cl_2h_each_bracket": 1,
            "trispectrum_1h": 4,
            "fullsky_q_cat": "infinity",
        },
        "qfrommap_f_sky": QFROMMAP_F_SKY,
        "cases": case_results,
        "noise_files": {
            "sigma_y0": str(SIGMA_Y0_FILE),
            "sky_fractions": str(SKYFRACS_FILE),
            "filter": FILTER_NAME,
        },
        "jax_devices": [str(device) for device in devices],
        "runtime_seconds": float(time.perf_counter() - started),
    }
    metadata_path = output_dir / "covariance_L1_m9_customgnfw_qfrommap_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return {"metadata_path": str(metadata_path), **metadata}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-sz", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--case",
        action="append",
        choices=CASES,
        dest="cases",
        help="case to compute; repeat for multiple cases (default: all)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    cases = CASES if args.cases is None else tuple(args.cases)
    result = compute_covariances(args.a_sz, args.output_dir, cases)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
