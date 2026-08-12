"""Full-sky L1_m9 theory covariance from simple GNFW hmfast theory."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from hmfast.halos import HaloModel
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.profiles import GNFWPressureProfile
from hmfast.tracers import tSZTracer

from flamingo.catalogue.frame import D3A_COSMOLOGY
from flamingo.powerspectra.bandpowers import (
    PLANCK_ELL_EFF as ELL_EFF,
    PLANCK_ELL_MAX as ELL_MAX,
    PLANCK_ELL_MIN as ELL_MIN,
)


REPO = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO / "data_paper" / "covariance"
F_SKY = 1.0
MASS_GRID = np.geomspace(1e11, 1e16, 64)
REDSHIFT_GRID = np.geomspace(0.005, 3.0, 96)
ELL_SMOOTH = np.geomspace(9.0, 1085.0, 50)
PROFILE_PARAMETERS = {
    "P0": 8.13,
    "c500": 1.156,
    "alpha": 1.062,
    "beta": 5.4807,
    "gamma": 0.3292,
    "B": 1.0,
}

_BIN_WIDTH_MAX = int(np.max(ELL_MAX - ELL_MIN))
_ELL_INTEGER = ELL_MIN[:, None] + np.arange(_BIN_WIDTH_MAX)[None, :]
_ELL_MASK = _ELL_INTEGER < ELL_MAX[:, None]


def bin_dl(ell: np.ndarray, dl: np.ndarray) -> np.ndarray:
    """Uniformly average a smooth D_ell curve over the 18 integer-ell bins."""
    ell = np.asarray(ell, dtype=float)
    dl = np.asarray(dl, dtype=float)
    sampled = np.empty(_ELL_INTEGER.shape, dtype=float)
    for index in range(18):
        sampled[index] = np.interp(
            np.log(_ELL_INTEGER[index]),
            np.log(ell),
            dl,
        )
    return np.sum(sampled * _ELL_MASK, axis=1) / np.sum(_ELL_MASK, axis=1)


def bin_trispectrum(
    ell: np.ndarray,
    trispectrum_cl: np.ndarray,
) -> np.ndarray:
    """Convert T_ell,ell' from C units and average over both bin axes."""
    ell = np.asarray(ell, dtype=float)
    trispectrum_cl = np.asarray(trispectrum_cl, dtype=float)
    log_ell = np.log(ell)
    log_integer = np.log(_ELL_INTEGER.astype(float))

    first_axis = np.empty((*_ELL_INTEGER.shape, ell.size), dtype=float)
    for column in range(ell.size):
        first_axis[..., column] = np.interp(
            log_integer, log_ell, trispectrum_cl[:, column]
        )

    interpolated = np.empty((*_ELL_INTEGER.shape, *_ELL_INTEGER.shape))
    for first_bin in range(18):
        for first_ell in range(_BIN_WIDTH_MAX):
            interpolated[first_bin, first_ell] = np.interp(
                log_integer,
                log_ell,
                first_axis[first_bin, first_ell],
            )

    dl_factor = _ELL_INTEGER * (_ELL_INTEGER + 1.0) / (2.0 * np.pi)
    trispectrum_dl = (
        interpolated
        * dl_factor[:, :, None, None]
        * dl_factor[None, None, :, :]
    )
    pair_mask = _ELL_MASK[:, :, None, None] * _ELL_MASK[None, None, :, :]
    return np.sum(trispectrum_dl * pair_mask, axis=(1, 3)) / np.sum(
        pair_mask, axis=(1, 3)
    )


def gaussian_covariance(
    dl_binned: np.ndarray,
    fsky: float = 1.0,
) -> np.ndarray:
    """Return the diagonal binned Knox covariance."""
    dl_binned = np.asarray(dl_binned, dtype=float)
    diagonal = 2.0 * dl_binned**2 / (
        (2.0 * ELL_EFF + 1.0) * (ELL_MAX - ELL_MIN) * fsky
    )
    return np.diag(diagonal)


def assemble_covariance(
    cov_gaussian: np.ndarray,
    trispectrum_binned: np.ndarray,
    fsky: float = 1.0,
) -> np.ndarray:
    """Combine Gaussian and connected one-halo terms."""
    return (
        np.asarray(cov_gaussian)
        + np.asarray(trispectrum_binned) / (4.0 * np.pi * fsky)
    )


def validate_covariance(
    cov_gaussian: np.ndarray,
    trispectrum_binned: np.ndarray,
    cov_full: np.ndarray,
) -> dict[str, float]:
    """Validate shapes, finiteness, assembly, symmetry, and definiteness."""
    arrays = {
        "cov_gaussian": np.asarray(cov_gaussian, dtype=float),
        "trispectrum_binned": np.asarray(trispectrum_binned, dtype=float),
        "cov_full": np.asarray(cov_full, dtype=float),
    }
    for name, array in arrays.items():
        if array.shape != (18, 18):
            raise ValueError(f"{name} has shape {array.shape}, expected (18, 18)")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite entries")

    expected = assemble_covariance(
        arrays["cov_gaussian"], arrays["trispectrum_binned"]
    )
    component_residual = float(np.max(np.abs(arrays["cov_full"] - expected)))
    scale = float(np.max(np.abs(expected)))
    if component_residual > 1e-12 * max(scale, np.finfo(float).tiny):
        raise ValueError("full covariance does not equal Gaussian + T/(4 pi)")

    asymmetry = float(
        np.max(np.abs(arrays["cov_full"] - arrays["cov_full"].T))
    )
    if asymmetry > 1e-12 * max(scale, np.finfo(float).tiny):
        raise ValueError("full covariance is not symmetric")
    if np.any(np.diag(arrays["cov_gaussian"]) <= 0.0):
        raise ValueError("Gaussian covariance diagonal is not positive")
    if np.any(np.diag(arrays["trispectrum_binned"]) < 0.0):
        raise ValueError("trispectrum has a negative diagonal")

    eigenvalues = np.linalg.eigvalsh(
        0.5 * (arrays["cov_full"] + arrays["cov_full"].T)
    )
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


def compute_covariance() -> dict[str, np.ndarray]:
    """Evaluate simple-GNFW power and covariance components with hmfast."""
    devices = jax.devices()
    if not devices or devices[0].platform != "gpu":
        raise RuntimeError(f"CUDA device required, got {devices}")

    halo_model = HaloModel(
        cosmology=D3A_COSMOLOGY,
        mass_definition=MassDefinition(500, "critical"),
        convert_masses=True,
        hm_consistency=False,
    )
    profile = GNFWPressureProfile(**PROFILE_PARAMETERS)
    tracer = tSZTracer(profile=profile)
    ell = jnp.asarray(ELL_SMOOTH)
    mass = jnp.asarray(MASS_GRID)
    redshift = jnp.asarray(REDSHIFT_GRID)

    cl_1h = np.asarray(halo_model.cl_1h(tracer, None, ell, mass, redshift))
    cl_2h = np.asarray(halo_model.cl_2h(tracer, None, ell, mass, redshift))
    prefactor = ELL_SMOOTH * (ELL_SMOOTH + 1.0) / (2.0 * np.pi)
    dl_1h = bin_dl(ELL_SMOOTH, prefactor * cl_1h)
    dl_2h = bin_dl(ELL_SMOOTH, prefactor * cl_2h)
    dl_total = dl_1h + dl_2h

    trispectrum_cl = np.asarray(
        halo_model.trispectrum_1h(
            tracer,
            None,
            ell,
            ell,
            mass,
            redshift,
        )
    )
    trispectrum_binned = bin_trispectrum(ELL_SMOOTH, trispectrum_cl)
    cov_gaussian = gaussian_covariance(dl_total, F_SKY)
    cov_full = assemble_covariance(
        cov_gaussian,
        trispectrum_binned,
        F_SKY,
    )
    return {
        "dl_1h": dl_1h,
        "dl_2h": dl_2h,
        "dl_total": dl_total,
        "cov_gaussian": cov_gaussian,
        "trispectrum_binned": trispectrum_binned,
        "cov_full": cov_full,
    }


def write_outputs(
    result: dict[str, np.ndarray],
    runtime_seconds: float,
) -> dict:
    """Validate and save the theory bandpowers and covariance artifacts."""
    diagnostics = validate_covariance(
        result["cov_gaussian"],
        result["trispectrum_binned"],
        result["cov_full"],
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(
        OUTPUT_DIR / "cov_gaussian_L1_m9_fullsky_Dl_yy_binned_18.npy",
        result["cov_gaussian"],
    )
    np.save(
        OUTPUT_DIR / "trispectrum_L1_m9_fullsky_Dl_yy_binned_18.npy",
        result["trispectrum_binned"],
    )
    np.save(
        OUTPUT_DIR / "cov_full_L1_m9_fullsky_Dl_yy_binned_18.npy",
        result["cov_full"],
    )
    np.savetxt(
        OUTPUT_DIR / "cov_full_L1_m9_fullsky_Dl_yy_binned_18.csv",
        result["cov_full"],
        delimiter=",",
        fmt="%.16e",
    )
    np.savetxt(
        OUTPUT_DIR / "Dl_yy_simplegnfw_B1_theory_binned_18.txt",
        np.column_stack(
            [
                ELL_EFF,
                result["dl_1h"],
                result["dl_2h"],
                result["dl_total"],
            ]
        ),
        header="ell_eff  D_ell_1h  D_ell_2h  D_ell_total",
        fmt="%.16e",
    )

    cosmology = {
        key: float(getattr(D3A_COSMOLOGY, key))
        for key in (
            "H0",
            "omega_cdm",
            "omega_b",
            "ln1e10A_s",
            "n_s",
            "tau_reio",
            "m_ncdm",
        )
    }
    metadata = {
        "data_file": str(
            REPO
            / "data_paper/binned_bandpowers/Dl_yy_L1_m9_fullsky_binned_18.txt"
        ),
        "cosmology": cosmology,
        "pressure_profile": "hmfast.GNFWPressureProfile",
        "profile_parameters": PROFILE_PARAMETERS,
        "sigma_lnY": 0.0,
        "f_sky": F_SKY,
        "mass_units": "physical M_sun",
        "mass_grid": {
            "minimum": float(MASS_GRID[0]),
            "maximum": float(MASS_GRID[-1]),
            "count": int(MASS_GRID.size),
        },
        "redshift_grid": {
            "minimum": float(REDSHIFT_GRID[0]),
            "maximum": float(REDSHIFT_GRID[-1]),
            "count": int(REDSHIFT_GRID.size),
        },
        "ell_grid_count": int(ELL_SMOOTH.size),
        "ell_effective": ELL_EFF.tolist(),
        "gaussian_power_terms": ["1h", "2h"],
        "non_gaussian_term": "connected 1h trispectrum / (4 pi f_sky)",
        "jax_devices": [str(device) for device in jax.devices()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runtime_seconds": float(runtime_seconds),
        "diagnostics": diagnostics,
    }
    with (
        OUTPUT_DIR / "covariance_L1_m9_fullsky_metadata.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata


def main() -> None:
    """Run the covariance calculation, write artifacts, and print diagnostics."""
    started = time.perf_counter()
    result = compute_covariance()
    metadata = write_outputs(result, time.perf_counter() - started)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print("Gaussian sigma:", np.sqrt(np.diag(result["cov_gaussian"])))
    print("Full sigma:", np.sqrt(np.diag(result["cov_full"])))


if __name__ == "__main__":
    main()
