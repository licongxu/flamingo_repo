"""Canonical L1_m9 bandpower binning and covariance operations."""
from __future__ import annotations

import numpy as np

from flamingo.powerspectra.bandpowers import (
    PLANCK_ELL_EFF as ELL_EFF,
    PLANCK_ELL_MAX as ELL_MAX,
    PLANCK_ELL_MIN as ELL_MIN,
)


def _bin_edges(
    ell_min: np.ndarray,
    ell_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ell_min = np.asarray(ell_min, dtype=int)
    ell_max = np.asarray(ell_max, dtype=int)
    if ell_min.shape != ell_max.shape:
        raise ValueError("ell_min and ell_max must have the same shape")
    if ell_min.ndim != 1 or np.any(ell_min > ell_max):
        raise ValueError("bin edges must be one-dimensional with ell_min <= ell_max")
    return ell_min, ell_max


def bin_dl_uniform(
    ell: np.ndarray,
    dl: np.ndarray,
    *,
    ell_min: np.ndarray = ELL_MIN,
    ell_max: np.ndarray = ELL_MAX,
) -> np.ndarray:
    """Log-interpolate positive ``D_ell`` and average inclusive integer bins."""
    ell = np.asarray(ell, dtype=float)
    dl = np.asarray(dl, dtype=float)
    ell_min, ell_max = _bin_edges(ell_min, ell_max)
    if ell.ndim != 1 or dl.shape != ell.shape:
        raise ValueError("ell and dl must be one-dimensional arrays of equal length")
    if np.any(ell <= 0.0) or np.any(dl <= 0.0):
        raise ValueError("ell and dl must be strictly positive for log interpolation")

    result = np.empty(ell_min.size, dtype=float)
    for index, (lo, hi) in enumerate(zip(ell_min, ell_max, strict=True)):
        integer_ell = np.arange(lo, hi + 1, dtype=float)
        sampled = np.exp(np.interp(np.log(integer_ell), np.log(ell), np.log(dl)))
        result[index] = np.mean(sampled)
    return result


def gaussian_bandpower_covariance(
    ell_cl: np.ndarray,
    cl: np.ndarray,
    f_sky: float,
    *,
    ell_min: np.ndarray = ELL_MIN,
    ell_max: np.ndarray = ELL_MAX,
) -> np.ndarray:
    """Return the exact Gaussian covariance, including shared bin edges."""
    ell_cl = np.asarray(ell_cl, dtype=float)
    cl = np.asarray(cl, dtype=float)
    ell_min, ell_max = _bin_edges(ell_min, ell_max)
    if f_sky <= 0.0:
        raise ValueError("f_sky must be positive")

    covariance = np.zeros((ell_min.size, ell_min.size), dtype=float)
    for row, (lo_row, hi_row) in enumerate(zip(ell_min, ell_max, strict=True)):
        ell_row = np.arange(lo_row, hi_row + 1, dtype=float)
        weight_row = ell_row * (ell_row + 1.0) / (2.0 * np.pi * ell_row.size)
        for col, (lo_col, hi_col) in enumerate(zip(ell_min, ell_max, strict=True)):
            lo, hi = max(lo_row, lo_col), min(hi_row, hi_col)
            if lo > hi:
                continue
            shared = np.arange(lo, hi + 1, dtype=float)
            cl_shared = np.interp(np.log(shared), np.log(ell_cl), cl)
            ell_col = np.arange(lo_col, hi_col + 1, dtype=float)
            weight_col = ell_col * (ell_col + 1.0) / (
                2.0 * np.pi * ell_col.size
            )
            covariance[row, col] = np.sum(
                weight_row[(shared - lo_row).astype(int)]
                * weight_col[(shared - lo_col).astype(int)]
                * 2.0
                * cl_shared**2
                / ((2.0 * shared + 1.0) * f_sky)
            )
    return covariance


def _bandpower_operator(
    ell: np.ndarray,
    ell_min: np.ndarray,
    ell_max: np.ndarray,
) -> np.ndarray:
    operator = np.empty((ell_min.size, ell.size), dtype=float)
    log_ell = np.log(ell)
    for column in range(ell.size):
        basis = np.zeros(ell.size, dtype=float)
        basis[column] = 1.0
        for row, (lo, hi) in enumerate(zip(ell_min, ell_max, strict=True)):
            integer_ell = np.arange(lo, hi + 1, dtype=float)
            weight = integer_ell * (integer_ell + 1.0) / (
                2.0 * np.pi * integer_ell.size
            )
            operator[row, column] = np.sum(
                weight * np.interp(np.log(integer_ell), log_ell, basis)
            )
    return operator


def trispectrum_bandpower_covariance(
    ell_tri: np.ndarray,
    trispectrum_cl: np.ndarray,
    f_sky: float,
    *,
    ell_min: np.ndarray = ELL_MIN,
    ell_max: np.ndarray = ELL_MAX,
) -> np.ndarray:
    """Project a C-ell-basis trispectrum into inclusive bandpower space."""
    ell_tri = np.asarray(ell_tri, dtype=float)
    trispectrum_cl = np.asarray(trispectrum_cl, dtype=float)
    ell_min, ell_max = _bin_edges(ell_min, ell_max)
    if trispectrum_cl.shape != (ell_tri.size, ell_tri.size):
        raise ValueError("trispectrum_cl shape must match ell_tri on both axes")
    if f_sky <= 0.0:
        raise ValueError("f_sky must be positive")
    operator = _bandpower_operator(ell_tri, ell_min, ell_max)
    return operator @ trispectrum_cl @ operator.T / (4.0 * np.pi * f_sky)


def validate_covariance(
    covariance_g: np.ndarray,
    covariance_t: np.ndarray,
    covariance: np.ndarray,
) -> dict[str, float]:
    """Validate component assembly, symmetry, finiteness, and definiteness."""
    covariance_g = np.asarray(covariance_g, dtype=float)
    covariance_t = np.asarray(covariance_t, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance_g.shape != covariance_t.shape or covariance.shape != covariance_g.shape:
        raise ValueError("covariance components must have matching shapes")
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance components must be square matrices")
    if not all(np.all(np.isfinite(item)) for item in (covariance_g, covariance_t, covariance)):
        raise ValueError("covariance contains non-finite entries")

    expected = covariance_g + covariance_t
    scale = max(float(np.max(np.abs(expected))), np.finfo(float).tiny)
    component_residual = float(np.max(np.abs(covariance - expected)))
    if component_residual > 1e-12 * scale:
        raise ValueError("covariance does not equal its Gaussian and trispectrum components")
    asymmetry = float(np.max(np.abs(covariance - covariance.T)))
    if asymmetry > 1e-12 * scale:
        raise ValueError("covariance is not symmetric")
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    if eigenvalues[0] <= 0.0:
        raise ValueError("covariance is not positive definite")
    return {
        "max_component_residual": component_residual,
        "max_asymmetry": asymmetry,
        "min_eigenvalue": float(eigenvalues[0]),
        "max_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
    }
