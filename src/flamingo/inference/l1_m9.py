"""Fixed-cosmology Cobaya components for the fiducial L1_m9 bandpowers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from cobaya.likelihood import Likelihood


def load_bandpower_likelihood(
    data_file: str | Path,
    covariance_file: str | Path,
    *,
    data_scale: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a two-column bandpower vector and its positive-definite covariance."""
    data = np.loadtxt(data_file)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f"bandpower data must have two columns, got {data.shape}")
    ell = np.asarray(data[:, 0], dtype=float)
    observed = np.asarray(data[:, 1], dtype=float) * float(data_scale)
    covariance = np.asarray(np.load(covariance_file), dtype=float)
    expected_shape = (observed.size, observed.size)
    if covariance.shape != expected_shape:
        raise ValueError(
            f"covariance has shape {covariance.shape}, expected {expected_shape}"
        )
    for name, array in (
        ("ell", ell),
        ("observed", observed),
        ("covariance", covariance),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=0.0):
        raise ValueError("covariance is not symmetric")
    np.linalg.cholesky(covariance)
    inverse = np.linalg.inv(covariance)
    return ell, observed, covariance, inverse


def gaussian_loglike(
    observed: np.ndarray,
    theory: np.ndarray,
    inverse_covariance: np.ndarray,
) -> float:
    """Return the unnormalized Gaussian log likelihood."""
    observed = np.asarray(observed, dtype=float)
    theory = np.asarray(theory, dtype=float)
    inverse_covariance = np.asarray(inverse_covariance, dtype=float)
    if theory.shape != observed.shape:
        raise ValueError(
            f"theory has shape {theory.shape}, expected {observed.shape}"
        )
    if inverse_covariance.shape != (observed.size, observed.size):
        raise ValueError("inverse covariance shape does not match the data")
    if not np.all(np.isfinite(theory)):
        raise ValueError("theory contains non-finite values")
    residual = observed - theory
    return float(-0.5 * residual @ inverse_covariance @ residual)


class L1M9BandPowerLikelihood(Likelihood):
    """Gaussian likelihood for all 18 fiducial L1_m9 full-sky bandpowers."""

    data_file: str
    covariance_file: str
    data_scale: float = 1e-12

    def initialize(self) -> None:
        self.ell, self.observed, self.covariance, self.inverse_covariance = (
            load_bandpower_likelihood(
                self.data_file,
                self.covariance_file,
                data_scale=self.data_scale,
            )
        )
        if self.observed.shape != (18,):
            raise ValueError(
                f"L1_m9 likelihood requires 18 bins, got {self.observed.size}"
            )
        super().initialize()

    def get_requirements(self) -> dict:
        return {"Cl_sz": {}}

    def logp(self, **params_values) -> float:
        theory = self.provider.get_Cl_sz()
        dl_1h = np.asarray(theory["1h"], dtype=float)
        dl_2h = np.asarray(theory["2h"], dtype=float)
        return gaussian_loglike(
            self.observed,
            dl_1h + dl_2h,
            self.inverse_covariance,
        )
