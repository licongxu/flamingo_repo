"""Fixed-cosmology Cobaya components for the fiducial L1_m9 bandpowers."""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from cobaya.likelihood import Likelihood
from cobaya.theory import Theory
from hmfast.halos import HaloModel
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.profiles import ParametricGNFWPressureProfile
from hmfast.tracers import tSZTracer

from ..catalogue.frame import D3A_COSMOLOGY
from .bandpowers import ELL_MAX, ELL_MIN, bin_dl_uniform

jax.config.update("jax_enable_x64", True)


ELL_SMOOTH = np.geomspace(9.0, 1085.0, 50)
MASS_GRID = np.geomspace(1e11, 1e16, 64)
REDSHIFT_GRID = np.geomspace(1e-6, 3.0, 96)


def _bin_dl(ell: np.ndarray, dl: np.ndarray) -> np.ndarray:
    """Uniformly average a smooth D_ell curve over the 18 integer-ell bins."""
    return bin_dl_uniform(ell, dl)


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


class L1M9CustomGNFWTheory(Theory):
    """Fixed-D3A custom-GNFW 1h+2h bandpowers for Cobaya."""

    output = ["Cl_sz"]
    params = {"A_SZ": 0, "alpha_SZ": 0}

    def get_requirements(self) -> dict:
        return {name: None for name in self.params}

    def initialize(self) -> None:
        self._halo_model = HaloModel(
            cosmology=D3A_COSMOLOGY,
            mass_definition=MassDefinition(500, "critical"),
            convert_masses=True,
            hm_consistency=False,
        )
        self._profile = ParametricGNFWPressureProfile(
            A_SZ=-4.1,
            alpha_SZ=1.12,
            P0=8.13,
            c500=1.156,
            alpha=1.062,
            beta=5.4807,
            gamma=0.3292,
            B=1.41,
        )
        self._tracer = tSZTracer(profile=self._profile)
        self._mass = jnp.asarray(MASS_GRID)
        self._redshift = jnp.asarray(REDSHIFT_GRID)
        self._evaluate_cl = jax.jit(self._evaluate_cl_impl)
        self._current_state = {}
        super().initialize()

    def _evaluate_cl_impl(
        self,
        A_SZ: float,
        alpha_SZ: float,
        ell: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        profile = self._profile.update(A_SZ=A_SZ, alpha_SZ=alpha_SZ)
        tracer = self._tracer.update(profile=profile)
        cl_1h = self._halo_model.cl_1h(
            tracer,
            None,
            ell,
            self._mass,
            self._redshift,
        )
        cl_2h = self._halo_model.cl_2h(
            tracer,
            None,
            ell,
            self._mass,
            self._redshift,
        )
        return cl_1h, cl_2h

    def evaluate_bandpowers(
        self,
        A_SZ: float,
        alpha_SZ: float,
    ) -> dict[str, np.ndarray]:
        """Evaluate and bin the custom-GNFW 1h and 2h spectra."""
        spectrum = self.evaluate_spectrum(A_SZ, alpha_SZ)
        return {
            "1h": _bin_dl(spectrum["ell"], spectrum["1h"]),
            "2h": _bin_dl(spectrum["ell"], spectrum["2h"]),
        }

    def evaluate_spectrum(
        self,
        A_SZ: float,
        alpha_SZ: float,
        ell: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Evaluate the custom-GNFW 1h, 2h, and total D_ell spectra."""
        ell = ELL_SMOOTH if ell is None else np.asarray(ell, dtype=float)
        cl_1h, cl_2h = self._evaluate_cl(
            float(A_SZ),
            float(alpha_SZ),
            jnp.asarray(ell),
        )
        prefactor = ell * (ell + 1.0) / (2.0 * np.pi)
        dl_1h = prefactor * np.asarray(cl_1h)
        dl_2h = prefactor * np.asarray(cl_2h)
        return {
            "ell": ell.copy(),
            "1h": dl_1h,
            "2h": dl_2h,
            "total": dl_1h + dl_2h,
        }

    def calculate(
        self,
        state: dict,
        want_derived: bool = True,
        **params_values,
    ) -> None:
        state["Cl_sz"] = self.evaluate_bandpowers(
            params_values["A_SZ"],
            params_values["alpha_SZ"],
        )
        self._current_state = state

    def get_Cl_sz(self) -> dict[str, np.ndarray] | None:
        return self._current_state.get("Cl_sz")
