"""Cobaya components for the masked tSZ bandpowers with sampled cosmology.

The full-sky components in :mod:`flamingo.inference.l1_m9` hold the cosmology
fixed and fit only the pressure-profile amplitude. This module is the
cosmology-sampling counterpart used for the signal-only chains: it evaluates
the tSZ bandpowers **after removing the clusters a Planck-like survey would
detect**, and it includes both halo terms.

Both terms matter, and they are masked *differently*. The 1-halo term is
quadratic in the profile of a single halo, so with log-normal intrinsic scatter
its mask weight is the conditional second moment
:math:`\\langle A^2 \\mathbf{1}(q<q_{\\rm cat})\\rangle` (``n_power=2``). The
2-halo term is a product of two separate mass integrals sourced by *distinct*
halos, so the scatter expectation factorises and each bracket carries the
conditional *first* moment (``n_power=1``). Using one weight for both would be
wrong; see :meth:`hmfast.halos.HaloModel.cl_2h_masked`.

The selection is the same one that built the masks: the custom-GNFW best fit
``A_SZ``/``alpha_SZ`` at ``B = 1.41``, evaluated on an explicit ``M_500c``
halo model. Declaring the mass definition is essential -- hmfast defaults to
``M_200c`` and would otherwise re-convert the already-``M_500c`` masses,
driving ``q`` down by a factor ~0.46.
"""
from __future__ import annotations

import numpy as np
from cobaya.likelihood import Likelihood
from cobaya.theory import Theory

from .l1_m9 import ELL_SMOOTH, _bin_dl, gaussian_loglike, load_bandpower_likelihood

#: Halo-mass and redshift integration grids, matching the covariance products.
MASS_GRID = np.geomspace(1e11, 1e16, 64)
REDSHIFT_GRID = np.geomspace(0.005, 3.0, 96)

#: Fixed custom-GNFW shape (Arnaud-like) fitted to the L1_m9 full-sky spectrum.
GNFW_SHAPE = {
    "P0": 8.13,
    "c500": 1.156,
    "alpha": 1.062,
    "beta": 5.4807,
    "gamma": 0.3292,
}

#: Hydrostatic bias shared by the pressure profile and the selection.
B_HYDROSTATIC = 1.41

#: Scatter quadrature settings for the conditional moments.
SCATTER_GRID = 512
SCATTER_NSIG = 8.0

#: Reference cosmology used to seed the emulator before parameters are sampled.
_FIDUCIAL_LN1E10AS = 2.9718


class MaskedBandPowerLikelihood(Likelihood):
    """Gaussian likelihood for the 18 masked (or full-sky) tSZ bandpowers."""

    data_file: str
    covariance_file: str
    data_scale: float = 1e-12

    def initialize(self) -> None:
        self.ell, self.observed, self.covariance, self.inverse_covariance = (
            load_bandpower_likelihood(
                self.data_file, self.covariance_file, data_scale=self.data_scale
            )
        )
        if self.observed.shape != (18,):
            raise ValueError(f"expected 18 bandpowers, got {self.observed.size}")
        super().initialize()

    def get_requirements(self) -> dict:
        return {"Cl_sz": {}}

    def logp(self, **params_values) -> float:
        theory = self.provider.get_Cl_sz()
        total = np.asarray(theory["1h"], dtype=float) + np.asarray(
            theory["2h"], dtype=float
        )
        if not np.all(np.isfinite(total)):
            return -np.inf
        return gaussian_loglike(self.observed, total, self.inverse_covariance)


class MaskedTSZTheory(Theory):
    """Masked 1h+2h tSZ bandpowers with a sampled cosmology.

    Set ``q_cat`` to the detection threshold that was masked in the data, or
    leave it ``None`` for the unmasked full-sky spectrum.
    """

    output = ["Cl_sz"]

    #: ``None`` -> full sky; otherwise the catalogue threshold that was masked.
    q_cat: float | None = None

    params = {
        "H0": None,
        "omega_cdm": None,
        "omega_b": None,
        "n_s": None,
        "sigma_8": None,
        "tau_reio": None,
        "A_SZ": None,
        "alpha_SZ": None,
        "sigma_lnY": None,
    }

    def get_requirements(self) -> dict:
        return {name: None for name in self.params}

    def initialize(self) -> None:
        import jax
        import jax.numpy as jnp
        from hmfast.cosmology import Cosmology
        from hmfast.halos import HaloModel
        from hmfast.halos.mass_definition import MassDefinition
        from hmfast.halos.profiles import ParametricGNFWPressureProfile
        from hmfast.tracers import tSZTracer
        from hmfast.tracers.tsz_completeness import load_sigma_y0_curve

        from ..cnc import FILTER_NAME, SIGMA_Y0_FILE, SKYFRACS_FILE

        jax.config.update("jax_enable_x64", True)
        self._jnp = jnp
        self._profile_cls = ParametricGNFWPressureProfile
        self._tracer_cls = tSZTracer

        self._cosmology_seed = Cosmology(emulator_set="lcdm:v1")
        # Touch the emulator once so the first likelihood call is not charged
        # with the load.
        self._cosmology_seed.sigma8(0.0)

        self._halo_model_seed = HaloModel(
            cosmology=self._cosmology_seed,
            mass_definition=MassDefinition(500, "critical"),
            convert_masses=True,
            hm_consistency=False,
        )
        self._mass = jnp.asarray(MASS_GRID)
        self._redshift = jnp.asarray(REDSHIFT_GRID)
        self._ell = jnp.asarray(ELL_SMOOTH)

        coefficients, _ = load_sigma_y0_curve(
            sigma_obj_file=str(SIGMA_Y0_FILE),
            skyfr_file=str(SKYFRACS_FILE),
            filter_name=FILTER_NAME,
        )
        self._noise_coeff = jnp.asarray(coefficients)
        self._current_state: dict = {}
        super().initialize()

    def _cosmology(self, H0, omega_cdm, omega_b, n_s, tau_reio, sigma_8):
        """Update the seed cosmology, solving ``A_s`` for the sampled ``sigma_8``."""
        jnp = self._jnp
        ln_as = _FIDUCIAL_LN1E10AS
        for _ in range(3):
            cosmology = self._cosmology_seed.update(
                H0=H0,
                omega_cdm=omega_cdm,
                omega_b=omega_b,
                ln1e10A_s=ln_as,
                n_s=n_s,
                tau_reio=tau_reio,
            )
            current = float(cosmology.sigma8(jnp.asarray(0.0)))
            ln_as = ln_as + 2.0 * np.log(sigma_8 / current)
        return self._cosmology_seed.update(
            H0=H0,
            omega_cdm=omega_cdm,
            omega_b=omega_b,
            ln1e10A_s=ln_as,
            n_s=n_s,
            tau_reio=tau_reio,
        )

    def evaluate_bandpowers(self, **values) -> dict[str, np.ndarray]:
        """Return the binned 1-halo and 2-halo masked (or full-sky) ``D_ell``."""
        from hmfast.tracers.tsz_completeness import (
            build_snr_grid,
            conditional_An_undetected,
        )

        jnp = self._jnp
        cosmology = self._cosmology(
            values["H0"],
            values["omega_cdm"],
            values["omega_b"],
            values["n_s"],
            values["tau_reio"],
            values["sigma_8"],
        )
        halo_model = self._halo_model_seed.update(cosmology=cosmology)
        profile = self._profile_cls(
            A_SZ=values["A_SZ"],
            alpha_SZ=values["alpha_SZ"],
            B=B_HYDROSTATIC,
            **GNFW_SHAPE,
        )
        tracer = self._tracer_cls(profile=profile)

        if self.q_cat is None:
            cl_1h = halo_model.cl_1h(
                tracer, None, self._ell, self._mass, self._redshift
            )
            cl_2h = halo_model.cl_2h(
                tracer, None, self._ell, self._mass, self._redshift
            )
        else:
            snr = build_snr_grid(
                halo_model,
                self._mass,
                self._redshift,
                values["A_SZ"],
                values["alpha_SZ"],
                B_HYDROSTATIC,
                coeff=self._noise_coeff,
            )
            masks = {
                n: conditional_An_undetected(
                    snr,
                    sigma_lnY=values["sigma_lnY"],
                    q_cat=float(self.q_cat),
                    n_power=n,
                    n_grid=SCATTER_GRID,
                    nsig=SCATTER_NSIG,
                )
                for n in (1, 2)
            }
            # n_power=2 for the quadratic 1-halo term, n_power=1 for each of the
            # two linear 2-halo brackets.
            cl_1h = halo_model.cl_1h_masked(
                tracer, None, self._ell, self._mass, self._redshift, masks[2],
                k_damp=0.0,
            )
            cl_2h = halo_model.cl_2h_masked(
                tracer, None, self._ell, self._mass, self._redshift, masks[1]
            )

        ell = np.asarray(ELL_SMOOTH, dtype=float)
        prefactor = ell * (ell + 1.0) / (2.0 * np.pi)
        return {
            "1h": _bin_dl(ell, prefactor * np.asarray(cl_1h)),
            "2h": _bin_dl(ell, prefactor * np.asarray(cl_2h)),
        }

    def calculate(self, state: dict, want_derived: bool = True, **params_values) -> None:
        state["Cl_sz"] = self.evaluate_bandpowers(
            **{name: float(params_values[name]) for name in self.params}
        )
        self._current_state = state

    def get_Cl_sz(self) -> dict[str, np.ndarray] | None:
        return self._current_state.get("Cl_sz")
