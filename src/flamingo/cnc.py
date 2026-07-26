"""Planck-like cluster detection signal-to-noise ``q`` from the SZ scaling relation.

The cluster-number-count (CNC) selection function used throughout the FLAMINGO
analysis is defined on the *matched-filter* signal-to-noise

.. math::

    q = \\frac{y_0(M_{500c}, z)\\, e^{\\epsilon}}{\\sigma_{y_0}(\\theta_{500})},
    \\qquad \\epsilon \\sim \\mathcal{N}(0, \\sigma_{\\ln Y}^2),

where :math:`y_0` is the central Compton parameter predicted by the Arnaud
et al. (2010) self-similar pressure profile evaluated at the *hydrostatic* mass
:math:`M_{500c}/B`, and :math:`\\sigma_{y_0}` is the Planck-like matched-filter
noise as a function of angular size (``szifi`` ``immf6`` filter curve).

Only ``(M_500c, z)`` are needed: no map or aperture photometry is involved, so
the same relation can be applied to any FLAMINGO catalogue (any feedback
variant, hydro or DMO). The heavy lifting -- the parametric A10 :math:`y_0`, the
angular size and the noise interpolation -- comes from
``hmfast.tracers.tsz_completeness``; this module fixes the FLAMINGO conventions
and the amplitude calibration around it.

The intrinsic scatter is drawn *deterministically* from a per-object integer
label (the SOAP index), so repeated runs and independent scripts assign the same
realisation to the same halo.

Example
-------
>>> from flamingo.cnc import SZScaling
>>> sz = SZScaling.calibrated()                      # doctest: +SKIP
>>> q = sz.q(M_500c_Msun, z, index=soap_index)       # doctest: +SKIP
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from hmfast.halos import HaloModel
from hmfast.halos.mass_definition import MassDefinition
from hmfast.tracers.tsz_completeness import (
    compute_theta500_arcmin,
    compute_y0_parametric,
    load_sigma_y0_curve,
    sigma_y0_from_theta,
)

from .catalogue.frame import D3A_COSMOLOGY
from . import paths

# --- FLAMINGO / Planck conventions -----------------------------------------

#: Hydrostatic mass bias: the scaling relation is evaluated at ``M_500c / B``.
B_HYDROSTATIC = 1.35

#: Intrinsic lognormal scatter in the Y-M relation (Planck 2015 XXIV).
SIGMA_LNY = 0.173

#: A10 self-similar exponent of ``y_0`` with mass: 2/3 (Y-M) + 0.12 (alpha_p)
#: + 1/3 (the ``1/theta_500`` of converting Y to a central amplitude).
ALPHA_SZ = 2.0 / 3.0 + 0.12 + 1.0 / 3.0

#: Matched-filter noise curve shipped with the repo (szifi, Planck-like).
SIGMA_Y0_FILE = paths.DATA / "noise" / "sigma_dict_szifi.npy"
SKYFRACS_FILE = paths.DATA / "noise" / "skyfracs_szifi_cosmology.npy"
FILTER_NAME = "immf6"

#: Seed for the deterministic per-object intrinsic scatter.
SCATTER_SEED = 20260630

# Physical constants, CGS unless noted.
_SIGMA_T_CM2 = 6.6524587e-25  # Thomson cross-section
_M_E_C2_EV = 510998.95  # electron rest energy
_MPC_CM = 3.085677581e24  # Mpc in cm
_KEV_CM3_TO_EV_CM3 = 8.13  # A10 P500 units -> the y0 integrand normalisation
_LOS_INTEGRAL = 0.470502095  # int p(sqrt(x^2+l^2)) dl for the A10 shape at x=0


def _a10_y0_reference(halo_model: HaloModel, m: np.ndarray, z: np.ndarray, B: float) -> np.ndarray:
    """Central Compton ``y_0`` from the analytic A10 self-similar profile.

    This is the amplitude the parametric ``hmfast`` model has to reproduce; it
    is used only to calibrate ``A_SZ`` (see :meth:`SZScaling.calibrated`).

    Parameters
    ----------
    halo_model : hmfast.halos.HaloModel
        Halo model carrying the cosmology.
    m : numpy.ndarray
        ``M_500c`` in solar masses, shape ``(n_m,)``.
    z : numpy.ndarray
        Redshift, shape ``(n_z,)``.
    B : float
        Hydrostatic mass bias.

    Returns
    -------
    numpy.ndarray
        ``y_0`` on the ``(n_m, n_z)`` grid.
    """
    cosmo = halo_model.cosmology
    h = cosmo.H0 / 100.0
    m = np.atleast_1d(np.asarray(m, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))

    r500 = MassDefinition(500, "critical").r_delta(cosmo, m[:, None], z[None, :])
    E = np.atleast_1d(np.asarray(cosmo.hubble_parameter(z))) / cosmo.H0

    # Arnaud et al. (2010) eq. 5: the self-similar pressure scale P_500.
    P500 = (
        1.65
        * (h / 0.7) ** 2
        * E[None, :] ** (8.0 / 3.0)
        * ((m[:, None] * h / B) / (0.7 * 3e14)) ** (2.0 / 3.0 + 0.12)
        * (0.7 / h) ** 1.5
    )
    return (
        2.0
        * (_SIGMA_T_CM2 / _M_E_C2_EV)
        * _KEV_CM3_TO_EV_CM3
        * P500
        * (r500 * _MPC_CM)
        * _LOS_INTEGRAL
    )


@dataclass(frozen=True)
class SZScaling:
    """The ``(M_500c, z) -> q`` selection relation with FLAMINGO conventions.

    Build with :meth:`calibrated`, which fits ``A_SZ`` so that the parametric
    ``hmfast`` amplitude matches the analytic A10 profile, and loads the
    matched-filter noise curve.

    Attributes
    ----------
    halo_model : hmfast.halos.HaloModel
        Halo model carrying the cosmology (FLAMINGO D3A by default).
    A_SZ : float
        log10 amplitude offset of the parametric ``y_0`` model.
    noise_coeff : jax.Array
        Polynomial coefficients of ``log sigma_y0`` vs ``log theta_500``.
    B : float
        Hydrostatic mass bias.
    alpha_SZ : float
        Mass exponent of ``y_0``.
    sigma_lnY : float
        Intrinsic lognormal scatter.
    seed : int
        Base seed for the deterministic per-object scatter.
    """

    halo_model: HaloModel
    A_SZ: float
    noise_coeff: jnp.ndarray
    B: float = B_HYDROSTATIC
    alpha_SZ: float = ALPHA_SZ
    sigma_lnY: float = SIGMA_LNY
    seed: int = SCATTER_SEED

    @classmethod
    def calibrated(
        cls,
        *,
        cosmology=D3A_COSMOLOGY,
        B: float = B_HYDROSTATIC,
        alpha_SZ: float = ALPHA_SZ,
        sigma_lnY: float = SIGMA_LNY,
        sigma_y0_file: Path | str = SIGMA_Y0_FILE,
        skyfracs_file: Path | str = SKYFRACS_FILE,
        filter_name: str = FILTER_NAME,
        seed: int = SCATTER_SEED,
    ) -> "SZScaling":
        """Calibrate ``A_SZ`` against the analytic A10 profile and load the noise curve.

        ``A_SZ`` is the median ``log10`` offset between the analytic A10 ``y_0``
        and the parametric model at ``A_SZ = 0``, taken over a
        ``10^13 - 10^15.5 Msun`` by ``z = 0.01 - 3`` grid.

        Parameters
        ----------
        cosmology : hmfast.Cosmology, optional
            Cosmology (default FLAMINGO D3A).
        B, alpha_SZ, sigma_lnY : float, optional
            Scaling-relation parameters (defaults are the FLAMINGO conventions).
        sigma_y0_file, skyfracs_file : Path or str, optional
            ``szifi`` matched-filter noise products.
        filter_name : str, optional
            Filter whose noise curve to use (default ``"immf6"``).
        seed : int, optional
            Base seed for the intrinsic scatter.

        Returns
        -------
        SZScaling
            Ready-to-evaluate scaling relation.
        """
        halo_model = HaloModel(cosmology=cosmology)

        m = np.logspace(13.0, 15.5, 48)
        z = np.geomspace(0.01, 3.0, 48)
        y0_ref = _a10_y0_reference(halo_model, m, z, B)
        y0_at_zero = np.asarray(
            compute_y0_parametric(halo_model, jnp.asarray(m), jnp.asarray(z), 0.0, alpha_SZ, B)
        )
        A_SZ = float(np.nanmedian(np.log10(y0_ref / y0_at_zero)))

        coeff, _ = load_sigma_y0_curve(
            sigma_obj_file=str(paths.require(Path(sigma_y0_file))),
            skyfr_file=str(paths.require(Path(skyfracs_file))),
            filter_name=filter_name,
        )
        return cls(
            halo_model=halo_model,
            A_SZ=A_SZ,
            noise_coeff=jnp.asarray(coeff),
            B=B,
            alpha_SZ=alpha_SZ,
            sigma_lnY=sigma_lnY,
            seed=seed,
        )

    # ``hmfast``'s ``compute_y0_parametric`` and ``compute_theta500_arcmin`` build
    # an outer (mass, redshift) *grid*. A halo catalogue instead needs the
    # elementwise value on paired ``(m_i, z_i)``, so everything below is vmapped
    # over scalars, matching the diagonal of the grid.

    def theta_500_arcmin(self, m: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Angular size ``theta_500`` in arcmin at the hydrostatic mass ``M/B``.

        Parameters
        ----------
        m : numpy.ndarray
            ``M_500c`` in solar masses, shape ``(n,)``.
        z : numpy.ndarray
            Redshift, shape ``(n,)``.

        Returns
        -------
        numpy.ndarray
            ``theta_500`` in arcmin, shape ``(n,)``.
        """
        return self._evaluate(self._theta_scalar, m, z)

    def y0(self, m: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Mean (scatter-free) central Compton parameter ``y_0``.

        Parameters
        ----------
        m : numpy.ndarray
            ``M_500c`` in solar masses, shape ``(n,)``.
        z : numpy.ndarray
            Redshift, shape ``(n,)``.

        Returns
        -------
        numpy.ndarray
            ``y_0``, dimensionless, shape ``(n,)``.
        """
        return self._evaluate(self._y0_scalar, m, z)

    def sigma_y0(self, theta_500_arcmin: np.ndarray) -> np.ndarray:
        """Matched-filter noise ``sigma_y0`` at a given angular size.

        Parameters
        ----------
        theta_500_arcmin : numpy.ndarray
            Angular size in arcmin.

        Returns
        -------
        numpy.ndarray
            ``sigma_y0``, dimensionless, same shape as the input.
        """
        sigma = sigma_y0_from_theta(
            jnp.asarray(theta_500_arcmin, dtype=jnp.float64), self.noise_coeff
        )
        return np.asarray(jax.device_get(sigma))

    def q(
        self,
        m: np.ndarray,
        z: np.ndarray,
        *,
        index: np.ndarray | None = None,
    ) -> np.ndarray:
        """Detection signal-to-noise ``q`` for each ``(M_500c, z)`` pair.

        Parameters
        ----------
        m : numpy.ndarray
            ``M_500c`` in solar masses, shape ``(n,)``.
        z : numpy.ndarray
            Redshift, shape ``(n,)``.
        index : numpy.ndarray, optional
            Per-object integer label (typically ``soap_index``) seeding the
            intrinsic scatter. If ``None`` the scatter-free ``q`` is returned.

        Returns
        -------
        numpy.ndarray
            ``q``, shape ``(n,)``.
        """
        q_mean = self._evaluate(self._q_scalar, m, z)
        if index is None:
            return q_mean
        return q_mean * np.exp(self.scatter(index))

    def scatter(self, index: np.ndarray) -> np.ndarray:
        """Deterministic ``N(0, sigma_lnY^2)`` draw keyed on a per-object integer.

        Parameters
        ----------
        index : numpy.ndarray
            Per-object integer label, typically the SOAP index. The same label
            always yields the same draw, so independent runs and scripts agree.

        Returns
        -------
        numpy.ndarray
            The log-amplitude offset ``epsilon``, shape ``(n,)``.
        """
        base = jax.random.PRNGKey(self.seed)
        sigma = self.sigma_lnY

        @jax.jit
        @jax.vmap
        def draw(i):
            return jax.random.normal(jax.random.fold_in(base, i)) * sigma

        return np.asarray(jax.device_get(draw(jnp.asarray(index, dtype=jnp.uint32))))

    # --- scalar kernels, vmapped by ``_evaluate`` ---------------------------

    def _theta_scalar(self, m: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        return compute_theta500_arcmin(self.halo_model, m, z, self.B).reshape(())

    def _y0_scalar(self, m: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        return compute_y0_parametric(
            self.halo_model, m, z, float(self.A_SZ), self.alpha_SZ, self.B
        ).reshape(())

    def _q_scalar(self, m: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        return self._y0_scalar(m, z) / sigma_y0_from_theta(
            self._theta_scalar(m, z), self.noise_coeff
        )

    @staticmethod
    def _evaluate(scalar_fn, m: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Run a scalar ``(m, z)`` kernel elementwise over paired arrays."""
        batched = jax.jit(jax.vmap(scalar_fn))
        values = batched(
            jnp.asarray(m, dtype=jnp.float64), jnp.asarray(z, dtype=jnp.float64)
        )
        return np.asarray(jax.device_get(values))
