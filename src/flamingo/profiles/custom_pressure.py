"""Custom-amplitude GNFW electron pressure profile (baseline tSZ theory).

This reuses **hmfast's GNFW shape** unchanged and only swaps the amplitude. The
GNFW core is

.. math::
    p(x) = (c_{500}x)^{-\\gamma}\\,[1+(c_{500}x)^{\\alpha}]^{(\\gamma-\\beta)/\\alpha},
    \\qquad x = r/r_{500c},

exactly as in :class:`hmfast.halos.profiles.pressure.GNFWPressureProfile`. The
electron pressure is

.. math::
    P_e(r,M,z) = P_{500}\\,P_0\\,p(x),
    \\qquad
    P_{500} = P_{500,0}\\,(M_{500c}/M_\\odot)^{\\alpha_\\mathrm{amp}}\\,E(z)^{2+\\beta_\\mathrm{amp}},

i.e. the Arnaud :math:`P_{500c}` normalization is replaced by a self-similar
power law whose exponents follow the **nb17 Y-M convention** (``alpha_amp`` is the
mass slope, ``beta_amp`` the :math:`E(z)` slope; self-similar values are
``alpha_amp=5/3``, ``beta_amp=2/3``, giving :math:`E^{8/3}`).

Implemented by subclassing hmfast's :class:`GNFWPressureProfile` and overriding
only :meth:`u_r`; the Hankel transform / :math:`u_\\ell` machinery
(:meth:`PressureProfile.u_k`) calls ``u_r`` internally, so nothing else changes.
hmfast itself is **not** modified.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from hmfast.halos.mass_definition import MassDefinition, convert_m_delta
from hmfast.halos.profiles.pressure import GNFWPressureProfile
from hmfast.utils import Const

# nb17 lowest-z catalogue Y-M fit (Y = D_A^2 Y in Mpc^2); see notebooks 17/19.
NB17_YM_INTERCEPT = -9.266
_SIGMA_T_CM2 = 6.6524587e-25
_M_E_C2_EV = 510998.95
_RHO_CRIT0_NORM = 2.77536627e11  # Msun/Mpc^3 at h=1


def gnfw_y_shape_integral(
    P0: float,
    c500: float,
    alpha: float,
    beta: float,
    gamma: float,
    *,
    r_out: float = 5.0,
    n: int = 5000,
) -> float:
    """Volume integral ``∫ 4π x² p(x) dx`` for the dimensionless GNFW shape.

    Used to connect ``P_{500,0}`` to the nb17 :math:`Y_{5R500c}` intercept
    (same convention as notebook 19).
    """
    xg = np.logspace(-4, np.log10(r_out), n)
    p = (c500 * xg) ** (-gamma) * (1 + (c500 * xg) ** alpha) ** ((gamma - beta) / alpha)
    return float(np.trapezoid(p * 4 * np.pi * xg ** 2, xg))


def p500_0_from_nb17(
    cosmology,
    alpha_amp: float,
    shape: dict,
    *,
    M_piv: float = 6.0e14,
    C17: float = NB17_YM_INTERCEPT,
) -> float:
    """Calibrate ``P500_0`` so ``Y_{5R500c}(M_\\mathrm{piv}, z=0) = e^{C17}``.

    Parameters
    ----------
    cosmology : hmfast.Cosmology
        Cosmology (for ``h``).
    alpha_amp : float
        Mass exponent in :math:`P_{500}` (nb17 pressure slope ``α_YM - 1``).
    shape : dict
        GNFW shape keys ``P0, c500, alpha, beta, gamma``.
    M_piv : float
        Pivot mass in :math:`M_\\odot` (default ``6e14``).
    C17 : float
        nb17 intercept in ``ln Y`` (default from catalogue fit).
    """
    h = cosmology.H0 / 100.0
    rho_crit0 = _RHO_CRIT0_NORM * h ** 2
    r500c_Mpc = (3 * M_piv / (4 * np.pi * 500 * rho_crit0)) ** (1.0 / 3.0)
    mpc_cm = Const._Mpc_over_m_ * 100.0
    I_shape = gnfw_y_shape_integral(
        shape["P0"], shape["c500"], shape["alpha"], shape["beta"], shape["gamma"]
    )
    Y_unit = (
        (_SIGMA_T_CM2 / _M_E_C2_EV)
        * (M_piv ** alpha_amp)
        * shape["P0"]
        * (r500c_Mpc * mpc_cm) ** 3
        * I_shape
        / mpc_cm ** 2
    )
    return float(np.exp(C17) / Y_unit)


class SelfSimilarGNFWPressureProfile(GNFWPressureProfile):
    """GNFW shape with a self-similar power-law amplitude (see module docstring).

    Parameters
    ----------
    x : array_like, optional
        Dimensionless radial grid (inherited; defaults to the hmfast grid).
    P500_0 : float
        Amplitude :math:`P_{500,0}` of the pressure normalization. Sets the
        overall scale and units; calibrate with :meth:`from_arnaud_pivot` to
        match Arnaud :math:`P_{500c}` at a pivot mass.
    alpha_amp : float
        Mass exponent of :math:`P_{500}` (nb17 mass slope; self-similar ``5/3``).
    beta_amp : float
        :math:`E(z)` exponent is ``2 + beta_amp`` (nb17 :math:`E` slope;
        self-similar ``2/3`` -> :math:`E^{8/3}`).
    P0, c500, alpha, beta, gamma : float
        GNFW shape parameters, identical to :class:`GNFWPressureProfile`.
    B : float
        Hydrostatic bias used in the mass-definition conversion (default 1.0;
        the amplitude scaling already absorbs the effective normalization).
    """

    def __init__(self, x=None, P500_0=1.0, alpha_amp=5.0 / 3.0, beta_amp=2.0 / 3.0,
                 P0=8.130, c500=1.156, alpha=1.0620, beta=5.4807, gamma=0.3292, B=1.0):
        super().__init__(x=x, P0=P0, c500=c500, alpha=alpha, beta=beta, gamma=gamma, B=B)
        self.P500_0 = P500_0
        self.alpha_amp = alpha_amp
        self.beta_amp = beta_amp

    @classmethod
    def from_arnaud_pivot(cls, cosmology, M_piv=6.0e14, z_piv=0.0,
                          alpha_amp=5.0 / 3.0, beta_amp=2.0 / 3.0, **shape):
        """Build a profile with ``P500_0`` calibrated to Arnaud at a pivot.

        ``P500_0`` is chosen so that :math:`P_{500}(M_\\mathrm{piv},z_\\mathrm{piv})`
        equals the Arnaud :math:`P_{500c}` at the same point, which fixes the
        units to hmfast's convention and anchors the amplitude physically.

        Parameters
        ----------
        cosmology : hmfast.Cosmology
            Cosmology (for :math:`h` and :math:`E(z)`).
        M_piv : float
            Pivot mass :math:`M_{500c}` in :math:`M_\\odot` (default 6e14).
        z_piv : float
            Pivot redshift (default 0).
        alpha_amp, beta_amp : float
            Amplitude exponents (see class docstring).
        **shape
            GNFW shape parameters (``P0, c500, alpha, beta, gamma, B, x``).
        """
        h = cosmology.H0 / 100.0
        B = shape.get("B", 1.0)
        E = float(np.asarray(cosmology.hubble_parameter(z_piv))) / cosmology.H0
        # Arnaud P_500c at the pivot (matches GNFWPressureProfile.u_r).
        P500c_arnaud = (1.65 * (h / 0.7) ** 2 * E ** (8.0 / 3.0)
                        * (M_piv * h / B / (0.7 * 3e14)) ** (2.0 / 3.0 + 0.12)
                        * (0.7 / h) ** 1.5)
        P500_0 = P500c_arnaud / (M_piv ** alpha_amp * E ** (2.0 + beta_amp))
        return cls(P500_0=P500_0, alpha_amp=alpha_amp, beta_amp=beta_amp, **shape)

    def _tree_flatten(self):
        leaves = (self.P0, self.c500, self.alpha, self.beta, self.gamma, self.B,
                  self.P500_0, self.alpha_amp, self.beta_amp)
        aux_data = (tuple(self._x.tolist()), self._hankel)
        return (leaves, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, leaves):
        x_tuple, hankel = aux_data
        obj = cls.__new__(cls)
        (obj.P0, obj.c500, obj.alpha, obj.beta, obj.gamma, obj.B,
         obj.P500_0, obj.alpha_amp, obj.beta_amp) = leaves
        obj._x = np.array(x_tuple)
        obj._hankel = hankel
        return obj

    def update(self, P500_0=None, alpha_amp=None, beta_amp=None,
               P0=None, c500=None, alpha=None, beta=None, gamma=None, B=None):
        """Return a new instance with updated parameters (any left ``None`` kept)."""
        leaves, treedef = self._tree_flatten()
        new = (
            P0 if P0 is not None else self.P0,
            c500 if c500 is not None else self.c500,
            alpha if alpha is not None else self.alpha,
            beta if beta is not None else self.beta,
            gamma if gamma is not None else self.gamma,
            B if B is not None else self.B,
            P500_0 if P500_0 is not None else self.P500_0,
            alpha_amp if alpha_amp is not None else self.alpha_amp,
            beta_amp if beta_amp is not None else self.beta_amp,
        )
        return self._tree_unflatten(treedef, new)

    @jax.jit
    def u_r(self, halo_model, r, m, z):
        """Electron pressure with the custom self-similar amplitude.

        Same signature/units as :meth:`GNFWPressureProfile.u_r`; only the
        normalization differs.
        """
        H0 = halo_model.cosmology.H0
        P0, c500, alpha, beta, gamma, B = (
            self.P0, self.c500, self.alpha, self.beta, self.gamma, self.B)
        r, m, z = jnp.atleast_1d(r), jnp.atleast_1d(m), jnp.atleast_1d(z)

        # Convert the input mass to M_500c (the profile's native calibration mass).
        mass_def_500c = MassDefinition(500, "critical")
        c_old = halo_model.concentration.c_delta(halo_model, m, z)
        m500c = convert_m_delta(halo_model.cosmology, m, z,
                                halo_model.mass_definition, mass_def_500c, c_old=c_old)
        r_500c = mass_def_500c.r_delta(halo_model.cosmology, m500c, z)  # (Nm, Nz)
        x_500c = r[:, None, None] / ((1.0 + z[None, None, :]) * r_500c[None, :, :])

        # Custom self-similar amplitude P_500 = P500_0 * (M500c/Msun)^a * E^(2+b).
        E_z = jnp.atleast_1d(halo_model.cosmology.hubble_parameter(z))[None, None, :] / H0
        m500c_b = m500c[None]  # (1, Nm, Nz), in Msun
        P_500 = self.P500_0 * m500c_b ** self.alpha_amp * E_z ** (2.0 + self.beta_amp)

        scaled_x = c500 * x_500c
        Pe = P_500 * P0 * scaled_x ** (-gamma) * (1 + scaled_x ** alpha) ** ((gamma - beta) / alpha)
        return Pe


jax.tree_util.register_pytree_node(
    SelfSimilarGNFWPressureProfile,
    lambda obj: obj._tree_flatten(),
    lambda aux_data, children: SelfSimilarGNFWPressureProfile._tree_unflatten(aux_data, children),
)
