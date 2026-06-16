"""Generalized NFW (GNFW) pressure profile.

A single dimensionless shape function ``p(x)`` of the scaled radius
``x = r / R_500c``,

.. math::
    p(x) = P_0 \\, (c_{500}\\, x)^{-\\gamma}
           \\left[1 + (c_{500}\\, x)^{\\alpha}\\right]^{-(\\beta - \\gamma)/\\alpha}.

The default parameters are the Arnaud et al. (2010) "universal pressure
profile" (UPP), so ``gnfw(x)`` returns the A10 shape. Override any parameter to
explore other GNFW fits.

Written in ``jax.numpy`` so the profile is jittable, differentiable, and
``vmap``-able over many haloes on GPU.

References
----------
Arnaud et al. 2010, A&A 517, A92.
"""
from __future__ import annotations

import jax.numpy as jnp

from .. import jaxconfig  # noqa: F401  (enables x64)

# Arnaud et al. (2010) universal pressure profile best-fit parameters.
A10_PARAMS = dict(P0=8.403, c500=1.177, gamma=0.3081, alpha=1.0510, beta=5.4905)


def gnfw(
    x: jnp.ndarray,
    P0: float = A10_PARAMS["P0"],
    c500: float = A10_PARAMS["c500"],
    gamma: float = A10_PARAMS["gamma"],
    alpha: float = A10_PARAMS["alpha"],
    beta: float = A10_PARAMS["beta"],
) -> jnp.ndarray:
    """Evaluate the GNFW pressure shape at scaled radius ``x``.

    Parameters
    ----------
    x : jax.Array
        Scaled radius ``r / R_500c``; clipped at a small floor to avoid the
        central singularity.
    P0 : float, optional
        Overall normalisation (default Arnaud A10).
    c500 : float, optional
        Concentration parameter multiplying ``x`` (default Arnaud A10).
    gamma, alpha, beta : float, optional
        Inner slope, transition sharpness, and outer slope (default Arnaud A10).

    Returns
    -------
    jax.Array
        Dimensionless pressure ``p(x)``, same shape as ``x``.
    """
    cx = c500 * jnp.clip(jnp.asarray(x, dtype=float), 1e-6, None)
    return P0 * cx ** (-gamma) * (1.0 + cx ** alpha) ** (-(beta - gamma) / alpha)
