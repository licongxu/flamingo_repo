"""Line-of-sight projection of the GNFW pressure profile.

A spherical pressure profile ``p(r)`` projects to a Compton-y shape

.. math::
    y(b) \\propto \\int p\\!\\left(\\sqrt{b^2 + s^2}\\right) \\, ds,

with ``b`` the projected (impact-parameter) radius in the same units as the
profile argument. The result is normalised to unity at the centre so it can be
multiplied by a measured central amplitude ``y0``.

Written in ``jax.numpy``: ``projected_shape`` is jittable and ``vmap``-able, so
projecting many haloes is a single batched GPU call.
"""
from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from .. import jaxconfig  # noqa: F401  (enables x64)
from .gnfw import gnfw


def projected_shape(
    b: jnp.ndarray,
    *,
    p_func: Callable[[jnp.ndarray], jnp.ndarray] = gnfw,
    s_max: float = 5.0,
    n_s: int = 4000,
) -> jnp.ndarray:
    """Project a 3-D profile along the line of sight, normalised to ``b -> 0``.

    Parameters
    ----------
    b : jax.Array
        Projected radii, same units as the argument of ``p_func`` (default
        ``x = r / R_500c``).
    p_func : callable, optional
        Dimensionless pressure shape ``p(r)``. Defaults to the GNFW/Arnaud
        profile :func:`flamingo.profiles.gnfw.gnfw`.
    s_max : float, optional
        Half-length of the line-of-sight integration, same units as ``b``;
        choose it well beyond the profile truncation (default ``5`` R_500c).
    n_s : int, optional
        Number of line-of-sight samples (trapezoidal rule).

    Returns
    -------
    jax.Array
        Projected shape ``y(b)/y(0)``, same shape as ``b``.
    """
    b = jnp.atleast_1d(jnp.asarray(b, dtype=float))
    s = jnp.linspace(0.0, s_max, n_s)
    integrand = p_func(jnp.sqrt(b[:, None] ** 2 + s[None, :] ** 2))
    los = jnp.trapezoid(integrand, s, axis=1)
    # Normalise by the b -> 0 line-of-sight integral, p(sqrt(0 + s^2)) = p(s),
    # so the returned shape is exactly unity at the centre.
    central = jnp.trapezoid(p_func(s), s)
    return los / central
