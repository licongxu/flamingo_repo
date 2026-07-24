"""Halo-model tSZ power spectrum ``C_ell^yy`` via hmfast.

Thin wrapper around hmfast's :class:`~hmfast.halos.HaloModel` with a GNFW
electron-pressure :class:`~hmfast.tracers.tSZTracer`: one call returns the
1-halo, 2-halo, and total spectra on an ``ell`` grid. Defaults are the
FLAMINGO D3A cosmology and the Arnaud A10 profile with hydrostatic bias
``B = M_true / M_estimated`` applied by hmfast.

Importing this module loads the hmfast emulators (via the D3A cosmology).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from hmfast.halos import HaloModel
from hmfast.halos.profiles import GNFWPressureProfile
from hmfast.tracers import tSZTracer

from ..catalogue.frame import D3A_COSMOLOGY
from ..profiles.gnfw import A10_PARAMS


def cl_yy(
    ell: np.ndarray,
    *,
    B: float = 1.0,
    gnfw_params: dict | None = None,
    cosmology=D3A_COSMOLOGY,
    m_grid: np.ndarray | None = None,
    z_grid: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute the halo-model tSZ auto-spectrum on an ``ell`` grid.

    Parameters
    ----------
    ell : numpy.ndarray
        Multipoles at which to evaluate the spectrum.
    B : float, optional
        Hydrostatic mass bias ``M_true / M_est`` (default 1.0; Planck-style
        analyses use ``B ~ 1.4``).
    gnfw_params : dict, optional
        GNFW parameters ``{P0, c500, gamma, alpha, beta}`` (default Arnaud
        A10, :data:`flamingo.profiles.gnfw.A10_PARAMS`).
    cosmology : hmfast.Cosmology, optional
        Background cosmology (default FLAMINGO D3A).
    m_grid : numpy.ndarray, optional
        Halo-mass integration grid in Msun (default 48 log-spaced points over
        ``10^13 .. 10^15.5``).
    z_grid : numpy.ndarray, optional
        Redshift integration grid (default 48 geometric points over
        ``0.01 .. 3``).

    Returns
    -------
    dict
        ``{"ell", "cl_1h", "cl_2h", "cl"}`` with ``cl = cl_1h + cl_2h``, all
        as 1-D float arrays on the input ``ell`` grid.
    """
    if gnfw_params is None:
        gnfw_params = A10_PARAMS
    if m_grid is None:
        m_grid = np.logspace(13.0, 15.5, 48)
    if z_grid is None:
        z_grid = np.geomspace(0.01, 3.0, 48)

    hm = HaloModel(cosmology=cosmology)
    tsz = tSZTracer(profile=GNFWPressureProfile(**gnfw_params, B=B))

    ell = jnp.asarray(ell, dtype=float)
    m = jnp.asarray(m_grid, dtype=float)
    z = jnp.asarray(z_grid, dtype=float)
    cl_1h = np.asarray(hm.cl_1h(tsz, tsz, l=ell, m=m, z=z))
    cl_2h = np.asarray(hm.cl_2h(tsz, tsz, l=ell, m=m, z=z))
    return {
        "ell": np.asarray(ell),
        "cl_1h": cl_1h,
        "cl_2h": cl_2h,
        "cl": cl_1h + cl_2h,
    }


def dl_of_cl(ell: np.ndarray, cl: np.ndarray) -> np.ndarray:
    """Convert ``C_ell`` to ``D_ell = ell(ell+1) C_ell / 2 pi``."""
    ell = np.asarray(ell, dtype=float)
    return ell * (ell + 1.0) * np.asarray(cl) / (2.0 * np.pi)
