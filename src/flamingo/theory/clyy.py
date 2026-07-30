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


#: Default halo-mass integration grid, in *physical* Msun (hmfast masses are
#: Msun, never Msun/h). The lower edge sits far below the halos that carry the
#: tSZ signal on purpose -- see the ``hm_consistency`` note in :func:`cl_yy`.
M_GRID = np.logspace(11.0, 16.0, 64)

#: Default redshift grid, spanning the FLAMINGO lightcone. The lower edge is a
#: small positive floor rather than 0: the Limber projection divides by the
#: comoving distance, so z = 0 returns NaN. Matches the grid used by the
#: ``cobaya`` tSZ theory modules on the ``main`` branch.
Z_GRID = np.geomspace(0.005, 3.0, 96)


def cl_yy(
    ell: np.ndarray,
    *,
    B: float = 1.0,
    gnfw_params: dict | None = None,
    cosmology=D3A_COSMOLOGY,
    m_grid: np.ndarray | None = None,
    z_grid: np.ndarray | None = None,
    hm_consistency: bool = False,
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
        Halo-mass integration grid in physical Msun (default :data:`M_GRID`).
    z_grid : numpy.ndarray, optional
        Redshift integration grid (default :data:`Z_GRID`).
    hm_consistency : bool, optional
        Whether to add hmfast's halo-model consistency counterterm (default
        ``False``). See the note below.

    Returns
    -------
    dict
        ``{"ell", "cl_1h", "cl_2h", "cl"}`` with ``cl = cl_1h + cl_2h``, all
        as 1-D float arrays on the input ``ell`` grid.

    Notes
    -----
    hmfast's consistency counterterm assigns every unit of mass missing from
    the grid, ``1 - int dn/dlnm (m/rho_m) dlnm``, to a phantom halo population
    sitting at ``m_grid[0]``, and gives it that halo's pressure profile. On a
    grid starting at ``10^13 Msun`` most of the matter budget is missing, so
    the counterterm dominates the small scales: it inflates ``C_ell^1h`` by a
    factor 3.3 at ``ell = 6000``. It is off by default. On :data:`M_GRID` the
    resolved halos account for nearly all the mass, and the switch then moves
    ``C_ell^1h`` by <0.1% and ``C_ell^2h`` -- which is subdominant everywhere
    here -- by ~16%.
    """
    if gnfw_params is None:
        gnfw_params = A10_PARAMS
    if m_grid is None:
        m_grid = M_GRID
    if z_grid is None:
        z_grid = Z_GRID

    # Mass grid is physical M_500c (tSZ / A10 convention). Do not use the
    # HaloModel default (M_200c), or convert_m_delta will treat M_500c as M_200c.
    from hmfast.halos.mass_definition import MassDefinition

    hm = HaloModel(
        cosmology=cosmology,
        mass_definition=MassDefinition(500, "critical"),
        convert_masses=True,
        hm_consistency=hm_consistency,
    )
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
