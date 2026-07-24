"""Full-sky power spectra via healpy ``anafast`` (no mask, no pymaster).

For masked maps use :mod:`flamingo.powerspectra.namaster`, which decouples the
mask; ``anafast`` on a masked map returns the coupled pseudo-Cl.
"""
from __future__ import annotations

import healpy as hp
import numpy as np


def full_sky_cl(m: np.ndarray, *, lmax: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Full-sky auto-spectrum of a scalar map, monopole subtracted.

    Parameters
    ----------
    m : numpy.ndarray
        Scalar HEALPix map in RING ordering (e.g. a Compton-y map).
    lmax : int, optional
        Maximum multipole (healpy default ``3*nside - 1``).

    Returns
    -------
    ell : numpy.ndarray
        Multipoles ``0 .. lmax``.
    cl : numpy.ndarray
        ``C_ell`` per multipole.
    """
    m = np.asarray(m, dtype=np.float64)
    cl = hp.anafast(m - m.mean(), lmax=lmax)
    return np.arange(cl.size), cl


def bin_cl(
    ell: np.ndarray,
    cl: np.ndarray,
    *,
    delta_ell: int = 30,
    ell_min: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average ``C_ell`` in linear bandpowers, matching the NaMaster binning.

    Parameters
    ----------
    ell, cl : numpy.ndarray
        Per-multipole spectrum, e.g. from :func:`full_sky_cl`.
    delta_ell : int, optional
        Linear bandpower width (default 30).
    ell_min : int, optional
        Lowest multipole to include (default 2; drop monopole and dipole).

    Returns
    -------
    ell_eff : numpy.ndarray
        Mean multipole of each bandpower.
    dl : numpy.ndarray
        ``D_ell = ell(ell+1) C_ell / 2 pi`` averaged per bandpower.
    cl_binned : numpy.ndarray
        ``C_ell`` averaged per bandpower.
    """
    ell = np.asarray(ell)
    cl = np.asarray(cl)
    keep = ell >= ell_min
    ell, cl = ell[keep], cl[keep]

    idx = (ell - ell_min) // delta_ell
    counts = np.bincount(idx)
    ell_eff = np.bincount(idx, weights=ell) / counts
    cl_binned = np.bincount(idx, weights=cl) / counts
    dl = np.bincount(idx, weights=ell * (ell + 1.0) * cl / (2.0 * np.pi)) / counts
    return ell_eff, dl, cl_binned
