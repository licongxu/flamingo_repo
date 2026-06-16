"""Spherical-geometry helpers shared by the masking and profile modules.

All angles are in radians unless a name ends in ``_arcmin`` or ``_deg``. Sky
positions follow the HEALPix convention: ``theta`` is colatitude in ``[0, pi]``
and ``phi`` is longitude in ``[0, 2 pi)``.
"""
from __future__ import annotations

import healpy as hp
import numpy as np

ARCMIN_PER_RAD = 180.0 * 60.0 / np.pi


def query_disc_separation(
    nside: int,
    theta_c: float,
    phi_c: float,
    radius_rad: float,
    *,
    nest: bool = False,
    inclusive: bool = True,
    fact: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Pixels inside a disc and their angular distance from the centre.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter of the target map.
    theta_c, phi_c : float
        Disc centre (colatitude, longitude) in radians.
    radius_rad : float
        Disc radius in radians.
    nest : bool, optional
        Pixel ordering of the map the returned indices will index into.
    inclusive : bool, optional
        Passed to :func:`healpy.query_disc`; include pixels that overlap the
        disc boundary.
    fact : int, optional
        Oversampling factor for the inclusive query.

    Returns
    -------
    pix : numpy.ndarray
        Integer pixel indices inside the disc.
    sep_rad : numpy.ndarray
        Great-circle separation of each pixel centre from ``(theta_c, phi_c)``,
        in radians, same shape as ``pix``.
    """
    cvec = hp.ang2vec(theta_c, phi_c)
    pix = hp.query_disc(nside, cvec, radius_rad, inclusive=inclusive, fact=fact, nest=nest)
    vecs = np.asarray(hp.pix2vec(nside, pix, nest=nest))
    cos_sep = np.clip(cvec @ vecs, -1.0, 1.0)
    return pix, np.arccos(cos_sep)


def pixel_size_arcmin(nside: int) -> float:
    """Approximate HEALPix pixel scale (square-root of pixel area) in arcmin."""
    return float(np.degrees(hp.nside2resol(nside)) * 60.0)
