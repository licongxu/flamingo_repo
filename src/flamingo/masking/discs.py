"""Binary cluster masks built from discs around catalogue positions.

Pure healpy (no NaMaster). Apodization, which the power-spectrum estimator
needs, lives in :mod:`flamingo.powerspectra` because it depends on pymaster.
"""
from __future__ import annotations

import healpy as hp
import numpy as np


def disc_mask(
    nside: int,
    theta: np.ndarray,
    phi: np.ndarray,
    radius_rad: np.ndarray,
    *,
    nest: bool = False,
    inclusive: bool = True,
    fact: int = 4,
) -> np.ndarray:
    """Build a binary mask that is zero inside discs around each position.

    Each cluster removes a disc of its own radius, so ``radius_rad`` is given
    per object (e.g. ``factor * theta_500``). The convention matches the
    flamingo_l2p8_lc0 masked power-spectrum pipeline: masked pixels are ``0``,
    kept pixels are ``1``.

    Parameters
    ----------
    nside : int
        HEALPix resolution of the mask.
    theta, phi : numpy.ndarray
        Disc centres (colatitude, longitude) in radians, length ``N``.
    radius_rad : numpy.ndarray
        Disc radius per centre in radians; scalar is broadcast to all ``N``.
    nest : bool, optional
        Pixel ordering of the returned mask (default RING).
    inclusive, fact : optional
        Passed to :func:`healpy.query_disc`.

    Returns
    -------
    numpy.ndarray
        Mask of length ``12 * nside**2``, dtype float64, values in ``{0, 1}``.
    """
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    radius_rad = np.broadcast_to(np.asarray(radius_rad, dtype=float), theta.shape)

    mask = np.ones(hp.nside2npix(nside), dtype=np.float64)
    for th, ph, r in zip(theta, phi, radius_rad):
        cvec = hp.ang2vec(th, ph)
        pix = hp.query_disc(nside, cvec, r, inclusive=inclusive, fact=fact, nest=nest)
        mask[pix] = 0.0
    return mask


def fsky(mask: np.ndarray) -> float:
    """Sky fraction retained by a mask (mean of the mask weights)."""
    return float(np.asarray(mask).mean())
