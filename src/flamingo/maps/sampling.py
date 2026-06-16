"""Sample HEALPix map values at sky positions.

Generalizes ``sample_y`` from the flamingo_l2p8_lc0 bundle: read a map value at
each catalogue position, optionally taking the maximum over the central pixel
and its neighbours to be robust to sub-pixel centroid offsets.
"""
from __future__ import annotations

import healpy as hp
import numpy as np

from .io import nside_of


def sample_at(
    m: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    nest: bool = False,
) -> np.ndarray:
    """Map value in the pixel containing each ``(theta, phi)`` position.

    Parameters
    ----------
    m : numpy.ndarray
        HEALPix map.
    theta, phi : numpy.ndarray
        Colatitude and longitude of the sample positions, in radians, in the
        same rotation frame as ``m``.
    nest : bool, optional
        Pixel ordering of ``m`` (default RING).

    Returns
    -------
    numpy.ndarray
        Map value at the central pixel of each position.
    """
    nside = nside_of(m)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    pix = hp.ang2pix(nside, theta, phi, nest=nest)
    return m[pix]


def sample_neighbour_max(
    m: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    nest: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Map value at each position, maximised over the pixel and its neighbours.

    Taking the maximum over the central pixel plus its (up to eight) HEALPix
    neighbours makes the sampled value robust to small offsets between the
    catalogue centroid and the brightest pixel.

    Parameters
    ----------
    m : numpy.ndarray
        HEALPix map.
    theta, phi : numpy.ndarray
        Sample positions in radians, in the same frame as ``m``.
    nest : bool, optional
        Pixel ordering of ``m`` (default RING).

    Returns
    -------
    value_max : numpy.ndarray
        Maximum map value over the central pixel and its neighbours.
    value_central : numpy.ndarray
        Map value at the central pixel only.
    """
    nside = nside_of(m)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    pix = hp.ang2pix(nside, theta, phi, nest=nest)
    value_central = m[pix]

    neigh = hp.get_all_neighbours(nside, theta, phi, nest=nest)  # (8, N)
    stack = np.vstack([pix[None, :], neigh])
    # query positions near a pole can return -1 neighbours; fall back to centre.
    stack = np.where(stack >= 0, stack, pix[None, :])
    value_max = m[stack].max(axis=0)
    return value_max, value_central
