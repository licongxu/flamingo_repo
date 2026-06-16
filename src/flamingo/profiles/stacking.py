"""Measure radial y-profiles from a HEALPix map around catalogue positions.

These routines query the map (healpy ``query_disc``) and bin pixel values by
angular separation, so they run on the CPU in NumPy. The model side (GNFW
evaluation and line-of-sight projection) is the JAX/GPU part, in
:mod:`flamingo.profiles.gnfw` and :mod:`flamingo.profiles.projection`.
"""
from __future__ import annotations

import numpy as np

from ..geometry import query_disc_separation
from ..maps.io import nside_of

ARCMIN_PER_RAD = 180.0 * 60.0 / np.pi


def radial_profile(
    m: np.ndarray,
    theta_c: float,
    phi_c: float,
    theta500_arcmin: float,
    *,
    r_out_factor: float = 8.0,
    n_bins: int = 14,
    bkg_in: float = 4.0,
    bkg_out: float = 6.0,
    nest: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Background-subtracted radial profile around one position, in arcmin.

    Parameters
    ----------
    m : numpy.ndarray
        HEALPix map (e.g. Compton-y), in the same frame as the position.
    theta_c, phi_c : float
        Centre (colatitude, longitude) in radians.
    theta500_arcmin : float
        Cluster angular scale ``theta_500`` in arcmin, used to set the disc and
        background annulus radii.
    r_out_factor : float, optional
        Outer profile radius in units of ``theta_500`` (default 8).
    n_bins : int, optional
        Number of linear radial bins from 0 to the outer radius.
    bkg_in, bkg_out : float, optional
        Inner/outer radii of the background annulus, in units of ``theta_500``.
    nest : bool, optional
        Pixel ordering of ``m`` (default RING).

    Returns
    -------
    r_arcmin : numpy.ndarray
        Bin-centre radii in arcmin.
    y : numpy.ndarray
        Mean background-subtracted map value per bin.
    yerr : numpy.ndarray
        Standard error on the mean per bin.
    bkg : float
        The subtracted background level.
    """
    nside = nside_of(m)
    r_out_arcmin = max(r_out_factor * theta500_arcmin, bkg_out * theta500_arcmin + 1.0)
    pix, sep_rad = query_disc_separation(
        nside, theta_c, phi_c, r_out_arcmin / ARCMIN_PER_RAD, nest=nest
    )
    vals = m[pix]
    sep_arcmin = sep_rad * ARCMIN_PER_RAD

    annulus = (sep_arcmin >= bkg_in * theta500_arcmin) & (sep_arcmin <= bkg_out * theta500_arcmin)
    bkg = float(np.median(vals[annulus])) if annulus.sum() > 5 else 0.0

    edges = np.linspace(0.0, r_out_arcmin, n_bins + 1)
    idx = np.digitize(sep_arcmin, edges) - 1
    r_mid, y_mean, y_err = [], [], []
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        v = vals[sel] - bkg
        r_mid.append(0.5 * (edges[b] + edges[b + 1]))
        y_mean.append(v.mean())
        y_err.append(v.std(ddof=1) / np.sqrt(sel.sum()) if sel.sum() > 1 else 0.0)
    return np.array(r_mid), np.array(y_mean), np.array(y_err), bkg


def normalized_profile(
    m: np.ndarray,
    theta_c: float,
    phi_c: float,
    theta500_arcmin: float,
    x_edges: np.ndarray,
    *,
    bkg_in: float = 4.0,
    bkg_out: float = 6.0,
    nest: bool = False,
) -> np.ndarray:
    """Profile binned in scaled radius ``x = theta / theta_500``.

    Returning the profile on a common ``x`` grid makes it stackable across
    clusters of different angular size.

    Parameters
    ----------
    m : numpy.ndarray
        HEALPix map.
    theta_c, phi_c : float
        Centre in radians.
    theta500_arcmin : float
        Cluster angular scale in arcmin.
    x_edges : numpy.ndarray
        Bin edges in units of ``theta_500`` (e.g. ``np.linspace(0, 6, 13)``).
    bkg_in, bkg_out : float, optional
        Background annulus in units of ``theta_500``.
    nest : bool, optional
        Pixel ordering of ``m``.

    Returns
    -------
    numpy.ndarray
        Mean background-subtracted value in each ``x`` bin; ``nan`` where a bin
        is empty. Length ``len(x_edges) - 1``.
    """
    nside = nside_of(m)
    r_out_arcmin = x_edges[-1] * theta500_arcmin
    pix, sep_rad = query_disc_separation(
        nside, theta_c, phi_c, r_out_arcmin / ARCMIN_PER_RAD, nest=nest
    )
    vals = m[pix]
    x = sep_rad * ARCMIN_PER_RAD / theta500_arcmin

    annulus = (x >= bkg_in) & (x <= bkg_out)
    bkg = float(np.median(vals[annulus])) if annulus.sum() > 5 else 0.0

    out = np.full(len(x_edges) - 1, np.nan)
    idx = np.digitize(x, x_edges) - 1
    for b in range(out.size):
        sel = idx == b
        if sel.any():
            out[b] = (vals[sel] - bkg).mean()
    return out


def stack_normalized(
    m: np.ndarray,
    theta_c: np.ndarray,
    phi_c: np.ndarray,
    theta500_arcmin: np.ndarray,
    x_edges: np.ndarray,
    *,
    bkg_in: float = 4.0,
    bkg_out: float = 6.0,
    nest: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack scaled-radius profiles over many clusters.

    Parameters
    ----------
    m : numpy.ndarray
        HEALPix map.
    theta_c, phi_c, theta500_arcmin : numpy.ndarray
        Per-cluster centres (radians) and angular scales (arcmin), length ``N``.
    x_edges : numpy.ndarray
        Common ``x = theta/theta_500`` bin edges.
    bkg_in, bkg_out : float, optional
        Background annulus in units of ``theta_500``.
    nest : bool, optional
        Pixel ordering of ``m``.

    Returns
    -------
    x_mid : numpy.ndarray
        Bin-centre scaled radii.
    y_stack : numpy.ndarray
        Mean profile across clusters (``nan`` bins ignored per cluster).
    """
    theta_c = np.asarray(theta_c, dtype=float)
    phi_c = np.asarray(phi_c, dtype=float)
    theta500_arcmin = np.asarray(theta500_arcmin, dtype=float)

    profiles = np.vstack(
        [
            normalized_profile(
                m, t, p, t5, x_edges, bkg_in=bkg_in, bkg_out=bkg_out, nest=nest
            )
            for t, p, t5 in zip(theta_c, phi_c, theta500_arcmin)
        ]
    )
    x_mid = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_stack = np.nanmean(profiles, axis=0)
    return x_mid, y_stack
