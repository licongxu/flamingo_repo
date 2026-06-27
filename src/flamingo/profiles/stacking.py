"""Measure radial y-profiles from a HEALPix map around catalogue positions.

These routines query the map (healpy ``query_disc``) and bin pixel values by
angular separation, so they run on the CPU in NumPy. The model side (GNFW
evaluation and line-of-sight projection) is the JAX/GPU part, in
:mod:`flamingo.profiles.gnfw` and :mod:`flamingo.profiles.projection`.
"""
from __future__ import annotations

import healpy as hp
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
    nest: bool = False,
) -> tuple[np.ndarray, float]:
    """Per-cluster annular profile and aperture amplitude in scaled radius.

    Empirical estimator: bin map pixels by ``x = theta / theta_500`` and also
    measure the aperture-mean signal inside ``x < 1``. Dividing the first by the
    second gives a dimensionless, mass-independent profile that stacks across
    clusters of different angular size (see :func:`stack_normalized`).

    Parameters
    ----------
    m : numpy.ndarray
        HEALPix map.
    theta_c, phi_c : float
        Centre (colatitude, longitude) in radians.
    theta500_arcmin : float
        Cluster angular scale in arcmin.
    x_edges : numpy.ndarray
        Bin edges in units of ``theta_500`` (e.g. ``np.linspace(0, 6, 13)``).
    nest : bool, optional
        Pixel ordering of ``m``.

    Returns
    -------
    ybar : numpy.ndarray
        Annular-mean map value per ``x`` bin, ``ybar_i = sum_{p in i} y_p / N_i``
        (equal pixel weights); ``nan`` where a bin is empty. Length
        ``len(x_edges) - 1``.
    y_norm : float
        Aperture-mean signal inside ``x < 1``,
        ``y_norm = Y500 / (pi theta500^2)`` with
        ``Y500 = sum_{x < 1} y_p Omega_pix``.
    """
    nside = nside_of(m)
    theta500_rad = theta500_arcmin / ARCMIN_PER_RAD
    r_out_rad = x_edges[-1] * theta500_rad
    pix, sep_rad = query_disc_separation(nside, theta_c, phi_c, r_out_rad, nest=nest)
    vals = m[pix]
    x = sep_rad / theta500_rad

    ybar = np.full(len(x_edges) - 1, np.nan)
    idx = np.digitize(x, x_edges) - 1
    for b in range(ybar.size):
        sel = idx == b
        if sel.any():
            ybar[b] = vals[sel].mean()

    inside = x < 1.0
    omega_pix = hp.nside2pixarea(nside)
    Y500 = vals[inside].sum() * omega_pix
    y_norm = Y500 / (np.pi * theta500_rad**2)
    return ybar, y_norm


def stack_normalized(
    m: np.ndarray,
    theta_c: np.ndarray,
    phi_c: np.ndarray,
    theta500_arcmin: np.ndarray,
    x_edges: np.ndarray,
    *,
    nest: bool = False,
) -> dict:
    """Stack self-normalised profiles over many clusters.

    Each cluster contributes ``g_i = ybar_i / y_norm`` (from
    :func:`normalized_profile`), so the stack measures the average *shape* with
    equal weight per cluster, independent of mass/amplitude.

    Parameters
    ----------
    m : numpy.ndarray
        HEALPix map.
    theta_c, phi_c, theta500_arcmin : numpy.ndarray
        Per-cluster centres (radians) and angular scales (arcmin), length ``N``.
    x_edges : numpy.ndarray
        Common ``x = theta/theta_500`` bin edges.
    nest : bool, optional
        Pixel ordering of ``m``.

    Returns
    -------
    dict
        ``x_mid`` (area-weighted bin-centre radii), ``fhat`` (mean stacked
        profile), ``sem`` (standard error on the stack), ``p16``/``p84``
        (cluster-to-cluster scatter percentiles), and ``n`` (number of clusters
        stacked). Per-cluster ``nan`` bins are ignored.
    """
    theta_c = np.asarray(theta_c, dtype=float)
    phi_c = np.asarray(phi_c, dtype=float)
    theta500_arcmin = np.asarray(theta500_arcmin, dtype=float)

    g_rows = []
    for t, p, t5 in zip(theta_c, phi_c, theta500_arcmin):
        ybar, y_norm = normalized_profile(m, t, p, t5, x_edges, nest=nest)
        if np.isfinite(y_norm) and y_norm > 0:
            g_rows.append(ybar / y_norm)
    G = np.vstack(g_rows)

    x_lo, x_hi = x_edges[:-1], x_edges[1:]
    x_mid = (2.0 / 3.0) * (x_hi**3 - x_lo**3) / (x_hi**2 - x_lo**2)
    n = np.sum(np.isfinite(G), axis=0)
    return dict(
        x_mid=x_mid,
        fhat=np.nanmean(G, axis=0),
        sem=np.nanstd(G, axis=0) / np.sqrt(np.maximum(n, 1)),
        p16=np.nanpercentile(G, 16, axis=0),
        p84=np.nanpercentile(G, 84, axis=0),
        n=G.shape[0],
    )
