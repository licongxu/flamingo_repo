"""Measure radial y-profiles from a HEALPix map around catalogue positions.

These routines query the map (healpy ``query_disc``) and bin pixel values by
angular separation, so they run on the CPU in NumPy. The model side (GNFW
evaluation and line-of-sight projection) is the JAX/GPU part, in
:mod:`flamingo.profiles.gnfw` and :mod:`flamingo.profiles.projection`.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import healpy as hp
import numpy as np

from ..geometry import query_disc_separation
from ..maps.io import nside_of

ARCMIN_PER_RAD = 180.0 * 60.0 / np.pi

# Inherited by fork workers so the HEALPix map is not pickled per cluster.
_STACK = {}


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

    Uses raw map pixels: no monopole, local-background, or other subtraction.
    Bin by ``x = theta / theta_500`` and measure the aperture mean inside
    ``x < 1``. Dividing the first by the second gives a dimensionless profile
    that stacks across clusters of different angular size
    (see :func:`stack_normalized`).

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


def default_n_jobs() -> int:
    """Process count for stacking; cap to keep map-cache traffic in check."""
    return max(1, min(64, os.cpu_count() or 1))


def _worker_profile(i: int) -> tuple[np.ndarray, float]:
    s = _STACK
    return normalized_profile(
        s["m"], s["theta"][i], s["phi"][i], s["t5"][i], s["x_edges"], nest=s["nest"]
    )


def _reduce_stack(
    rows: list[tuple[np.ndarray, float]],
    x_edges: np.ndarray,
    y_norms: np.ndarray | None = None,
) -> dict:
    triples = []
    for i, (ybar, y_map) in enumerate(rows):
        denom = y_map if y_norms is None else y_norms[i]
        if np.isfinite(denom) and denom > 0 and np.isfinite(y_map) and y_map > 0:
            triples.append((ybar, denom, y_map))
    if not triples:
        raise ValueError("no clusters with a positive Y_500 aperture")
    G = np.vstack([ybar / denom for ybar, denom, y_map in triples])
    amp = np.array([y_map / denom for ybar, denom, y_map in triples], dtype=float)
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
        A_median=float(np.median(amp)),
        A_p16=float(np.percentile(amp, 16)),
        A_p84=float(np.percentile(amp, 84)),
    )


def measure_profiles(
    m: np.ndarray,
    theta_c: np.ndarray,
    phi_c: np.ndarray,
    theta500_arcmin: np.ndarray,
    x_edges: np.ndarray,
    *,
    nest: bool = False,
    n_jobs: int | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Per-cluster annular profiles and map aperture means."""
    theta_c = np.asarray(theta_c, dtype=float)
    phi_c = np.asarray(phi_c, dtype=float)
    theta500_arcmin = np.asarray(theta500_arcmin, dtype=float)
    n = int(theta_c.size)
    if not (phi_c.size == n == theta500_arcmin.size):
        raise ValueError("theta_c, phi_c, and theta500_arcmin must have the same length")
    jobs = 1 if n_jobs == 1 else (default_n_jobs() if n_jobs is None else int(n_jobs))
    if jobs < 1:
        raise ValueError("n_jobs must be >= 1")

    def _one(i: int) -> tuple[np.ndarray, float]:
        return normalized_profile(m, theta_c[i], phi_c[i], theta500_arcmin[i], x_edges, nest=nest)

    if jobs == 1 or n < 8:
        return [_one(i) for i in range(n)]
    _STACK.update(
        m=m, theta=theta_c, phi=phi_c, t5=theta500_arcmin, x_edges=x_edges, nest=nest
    )
    chunksize = max(32, n // (jobs * 8))
    try:
        with mp.get_context("fork").Pool(jobs) as pool:
            return pool.map(_worker_profile, range(n), chunksize=chunksize)
    finally:
        _STACK.clear()


def stack_normalized(
    m: np.ndarray,
    theta_c: np.ndarray,
    phi_c: np.ndarray,
    theta500_arcmin: np.ndarray,
    x_edges: np.ndarray,
    *,
    nest: bool = False,
    n_jobs: int | None = None,
    y_norms: np.ndarray | None = None,
) -> dict:
    """Stack self-normalised profiles over many clusters.

    Each cluster contributes ``g_i = ybar_i / y_norm``. By default ``y_norm``
    is the map aperture mean inside ``theta_500`` (shape-only). Pass ``y_norms``
    to use a model aperture mean and keep amplitude.

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
    n_jobs : int, optional
        Forked workers (default :func:`default_n_jobs`). ``1`` is serial.
    y_norms : numpy.ndarray, optional
        Per-cluster denominators, length ``N``. Default is the map ``Y_500``
        aperture mean (shape-only). Pass the A10-predicted aperture mean to
        keep amplitude.

    Returns
    -------
    dict
        ``x_mid`` (area-weighted bin-centre radii), ``fhat`` (mean stacked
        profile), ``sem`` (standard error on the stack), ``p16``/``p84``
        (cluster-to-cluster scatter percentiles), and ``n`` (number of clusters
        stacked). Per-cluster ``nan`` bins are ignored.
    """
    if y_norms is not None:
        y_norms = np.asarray(y_norms, dtype=float)
        if y_norms.shape != (int(np.asarray(theta_c).size),):
            raise ValueError(
                f"y_norms has shape {y_norms.shape}, expected {(int(np.asarray(theta_c).size),)}"
            )
    rows = measure_profiles(
        m, theta_c, phi_c, theta500_arcmin, x_edges, nest=nest, n_jobs=n_jobs
    )
    return _reduce_stack(rows, x_edges, y_norms)
