"""Flat-sky FFT power spectrum of a pixell stamp.

This module needs ``pixell``, an optional dependency::

    pip install -e ".[stamps]"

The estimator is the standard flat-sky one: with ``y_ell = Omega_pix * FFT``
(units sr), the auto-spectrum is ``C_ell = |y_ell|^2 / Omega_patch``,
azimuthally averaged in linear ``ell`` bins on the ``enmap.modlmap`` grid.
"""
from __future__ import annotations

import numpy as np

try:
    from pixell import enmap
except ImportError as exc:  # pragma: no cover - exercised only without pixell
    raise ImportError(
        "flamingo.powerspectra.flat requires pixell. "
        'Install it with: pip install -e ".[stamps]"'
    ) from exc


def stamp_cl(
    stamp,
    *,
    delta_ell: int = 200,
    lmax: float | None = None,
    subtract_mean: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally binned auto-spectrum of a single flat-sky stamp.

    Parameters
    ----------
    stamp : pixell.enmap.ndmap
        Square stamp, e.g. from :func:`flamingo.maps.stamps.gnomonic_stamp`.
    delta_ell : int, optional
        Linear ``ell`` bin width (default 200; flat-sky ``ell`` modes are
        sparse at low ``ell`` for small stamps).
    lmax : float, optional
        Discard modes above this ``ell`` (default: keep all, up to the grid
        Nyquist frequency).
    subtract_mean : bool, optional
        Subtract the stamp mean before transforming (default ``True``).

    Returns
    -------
    ell_eff : numpy.ndarray
        Mean ``|ell|`` of the modes in each bin.
    cl : numpy.ndarray
        ``C_ell`` per bin, same units as ``stamp**2`` times steradian.
    """
    lmap = np.asarray(enmap.modlmap(stamp.shape, stamp.wcs))
    omega_pix = float(stamp.pixsize())
    omega_patch = omega_pix * stamp.size

    data = np.asarray(stamp, dtype=np.float64)
    if subtract_mean:
        data = data - data.mean()
    y_ell = omega_pix * np.fft.fft2(data)
    p2d = np.abs(y_ell) ** 2 / omega_patch

    if lmax is None:
        lmax = lmap.max()
    keep = (lmap > 0) & (lmap <= lmax)
    idx = (lmap[keep] // delta_ell).astype(int)
    counts = np.bincount(idx)
    good = counts > 0
    ell_eff = (np.bincount(idx, weights=lmap[keep]) / np.where(good, counts, 1))[good]
    cl = (np.bincount(idx, weights=p2d[keep]) / np.where(good, counts, 1))[good]
    return ell_eff, cl
