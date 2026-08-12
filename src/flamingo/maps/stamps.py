"""Flat-sky stamps cut from HEALPix maps via pixell.

This module needs ``pixell``, an optional dependency::

    pip install -e ".[stamps]"

A stamp is a gnomonic (tangent-plane) projection of the HEALPix map centred on
a sky position, returned as a :class:`pixell.enmap.ndmap` so its WCS travels
with the pixel data. Stamps feed the flat-sky FFT power spectrum in
:mod:`flamingo.powerspectra.flat`.
"""
from __future__ import annotations

import numpy as np

try:
    from pixell import enmap, reproject
except ImportError as exc:  # pragma: no cover - exercised only without pixell
    raise ImportError(
        "flamingo.maps.stamps requires pixell. "
        'Install it with: pip install -e ".[stamps]"'
    ) from exc


def gnomonic_stamp(
    m: np.ndarray,
    lon_deg: float,
    lat_deg: float,
    *,
    width_rad: float,
    res_rad: float,
    order: int = 1,
):
    """Cut a square tangent-plane stamp from a HEALPix map.

    Parameters
    ----------
    m : numpy.ndarray
        HEALPix map in RING ordering.
    lon_deg, lat_deg : float
        Stamp centre in degrees (longitude, latitude), in the same rotation
        frame as ``m``.
    width_rad : float
        Side length of the stamp in radians.
    res_rad : float
        Pixel size of the stamp in radians; choose it to oversample the
        HEALPix resolution (a few times smaller than the HEALPix pixel scale).
    order : int, optional
        Spline interpolation order for the HEALPix -> flat resampling
        (default 1, bilinear).

    Returns
    -------
    pixell.enmap.ndmap
        Square stamp of shape ``(npix, npix)`` with an attached WCS, where
        ``npix = ceil(width_rad / res_rad)`` rounded up to even.
    """
    npix = int(np.ceil(width_rad / res_rad))
    npix += npix % 2
    dec, ra = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    shape, wcs = enmap.geometry(pos=(dec, ra), shape=(npix, npix), res=res_rad, proj="tan")
    stamp = reproject.healpix2map(m, shape, wcs, rot=None, method="spline", order=order)
    return enmap.ndmap(np.asarray(stamp, dtype=np.float64), wcs)


def zero_outside_aperture(stamp, radius_rad: float):
    """Zero all stamp pixels beyond ``radius_rad`` from the stamp centre.

    Parameters
    ----------
    stamp : pixell.enmap.ndmap
        Stamp from :func:`gnomonic_stamp` (or any enmap).
    radius_rad : float
        Hard circular aperture radius in radians.

    Returns
    -------
    pixell.enmap.ndmap
        Copy of ``stamp`` with pixels outside the aperture set to zero.
    """
    r = np.asarray(enmap.modrmap(stamp.shape, stamp.wcs))
    out = stamp.copy()
    out[r > radius_rad] = 0.0
    return out
