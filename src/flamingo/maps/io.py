"""Reading and writing FLAMINGO HEALPix maps."""
from __future__ import annotations

from pathlib import Path

import healpy as hp
import numpy as np


def read_map(path: str | Path, *, field: int = 0, nest: bool = False) -> np.ndarray:
    """Read a single-field HEALPix map from a FITS file.

    Parameters
    ----------
    path : str or Path
        FITS file written in HEALPix format (e.g. a Compton-y map).
    field : int, optional
        Which map column to read (default 0).
    nest : bool, optional
        Return the map in NESTED ordering if ``True``; RING otherwise. The
        FLAMINGO products are stored in RING.

    Returns
    -------
    numpy.ndarray
        1-D array of length ``12 * nside**2``.
    """
    return hp.read_map(str(path), field=field, nest=nest)


def write_map(path: str | Path, m: np.ndarray, *, nest: bool = False, overwrite: bool = True) -> None:
    """Write a HEALPix map to a FITS file.

    Parameters
    ----------
    path : str or Path
        Output FITS path.
    m : numpy.ndarray
        Map to write; its length must be a valid HEALPix size.
    nest : bool, optional
        Ordering of ``m`` (default RING).
    overwrite : bool, optional
        Overwrite an existing file (default ``True``).
    """
    hp.write_map(str(path), m, nest=nest, overwrite=overwrite)


def nside_of(m: np.ndarray) -> int:
    """Return the ``nside`` of a HEALPix map from its length."""
    return hp.npix2nside(np.asarray(m).size)
