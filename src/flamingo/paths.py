"""Resolve paths to the bundled FLAMINGO data products.

The repository ships a small ``data/`` tree with one map and a few catalogues::

    data/
      hydro_L2p8m9/map/y_unlensed_L2p8_m9_lc0.fits
      hydro_L2p8m9/catalogue/halo_catalogue_M500c_5e13_zlt3_soap_snr_d3a_yang26rot.csv
      dmo_L2p8m9/catalogue/halo_catalogue_M500c_5e13_dmo_snaps.csv

Set ``FLAMINGO_ROOT`` to point at a different copy of this tree (for example a
full run on ``/scratch`` or ``/rds``); otherwise the in-repo ``data/`` is used.
"""
from __future__ import annotations

import os
from pathlib import Path

# This file is ``<repo>/src/flamingo/paths.py``; the repo root is three up.
_REPO_ROOT = Path(__file__).resolve().parents[2]

ROOT = Path(os.environ.get("FLAMINGO_ROOT", _REPO_ROOT)).resolve()
DATA = ROOT / "data"

HYDRO = DATA / "hydro_L2p8m9"
DMO = DATA / "dmo_L2p8m9"

# Default products shipped with the repo.
HYDRO_MAP = HYDRO / "map" / "y_unlensed_L2p8_m9_lc0.fits"
HYDRO_CATALOGUE = HYDRO / "catalogue" / "halo_catalogue_M500c_5e13_zlt3_soap_snr_d3a_yang26rot.csv"
DMO_CATALOGUE = DMO / "catalogue" / "halo_catalogue_M500c_5e13_dmo_snaps.csv"


def require(path: Path) -> Path:
    """Return ``path`` if it exists, else raise a helpful ``FileNotFoundError``.

    Parameters
    ----------
    path : Path
        Candidate data path, typically one of the module-level constants.

    Returns
    -------
    Path
        The same path, guaranteed to exist on disk.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Set FLAMINGO_ROOT to a directory containing the "
            f"expected data/ tree (current ROOT={ROOT})."
        )
    return path
