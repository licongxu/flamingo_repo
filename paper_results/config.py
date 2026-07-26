"""Single source of truth for inputs, analysis settings, and output locations.

Nothing here is computed; every constant that any figure or data product
depends on is declared once in this module so the pipeline is reproducible from
one file.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Locations --------------------------------------------------------------

#: This checkout (the ``paper-results`` working tree).
REPO = Path(__file__).resolve().parents[1]

#: Shared FLAMINGO data tree (large inputs and caches; not under version control).
FLAMINGO_ROOT = Path(os.environ.get("FLAMINGO_ROOT", "/scratch/scratch-lxu/flamingo_repo"))
FLAMINGO_DATA = FLAMINGO_ROOT / "data"

#: L1_m9 lightcone-0 Compton-y maps and halo catalogues, one per feedback model.
MAP_DIR = Path(os.environ.get("L1M9_MAP_DIR", "/rds/rds-lxu/flamingo/L1_m9/maps"))
CAT_DIR = Path(os.environ.get("L1M9_CAT_DIR", "/rds/rds-lxu/flamingo/L1_m9/catalogues"))

#: Planck-like matched-filter noise curves (szifi).
SIGMA_Y0_FILE = FLAMINGO_DATA / "noise" / "sigma_dict_szifi.npy"
SKYFRACS_FILE = FLAMINGO_DATA / "noise" / "skyfracs_szifi_cosmology.npy"

#: Small, version-controlled outputs: bandpowers and summary tables.
RESULTS = REPO / "results"
BANDPOWERS = RESULTS / "bandpowers"

#: Large intermediates (per-halo q catalogues); regenerable, not committed.
CACHE = FLAMINGO_DATA / "paper_results_cache"
QCAT_CACHE = CACHE / "qcat"

#: Figures.
FIGURES = REPO / "figures"


# --- Inputs -----------------------------------------------------------------

#: Fiducial FLAMINGO L1_m9 run.
FIDUCIAL = "L1_m9"

#: All L1_m9 feedback variants. Order is for legends only: the gas-fraction
#: series from weakest to strongest feedback, then the stellar-mass and jet runs.
VARIANTS = [
    "fgas+2sigma",
    "L1_m9",
    "fgas-2sigma",
    "fgas-4sigma",
    "fgas-8sigma",
    "Mstar-1sigma",
    "Mstar-1sigma_fgas-4sigma",
    "Jet",
    "Jet_fgas-4sigma",
]

#: Display labels for figures.
LABELS = {
    "L1_m9": "fiducial (L1_m9)",
    "fgas+2sigma": r"$f_{\rm gas}+2\sigma$",
    "fgas-2sigma": r"$f_{\rm gas}-2\sigma$",
    "fgas-4sigma": r"$f_{\rm gas}-4\sigma$",
    "fgas-8sigma": r"$f_{\rm gas}-8\sigma$",
    "Mstar-1sigma": r"$M_*-1\sigma$",
    "Mstar-1sigma_fgas-4sigma": r"$M_*-1\sigma$, $f_{\rm gas}-4\sigma$",
    "Jet": r"Jet",
    "Jet_fgas-4sigma": r"Jet, $f_{\rm gas}-4\sigma$",
}

#: Colours for the feedback comparison figures.
COLORS = {
    "L1_m9": "k",
    "fgas+2sigma": "#d62728",
    "fgas-2sigma": "#ff7f0e",
    "fgas-4sigma": "#bcbd22",
    "fgas-8sigma": "#2ca02c",
    "Mstar-1sigma": "#17becf",
    "Mstar-1sigma_fgas-4sigma": "#1f77b4",
    "Jet": "#9467bd",
    "Jet_fgas-4sigma": "#e377c2",
}


def map_path(variant: str) -> Path:
    """Path to the ``nside=4096`` unlensed Compton-y map of a feedback variant."""
    return MAP_DIR / f"y_unlensed_{variant}_lc0_nside4096.fits"


def catalogue_path(variant: str) -> Path:
    """Path to the ``M_500c > 5e13 Msun``, ``z < 3`` yang26-rotated halo catalogue."""
    return CAT_DIR / f"halo_catalogue_M500c_5e13_zlt3_{variant}_yang26rot_qfrommz.csv"


def qcat_path(variant: str) -> Path:
    """Path to the cached compact ``q`` catalogue produced by :mod:`~paper_results.compute_q`."""
    return QCAT_CACHE / f"{variant}.npz"


# --- Analysis settings ------------------------------------------------------

#: Detection significance thresholds at which clusters are masked.
Q_CUTS = [50.0, 20.0, 10.0, 5.0, 1.0]

#: Filename tag per cut, plus the unmasked case.
CUT_TAGS = ["qgt50", "qgt20", "qgt10", "qgt5", "qgt1"]
FULLSKY_TAG = "fullsky"

#: Masking radius in units of the cluster angular size ``theta_500``.
R_MASK = 5.0

#: NaMaster C1 apodization scale of the binary mask, in degrees.
APOD_DEG = 0.5

#: Pseudo-Cl estimation: linear bandpowers of width ``DELTA_ELL`` up to ``LMAX``.
LMAX = 6000
DELTA_ELL = 30

#: Lowest column-mass kept by the catalogues (documented, not applied here).
M_MIN_MSUN = 5.0e13
