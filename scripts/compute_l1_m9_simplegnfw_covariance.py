"""Full-sky L1_m9 theory covariance from simple GNFW hmfast theory."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from hmfast.halos import HaloModel
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.profiles import GNFWPressureProfile
from hmfast.tracers import tSZTracer

from flamingo.catalogue.frame import D3A_COSMOLOGY


REPO = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO / "data_paper" / "covariance"

ELL_MIN = np.array(
    [9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835],
    dtype=int,
)
ELL_MAX = np.array(
    [12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835, 1085],
    dtype=int,
)
ELL_EFF = np.array(
    [
        10.0,
        13.5,
        18.0,
        23.5,
        30.5,
        40.0,
        52.5,
        68.5,
        89.5,
        117.0,
        152.5,
        198.0,
        257.5,
        335.5,
        436.5,
        567.5,
        738.0,
        959.5,
    ],
)
_BIN_WIDTH_MAX = int(np.max(ELL_MAX - ELL_MIN))
_ELL_INTEGER = ELL_MIN[:, None] + np.arange(_BIN_WIDTH_MAX)[None, :]
_ELL_MASK = _ELL_INTEGER < ELL_MAX[:, None]


def bin_dl(ell: np.ndarray, dl: np.ndarray) -> np.ndarray:
    """Uniformly average a smooth D_ell curve over the 18 integer-ell bins."""
    ell = np.asarray(ell, dtype=float)
    dl = np.asarray(dl, dtype=float)
    sampled = np.empty(_ELL_INTEGER.shape, dtype=float)
    for index in range(18):
        sampled[index] = np.interp(
            np.log(_ELL_INTEGER[index]),
            np.log(ell),
            dl,
        )
    return np.sum(sampled * _ELL_MASK, axis=1) / np.sum(_ELL_MASK, axis=1)


def bin_trispectrum(
    ell: np.ndarray,
    trispectrum_cl: np.ndarray,
) -> np.ndarray:
    """Convert T_ell,ell' from C units and average over both bin axes."""
    ell = np.asarray(ell, dtype=float)
    trispectrum_cl = np.asarray(trispectrum_cl, dtype=float)
    log_ell = np.log(ell)
    log_integer = np.log(_ELL_INTEGER.astype(float))

    first_axis = np.empty((*_ELL_INTEGER.shape, ell.size), dtype=float)
    for column in range(ell.size):
        first_axis[..., column] = np.interp(
            log_integer, log_ell, trispectrum_cl[:, column]
        )

    interpolated = np.empty((*_ELL_INTEGER.shape, *_ELL_INTEGER.shape))
    for first_bin in range(18):
        for first_ell in range(_BIN_WIDTH_MAX):
            interpolated[first_bin, first_ell] = np.interp(
                log_integer,
                log_ell,
                first_axis[first_bin, first_ell],
            )

    dl_factor = _ELL_INTEGER * (_ELL_INTEGER + 1.0) / (2.0 * np.pi)
    trispectrum_dl = (
        interpolated
        * dl_factor[:, :, None, None]
        * dl_factor[None, None, :, :]
    )
    pair_mask = _ELL_MASK[:, :, None, None] * _ELL_MASK[None, None, :, :]
    return np.sum(trispectrum_dl * pair_mask, axis=(1, 3)) / np.sum(
        pair_mask, axis=(1, 3)
    )


def gaussian_covariance(
    dl_binned: np.ndarray,
    fsky: float = 1.0,
) -> np.ndarray:
    """Return the diagonal binned Knox covariance."""
    dl_binned = np.asarray(dl_binned, dtype=float)
    diagonal = 2.0 * dl_binned**2 / (
        (2.0 * ELL_EFF + 1.0) * (ELL_MAX - ELL_MIN) * fsky
    )
    return np.diag(diagonal)


def assemble_covariance(
    cov_gaussian: np.ndarray,
    trispectrum_binned: np.ndarray,
    fsky: float = 1.0,
) -> np.ndarray:
    """Combine Gaussian and connected one-halo terms."""
    return (
        np.asarray(cov_gaussian)
        + np.asarray(trispectrum_binned) / (4.0 * np.pi * fsky)
    )
