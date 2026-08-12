"""Bandpower conventions shared by FLAMINGO production pipelines."""
from __future__ import annotations

from pathlib import Path

import numpy as np


PLANCK_ELL_MIN = np.array(
    [9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835]
)
PLANCK_ELL_MAX = np.array(
    [12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835, 1085]
)
PLANCK_ELL_EFF = (PLANCK_ELL_MIN + PLANCK_ELL_MAX - 1) / 2


def bin_planck_dl(ell: np.ndarray, cl: np.ndarray) -> np.ndarray:
    """Average ``D_ell`` over the 18 inclusive Planck bins."""
    dl = ell * (ell + 1.0) * cl / (2.0 * np.pi)
    return np.array(
        [
            np.nanmean(dl[(ell >= lo) & (ell <= hi)])
            for lo, hi in zip(PLANCK_ELL_MIN, PLANCK_ELL_MAX, strict=True)
        ]
    )


def bin_log_dl(
    ell: np.ndarray,
    cl: np.ndarray,
    *,
    lmax: int = 10_000,
    dln_ell: float = 0.4,
    n_bins: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Average ``C_ell`` in log bins and return ``D_ell`` at their centres."""
    edges = lmax * np.exp(-n_bins * dln_ell) * np.exp(dln_ell * np.arange(n_bins + 1))
    centres = np.sqrt(edges[:-1] * edges[1:])
    binned_cl = np.empty(n_bins)
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        inside = (ell >= lo) & (ell <= hi if index == n_bins - 1 else ell < hi)
        if not np.any(inside):
            raise ValueError(f"log bin [{lo}, {hi}] is empty")
        binned_cl[index] = np.nanmean(cl[inside])
    return centres, centres * (centres + 1.0) * binned_cl / (2.0 * np.pi)


def write_bandpowers(path: str | Path, ell: np.ndarray, dl: np.ndarray, header: str) -> None:
    """Write the standard two-column FLAMINGO bandpower product."""
    np.savetxt(path, np.column_stack([ell, dl]), fmt="%.6e", header=header)
