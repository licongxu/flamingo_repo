"""Pseudo-Cl power spectra of HEALPix maps via NaMaster.

This module needs ``pymaster`` (NaMaster), an optional dependency::

    pip install -e ".[powerspectra]"

It generalizes the masked-spectrum pipeline from the flamingo_l2p8_lc0 bundle:
apodize a binary mask, subtract the mask-weighted monopole, and return the
mask-decoupled binned bandpowers as ``D_ell = ell(ell+1) C_ell / 2 pi``.
"""
from __future__ import annotations

import numpy as np

try:
    import pymaster as nmt
except ImportError as exc:  # pragma: no cover - exercised only without pymaster
    raise ImportError(
        "flamingo.powerspectra requires NaMaster (pymaster). "
        'Install it with: pip install -e ".[powerspectra]"'
    ) from exc


def apodize(mask: np.ndarray, aperture_deg: float = 0.5, *, apotype: str = "C1") -> np.ndarray:
    """Apodize a binary mask (NaMaster ``mask_apodization``).

    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask in RING ordering (1 = keep, 0 = masked).
    aperture_deg : float, optional
        Apodization scale in degrees (default 0.5).
    apotype : str, optional
        NaMaster apodization type (``"C1"``, ``"C2"``, or ``"Smooth"``).

    Returns
    -------
    numpy.ndarray
        Apodized (smoothly tapered) mask.
    """
    return nmt.mask_apodization(mask, aperture_deg, apotype=apotype)


def decoupled_dl(
    m: np.ndarray,
    mask: np.ndarray | None,
    *,
    delta_ell: int = 30,
    lmax: int | None = None,
    lmax_cap: int = 6000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mask-decoupled binned auto-spectrum of a scalar map.

    Subtracts the mask-weighted monopole before estimating the pseudo-Cl, then
    decouples with the NaMaster workspace and bins linearly in ``ell``.

    Parameters
    ----------
    m : numpy.ndarray
        Scalar HEALPix map (RING), e.g. a Compton-y map.
    mask : numpy.ndarray or None
        Apodized mask. If ``None``, the full sky is used (mask of ones) and a
        simple-mean monopole is subtracted.
    delta_ell : int, optional
        Linear bandpower width (default 30).
    lmax : int, optional
        Maximum multipole. Defaults to ``min(3*nside - 1, lmax_cap)``.
    lmax_cap : int, optional
        Hard ceiling on ``lmax`` to bound NaMaster cost (default 6000).

    Returns
    -------
    ell_eff : numpy.ndarray
        Effective multipole of each bandpower.
    dl : numpy.ndarray
        ``D_ell = ell(ell+1) C_ell / 2 pi`` per bandpower.
    cl : numpy.ndarray
        Decoupled ``C_ell`` per bandpower.
    """
    nside = int(np.sqrt(m.size / 12.0))
    if lmax is None:
        lmax = int(min(3 * nside - 1, lmax_cap))
    bands = nmt.NmtBin.from_lmax_linear(lmax, delta_ell)

    if mask is None:
        mask = np.ones_like(m)
        monopole = float(m.mean())
    else:
        monopole = float(np.average(m, weights=mask))

    field = nmt.NmtField(mask, [m - monopole], lmax=lmax)
    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(field, field, bands)
    cl = workspace.decouple_cell(nmt.compute_coupled_cell(field, field))[0]

    ell_eff = bands.get_effective_ells()
    dl = ell_eff * (ell_eff + 1.0) * cl / (2.0 * np.pi)
    return ell_eff, dl, cl
