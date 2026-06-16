"""Angular sizes, distances, and rotation-frame checks for FLAMINGO catalogues.

Cosmological distances come from **hmfast** (physical units: distances in Mpc,
``H(z)`` in km/s/Mpc), not astropy. The L2p8 y-map is stored in the yang26-
rotated frame, so cluster positions must be sampled with the rotated columns
(``theta_rot_rad``, ``phi_rot_rad``); :func:`rotation_sanity` checks that.

Importing this module constructs the FLAMINGO fiducial (D3A) cosmology, which
loads hmfast emulator assets as a side effect.
"""
from __future__ import annotations

import healpy as hp
import numpy as np
from hmfast import Cosmology

# FLAMINGO fiducial cosmology (D3A): Omega_m = 0.306, Omega_b = 0.0486,
# h = 0.681, one massive neutrino m_ncdm = 0.06 eV. hmfast takes *physical*
# densities omega_x = Omega_x h^2.
_H0 = 68.1
_h = _H0 / 100.0
_OMEGA_M = 0.306
_OMEGA_B = 0.0486
_M_NCDM = 0.06
_omega_b = _OMEGA_B * _h ** 2
_omega_nu = _M_NCDM / 93.14  # standard m_nu -> omega_nu approximation
_omega_cdm = _OMEGA_M * _h ** 2 - _omega_b - _omega_nu

D3A_COSMOLOGY = Cosmology(H0=_H0, omega_cdm=_omega_cdm, omega_b=_omega_b, m_ncdm=_M_NCDM)


def angular_diameter_distance(z: np.ndarray, *, cosmology: Cosmology = D3A_COSMOLOGY) -> np.ndarray:
    """Proper angular-diameter distance ``D_A(z)`` in Mpc (physical units).

    Parameters
    ----------
    z : numpy.ndarray
        Redshift.
    cosmology : hmfast.Cosmology, optional
        Cosmology to evaluate (default FLAMINGO D3A).

    Returns
    -------
    numpy.ndarray
        ``D_A(z)`` in proper Mpc.
    """
    return np.asarray(cosmology.angular_diameter_distance(np.asarray(z, dtype=float)))


def efunc(z: np.ndarray, *, cosmology: Cosmology = D3A_COSMOLOGY) -> np.ndarray:
    """Dimensionless expansion rate ``E(z) = H(z) / H_0``.

    Parameters
    ----------
    z : numpy.ndarray
        Redshift.
    cosmology : hmfast.Cosmology, optional
        Cosmology to evaluate (default FLAMINGO D3A).

    Returns
    -------
    numpy.ndarray
        ``E(z)``.
    """
    z = np.asarray(z, dtype=float)
    Hz = np.asarray(cosmology.hubble_parameter(z))
    H0 = float(np.asarray(cosmology.hubble_parameter(0.0)))
    return Hz / H0


def theta_500(
    R_500c_Mpc: np.ndarray,
    z: np.ndarray,
    *,
    cosmology: Cosmology = D3A_COSMOLOGY,
) -> np.ndarray:
    """Angular size ``theta_500 = R_500c / D_A(z)`` in radians.

    Parameters
    ----------
    R_500c_Mpc : numpy.ndarray
        Cluster radius ``R_500c`` in proper Mpc.
    z : numpy.ndarray
        Redshift.
    cosmology : hmfast.Cosmology, optional
        Cosmology for the angular-diameter distance (default FLAMINGO D3A).

    Returns
    -------
    numpy.ndarray
        Angular scale ``theta_500`` in radians.
    """
    D_A = angular_diameter_distance(z, cosmology=cosmology)
    return np.asarray(R_500c_Mpc, dtype=float) / D_A


def rotation_sanity(
    m: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    *,
    seed: int = 0,
    nest: bool = False,
) -> dict[str, float]:
    """Check that catalogue positions land on bright y-peaks, not random sky.

    Compares the mean map value at the supplied positions to the mean at the
    same number of random pixels. For a correctly rotated catalogue the ratio
    is large (clusters sit on tSZ peaks).

    Parameters
    ----------
    m : numpy.ndarray
        HEALPix map.
    theta, phi : numpy.ndarray
        Candidate cluster positions in radians.
    seed : int, optional
        Seed for the random comparison pixels.
    nest : bool, optional
        Pixel ordering of ``m``.

    Returns
    -------
    dict
        ``{"mean_at_positions", "mean_at_random", "ratio"}``.
    """
    nside = hp.npix2nside(m.size)
    pix = hp.ang2pix(nside, np.asarray(theta, float), np.asarray(phi, float), nest=nest)
    rng = np.random.default_rng(seed)
    pix_rnd = rng.integers(0, m.size, pix.size)

    mean_pos = float(m[pix].mean())
    mean_rnd = float(m[pix_rnd].mean())
    return {
        "mean_at_positions": mean_pos,
        "mean_at_random": mean_rnd,
        "ratio": mean_pos / mean_rnd if mean_rnd != 0 else np.inf,
    }
