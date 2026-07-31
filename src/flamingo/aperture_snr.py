"""Empirical cylindrical tSZ aperture signals and matched-filter noise."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import healpy as hp
import numpy as np
import pandas as pd

from .catalogue import theta_500


APERTURE_COLUMNS = [
    "theta_500_arcmin",
    "Y_500cyl_arcmin2",
    "sigma_Y500_arcmin2",
    "npix_in_aperture",
    "q_from_aperture",
]

_REQUIRED_CATALOGUE_COLUMNS = [
    "z",
    "R_500c_Mpc",
    "theta_rot_rad",
    "phi_rot_rad",
    "q_from_mz",
]


def fit_sigma_y500(
    noise_file: str | Path,
    skyfracs_file: str | Path,
    *,
    filter_name: str = "immf6",
    theta_min_arcmin: float = 0.5,
    theta_max_arcmin: float = 32.0,
    poly_deg: int = 3,
) -> np.ndarray:
    """Fit ``log sigma_Y500`` against ``log theta_500`` after tile averaging."""
    noise = np.load(noise_file, allow_pickle=True).item()
    if filter_name not in noise:
        raise ValueError(f"noise file has no filter {filter_name!r}")
    tiles = noise[filter_name]
    if not tiles:
        raise ValueError(f"filter {filter_name!r} has no tile curves")

    skyfracs = np.asarray(np.load(skyfracs_file), dtype=np.float64).ravel()
    lengths = {np.asarray(curve).size for curve in tiles.values()}
    if len(lengths) != 1:
        raise ValueError(f"tile curves have inconsistent lengths: {sorted(lengths)}")
    n_theta = lengths.pop()

    weighted = np.zeros(n_theta, dtype=np.float64)
    weight_sum = 0.0
    for tile_idx, curve in tiles.items():
        index = int(tile_idx)
        if index < 0 or index >= skyfracs.size:
            raise ValueError(f"tile index {index} is outside sky-fraction array")
        values = np.asarray(curve, dtype=np.float64)
        if values.shape != (n_theta,):
            raise ValueError(f"tile {index} curve has invalid shape {values.shape}")
        if not np.all(np.isfinite(values)) or not np.all(values > 0.0):
            raise ValueError(f"tile {index} curve must be finite and positive")
        weight = float(skyfracs[index])
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"tile {index} has invalid sky-fraction weight {weight}")
        weighted += weight * values
        weight_sum += weight

    if weight_sum <= 0.0:
        raise ValueError("sum of selected sky-fraction weights must be positive")
    theta = np.geomspace(theta_min_arcmin, theta_max_arcmin, n_theta)
    sigma_skyavg = weighted / weight_sum
    return np.polyfit(np.log(theta), np.log(sigma_skyavg), deg=poly_deg)


def sigma_y500_from_theta(theta_500_arcmin: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """Evaluate a log-log polynomial fit to ``sigma_Y500(theta_500)``."""
    theta = np.asarray(theta_500_arcmin, dtype=np.float64)
    if not np.all(np.isfinite(theta)) or not np.all(theta > 0.0):
        raise ValueError("theta_500_arcmin must be finite and positive")
    values = np.exp(np.polyval(np.asarray(coeff, dtype=np.float64), np.log(theta)))
    if not np.all(np.isfinite(values)) or not np.all(values > 0.0):
        raise ValueError("fitted sigma_Y500 must be finite and positive")
    return values


def aperture_y500(
    ymap: np.ndarray,
    theta_rad: np.ndarray,
    phi_rad: np.ndarray,
    theta_500_rad: np.ndarray,
    *,
    nest: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum raw Compton-y map pixels whose centres lie inside each aperture."""
    ymap = np.asarray(ymap)
    if ymap.ndim != 1:
        raise ValueError(f"ymap must be one-dimensional, got shape {ymap.shape}")
    nside = hp.npix2nside(ymap.size)

    theta = np.asarray(theta_rad, dtype=np.float64)
    phi = np.asarray(phi_rad, dtype=np.float64)
    radius = np.asarray(theta_500_rad, dtype=np.float64)
    if theta.shape != phi.shape or theta.shape != radius.shape:
        raise ValueError("theta, phi, and theta_500 must have the same shape")
    if theta.ndim != 1:
        raise ValueError("aperture geometry arrays must be one-dimensional")
    if not all(np.all(np.isfinite(values)) for values in (theta, phi, radius)):
        raise ValueError("aperture geometry must be finite")
    if not np.all(radius > 0.0):
        raise ValueError("theta_500 radii must be positive")

    omega_pix = hp.nside2pixarea(nside)
    arcmin2_per_sr = (180.0 * 60.0 / np.pi) ** 2
    y500 = np.empty(theta.size, dtype=np.float64)
    pixel_counts = np.empty(theta.size, dtype=np.int32)
    for index, (th, ph, aperture_radius) in enumerate(zip(theta, phi, radius)):
        pixels = hp.query_disc(
            nside,
            hp.ang2vec(th, ph),
            float(aperture_radius),
            nest=nest,
        )
        pixel_counts[index] = pixels.size
        y500[index] = (
            np.sum(ymap[pixels], dtype=np.float64) * omega_pix * arcmin2_per_sr
        )

    if not np.all(np.isfinite(y500)):
        raise ValueError("integrated Y_500 apertures must be finite")
    return y500, pixel_counts


def catalogue_chunk_to_qfrommap(
    frame: pd.DataFrame,
    ymap: np.ndarray,
    noise_coeff: np.ndarray,
    *,
    theta500_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = theta_500,
) -> pd.DataFrame:
    """Replace a chunk's parametric q with raw map-aperture observables."""
    for column in _REQUIRED_CATALOGUE_COLUMNS:
        if column not in frame.columns:
            raise ValueError(f"missing column {column}")
        values = frame[column].to_numpy(np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"column {column} must be finite")

    theta_500_rad = np.asarray(
        theta500_fn(
            frame["R_500c_Mpc"].to_numpy(np.float64),
            frame["z"].to_numpy(np.float64),
        ),
        dtype=np.float64,
    )
    if theta_500_rad.shape != (len(frame),):
        raise ValueError(
            f"theta_500 has shape {theta_500_rad.shape}, expected {(len(frame),)}"
        )
    if not np.all(np.isfinite(theta_500_rad)) or not np.all(theta_500_rad > 0.0):
        raise ValueError("theta_500 must be finite and positive")

    theta_500_arcmin = np.rad2deg(theta_500_rad) * 60.0
    y500, pixel_counts = aperture_y500(
        ymap,
        frame["theta_rot_rad"].to_numpy(np.float64),
        frame["phi_rot_rad"].to_numpy(np.float64),
        theta_500_rad,
    )
    sigma = sigma_y500_from_theta(theta_500_arcmin, noise_coeff)
    q = y500 / sigma
    if not np.all(np.isfinite(q)):
        raise ValueError("q_from_aperture must be finite")

    output = frame.drop(columns="q_from_mz").copy()
    output["theta_500_arcmin"] = theta_500_arcmin
    output["Y_500cyl_arcmin2"] = y500
    output["sigma_Y500_arcmin2"] = sigma
    output["npix_in_aperture"] = pixel_counts
    output["q_from_aperture"] = q
    return output
