"""Tests for HEALPix sampling and disc masking on a small synthetic map."""
import healpy as hp
import numpy as np

from flamingo.masking import disc_mask, fsky
from flamingo.maps import nside_of, sample_at, sample_neighbour_max

NSIDE = 64


def _peaked_map(theta, phi):
    """Map that is 1 in the pixel at (theta, phi), 0 elsewhere."""
    m = np.zeros(hp.nside2npix(NSIDE))
    m[hp.ang2pix(NSIDE, theta, phi)] = 1.0
    return m


def test_nside_of_roundtrip():
    m = np.zeros(hp.nside2npix(NSIDE))
    assert nside_of(m) == NSIDE


def test_sample_at_hits_peak():
    theta, phi = np.array([1.0]), np.array([2.0])
    m = _peaked_map(theta[0], phi[0])
    assert np.allclose(sample_at(m, theta, phi), 1.0)


def test_neighbour_max_recovers_offset_peak():
    # Place the peak in a neighbour of the queried pixel; neighbour-max finds it.
    theta, phi = 1.0, 2.0
    pix = hp.ang2pix(NSIDE, theta, phi)
    neigh = hp.get_all_neighbours(NSIDE, theta, phi)
    neigh = neigh[neigh >= 0]
    m = np.zeros(hp.nside2npix(NSIDE))
    m[neigh[0]] = 1.0
    vmax, vcen = sample_neighbour_max(m, np.array([theta]), np.array([phi]))
    assert np.allclose(vmax, 1.0)
    assert np.allclose(vcen, 0.0)


def test_disc_mask_zeros_a_region_and_fsky():
    theta, phi = np.array([np.pi / 2]), np.array([0.0])
    radius = np.array([np.radians(5.0)])  # 5 deg disc
    mask = disc_mask(NSIDE, theta, phi, radius)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert mask.min() == 0.0  # something was masked
    assert 0.0 < fsky(mask) < 1.0
