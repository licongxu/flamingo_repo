"""Tests for the healpy full-sky spectra and the pixell stamp pipeline."""
import healpy as hp
import numpy as np
import pytest

from flamingo.powerspectra import bin_cl, full_sky_cl

NSIDE = 64


def test_full_sky_cl_white_noise_is_flat():
    # White noise of variance sigma^2 has C_ell = sigma^2 * Omega_pix.
    rng = np.random.default_rng(0)
    sigma = 1.0
    m = rng.normal(0.0, sigma, hp.nside2npix(NSIDE))
    ell, cl = full_sky_cl(m)
    expected = sigma**2 * hp.nside2pixarea(NSIDE)
    assert np.allclose(cl[2:].mean(), expected, rtol=0.05)


def test_bin_cl_recovers_flat_spectrum():
    ell = np.arange(300)
    cl = np.full(ell.size, 3.0)
    ell_eff, dl, cl_b = bin_cl(ell, cl, delta_ell=30)
    assert np.allclose(cl_b, 3.0)
    # dl is the bin-average of the per-ell D_ell.
    lf = ell[ell >= 2].astype(float)
    idx = ((lf - 2) // 30).astype(int)
    expected_dl = np.bincount(idx, weights=lf * (lf + 1.0) * 3.0 / (2.0 * np.pi)) / np.bincount(idx)
    assert np.allclose(dl, expected_dl)
    assert ell_eff.min() >= 2


pixell = pytest.importorskip("pixell")


def test_gnomonic_stamp_of_constant_map_is_constant():
    from flamingo.maps.stamps import gnomonic_stamp, zero_outside_aperture

    m = np.full(hp.nside2npix(NSIDE), 7.0)
    stamp = gnomonic_stamp(m, lon_deg=30.0, lat_deg=45.0,
                           width_rad=np.radians(2.0), res_rad=np.radians(0.02))
    assert stamp.shape[0] == stamp.shape[1]
    assert np.allclose(np.asarray(stamp), 7.0)

    r_ap = np.radians(0.5)
    cut = zero_outside_aperture(stamp, r_ap)
    assert np.asarray(cut).min() == 0.0  # corners zeroed
    assert np.isclose(np.asarray(cut).max(), 7.0)  # centre untouched


def test_stamp_cl_white_noise_amplitude():
    # Uncorrelated pixel noise of variance sigma^2 gives C_ell = sigma^2 * Omega_pix.
    from pixell import enmap
    from flamingo.powerspectra.flat import stamp_cl

    res = np.radians(0.01)
    shape, wcs = enmap.geometry(pos=(0.0, 0.0), shape=(128, 128), res=res, proj="tan")
    rng = np.random.default_rng(1)
    stamp = enmap.ndmap(rng.normal(0.0, 1.0, shape), wcs)

    ell_eff, cl = stamp_cl(stamp, delta_ell=500)
    omega_pix = float(stamp.pixsize())
    assert np.all(np.isfinite(cl))
    assert np.allclose(cl.mean(), omega_pix, rtol=0.1)
