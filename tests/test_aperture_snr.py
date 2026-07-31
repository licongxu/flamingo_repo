"""Tests for map-aperture tSZ signal and noise calculations."""

import healpy as hp
import numpy as np
import pytest

from flamingo.aperture_snr import aperture_y500, fit_sigma_y500, sigma_y500_from_theta


def _write_noise(tmp_path, noise, skyfracs):
    noise_file = tmp_path / "noise.npy"
    sky_file = tmp_path / "sky.npy"
    np.save(noise_file, noise)
    np.save(sky_file, np.asarray(skyfracs, dtype=np.float64))
    return noise_file, sky_file


def test_fit_sigma_y500_uses_hmfast_sky_weighting(tmp_path):
    """Changing tile weights must change the fitted sky-average curve."""
    theta = np.geomspace(0.5, 32.0, 25)
    base = 2.0e-4 * theta**1.5
    noise_file, sky_file = _write_noise(
        tmp_path,
        {"immf6": {0: base, 1: 2.0 * base}},
        [1.0, 3.0],
    )

    coeff = fit_sigma_y500(noise_file, sky_file)
    actual = sigma_y500_from_theta(theta, coeff)

    assert np.allclose(actual, 1.75 * base, rtol=1e-12)


@pytest.mark.parametrize(
    ("noise", "skyfracs", "match"),
    [
        ({"other": {0: np.ones(25)}}, [1.0], "filter.*immf6"),
        ({"immf6": {0: np.r_[np.ones(24), 0.0]}}, [1.0], "positive"),
        ({"immf6": {0: np.ones(25), 1: np.ones(24)}}, [1.0, 1.0], "length"),
        ({"immf6": {2: np.ones(25)}}, [1.0, 1.0], "tile index"),
    ],
)
def test_fit_sigma_y500_rejects_invalid_noise_inputs(tmp_path, noise, skyfracs, match):
    """Malformed tile data must fail before producing a plausible polynomial."""
    noise_file, sky_file = _write_noise(tmp_path, noise, skyfracs)

    with pytest.raises(ValueError, match=match):
        fit_sigma_y500(noise_file, sky_file)


def test_aperture_y500_sums_ring_pixel_centres():
    """Changing a selected map pixel must change the integrated aperture signal."""
    nside = 8
    ymap = np.arange(hp.nside2npix(nside), dtype=np.float32) * 1e-8
    theta = np.array([1.0])
    phi = np.array([2.0])
    radius = np.array([0.2])
    pixels = hp.query_disc(nside, hp.ang2vec(theta[0], phi[0]), radius[0])
    arcmin2_per_sr = (180.0 * 60.0 / np.pi) ** 2
    expected = (
        ymap[pixels].sum(dtype=np.float64)
        * hp.nside2pixarea(nside)
        * arcmin2_per_sr
    )

    y500, npix = aperture_y500(ymap, theta, phi, radius)

    assert y500 == pytest.approx([expected])
    assert np.array_equal(npix, [len(pixels)])


def test_aperture_y500_accepts_an_empty_pixel_centre_aperture():
    """A subpixel aperture with no selected centre must return an explicit zero."""
    nside = 8
    theta = np.array([1.0])
    phi = np.array([2.0])
    radius = np.array([1e-10])
    assert hp.query_disc(nside, hp.ang2vec(theta[0], phi[0]), radius[0]).size == 0

    y500, npix = aperture_y500(
        np.ones(hp.nside2npix(nside)), theta, phi, radius
    )

    assert np.array_equal(y500, [0.0])
    assert np.array_equal(npix, [0])


@pytest.mark.parametrize(
    ("theta", "phi", "radius", "match"),
    [
        ([1.0, 1.2], [2.0], [0.2], "same shape"),
        ([np.nan], [2.0], [0.2], "finite"),
        ([1.0], [2.0], [0.0], "positive"),
    ],
)
def test_aperture_y500_rejects_invalid_geometry(theta, phi, radius, match):
    """Invalid geometry must not silently produce a catalogue measurement."""
    ymap = np.ones(hp.nside2npix(8))

    with pytest.raises(ValueError, match=match):
        aperture_y500(ymap, np.asarray(theta), np.asarray(phi), np.asarray(radius))
