"""Tests for the SZ scaling relation that defines the cluster selection."""
import numpy as np
import pytest

from flamingo.cnc import SZScaling, _a10_y0_reference
from flamingo.catalogue import D3A_COSMOLOGY

pytest.importorskip("hmfast")

from hmfast.halos import HaloModel  # noqa: E402


@pytest.fixture(scope="module")
def halo_model():
    return HaloModel(cosmology=D3A_COSMOLOGY)


def test_a10_reference_y0_scales_with_mass_and_redshift(halo_model):
    # y0 ~ P_500 * R_500c ~ [M^(2/3+0.12) E^(8/3)] * [M^(1/3) E^(-2/3)],
    # so it rises with mass and, being a surface brightness, also with E(z).
    m = np.array([1e14, 1e15])
    z = np.array([0.1, 1.0])
    y0 = np.asarray(_a10_y0_reference(halo_model, m, z, B=1.35))

    assert y0.shape == (2, 2)
    assert np.all(y0 > 0)
    assert np.all(y0[1] > y0[0])  # more massive halos are brighter
    assert np.all(y0[:, 1] > y0[:, 0])  # no dimming: y0 grows roughly as E(z)^2


def test_a10_reference_y0_suppressed_by_hydrostatic_bias(halo_model):
    m, z = np.array([5e14]), np.array([0.3])
    unbiased = _a10_y0_reference(halo_model, m, z, B=1.0)
    biased = _a10_y0_reference(halo_model, m, z, B=1.35)
    # Evaluating at M/B lowers P_500 by B^-(2/3 + 0.12) and nothing else.
    assert np.allclose(biased / unbiased, 1.35 ** -(2.0 / 3.0 + 0.12))


def _scaling(**kwargs):
    """A calibrated relation, or skip if the szifi noise curves are absent."""
    from paper_results import config

    if not (config.SIGMA_Y0_FILE.exists() and config.SKYFRACS_FILE.exists()):
        pytest.skip("szifi matched-filter noise curves not available")
    return SZScaling.calibrated(
        sigma_y0_file=config.SIGMA_Y0_FILE, skyfracs_file=config.SKYFRACS_FILE, **kwargs
    )


def test_scatter_is_deterministic_and_correctly_dispersed():
    sz = _scaling()
    index = np.arange(200_000, dtype=np.uint32)
    draw = sz.scatter(index)

    assert np.allclose(draw, sz.scatter(index))  # same labels -> same realisation
    assert np.allclose(draw[:10], sz.scatter(index[:10]))  # and independent of batching
    assert np.isclose(draw.std(), sz.sigma_lnY, rtol=0.02)
    assert abs(draw.mean()) < 0.01


def test_q_is_elementwise_and_monotonic_in_mass():
    sz = _scaling()
    m = np.array([1e14, 5e14, 1e15])
    z = np.full(3, 0.2)

    q = sz.q(m, z)
    assert q.shape == (3,)  # not the (3, 3) hmfast grid
    assert np.all(np.diff(q) > 0)

    # The scattered q is the mean q times exp(epsilon), object by object.
    index = np.array([7, 11, 13], dtype=np.uint32)
    assert np.allclose(sz.q(m, z, index=index), q * np.exp(sz.scatter(index)))


def test_theta_500_shrinks_with_redshift():
    sz = _scaling()
    z = np.array([0.1, 0.5, 1.5])
    theta = sz.theta_500_arcmin(np.full(3, 5e14), z)
    assert theta.shape == (3,)
    assert np.all(np.diff(theta) < 0)
