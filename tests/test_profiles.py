"""Tests for the GNFW pressure model and its line-of-sight projection."""
import healpy as hp
import jax.numpy as jnp
import numpy as np

from flamingo.profiles import (
    A10_PARAMS,
    gnfw,
    normalized_profile,
    projected_shape,
    stack_normalized,
    y500_normalized_projected,
)


def test_gnfw_shape_and_finite():
    x = jnp.logspace(-2, 1, 50)
    p = gnfw(x)
    assert p.shape == x.shape
    assert bool(jnp.all(jnp.isfinite(p)))


def test_gnfw_x64_enabled():
    # jaxconfig should have promoted the default float to float64.
    assert gnfw(jnp.array([1.0])).dtype == jnp.float64


def test_gnfw_default_is_arnaud():
    x = jnp.array([0.3, 1.0, 3.0])
    explicit = gnfw(x, **A10_PARAMS)
    assert bool(jnp.allclose(gnfw(x), explicit, rtol=1e-12))


def test_gnfw_monotonic_decreasing():
    x = jnp.linspace(0.05, 5.0, 100)
    p = gnfw(x)
    assert bool(jnp.all(jnp.diff(p) < 0.0))


def test_projected_shape_normalized_and_decreasing():
    b = jnp.linspace(0.0, 3.0, 40)
    y = projected_shape(b)
    assert y.shape == b.shape
    assert bool(jnp.all(jnp.isfinite(y)))
    # Normalised to unity at the centre and falling outward.
    assert np.isclose(float(y[0]), 1.0, atol=1e-3)
    assert bool(jnp.all(jnp.diff(y) < 0.0))


def test_projected_shape_jit():
    import jax

    f = jax.jit(projected_shape)
    b = jnp.linspace(0.0, 2.0, 16)
    assert bool(jnp.allclose(f(b), projected_shape(b), rtol=1e-10))


def test_y500_normalized_projected_has_unit_aperture_mean():
    x = jnp.linspace(0.0, 1.0, 400)
    f = y500_normalized_projected(x)
    aperture_mean = float(2.0 * jnp.trapezoid(x * f, x))
    assert np.isclose(aperture_mean, 1.0, atol=2e-3)
    assert bool(jnp.all(jnp.diff(f) < 0.0))


def test_cylindrical_y500_shape_differs_from_spherical():
    x = jnp.linspace(0.05, 3.0, 80)
    y_cyl = y500_normalized_projected(x)
    p = gnfw(x)
    xx = jnp.linspace(0.0, 1.0, 400)
    p_mean = 3.0 * jnp.trapezoid(xx**2 * gnfw(xx), xx)
    y_sph = p / p_mean
    i1 = int(jnp.argmin(jnp.abs(x - 1.0)))
    assert abs(float(y_cyl[i1] / y_sph[i1]) - 1.0) > 0.05


def test_stacking_does_not_subtract_background_or_monopole():
    import inspect

    from flamingo.profiles import stacking

    src = inspect.getsource(stacking)
    assert "bkg" not in src
    assert "annulus" not in src
    assert "radial_profile" not in src
    assert not hasattr(stacking, "radial_profile")


def test_normalized_profile_uses_raw_pixels():
    nside = 64
    offset = 2.5e-6
    m = np.full(hp.nside2npix(nside), offset)
    x_edges = np.linspace(0.0, 2.0, 5)
    ybar, y_norm = normalized_profile(m, 0.4, 0.1, 180.0, x_edges)
    np.testing.assert_allclose(ybar[np.isfinite(ybar)], offset, atol=1e-18)
    np.testing.assert_allclose(y_norm, offset, rtol=0.05)


def test_stack_normalized_flat_map_is_unity():
    nside = 64
    m = np.ones(hp.nside2npix(nside))
    x_edges = np.linspace(0.0, 2.0, 5)
    stacked = stack_normalized(m, [0.4], [0.1], [180.0], x_edges, n_jobs=1)
    np.testing.assert_allclose(stacked["fhat"], 1.0, atol=0.05)
    assert stacked["n"] == 1


def test_stack_normalized_n_jobs_match_serial():
    nside = 64
    m = np.ones(hp.nside2npix(nside))
    x_edges = np.linspace(0.0, 2.0, 5)
    theta = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
    phi = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    t5 = [180.0] * 8
    serial = stack_normalized(m, theta, phi, t5, x_edges, n_jobs=1)
    threaded = stack_normalized(m, theta, phi, t5, x_edges, n_jobs=2)
    np.testing.assert_allclose(threaded["fhat"], serial["fhat"])
    assert threaded["n"] == serial["n"]


def test_stack_y_norms_scales_amplitude():
    nside = 64
    m = np.ones(hp.nside2npix(nside))
    x_edges = np.linspace(0.0, 2.0, 5)
    map_norm = stack_normalized(m, [0.4], [0.1], [180.0], x_edges, n_jobs=1)
    scaled = stack_normalized(
        m, [0.4], [0.1], [180.0], x_edges, n_jobs=1, y_norms=np.array([2.0])
    )
    np.testing.assert_allclose(scaled["fhat"], 0.5 * map_norm["fhat"], atol=0.05)
    assert abs(scaled["A_median"] - 0.5) < 0.05
