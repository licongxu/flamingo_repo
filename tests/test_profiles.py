"""Tests for the GNFW pressure model and its line-of-sight projection."""
import jax.numpy as jnp
import numpy as np

from flamingo.profiles import A10_PARAMS, gnfw, projected_shape


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
