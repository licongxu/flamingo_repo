import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hmfast.tracers.tsz_completeness import conditional_An_undetected

from flamingo.inference.masked_ps import MaskedTSZTheory


def test_masked_theory_declares_explicit_B():
    assert "B" in MaskedTSZTheory.params


@pytest.mark.parametrize("n_power, exponent", [(1, 0.5), (2, 2.0), (4, 8.0)])
def test_infinite_q_moment_matches_lognormal_limit(n_power, exponent):
    snr = jnp.array([[1.0, 10.0], [100.0, 1000.0]])
    sigma = 0.173

    got = conditional_An_undetected(
        snr,
        sigma_lnY=sigma,
        q_cat=np.inf,
        n_power=n_power,
        n_grid=512,
        nsig=8.0,
    )

    np.testing.assert_allclose(got, np.exp(exponent * sigma**2), rtol=2e-6)


def test_fullsky_none_matches_infinite_threshold_with_scatter():
    assert jax.devices()[0].platform == "gpu"
    theory = MaskedTSZTheory()
    values = {
        "H0": 68.1,
        "omega_cdm": 0.11872788986038219,
        "omega_b": 0.022538784599999993,
        "n_s": 0.965,
        "sigma_8": 0.8025701499024616,
        "tau_reio": 0.0544,
        "A_SZ": -4.0953238,
        "alpha_SZ": 1.12,
        "sigma_lnY": 0.173,
        "B": 1.41,
    }

    theory.q_cat = None
    fullsky = theory.evaluate_bandpowers(**values)
    theory.q_cat = np.inf
    infinite_threshold = theory.evaluate_bandpowers(**values)
    zero_scatter_values = {**values, "sigma_lnY": 0.0}
    zero_scatter = theory.evaluate_bandpowers(**zero_scatter_values)

    for component in ("1h", "2h"):
        assert fullsky[component].shape == (18,)
        assert np.all(np.isfinite(fullsky[component]))
        assert np.all(fullsky[component] > 0.0)
        np.testing.assert_allclose(
            fullsky[component],
            infinite_threshold[component],
            rtol=1e-10,
            atol=0.0,
        )
    np.testing.assert_allclose(
        fullsky["1h"] / zero_scatter["1h"],
        np.exp(2.0 * values["sigma_lnY"] ** 2),
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        fullsky["2h"] / zero_scatter["2h"],
        np.exp(values["sigma_lnY"] ** 2),
        rtol=4e-6,
    )
