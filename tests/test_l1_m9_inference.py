import numpy as np

from flamingo.inference.l1_m9 import (
    L1M9CustomGNFWTheory,
    gaussian_loglike,
    load_bandpower_likelihood,
)


def test_load_bandpower_likelihood_applies_data_scale(tmp_path):
    data = tmp_path / "data.txt"
    covariance = tmp_path / "cov.npy"
    np.savetxt(data, [[10.0, 2.0], [20.0, 5.0]])
    np.save(covariance, np.diag([4e-24, 9e-24]))

    ell, observed, cov, inverse = load_bandpower_likelihood(
        data, covariance, data_scale=1e-12
    )

    np.testing.assert_array_equal(ell, [10.0, 20.0])
    np.testing.assert_array_equal(observed, [2e-12, 5e-12])
    np.testing.assert_array_equal(cov, np.diag([4e-24, 9e-24]))
    np.testing.assert_allclose(inverse @ cov, np.eye(2))


def test_gaussian_loglike_matches_literal_quadratic_form():
    observed = np.array([1.0, 2.0])
    theory = np.zeros(2)
    inverse = np.diag([1.0 / 4.0, 1.0 / 9.0])
    expected = -0.5 * (1.0 / 4.0 + 4.0 / 9.0)
    assert gaussian_loglike(observed, theory, inverse) == expected


def test_custom_gnfw_theory_returns_positive_1h_and_2h_bandpowers():
    theory = L1M9CustomGNFWTheory()
    theory.initialize()

    bandpowers = theory.evaluate_bandpowers(-4.1, 1.12)

    assert set(bandpowers) == {"1h", "2h"}
    for term in bandpowers.values():
        assert term.shape == (18,)
        assert np.all(np.isfinite(term))
        assert np.all(term > 0.0)
    assert np.all(bandpowers["1h"] + bandpowers["2h"] > 0.0)


def test_custom_gnfw_theory_returns_total_spectrum_for_best_fit():
    theory = L1M9CustomGNFWTheory()
    theory.initialize()

    spectrum = theory.evaluate_spectrum(-4.1095805, 0.97447729)

    assert set(spectrum) == {"ell", "1h", "2h", "total"}
    assert spectrum["ell"].shape == spectrum["total"].shape
    for term in ("1h", "2h", "total"):
        assert np.all(np.isfinite(spectrum[term]))
        assert np.all(spectrum[term] > 0.0)
    np.testing.assert_allclose(
        spectrum["total"],
        spectrum["1h"] + spectrum["2h"],
        rtol=1e-14,
    )


def test_custom_gnfw_theory_evaluates_a_requested_multipole_grid():
    theory = L1M9CustomGNFWTheory()
    theory.initialize()
    requested_ell = np.geomspace(100.0, 10000.0, 8)

    spectrum = theory.evaluate_spectrum(
        -4.1095805,
        0.97447729,
        ell=requested_ell,
    )

    np.testing.assert_array_equal(spectrum["ell"], requested_ell)
    for term in ("1h", "2h", "total"):
        assert spectrum[term].shape == requested_ell.shape
        assert np.all(np.isfinite(spectrum[term]))
        assert np.all(spectrum[term] > 0.0)
