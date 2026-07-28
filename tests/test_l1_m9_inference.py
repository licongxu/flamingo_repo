import numpy as np

from flamingo.inference.l1_m9 import (
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
