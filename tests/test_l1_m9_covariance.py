import numpy as np

from scripts.compute_l1_m9_simplegnfw_covariance import (
    ELL_EFF,
    ELL_MAX,
    ELL_MIN,
    assemble_covariance,
    bin_dl,
    bin_trispectrum,
    gaussian_covariance,
    validate_covariance,
)


def test_bin_dl_preserves_a_constant():
    ell = np.geomspace(9.0, 1085.0, 80)
    got = bin_dl(ell, np.full(ell.shape, 3.25))
    np.testing.assert_allclose(got, 3.25, rtol=0.0, atol=1e-14)


def test_bin_trispectrum_is_symmetric_and_has_18_bins():
    ell = np.geomspace(9.0, 1085.0, 40)
    trispectrum_cl = np.outer(ell**-1.2, ell**-1.2)
    got = bin_trispectrum(ell, trispectrum_cl)
    assert got.shape == (18, 18)
    np.testing.assert_allclose(got, got.T, rtol=1e-12, atol=0.0)
    assert np.all(got > 0.0)


def test_gaussian_covariance_matches_fullsky_knox_formula():
    dl = np.linspace(0.2, 1.9, 18) * 1e-12
    got = gaussian_covariance(dl)
    expected_diag = 2.0 * dl**2 / (
        (2.0 * ELL_EFF + 1.0) * (ELL_MAX - ELL_MIN)
    )
    np.testing.assert_allclose(np.diag(got), expected_diag)
    np.testing.assert_array_equal(got - np.diag(np.diag(got)), 0.0)


def test_assemble_covariance_adds_trispectrum_over_4pi():
    gaussian = np.eye(18) * 2.0
    trispectrum = np.full((18, 18), 3.0)
    got = assemble_covariance(gaussian, trispectrum)
    np.testing.assert_allclose(got, gaussian + trispectrum / (4.0 * np.pi))


def test_validate_covariance_reports_a_positive_definite_component_sum():
    gaussian = np.eye(18) * 2.0
    trispectrum = np.ones((18, 18))
    full = assemble_covariance(gaussian, trispectrum)
    diagnostics = validate_covariance(gaussian, trispectrum, full)
    assert diagnostics["max_component_residual"] == 0.0
    assert diagnostics["max_asymmetry"] == 0.0
    assert diagnostics["min_eigenvalue"] > 0.0
