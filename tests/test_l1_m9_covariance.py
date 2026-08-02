import importlib.util
from pathlib import Path

import numpy as np

from flamingo.inference.bandpowers import (
    bin_dl_uniform,
    gaussian_bandpower_covariance,
    trispectrum_bandpower_covariance,
    validate_covariance as validate_exact_covariance,
)
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


def test_bin_dl_uniform_uses_both_inclusive_edges():
    ell = np.arange(9.0, 17.0)
    dl = ell.copy()

    got = bin_dl_uniform(
        ell,
        dl,
        ell_min=np.array([9, 12]),
        ell_max=np.array([12, 16]),
    )

    np.testing.assert_allclose(
        got,
        [np.mean([9, 10, 11, 12]), np.mean([12, 13, 14, 15, 16])],
    )


def test_gaussian_covariance_keeps_shared_edge():
    ell = np.geomspace(9.0, 16.0, 10)
    cl = np.full_like(ell, 3e-12)

    covariance = gaussian_bandpower_covariance(
        ell,
        cl,
        0.61,
        ell_min=np.array([9, 12]),
        ell_max=np.array([12, 16]),
    )

    assert covariance[0, 1] > 0.0
    assert covariance[1, 0] == covariance[0, 1]


def test_gaussian_covariance_matches_synthetic_reference_in_physical_units():
    reference_path = Path(
        "/scratch/scratch-lxu/tsz_cnc_paper_plots/tsz_only/bandpower_theory.py"
    )
    spec = importlib.util.spec_from_file_location(
        "synthetic_bandpower_theory",
        reference_path,
    )
    assert spec is not None and spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    ell = np.geomspace(9.0, 1085.0, 47)
    cl = 2.3e-12 * (ell / 80.0) ** -1.17
    f_sky = 0.8595492784996496

    expected = reference.gaussian_bandpower_covariance(
        ell,
        cl,
        f_sky,
        ell_min=ELL_MIN,
        ell_max=ELL_MAX,
    ) / 1e24
    got = gaussian_bandpower_covariance(ell, cl, f_sky)

    np.testing.assert_allclose(got, expected, rtol=1e-13, atol=0.0)


def test_trispectrum_covariance_matches_synthetic_reference_in_physical_units():
    reference_path = Path(
        "/scratch/scratch-lxu/tsz_cnc_paper_plots/tsz_only/bandpower_theory.py"
    )
    spec = importlib.util.spec_from_file_location(
        "synthetic_trispectrum_bandpower_theory",
        reference_path,
    )
    assert spec is not None and spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    ell = np.geomspace(9.0, 1085.0, 29)
    shape = (ell / 80.0) ** -0.83
    trispectrum = 1.7e-27 * np.outer(shape, shape)
    f_sky = 0.9435688947539681

    expected = reference.trispectrum_bandpower_covariance(
        ell,
        trispectrum,
        f_sky,
        ell_min=ELL_MIN,
        ell_max=ELL_MAX,
    ) / 1e24
    got = trispectrum_bandpower_covariance(ell, trispectrum, f_sky)

    np.testing.assert_allclose(got, expected, rtol=1e-13, atol=0.0)


def test_exact_covariance_validation_uses_already_normalized_components():
    gaussian = np.eye(2) * 2.0
    trispectrum = np.ones((2, 2)) * 0.25
    covariance = gaussian + trispectrum

    diagnostics = validate_exact_covariance(gaussian, trispectrum, covariance)

    assert diagnostics["max_component_residual"] == 0.0
    assert diagnostics["min_eigenvalue"] > 0.0


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
