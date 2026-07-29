from pathlib import Path
import subprocess
import sys

import healpy as hp
import numpy as np
import pytest


def test_make_log_bins_has_exact_log_width_and_geometric_centres():
    try:
        from scripts.plot_l1_m9_bestfit_customgnfw_highell import make_log_bins
    except ModuleNotFoundError:
        pytest.fail("high-ell plotting module is missing")

    edges = make_log_bins(2.0, 2.0 * np.exp(0.8), 0.4)

    np.testing.assert_allclose(np.diff(np.log(edges)), [0.4, 0.4])
    np.testing.assert_allclose(edges, 2.0 * np.exp(0.4 * np.arange(3)))


def test_bin_cl_log_excludes_internal_right_edges_and_includes_final_edge():
    try:
        from scripts.plot_l1_m9_bestfit_customgnfw_highell import bin_cl_log
    except ModuleNotFoundError:
        pytest.fail("high-ell plotting module is missing")

    centres, binned = bin_cl_log(
        ell=np.array([1.0, 2.0, 3.0, 4.0]),
        cl=np.array([10.0, 20.0, 30.0, 40.0]),
        edges=np.array([1.0, 3.0, 4.0]),
    )

    np.testing.assert_allclose(centres, [np.sqrt(3.0), np.sqrt(12.0)])
    np.testing.assert_allclose(binned, [15.0, 35.0])


def test_compute_theory_bandpowers_returns_positive_total_equal_to_sum():
    try:
        from scripts.plot_l1_m9_bestfit_customgnfw_highell import (
            compute_theory_bandpowers,
        )
    except ImportError:
        pytest.fail("compute_theory_bandpowers is missing")

    edges = 100.0 * np.exp(0.4 * np.arange(3))
    theory = compute_theory_bandpowers(edges, n_per_bin=4)

    assert set(theory) == {"ell", "1h", "2h", "total"}
    for term in ("1h", "2h", "total"):
        assert theory[term].shape == (2,)
        assert np.all(np.isfinite(theory[term]))
        assert np.all(theory[term] > 0.0)
    np.testing.assert_allclose(theory["total"], theory["1h"] + theory["2h"])


def test_compute_theory_bandpowers_uses_requested_custom_parameters():
    from scripts.plot_l1_m9_bestfit_customgnfw_highell import (
        compute_theory_bandpowers,
    )

    edges = 100.0 * np.exp(0.4 * np.arange(3))
    free_fit = compute_theory_bandpowers(
        edges,
        A_SZ=-4.1095805,
        alpha_SZ=0.97447729,
        n_per_bin=4,
    )
    fixed_alpha_fit = compute_theory_bandpowers(
        edges,
        A_SZ=-4.0953238,
        alpha_SZ=1.12,
        n_per_bin=4,
    )

    fractional_change = np.abs(fixed_alpha_fit["total"] / free_fit["total"] - 1.0)
    assert np.max(fractional_change) > 0.01


def test_compute_simple_gnfw_bandpowers_uses_requested_mass_bias():
    try:
        from scripts.plot_l1_m9_bestfit_customgnfw_highell import (
            compute_simple_gnfw_bandpowers,
        )
    except ImportError:
        pytest.fail("compute_simple_gnfw_bandpowers is missing")

    edges = 100.0 * np.exp(0.4 * np.arange(3))
    theory_b1 = compute_simple_gnfw_bandpowers(edges, B=1.0, n_per_bin=4)
    theory_b135 = compute_simple_gnfw_bandpowers(edges, B=1.35, n_per_bin=4)

    for theory in (theory_b1, theory_b135):
        assert set(theory) == {"ell", "total"}
        assert theory["total"].shape == (2,)
        assert np.all(np.isfinite(theory["total"]))
        assert np.all(theory["total"] > 0.0)
    fractional_change = np.abs(theory_b135["total"] / theory_b1["total"] - 1.0)
    assert np.max(fractional_change) > 0.01


def test_fit_simple_gnfw_mass_bias_recovers_log_space_minimum(monkeypatch):
    import scripts.plot_l1_m9_bestfit_customgnfw_highell as module

    edges = np.array([100.0, 200.0, 400.0])

    def synthetic_bandpowers(edges, *, B, n_per_bin):
        return {
            "ell": np.sqrt(edges[:-1] * edges[1:]),
            "total": B * np.array([1.0, 2.0]),
        }

    monkeypatch.setattr(module, "compute_simple_gnfw_bandpowers", synthetic_bandpowers)
    best_B, best_model = module.fit_simple_gnfw_mass_bias(
        edges,
        measured_dl=np.array([1.25, 2.5]),
        n_per_bin=4,
    )

    assert best_B == pytest.approx(1.25, abs=1e-5)
    np.testing.assert_allclose(best_model["total"], [1.25, 2.5], atol=1e-5)


def test_compute_map_bandpowers_applies_pixel_window_deconvolution(tmp_path):
    try:
        from scripts.plot_l1_m9_bestfit_customgnfw_highell import (
            compute_map_bandpowers,
        )
    except ImportError:
        pytest.fail("compute_map_bandpowers is missing")

    nside = 8
    map_file = tmp_path / "map.fits"
    hp.write_map(map_file, np.arange(hp.nside2npix(nside), dtype=float))
    edges = np.array([2.0, 4.0, 8.0, 16.0])

    ell_raw, dl_raw = compute_map_bandpowers(
        map_file,
        edges,
        lmax=16,
        deconvolve_pixwin=False,
    )
    ell_corrected, dl_corrected = compute_map_bandpowers(
        map_file,
        edges,
        lmax=16,
        deconvolve_pixwin=True,
    )

    np.testing.assert_array_equal(ell_corrected, ell_raw)
    assert np.all(dl_corrected >= dl_raw)
    assert np.any(dl_corrected > dl_raw)


def test_write_empirical_bandpowers_roundtrips_values_and_metadata(tmp_path):
    try:
        from scripts.plot_l1_m9_bestfit_customgnfw_highell import (
            write_empirical_bandpowers,
        )
    except ImportError:
        pytest.fail("write_empirical_bandpowers is missing")

    output = tmp_path / "bandpowers.txt"
    ell = np.array([100.0, 200.0])
    displayed_dl = np.array([0.1, 0.2])
    write_empirical_bandpowers(output, ell, displayed_dl)

    np.testing.assert_allclose(np.loadtxt(output), np.column_stack((ell, displayed_dl)))
    text = output.read_text()
    assert "Delta ln ell = 0.4" in text
    assert "pixel window deconvolved" in text
    assert "ell_eff  1e12_D_ell_yy" in text


def test_make_high_ell_figure_has_full_range_ratio_and_no_2h_line():
    try:
        from scripts.plot_l1_m9_bestfit_customgnfw_highell import (
            make_high_ell_figure,
        )
    except ImportError:
        pytest.fail("make_high_ell_figure is missing")

    ell = np.geomspace(100.0, 10000.0, 4)
    figure = make_high_ell_figure(
        ell,
        map_dl=np.array([0.1, 0.2, 0.4, 0.8]),
        total_dl=np.array([0.08, 0.18, 0.38, 0.78]),
        one_halo_dl=np.array([0.07, 0.17, 0.37, 0.77]),
        simple_gnfw_curves={
            1.0: np.array([0.12, 0.24, 0.48, 0.96]),
            1.1: np.array([0.11, 0.22, 0.44, 0.88]),
            1.35: np.array([0.09, 0.18, 0.36, 0.72]),
        },
        best_simple_gnfw_B=1.23,
        best_simple_gnfw_dl=np.array([0.1, 0.2, 0.4, 0.8]),
        fixed_alpha_dl=np.array([0.09, 0.19, 0.39, 0.79]),
    )

    upper, ratio = figure.axes
    assert upper.get_legend_handles_labels()[1] == [
        "FLAMINGO L1_m9 full sky",
        "customGNFW best fit, total",
        "customGNFW best fit, 1-halo",
        "simple GNFW, B=1, total",
        "simple GNFW, B=1.1, total",
        "simple GNFW, B=1.35, total",
        "simple GNFW, best-fit B=1.230, total",
        "customGNFW fixed-alpha fit, total",
    ]
    np.testing.assert_allclose(upper.get_xlim(), [100.0, 10000.0])
    np.testing.assert_allclose(ratio.get_xlim(), [100.0, 10000.0])
    np.testing.assert_allclose(ratio.lines[0].get_xdata(), ell)


def test_high_ell_script_imports_from_the_scripts_directory():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"

    result = subprocess.run(
        [sys.executable, "-c", "import plot_l1_m9_bestfit_customgnfw_highell"],
        cwd=scripts_dir,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
