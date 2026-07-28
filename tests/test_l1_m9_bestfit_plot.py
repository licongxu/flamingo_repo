import numpy as np
import pytest


def test_load_plot_data_returns_all_empirical_points_and_positive_theory(tmp_path):
    try:
        from scripts.plot_l1_m9_bestfit_customgnfw import load_plot_data
    except ModuleNotFoundError:
        pytest.fail("best-fit plotting module is missing")

    data_file = tmp_path / "bandpowers.txt"
    empirical_ell = np.geomspace(10.0, 1000.0, 18)
    empirical_dl = np.geomspace(0.1, 1.0, 18)
    np.savetxt(data_file, np.column_stack((empirical_ell, empirical_dl)))

    loaded_ell, loaded_dl, theory_ell, theory_dl = load_plot_data(data_file)

    np.testing.assert_array_equal(loaded_ell, empirical_ell)
    np.testing.assert_array_equal(loaded_dl, empirical_dl)
    assert loaded_ell.shape == (18,)
    assert theory_ell.shape == theory_dl.shape
    assert np.all(np.isfinite(theory_dl))
    assert np.all(theory_dl > 0.0)


def test_make_figure_contains_exactly_the_two_requested_series():
    try:
        from scripts.plot_l1_m9_bestfit_customgnfw import make_figure
    except ImportError:
        pytest.fail("make_figure is missing")

    figure = make_figure(
        np.array([10.0, 20.0]),
        np.array([0.1, 0.2]),
        np.array([10.0, 15.0, 20.0]),
        np.array([0.08, 0.15, 0.22]),
    )

    labels = figure.axes[0].get_legend_handles_labels()[1]
    assert labels == [
        "FLAMINGO L1_m9 full sky",
        "customGNFW best fit",
    ]


def test_save_figure_writes_nonempty_png_and_pdf(tmp_path):
    try:
        from scripts.plot_l1_m9_bestfit_customgnfw import make_figure, save_figure
    except ImportError:
        pytest.fail("save_figure is missing")

    figure = make_figure(
        np.array([10.0, 20.0]),
        np.array([0.1, 0.2]),
        np.array([10.0, 15.0, 20.0]),
        np.array([0.08, 0.15, 0.22]),
    )

    outputs = save_figure(figure, tmp_path / "comparison")

    assert [path.suffix for path in outputs] == [".png", ".pdf"]
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs)
