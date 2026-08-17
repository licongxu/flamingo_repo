import numpy as np

from scripts.plot_l1_m9_y5r500_inferred_vs_truth import (
    GNFW_SHAPE,
    build_figure,
    sph_over_cyl,
    y_arcmin2_to_mpc2,
)


def test_sph_over_cyl_is_shape_only_and_near_unity_at_5r500():
    ratio5 = sph_over_cyl(5.0)
    ratio1 = sph_over_cyl(1.0)
    ratio_p0 = sph_over_cyl(5.0, P0=GNFW_SHAPE["P0"] * 3.0)
    assert 0.0 < ratio1 < ratio5 < 1.0
    np.testing.assert_allclose(ratio1, 0.82747, rtol=1e-4)
    np.testing.assert_allclose(ratio5, 0.98578, rtol=1e-4)
    assert abs(ratio5 - ratio_p0) < 1e-12


def test_arcmin2_to_mpc2_is_da_squared():
    y = np.array([2.0])
    r500 = np.array([1.5])
    theta = np.array([3.0])
    np.testing.assert_allclose(y_arcmin2_to_mpc2(y, r500, theta), [0.5])


def test_inferred_vs_truth_figure_is_one_log_panel_with_two_series():
    rng = np.random.default_rng(0)
    truth = 10.0 ** rng.uniform(-7.0, -4.0, 200)
    fig = build_figure(truth, 1.1 * truth, 1.2 * truth, 0.9 * truth)
    try:
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        assert len(ax.get_legend_handles_labels()[1]) == 2
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
