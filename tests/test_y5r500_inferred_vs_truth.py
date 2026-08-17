import numpy as np

from scripts.plot_l1_m9_y5r500_inferred_vs_truth import (
    A_SZ,
    GNFW_SHAPE,
    build_figure,
    model_y5r500_mpc2,
    sph_over_cyl,
    y_arcmin2_to_mpc2,
    y0_parametric,
)


def test_sph_over_cyl_is_shape_only_and_near_unity_at_5r500():
    ratio5 = sph_over_cyl(5.0)
    ratio1 = sph_over_cyl(1.0)
    ratio_p0 = sph_over_cyl(5.0, P0=GNFW_SHAPE["P0"] * 3.0)
    assert 0.0 < ratio1 < ratio5 < 1.0
    assert abs(ratio5 - ratio_p0) < 1e-12


def test_model_y5r500_scales_with_a_sz_and_is_positive():
    m = np.array([1.0e14, 5.0e14])
    z = np.array([0.2, 0.5])
    r500 = np.array([0.7, 1.1])
    y = model_y5r500_mpc2(m, z, r500)
    assert np.all(y > 0.0)
    y0 = y0_parametric(m, z)
    np.testing.assert_allclose(y / y0, y[0] / y0[0] * (r500 / r500[0]) ** 2)
    assert A_SZ < 0.0


def test_arcmin2_to_mpc2_is_da_squared():
    y = np.array([2.0])
    r500 = np.array([1.5])
    theta = np.array([3.0])
    np.testing.assert_allclose(y_arcmin2_to_mpc2(y, r500, theta), [0.5])


def test_inferred_vs_truth_figure_has_two_log_panels():
    rng = np.random.default_rng(0)
    truth = 10.0 ** rng.uniform(-7.0, -4.0, 200)
    fig = build_figure(truth, 1.1 * truth, 0.9 * truth)
    try:
        assert len(fig.axes) == 2
        assert fig.axes[0].get_xscale() == "log"
        assert fig.axes[0].get_yscale() == "log"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
