from __future__ import annotations

import numpy as np
import pandas as pd

from flamingo.powerspectra.q_selection import resolve_q_selection
from scripts.plot_l1_m9_ym_relation import (
    Q_COLUMN,
    SELF_SIM_ALPHA_SZ,
    build_figure,
    catalogue_path,
    fit_asz,
    fit_ym,
    gnfw_y_over_y0_r2,
    load_ym,
    median_band,
    output_stem,
    scaled_y,
)


def test_catalogue_path_selects_qfrommap():
    selection = resolve_q_selection("qfrommap")
    path = catalogue_path()
    assert path.name == "halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommap.csv"
    assert path.parent == selection.l1_catalogue_dir
    assert Q_COLUMN == selection.q_column
    assert output_stem().name == "l1_m9_cnc_ym_relation_qgt5_qfrommap"


def test_load_ym_keeps_positive_rows(tmp_path):
    path = tmp_path / "cat.csv"
    pd.DataFrame(
        {
            "z": [0.1, 0.2, 0.3],
            "M_500c_Msun": [6e13, 0.0, 8e13],
            "R_500c_Mpc": [0.5, 0.4, 0.6],
            "Y_5R500c_Mpc2": [1e-5, 2e-5, -1.0],
            "q_from_aperture": [1.0, 6.0, 8.0],
        }
    ).to_csv(path, index=False)

    frame = load_ym(path)

    assert list(frame["M_500c_Msun"]) == [6e13]


def test_scaled_y_is_identity_at_z0():
    y = np.array([1.5e-5, 2.0e-5])
    np.testing.assert_allclose(scaled_y(y, np.zeros(2)), y)


def test_median_band_recovers_constant():
    mass = np.logspace(13.8, 14.8, 400)
    y = np.full(mass.shape, 3.0e-5)
    mids, med, p16, p84 = median_band(mass, y, min_count=10)
    assert mids.size > 0
    np.testing.assert_allclose(med, 3.0e-5)
    np.testing.assert_allclose(p16, 3.0e-5)
    np.testing.assert_allclose(p84, 3.0e-5)


def test_figure_marks_resolved_in_red():
    rng = np.random.default_rng(0)
    mass = 10.0 ** rng.uniform(13.7, 15.0, 300)
    y = 1e-6 * (mass / 1e14) ** (5.0 / 3.0)
    resolved = np.zeros(mass.shape, dtype=bool)
    resolved[-12:] = True
    fig = build_figure(mass, y, resolved)
    try:
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        red = [
            line
            for line in ax.lines
            if line.get_color() in ("#d62728", "r", "red") and line.get_marker() == "."
        ]
        assert len(red) == 1
        assert red[0].get_xdata().size == 12
        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert any("q>5" in label for label in labels)
        assert any("OLS" in label for label in labels)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_fit_ym_recovers_self_similar_slope_and_scatter():
    rng = np.random.default_rng(1)
    mass = 10.0 ** rng.uniform(13.7, 15.0, 2000)
    eps = rng.normal(0.0, 0.17, mass.size)
    y = 1.0e-5 * (mass / 1.0e14) ** (5.0 / 3.0) * np.exp(eps)
    fit = fit_ym(mass, y)
    assert abs(fit["alpha_Y"] - 5.0 / 3.0) < 0.02
    assert abs(fit["alpha_SZ"] - 1.0) < 0.02
    assert abs(fit["sigma_ln"] - 0.17) < 0.02


def test_fit_asz_recovers_injected_scaling():
    rng = np.random.default_rng(2)
    mass = 10.0 ** rng.uniform(13.7, 15.2, 1500)
    z = rng.uniform(0.05, 0.8, mass.size)
    r500 = 0.7 * (mass / 1.0e14) ** (1.0 / 3.0)
    a_sz, alpha = -4.20, SELF_SIM_ALPHA_SZ
    from scripts.plot_l1_m9_ym_relation import H, efunc, m_tilde

    k = gnfw_y_over_y0_r2()
    ez = efunc(z)
    y0 = (10.0 ** a_sz) * m_tilde(mass) ** alpha * ez**2 * (H / 0.7) ** (-0.5)
    y_sph = k * y0 * r500**2 * np.exp(rng.normal(0.0, 0.05, mass.size))
    fit = fit_asz(mass, y_sph, r500, z, k=k)
    assert abs(fit["A_SZ"] - a_sz) < 0.02
    assert abs(fit["alpha_SZ"] - alpha) < 0.02
    assert abs(fit["sigma_ln"] - 0.05) < 0.01
