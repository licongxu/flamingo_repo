from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.plot_l1_m9_y500cyl_vs_soap import (
    build_figure,
    cyl_mpc2,
    inferred_sph_mpc2,
    output_stem,
)
from scripts.plot_l1_m9_y5r500_inferred_vs_truth import CATALOGUE, sph_over_cyl, y_arcmin2_to_mpc2
from flamingo.powerspectra.q_selection import resolve_q_selection


def test_output_is_nobg_inferred_sph_vs_soap():
    assert output_stem().name == "l1_m9_y500_inferred_sph_vs_soap_nobg"
    assert output_stem().parent.name == "diagnostics"
    assert CATALOGUE.parent == resolve_q_selection("qfrommap").l1_catalogue_dir


def test_inferred_sph_is_gnfw_times_catalogue_cyl():
    frame = pd.DataFrame(
        {
            "Y_500cyl_arcmin2": [2.0],
            "R_500c_Mpc": [1.5],
            "theta_500_arcmin": [3.0],
        }
    )
    cyl = y_arcmin2_to_mpc2(np.array([2.0]), np.array([1.5]), np.array([3.0]))
    np.testing.assert_allclose(cyl_mpc2(frame), cyl)
    np.testing.assert_allclose(inferred_sph_mpc2(frame), sph_over_cyl(1.0) * cyl)


def test_figure_is_inferred_sph_vs_soap():
    rng = np.random.default_rng(0)
    soap = 10.0 ** rng.uniform(-7.0, -4.0, 80)
    fig = build_figure(soap, 1.1 * soap)
    try:
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        assert ax.get_yscale() == "log"
        labels = " ".join(ax.get_legend_handles_labels()[1]).lower()
        ylabel = ax.get_ylabel().lower()
        assert "sph" in ylabel
        assert "f" in ylabel or "f" in labels
        assert len(ax.get_legend_handles_labels()[1]) == 1
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
