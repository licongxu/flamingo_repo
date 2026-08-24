from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from flamingo.powerspectra.q_selection import resolve_q_selection
from scripts.plot_l1_m9_stacked_pressure_profile import (
    MASS_EDGES,
    M_MIN,
    Q_MIN,
    a10_cyl_at,
    a10_y_norm,
    build_figure,
    catalogue_path,
    mass_label,
    output_stem,
    output_stem_map,
    output_stem_qgt5,
    output_stem_top,
    read_ymap,
    split_mass_bins,
)


def test_catalogue_is_m5e13_and_output_is_massbins():
    selection = resolve_q_selection("qfrommap")
    assert catalogue_path().name == "halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommap.csv"
    assert catalogue_path().parent == selection.l1_catalogue_dir
    assert output_stem().name == "l1_m9_fiducial_mgt5e13_massbins_ycyl_vs_a10"
    assert output_stem_map().name == "l1_m9_fiducial_mgt5e13_massbins_ycyl_stack"
    assert output_stem_qgt5().name == "l1_m9_fiducial_qgt5_ycyl_stack"
    assert output_stem_top(1000).name == "l1_m9_fiducial_top1000_ycyl_stack"
    assert output_stem().parent.name == "stack"
    assert M_MIN == 5.0e13
    assert Q_MIN == 5.0
    assert MASS_EDGES[0] == 5.0e13
    src = inspect.getsource(read_ymap)
    assert "mean" not in src
    assert "-=" not in src


def test_a10_y_norm_scales_with_mass_and_shape_has_unit_aperture():
    z = np.array([0.3, 0.3])
    r = np.array([0.8, 0.8])
    y = a10_y_norm(np.array([5e14, 1e15]), z, r)
    assert y[1] > y[0]
    assert np.all(y > 0.0)
    y0_scale = a10_y_norm(np.array([5e14]), np.array([0.3]), np.array([1.0]))[0]
    assert 1e-6 < y0_scale < 1e-3
    x = np.linspace(0.0, 1.0, 400)
    f = a10_cyl_at(x)
    assert abs(float(2.0 * np.trapezoid(x * f, x)) - 1.0) < 2e-3


def test_split_mass_bins_keeps_every_row():
    mass = np.concatenate(
        [np.full(50, 6e13), np.full(5, 2.0e14), np.full(3, 8.0e14)]
    )
    frame = pd.DataFrame({"M_500c_Msun": mass, "theta_rot_rad": np.zeros(mass.size)})
    bins = split_mass_bins(frame)
    assert sum(len(chunk) for _, _, chunk in bins) == len(frame)
    top = [chunk for lo, hi, chunk in bins if lo >= 6.0e14]
    assert top and len(top[0]) == 3


def test_mass_label_and_figure_panels():
    import matplotlib.pyplot as plt

    assert "10^{13}" in mass_label(5e13, 8e13) or "5" in mass_label(5e13, 8e13)
    x = np.array([0.3, 1.0, 2.0])
    y = np.array([2.0, 1.0, 0.4])
    stacked = {
        "x_mid": x,
        "fhat": y,
        "sem": 0.05 * y,
        "n": 12,
        "A_median": 0.82,
        "A_p16": 0.7,
        "A_p84": 0.9,
    }
    fig = build_figure(
        [(5e13, 8e13, stacked), (6.0e14, np.inf, stacked)],
        a10_cyl_at(x),
        x,
    )
    fig_map = build_figure(
        [(5e13, 8e13, stacked), (6.0e14, np.inf, stacked)],
        a10_cyl_at(x),
        x,
        vs_a10=False,
    )
    try:
        visible = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
        assert len(visible) == 2
        labels = " ".join(visible[0].get_legend_handles_labels()[1]).lower()
        assert "a10" in labels
        assert "a=" in labels.replace(" ", "")
        assert visible[0].get_xscale() == "log"
        map_labels = " ".join(fig_map.axes[0].get_legend_handles_labels()[1])
        assert "A=" not in map_labels.replace(" ", "")
    finally:
        plt.close(fig)
        plt.close(fig_map)
