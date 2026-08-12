from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO / "scripts/figures/plot_l1_m9_binned_cnc.py"
    spec = importlib.util.spec_from_file_location("l1_m9_binned_cnc_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_catalogue_path_selects_qfrommap():
    module = _load_script()

    assert module.catalogue_path("Jet").name == (
        "halo_catalogue_M500c_5e13_zlt3_Jet_yang26rot_qfrommap.csv"
    )
    assert module.Q_COLUMN == "q_from_aperture"


def test_script_loads_without_repository_on_python_path(tmp_path):
    script = REPO / "scripts/figures/plot_l1_m9_binned_cnc.py"
    command = (
        "import runpy; "
        f"runpy.run_path({str(script)!r}, run_name='l1_m9_binned_cnc_entrypoint_test')"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", command],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_paper_edges_are_exact():
    module = _load_script()

    assert np.allclose(module.Z_EDGES, np.linspace(0.005, 1.0, 11))
    assert np.allclose(module.Q_EDGES, np.geomspace(5.0, 40.0, 6))


def test_bin_cnc_uses_bounds_and_returns_joint_histogram(tmp_path):
    module = _load_script()
    path = tmp_path / "catalogue.csv"
    pd.DataFrame(
        {
            "z": [0.005, 0.2, 1.0, 0.004, 0.5, 0.5],
            "q_from_aperture": [5.0, 6.0, 40.0, 10.0, 4.99, 40.01],
        }
    ).to_csv(path, index=False)

    n_zq = module.bin_cnc(path, chunksize=2)

    assert n_zq.shape == (10, 5)
    assert n_zq.sum() == 3
    assert np.array_equal(
        n_zq.sum(axis=0),
        np.histogram([5.0, 6.0, 40.0], module.Q_EDGES)[0],
    )
    assert np.array_equal(
        n_zq.sum(axis=1),
        np.histogram([0.005, 0.2, 1.0], module.Z_EDGES)[0],
    )


def test_output_stems_are_explicit_qfrommap_names():
    module = _load_script()

    assert module.output_stem("q").name == (
        "l1_m9_cnc_binned_Nq_qgt5_feedback_qfrommap"
    )
    assert module.output_stem("z").name == (
        "l1_m9_cnc_binned_Nz_qgt5_feedback_qfrommap"
    )


def test_legend_label_uses_latex_and_includes_catalogue_total():
    module = _load_script()

    assert module.legend_label("L1_m9", 1330) == (
        r"fiducial (L1\_m9) ($N=1{,}330$)"
    )


def test_separate_figures_group_all_nine_marginals():
    module = _load_script()
    histograms = {
        variant: np.full((10, 5), index + 1, dtype=int)
        for index, variant in enumerate(module.VARIANTS)
    }

    fig_q = module.build_figure(histograms, "q")
    fig_z = module.build_figure(histograms, "z")

    try:
        assert len(fig_q.axes) == len(fig_z.axes) == 1
        assert len(fig_q.axes[0].patches) == 5 * len(module.VARIANTS) == 45
        assert len(fig_z.axes[0].patches) == 10 * len(module.VARIANTS) == 90
        assert fig_q.axes[0].get_xscale() == "log"
        assert fig_z.axes[0].get_xscale() == "linear"
        assert np.allclose(fig_q.get_size_inches(), [7.1, 5.4])
        assert np.allclose(fig_z.get_size_inches(), [7.1, 5.4])
        assert fig_q.axes[0].get_xlabel() == r"$q$"
        assert fig_z.axes[0].get_xlabel() == r"$z$"
        assert fig_q.axes[0].get_ylabel() == fig_z.axes[0].get_ylabel() == r"$N$"
        assert fig_q.axes[0].xaxis.label.get_usetex()
        q_bars = fig_q.axes[0].patches
        assert q_bars[5].get_alpha() == 1.0
        assert all(bar.get_alpha() == 0.68 for bar in q_bars[:5])
        assert all(bar.get_alpha() == 0.68 for bar in q_bars[10:])
        labels = [
            text.get_text() for text in fig_q.axes[0].get_legend().get_texts()
        ]
        assert len(labels) == 9
        for index, label in enumerate(labels):
            assert rf"($N={50 * (index + 1)}$)" in label
    finally:
        module.plt.close(fig_q)
        module.plt.close(fig_z)
