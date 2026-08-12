from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import paper_results.figures as paper_figures

REPO = Path(__file__).resolve().parents[1]
MASKED_DATA = REPO / "data_paper" / "binned_bandpowers"


def _load_script(name: str, filename: str):
    path = REPO / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_fiducial_plot_selects_qfrommap_masked_bandpower():
    module = _load_script(
        "l1_fiducial_qfrommap_plot_test",
        "plot_l1_m9_masked_ps_alpha_fixed_1p12.py",
    )

    path = module._masked_paths("qgt5", log=False, selection_tag="qfrommap")

    assert path == MASKED_DATA / "Dl_yy_L1_m9_masked_qgt5_qfrommap_binned_18.txt"


def test_fiducial_qfrommap_plot_includes_completed_q3_and_q6_thresholds():
    module = _load_script(
        "l1_fiducial_qfrommap_thresholds_test",
        "plot_l1_m9_masked_ps_alpha_fixed_1p12.py",
    )

    cuts, tags = module.selection_cuts("qfrommap")

    assert cuts == [50.0, 20.0, 10.0, 6.0, 5.0, 3.0, 1.0]
    assert tags == ["qgt50", "qgt20", "qgt10", "qgt6", "qgt5", "qgt3", "qgt1"]


def test_combined_fiducial_planck_plot_routes_masked_curves_to_qfrommap():
    module = _load_script(
        "combined_fiducial_planck_qfrommap_path_test",
        "plot_flamingo_planck_tszps.py",
    )

    assert module._bin18_path(
        "L1_m9", masked=True, selection_tag="qfrommap"
    ) == MASKED_DATA / "Dl_yy_L1_m9_masked_qgt6_qfrommap_binned_18.txt"
    assert module.output_stem("qfrommap").name == (
        "l1_m9_fiducial_masking_planck_qfrommap"
    )


def test_combined_fiducial_planck_figure_has_two_title_free_spectra_panels():
    module = _load_script(
        "combined_fiducial_planck_qfrommap_layout_test",
        "plot_flamingo_planck_tszps.py",
    )

    fig = module.build_figure()
    try:
        assert len(fig.axes) == 2
        assert [ax.get_title() for ax in fig.axes] == ["", ""]
        assert fig.axes[0].get_legend_handles_labels()[1] == [
            "full sky",
            "$q>50$",
            "$q>20$",
            r"$q>10$ ($N=589$, $f_{\rm sky}=0.944$)",
            r"$q>5$ ($N=3,090$, $f_{\rm sky}=0.860$)",
        ]
        assert [text.get_text() for text in fig.axes[1].get_legend().get_texts()] == [
            "FLAMINGO full sky",
            "Bolliet et al. (2018)",
            "$q>6$ FLAMINGO",
            "Rotti et al. (2021)",
        ]
    finally:
        module.plt.close(fig)


def test_feedback_q5_plot_reads_qfrommap_from_binned_bandpowers():
    module = _load_script(
        "l1_feedback_q5_qfrommap_plot_test",
        "plot_l1_m9_feedback_bandpowers.py",
    )

    path = module._path("Jet", masked=True, log=False, selection_tag="qfrommap")

    assert path == MASKED_DATA / "Dl_yy_L1_m9_Jet_masked_qgt5_qfrommap_binned_18.txt"


def test_feedback_output_stem_preserves_legacy_name_and_tags_multiq_qfrommap():
    module = _load_script(
        "l1_feedback_q5_output_stem_test",
        "plot_l1_m9_feedback_bandpowers.py",
    )

    assert module.output_stem(log=False).name == ("l1_m9_feedback_ps_binned_18_alpha_fixed_1p12")
    assert module.output_stem(log=True, selection_tag="qfrommap").name == (
        "l1_m9_feedback_ps_logbins_multiq_qfrommap"
    )
    assert module.covariance_note("qfrommap") == ""
    assert "covariance" in module.covariance_note("qfrommz_alpha_fixed_1p12")


def test_feedback_multiq_plot_routes_all_masked_panels_to_qfrommap():
    module = _load_script(
        "l1_feedback_multiq_qfrommap_plot_test",
        "plot_l1_m9_feedback_bandpowers.py",
    )

    assert module.PANEL_CUTS == (
        ("full sky", None),
        (r"$q>20$", "qgt20"),
        (r"$q>10$", "qgt10"),
        (r"$q>5$", "qgt5"),
    )
    assert module._path(
        "Jet",
        masked=True,
        log=True,
        selection_tag="qfrommap",
        cut_tag="qgt20",
    ) == MASKED_DATA / (
        "Dl_yy_L1_m9_Jet_masked_qgt20_qfrommap_"
        "logbins_dln0p4_lmax10000.txt"
    )
    assert module.output_stem(log=True, selection_tag="qfrommap").name == (
        "l1_m9_feedback_ps_logbins_multiq_qfrommap"
    )


def test_feedback_multiq_figure_has_four_spectra_panels_and_no_main_title():
    module = _load_script(
        "l1_feedback_multiq_qfrommap_layout_test",
        "plot_l1_m9_feedback_bandpowers.py",
    )

    fig = module.build_figure(
        log=True,
        ell_range=(100.0, 10000.0),
        selection_tag="qfrommap",
    )
    try:
        assert len(fig.axes) == 4
        assert [ax.get_title() for ax in fig.axes] == [
            "full sky",
            "$q>20$",
            "$q>10$",
            "$q>5$",
        ]
        assert fig._suptitle is None
        assert [len(ax.lines) for ax in fig.axes] == [9, 9, 9, 9]
    finally:
        module.plt.close(fig)


def test_feedback_q1_plot_reads_qfrommap_from_binned_bandpowers():
    module = _load_script(
        "l1_feedback_q1_qfrommap_plot_test",
        "plot_l1_m9_feedback_bandpowers_qgt1.py",
    )

    path = module._path("Jet", masked=True, log=True, selection_tag="qfrommap")

    assert path == MASKED_DATA / (
        "Dl_yy_L1_m9_Jet_masked_qgt1_qfrommap_" "logbins_dln0p4_lmax10000.txt"
    )


def test_feedback_ratio_plot_selects_qfrommap_bandpower():
    module = _load_script(
        "l1_feedback_ratio_qfrommap_plot_test",
        "plot_l1_m9_feedback_ratio_vs_q.py",
    )

    path = module.bandpower_path("fgas-8sigma", 5.0, log=False, selection_tag="qfrommap")

    assert path == MASKED_DATA / ("Dl_yy_L1_m9_fgas-8sigma_masked_qgt5_qfrommap_binned_18.txt")
    description = module.error_band_description((("fullsky", "full sky"),))
    assert "full sky" in description
    assert "q>5" not in description


def test_feedback_ratio_qfrommap_uses_fullsky_and_q5_theory_bands():
    module = _load_script(
        "l1_feedback_ratio_qfrommap_bands_test",
        "plot_l1_m9_feedback_ratio_vs_q.py",
    )

    assert module.selection_error_band_kinds("qfrommap") == (
        ("fullsky", "full sky"),
        (5.0, r"$q>5$"),
    )
    assert module.cov_path_for_band(5.0, log=False).name == (
        "cov_full_L1_m9_customgnfw_bestfit_masked_qgt5_Dl_yy_binned_18.npy"
    )


def test_feedback_ratio_binned_figure_has_no_main_title_and_both_band_labels(monkeypatch):
    module = _load_script(
        "l1_feedback_ratio_qfrommap_layout_test",
        "plot_l1_m9_feedback_ratio_vs_q.py",
    )
    module.TAG = "qfrommap"
    module.OUTPUT_TAG = "qfrommap"
    module.ERROR_BAND_KINDS = module.selection_error_band_kinds("qfrommap")
    captured = {}

    def capture_figure(fig, stem):
        captured["fig"] = fig
        return stem.with_suffix(".png")

    monkeypatch.setattr(module, "_save", capture_figure)
    module.plot_all_feedback_ratio_vs_q(
        log=False,
        ell_range=(10.0, 959.5),
        show_title=False,
    )

    fig = captured["fig"]
    try:
        assert len(fig.axes) == 8
        assert fig._suptitle is None
        legend_labels = [text.get_text() for text in fig.legends[0].get_texts()]
        assert legend_labels[-2:] == [r"full sky $1\sigma$", r"$q>5$ $1\sigma$"]
    finally:
        module.plt.close(fig)


def test_paper_plot_selects_qfrommap_bandpower():
    path = paper_figures.masked_bandpower_path("Jet", "qgt5", mask_tag="qfrommap")

    assert path == MASKED_DATA / "Dl_yy_L1_m9_Jet_masked_qgt5_qfrommap_logbins_dln0p4_lmax10000.txt"


def test_legacy_selection_remains_default():
    module = _load_script(
        "l1_feedback_ratio_legacy_plot_test",
        "plot_l1_m9_feedback_ratio_vs_q.py",
    )

    assert "qfrommz_alpha_fixed_1p12" in module.bandpower_path("Jet", 5.0, log=False).name


def test_asz_fit_bandpower_figure_is_single_panel():
    module = _load_script(
        "l1_asz_fit_bandpower_layout_test",
        "plot_l1_m9_asz_fit_bandpowers.py",
    )

    summaries = module._load_summary()
    fig = module.build_figure(summaries)
    try:
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert len(ax.lines) == 15
        assert len(ax.containers) == 0
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "full sky",
            r"$q>50$",
            r"$q>20$",
            r"$q>10$",
            r"$q>5$",
        ]
        assert fig._suptitle is None
    finally:
        module.plt.close(fig)
