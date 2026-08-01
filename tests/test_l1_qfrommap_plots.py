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


def test_feedback_q5_plot_reads_qfrommap_from_binned_bandpowers():
    module = _load_script(
        "l1_feedback_q5_qfrommap_plot_test",
        "plot_l1_m9_feedback_bandpowers.py",
    )

    path = module._path("Jet", masked=True, log=False, selection_tag="qfrommap")

    assert path == MASKED_DATA / "Dl_yy_L1_m9_Jet_masked_qgt5_qfrommap_binned_18.txt"


def test_feedback_q5_output_stem_preserves_legacy_name_and_tags_qfrommap():
    module = _load_script(
        "l1_feedback_q5_output_stem_test",
        "plot_l1_m9_feedback_bandpowers.py",
    )

    assert module.output_stem(log=False).name == ("l1_m9_feedback_ps_binned_18_alpha_fixed_1p12")
    assert module.output_stem(log=True, selection_tag="qfrommap").name == (
        "l1_m9_feedback_ps_logbins_qfrommap_qgt5"
    )
    assert module.covariance_note("qfrommap") == ""
    assert "covariance" in module.covariance_note("qfrommz_alpha_fixed_1p12")


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


def test_paper_plot_selects_qfrommap_bandpower():
    path = paper_figures.masked_bandpower_path("Jet", "qgt5", mask_tag="qfrommap")

    assert path == MASKED_DATA / "Dl_yy_L1_m9_Jet_masked_qgt5_qfrommap_logbins_dln0p4_lmax10000.txt"


def test_legacy_selection_remains_default():
    module = _load_script(
        "l1_feedback_ratio_legacy_plot_test",
        "plot_l1_m9_feedback_ratio_vs_q.py",
    )

    assert "qfrommz_alpha_fixed_1p12" in module.bandpower_path("Jet", 5.0, log=False).name
