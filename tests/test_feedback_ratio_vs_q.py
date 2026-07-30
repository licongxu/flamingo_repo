"""Tests for feedback ratio-vs-q plot (full sky + q cuts; bands about 1)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "plot_l1_m9_feedback_ratio_vs_q.py"
DATA = REPO / "data_paper" / "binned_bandpowers"
DATA_FB = REPO / "data_paper" / "feedback_bandpower"
TAG = "qfrommz_alpha_fixed_1p12"


def _load_module():
    spec = importlib.util.spec_from_file_location("plot_feedback_ratio_vs_q", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    if not SCRIPT.is_file():
        pytest.fail(f"missing plot script: {SCRIPT}")
    return _load_module()


def test_default_curves_fullsky_not_q50(mod):
    assert "fullsky" in mod.DEFAULT_CURVES
    assert 50.0 not in mod.DEFAULT_CURVES
    assert 1.0 not in mod.DEFAULT_CURVES
    assert 3.0 not in mod.DEFAULT_CURVES
    assert set(mod.DEFAULT_CURVES) == {"fullsky", 20.0, 10.0, 5.0}


def test_error_bands_only_fullsky_and_qgt5(mod):
    kinds = [b[0] for b in mod.ERROR_BAND_KINDS]
    assert kinds == ["fullsky", 5.0]


def test_q5_error_band_color_darker_than_q5_curve(mod):
    curves = list(mod.DEFAULT_CURVES)
    cmap = mod._color_map_for_curves(curves)
    band = np.asarray(mod._error_band_color(5.0, cmap), dtype=float)
    line = np.asarray(cmap[5.0], dtype=float)
    # Band is a darkened version of the q>5 line color (more visible fill).
    assert np.all(band[:3] <= line[:3] + 1e-12)
    assert np.any(band[:3] < line[:3] - 1e-6)
    assert mod._error_band_alpha(5.0) > mod._error_band_alpha("fullsky")
    # full-sky band is fixed light grey
    assert mod._error_band_color("fullsky", cmap) == "#d0d0d0"


def test_ylim_from_ratios_ignores_wide_bands(mod):
    # Fake ratios near 1; ylim should stay tight even if bands would be huge.
    n = 5
    inside = np.ones(n, dtype=bool)
    ratios = {
        "fullsky": np.array([0.95, 0.96, 0.97, 0.98, 0.99]),
        5.0: np.array([0.94, 0.95, 0.96, 0.97, 0.98]),
    }
    lo, hi = mod._ylim_from_ratios(ratios, inside)
    assert lo < 0.94 and hi > 0.99
    assert hi - lo < 0.5  # not expanded to full-sky band width


def test_ratio_fullsky_matches_direct_division(mod):
    path_v = mod.bandpower_path("fgas-8sigma", "fullsky", log=False)
    path_f = mod.bandpower_path("fiducial", "fullsky", log=False)
    if not path_v.is_file() or not path_f.is_file():
        pytest.skip("missing fullsky bandpowers")
    ell, ratio = mod.ratio_at_cut("fgas-8sigma", "fullsky", log=False)
    _, dl_v = mod.load_bandpowers(path_v)
    _, dl_f = mod.load_bandpowers(path_f)
    np.testing.assert_allclose(ratio, dl_v / dl_f, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("log", [False, True])
def test_relative_error_band_is_sigma_over_fiducial(mod, log):
    for kind in ("fullsky", 5.0):
        ell, rel = mod.relative_error_band(kind, log=log)
        sigma = mod.load_sigma_1e12(kind, log=log)
        _, dl_f = mod.load_bandpowers(mod.bandpower_path(mod.FIDUCIAL, kind, log=log))
        np.testing.assert_allclose(rel, sigma / dl_f, rtol=0.0, atol=0.0)
        assert np.all(rel > 0.0)


def test_plot_writes_both_binnings(mod, tmp_path):
    curves = list(mod.DEFAULT_CURVES)
    for kind in curves:
        if not mod.bandpower_path(mod.DEFAULT_VARIANT, kind, log=False).is_file():
            pytest.skip(f"missing {kind}")
    for log, stem_name, erange in (
        (False, "ratio_18", (10.0, 959.5)),
        (True, "ratio_log", (100.0, 10000.0)),
    ):
        stem = tmp_path / stem_name
        png = mod.plot_ratio_vs_q(
            mod.DEFAULT_VARIANT,
            q_cuts=curves,
            log=log,
            ell_range=erange,
            stem=stem,
        )
        assert png is not None and png.is_file() and png.stat().st_size > 0


def test_plot_all_feedback_smoke(mod, tmp_path):
    curves = ["fullsky", 5.0]
    for v in ("fgas-8sigma", "Jet"):
        for kind in curves:
            if not mod.bandpower_path(v, kind, log=False).is_file():
                pytest.skip(f"missing {v} {kind}")
    png = mod.plot_all_feedback_ratio_vs_q(
        q_cuts=curves,
        log=False,
        ell_range=(10.0, 959.5),
        stem=tmp_path / "all",
        variants=["fgas-8sigma", "Jet"],
        shared_ylim=True,
    )
    assert png.is_file() and png.stat().st_size > 0
