"""Helpers for the L1_m9 JXPaint run: angles, bins, no pixwin."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "paint_l1_m9", REPO / "painting" / "paint_l1_m9.py"
)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def test_healpy_to_jxpaint_north_pole():
    lon, lat = mod.healpy_to_jxpaint(theta=0.0, phi=1.2)
    assert lon == 1.2
    assert lat == np.pi


def test_bins_end_at_959p5():
    assert mod.ELL_EFF[-1] == 959.5
    assert mod.ELL_MAX[-1] == 1085
    assert mod.LMAX >= int(mod.ELL_MAX[-1])


def test_inference_params_match_hydro_fit():
    assert mod.B == 1.41
    assert mod.ALPHA_SZ == 1.12
    assert mod.H0 == 68.1
    assert mod.GNFW_SHAPE["P0"] == 8.13
    assert mod.A_SZ == -4.1075214
    F0, B = 7.789341272592724, 1.41
    y0_param = 7.137e-5
    y0_true = mod.y0_for_painter(y0_param, F0, B=B)
    painted_center = y0_true / B ** (1.0 / 3.0) * F0
    np.testing.assert_allclose(painted_center, y0_param, rtol=1e-12)


def test_deconvolve_beam_preserves_monopole():
    import healpy as hp

    m = np.full(hp.nside2npix(32), 1e-6)
    out = mod.deconvolve_beam(m, fwhm_arcmin=10.0, lmax=64)
    np.testing.assert_allclose(out.mean(), m.mean(), rtol=1e-5)


def test_m1e13_outputs_do_not_clobber_cnc():
    assert mod.OUT_TAG == "m1e13"
    assert "1e13" in mod.CAT_FILE.name
    assert "5e13" in mod.CAT_MASK.name
    assert "qfrommap" in mod.CAT_MASK.name


def test_bin_dl_18_no_pixwin_mean():
    ell = np.arange(mod.LMAX + 1, dtype=float)
    cl = np.full(ell.shape, 2.0 * np.pi / np.maximum(ell * (ell + 1.0), 1.0))
    dl = mod.bin_dl_18(ell, cl)
    assert dl.shape == (18,)
    np.testing.assert_allclose(dl, 1e12, rtol=1e-12)
