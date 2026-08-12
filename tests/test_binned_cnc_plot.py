"""Tests for paper-style L1_m9 binned CNC plotting."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "plot_l1_m9_binned_cnc.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("plot_binned_cnc", SCRIPT)
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


def test_q_edges_match_paper(mod):
    qe = mod.q_edges(5.0)
    assert qe.shape == (6,)
    assert qe[0] == pytest.approx(5.0)
    assert qe[-1] == pytest.approx(40.0)
    assert qe[-1] / qe[0] == pytest.approx(mod.Q_MAX_RATIO)
    assert np.all(np.diff(qe) > 0)

    qe50 = mod.q_edges(50.0)
    assert qe50[-1] == pytest.approx(400.0)


def test_bin_cnc_synthetic(mod, tmp_path):
    csv = tmp_path / "tiny.csv"
    csv.write_text(
        "z,q_from_aperture\n"
        "0.05,6\n"
        "0.15,12\n"
        "0.25,8\n"
        "0.35,3\n"  # below q-cut window
        "1.50,20\n"  # outside z window
    )
    n_zq, nz, nq, n_binned = mod.bin_cnc(csv, q_column="q_from_aperture", q_cut=5.0)
    assert n_zq.shape == (10, 5)
    assert n_binned == 3
    assert int(nz.sum()) == 3
    assert int(nq.sum()) == 3


def test_plot_smoke(mod, tmp_path):
    csv = tmp_path / "fid.csv"
    rows = ["z,q_from_aperture"]
    for iz in range(3):
        for iq, q in enumerate([6.0, 12.0, 25.0]):
            rows.append(f"{0.1 + 0.2 * iz},{q + 0.1 * iq}")
    csv.write_text("\n".join(rows) + "\n")

    mod.CAT_DIR = tmp_path
    orig = mod.catalogue_path

    def _fake_cat(variant: str, *, selection_suffix: str) -> Path:
        _ = variant, selection_suffix
        return csv

    mod.catalogue_path = _fake_cat
    try:
        png1 = mod.plot_cnc_vs_q_cuts(
            selection_suffix="qfrommap",
            q_column="q_from_aperture",
            selection_description="test",
            output_tag="test",
            stem=tmp_path / "qcuts",
        )
        png2 = mod.plot_cnc_feedback_qgt5(
            selection_suffix="qfrommap",
            q_column="q_from_aperture",
            selection_description="test",
            output_tag="test",
            stem=tmp_path / "feedback",
        )
    finally:
        mod.catalogue_path = orig

    assert png1.is_file() and png1.stat().st_size > 0
    assert png2.is_file() and png2.stat().st_size > 0
