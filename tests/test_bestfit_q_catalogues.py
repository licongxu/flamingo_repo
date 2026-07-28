"""Tests for the streamed best-fit ``q_from_mz`` catalogue generator."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "create_l1_m9_bestfit_q_catalogues.py"
)
SPEC = importlib.util.spec_from_file_location("create_l1_m9_bestfit_q_catalogues", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class FakeScaling:
    A_SZ = -4.1095805
    alpha_SZ = 0.97447729
    B = 1.41
    sigma_lnY = 0.173
    seed = 20260630

    def q(self, mass, redshift, *, index):
        return mass / 1e14 + redshift + index / 100


def _source_catalogue(path: Path) -> None:
    path.write_text(
        "# fiducial B=1.35\n"
        "soap_index,z,M_500c_Msun,R_500c_Mpc,q_from_mz\n"
        "10,0.2,1e14,0.7,99\n"
        "20,0.4,2e14,0.9,88\n"
    )


def test_bestfit_path_adds_suffix_before_csv(tmp_path):
    source = tmp_path / "halo_qfrommz.csv"
    assert module.bestfit_path(source) == tmp_path / "halo_qfrommz_bestfit.csv"


def test_bestfit_scaling_uses_rerun_parameters():
    scaling = module.bestfit_scaling()

    assert scaling.A_SZ == pytest.approx(-4.1095805)
    assert scaling.alpha_SZ == pytest.approx(0.97447729)
    assert scaling.B == pytest.approx(1.41)
    assert scaling.sigma_lnY == pytest.approx(0.173)
    assert scaling.seed == 20260630


def test_rewrite_preserves_catalogue_and_replaces_only_q(tmp_path):
    source = tmp_path / "halo_qfrommz.csv"
    output = module.bestfit_path(source)
    _source_catalogue(source)

    summary = module.rewrite_catalogue(
        source,
        output,
        FakeScaling(),
        chunksize=1,
    )

    source_data = pd.read_csv(source, comment="#")
    output_data = pd.read_csv(output, comment="#")
    assert list(output_data.columns) == list(source_data.columns)
    pd.testing.assert_frame_equal(
        output_data.drop(columns="q_from_mz"),
        source_data.drop(columns="q_from_mz"),
        check_dtype=False,
    )
    assert np.allclose(output_data["q_from_mz"], [1.3, 2.6])
    assert summary.rows == 2
    assert summary.q_min == pytest.approx(1.3)
    assert summary.q_max == pytest.approx(2.6)

    comments = [
        line.strip()
        for line in output.read_text().splitlines()
        if line.startswith("#")
    ]
    assert any(str(source) in line for line in comments)
    assert any("A_SZ=-4.1095805" in line for line in comments)
    assert any("alpha_SZ=0.97447729" in line for line in comments)
    assert any("B=1.41" in line for line in comments)


def test_rewrite_refuses_to_overwrite_without_force(tmp_path):
    source = tmp_path / "halo_qfrommz.csv"
    output = module.bestfit_path(source)
    _source_catalogue(source)
    output.write_text("keep me\n")

    with pytest.raises(FileExistsError):
        module.rewrite_catalogue(source, output, FakeScaling())

    assert output.read_text() == "keep me\n"


def test_rewrite_removes_temporary_file_after_calculation_failure(tmp_path):
    class BrokenScaling(FakeScaling):
        def q(self, mass, redshift, *, index):
            raise RuntimeError("calculation failed")

    source = tmp_path / "halo_qfrommz.csv"
    output = module.bestfit_path(source)
    _source_catalogue(source)

    with pytest.raises(RuntimeError, match="calculation failed"):
        module.rewrite_catalogue(source, output, BrokenScaling())

    assert not output.exists()
    assert not list(tmp_path.glob("*.tmp-*"))
