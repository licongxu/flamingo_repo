from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO / "scripts" / "plot_l1_m9_binned_cnc.py"
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
