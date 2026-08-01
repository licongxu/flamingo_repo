from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from flamingo.powerspectra.q_selection import resolve_q_selection


REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str, relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_qfrommap_selection_uses_corrected_l1_and_canonical_l2_catalogues():
    selection = resolve_q_selection("qfrommap")

    assert selection.tag == "qfrommap"
    assert selection.q_column == "q_from_aperture"
    assert selection.l1_catalogue_dir == Path(
        "/rds/rds-lxu/flamingo/.hbt_join_fix_staging/20260731/"
        "L1_m9/catalogues"
    )
    assert selection.l2_root == Path("/rds/rds-lxu/flamingo/L2p8_m9")


def test_legacy_selection_defaults_are_unchanged():
    selection = resolve_q_selection("qfrommz_alpha_fixed_1p12")

    assert selection.tag == "qfrommz_alpha_fixed_1p12"
    assert selection.q_column == "q_from_mz"
    assert selection.l1_catalogue_dir == Path(
        "/rds/rds-lxu/flamingo/L1_m9/catalogues"
    )


def test_feedback_catalogue_loader_selects_requested_q_column(tmp_path):
    module = _load_script(
        "feedback_bandpowers_q_selection_test",
        "scripts/compute_l1_m9_feedback_bandpowers.py",
    )
    rows = 10_000
    path = tmp_path / "catalogue.csv"
    pd.DataFrame(
        {
            "z": np.full(rows, 0.2),
            "R_500c_Mpc": np.full(rows, 1.0),
            "theta_rot_rad": np.full(rows, 1.0),
            "phi_rot_rad": np.full(rows, 2.0),
            "theta_nat_rad": np.full(rows, 1.1),
            "phi_nat_rad": np.full(rows, 2.1),
            "q_from_mz": np.full(rows, 3.0),
            "q_from_aperture": np.full(rows, 7.0),
        }
    ).to_csv(path, index=False)

    legacy = module.load_catalogue(path)
    aperture = module.load_catalogue(path, q_column="q_from_aperture")

    assert np.all(legacy["q"] == 3.0)
    assert np.all(aperture["q"] == 7.0)
