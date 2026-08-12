from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def test_validation_reader_round_trips_17_digit_soap_value(tmp_path):
    script = REPO / "scripts/catalogue/backfill_l2_y5r500.py"
    spec = importlib.util.spec_from_file_location("backfill_l2_roundtrip_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    path = tmp_path / "catalogue.csv"
    expected = np.array([1.7732381820678711e-06])
    path.write_text(
        "snap,soap_index,Y_5R500c_Mpc2\n"
        "50,123,1.7732381820678711e-06\n"
    )

    frame = module.read_backfill_for_validation(path)

    assert np.array_equal(frame["Y_5R500c_Mpc2"].to_numpy(), expected)
