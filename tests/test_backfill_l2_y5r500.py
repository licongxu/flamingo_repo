from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO / "scripts/backfill_l2_y5r500.py"
    spec = importlib.util.spec_from_file_location("backfill_l2_y5r500_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_append_column_preserves_comments_and_complete_old_rows(tmp_path):
    module = _load_module()
    source = tmp_path / "source.csv"
    output = tmp_path / "output.csv"
    source.write_text(
        "# catalogue provenance\n"
        "snap,soap_index,z\n"
        "18,2,0.10000000000000001\n"
        "19,3,0.2\n"
    )

    summary = module.append_column_preserving_rows(
        source, output, "Y_5R500c_Mpc2", np.array([1.25, 2.5])
    )

    assert output.read_text() == (
        "# catalogue provenance\n"
        "snap,soap_index,z,Y_5R500c_Mpc2\n"
        "18,2,0.10000000000000001,1.25\n"
        "19,3,0.2,2.5\n"
    )
    assert summary == {"rows": 2, "old_rows_sha256": summary["new_prefix_sha256"],
                       "new_prefix_sha256": summary["old_rows_sha256"]}


def test_append_column_rejects_existing_target_column(tmp_path):
    module = _load_module()
    source = tmp_path / "source.csv"
    source.write_text("snap,soap_index,Y_5R500c_Mpc2\n18,2,1.0\n")

    with pytest.raises(ValueError, match="already exists"):
        module.append_column_preserving_rows(
            source, tmp_path / "output.csv", "Y_5R500c_Mpc2", np.array([1.25])
        )


@pytest.mark.parametrize("values", [np.array([np.nan]), np.array([-1.0])])
def test_append_column_rejects_invalid_soap_values(tmp_path, values):
    module = _load_module()
    source = tmp_path / "source.csv"
    source.write_text("snap,soap_index,z\n18,2,0.1\n")

    with pytest.raises(ValueError, match="finite and non-negative"):
        module.append_column_preserving_rows(
            source, tmp_path / "output.csv", "Y_5R500c_Mpc2", values
        )
