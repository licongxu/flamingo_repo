from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
L2_ROOT = Path("/rds/rds-lxu/flamingo/L2p8_m9")


def _load_module():
    path = REPO / "scripts/backfill_l2_y5r500.py"
    spec = importlib.util.spec_from_file_location("backfill_l2_lightcones_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("lightcone", [0, 2, 7])
def test_catalogue_path_selects_requested_lightcone(lightcone):
    module = _load_module()

    assert module.catalogue_path(lightcone) == L2_ROOT / (
        f"lightcone{lightcone}/catalogues/"
        "halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_qfrommap.csv"
    )


@pytest.mark.parametrize("lightcone", [0, 2, 7])
def test_target_key_selects_requested_lightcone(lightcone):
    module = _load_module()

    assert module.target_key(lightcone) == f"L2p8_m9/lightcone{lightcone}"


@pytest.mark.parametrize("lightcone", [-1, 8])
def test_lightcone_helpers_reject_out_of_range_indices(lightcone):
    module = _load_module()

    with pytest.raises(ValueError, match="lightcone must be in 0..7"):
        module.catalogue_path(lightcone)
    with pytest.raises(ValueError, match="lightcone must be in 0..7"):
        module.target_key(lightcone)
