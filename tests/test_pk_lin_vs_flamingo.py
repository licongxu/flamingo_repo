"""FLAMINGO hdf5 is Mpc^{-1} / Mpc^3. hmfast pk() docstring is not proof of that."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO / "scripts" / "plot_pk_lin_vs_flamingo_sim.py"
    spec = importlib.util.spec_from_file_location("pk_lin_vs_flamingo_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_flamingo_matches_colossus_in_mpc_units_on_linear_scales():
    module = _load_script()
    k, p = module._sim_pk(module.HYDRO, 0.0)
    k_lin = 0.01
    p_sim = np.interp(k_lin, k, p)
    p_col = float(module._colossus_pk_mpc(0.0, np.array([k_lin]))[0])
    np.testing.assert_allclose(p_sim / p_col, 1.0, rtol=0.05)


def test_hmfast_pk_is_not_the_same_mpc3_amplitude_as_colossus():
    module = _load_script()
    k, p = module._hmfast_pk(0.0)
    k_lin = 0.01
    p_hm = np.interp(k_lin, k, p)
    p_col = float(module._colossus_pk_mpc(0.0, np.array([k_lin]))[0])
    ratio = p_hm / p_col
    # Same k [1/Mpc]. If pk() were Mpc^3, this would be ~1. It is ~1/h^3.
    assert 2.8 < ratio < 3.5
