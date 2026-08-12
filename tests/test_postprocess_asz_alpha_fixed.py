from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO / "scripts/inference/postprocess_asz_alpha_fixed.py"
    spec = importlib.util.spec_from_file_location("postprocess_asz_alpha_fixed", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_case_config_preserves_l1_outputs():
    config = _load_module().case_config("L1_m9")

    assert config["chain_dir"] == Path("chains/l1_m9_customgnfw_asz_alpha_fixed_1p12")
    assert config["fixed"] == {"alpha_SZ": 1.12, "B": 1.41}
    assert config["summary_prefix"] == []
    assert config["figure_stem"] == "posterior_A_SZ"
    assert config["plot_label"] is None


def test_case_config_preserves_l2_outputs():
    config = _load_module().case_config("L2p8_m9")

    assert config["chain_dir"] == Path("chains/L2p8_m9_customgnfw_asz_alpha_fixed_1p12_B1p41")
    assert config["fixed"] == {"alpha_SZ": 1.12, "B": 1.41, "cosmology": "D3A"}
    assert config["summary_prefix"] == ["simulation = L2p8_m9", "fixed cosmology = D3A"]
    assert config["figure_stem"] == "posterior_L2p8_m9_A_SZ"
    assert config["plot_label"] == "L2p8_m9"
