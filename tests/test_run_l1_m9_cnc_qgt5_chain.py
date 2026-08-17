import os

import numpy as np
import pandas as pd

from scripts import plot_l1_m9_cnc_qgt5_triangle as triangle
from scripts import run_l1_m9_cnc_qgt5_chain as runner


def test_only_scalrel_params_are_sampled():
    params = runner.parameters()
    sampled = [name for name, config in params.items() if "prior" in config]
    assert sampled == ["A_SZ", "alpha_SZ", "sigma_lnY"]
    assert params["A_SZ"]["prior"] == {"min": -4.41, "max": -4.00}
    assert params["alpha_SZ"]["prior"] == {"min": 0.8, "max": 1.2}
    assert params["sigma_lnY"]["prior"] == {
        "dist": "norm",
        "loc": 0.173,
        "scale": 0.05,
    }
    assert params["H0"]["value"] == 68.1
    assert params["sigma_8"]["value"] == 0.8025701499024616
    assert params["B"]["value"] == 1.41
    assert params["Omega_m"]["value"] == 0.306


def test_theory_matches_case03_cnc_chain(tmp_path):
    data = tmp_path / "N2d.txt"
    data.write_text("0\n")
    info = runner.build_info(data_file=data)
    likelihood = next(iter(info["likelihood"].values()))
    assert next(iter(info["likelihood"])) == runner.LIKELIHOOD
    assert likelihood["n_points"] == 2048
    assert likelihood["n_z"] == 50
    assert likelihood["q_min"] == 5.0
    assert likelihood["q_max"] == 40.0
    assert likelihood["n_q_bins"] == 5
    assert likelihood["n_z_bins"] == 10
    assert likelihood["M_pivot"] == 2.1e14
    assert likelihood["hmfast_path"] == str(runner.HMFAST_SRC)
    assert likelihood["data_file"] == str(data)
    assert info["output"] == str(runner.CHAINS / "chain")
    assert set(likelihood["input_params"]) == set(info["params"])


def test_write_counts_bins_qfrommap_qgt5(tmp_path):
    catalogue = tmp_path / "cat.csv"
    pd.DataFrame(
        {
            "z": [0.005, 0.2, 1.0, 0.004, 0.5, 0.5],
            "q_from_aperture": [5.0, 6.0, 40.0, 10.0, 4.99, 40.01],
        }
    ).to_csv(catalogue, index=False)
    output = tmp_path / "N2d.txt"

    counts = runner.write_counts(output, catalogue=catalogue)

    assert counts.shape == (10, 5)
    assert counts.sum() == 3
    loaded = np.loadtxt(output)
    assert np.array_equal(loaded, counts)


def test_empty_cuda_visible_devices_defaults_to_gpu1(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    runner._prepare_runtime()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_triangle_plots_the_three_sampled_scalrel_params():
    sampled = [name for name, config in runner.parameters().items() if "prior" in config]
    assert list(triangle.PARAMS) == sampled
    assert triangle.CHAIN_ROOT == runner.CHAINS / "chain"
    assert triangle.STEM.name == "triangle_A_SZ_alpha_SZ_sigma_lnY"
