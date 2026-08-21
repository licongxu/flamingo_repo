import os

import numpy as np
import pandas as pd
import pytest

from scripts import plot_l1_m9_cnc_qgt5_triangle as triangle
from scripts import run_l1_m9_cnc_qgt5_chain as runner


def test_only_scalrel_params_are_sampled():
    params = runner.parameters()
    sampled = [name for name, config in params.items() if "prior" in config]
    assert sampled == ["A_SZ", "alpha_SZ", "sigma_lnY"]
    assert params["A_SZ"]["prior"] == {"min": -4.41, "max": -4.00}
    assert params["alpha_SZ"]["prior"] == {"min": 0.5, "max": 1.3}
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

    sub_info = runner.build_info(
        data_file=data,
        output=tmp_path / "submono" / "chain",
    )
    assert sub_info["output"] == str(tmp_path / "submono" / "chain")
    assert sub_info["likelihood"][runner.LIKELIHOOD]["data_file"] == str(data)

    mono = runner.build_info(
        data_file=data,
        output=tmp_path / "addmono" / "chain",
        add_monopole=True,
        y_monopole=1.5e-6,
    )
    lik = mono["likelihood"][runner.LIKELIHOOD]
    assert lik["add_monopole"] is True
    assert lik["y_monopole"] == 1.5e-6
    assert lik["tszsbi_sigma_Y500_file"] == str(runner.Y500_NOISE)
    assert "add_monopole" not in info["likelihood"][runner.LIKELIHOOD]


def test_q_aperture_submono_subtracts_mean_times_area():
    pixarea_sr = 1.0 / runner.ARCMIN2_PER_SR
    q = runner.q_aperture_submono(
        np.array([2.0]), np.array([10.0]), np.array([0.5]), 0.1, pixarea_sr
    )
    assert q == pytest.approx((2.0 - 0.1 * 10.0) / 0.5)


def test_write_counts_submono_bins_corrected_q(tmp_path):
    catalogue = tmp_path / "cat.csv"
    pd.DataFrame(
        {
            "z": [0.2, 0.2, 0.2],
            "Y_500cyl_arcmin2": [6.0, 6.0, 6.0],
            "sigma_Y500_arcmin2": [1.0, 1.0, 1.0],
            "npix_in_aperture": [10.0, 10.0, 10.0],
        }
    ).to_csv(catalogue, index=False)
    pixarea_sr = 1.0 / runner.ARCMIN2_PER_SR
    output = tmp_path / "N2d.txt"
    # ybar=0 -> q=6, all three enter; ybar=0.2 -> q=4, all drop below q=5.
    counts_raw = runner.write_counts_submono(
        output, catalogue, ybar=0.0, pixarea_sr=pixarea_sr
    )
    counts_sub = runner.write_counts_submono(
        output, catalogue, ybar=0.2, pixarea_sr=pixarea_sr
    )
    assert int(counts_raw.sum()) == 3
    assert int(counts_sub.sum()) == 0


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
    assert triangle.SUBMONO_STEM.name == "l1_m9_qgt5_cnc_submono_triangle"
    assert triangle.RAW_VS_SUBMONO_STEM.name == (
        "l1_m9_qgt5_cnc_raw_vs_submono_triangle"
    )
    assert triangle.ADDMONO_STEM.name == "l1_m9_qgt5_cnc_addmono_triangle"
