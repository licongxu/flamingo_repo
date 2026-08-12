import json

import pytest

from scripts.inference import run_masked_ps_asz_only_chains as runner


def _artifacts(tmp_path):
    covariance = tmp_path / "cov.npy"
    covariance.touch()
    return {
        "A_SZ": -4.1075073,
        "covariance_paths": {case: covariance for case in runner.CASES},
        "metadata": {"A_SZ": -4.1075073},
        "summary": {
            "fixed_parameters": runner.FIXED,
            "final_metadata_path": str(tmp_path / "metadata.json"),
        },
    }


def test_only_asz_is_sampled_and_everything_else_is_fixed():
    params = runner.parameters()

    sampled = [name for name, config in params.items() if "prior" in config]
    assert sampled == ["A_SZ"]
    assert params["A_SZ"]["prior"] == {
        "dist": "norm",
        "loc": -4.09532,
        "scale": 0.03,
    }
    assert params["H0"]["value"] == 68.1
    assert params["alpha_SZ"]["value"] == 1.12
    assert params["sigma_lnY"]["value"] == 0.173
    assert params["B"]["value"] == 1.41


@pytest.mark.parametrize(
    ("case", "q_cat"),
    [("fullsky", None), ("qgt50", 50.0), ("qgt20", 20.0), ("qgt10", 10.0), ("qgt5", 5.0)],
)
def test_each_case_uses_its_q_theory_and_fixed_fullsky_point_covariance(
    tmp_path, case, q_cat
):
    artifacts = _artifacts(tmp_path)

    info = runner.build_info(case, artifacts=artifacts)

    theory = next(iter(info["theory"].values()))
    likelihood = next(iter(info["likelihood"].values()))
    assert theory["q_cat"] == q_cat
    assert likelihood["covariance_file"] == str(artifacts["covariance_paths"][case])
    assert info["params"]["A_SZ"]["prior"]["loc"] != artifacts["A_SZ"]


def test_preflight_records_distinct_covariance_and_prior_amplitudes(
    tmp_path, monkeypatch
):
    artifacts = _artifacts(tmp_path)
    output = tmp_path / "preflight.json"
    monkeypatch.setattr(runner, "gpu_devices", lambda: ["cuda:0"])

    runner.write_preflight(("fullsky",), artifacts, output_file=output)

    written = json.loads(output.read_text())
    assert written["covariance_parameter_point"]["A_SZ"] == -4.1075073
    assert written["fit_prior"]["A_SZ"] == {
        "dist": "norm",
        "loc": -4.09532,
        "scale": 0.03,
    }
