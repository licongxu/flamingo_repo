import json

import pytest

from scripts.inference import run_masked_ps_chains as runner


@pytest.mark.parametrize("case", ["qgt50", "qgt20", "qgt10", "qgt5"])
def test_masked_cases_use_qfrommap_data(case):
    data, _ = runner.data_files(case)

    assert data.name == f"Dl_yy_L1_m9_masked_{case}_qfrommap_binned_18.txt"


def test_d3a_prior_centers_wider_asz_and_fixed_B():
    params = runner.parameters(a_sz_best_fit=-4.1075073)

    assert params["H0"]["prior"]["loc"] == 68.1
    assert params["n_s"]["prior"]["loc"] == 0.965
    assert params["omega_b"]["prior"]["loc"] == pytest.approx(0.0225387846)
    assert params["A_SZ"]["prior"] == {
        "dist": "norm",
        "loc": -4.1075073,
        "scale": 0.03,
    }
    assert params["B"]["value"] == 1.41
    assert params["tau_reio"]["value"] == 0.0544
    assert params["sigma_8"]["prior"] == {"min": 0.6, "max": 1.0}
    assert params["Omega_m"]["prior"] == {"min": 0.2, "max": 0.5}


def test_omega_cdm_subtracts_the_fixed_neutrino_density():
    got = runner.omega_cdm_from_omegam(
        Omega_m=0.306,
        H0=68.1,
        omega_b=0.022538784599999993,
    )

    assert got == pytest.approx(0.11872788986038219)


def test_converged_artifacts_reject_covariance_at_a_different_amplitude(
    tmp_path,
    monkeypatch,
):
    covariance = tmp_path / "cov.npy"
    covariance.write_bytes(b"placeholder")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "A_SZ": -4.2,
                "cases": {
                    case: {"paths": {"cov_full": str(covariance)}}
                    for case in runner.CASES
                },
            }
        )
    )
    summary = tmp_path / "converged.json"
    summary.write_text(
        json.dumps(
            {
                "converged": True,
                "best_fit": {"A_SZ": -4.1},
                "final_metadata_path": str(metadata),
            }
        )
    )
    monkeypatch.setattr(runner, "CONVERGED_FILE", summary)

    with pytest.raises(ValueError, match="A_SZ"):
        runner.load_converged_artifacts()


def test_preflight_can_be_written_to_a_case_isolated_path(tmp_path, monkeypatch):
    covariance = tmp_path / "cov.npy"
    covariance.touch()
    artifacts = {
        "A_SZ": -4.1075073,
        "covariance_paths": {case: covariance for case in runner.CASES},
        "metadata": {"A_SZ": -4.1075073},
        "summary": {"final_metadata_path": str(tmp_path / "metadata.json")},
    }
    output_file = tmp_path / "qgt50" / "preflight.json"
    monkeypatch.setattr(runner, "gpu_devices", lambda: ["cuda:0"])

    runner.write_preflight(
        ("qgt50",),
        artifacts,
        output_file=output_file,
    )

    assert output_file.is_file()
    written = json.loads(output_file.read_text())
    assert set(written["cases"]) == {"qgt50"}


def test_science_chain_can_use_a_learned_proposal_covariance(tmp_path):
    covariance = tmp_path / "likelihood_cov.npy"
    covariance.touch()
    proposal_covariance = tmp_path / "proposal.covmat"
    proposal_covariance.write_text("# H0\n1.0\n")
    artifacts = {
        "A_SZ": -4.1075073,
        "covariance_paths": {case: covariance for case in runner.CASES},
    }

    info = runner.build_info(
        "fullsky",
        artifacts=artifacts,
        covmat_file=proposal_covariance,
    )

    assert info["sampler"]["mcmc"]["covmat"] == str(proposal_covariance.resolve())
    assert info["sampler"]["mcmc"]["learn_proposal_Rminus1_max"] == 100.0
    assert info["sampler"]["mcmc"]["learn_proposal_Rminus1_max_early"] == 100.0
