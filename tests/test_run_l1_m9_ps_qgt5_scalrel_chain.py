from scripts import run_l1_m9_cnc_qgt5_chain as cnc
from scripts import run_l1_m9_ps_qgt5_scalrel_chain as ps
from scripts import plot_l1_m9_cnc_qgt5_triangle as triangle


def test_ps_uses_exactly_the_cnc_scalrel_priors():
    cnc_params = cnc.parameters()
    ps_params = ps.parameters()
    sampled = [name for name, config in ps_params.items() if "prior" in config]
    assert sampled == ["A_SZ", "alpha_SZ", "sigma_lnY"]
    for name in sampled:
        assert ps_params[name]["prior"] == cnc_params[name]["prior"]
        assert ps_params[name]["ref"] == cnc_params[name]["ref"]
        assert ps_params[name]["proposal"] == cnc_params[name]["proposal"]
    for name in ("H0", "sigma_8", "omega_cdm", "omega_b", "n_s", "tau_reio", "B"):
        assert ps_params[name]["value"] == cnc_params[name]["value"]


def test_ps_chain_is_qgt5_bandpowers(tmp_path):
    covariance = tmp_path / "cov.npy"
    covariance.touch()
    artifacts = {
        "A_SZ": -4.1075073,
        "covariance_paths": {"qgt5": covariance},
        "metadata": {"A_SZ": -4.1075073},
        "summary": {"fixed_parameters": {**cnc.FIXED, "alpha_SZ": 1.12, "sigma_lnY": 0.173}},
    }
    data = tmp_path / "Dl.txt"
    data.write_text("1 1\n")

    def fake_data_files(case, paths):
        assert case == "qgt5"
        return data, paths[case]

    import scripts.run_l1_m9_ps_qgt5_scalrel_chain as mod

    original = mod.data_files
    mod.data_files = fake_data_files
    try:
        info = ps.build_info(artifacts=artifacts)
    finally:
        mod.data_files = original

    assert next(iter(info["theory"].values()))["q_cat"] == 5.0
    assert info["params"]["A_SZ"]["prior"] == cnc.parameters()["A_SZ"]["prior"]
    assert info["output"] == str(ps.CHAINS / "chain")
    likelihood = next(iter(info["likelihood"].values()))
    assert likelihood["data_file"] == str(data)
    assert likelihood["covariance_file"] == str(covariance)


def test_comparison_triangle_loads_both_chains():
    assert triangle.PARAMS == ("A_SZ", "alpha_SZ", "sigma_lnY")
    assert triangle.PS_CHAIN_ROOT == ps.CHAINS / "chain"
    assert triangle.COMPARE_STEM.name == "l1_m9_qgt5_cnc_vs_ps_triangle"


def test_bestfit_bandpower_plot_uses_qgt5_map():
    from scripts import plot_l1_m9_ps_qgt5_scalrel_bandpowers as band

    assert band.CASE == "qgt5"
    assert band.STEM.name == "l1_m9_ps_qgt5_scalrel_bestfit_qfrommap"
    params = band.map_params()
    assert params["sigma_lnY"] > 0.05
    assert params["chi2"] > 0.0
