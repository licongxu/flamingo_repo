import numpy as np

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
        info = ps.build_info("qgt5", artifacts=artifacts)
    finally:
        mod.data_files = original

    assert next(iter(info["theory"].values()))["q_cat"] == 5.0
    assert info["params"]["A_SZ"]["prior"] == cnc.parameters()["A_SZ"]["prior"]
    assert info["output"] == str(ps.chain_dir("qgt5") / "chain")
    likelihood = next(iter(info["likelihood"].values()))
    assert likelihood["data_file"] == str(data)
    assert likelihood["covariance_file"] == str(covariance)


def test_other_cases_keep_cnc_priors_and_q_thresholds(tmp_path):
    covariance = tmp_path / "cov.npy"
    covariance.touch()
    artifacts = {
        "A_SZ": -4.1075073,
        "covariance_paths": {
            case: covariance for case in ("fullsky", "qgt50", "qgt20", "qgt10")
        },
        "metadata": {"A_SZ": -4.1075073},
        "summary": {
            "fixed_parameters": {**cnc.FIXED, "alpha_SZ": 1.12, "sigma_lnY": 0.173}
        },
    }
    data = tmp_path / "Dl.txt"
    data.write_text("1 1\n")

    import scripts.run_l1_m9_ps_qgt5_scalrel_chain as mod

    original = mod.data_files
    mod.data_files = lambda case, paths: (data, paths[case])
    try:
        infos = {
            case: ps.build_info(case, artifacts=artifacts)
            for case in ("fullsky", "qgt50", "qgt20", "qgt10")
        }
    finally:
        mod.data_files = original

    assert [next(iter(info["theory"].values()))["q_cat"] for info in infos.values()] == [
        None,
        50.0,
        20.0,
        10.0,
    ]
    for case, info in infos.items():
        assert info["params"]["A_SZ"]["prior"] == cnc.parameters()["A_SZ"]["prior"]
        assert info["params"]["alpha_SZ"]["prior"] == cnc.parameters()["alpha_SZ"]["prior"]
        assert info["params"]["sigma_lnY"]["prior"] == cnc.parameters()["sigma_lnY"]["prior"]
        assert info["output"] == str(ps.chain_dir(case) / "chain")


def test_comparison_triangle_loads_both_chains():
    assert triangle.PARAMS == ("A_SZ", "alpha_SZ", "sigma_lnY")
    assert triangle.PS_CHAIN_ROOT == ps.CHAINS / "chain"
    assert triangle.COMPARE_STEM.name == "l1_m9_qgt5_cnc_vs_ps_triangle"
    assert triangle.PS_CASES == ("fullsky", "qgt50", "qgt20", "qgt10", "qgt5")
    assert triangle.PS_STEM.name == "l1_m9_ps_scalrel_triangle"


def test_bestfit_bandpower_plot_uses_qgt5_map():
    from scripts import plot_l1_m9_ps_qgt5_scalrel_bandpowers as band

    assert band.CASE == "qgt5"
    assert band.STEM.name == "l1_m9_ps_qgt5_scalrel_bestfit_qfrommap"
    assert band.STEM_ALL.name == "l1_m9_ps_scalrel_bestfit_qfrommap"
    assert [case for case, _ in band.PANELS] == [
        "fullsky", "qgt50", "qgt20", "qgt10", "qgt5",
    ]
    params = band.map_params()
    assert params["sigma_lnY"] > 0.05
    assert params["chi2"] > 0.0


def test_shot_noise_is_sum_y5r500_sq_over_4pi():
    from scripts import plot_l1_m9_ps_qgt5_scalrel_bandpowers as band

    y_mpc2 = np.array([1.0, 2.0])
    d_a = np.array([10.0, 20.0])
    q = np.array([3.0, 30.0])
    y_sr = y_mpc2 / d_a**2
    assert np.isclose(band.shot_noise_c(y_mpc2, d_a), np.sum(y_sr**2) / (4.0 * np.pi))
    assert np.isclose(
        band.shot_noise_c(y_mpc2, d_a, q=q, q_cut=5.0),
        y_sr[0] ** 2 / (4.0 * np.pi),
    )
    ell = np.array([10.0, 100.0])
    c_shot = band.shot_noise_c(y_mpc2, d_a)
    assert np.allclose(
        band.shot_noise_dl(ell, c_shot),
        ell * (ell + 1.0) / (2.0 * np.pi) * c_shot,
    )


def test_all_figure_draws_1h_2h_and_shot_noise():
    from scripts import plot_l1_m9_ps_qgt5_scalrel_bandpowers as band

    ell = np.geomspace(10.0, 900.0, 18)
    ones = np.full(18, 1.0e-12)
    fits = {
        case: {
            "ell": ell,
            "observed": ones,
            "err": ones * 0.1,
            "total": ones * 2.0,
            "1h": ones,
            "2h": ones,
            "shot_noise": ones * np.linspace(0.2, 3.0, 18),
        }
        for case, _ in band.PANELS
    }
    rc = dict(band.PAPER_RC)
    rc["text.usetex"] = False
    band.plt.rcParams.update(rc)
    fig = band.build_all_figure(fits)
    try:
        styles = {line.get_linestyle() for line in fig.axes[0].lines}
        assert ":" in styles
        assert "--" in styles
        assert "-." in styles
    finally:
        band.plt.close(fig)
