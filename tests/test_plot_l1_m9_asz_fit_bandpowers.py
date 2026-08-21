from pathlib import Path

from flamingo.inference.masked_ps import GNFW_SHAPE


def test_asz_plot_finds_summary_and_writes_1h_2h_stem():
    from scripts.plot_l1_m9_asz_fit_bandpowers import PANELS, chains_dir

    summary = chains_dir() / "posterior_summary.json"
    assert summary.is_file()
    assert [case for case, _ in PANELS] == ["fullsky", "qgt50", "qgt20", "qgt10", "qgt5"]
    assert Path("figures/masked_ps")


def test_extended_profile_plot_is_shallower_and_not_the_paper_stem():
    from scripts.plot_l1_m9_extended_profile_bandpowers import BETAS, STEM

    betas = [beta for beta, _, _ in BETAS]
    assert betas[0] == GNFW_SHAPE["beta"]
    assert all(beta <= GNFW_SHAPE["beta"] for beta in betas)
    assert min(betas) < GNFW_SHAPE["beta"]
    assert STEM.name == "l1_m9_extended_profile_qfrommap"
    assert "1h_2h" not in STEM.name
