from pathlib import Path


def test_asz_plot_finds_summary_and_writes_1h_2h_stem():
    from scripts.plot_l1_m9_asz_fit_bandpowers import PANELS, chains_dir

    summary = chains_dir() / "posterior_summary.json"
    assert summary.is_file()
    assert [case for case, _ in PANELS] == ["fullsky", "qgt50", "qgt20", "qgt10", "qgt5"]
    assert Path("figures/masked_ps")
