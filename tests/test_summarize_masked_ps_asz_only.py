import numpy as np
import pytest
from getdist import MCSamples

from scripts.inference.summarize_masked_ps_asz_only import summarize_asz_samples


def test_asz_summary_reports_constraint_and_data_chi2():
    names = [
        "A_SZ",
        "minuslogprior",
        "chi2",
        "chi2__flamingo.inference.masked_ps.MaskedBandPowerLikelihood",
    ]
    values = np.zeros((200, len(names)))
    values[:, 0] = np.linspace(-4.2, -4.0, 200)
    values[:, 1] = np.linspace(0.0, 2.0, 200)
    values[:, 2] = np.linspace(5.0, 30.0, 200)
    values[:, 3] = np.linspace(60.0, 20.0, 200)
    samples = MCSamples(
        samples=values,
        names=names,
        labels=names,
        weights=np.linspace(0.5, 1.5, 200),
        settings={"ignore_rows": 0},
    )

    summary = summarize_asz_samples(samples)

    assert summary["A_SZ"]["mean"] == pytest.approx(
        np.average(values[:, 0], weights=samples.weights)
    )
    assert summary["A_SZ"]["standard_deviation"] > 0
    assert summary["A_SZ"]["interval_68"][0] < summary["A_SZ"]["interval_68"][1]
    assert summary["best_likelihood_sample"]["chi2"] == pytest.approx(20.0)
    assert summary["best_likelihood_sample"]["A_SZ"] == pytest.approx(-4.0)
    assert summary["best_likelihood_sample"]["chi2_column"].startswith("chi2__")
