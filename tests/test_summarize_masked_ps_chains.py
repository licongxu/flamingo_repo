import numpy as np
import pytest
from getdist import MCSamples

from scripts.summarize_masked_ps_chains import (
    REQUIRED_PARAMETERS,
    read_convergence,
    summarize_samples,
)


def test_checkpoint_must_report_converged(tmp_path):
    checkpoint = tmp_path / "chain.checkpoint"
    checkpoint.write_text(
        "sampler:\n  mcmc:\n    converged: false\n    Rminus1_last: 0.03\n"
    )

    with pytest.raises(RuntimeError, match="not converged"):
        read_convergence(checkpoint)


def test_checkpoint_must_meet_rminus1_threshold(tmp_path):
    checkpoint = tmp_path / "chain.checkpoint"
    checkpoint.write_text(
        "sampler:\n  mcmc:\n    converged: true\n    Rminus1_last: 0.02\n"
    )

    with pytest.raises(RuntimeError, match="Rminus1"):
        read_convergence(checkpoint)


def test_summary_contains_numerical_constraints_for_required_parameters():
    rng = np.random.default_rng(20260802)
    names = [*REQUIRED_PARAMETERS, "chi2__masked_bandpowers"]
    samples_array = rng.normal(size=(600, len(names)))
    samples_array[:, -1] = np.linspace(30.0, 80.0, 600)
    samples = MCSamples(
        samples=samples_array,
        names=names,
        labels=names,
        weights=np.linspace(0.5, 1.5, 600),
        settings={"ignore_rows": 0},
    )

    summary = summarize_samples(samples)

    assert set(REQUIRED_PARAMETERS) <= set(summary["parameters"])
    for name in REQUIRED_PARAMETERS:
        constraint = summary["parameters"][name]
        assert set(constraint) == {"mean", "standard_deviation", "interval_68"}
        assert len(constraint["interval_68"]) == 2
        assert constraint["interval_68"][0] < constraint["interval_68"][1]
    assert summary["best_sample"]["chi2"] == pytest.approx(30.0)
    assert summary["retained_weight"] == pytest.approx(np.sum(samples.weights))


def test_summary_uses_data_likelihood_chi2_not_total_posterior_chi2():
    rng = np.random.default_rng(42)
    names = [*REQUIRED_PARAMETERS, "chi2", "chi2__MaskedBandPowerLikelihood"]
    samples_array = rng.normal(size=(100, len(names)))
    samples_array[:, -2] = np.linspace(1.0, 100.0, 100)
    samples_array[:, -1] = np.linspace(50.0, 20.0, 100)
    samples = MCSamples(
        samples=samples_array,
        names=names,
        labels=names,
        settings={"ignore_rows": 0},
    )

    summary = summarize_samples(samples)

    assert summary["best_sample"]["chi2_column"] == "chi2__MaskedBandPowerLikelihood"
    assert summary["best_sample"]["chi2"] == pytest.approx(20.0)
