"""Summarize and plot the five L1_m9 qfrommap A_SZ-only chains."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from flamingo.inference.l1_m9 import load_bandpower_likelihood  # noqa: E402
from flamingo.inference.masked_ps import MaskedTSZTheory  # noqa: E402
from scripts.run_masked_ps_asz_only_chains import (  # noqa: E402
    ASZ_PRIOR,
    CASES,
    CHAINS,
    FIXED,
    PREFLIGHT_FILE,
    load_converged_artifacts,
)
from scripts.summarize_masked_ps_chains import (  # noqa: E402
    _weighted_quantile,
    read_convergence,
)

BURN_IN_FRACTION = 0.3
MAX_RMINUS1 = 0.005


def summarize_asz_samples(samples) -> dict:
    """Return the A_SZ posterior and minimum data-likelihood chi-square sample."""
    names = samples.getParamNames().list()
    if "A_SZ" not in names:
        raise ValueError("chain is missing A_SZ")
    likelihood_chi2 = [name for name in names if name.startswith("chi2__")]
    if not likelihood_chi2:
        raise ValueError("chain is missing the data-likelihood chi2 column")
    weights = np.asarray(samples.weights, dtype=float)
    values = np.asarray(samples.samples[:, names.index("A_SZ")], dtype=float)
    mean = float(np.average(values, weights=weights))
    standard_deviation = float(
        np.sqrt(np.average((values - mean) ** 2, weights=weights))
    )
    chi2_name = likelihood_chi2[0]
    chi2_values = np.asarray(samples.samples[:, names.index(chi2_name)], dtype=float)
    best_index = int(np.argmin(chi2_values))
    return {
        "A_SZ": {
            "mean": mean,
            "standard_deviation": standard_deviation,
            "interval_68": [
                _weighted_quantile(values, weights, 0.16),
                _weighted_quantile(values, weights, 0.84),
            ],
        },
        "best_likelihood_sample": {
            "A_SZ": float(values[best_index]),
            "chi2": float(chi2_values[best_index]),
            "chi2_column": chi2_name,
        },
        "retained_rows": int(values.size),
        "retained_weight": float(np.sum(weights)),
        "effective_samples": float(samples.getEffectiveSamples(names.index("A_SZ"))),
    }


def plot_getdist(samples_by_case: dict[str, object]) -> dict[str, str]:
    """Plot the five one-dimensional GetDist posteriors."""
    import matplotlib.pyplot as plt
    from getdist import plots

    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    labels = ["full sky", "q>50", "q>20", "q>10", "q>5"]
    plotter = plots.get_single_plotter(width_inch=7)
    plotter.settings.legend_fontsize = 10
    plotter.plot_1d(
        [samples_by_case[case] for case in CASES],
        "A_SZ",
        normalized=True,
        colors=colors,
    )
    plotter.add_legend(labels)
    products = {}
    for suffix in ("png", "pdf"):
        path = CHAINS / f"getdist_posterior_A_SZ.{suffix}"
        plotter.export(str(path))
        products[suffix] = str(path.resolve())
    plt.close("all")
    return products


def evaluate_fit(case: str, a_sz: float, artifacts: dict) -> dict:
    """Re-evaluate theory and per-bin residuals at one fitted amplitude."""
    preflight = json.loads(PREFLIGHT_FILE.read_text())
    case_config = preflight["cases"][case]
    ell, observed, covariance, inverse_covariance = load_bandpower_likelihood(
        case_config["data_file"],
        case_config["covariance_file"],
        data_scale=1e-12,
    )
    theory_component = MaskedTSZTheory({"q_cat": CASES[case]}, timing=True)
    terms = theory_component.evaluate_bandpowers(A_SZ=a_sz, **FIXED)
    theory = np.asarray(terms["1h"] + terms["2h"], dtype=float)
    residual = observed - theory
    cholesky = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(cholesky, residual)
    chi2 = float(residual @ inverse_covariance @ residual)
    from scipy.stats import chi2 as chi2_distribution

    return {
        "ell": np.asarray(ell),
        "observed": observed,
        "theory": theory,
        "covariance": covariance,
        "diagonal_pull": residual / np.sqrt(np.diag(covariance)),
        "whitened_residual": whitened,
        "chi2": chi2,
        "degrees_of_freedom": 17,
        "reduced_chi2": chi2 / 17.0,
        "p_value": float(chi2_distribution.sf(chi2, 17)),
        "covariance_file": case_config["covariance_file"],
        "covariance_A_SZ": float(artifacts["A_SZ"]),
    }


def plot_fit(case: str, fit: dict) -> dict[str, str]:
    """Plot bandpowers and per-bin residual diagnostics for one case."""
    import matplotlib.pyplot as plt

    labels = {
        "fullsky": "full sky",
        "qgt50": "q>50",
        "qgt20": "q>20",
        "qgt10": "q>10",
        "qgt5": "q>5",
    }
    ell = fit["ell"]
    scale = 1e12
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0, 1.0]},
    )
    axes[0].errorbar(
        ell,
        scale * fit["observed"],
        yerr=scale * np.sqrt(np.diag(fit["covariance"])),
        fmt="o",
        ms=3.5,
        capsize=2,
        label="L1_m9 data",
    )
    axes[0].plot(ell, scale * fit["theory"], label="best-fit theory")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$10^{12}D_\ell^{yy}$")
    axes[0].set_title(
        f"{labels[case]}: $\\chi^2={fit['chi2']:.2f}$ for 17 dof"
    )
    axes[0].legend()
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].plot(ell, fit["diagonal_pull"], "o-")
    axes[1].set_ylabel("diag. pull")
    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[2].plot(ell, fit["whitened_residual"], "o-")
    axes[2].set_ylabel("whitened")
    axes[2].set_xlabel(r"$\ell$")
    axes[2].set_xscale("log")
    fig.tight_layout()
    products = {}
    for suffix in ("png", "pdf"):
        path = CHAINS / case / f"bandpowers_residuals.{suffix}"
        fig.savefig(path, dpi=180)
        products[suffix] = str(path.resolve())
    plt.close(fig)
    return products


def _constraint_table(summaries: dict[str, dict]) -> str:
    lines = [
        "case mean standard_deviation lower_68 upper_68 best_likelihood_A_SZ chi2 dof reduced_chi2 p_value"
    ]
    for case in CASES:
        constraint = summaries[case]["A_SZ"]
        best = summaries[case]["best_likelihood_sample"]
        fit = summaries[case]["fit_diagnostics"]
        lines.append(
            f"{case} {constraint['mean']:.10g} {constraint['standard_deviation']:.10g} "
            f"{constraint['interval_68'][0]:.10g} {constraint['interval_68'][1]:.10g} "
            f"{best['A_SZ']:.10g} {fit['chi2']:.10g} {fit['degrees_of_freedom']} "
            f"{fit['reduced_chi2']:.10g} {fit['p_value']:.10g}"
        )
    return "\n".join(lines) + "\n"


def main() -> dict:
    from getdist import loadMCSamples

    artifacts = load_converged_artifacts()
    preflight = json.loads(PREFLIGHT_FILE.read_text())
    samples_by_case = {}
    summaries = {}
    for case in CASES:
        case_dir = CHAINS / case
        convergence = read_convergence(
            case_dir / "chain.checkpoint", max_rminus1=MAX_RMINUS1
        )
        samples = loadMCSamples(
            str(case_dir / "chain"),
            settings={"ignore_rows": BURN_IN_FRACTION},
        )
        summary = {
            "case": case,
            "q_cat": CASES[case],
            "burn_in_fraction": BURN_IN_FRACTION,
            "convergence": convergence,
            **summarize_asz_samples(samples),
        }
        fit = evaluate_fit(
            case, summary["best_likelihood_sample"]["A_SZ"], artifacts
        )
        summary["fit_diagnostics"] = {
            key: value
            for key, value in fit.items()
            if key
            not in {
                "ell",
                "observed",
                "theory",
                "covariance",
                "diagonal_pull",
                "whitened_residual",
            }
        }
        summary["residual_plot"] = plot_fit(case, fit)
        (case_dir / "posterior_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        samples_by_case[case] = samples
        summaries[case] = summary

    posterior_plot = plot_getdist(samples_by_case)
    checks = {
        "one_sampled_parameter": all(
            [name for name, config in preflight["params"].items() if "prior" in config]
            == ["A_SZ"]
            for _ in [0]
        ),
        "approved_prior": preflight["fit_prior"]["A_SZ"] == ASZ_PRIOR,
        "common_fullsky_covariance_point": (
            preflight["covariance_parameter_point"]["A_SZ"] == artifacts["A_SZ"]
            and {
                name: preflight["covariance_parameter_point"][name]
                for name in FIXED
            }
            == FIXED
        ),
        "all_converged": all(
            summary["convergence"]["Rminus1_last"] <= MAX_RMINUS1
            for summary in summaries.values()
        ),
        "direct_chi2_matches_chain": all(
            abs(
                summary["fit_diagnostics"]["chi2"]
                - summary["best_likelihood_sample"]["chi2"]
            )
            < 1e-5
            for summary in summaries.values()
        ),
        "plots_exist": all(Path(path).is_file() for path in posterior_plot.values())
        and all(
            Path(path).is_file()
            for summary in summaries.values()
            for path in summary["residual_plot"].values()
        ),
    }
    combined = {
        "analysis": "L1_m9 qfrommap A_SZ-only with fixed full-sky best-fit covariance point",
        "fit_prior": {"A_SZ": ASZ_PRIOR},
        "fixed_fit_parameters": FIXED,
        "covariance_parameter_point": preflight["covariance_parameter_point"],
        "cases": summaries,
        "posterior_plot": posterior_plot,
        "audit": {"passed": all(checks.values()), "checks": checks},
    }
    summary_path = CHAINS / "posterior_summary.json"
    summary_path.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    (CHAINS / "posterior_constraints.txt").write_text(_constraint_table(summaries))
    print(json.dumps({"summary": str(summary_path), "audit": combined["audit"]}, indent=2))
    if not combined["audit"]["passed"]:
        raise RuntimeError(f"completion audit failed: {summary_path}")
    return combined


if __name__ == "__main__":
    main()
