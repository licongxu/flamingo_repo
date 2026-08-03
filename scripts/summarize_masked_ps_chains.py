"""Validate, summarize, and plot the five L1_m9 qfrommap Cobaya chains."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("MPLBACKEND", "Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from scripts.run_masked_ps_chains import (  # noqa: E402
    CASES,
    CHAINS,
    CONVERGED_FILE,
    PREFLIGHT_FILE,
    load_converged_artifacts,
    parameters,
)

BURN_IN_FRACTION = 0.3
MAX_RMINUS1 = 0.01
REQUIRED_PARAMETERS = (
    "H0",
    "sigma_8",
    "n_s",
    "omega_b",
    "Omega_m",
    "A_SZ",
    "alpha_SZ",
    "sigma_lnY",
    "S8",
)
COSMOLOGY_PLOT_PARAMETERS = ("H0", "sigma_8", "n_s", "omega_b", "Omega_m", "S8")
SZ_PLOT_PARAMETERS = ("A_SZ", "alpha_SZ", "sigma_lnY")
EXPECTED_F_SKY = {
    "fullsky": 1.0,
    "qgt50": 0.9972647840959187,
    "qgt20": 0.9858059150434103,
    "qgt10": 0.9435688947539681,
    "qgt5": 0.8595492784996496,
}


def read_convergence(checkpoint: Path, *, max_rminus1: float = MAX_RMINUS1) -> dict:
    """Require a genuine Cobaya convergence checkpoint at the target threshold."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = yaml.safe_load(checkpoint.read_text())
    mcmc = payload.get("sampler", {}).get("mcmc", {})
    if mcmc.get("converged") is not True:
        raise RuntimeError(f"chain is not converged: {checkpoint}")
    rminus1 = float(mcmc["Rminus1_last"])
    if rminus1 > max_rminus1:
        raise RuntimeError(
            f"chain Rminus1={rminus1} exceeds {max_rminus1}: {checkpoint}"
        )
    return {
        "converged": True,
        "Rminus1_last": rminus1,
        "checkpoint": str(checkpoint.resolve()),
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= np.sum(weights)
    return float(np.interp(probability, cumulative, values))


def summarize_samples(samples) -> dict:
    """Return posterior constraints and the minimum-chi-square retained sample."""
    names = samples.getParamNames().list()
    missing = set(REQUIRED_PARAMETERS) - set(names)
    if missing:
        raise ValueError(f"chain is missing required parameters: {sorted(missing)}")
    weights = np.asarray(samples.weights, dtype=float)
    if np.any(weights < 0.0) or np.sum(weights) <= 0.0:
        raise ValueError("sample weights must be non-negative with positive sum")

    constraints = {}
    effective_samples = {}
    for name in REQUIRED_PARAMETERS:
        index = names.index(name)
        values = np.asarray(samples.samples[:, index], dtype=float)
        mean = float(np.average(values, weights=weights))
        variance = float(np.average((values - mean) ** 2, weights=weights))
        constraints[name] = {
            "mean": mean,
            "standard_deviation": float(np.sqrt(variance)),
            "interval_68": [
                _weighted_quantile(values, weights, 0.16),
                _weighted_quantile(values, weights, 0.84),
            ],
        }
        effective_samples[name] = float(samples.getEffectiveSamples(index))

    likelihood_chi2 = [name for name in names if name.startswith("chi2__")]
    chi2_name = likelihood_chi2[0] if likelihood_chi2 else "chi2"
    chi2_index = names.index(chi2_name)
    best_index = int(np.argmin(samples.samples[:, chi2_index]))
    best_sample = {
        name: float(samples.samples[best_index, names.index(name)])
        for name in REQUIRED_PARAMETERS
    }
    best_sample.update(
        {
            "chi2": float(samples.samples[best_index, chi2_index]),
            "chi2_column": chi2_name,
        }
    )
    return {
        "parameters": constraints,
        "best_sample": best_sample,
        "retained_rows": int(samples.samples.shape[0]),
        "retained_weight": float(np.sum(weights)),
        "effective_samples": effective_samples,
    }


def plot_getdist_posteriors(samples_by_case: dict[str, object]) -> dict[str, str]:
    """Write combined GetDist posterior plots for cosmology and tSZ parameters."""
    import matplotlib.pyplot as plt
    from getdist import plots

    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    sample_list = [samples_by_case[case] for case in CASES]
    labels = ["full sky", "q>50", "q>20", "q>10", "q>5"]
    products = {}
    for label, plot_parameters in (
        ("cosmology", COSMOLOGY_PLOT_PARAMETERS),
        ("sz", SZ_PLOT_PARAMETERS),
    ):
        plotter = plots.get_subplot_plotter(width_inch=11)
        plotter.settings.num_plot_contours = 2
        plotter.settings.axes_fontsize = 9
        plotter.settings.lab_fontsize = 11
        plotter.settings.legend_fontsize = 9
        plotter.triangle_plot(
            sample_list,
            list(plot_parameters),
            filled=False,
            contour_colors=colors,
            legend_labels=labels,
        )
        for suffix in ("png", "pdf"):
            path = CHAINS / f"getdist_posterior_{label}.{suffix}"
            plotter.export(str(path))
            products[f"{label}_{suffix}"] = str(path.resolve())
        plt.close("all")
    return products


def _constraint_table(case_summaries: dict[str, dict]) -> str:
    lines = ["case parameter mean standard_deviation lower_68 upper_68"]
    for case in CASES:
        for name in REQUIRED_PARAMETERS:
            constraint = case_summaries[case]["parameters"][name]
            lines.append(
                f"{case} {name} {constraint['mean']:.10g} "
                f"{constraint['standard_deviation']:.10g} "
                f"{constraint['interval_68'][0]:.10g} "
                f"{constraint['interval_68'][1]:.10g}"
            )
    return "\n".join(lines) + "\n"


def build_completion_audit(
    case_summaries: dict[str, dict],
    plots: dict[str, str],
) -> dict:
    """Audit every scientific input, covariance, GPU, and convergence invariant."""
    artifacts = load_converged_artifacts()
    metadata = artifacts["metadata"]
    global_preflight = json.loads(PREFLIGHT_FILE.read_text())
    resolved_params = parameters(artifacts["A_SZ"])
    expected_data = {
        "fullsky": "Dl_yy_L1_m9_fullsky_binned_18.txt",
        **{
            case: f"Dl_yy_L1_m9_masked_{case}_qfrommap_binned_18.txt"
            for case in CASES
            if case != "fullsky"
        },
    }

    component_checks = {}
    covariance_positive_definite = {}
    for case in CASES:
        paths = metadata["cases"][case]["paths"]
        gaussian = np.load(paths["cov_gaussian"])
        trispectrum = np.load(paths["cov_trispectrum"])
        covariance = np.load(paths["cov_full"])
        component_checks[case] = bool(np.array_equal(covariance, gaussian + trispectrum))
        covariance_positive_definite[case] = bool(
            np.linalg.eigvalsh(0.5 * (covariance + covariance.T)).min() > 0.0
        )

    case_gpu = {}
    for case in CASES:
        preflight_path = CHAINS / case / "preflight.json"
        case_preflight = json.loads(preflight_path.read_text())
        case_gpu[case] = {
            "preflight": str(preflight_path.resolve()),
            "devices": case_preflight["jax_devices"],
            "passed": all("cuda" in device.lower() for device in case_preflight["jax_devices"]),
        }

    checks = {
        "five_cases": set(case_summaries) == set(CASES),
        "correct_data_files": all(
            Path(global_preflight["cases"][case]["data_file"]).name == expected_data[case]
            for case in CASES
        ),
        "covariance_at_best_fit": metadata["A_SZ"] == artifacts["A_SZ"],
        "qfrommap_f_sky": metadata["qfrommap_f_sky"] == EXPECTED_F_SKY,
        "exact_covariance_components": all(component_checks.values()),
        "positive_definite_covariances": all(covariance_positive_definite.values()),
        "d3a_gaussian_centers": (
            resolved_params["H0"]["prior"]["loc"] == 68.1
            and resolved_params["n_s"]["prior"]["loc"] == 0.965
            and resolved_params["omega_b"]["prior"]["loc"]
            == 0.022538784599999993
        ),
        "wider_A_SZ_prior": resolved_params["A_SZ"]["prior"]
        == {"dist": "norm", "loc": artifacts["A_SZ"], "scale": 0.03},
        "fixed_B": resolved_params["B"]["value"] == 1.41,
        "gpu_execution": all(entry["passed"] for entry in case_gpu.values()),
        "five_converged_checkpoints": all(
            summary["convergence"]["converged"]
            and summary["convergence"]["Rminus1_last"] <= MAX_RMINUS1
            for summary in case_summaries.values()
        ),
        "getdist_plots": all(Path(path).is_file() for path in plots.values()),
        "constraint_metadata": all(
            set(summary["parameters"]) == set(REQUIRED_PARAMETERS)
            for summary in case_summaries.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "details": {
            "component_identity": component_checks,
            "positive_definite": covariance_positive_definite,
            "case_gpu": case_gpu,
            "best_fit_A_SZ": artifacts["A_SZ"],
            "covariance_A_SZ": metadata["A_SZ"],
            "qfrommap_f_sky": metadata["qfrommap_f_sky"],
            "fixed_point_summary": str(CONVERGED_FILE.resolve()),
            "global_preflight": str(PREFLIGHT_FILE.resolve()),
        },
    }


def main() -> dict:
    from getdist import loadMCSamples

    CHAINS.mkdir(parents=True, exist_ok=True)
    samples_by_case = {}
    case_summaries = {}
    for case in CASES:
        case_dir = CHAINS / case
        convergence = read_convergence(case_dir / "chain.checkpoint")
        samples = loadMCSamples(
            str(case_dir / "chain"),
            settings={"ignore_rows": BURN_IN_FRACTION},
        )
        summary = {
            "case": case,
            "q_cat": CASES[case],
            "burn_in_fraction": BURN_IN_FRACTION,
            "convergence": convergence,
            **summarize_samples(samples),
        }
        (case_dir / "posterior_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        samples_by_case[case] = samples
        case_summaries[case] = summary

    plot_paths = plot_getdist_posteriors(samples_by_case)
    audit = build_completion_audit(case_summaries, plot_paths)
    audit_path = CHAINS / "completion_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    combined = {
        "analysis": "fiducial L1_m9 fullsky and qfrommap q>50,20,10,5",
        "burn_in_fraction": BURN_IN_FRACTION,
        "required_parameters": list(REQUIRED_PARAMETERS),
        "cases": case_summaries,
        "plots": plot_paths,
        "fixed_point_summary": str(CONVERGED_FILE.resolve()),
        "completion_audit": str(audit_path.resolve()),
        "completion_audit_passed": audit["passed"],
    }
    combined_path = CHAINS / "posterior_summary.json"
    combined_path.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    (CHAINS / "posterior_constraints.txt").write_text(
        _constraint_table(case_summaries)
    )
    print(
        json.dumps(
            {
                "posterior_summary": str(combined_path.resolve()),
                "completion_audit": str(audit_path.resolve()),
                "completion_audit_passed": audit["passed"],
                "plots": plot_paths,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not audit["passed"]:
        raise RuntimeError(f"completion audit failed: {audit_path}")
    return combined


if __name__ == "__main__":
    main()
