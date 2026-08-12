"""Post-process DMB tSZ PS chains: posterior summary, P(k) suppression, figures."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from flamingo.inference.dmb_ps import (  # noqa: E402
    ALL_FREE_PARAMS,
    D3A_VALUES,
    DMB_TABLE1_PRIORS,
    PRIMARY_DMB_DEFAULTS,
    PRIMARY_DMB_PARAMS,
    SAMPLED_DMB_PARAMS,
    evaluate_dmb_bandpowers,
    evaluate_pk_suppression,
)
from flamingo.inference.l1_m9 import load_bandpower_likelihood  # noqa: E402
from scripts.run_dmb_masked_ps_chains import CHAINS  # noqa: E402
from scripts.run_masked_ps_chains import CASES, data_files, load_converged_artifacts  # noqa: E402
from scripts.summarize_masked_ps_chains import _weighted_quantile  # noqa: E402

FIGURES = REPO / "figures" / "dmb"
BURN_IN_FRACTION = 0.3
N_SUPPRESSION_SAMPLES = 1000  # fast after JIT cache; use many draws for smooth bands

PAPER_RC = {
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "axes.linewidth": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
}

# Triangle: ACT Table-1 DMB knobs (10 params, Dalal et al. 2026)
TRIANGLE_PARAMS = list(ALL_FREE_PARAMS)
# P(k) uses profile params only (not sigma_lnY)
PK_DRAW_PARAMS = [p for p in SAMPLED_DMB_PARAMS if p != "sigma_lnY"]


def _load_samples(case: str, *, burn_in_fraction: float = BURN_IN_FRACTION):
    from getdist import loadMCSamples

    return loadMCSamples(
        str(CHAINS / case / "chain"), settings={"ignore_rows": burn_in_fraction}
    )


def summarize_params(samples) -> dict:
    names = samples.getParamNames().list()
    weights = np.asarray(samples.weights, dtype=float)
    out: dict = {"parameters": {}, "retained_rows": int(samples.samples.shape[0])}
    for name in ALL_FREE_PARAMS:
        if name not in names:
            continue
        values = np.asarray(samples.samples[:, names.index(name)], dtype=float)
        mean = float(np.average(values, weights=weights))
        std = float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))
        out["parameters"][name] = {
            "mean": mean,
            "standard_deviation": std,
            "median": _weighted_quantile(values, weights, 0.50),
            "interval_68": [
                _weighted_quantile(values, weights, 0.16),
                _weighted_quantile(values, weights, 0.84),
            ],
            "interval_95": [
                _weighted_quantile(values, weights, 0.025),
                _weighted_quantile(values, weights, 0.975),
            ],
        }
    for name in SAMPLED_DMB_PARAMS:
        if name not in out["parameters"]:
            raise ValueError(f"chain missing {name}; has {names}")
    chi2_cols = [n for n in names if n.startswith("chi2__")]
    if chi2_cols:
        chi2 = np.asarray(samples.samples[:, names.index(chi2_cols[0])], dtype=float)
        best = int(np.argmin(chi2))
        out["best_likelihood_sample"] = {
            name: float(samples.samples[best, names.index(name)])
            for name in ALL_FREE_PARAMS
            if name in names
        }
        out["best_likelihood_sample"]["chi2"] = float(chi2[best])
    else:
        out["best_likelihood_sample"] = {
            name: out["parameters"][name]["median"] for name in out["parameters"]
        }
    return out


def prior_predictive_draws(n: int = N_SUPPRESSION_SAMPLES) -> list[dict[str, float]]:
    """Uniform samples from ACT Table 1 priors (Dalal et al. Fig. 6 convention)."""
    rng = np.random.default_rng(20260811)
    draws = []
    for _ in range(n):
        draws.append(
            {
                name: float(rng.uniform(DMB_TABLE1_PRIORS[name]["min"], DMB_TABLE1_PRIORS[name]["max"]))
                for name in PK_DRAW_PARAMS
            }
        )
    return draws


def thin_param_draws(samples, n: int = N_SUPPRESSION_SAMPLES) -> list[dict[str, float]]:
    names = samples.getParamNames().list()
    weights = np.asarray(samples.weights, dtype=float)
    weights = weights / weights.sum()
    rng = np.random.default_rng(20260811)
    n_draw = min(n, samples.samples.shape[0])
    idx = rng.choice(samples.samples.shape[0], size=n_draw, replace=False, p=weights)
    keys = [p for p in PK_DRAW_PARAMS if p in names]
    return [
        {name: float(samples.samples[i, names.index(name)]) for name in keys}
        for i in idx
    ]


def compute_pk_suppression_band(draws: list[dict[str, float]]) -> dict[str, np.ndarray]:
    """Dalal et al. Eq. (5.4) S(k)=P_DMB/P_NFW; 68% and 95% bands."""
    ratios = []
    k = None
    for i, params in enumerate(draws):
        result = evaluate_pk_suppression(**params)
        if k is None:
            k = result["k_h"]
        ratios.append(result["ratio"])
        print(f"  pk draw {i + 1}/{len(draws)}", flush=True)
    stack = np.vstack(ratios)
    return {
        "k_h": k,
        "k": k,  # h/Mpc
        "ratio_median": np.nanmedian(stack, axis=0),
        "ratio_p16": np.nanpercentile(stack, 16, axis=0),
        "ratio_p84": np.nanpercentile(stack, 84, axis=0),
        "ratio_p025": np.nanpercentile(stack, 2.5, axis=0),
        "ratio_p975": np.nanpercentile(stack, 97.5, axis=0),
        "ratio_mean": np.nanmean(stack, axis=0),
        "n_draws": np.asarray([len(draws)]),
    }


def plot_triangle(samples, case: str, out_dir: Path) -> dict[str, Path]:
    from getdist import plots

    with plt.rc_context(PAPER_RC):
        g = plots.get_subplot_plotter(width_inch=7)
        g.settings.axes_labelsize = 14
        plot_params = [p for p in TRIANGLE_PARAMS if p in samples.getParamNames().list()]
        g.triangle_plot([samples], plot_params, filled=True, contour_colors=["#1b9e77"])
        products = {}
        for suffix in ("png", "pdf"):
            path = out_dir / f"dmb_triangle_{case}.{suffix}"
            g.export(str(path))
            products[suffix] = path
        plt.close("all")
    return products


def plot_bandpower_fit(case: str, best_params: dict[str, float], out_dir: Path) -> dict[str, Path]:
    artifacts = load_converged_artifacts()
    data_path, cov_path = data_files(case, artifacts["covariance_paths"])
    ell, observed, cov, _ = load_bandpower_likelihood(data_path, cov_path)
    err = np.sqrt(np.diag(cov))
    q_cat = CASES[case]
    bp = evaluate_dmb_bandpowers(
        **{**PRIMARY_DMB_DEFAULTS, **best_params}, q_cat=q_cat
    )
    theory = bp["1h"] + bp["2h"]
    defaults = evaluate_dmb_bandpowers(**PRIMARY_DMB_DEFAULTS, q_cat=q_cat)
    theory_def = defaults["1h"] + defaults["2h"]

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(
            2, 1, figsize=(6.5, 6.2), sharex=True,
            gridspec_kw={"height_ratios": [2.4, 1.0], "hspace": 0.05},
        )
        ax, axr = axes
        scale = 1e12
        ax.errorbar(
            ell, observed * scale, yerr=err * scale,
            fmt="o", color="k", ms=4, capsize=2, label="L1_m9 data", zorder=3,
        )
        ax.plot(ell, theory * scale, color="#d95f02", lw=2.0, label="DMB best fit")
        ax.plot(
            ell, theory_def * scale, color="#7570b3", lw=1.5, ls="--", label="DMB defaults"
        )
        ax.plot(ell, bp["1h"] * scale, color="#1b9e77", lw=1.0, ls=":", label="1h")
        ax.plot(ell, bp["2h"] * scale, color="#e7298a", lw=1.0, ls=":", label="2h")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(r"$10^{12}\,D_\ell^{yy}$")
        title = "full sky" if case == "fullsky" else case.replace("qgt", r"$q>$")
        ax.set_title(f"L1_m9 masked tSZ PS — DMB fit ({title})")
        ax.legend(frameon=False, loc="upper left")
        resid = (observed - theory) / err
        axr.axhline(0.0, color="0.5", lw=0.8)
        axr.errorbar(ell, resid, yerr=np.ones_like(resid), fmt="o", color="k", ms=4, capsize=2)
        axr.set_xscale("log")
        axr.set_xlabel(r"$\ell$")
        axr.set_ylabel(r"$(d-t)/\sigma$")
        axr.set_ylim(-4, 4)
        products = {}
        for suffix in ("png", "pdf"):
            path = out_dir / f"dmb_bandpower_fit_{case}.{suffix}"
            fig.savefig(path, bbox_inches="tight", dpi=200)
            products[suffix] = path
        plt.close(fig)
    return products


def plot_pk_suppression(
    band: dict[str, np.ndarray],
    case: str,
    out_dir: Path,
    *,
    kind: str = "posterior",
) -> dict[str, Path]:
    """S(k)=P_DMB/P_NFW with 68% + 95% bands, k in h/Mpc."""
    if kind == "posterior":
        med_label = "Posterior (median)"
        title = "full sky" if case == "fullsky" else case.replace("qgt", r"$q>$")
        fig_stem = f"dmb_pk_suppression_{case}"
    else:
        med_label = "Prior predictive (median)"
        fig_stem = f"dmb_pk_suppression_prior_{case}"
        title = None

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        k = np.asarray(band.get("k_h", band["k"]), dtype=float)
        med = np.asarray(band["ratio_median"], dtype=float)
        lo68 = np.asarray(band["ratio_p16"], dtype=float)
        hi68 = np.asarray(band["ratio_p84"], dtype=float)
        lo95 = np.asarray(band.get("ratio_p025", lo68), dtype=float)
        hi95 = np.asarray(band.get("ratio_p975", hi68), dtype=float)

        ax.fill_between(
            k, lo95, hi95, color="#4e79a7", alpha=0.25, label="95% CL", zorder=1
        )
        ax.fill_between(
            k, lo68, hi68, color="#4e79a7", alpha=0.45, label="68% CL", zorder=2
        )
        ax.plot(k, med, color="#1f4e79", lw=2.0, label=med_label, zorder=3)
        ax.axhline(1.0, color="0.35", ls="--", lw=0.9, zorder=0)

        ax.set_xscale("log")
        ax.set_xlabel(r"$k\ (h\,{\rm Mpc}^{-1})$")
        ax.set_ylabel(r"$P_{\rm DMB}(k)\,/\,P_{\rm NFW}(k)$")
        if kind == "posterior":
            ax.set_title(rf"Matter power suppression at $z=0$ (L1\_m9 {title} tSZ PS)")
        else:
            ax.set_title(r"Matter power suppression at $z=0$ (ACT Table 1 prior predictive)")
        ax.set_xlim(float(k.min()), float(k.max()))
        y_lo = min(0.70, 0.95 * float(np.nanmin(lo95)))
        ax.set_ylim(y_lo, 1.10)
        ax.legend(frameon=False, loc="lower left", fontsize=10)

        products = {}
        for suffix in ("png", "pdf"):
            path = out_dir / f"{fig_stem}.{suffix}"
            fig.savefig(path, bbox_inches="tight", dpi=200)
            products[suffix] = path
        plt.close(fig)
    return products


def write_summary_text(summary: dict, path: Path) -> None:
    lines = [
        "DMB baryonic feedback from L1_m9 large-scale masked tSZ PS",
        f"case: {summary.get('case')}",
        "cosmology: D3A fixed (papers fix Planck for tSZ DMB)",
        f"d3a: {json.dumps(D3A_VALUES)}",
        "DMB priors: To et al. (2024) / Dalal et al. (2026) Table 1",
        f"retained rows: {summary.get('retained_rows')}",
        "",
        "parameter  median  16%  84%  mean  std",
    ]
    for name, stats in summary["parameters"].items():
        lo, hi = stats["interval_68"]
        lines.append(
            f"{name:12s}  {stats['median']:.4f}  {lo:.4f}  {hi:.4f}  "
            f"{stats['mean']:.4f}  {stats['standard_deviation']:.4f}"
        )
    lines.append("")
    lines.append(f"best_likelihood_sample: {json.dumps(summary.get('best_likelihood_sample', {}))}")
    path.write_text("\n".join(lines) + "\n")


def process_case(case: str) -> dict:
    FIGURES.mkdir(parents=True, exist_ok=True)
    case_dir = CHAINS / case
    case_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading samples for {case} ...", flush=True)
    samples = _load_samples(case)
    summary = summarize_params(samples)
    summary["case"] = case
    summary["cosmology"] = "D3A_fixed"
    summary["d3a_fixed"] = dict(D3A_VALUES)
    summary["prior_source"] = "To et al. 2024 / Dalal et al. 2026 Table 1; cosmo fixed D3A"

    summary_json = case_dir / "posterior_summary.json"
    summary_txt = case_dir / "posterior_summary.txt"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_summary_text(summary, summary_txt)
    (CHAINS / "dmb_posterior_summary.txt").write_text(summary_txt.read_text())

    print("Triangle plot ...", flush=True)
    tri = plot_triangle(samples, case, FIGURES)

    best = summary["best_likelihood_sample"]
    best_params = {
        name: float(best[name]) for name in ALL_FREE_PARAMS if name in best
    }
    print("Bandpower fit figure ...", flush=True)
    fit = plot_bandpower_fit(case, best_params, FIGURES)

    print("P(k) suppression — posterior (constraining) ...", flush=True)
    post_draws = thin_param_draws(samples, n=N_SUPPRESSION_SAMPLES)
    post_band = compute_pk_suppression_band(post_draws)
    np.savez(case_dir / "pk_suppression_posterior.npz", **post_band)
    pk_fig = plot_pk_suppression(post_band, case, FIGURES, kind="posterior")

    print("P(k) suppression — prior predictive (reference) ...", flush=True)
    prior_draws = prior_predictive_draws(n=N_SUPPRESSION_SAMPLES)
    prior_band = compute_pk_suppression_band(prior_draws)
    np.savez(case_dir / "pk_suppression_prior.npz", **prior_band)
    pk_prior_fig = plot_pk_suppression(prior_band, case, FIGURES, kind="prior")

    products = {
        "summary_json": str(summary_json),
        "summary_txt": str(summary_txt),
        "triangle": {k: str(v) for k, v in tri.items()},
        "bandpower_fit": {k: str(v) for k, v in fit.items()},
        "pk_suppression": {k: str(v) for k, v in pk_fig.items()},
        "pk_suppression_posterior_npz": str(case_dir / "pk_suppression_posterior.npz"),
        "pk_suppression_prior": {k: str(v) for k, v in pk_prior_fig.items()},
        "pk_suppression_prior_npz": str(case_dir / "pk_suppression_prior.npz"),
    }
    (case_dir / "postprocess_products.json").write_text(
        json.dumps(products, indent=2, sort_keys=True) + "\n"
    )
    return products


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="qgt5", choices=tuple(CASES))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(json.dumps(process_case(args.case), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
