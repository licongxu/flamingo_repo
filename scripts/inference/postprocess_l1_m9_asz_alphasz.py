#!/usr/bin/env python3
"""Summarize and plot the converged L1_m9 A_SZ-alpha_SZ Cobaya chain."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from getdist import loadMCSamples, plots


CHAIN_DIR = Path("chains/l1_m9_customgnfw_asz_alphasz")
CHAIN_ROOT = CHAIN_DIR / "chain"
PARAMS = ("A_SZ", "alpha_SZ")


def main() -> None:
    samples = loadMCSamples(str(CHAIN_ROOT), settings={"ignore_rows": 0.3})
    names = samples.getParamNames().list()
    param_indices = [names.index(name) for name in PARAMS]
    chi2_name = next(name for name in names if name.startswith("chi2__"))
    chi2_index = names.index(chi2_name)

    best_index = int(np.argmin(samples.samples[:, chi2_index]))
    best = {
        name: float(samples.samples[best_index, index])
        for name, index in zip(PARAMS, param_indices, strict=True)
    }
    best["chi2"] = float(samples.samples[best_index, chi2_index])

    values = samples.samples[:, param_indices]
    weights = samples.weights
    mean = np.average(values, axis=0, weights=weights)
    centered = values - mean
    covariance = np.einsum(
        "n,ni,nj->ij", weights, centered, centered
    ) / np.sum(weights)
    variance = np.diag(covariance)

    marge = samples.getMargeStats()
    intervals_68 = {
        name: [
            float(marge.parWithName(name).limits[0].lower),
            float(marge.parWithName(name).limits[0].upper),
        ]
        for name in PARAMS
    }

    summary = {
        "burn_in_fraction": 0.3,
        "best_fit": best,
        "posterior_mean": dict(zip(PARAMS, map(float, mean), strict=True)),
        "posterior_variance": dict(zip(PARAMS, map(float, variance), strict=True)),
        "posterior_covariance": covariance.tolist(),
        "credible_interval_68": intervals_68,
        "retained_rows": int(len(samples.samples)),
        "retained_weight": float(np.sum(weights)),
        "effective_samples": float(samples.getEffectiveSamples()),
    }
    (CHAIN_DIR / "best_fit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        f"best_fit A_SZ = {best['A_SZ']:.9g}",
        f"best_fit alpha_SZ = {best['alpha_SZ']:.9g}",
        f"best_fit chi2 = {best['chi2']:.9g}",
        f"posterior_mean A_SZ = {mean[0]:.9g}",
        f"posterior_mean alpha_SZ = {mean[1]:.9g}",
        f"posterior_variance A_SZ = {variance[0]:.9g}",
        f"posterior_variance alpha_SZ = {variance[1]:.9g}",
        "posterior_covariance =",
        np.array2string(covariance, precision=9),
    ]
    (CHAIN_DIR / "posterior_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    plotter = plots.get_subplot_plotter(width_inch=6)
    plotter.settings.num_plot_contours = 2
    plotter.settings.axes_fontsize = 11
    plotter.settings.lab_fontsize = 13
    plotter.triangle_plot(
        [samples],
        list(PARAMS),
        filled=True,
        contour_colors=["#2c7fb8"],
        legend_labels=["L1_m9"],
    )
    plotter.subplots[0, 0].axvline(best["A_SZ"], color="#d7301f", lw=1.2)
    plotter.subplots[1, 1].axvline(best["alpha_SZ"], color="#d7301f", lw=1.2)
    plotter.subplots[1, 0].plot(
        best["A_SZ"],
        best["alpha_SZ"],
        marker="*",
        color="#d7301f",
        ms=10,
        label="best fit",
        zorder=10,
    )
    plotter.subplots[1, 0].legend(frameon=False, fontsize=9, loc="best")
    for suffix in ("png", "pdf"):
        plotter.export(str(CHAIN_DIR / f"posterior_A_SZ_alpha_SZ.{suffix}"))
    plt.close("all")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
