#!/usr/bin/env python3
"""Summarize and plot the L2p8_m9 A_SZ chain with alpha_SZ fixed."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from getdist import loadMCSamples, plots


CHAIN_DIR = Path(
    "chains/L2p8_m9_customgnfw_asz_alpha_fixed_1p12_B1p41"
)
CHAIN_ROOT = CHAIN_DIR / "chain"


def main() -> None:
    samples = loadMCSamples(str(CHAIN_ROOT), settings={"ignore_rows": 0.3})
    names = samples.getParamNames().list()
    asz_index = names.index("A_SZ")
    chi2_index = names.index(next(name for name in names if name.startswith("chi2__")))
    best_index = int(np.argmin(samples.samples[:, chi2_index]))

    asz = samples.samples[:, asz_index]
    weights = samples.weights
    best_asz = float(asz[best_index])
    best_chi2 = float(samples.samples[best_index, chi2_index])
    mean = float(np.average(asz, weights=weights))
    variance = float(np.average((asz - mean) ** 2, weights=weights))
    limit = samples.getMargeStats().parWithName("A_SZ").limits[0]

    summary = {
        "simulation": "L2p8_m9",
        "fixed": {"alpha_SZ": 1.12, "B": 1.41, "cosmology": "D3A"},
        "burn_in_fraction": 0.3,
        "best_fit": {"A_SZ": best_asz, "chi2": best_chi2},
        "posterior_mean": {"A_SZ": mean},
        "posterior_variance": {"A_SZ": variance},
        "credible_interval_68": {
            "A_SZ": [float(limit.lower), float(limit.upper)]
        },
        "retained_rows": int(len(samples.samples)),
        "retained_weight": float(np.sum(weights)),
        "effective_samples": float(samples.getEffectiveSamples()),
    }
    (CHAIN_DIR / "best_fit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "simulation = L2p8_m9",
        "fixed cosmology = D3A",
        "fixed alpha_SZ = 1.12",
        "fixed B = 1.41",
        f"best_fit A_SZ = {best_asz:.9g}",
        f"best_fit chi2 = {best_chi2:.9g}",
        f"posterior_mean A_SZ = {mean:.9g}",
        f"posterior_variance A_SZ = {variance:.9g}",
        f"68% interval A_SZ = [{limit.lower:.9g}, {limit.upper:.9g}]",
    ]
    (CHAIN_DIR / "posterior_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    plotter = plots.get_single_plotter(width_inch=6)
    plotter.plot_1d(
        samples,
        "A_SZ",
        colors=["#2c7fb8"],
        line_args=[{"label": "L2p8_m9"}],
    )
    plotter.add_x_marker(best_asz, color="#d7301f", lw=1.4)
    for suffix in ("png", "pdf"):
        plotter.export(str(CHAIN_DIR / f"posterior_L2p8_m9_A_SZ.{suffix}"))
    plt.close("all")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
