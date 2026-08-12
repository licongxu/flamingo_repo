#!/usr/bin/env python3
"""Summarize and plot an A_SZ chain with alpha_SZ fixed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from getdist import loadMCSamples, plots


CASES = {
    "L1_m9": {
        "chain_dir": Path("chains/l1_m9_customgnfw_asz_alpha_fixed_1p12"),
        "fixed": {"alpha_SZ": 1.12, "B": 1.41},
        "summary_extra": {},
        "summary_prefix": [],
        "figure_stem": "posterior_A_SZ",
        "plot_label": None,
    },
    "L2p8_m9": {
        "chain_dir": Path("chains/L2p8_m9_customgnfw_asz_alpha_fixed_1p12_B1p41"),
        "fixed": {"alpha_SZ": 1.12, "B": 1.41, "cosmology": "D3A"},
        "summary_extra": {"simulation": "L2p8_m9"},
        "summary_prefix": ["simulation = L2p8_m9", "fixed cosmology = D3A"],
        "figure_stem": "posterior_L2p8_m9_A_SZ",
        "plot_label": "L2p8_m9",
    },
}


def case_config(case: str) -> dict:
    return CASES[case]


def main(case: str) -> None:
    config = case_config(case)
    chain_dir = config["chain_dir"]
    samples = loadMCSamples(str(chain_dir / "chain"), settings={"ignore_rows": 0.3})
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
        **config["summary_extra"],
        "fixed": config["fixed"],
        "burn_in_fraction": 0.3,
        "best_fit": {"A_SZ": best_asz, "chi2": best_chi2},
        "posterior_mean": {"A_SZ": mean},
        "posterior_variance": {"A_SZ": variance},
        "credible_interval_68": {"A_SZ": [float(limit.lower), float(limit.upper)]},
        "retained_rows": int(len(samples.samples)),
        "retained_weight": float(np.sum(weights)),
        "effective_samples": float(samples.getEffectiveSamples()),
    }
    (chain_dir / "best_fit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        *config["summary_prefix"],
        "fixed alpha_SZ = 1.12",
        "fixed B = 1.41",
        f"best_fit A_SZ = {best_asz:.9g}",
        f"best_fit chi2 = {best_chi2:.9g}",
        f"posterior_mean A_SZ = {mean:.9g}",
        f"posterior_variance A_SZ = {variance:.9g}",
        f"68% interval A_SZ = [{limit.lower:.9g}, {limit.upper:.9g}]",
    ]
    (chain_dir / "posterior_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    plotter = plots.get_single_plotter(width_inch=6)
    plot_args = {}
    if config["plot_label"] is not None:
        plot_args["line_args"] = [{"label": config["plot_label"]}]
    plotter.plot_1d(samples, "A_SZ", colors=["#2c7fb8"], **plot_args)
    plotter.add_x_marker(best_asz, color="#d7301f", lw=1.4)
    for suffix in ("png", "pdf"):
        plotter.export(str(chain_dir / f"{config['figure_stem']}.{suffix}"))
    plt.close("all")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    main(parser.parse_args().case)
