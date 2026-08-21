"""Triangle plot for the converged L1_m9 qfrommap CNC q>5 chain."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from getdist import loadMCSamples, plots

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_l1_m9_cnc_qgt5_chain import CHAINS  # noqa: E402
from scripts.run_l1_m9_ps_qgt5_scalrel_chain import (  # noqa: E402
    CHAINS as PS_CHAINS,
    CHAINS_ROOT as PS_CHAINS_ROOT,
    chain_dir as ps_chain_dir,
)
from scripts.summarize_masked_ps_chains import read_convergence  # noqa: E402

CHAIN_ROOT = CHAINS / "chain"
PS_CHAIN_ROOT = PS_CHAINS / "chain"
PARAMS = ("A_SZ", "alpha_SZ", "sigma_lnY")
STEM = CHAINS / "triangle_A_SZ_alpha_SZ_sigma_lnY"
SUBMONO_CHAINS = REPO / "chains" / "l1_m9_qfrommap_submono_cnc_qgt5"
SUBMONO_CHAIN_ROOT = SUBMONO_CHAINS / "chain"
SUBMONO_STEM = (
    REPO / "figures" / "masked_ps" / "l1_m9_qgt5_cnc_submono_triangle"
)
RAW_VS_SUBMONO_STEM = (
    REPO / "figures" / "masked_ps" / "l1_m9_qgt5_cnc_raw_vs_submono_triangle"
)
ADDMONO_CHAINS = REPO / "chains" / "l1_m9_qfrommap_cnc_qgt5_addmono"
ADDMONO_CHAIN_ROOT = ADDMONO_CHAINS / "chain"
ADDMONO_STEM = (
    REPO / "figures" / "masked_ps" / "l1_m9_qgt5_cnc_addmono_triangle"
)
RAW_VS_ADDMONO_STEM = (
    REPO / "figures" / "masked_ps" / "l1_m9_qgt5_cnc_raw_vs_addmono_triangle"
)
COMPARE_STEM = (
    REPO / "figures" / "masked_ps" / "l1_m9_qgt5_cnc_vs_ps_triangle"
)
PS_CASES = ("fullsky", "qgt50", "qgt20", "qgt10", "qgt5")
PS_LABELS = {
    "fullsky": "full sky",
    "qgt50": r"$q>50$",
    "qgt20": r"$q>20$",
    "qgt10": r"$q>10$",
    "qgt5": r"$q>5$",
}
PS_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
PS_STEM = REPO / "figures" / "masked_ps" / "l1_m9_ps_scalrel_triangle"
BURN_IN_FRACTION = 0.3


def summarize(samples) -> dict:
    """Return posterior mean/std and the maximum-likelihood sample."""
    names = samples.getParamNames().list()
    chi2_name = next(name for name in names if name.startswith("chi2__"))
    best_index = int(samples.samples[:, names.index(chi2_name)].argmin())
    stats = samples.getMargeStats()
    constraints = {}
    for name in PARAMS:
        par = stats.parWithName(name)
        constraints[name] = {
            "mean": float(par.mean),
            "standard_deviation": float(par.err),
            "interval_68": [float(par.limits[0].lower), float(par.limits[0].upper)],
            "best_fit": float(samples.samples[best_index, names.index(name)]),
        }
    return {
        "burn_in_fraction": BURN_IN_FRACTION,
        "constraints": constraints,
        "best_fit_chi2": float(samples.samples[best_index, names.index(chi2_name)]),
        "retained_rows": int(len(samples.samples)),
        "effective_samples": float(samples.getEffectiveSamples()),
    }


def plot_triangle(
    samples,
    summary: dict,
    *,
    stem: Path = STEM,
    label: str = r"L1\_m9 CNC $q>5$",
) -> None:
    """Write the filled GetDist triangle for the three scaling parameters."""
    plotter = plots.get_subplot_plotter(width_inch=7)
    plotter.settings.num_plot_contours = 2
    plotter.settings.axes_fontsize = 11
    plotter.settings.lab_fontsize = 13
    plotter.triangle_plot(
        [samples],
        list(PARAMS),
        filled=True,
        contour_colors=["#2c7fb8"],
        legend_labels=[label],
    )
    means = [summary["constraints"][name]["mean"] for name in PARAMS]
    for i, value in enumerate(means):
        plotter.subplots[i, i].axvline(value, color="#d7301f", lw=1.2)
    plotter.subplots[1, 0].plot(means[0], means[1], marker="*", color="#d7301f", ms=10)
    plotter.subplots[2, 0].plot(means[0], means[2], marker="*", color="#d7301f", ms=10)
    plotter.subplots[2, 1].plot(means[1], means[2], marker="*", color="#d7301f", ms=10)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        plotter.export(str(stem.with_suffix(f".{suffix}")))
    plt.close("all")


def plot_raw_vs_submono(raw, submono) -> None:
    """Overlay raw-aperture and monopole-subtracted CNC posteriors."""
    plotter = plots.get_subplot_plotter(width_inch=7)
    plotter.settings.num_plot_contours = 2
    plotter.settings.axes_fontsize = 11
    plotter.settings.lab_fontsize = 13
    plotter.settings.legend_fontsize = 11
    plotter.triangle_plot(
        [raw, submono],
        list(PARAMS),
        filled=True,
        contour_colors=["#7570b3", "#2c7fb8"],
        legend_labels=[r"raw aperture $q$", r"monopole subtracted"],
    )
    RAW_VS_SUBMONO_STEM.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        plotter.export(str(RAW_VS_SUBMONO_STEM.with_suffix(f".{suffix}")))
        plotter.export(str(SUBMONO_CHAINS / f"triangle_raw_vs_submono.{suffix}"))
    plt.close("all")


def plot_raw_vs_addmono(raw, addmono) -> None:
    """Overlay no-monopole theory vs theory-with-monopole, both on raw q."""
    plotter = plots.get_subplot_plotter(width_inch=7)
    plotter.settings.num_plot_contours = 2
    plotter.settings.axes_fontsize = 11
    plotter.settings.lab_fontsize = 13
    plotter.settings.legend_fontsize = 11
    plotter.triangle_plot(
        [raw, addmono],
        list(PARAMS),
        filled=True,
        contour_colors=["#7570b3", "#d95f02"],
        legend_labels=[r"theory without $\langle y\rangle$", r"theory add_monopole"],
    )
    RAW_VS_ADDMONO_STEM.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        plotter.export(str(RAW_VS_ADDMONO_STEM.with_suffix(f".{suffix}")))
        plotter.export(str(ADDMONO_CHAINS / f"triangle_raw_vs_addmono.{suffix}"))
    plt.close("all")


def plot_cnc_vs_ps(cnc, ps) -> None:
    """Overlay CNC and q>5 tSZ PS posteriors on one triangle."""
    plotter = plots.get_subplot_plotter(width_inch=7)
    plotter.settings.num_plot_contours = 2
    plotter.settings.axes_fontsize = 11
    plotter.settings.lab_fontsize = 13
    plotter.settings.legend_fontsize = 11
    plotter.triangle_plot(
        [cnc, ps],
        list(PARAMS),
        filled=True,
        contour_colors=["#2c7fb8", "#d95f02"],
        legend_labels=[r"CNC $q>5$", r"tSZ PS $q>5$"],
    )
    COMPARE_STEM.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        plotter.export(str(COMPARE_STEM.with_suffix(f".{suffix}")))
        plotter.export(str(PS_CHAINS / f"triangle_cnc_vs_ps.{suffix}"))
    plt.close("all")


def plot_ps_cases(samples_by_case: dict) -> None:
    """Overlay tSZ PS scalrel posteriors for every masking threshold."""
    plotter = plots.get_subplot_plotter(width_inch=8)
    plotter.settings.num_plot_contours = 2
    plotter.settings.axes_fontsize = 11
    plotter.settings.lab_fontsize = 13
    plotter.settings.legend_fontsize = 10
    plotter.triangle_plot(
        [samples_by_case[case] for case in PS_CASES],
        list(PARAMS),
        filled=True,
        contour_colors=PS_COLORS,
        legend_labels=[PS_LABELS[case] for case in PS_CASES],
    )
    PS_STEM.parent.mkdir(parents=True, exist_ok=True)
    PS_CHAINS_ROOT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        plotter.export(str(PS_STEM.with_suffix(f".{suffix}")))
        plotter.export(str(PS_CHAINS_ROOT / f"triangle_scalrel.{suffix}"))
    plt.close("all")


def load_converged(root: Path, checkpoint: Path):
    read_convergence(checkpoint)
    return loadMCSamples(str(root), settings={"ignore_rows": BURN_IN_FRACTION})


def plot_submono() -> dict:
    """Triangle for the monopole-subtracted chain, plus overlay on raw q."""
    submono = load_converged(
        SUBMONO_CHAIN_ROOT, SUBMONO_CHAINS / "chain.checkpoint"
    )
    summary = {
        "convergence": read_convergence(SUBMONO_CHAINS / "chain.checkpoint"),
        **summarize(submono),
    }
    plot_triangle(
        submono,
        summary,
        stem=SUBMONO_STEM,
        label=r"CNC $q>5$, monopole subtracted",
    )
    (SUBMONO_CHAINS / "posterior_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        f"{name} = {row['mean']:.5f} ± {row['standard_deviation']:.5f}  "
        f"(68% [{row['interval_68'][0]:.5f}, {row['interval_68'][1]:.5f}]; "
        f"best {row['best_fit']:.5f})"
        for name, row in summary["constraints"].items()
    ]
    (SUBMONO_CHAINS / "posterior_constraints.txt").write_text("\n".join(lines) + "\n")
    print("CNC submono:", *lines, sep="\n  ")
    if (CHAINS / "chain.checkpoint").is_file():
        raw = load_converged(CHAIN_ROOT, CHAINS / "chain.checkpoint")
        plot_raw_vs_submono(raw, submono)
    return summary


def plot_addmono() -> dict:
    """Triangle for the theory-monopole chain on raw aperture q."""
    addmono = load_converged(
        ADDMONO_CHAIN_ROOT, ADDMONO_CHAINS / "chain.checkpoint"
    )
    summary = {
        "convergence": read_convergence(ADDMONO_CHAINS / "chain.checkpoint"),
        **summarize(addmono),
    }
    plot_triangle(
        addmono,
        summary,
        stem=ADDMONO_STEM,
        label=r"CNC $q>5$, theory $\langle y\rangle$",
    )
    (ADDMONO_CHAINS / "posterior_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        f"{name} = {row['mean']:.5f} ± {row['standard_deviation']:.5f}  "
        f"(68% [{row['interval_68'][0]:.5f}, {row['interval_68'][1]:.5f}]; "
        f"best {row['best_fit']:.5f})"
        for name, row in summary["constraints"].items()
    ]
    (ADDMONO_CHAINS / "posterior_constraints.txt").write_text("\n".join(lines) + "\n")
    print("CNC add_monopole:", *lines, sep="\n  ")
    if (CHAINS / "chain.checkpoint").is_file():
        raw = load_converged(CHAIN_ROOT, CHAINS / "chain.checkpoint")
        plot_raw_vs_addmono(raw, addmono)
    return summary


def main() -> dict:
    if (ADDMONO_CHAINS / "chain.checkpoint").is_file():
        plot_addmono()
    if (SUBMONO_CHAINS / "chain.checkpoint").is_file():
        plot_submono()
    cnc = load_converged(CHAIN_ROOT, CHAINS / "chain.checkpoint")
    summary = {
        "convergence": read_convergence(CHAINS / "chain.checkpoint"),
        **summarize(cnc),
    }
    plot_triangle(cnc, summary)
    (CHAINS / "posterior_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        f"{name} = {row['mean']:.5f} ± {row['standard_deviation']:.5f}  "
        f"(68% [{row['interval_68'][0]:.5f}, {row['interval_68'][1]:.5f}]; "
        f"best {row['best_fit']:.5f})"
        for name, row in summary["constraints"].items()
    ]
    (CHAINS / "posterior_constraints.txt").write_text("\n".join(lines) + "\n")
    print("CNC:", *lines, sep="\n  ")
    if (PS_CHAINS / "chain.checkpoint").is_file():
        ps = load_converged(PS_CHAIN_ROOT, PS_CHAINS / "chain.checkpoint")
        ps_summary = {
            "convergence": read_convergence(PS_CHAINS / "chain.checkpoint"),
            **summarize(ps),
        }
        plot_cnc_vs_ps(cnc, ps)
        (PS_CHAINS / "posterior_summary.json").write_text(
            json.dumps(ps_summary, indent=2, sort_keys=True) + "\n"
        )
        ps_lines = [
            f"{name} = {row['mean']:.5f} ± {row['standard_deviation']:.5f}  "
            f"(68% [{row['interval_68'][0]:.5f}, {row['interval_68'][1]:.5f}]; "
            f"best {row['best_fit']:.5f})"
            for name, row in ps_summary["constraints"].items()
        ]
        (PS_CHAINS / "posterior_constraints.txt").write_text("\n".join(ps_lines) + "\n")
        print("tSZ PS:", *ps_lines, sep="\n  ")
        summary["ps"] = ps_summary
    if all((ps_chain_dir(case) / "chain.checkpoint").is_file() for case in PS_CASES):
        samples_by_case = {
            case: load_converged(
                ps_chain_dir(case) / "chain",
                ps_chain_dir(case) / "chain.checkpoint",
            )
            for case in PS_CASES
        }
        plot_ps_cases(samples_by_case)
        lines = []
        for case, samples in samples_by_case.items():
            row = summarize(samples)
            print(f"tSZ PS {case}:", flush=True)
            for name, constraint in row["constraints"].items():
                line = (
                    f"{name} = {constraint['mean']:.5f} ± "
                    f"{constraint['standard_deviation']:.5f}"
                )
                print("  ", line, flush=True)
                lines.append(f"{case} {line}")
        (PS_CHAINS_ROOT / "posterior_constraints.txt").write_text(
            "\n".join(lines) + "\n"
        )
    return summary


if __name__ == "__main__":
    main()
