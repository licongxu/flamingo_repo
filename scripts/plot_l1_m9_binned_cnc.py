"""Plot q-from-map cluster-count marginals for L1_m9 feedback variants."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from paper_results.config import (
    CAT_DIR,
    COLORS,
    FIGURES_FEEDBACK,
    LABELS,
    VARIANTS,
)

Q_COLUMN = "q_from_aperture"
Z_EDGES = np.linspace(0.005, 1.0, 11)
Q_EDGES = np.geomspace(5.0, 40.0, 6)

PAPER_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    # Match tick labels: figure is ~7.1in wide but shown at columnwidth (~3.4in).
    "legend.fontsize": 12,
    "mathtext.fontset": "cm",
    "text.latex.preamble": r"\usepackage{amsmath}",
}


def catalogue_path(variant: str) -> Path:
    """Return the canonical q-from-map catalogue for one feedback variant."""
    return CAT_DIR / (
        f"halo_catalogue_M500c_5e13_zlt3_{variant}_yang26rot_qfrommap.csv"
    )


def bin_cnc(path: Path, chunksize: int = 1_000_000) -> np.ndarray:
    """Return the joint counts in the paper's fixed ``(z, q)`` bins."""
    if not path.is_file():
        raise FileNotFoundError(f"missing catalogue: {path}")

    counts = np.zeros((len(Z_EDGES) - 1, len(Q_EDGES) - 1), dtype=np.int64)
    for chunk in pd.read_csv(
        path,
        comment="#",
        usecols=["z", Q_COLUMN],
        chunksize=chunksize,
    ):
        chunk_counts, _, _ = np.histogram2d(
            chunk["z"].to_numpy(np.float64),
            chunk[Q_COLUMN].to_numpy(np.float64),
            bins=[Z_EDGES, Q_EDGES],
        )
        counts += chunk_counts.astype(np.int64)
    return counts


def load_histograms() -> dict[str, np.ndarray]:
    """Load the joint histogram for every L1_m9 feedback prescription."""
    return {variant: bin_cnc(catalogue_path(variant)) for variant in VARIANTS}


def output_stem(marginal: str) -> Path:
    """Return the explicit q-from-map output stem for one marginal."""
    token = {"q": "Nq", "z": "Nz", "qz": "Nq_Nz"}[marginal]
    return FIGURES_FEEDBACK / f"l1_m9_cnc_binned_{token}_qgt5_feedback_qfrommap"


def legend_label(variant: str, total: int) -> str:
    """Return a TeX-safe feedback label with its total cluster count."""
    label = LABELS[variant].replace("L1_m9", r"L1\_m9")
    formatted_total = f"{total:,d}".replace(",", "{,}")
    return rf"{label} ($N={formatted_total}$)"


def _grouped_bar_geometry(
    edges: np.ndarray,
    series_index: int,
    n_series: int,
    *,
    log_x: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return left edges and widths for one series within every bin."""
    coordinates = np.log(edges) if log_x else edges
    bin_widths = np.diff(coordinates)
    bar_widths = 0.9 * bin_widths / n_series
    left = coordinates[:-1] + 0.05 * bin_widths + series_index * bar_widths
    right = left + bar_widths
    if log_x:
        left = np.exp(left)
        right = np.exp(right)
    return left, right - left


def _draw_marginal(
    ax: plt.Axes,
    histograms: dict[str, np.ndarray],
    marginal: str,
) -> None:
    """Draw one grouped-bar marginal on an existing axis."""
    if marginal == "q":
        edges = Q_EDGES
        sum_axis = 0
        xlabel = r"$q$"
        log_x = True
    elif marginal == "z":
        edges = Z_EDGES
        sum_axis = 1
        xlabel = r"$z$"
        log_x = False
    else:
        raise ValueError(f"unknown marginal: {marginal}")

    for index, variant in enumerate(VARIANTS):
        counts = histograms[variant].sum(axis=sum_axis)
        left, widths = _grouped_bar_geometry(
            edges,
            index,
            len(VARIANTS),
            log_x=log_x,
        )
        ax.bar(
            left,
            counts,
            width=widths,
            align="edge",
            color=COLORS[variant],
            alpha=1.0 if variant == "L1_m9" else 0.68,
            edgecolor="white",
            linewidth=0.35,
            label=legend_label(variant, int(histograms[variant].sum())),
        )

    if log_x:
        ax.set_xscale("log")
        ax.set_xticks([5.0, 10.0, 20.0, 40.0], labels=[r"$5$", r"$10$", r"$20$", r"$40$"])
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlim(4.7, 42.0)
    else:
        ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$N$")
    ax.grid(False)


def build_figure(
    histograms: dict[str, np.ndarray],
    marginal: str,
) -> plt.Figure:
    """Build one standalone marginal-count figure."""
    plt.rcParams.update(PAPER_RC)
    fig, ax = plt.subplots(figsize=(7.1, 4.4), layout="constrained")
    _draw_marginal(ax, histograms, marginal)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        frameon=False,
        ncol=3,
        fontsize=12,
        columnspacing=0.8,
        handlelength=1.15,
        handletextpad=0.35,
    )
    return fig


def build_combined_figure(histograms: dict[str, np.ndarray]) -> plt.Figure:
    """Build the paper-ready ``N(q)`` and ``N(z)`` two-panel figure."""
    plt.rcParams.update(PAPER_RC)
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 6.1), layout="constrained")
    _draw_marginal(axes[0], histograms, "q")
    _draw_marginal(axes[1], histograms, "z")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        frameon=False,
        ncol=3,
        fontsize=12,
        columnspacing=0.8,
        handlelength=1.15,
        handletextpad=0.35,
    )
    return fig


def save_figure(fig: plt.Figure, stem: Path) -> None:
    """Write one figure as PNG and PDF."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        output = stem.with_suffix(f".{suffix}")
        fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"wrote {output}", flush=True)
    plt.close(fig)


def main() -> None:
    histograms = load_histograms()
    for variant, counts in histograms.items():
        print(f"{variant}: N={int(counts.sum()):,d}", flush=True)
    for marginal in ("q", "z"):
        save_figure(build_figure(histograms, marginal), output_stem(marginal))
    save_figure(build_combined_figure(histograms), output_stem("qz"))


if __name__ == "__main__":
    main()
