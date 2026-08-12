"""Plot paper-binned L1_m9 cluster number counts vs q cut and feedback.

Binning matches ``tsz_cnc_paper_plots`` / cosmocnc joint analysis:
10 linear redshift bins over ``0.005 <= z <= 1`` and 5 log-spaced q bins
from ``q_cut`` to 40.

Figures (under ``figures/feedback/``), each with ``N(z)`` and ``N(q)`` panels:

* fiducial binned CNC for several detection cuts;
* binned CNC at ``q > 5`` for fiducial, ``f_gas-4sigma``, Jet, and ``M_*-1sigma``.

``N(z)`` and ``N(q)`` are marginals of the same 10×5 ``(z, q)`` histogram.

Run::

    python scripts/plot_l1_m9_binned_cnc.py
    python scripts/plot_l1_m9_binned_cnc.py --selection qfrommap
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "figures" / "feedback"
DPI = 300

CAT_DIR = Path(
    os.environ.get(
        "L1M9_CAT_DIR",
        "/rds/rds-lxu/flamingo/.hbt_join_fix_staging/20260731/L1_m9/catalogues",
    )
)

# Paper / tsz_cnc_paper_plots CNC binning.
Z_MIN, Z_MAX, N_Z = 0.005, 1.0, 10
Q_MAX_RATIO = 8.0
N_Q = 5
Z_EDGES = np.linspace(Z_MIN, Z_MAX, N_Z + 1)
CHUNK = 1_000_000

FIDUCIAL = "L1_m9"
Q_CUTS = [20.0, 10.0, 5.0]

FEEDBACK_VARIANTS = [
    "fgas+2sigma",
    "fgas-2sigma",
    "fgas-4sigma",
    "fgas-8sigma",
    "Jet",
    "Jet_fgas-4sigma",
    "Mstar-1sigma",
    "Mstar-1sigma_fgas-4sigma",
]

PLOT_FEEDBACK_VARIANTS = [
    FIDUCIAL,
    "fgas-4sigma",
    "Jet",
    "Mstar-1sigma",
]

VARIANT_LABELS = {
    "L1_m9": "fiducial",
    "fgas+2sigma": r"$f_{\rm gas}+2\sigma$",
    "fgas-2sigma": r"$f_{\rm gas}-2\sigma$",
    "fgas-4sigma": r"$f_{\rm gas}-4\sigma$",
    "fgas-8sigma": r"$f_{\rm gas}-8\sigma$",
    "Jet": "Jet",
    "Jet_fgas-4sigma": r"Jet $f_{\rm gas}-4\sigma$",
    "Mstar-1sigma": r"$M_*-1\sigma$",
    "Mstar-1sigma_fgas-4sigma": r"$M_*-1\sigma$ $f_{\rm gas}-4\sigma$",
}

SELECTIONS = {
    "qfrommap": {
        "suffix": "qfrommap",
        "q_column": "q_from_aperture",
        "description": "empirical aperture q",
    },
    "qfrommz_alpha_fixed_1p12": {
        "suffix": "qfrommz_alpha_fixed_1p12",
        "q_column": "q_from_mz",
        "description": r"$\alpha_{\rm SZ}=1.12$",
        "filename_token": "alpha_fixed_1p12",
    },
}


def q_edges(q_cut: float) -> np.ndarray:
    q_min = float(q_cut)
    q_max = q_min * Q_MAX_RATIO
    return np.geomspace(q_min, q_max, N_Q + 1)


def z_centres() -> np.ndarray:
    return 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])


def catalogue_path(variant: str, *, selection_suffix: str) -> Path:
    return CAT_DIR / (
        f"halo_catalogue_M500c_5e13_zlt3_{variant}_yang26rot_{selection_suffix}.csv"
    )


def bin_cnc(
    path: Path,
    *,
    q_column: str,
    q_cut: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return ``(N_zq, N_z, N_q, n_binned)`` in the paper (z, q) bins."""
    if not path.is_file():
        raise FileNotFoundError(f"missing catalogue: {path}")

    qe = q_edges(q_cut)
    z_chunks: list[np.ndarray] = []
    q_chunks: list[np.ndarray] = []
    for chunk in pd.read_csv(path, comment="#", usecols=["z", q_column], chunksize=CHUNK):
        z = chunk["z"].to_numpy(np.float64)
        q = chunk[q_column].to_numpy(np.float64)
        sel = (z >= Z_EDGES[0]) & (z <= Z_EDGES[-1])
        z_chunks.append(z[sel])
        q_chunks.append(q[sel])

    if z_chunks:
        z_all = np.concatenate(z_chunks)
        q_all = np.concatenate(q_chunks)
    else:
        z_all = np.empty(0, dtype=float)
        q_all = np.empty(0, dtype=float)

    n_zq, _, _ = np.histogram2d(z_all, q_all, bins=[Z_EDGES, qe])
    n_binned = int(n_zq.sum())
    return n_zq, n_zq.sum(axis=1), n_zq.sum(axis=0), n_binned


def _curve_colors(n: int) -> np.ndarray:
    return plt.cm.plasma(np.linspace(0.15, 0.85, n))


def _save(fig: plt.Figure, stem: Path) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = Path(stem)
    for suffix in ("png", "pdf"):
        out = stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        try:
            shown = out.resolve().relative_to(REPO.resolve())
        except ValueError:
            shown = out
        print(f"wrote {shown}", flush=True)
    plt.close(fig)
    return stem.with_suffix(".png")


def q_centres(q_cut: float) -> np.ndarray:
    qe = q_edges(q_cut)
    return 0.5 * (qe[:-1] + qe[1:])


def _draw_step_hist(
    ax: plt.Axes,
    edges: np.ndarray,
    counts: np.ndarray,
    *,
    color,
    label: str,
    log_x: bool = False,
) -> None:
    """Draw binned counts as a step histogram (reference plot style)."""
    ax.stairs(
        counts,
        edges,
        color=color,
        label=label,
        linewidth=1.8,
        baseline=0.0,
    )
    if log_x:
        ax.set_xscale("log")


def _style_cnc_panels(
    axes: tuple[plt.Axes, plt.Axes],
    *,
    title: str,
    selection_description: str,
    xlim_z: tuple[float, float],
    legend_kwargs: dict | None = None,
) -> None:
    ax_z, ax_q = axes
    note = (
        rf"10 linear $z$ bins, 5 log $q$ bins ($q_{{\rm cut}}$ to $8 q_{{\rm cut}}$); "
        rf"{selection_description}"
    )
    for ax in axes:
        ax.set_ylim(bottom=0.0)
        ax.grid(False)
    ax_z.set_xlim(*xlim_z)
    ax_z.set_xlabel(r"redshift $z$")
    ax_z.set_ylabel(r"$N$")
    ax_q.set_xlabel(r"detection significance $q$")
    ax_q.set_ylabel(r"$N$")
    fig = ax_z.figure
    fig.suptitle(f"{title}\n({note})", fontsize=10.5, y=1.02)
    if legend_kwargs is None:
        legend_kwargs = {}
    ax_z.legend(loc="upper right", frameon=False, **legend_kwargs)


def plot_cnc_vs_q_cuts(
    *,
    selection_suffix: str,
    q_column: str,
    selection_description: str,
    output_tag: str,
    stem: Path | None = None,
) -> Path:
    """Fiducial ``N(z)`` and ``N(q)`` for several q-cut binnings."""
    path = catalogue_path(FIDUCIAL, selection_suffix=selection_suffix)
    colors = _curve_colors(len(Q_CUTS))

    fig, (ax_z, ax_q) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for q_cut, color in zip(Q_CUTS, colors):
        _, nz, nq, n_tot = bin_cnc(path, q_column=q_column, q_cut=q_cut)
        qe = q_edges(q_cut)
        label = rf"$q>{int(q_cut) if float(q_cut).is_integer() else q_cut:g}$"
        series_label = f"{label} ($N={n_tot:,d}$)"
        _draw_step_hist(ax_z, Z_EDGES, nz, color=color, label=series_label)
        _draw_step_hist(ax_q, qe, nq, color=color, label=series_label, log_x=True)
        ax_q.set_xlim(qe[0], qe[-1])

    _style_cnc_panels(
        (ax_z, ax_q),
        title=r"FLAMINGO L1_m9 binned CNC vs detection cut",
        selection_description=selection_description,
        xlim_z=(Z_EDGES[0], Z_EDGES[-1]),
        legend_kwargs={"fontsize": 8.5, "ncol": 2},
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    if stem is None:
        stem = FIGURES / f"l1_m9_cnc_binned_qcuts_{output_tag}"
    return _save(fig, stem)


def plot_cnc_feedback_qgt5(
    *,
    selection_suffix: str,
    q_column: str,
    selection_description: str,
    output_tag: str,
    stem: Path | None = None,
) -> Path:
    """``N(z)`` and ``N(q)`` at ``q > 5`` for selected feedback variants."""
    qe = q_edges(5.0)
    variants = list(PLOT_FEEDBACK_VARIANTS)
    colors = _curve_colors(len(variants))

    fig, (ax_z, ax_q) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for variant, color in zip(variants, colors):
        path = catalogue_path(variant, selection_suffix=selection_suffix)
        _, nz, nq, n_tot = bin_cnc(path, q_column=q_column, q_cut=5.0)
        label = VARIANT_LABELS.get(variant, variant)
        series_label = f"{label} ($N={n_tot:,d}$)"
        _draw_step_hist(ax_z, Z_EDGES, nz, color=color, label=series_label)
        _draw_step_hist(ax_q, qe, nq, color=color, label=series_label, log_x=True)

    ax_q.set_xlim(qe[0], qe[-1])
    _style_cnc_panels(
        (ax_z, ax_q),
        title=r"FLAMINGO L1_m9 binned CNC at $q>5$ by feedback",
        selection_description=selection_description,
        xlim_z=(Z_EDGES[0], Z_EDGES[-1]),
        legend_kwargs={"fontsize": 8.5, "ncol": 2},
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    if stem is None:
        stem = FIGURES / f"l1_m9_cnc_binned_qgt5_feedback_{output_tag}"
    return _save(fig, stem)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--selection",
        choices=tuple(SELECTIONS),
        default="qfrommap",
    )
    args = parser.parse_args()
    sel = SELECTIONS[args.selection]
    suffix = sel["suffix"]
    q_column = sel["q_column"]
    description = sel["description"]
    output_tag = sel.get("filename_token", suffix)

    plt.rcParams.update(
        {
            "font.size": 10,
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "axes.grid": False,
        }
    )

    print(f"catalogues: {CAT_DIR}", flush=True)
    print(f"selection: {args.selection} ({q_column})", flush=True)
    print(f"z edges: {Z_EDGES[0]:g} … {Z_EDGES[-1]:g} ({N_Z} bins)", flush=True)
    print(f"q edges (q>5): {q_edges(5.0)}", flush=True)

    plot_cnc_vs_q_cuts(
        selection_suffix=suffix,
        q_column=q_column,
        selection_description=description,
        output_tag=output_tag,
    )
    plot_cnc_feedback_qgt5(
        selection_suffix=suffix,
        q_column=q_column,
        selection_description=description,
        output_tag=output_tag,
    )


if __name__ == "__main__":
    main()
