"""Plot fiducial L1_m9 masked tSZ bandpowers (alpha-fixed q catalogue).

Paper-style figure uses the 12 logarithmic bins
(``Delta ln ell = 0.4``, ``ell_max = 10000``) with ``text.usetex=True``.
An optional 18-bin version is still written for diagnostics.

Run::

    python scripts/figures/plot_l1_m9_masked_ps_alpha_fixed_1p12.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data_paper" / "binned_bandpowers"
FIGURES = REPO / "figures" / "masked_ps"
TAG = "qfrommz_alpha_fixed_1p12"

Q_CUTS = [50.0, 20.0, 10.0, 5.0]
CUT_TAGS = ["qgt50", "qgt20", "qgt10", "qgt5"]

FULLSKY_18 = DATA / "Dl_yy_L1_m9_fullsky_binned_18.txt"
FULLSKY_LOG = DATA / "Dl_yy_L1_m9_fiducial_fullsky_logbins_dln0p4_lmax10000_pixwin_deconvolved.txt"
META = DATA / f"L1_m9_masked_{TAG}_metadata.json"

OUT_18 = FIGURES / "l1_m9_masked_ps_binned_18_alpha_fixed_1p12"
OUT_LOG = FIGURES / "l1_m9_masked_ps_logbins_alpha_fixed_1p12"

# Paper-style fonts (aligned with ~/scratch/tsz_project/tsz_ps_plot).
PAPER_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "text.latex.preamble": r"\usepackage{amsmath}",
}


def selection_cuts(selection_tag: str) -> tuple[list[float], list[str]]:
    if selection_tag == "qfrommap":
        return (
            [50.0, 20.0, 10.0, 6.0, 5.0, 3.0, 1.0],
            ["qgt50", "qgt20", "qgt10", "qgt6", "qgt5", "qgt3", "qgt1"],
        )
    return Q_CUTS, CUT_TAGS


def _load_two_column(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def _masked_paths(tag: str, *, log: bool, selection_tag: str | None = None) -> Path:
    selection_tag = TAG if selection_tag is None else selection_tag
    suffix = "logbins_dln0p4_lmax10000" if log else "binned_18"
    return DATA / f"Dl_yy_L1_m9_masked_{tag}_{selection_tag}_{suffix}.txt"


def _save(fig: plt.Figure, stem: Path, *, dpi: int = 300) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        out = stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"wrote {out.relative_to(REPO)}", flush=True)
    plt.close(fig)


def _plot(
    ell_full: np.ndarray,
    dl_full: np.ndarray,
    masked: list[tuple[float, str, np.ndarray, np.ndarray]],
    masked_meta: dict,
    *,
    ell_range: tuple[float, float],
    stem: Path,
    paper: bool = True,
) -> None:
    inside = (ell_full >= ell_range[0]) & (ell_full <= ell_range[1])
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(masked)))

    fig, (ax, axr) = plt.subplots(
        2,
        1,
        figsize=(6.8, 6.6) if paper else (6.4, 6.4),
        sharex=True,
        height_ratios=[2.4, 1.0],
        gridspec_kw={"hspace": 0.06},
    )
    ax.loglog(
        ell_full[inside],
        dl_full[inside],
        "k-",
        lw=2.2,
        marker="o",
        markersize=4.5,
        label=r"$D_\ell^{\mathrm{full}}$",
        zorder=3,
    )
    for (cut, tag, ell, dl), color in zip(masked, colors):
        keep = (ell >= ell_range[0]) & (ell <= ell_range[1])
        meta = masked_meta.get(tag)
        label = rf"$q>{cut:g}$"
        if meta is not None:
            label += (
                rf"~($N={meta['n_masked']}$, " rf"$f_{{\mathrm{{sky}}}}={meta['f_sky_eff']:.3f}$)"
            )
        ax.loglog(
            ell[keep],
            dl[keep],
            lw=1.8,
            color=color,
            marker="o",
            markersize=4.0,
            label=label,
        )
        ratio_ell = ell_full[inside]
        ratio = np.interp(ratio_ell, ell[keep], dl[keep]) / dl_full[inside]
        axr.semilogx(ratio_ell, ratio, lw=1.8, color=color)

    axr.axhline(1.0, color="k", lw=1.2)
    ax.set_ylabel(r"$10^{12}\,D_\ell^{yy}$")
    axr.set_ylabel(r"$D_\ell^{\mathrm{masked}}/D_\ell^{\mathrm{full}}$")
    axr.set_ylim(0.0, 1.05)
    # No title (caption carries the description).
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=14 if paper else 8,
        handlelength=1.6,
        borderaxespad=0.4,
    )
    for panel in (ax, axr):
        panel.set_xscale("log")
        panel.set_xlim(*ell_range)
        panel.grid(False)
    ax.tick_params(labelbottom=False)
    axr.set_xlabel(r"$\ell$")
    _save(fig, stem, dpi=300 if paper else 180)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", choices=(TAG, "qfrommap"), default=TAG)
    args = parser.parse_args()
    TAG = args.selection
    if TAG == "qfrommap":
        Q_CUTS, CUT_TAGS = selection_cuts(TAG)
        meta_path = DATA / "L1_m9_feedback_multi_q_bandpowers_qfrommap_metadata.json"
        with meta_path.open() as handle:
            masked_meta = json.load(handle)["variants"]["fiducial"]["cuts"]
        OUT_18 = FIGURES / "l1_m9_masked_ps_binned_18_qfrommap"
        OUT_LOG = FIGURES / "l1_m9_masked_ps_logbins_qfrommap"
    else:
        with META.open() as handle:
            masked_meta = json.load(handle)["cuts"]

    # Paper figure: 12 log bins, LaTeX, no title.
    plt.rcParams.update(PAPER_RC)
    ell_log, dl_log_full = _load_two_column(FULLSKY_LOG)
    masked_log = []
    for cut, tag in zip(Q_CUTS, CUT_TAGS):
        ell, dl = _load_two_column(_masked_paths(tag, log=True))
        masked_log.append((cut, tag, ell, dl))
    _plot(
        ell_log,
        dl_log_full,
        masked_log,
        masked_meta,
        ell_range=(100.0, 10000.0),
        stem=OUT_LOG,
        paper=True,
    )

    # Diagnostic 18-bin version (same style, different multipole range).
    ell18, dl18_full = _load_two_column(FULLSKY_18)
    masked18 = []
    for cut, tag in zip(Q_CUTS, CUT_TAGS):
        ell, dl = _load_two_column(_masked_paths(tag, log=False))
        masked18.append((cut, tag, ell, dl))
    _plot(
        ell18,
        dl18_full,
        masked18,
        masked_meta,
        ell_range=(10.0, 959.5),
        stem=OUT_18,
        paper=True,
    )
