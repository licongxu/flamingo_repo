"""Compare L1_m9 and L2p8_m9 masked tSZ bandpowers on one figure.

L1_m9 uses solid lines; L2p8_m9 uses dashed lines. One figure per binning
convention (18 Planck bins and 12 log bins).

Run::

    python scripts/plot_l1_l2p8_m9_masked_ps_comparison.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data_paper" / "binned_bandpowers"
FIGURES = REPO / "figures" / "masked_ps"
TAG = "qfrommz_alpha_fixed_1p12"

RESOLUTIONS = (
    ("L1_m9", "L1_m9", "-"),
    ("L2p8_m9", "L2p8_m9", "--"),
)

Q_CUTS = [50.0, 20.0, 10.0, 5.0, 1.0]
CUT_TAGS = ["qgt50", "qgt20", "qgt10", "qgt5", "qgt1"]


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def _paths(variant: str, tag: str, *, log: bool) -> Path:
    suffix = "logbins_dln0p4_lmax10000" if log else "binned_18"
    return DATA / f"Dl_yy_{variant}_masked_{tag}_{TAG}_{suffix}.txt"


def _fullsky_paths(variant: str, *, log: bool) -> Path:
    if log:
        return DATA / (
            f"Dl_yy_{variant}_fiducial_fullsky_logbins_dln0p4_lmax10000_pixwin_deconvolved.txt"
        )
    return DATA / f"Dl_yy_{variant}_fullsky_binned_18.txt"


def _save(fig: plt.Figure, stem: Path) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=180, bbox_inches="tight")
        print(f"wrote {stem.with_suffix('.' + suffix).relative_to(REPO)}", flush=True)
    plt.close(fig)


def _plot(*, log: bool, ell_range: tuple[float, float], title: str, stem: Path) -> None:
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(CUT_TAGS)))
    fig, (ax, axr) = plt.subplots(
        2,
        1,
        figsize=(6.8, 6.8),
        sharex=True,
        height_ratios=[2.4, 1.0],
        gridspec_kw={"hspace": 0.06},
    )

    for variant, _, ls in RESOLUTIONS:
        meta = json.loads((DATA / f"{variant}_masked_{TAG}_metadata.json").read_text())["cuts"]
        ell_fs, dl_fs = _load(_fullsky_paths(variant, log=log))
        inside_fs = (ell_fs >= ell_range[0]) & (ell_fs <= ell_range[1])
        ax.loglog(
            ell_fs[inside_fs],
            dl_fs[inside_fs],
            color="k",
            ls=ls,
            lw=2.0,
            marker="o",
            markersize=3.8,
            zorder=3,
        )
        axr.semilogx(
            ell_fs[inside_fs],
            np.ones(inside_fs.sum()),
            color="k",
            ls=ls,
            lw=1.0,
        )

        for cut, tag, color in zip(Q_CUTS, CUT_TAGS, colors):
            ell, dl = _load(_paths(variant, tag, log=log))
            keep = (ell >= ell_range[0]) & (ell <= ell_range[1])
            ax.loglog(
                ell[keep],
                dl[keep],
                color=color,
                ls=ls,
                lw=1.5,
                marker="o",
                markersize=3.2,
            )
            ratio_ell = ell_fs[inside_fs]
            ratio = np.interp(ratio_ell, ell[keep], dl[keep]) / np.interp(
                ratio_ell, ell_fs[inside_fs], dl_fs[inside_fs]
            )
            axr.semilogx(ratio_ell, ratio, color=color, ls=ls, lw=1.5)

    style_handles = [
        Line2D([0], [0], color="k", ls="-", lw=2.0, marker="o", markersize=4, label="L1_m9"),
        Line2D([0], [0], color="k", ls="--", lw=2.0, marker="o", markersize=4, label="L2p8_m9"),
    ]
    cut_handles = [
        Line2D([0], [0], color=color, ls="-", lw=1.5, marker="o", markersize=3, label=rf"$q>{cut:g}$")
        for cut, color in zip(Q_CUTS, colors)
    ]
    ax.legend(
        handles=style_handles + cut_handles,
        fontsize=8,
        loc="lower right",
        frameon=False,
        ncol=2,
        columnspacing=1.0,
    )

    ax.set_ylabel(r"$10^{12}\,\ell(\ell+1)C_\ell^{yy}/(2\pi)$")
    axr.set_ylabel("masked / full sky")
    axr.set_ylim(0.0, 1.05)
    ax.set_title(title, fontsize=11)
    for panel in (ax, axr):
        panel.set_xscale("log")
        panel.set_xlim(*ell_range)
    ax.tick_params(labelbottom=False)
    axr.set_xlabel(r"multipole $\ell$")
    _save(fig, stem)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 10, "text.usetex": False, "mathtext.fontset": "cm"})

    _plot(
        log=False,
        ell_range=(10.0, 959.5),
        title=(
            "FLAMINGO masked tSZ: L1_m9 (solid) vs L2p8_m9 (dashed), "
            r"18 Planck bins, $\alpha_{\rm SZ}=1.12$ best-fit $q$"
        ),
        stem=FIGURES / "l1_l2p8_m9_masked_ps_binned_18_alpha_fixed_1p12",
    )
    _plot(
        log=True,
        ell_range=(100.0, 10000.0),
        title=(
            "FLAMINGO masked tSZ: L1_m9 (solid) vs L2p8_m9 (dashed), "
            r"$\Delta\ln\ell=0.4$ log bins, $\alpha_{\rm SZ}=1.12$ best-fit $q$"
        ),
        stem=FIGURES / "l1_l2p8_m9_masked_ps_logbins_alpha_fixed_1p12",
    )
