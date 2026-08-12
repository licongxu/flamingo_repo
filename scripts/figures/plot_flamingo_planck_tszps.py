"""Paper figure: FLAMINGO progressive masking and Planck comparison.

The left panel shows the fiducial L1_m9 spectrum for four empirical
``qfrommap`` masking thresholds. The right panel compares the full-sky and
``q > 6`` L1_m9 spectra with the Planck measurements of Bolliet et al. (2018)
and Rotti et al. (2021), respectively.

Run::

    python scripts/figures/plot_flamingo_planck_tszps.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data_paper" / "binned_bandpowers"
TOPOSZ = REPO / "ref_package" / "TopoSZ" / "fig12 data"
FIGURES = REPO / "figures" / "planck_comparison"
TAG = "qfrommap"
META = DATA / "L1_m9_feedback_multi_q_bandpowers_qfrommap_metadata.json"

LEFT_CUTS = (
    (50.0, "qgt50"),
    (20.0, "qgt20"),
    (10.0, "qgt10"),
    (5.0, "qgt5"),
)

PLANCK_B18 = TOPOSZ / "planck_sz_1712_00788v1.txt"
PLANCK_MASKED = TOPOSZ / (
    "szpowespectrum_measurement_urc_snr6_p18cmb_bf_fg_from_TSZ+P18_"
    "l_clyy_sigclyy_cib_ir_rs_cn.txt"
)

N_PLANCK_BINS = 18
ELL_PLANCK_MAX = 959.5
ELL_XMIN = 10.0
ELL_XMAX = 1.0e4

PAPER_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "text.latex.preamble": r"\usepackage{amsmath}",
}


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    return arr[:, 0], arr[:, 1]


def _bin18_path(
    variant: str,
    *,
    masked: bool,
    selection_tag: str = TAG,
) -> Path:
    if masked:
        return DATA / f"Dl_yy_{variant}_masked_qgt6_{selection_tag}_binned_18.txt"
    return DATA / f"Dl_yy_{variant}_fullsky_binned_18.txt"


def _logbin_path(
    variant: str,
    *,
    masked: bool,
    selection_tag: str = TAG,
) -> Path:
    if masked:
        return DATA / (
            f"Dl_yy_{variant}_masked_qgt6_{selection_tag}_"
            "logbins_dln0p4_lmax10000.txt"
        )
    return DATA / (
        f"Dl_yy_{variant}_fiducial_fullsky_"
        "logbins_dln0p4_lmax10000_pixwin_deconvolved.txt"
    )


def _masked_logbin_path(cut_tag: str, selection_tag: str = TAG) -> Path:
    return DATA / (
        f"Dl_yy_L1_m9_masked_{cut_tag}_{selection_tag}_"
        "logbins_dln0p4_lmax10000.txt"
    )


def _hybrid_curve(variant: str, *, masked: bool) -> tuple[np.ndarray, np.ndarray]:
    """Use 18 Planck bins through ell=959.5 and logarithmic bins above it."""
    ell18, dl18 = _load(_bin18_path(variant, masked=masked))
    elllog, dllog = _load(_logbin_path(variant, masked=masked))
    high_ell = elllog > ELL_PLANCK_MAX
    return np.concatenate([ell18, elllog[high_ell]]), np.concatenate([dl18, dllog[high_ell]])


def _planck_bolliet() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(PLANCK_B18)[:N_PLANCK_BINS]
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _planck_masked_marginalised() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(PLANCK_MASKED)[:N_PLANCK_BINS]
    return arr[:, 0], arr[:, 1], arr[:, 2]


def output_stem(selection_tag: str = TAG) -> Path:
    return FIGURES / f"l1_m9_fiducial_masking_planck_{selection_tag}"


def _save(fig: plt.Figure, stem: Path) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        out = stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out.relative_to(REPO)}", flush=True)
    plt.close(fig)


def build_figure() -> plt.Figure:
    plt.rcParams.update(PAPER_RC)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.8, 4.8))

    ell_full_log, dl_full_log = _load(_logbin_path("L1_m9", masked=False))
    keep_full_log = (ell_full_log >= 100.0) & (ell_full_log <= ELL_XMAX)
    ax_left.loglog(
        ell_full_log[keep_full_log],
        dl_full_log[keep_full_log],
        color="k",
        lw=2.3,
        label="full sky",
        zorder=4,
    )
    cut_colors = ("#CC79A7", "#56B4E9", "#009E73", "#E69F00")
    metadata = json.loads(META.read_text())["variants"]["fiducial"]["cuts"]
    for (cut, cut_tag), color in zip(LEFT_CUTS, cut_colors):
        ell, dl = _load(_masked_logbin_path(cut_tag))
        keep = (ell >= 100.0) & (ell <= ELL_XMAX)
        label = rf"$q>{cut:g}$"
        entry = metadata.get(cut_tag)
        if entry:
            label += rf" ($N={entry['n_masked']:,}$, $f_{{\rm sky}}={entry['f_sky_eff']:.3f}$)"
        ax_left.loglog(ell[keep], dl[keep], color=color, lw=1.9, label=label)

    ell_b, dl_b, sig_b = _planck_bolliet()
    ell_r, dl_r, sig_r = _planck_masked_marginalised()
    full_color = "#0072B2"
    masked_color = "#D55E00"

    ell_full, dl_full = _hybrid_curve("L1_m9", masked=False)
    keep_full = (ell_full >= ELL_XMIN) & (ell_full <= ELL_XMAX)
    (full_line,) = ax_right.loglog(
        ell_full[keep_full],
        dl_full[keep_full],
        color=full_color,
        lw=2.3,
        label="FLAMINGO full sky",
        zorder=3,
    )
    bolliet_points = ax_right.errorbar(
        ell_b,
        dl_b,
        yerr=sig_b,
        fmt="o",
        mfc="white",
        mec=full_color,
        ecolor=full_color,
        capsize=2.5,
        elinewidth=1.1,
        markersize=4.5,
        label="Bolliet et al. (2018)",
        zorder=5,
    )

    ell_masked, dl_masked = _hybrid_curve("L1_m9", masked=True)
    keep_masked = (ell_masked >= ELL_XMIN) & (ell_masked <= ELL_XMAX)
    (masked_line,) = ax_right.loglog(
        ell_masked[keep_masked],
        dl_masked[keep_masked],
        color=masked_color,
        ls="--",
        lw=2.3,
        label=r"$q>6$ FLAMINGO",
        zorder=3,
    )
    rotti_points = ax_right.errorbar(
        ell_r,
        dl_r,
        yerr=sig_r,
        fmt="s",
        mfc="white",
        mec=masked_color,
        ecolor=masked_color,
        capsize=2.5,
        elinewidth=1.1,
        markersize=4.5,
        label="Rotti et al. (2021)",
        zorder=5,
    )

    for panel, xmin in ((ax_left, 100.0), (ax_right, ELL_XMIN)):
        panel.set_xscale("log")
        panel.set_yscale("log")
        panel.set_xlim(xmin, ELL_XMAX)
        panel.set_ylim(5.0e-3, 3.0)
        panel.set_xlabel(r"$\ell$")
        panel.grid(False)
    ax_left.legend(frameon=False, loc="lower right", handlelength=2.1)
    ax_right.legend(
        handles=(full_line, bolliet_points, masked_line, rotti_points),
        frameon=False,
        loc="lower right",
        handlelength=2.1,
    )
    ax_left.set_ylabel(r"$10^{12}D_\ell^{yy}$")
    ax_right.tick_params(labelleft=False)
    ax_left.text(0.03, 0.95, r"\textbf{(a)}", transform=ax_left.transAxes, va="top")
    ax_right.text(0.03, 0.95, r"\textbf{(b)}", transform=ax_right.transAxes, va="top")
    fig.subplots_adjust(wspace=0.08)
    return fig


def main() -> None:
    fig = build_figure()
    _save(fig, output_stem())


if __name__ == "__main__":
    main()
