"""Plot L1_m9 feedback spectra at full sky and three qfrommap cuts.

The paper figure is a spectra-only 2x2 grid ordered as full sky, q>20,
q>10, and q>5. Both Planck-style and logarithmic-bin diagnostics are written.

Run::

    python scripts/plot_l1_m9_feedback_bandpowers.py --selection qfrommap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data_paper" / "feedback_bandpower"
MASKED_DATA = REPO / "data_paper" / "binned_bandpowers"
COV = REPO / "data_paper" / "covariance"
FIGURES = REPO / "figures" / "feedback"
TAG = "qfrommz_alpha_fixed_1p12"
CUT_TAG = "qgt5"

PANEL_CUTS = (
    ("full sky", None),
    (r"$q>20$", "qgt20"),
    (r"$q>10$", "qgt10"),
    (r"$q>5$", "qgt5"),
)

VARIANTS = [
    "fiducial",
    "fgas+2sigma",
    "fgas-2sigma",
    "fgas-4sigma",
    "fgas-8sigma",
    "Jet",
    "Jet_fgas-4sigma",
    "Mstar-1sigma",
    "Mstar-1sigma_fgas-4sigma",
]
LABELS = {
    "fiducial": "fiducial",
    "fgas+2sigma": r"$f_{\rm gas}+2\sigma$",
    "fgas-2sigma": r"$f_{\rm gas}-2\sigma$",
    "fgas-4sigma": r"$f_{\rm gas}-4\sigma$",
    "fgas-8sigma": r"$f_{\rm gas}-8\sigma$",
    "Jet": "Jet",
    "Jet_fgas-4sigma": r"Jet $f_{\rm gas}-4\sigma$",
    "Mstar-1sigma": r"$M_*-1\sigma$",
    "Mstar-1sigma_fgas-4sigma": r"$M_*-1\sigma$ $f_{\rm gas}-4\sigma$",
}

PAPER_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "text.latex.preamble": r"\usepackage{amsmath}",
}


def _path(
    variant: str,
    *,
    masked: bool,
    log: bool,
    selection_tag: str | None = None,
    cut_tag: str = CUT_TAG,
) -> Path:
    selection_tag = TAG if selection_tag is None else selection_tag
    suffix = "logbins_dln0p4_lmax10000" if log else "binned_18"
    if not masked:
        return DATA / f"Dl_yy_L1_m9_{variant}_fullsky_{suffix}.txt"
    root = MASKED_DATA if selection_tag == "qfrommap" else DATA
    token = "" if selection_tag == "qfrommap" and variant == "fiducial" else f"_{variant}"
    return root / f"Dl_yy_L1_m9{token}_masked_{cut_tag}_{selection_tag}_{suffix}.txt"


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def output_stem(*, log: bool, selection_tag: str | None = None) -> Path:
    """Return an explicit output stem without overwriting legacy figures."""
    selection_tag = TAG if selection_tag is None else selection_tag
    bin_tag = "logbins" if log else "binned_18"
    suffix = "multiq_qfrommap" if selection_tag == "qfrommap" else "alpha_fixed_1p12"
    return FIGURES / f"l1_m9_feedback_ps_{bin_tag}_{suffix}"


def covariance_note(selection_tag: str) -> str:
    """Compatibility helper describing legacy covariance plots."""
    if selection_tag == "qfrommap":
        return ""
    return (
        "\n"
        r"errors: custom-GNFW $\alpha_{\rm SZ}=1.12$, $B=1.41$, "
        r"best-fit $A_{\rm SZ}$ covariance"
    )


def _load_error_sigmas(*, log: bool) -> dict[bool, tuple[np.ndarray, np.ndarray]]:
    """Compatibility loader for the previous two-panel diagnostic."""
    name = (
        "Dl_yy_customgnfw_bestfit_theory_logbins_dln0p4_lmax10000.txt"
        if log
        else "Dl_yy_customgnfw_bestfit_theory_binned_18.txt"
    )
    data = np.loadtxt(COV / name)
    return {
        False: (data[:, 0], data[:, 7] * 1e12),
        True: (data[:, 0], data[:, 8] * 1e12),
    }


def _save(fig: plt.Figure, stem: Path) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        out = stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out.relative_to(REPO)}", flush=True)
    plt.close(fig)


def _ell_xmax(*, log: bool, selection_tag: str) -> float:
    ell, _ = _load(_path("fiducial", masked=False, log=log, selection_tag=selection_tag))
    return float(ell.max())


def build_figure(
    *,
    log: bool,
    ell_range: tuple[float, float],
    selection_tag: str,
) -> plt.Figure:
    plt.rcParams.update(PAPER_RC)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.0, 7.8),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.10, "wspace": 0.08},
    )
    variant_colors = ["k", *plt.cm.viridis(np.linspace(0.05, 0.9, len(VARIANTS) - 1))]

    for index, (ax, (panel_label, cut_tag)) in enumerate(zip(axes.ravel(), PANEL_CUTS)):
        masked = cut_tag is not None
        for variant, color in zip(VARIANTS, variant_colors):
            ell, dl = _load(
                _path(
                    variant,
                    masked=masked,
                    log=log,
                    selection_tag=selection_tag,
                    cut_tag=cut_tag or CUT_TAG,
                )
            )
            keep = (ell >= ell_range[0]) & (ell <= ell_range[1])
            ax.loglog(
                ell[keep],
                dl[keep],
                color=color,
                lw=2.3 if variant == "fiducial" else 1.5,
                label=LABELS[variant],
                zorder=3 if variant == "fiducial" else 2,
            )
        ax.set_title(panel_label)
        ax.set_xlim(*ell_range)
        ax.grid(False)
        if index // 2 == 1:
            ax.set_xlabel(r"$\ell$")
        if index % 2 == 0:
            ax.set_ylabel(r"$10^{12}D_\ell^{yy}$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(
        handles,
        labels,
        loc="lower right",
        frameon=False,
        handlelength=2.0,
    )
    return fig


def main() -> None:
    global TAG
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", choices=(TAG, "qfrommap"), default=TAG)
    args = parser.parse_args()
    TAG = args.selection

    for log in (False, True):
        if log:
            ell_range = (100.0, _ell_xmax(log=True, selection_tag=TAG))
        else:
            ell_range = (10.0, 959.5)
        fig = build_figure(log=log, ell_range=ell_range, selection_tag=TAG)
        _save(fig, output_stem(log=log, selection_tag=TAG))


if __name__ == "__main__":
    main()
