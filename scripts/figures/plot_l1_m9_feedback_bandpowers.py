"""Plot L1_m9 feedback spectra at full sky and three qfrommap cuts.

The paper figure is a spectra-only 2x2 grid ordered as full sky, q>20,
q>10, and q>5. Both Planck-style and logarithmic-bin diagnostics are written.

Run::

    python scripts/figures/plot_l1_m9_feedback_bandpowers.py --selection qfrommap
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


def _load_error_sigmas(
    *, log: bool, masked_column: int = 8
) -> dict[bool, tuple[np.ndarray, np.ndarray]]:
    """Compatibility loader for the previous two-panel diagnostic."""
    name = (
        "Dl_yy_customgnfw_bestfit_theory_logbins_dln0p4_lmax10000.txt"
        if log
        else "Dl_yy_customgnfw_bestfit_theory_binned_18.txt"
    )
    data = np.loadtxt(COV / name)
    return {
        False: (data[:, 0], data[:, 7] * 1e12),
        True: (data[:, 0], data[:, masked_column] * 1e12),
    }


def _save(fig: plt.Figure, stem: Path, *, dpi: int = 300) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        out = stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"wrote {out.relative_to(REPO)}", flush=True)
    plt.close(fig)


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
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=2.0,
    )
    fig.subplots_adjust(bottom=0.15)
    return fig


def single_cut_output_stem(*, log: bool, selection_tag: str, cut_tag: str) -> Path:
    bin_tag = "logbins" if log else "binned_18"
    output_tag = "qfrommap" if selection_tag == "qfrommap" else "alpha_fixed_1p12"
    return FIGURES / f"l1_m9_feedback_ps_{bin_tag}_{output_tag}_{cut_tag}"


def _q_cut(cut_tag: str) -> float:
    return float(cut_tag.removeprefix("qgt").replace("p", "."))


def build_single_cut_figure(
    *,
    log: bool,
    ell_range: tuple[float, float],
    selection_tag: str,
    cut_tag: str,
    title: str,
    error_sigmas: dict[bool, tuple[np.ndarray, np.ndarray]] | None = None,
) -> plt.Figure:
    """Build the legacy full-sky/masked comparison for one q threshold."""
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(VARIANTS) - 1))
    q_cut = _q_cut(cut_tag)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.0, 6.6),
        sharex=True,
        height_ratios=[2.2, 1.0],
        gridspec_kw={"hspace": 0.06, "wspace": 0.22},
    )
    for col, masked in enumerate((False, True)):
        ax, axr = axes[0, col], axes[1, col]
        ell_ref, dl_ref = _load(
            _path(
                "fiducial",
                masked=masked,
                log=log,
                selection_tag=selection_tag,
                cut_tag=cut_tag,
            )
        )
        inside = (ell_ref >= ell_range[0]) & (ell_ref <= ell_range[1])
        sigma = None
        if error_sigmas is not None:
            ell_err, sigma = error_sigmas[masked]
            np.testing.assert_allclose(ell_ref, ell_err, rtol=0.0, atol=5e-4)
        if sigma is None:
            ax.loglog(
                ell_ref[inside],
                dl_ref[inside],
                "k-",
                lw=2.2,
                marker="o",
                markersize=4.0,
                label=LABELS["fiducial"],
                zorder=3,
            )
        else:
            ax.errorbar(
                ell_ref[inside],
                dl_ref[inside],
                yerr=sigma[inside],
                color="k",
                lw=2.2,
                marker="o",
                markersize=4.0,
                elinewidth=1.1,
                capsize=2.4,
                capthick=1.1,
                label=LABELS["fiducial"],
                zorder=3,
            )
        axr.axhline(1.0, color="k", lw=1.6, zorder=3)
        if sigma is not None:
            axr.errorbar(
                ell_ref[inside],
                np.ones_like(ell_ref[inside]),
                yerr=(sigma / dl_ref)[inside],
                fmt="none",
                ecolor="k",
                elinewidth=1.1,
                capsize=2.4,
                capthick=1.1,
                zorder=4,
            )
            axr.set_ylim(0.75, 1.35)
        for variant, color in zip(VARIANTS[1:], colors):
            ell, dl = _load(
                _path(
                    variant,
                    masked=masked,
                    log=log,
                    selection_tag=selection_tag,
                    cut_tag=cut_tag,
                )
            )
            ax.loglog(
                ell[inside],
                dl[inside],
                lw=1.4,
                color=color,
                marker="o",
                markersize=3.0,
                label=LABELS[variant],
            )
            axr.semilogx(ell[inside], dl[inside] / dl_ref[inside], lw=1.4, color=color)

        ax.set_title("full sky" if not masked else rf"masked, $q>{q_cut:g}$", fontsize=10)
        ax.set_xscale("log")
        axr.set_xscale("log")
        ax.set_xlim(*ell_range)
        axr.set_xlim(*ell_range)
        ax.tick_params(labelbottom=False)
        axr.set_xlabel(r"multipole $\ell$")
        if col == 0:
            ax.set_ylabel(r"$10^{12}\,\ell(\ell+1)C_\ell^{yy}/(2\pi)$")
            axr.set_ylabel("variant / fiducial")
            ax.legend(fontsize=7.5, loc="upper left", frameon=False, ncol=2)

    fig.suptitle(title, fontsize=11, y=0.96)
    return fig


def _print_single_cut_metadata(selection_tag: str, cut_tag: str) -> None:
    if selection_tag == "qfrommap":
        path = MASKED_DATA / "L1_m9_feedback_multi_q_bandpowers_qfrommap_metadata.json"
    else:
        path = DATA / f"L1_m9_feedback_bandpowers_{cut_tag}_metadata.json"
    if not path.exists():
        return
    meta = json.loads(path.read_text())
    print("f_sky_eff per variant:")
    for variant in VARIANTS:
        entry = meta["variants"][variant]
        if selection_tag == "qfrommap":
            entry = entry["cuts"][cut_tag]
        print(
            f"  {variant:26s} N_masked={entry['n_masked']:5d}  "
            f"f_sky={entry['f_sky_eff']:.4f}"
        )


def main() -> None:
    global TAG
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", choices=(TAG, "qfrommap"), default=TAG)
    parser.add_argument(
        "--single-cut",
        metavar="QTAG",
        help="write the legacy full-sky/masked diagnostic for one cut (for example qgt1)",
    )
    args = parser.parse_args()
    TAG = args.selection

    if args.single_cut:
        plt.rcParams.update({"font.size": 10, "text.usetex": False, "mathtext.fontset": "cm"})
        cut_tag = args.single_cut
        q_cut = _q_cut(cut_tag)
        _print_single_cut_metadata(TAG, cut_tag)
        q_description = "empirical aperture q" if TAG == "qfrommap" else r"$q$ from $\alpha_{\rm SZ}=1.12$ best fit"
        for log, ell_range in ((False, (10.0, 959.5)), (True, (100.0, 10000.0))):
            bin_description = (
                r"$\Delta\ln\ell=0.4$ log bins" if log else "18 Planck bins"
            )
            title = (
                "FLAMINGO L1_m9 feedback variants: tSZ power spectrum "
                rf"({bin_description}; masked $q>{q_cut:g}$; {q_description})"
                + covariance_note(TAG)
            )
            errors = None if TAG == "qfrommap" else _load_error_sigmas(log=log, masked_column=12)
            fig = build_single_cut_figure(
                log=log,
                ell_range=ell_range,
                selection_tag=TAG,
                cut_tag=cut_tag,
                title=title,
                error_sigmas=errors,
            )
            _save(
                fig,
                single_cut_output_stem(log=log, selection_tag=TAG, cut_tag=cut_tag),
                dpi=180,
            )
        return

    for log, ell_range in ((False, (10.0, 959.5)), (True, (100.0, 10000.0))):
        fig = build_figure(log=log, ell_range=ell_range, selection_tag=TAG)
        _save(fig, output_stem(log=log, selection_tag=TAG))


if __name__ == "__main__":
    main()
