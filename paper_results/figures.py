"""Step 3: the paper figures, drawn from the bandpowers written by :mod:`~paper_results.compute_ps`.

``fiducial``
    The fiducial L1_m9 tSZ power spectrum, unmasked and after masking clusters
    above each detection threshold, with the fraction of power removed.

``feedback``
    The unmasked and the ``q > 5``-masked tSZ power spectrum for every L1_m9
    feedback variant, each with its ratio to the fiducial run. Masking the
    detected clusters removes the well-measured massive halos and leaves the
    diffuse, feedback-sensitive component, so the spread between variants is
    the quantity of interest.

``theory``
    The total (unmasked) fiducial tSZ power spectrum against the ``hmfast``
    halo-model prediction, split into its 1-halo and 2-halo terms. Both use the
    same FLAMINGO D3A cosmology and the Arnaud A10 pressure profile at
    ``B = 1``; nothing is fitted.

Run::

    python -m paper_results.figures            # all figures
    python -m paper_results.figures --figure fiducial
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config

#: Multipole range shown; below ~100 the lightcone box scale bites.
ELL_RANGE = (100.0, 6000.0)

#: Multipole range of the halo-model comparison. The measurement stops at
#: ``config.LMAX``; the prediction is drawn out to ``10^4``.
THEORY_ELL_RANGE = (100.0, 1.0e4)

#: Reference cut for the feedback comparison.
FEEDBACK_CUT = 5.0


def load(variant: str) -> dict:
    """Load the bandpower file of one variant.

    Parameters
    ----------
    variant : str
        Feedback-variant name.

    Returns
    -------
    dict
        Contents of ``results/bandpowers/<variant>.npz``.
    """
    path = config.BANDPOWERS / f"{variant}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `python -m paper_results.compute_ps` first")
    return dict(np.load(path, allow_pickle=True))


def _finish(ax, *, xlabel: bool) -> None:
    """Common axis styling for a bandpower panel."""
    ax.set_xscale("log")
    ax.set_xlim(*ELL_RANGE)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    if xlabel:
        ax.set_xlabel(r"multipole $\ell$")
    else:
        ax.tick_params(labelbottom=False)


def figure_fiducial() -> None:
    """Draw the fiducial unmasked and ``q``-masked tSZ power spectra."""
    data = load(config.FIDUCIAL)
    ell = data["ell"]
    inside = (ell >= ELL_RANGE[0]) & (ell <= ELL_RANGE[1])
    full = data["dl_fullsky"]

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(6.4, 6.4), sharex=True, height_ratios=[2.4, 1],
        gridspec_kw=dict(hspace=0.06),
    )

    ax.loglog(ell[inside], full[inside], "k-", lw=2.0, label="unmasked")
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(config.Q_CUTS)))
    for cut, dl, n, f, color in zip(
        data["q_cuts"], data["dl_masked"], data["n_masked"], data["fsky"], colors
    ):
        label = rf"$q>{cut:g}$  ($N={int(n):,}$, $f_{{\rm sky}}={f:.3f}$)"
        ax.loglog(ell[inside], dl[inside], lw=1.5, color=color, label=label)
        axr.semilogx(ell[inside], (dl / full)[inside], lw=1.5, color=color)

    axr.axhline(1.0, color="k", lw=1.0)
    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{yy}/2\pi$")
    axr.set_ylabel("masked / unmasked")
    axr.set_ylim(0, 1.05)
    ax.set_title(
        "FLAMINGO L1_m9 fiducial: tSZ power spectrum masked at "
        rf"${config.R_MASK:g}\,\theta_{{500}}$",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    _finish(ax, xlabel=False)
    _finish(axr, xlabel=True)

    _save(fig, "fiducial_masked_tsz_ps")


def figure_feedback() -> None:
    """Draw the unmasked and ``q>5``-masked tSZ power spectra of all feedback variants."""
    data = {v: load(v) for v in config.VARIANTS}
    ell = data[config.FIDUCIAL]["ell"]
    inside = (ell >= ELL_RANGE[0]) & (ell <= ELL_RANGE[1])
    icut = config.Q_CUTS.index(FEEDBACK_CUT)

    def spectrum(variant: str, masked: bool) -> np.ndarray:
        d = data[variant]
        return d["dl_masked"][icut] if masked else d["dl_fullsky"]

    fig, axes = plt.subplots(
        2, 2, figsize=(11.0, 6.6), sharex=True, height_ratios=[2.4, 1],
        gridspec_kw=dict(hspace=0.06, wspace=0.22),
    )
    titles = ("unmasked (total tSZ)", rf"masked, $q>{FEEDBACK_CUT:g}$")

    for col, (masked, title) in enumerate(zip((False, True), titles)):
        ax, axr = axes[0, col], axes[1, col]
        reference = spectrum(config.FIDUCIAL, masked)
        for variant in config.VARIANTS:
            dl = spectrum(variant, masked)
            style = dict(
                color=config.COLORS[variant],
                lw=2.2 if variant == config.FIDUCIAL else 1.4,
                label=config.LABELS[variant],
            )
            ax.loglog(ell[inside], dl[inside], **style)
            axr.semilogx(ell[inside], (dl / reference)[inside], **{**style, "label": None})

        axr.axhline(1.0, color="k", lw=1.0, zorder=0)
        ax.set_title(title, fontsize=11)
        _finish(ax, xlabel=False)
        _finish(axr, xlabel=True)

    axes[0, 0].set_ylabel(r"$\ell(\ell+1)C_\ell^{yy}/2\pi$")
    axes[1, 0].set_ylabel("ratio to fiducial")
    axes[0, 0].legend(fontsize=7.5, loc="upper left", ncol=1, frameon=False)
    fig.suptitle(
        "FLAMINGO L1_m9 feedback variants: tSZ power spectrum before and after cluster masking",
        fontsize=12,
    )

    _save(fig, "feedback_tsz_ps")


def figure_theory() -> None:
    """Draw the fiducial unmasked tSZ power spectrum against the hmfast halo model."""
    from flamingo.theory import cl_yy, dl_of_cl
    from flamingo.theory.clyy import M_GRID, Z_GRID

    data = load(config.FIDUCIAL)
    ell = data["ell"]
    inside = (ell >= THEORY_ELL_RANGE[0]) & (ell <= THEORY_ELL_RANGE[1])
    measured = data["dl_fullsky"]

    ell_th = np.geomspace(*THEORY_ELL_RANGE, 64)
    model = cl_yy(ell_th, B=config.THEORY_B)
    dl = {key: dl_of_cl(ell_th, model[f"cl{suffix}"]) for key, suffix in
          (("total", ""), ("1h", "_1h"), ("2h", "_2h"))}

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(6.4, 7.4), sharex=True, height_ratios=[3.0, 1],
        gridspec_kw=dict(hspace=0.06),
    )

    ax.loglog(ell[inside], measured[inside], "k-", lw=2.2, label="FLAMINGO L1_m9 map")
    # The 2-halo term is ~2% of the signal, so the total sits on top of the
    # 1-halo curve: draw it wide and pale so the 1-halo dashes stay visible.
    ax.loglog(ell_th, dl["total"], color="#c0392b", lw=4.0, alpha=0.35,
              label="halo model, 1h + 2h", solid_capstyle="round")
    ax.loglog(ell_th, dl["1h"], color="#c0392b", lw=1.3, ls="--", label="1-halo")
    ax.loglog(ell_th, dl["2h"], color="#c0392b", lw=1.3, ls=":", label="2-halo")

    # The ratio is only defined where the measurement exists.
    axr.semilogx(
        ell[inside],
        measured[inside] / np.interp(ell[inside], ell_th, dl["total"]),
        "k-", lw=1.6,
    )
    axr.axhline(1.0, color="#c0392b", lw=1.0)
    for band in (0.9, 1.1):
        axr.axhline(band, color="#c0392b", lw=0.7, ls=":")

    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{yy}/2\pi$")
    axr.set_ylabel("map / halo model")
    axr.set_ylim(0.6, 1.4)
    ax.set_title(
        "FLAMINGO L1_m9 total tSZ power spectrum vs. the A10 halo model "
        rf"($B={config.THEORY_B:g}$, D3A)",
        fontsize=11,
    )
    ax.legend(fontsize=8.5, loc="upper left", frameon=False)
    for panel in (ax, axr):
        panel.set_xscale("log")
        panel.set_xlim(*THEORY_ELL_RANGE)
        panel.grid(alpha=0.25, which="both", lw=0.4)
    ax.tick_params(labelbottom=False)
    axr.set_xlabel(r"multipole $\ell$")

    ax.text(
        0.97, 0.05,
        rf"$M\in[10^{{{np.log10(M_GRID[0]):.0f}}},\,10^{{{np.log10(M_GRID[-1]):.0f}}}]\,M_\odot$,"
        rf"  $z\in[{Z_GRID[0]:g},\,{Z_GRID[-1]:g}]$",
        transform=ax.transAxes, fontsize=8.5, color="0.35", ha="right",
    )

    _save(fig, "fiducial_tsz_ps_vs_halo_model")


def _save(fig, name: str) -> None:
    """Write a figure as PDF and PNG into ``figures/``."""
    config.FIGURES_PAPER.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = config.FIGURES_PAPER / f"{name}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"wrote {path.relative_to(config.REPO)}", flush=True)
    plt.close(fig)


FIGURES = {
    "fiducial": figure_fiducial,
    "feedback": figure_feedback,
    "theory": figure_theory,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figure", choices=sorted(FIGURES), action="append")
    args = parser.parse_args()

    plt.rcParams.update({"font.size": 10, "text.usetex": False, "mathtext.fontset": "cm"})
    for name in args.figure or sorted(FIGURES):
        FIGURES[name]()


if __name__ == "__main__":
    main()
