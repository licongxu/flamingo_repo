"""Redshift decomposition of the tSZ power spectrum into lightcone shell groups.

The yang26 lightcone applies a fresh random rotation of the periodic box after
every group of shells, so the 13 rotation groups of the L1_m9 lightcone are
near-independent redshift slices: their power spectra add up to the spectrum of
the summed map, and the residual of that closure is a measure of how much
structure is shared between slices.

``decomposition``
    The fiducial L1_m9 total tSZ power spectrum broken into the contribution of
    each rotation group, with the fractional contribution per group underneath.

``shells``
    Each group's measured spectrum against the ``hmfast`` halo model integrated
    over that group's own redshift range, at ``B = 1``. Nothing is fitted.

``feedback`` / ``feedback_ratio``
    The same per-group spectra for all nine L1_m9 feedback variants, in absolute
    terms and as a ratio to the fiducial run: which redshifts carry the
    feedback sensitivity of the total signal.

Inputs are the cached ``D_ell`` under :data:`~paper_results.config.ROTGROUP_DIR`
(monopole subtracted, ``anafast`` with ``iter=0``, pixel window deconvolved,
``f_sky`` corrected).

Run::

    python -m paper_results.rotation_groups            # all figures
    python -m paper_results.rotation_groups --figure decomposition
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import config

#: Multipole range shown. As elsewhere in the pipeline, below ~100 the finite
#: lightcone box scale bites; the maps are measured up to ``config.LMAX``.
ELL_RANGE = (100.0, 6000.0)

#: Number of logarithmic bandpowers the raw per-multipole spectra are binned to.
N_BANDPOWERS = 24

#: Panel grid used by the per-group figures: 13 groups into 4x4, three spare.
GRID = (4, 4)

#: Redshift grid resolution of the per-group halo-model prediction.
N_Z_THEORY = 48

#: LaTeX-safe rendering of the run name (the underscore needs escaping).
RUN = r"\texttt{L1\_m9}"


def load(variant: str) -> dict:
    """Load the cached rotation-group spectra of one feedback variant.

    Parameters
    ----------
    variant : str
        Feedback-variant name.

    Returns
    -------
    dict
        Contents of ``<ROTGROUP_DIR>/<variant>.npz``: per-group ``dl`` of shape
        ``(n_groups, lmax + 1)``, the summed-map spectrum ``dl_all``, and the
        shell and redshift bounds of every group.
    """
    path = config.ROTGROUP_DIR / f"{variant}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; build the rotation-group maps first")
    return dict(np.load(path, allow_pickle=True))


def log_bin(ell: np.ndarray, dl: np.ndarray, n_bins: int = N_BANDPOWERS) -> tuple:
    """Average a per-multipole spectrum into logarithmic bandpowers.

    Parameters
    ----------
    ell : numpy.ndarray
        Multipoles, one per element of ``dl``.
    dl : numpy.ndarray
        Spectrum to bin.
    n_bins : int, optional
        Number of bins spanning :data:`ELL_RANGE`.

    Returns
    -------
    tuple of numpy.ndarray
        Bin-centre multipoles and binned spectrum, skipping empty bins.
    """
    edges = np.geomspace(*ELL_RANGE, n_bins + 1)
    which = np.digitize(ell, edges) - 1
    keep = [b for b in range(n_bins) if np.any(which == b)]
    centres = np.array([ell[which == b].mean() for b in keep])
    binned = np.array([dl[which == b].mean() for b in keep])
    return centres, binned


def variant_label(variant: str) -> str:
    """Legend label of a feedback variant, safe to typeset with LaTeX."""
    return "fiducial" if variant == config.FIDUCIAL else config.LABELS[variant]


def group_title(data: dict, gi: int) -> str:
    """Panel title of one rotation group: its shells and its redshift range."""
    return (
        rf"group {gi}: shells {data['shell_first'][gi]}--{data['shell_last'][gi]}"
        "\n"
        rf"$z \in [{data['z_inner'][gi]:.2f},\,{data['z_outer'][gi]:.2f}]$"
    )


def group_colors(n_groups: int) -> mpl.colors.ListedColormap:
    """One colour per rotation group, ordered by redshift."""
    return mpl.colors.ListedColormap(plt.cm.viridis(np.linspace(0.05, 0.95, n_groups)))


def redshift_colorbar(fig, axes, data: dict, cmap) -> None:
    """Attach a colour bar whose segments are the groups' redshift ranges."""
    edges = np.concatenate(([data["z_inner"][0]], data["z_outer"]))
    mappable = mpl.cm.ScalarMappable(
        norm=mpl.colors.BoundaryNorm(edges, cmap.N), cmap=cmap
    )
    # 'uniform' spacing gives every group an equally tall segment, so the bar
    # doubles as the legend: segment i is exactly the colour of group i.
    bar = fig.colorbar(mappable, ax=axes, spacing="uniform", ticks=edges,
                       pad=0.02, fraction=0.05)
    bar.ax.set_yticklabels([f"{z:.2f}" for z in edges], fontsize=6.5)
    bar.ax.tick_params(length=2)
    bar.set_label(r"rotation-group redshift $z$", fontsize=9, labelpad=8)


def theory_dl(z_lo: float, z_hi: float) -> tuple:
    """Halo-model tSZ ``D_ell`` integrated over one group's redshift range.

    Parameters
    ----------
    z_lo, z_hi : float
        Redshift bounds of the group.

    Returns
    -------
    tuple of numpy.ndarray
        Multipoles and the total (1-halo + 2-halo) prediction at ``B = 1``.
    """
    from flamingo.theory import cl_yy, dl_of_cl

    ell = np.geomspace(*ELL_RANGE, 48)
    # The Limber projection divides by the comoving distance, so the innermost
    # group needs the same small positive floor as flamingo.theory.clyy.Z_GRID.
    z_grid = np.geomspace(max(z_lo, 0.005), z_hi, N_Z_THEORY)
    model = cl_yy(ell, B=config.THEORY_B, z_grid=z_grid)
    return ell, dl_of_cl(ell, model["cl"])


def _panels(figsize: tuple, **kwargs):
    """A 4x4 panel grid with the three unused axes turned off."""
    fig, axes = plt.subplots(*GRID, figsize=figsize, sharex=True, **kwargs)
    return fig, axes.ravel()


def _finish(ax, *, xlabel: bool) -> None:
    """Common axis styling for a bandpower panel."""
    ax.set_xscale("log")
    ax.set_xlim(*ELL_RANGE)
    if xlabel:
        ax.set_xlabel(r"multipole $\ell$")
        # The spare panels of the grid are blanked, so the bottom row of a
        # column is not always the bottom row of the figure: sharex would
        # otherwise strip the tick labels off the last populated panel.
        ax.tick_params(labelbottom=True)


def figure_decomposition() -> None:
    """Draw the fiducial tSZ power spectrum split into rotation-group shells."""
    data = load(config.FIDUCIAL)
    ell = data["ell"]
    n_groups = data["dl"].shape[0]
    cmap = group_colors(n_groups)

    centres, total = log_bin(ell, data["dl_all"])
    groups = np.array([log_bin(ell, data["dl"][gi])[1] for gi in range(n_groups)])

    fig, (ax, axf) = plt.subplots(
        2, 1, figsize=(6.6, 6.8), sharex=True, height_ratios=[2.2, 1],
        gridspec_kw=dict(hspace=0.07),
    )

    # The box is re-rotated between groups, so the slices are near-independent
    # and their spectra add back up to the total. The two curves therefore lie
    # on top of each other: draw the total wide and pale so the sum stays visible.
    ax.loglog(centres, total, "k-", lw=4.0, alpha=0.3, solid_capstyle="round",
              label=r"total map, $z<3$", zorder=4)
    ax.loglog(centres, groups.sum(axis=0), "k--", lw=1.2,
              label=r"$\sum_g D_\ell^{\,g}$", zorder=5)
    for gi in range(n_groups):
        ax.loglog(centres, groups[gi], lw=1.3, color=cmap(gi))
        axf.semilogx(centres, groups[gi] / total, lw=1.3, color=cmap(gi))

    ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{yy}/2\pi$")
    ax.legend(fontsize=9, loc="lower right", frameon=False)
    ax.set_title(
        rf"FLAMINGO {RUN}: the tSZ power spectrum by lightcone redshift shell",
        fontsize=11,
    )
    axf.set_ylabel(r"$D_\ell^{\,g}\,/\,D_\ell^{\,\rm total}$")
    axf.set_ylim(0, None)
    _finish(ax, xlabel=False)
    _finish(axf, xlabel=True)
    redshift_colorbar(fig, [ax, axf], data, cmap)

    _save(fig, "l1_m9_rotation_group_decomposition")


def figure_shells() -> None:
    """Draw each rotation group's spectrum against the halo model of its own z-range."""
    data = load(config.FIDUCIAL)
    ell = data["ell"]
    n_groups = data["dl"].shape[0]
    cmap = group_colors(n_groups)

    fig, axes = _panels((13.0, 11.0), sharey=True,
                        gridspec_kw=dict(hspace=0.32, wspace=0.08))
    for gi in range(n_groups):
        ax = axes[gi]
        centres, binned = log_bin(ell, data["dl"][gi])
        ell_th, dl_th = theory_dl(data["z_inner"][gi], data["z_outer"][gi])
        ax.loglog(centres, binned, "o", ms=3.5, color=cmap(gi), label="FLAMINGO map")
        ax.loglog(ell_th, dl_th, "-", lw=1.6, color="0.25",
                  label=rf"halo model, $B={config.THEORY_B:g}$")
        ax.set_title(group_title(data, gi), fontsize=9)
        _finish(ax, xlabel=gi >= n_groups - GRID[1])
        if gi % GRID[1] == 0:
            ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{yy}/2\pi$")

    _legend_in_spare(fig, axes, n_groups, fontsize=10)
    fig.suptitle(
        rf"FLAMINGO {RUN}: tSZ power spectrum per rotation-group shell "
        r"vs.\ the A10 halo model",
        fontsize=12,
    )
    _save(fig, "l1_m9_rotation_group_shells_vs_halo_model")


def figure_feedback() -> None:
    """Draw the per-group spectra of every feedback variant."""
    data = {v: load(v) for v in config.VARIANTS}
    fid = data[config.FIDUCIAL]
    ell = fid["ell"]
    n_groups = fid["dl"].shape[0]

    fig, axes = _panels((13.0, 11.0), sharey=True,
                        gridspec_kw=dict(hspace=0.32, wspace=0.08))
    for gi in range(n_groups):
        ax = axes[gi]
        for variant in config.VARIANTS:
            centres, binned = log_bin(ell, data[variant]["dl"][gi])
            ax.loglog(centres, binned, lw=2.0 if variant == config.FIDUCIAL else 1.3,
                      color=config.COLORS[variant], label=variant_label(variant))
        ax.set_title(group_title(fid, gi), fontsize=9)
        _finish(ax, xlabel=gi >= n_groups - GRID[1])
        if gi % GRID[1] == 0:
            ax.set_ylabel(r"$\ell(\ell+1)C_\ell^{yy}/2\pi$")

    _legend_in_spare(fig, axes, n_groups, fontsize=10)
    fig.suptitle(
        rf"FLAMINGO {RUN} feedback variants: tSZ power spectrum per rotation-group shell",
        fontsize=12,
    )
    _save(fig, "l1_m9_rotation_group_feedback")


def figure_feedback_ratio() -> None:
    """Draw the per-group feedback spectra as a ratio to the fiducial run."""
    data = {v: load(v) for v in config.VARIANTS}
    fid = data[config.FIDUCIAL]
    ell = fid["ell"]
    n_groups = fid["dl"].shape[0]

    fig, axes = _panels((13.0, 11.0), sharey=True,
                        gridspec_kw=dict(hspace=0.32, wspace=0.08))
    for gi in range(n_groups):
        ax = axes[gi]
        reference = log_bin(ell, fid["dl"][gi])[1]
        for variant in config.VARIANTS:
            if variant == config.FIDUCIAL:
                continue
            centres, binned = log_bin(ell, data[variant]["dl"][gi])
            ax.semilogx(centres, binned / reference, lw=1.3,
                        color=config.COLORS[variant], label=variant_label(variant))
        ax.axhline(1.0, color="k", lw=1.0, zorder=0)
        ax.set_ylim(0.0, 2.0)
        ax.set_title(group_title(fid, gi), fontsize=9)
        _finish(ax, xlabel=gi >= n_groups - GRID[1])
        if gi % GRID[1] == 0:
            ax.set_ylabel(r"$D_\ell\,/\,D_\ell^{\,\rm fiducial}$")

    _legend_in_spare(fig, axes, n_groups, fontsize=10)
    fig.suptitle(
        rf"FLAMINGO {RUN} feedback variants: tSZ power spectrum per rotation-group "
        r"shell, relative to the fiducial run",
        fontsize=12,
    )
    _save(fig, "l1_m9_rotation_group_feedback_ratio")


def _legend_in_spare(fig, axes, n_groups: int, **kwargs) -> None:
    """Put one shared legend into the unused panels of the grid."""
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[n_groups:]:
        ax.axis("off")
    axes[n_groups].legend(handles, labels, loc="center", frameon=False, **kwargs)


def _save(fig, name: str) -> None:
    """Write a figure as PDF and PNG into ``figures/``."""
    config.FIGURES_ROTATION_GROUPS.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = config.FIGURES_ROTATION_GROUPS / f"{name}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"wrote {path.relative_to(config.REPO)}", flush=True)
    plt.close(fig)


FIGURES = {
    "decomposition": figure_decomposition,
    "shells": figure_shells,
    "feedback": figure_feedback,
    "feedback_ratio": figure_feedback_ratio,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figure", choices=sorted(FIGURES), action="append")
    args = parser.parse_args()

    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
    })
    for name in args.figure or list(FIGURES):
        FIGURES[name]()


if __name__ == "__main__":
    main()
