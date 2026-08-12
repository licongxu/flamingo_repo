"""Plot all L1_m9 feedback / fiducial masked tSZ ratios at several sky cuts.

Bandpowers: paper NaMaster products under ``data_paper/binned_bandpowers``
and full-sky products under ``data_paper/feedback_bandpower``.

Ratio curves: full sky, ``q > 20``, ``q > 10``, ``q > 5`` (no q=50, q=3, q=1).

Error bands (around ratio = 1 only) as shaded regions:

* full-sky theory 1σ: ``σ_full / D_ℓ^{fid, fullsky}``
* masked q>5 theory 1σ: ``σ_{q>5} / D_ℓ^{fid, q>5}``

y-axis range follows the ratio curves (with padding), so lines stay visible
even when the full-sky error band is wide.

Style: dpi=300, no gridlines.

Run::

    python scripts/figures/plot_l1_m9_feedback_ratio_vs_q.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data_paper" / "binned_bandpowers"
DATA_FB = REPO / "data_paper" / "feedback_bandpower"
COV = REPO / "data_paper" / "covariance"
FIGURES = REPO / "figures" / "feedback"
TAG = "qfrommz_alpha_fixed_1p12"
OUTPUT_TAG = "alpha_fixed_1p12"
SELECTION_DESCRIPTION = r"$\alpha_{\rm SZ}=1.12$"

FIDUCIAL = "fiducial"
DEFAULT_VARIANT = "fgas-8sigma"
DPI = 300

# Ratio curves: full sky replaces former q>50; no q=1,3.
DEFAULT_CURVES: list[str | float] = ["fullsky", 20.0, 10.0, 5.0]

# Error bands about 1.0 only (colors taken from matching curve colors at draw time).
ERROR_BAND_KINDS = (
    ("fullsky", "full sky"),
    (5.0, r"$q>5$"),
)

# y-padding around ratio curves (fraction of span, plus absolute floor).
Y_PAD_FRAC = 0.12
Y_PAD_ABS = 0.03

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

VARIANT_LABELS = {
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


def cut_tag(q: float) -> str:
    if float(q).is_integer():
        return f"qgt{int(q)}"
    return f"qgt{str(q).replace('.', 'p')}"


def curve_label(kind: str | float) -> str:
    if kind == "fullsky":
        return "full sky"
    if float(kind).is_integer():
        return rf"$q>{int(kind)}$"
    return rf"$q>{float(kind):g}$"


def error_band_description(
    kinds: tuple[tuple[str | float, str], ...] | None = None,
) -> str:
    """Human-readable list of uncertainty bands actually drawn."""
    kinds = ERROR_BAND_KINDS if kinds is None else kinds
    return " and ".join(label for _, label in kinds)


def bandpower_path(
    variant: str,
    kind: str | float,
    *,
    log: bool,
    selection_tag: str | None = None,
) -> Path:
    """Path to paper bandpowers for full sky or a masked q cut."""
    selection_tag = TAG if selection_tag is None else selection_tag
    suffix = "logbins_dln0p4_lmax10000" if log else "binned_18"
    if kind == "fullsky":
        if variant in ("fiducial", "L1_m9"):
            p = DATA_FB / f"Dl_yy_L1_m9_fiducial_fullsky_{suffix}.txt"
            if p.is_file():
                return p
            if log:
                return (
                    DATA
                    / "Dl_yy_L1_m9_fiducial_fullsky_logbins_dln0p4_lmax10000_pixwin_deconvolved.txt"
                )
            return DATA / "Dl_yy_L1_m9_fullsky_binned_18.txt"
        return DATA_FB / f"Dl_yy_L1_m9_{variant}_fullsky_{suffix}.txt"

    tag = cut_tag(float(kind))
    if variant in ("fiducial", "L1_m9"):
        return DATA / f"Dl_yy_L1_m9_masked_{tag}_{selection_tag}_{suffix}.txt"
    return DATA / f"Dl_yy_L1_m9_{variant}_masked_{tag}_{selection_tag}_{suffix}.txt"


def fullsky_bandpower_path(*, log: bool) -> Path:
    """Fiducial full-sky bandpowers (compat helper)."""
    return bandpower_path(FIDUCIAL, "fullsky", log=log)


def cov_path_for_band(kind: str | float, *, log: bool) -> Path:
    suffix = "logbins_dln0p4_lmax10000" if log else "binned_18"
    if kind == "fullsky":
        label = "fullsky"
    else:
        label = f"masked_{cut_tag(float(kind))}"
    return COV / f"cov_full_L1_m9_customgnfw_bestfit_{label}_Dl_yy_{suffix}.npy"


def load_bandpowers(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing bandpower product: {path}")
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def load_sigma_1e12(kind: str | float, *, log: bool) -> np.ndarray:
    path = cov_path_for_band(kind, log=log)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing theory covariance: {path}\n"
            "Run scripts/powerspectra/compute_l1_m9_customgnfw_bestfit_covariance.py first."
        )
    cov = np.load(path)
    return np.sqrt(np.diag(np.asarray(cov, dtype=float))) * 1e12


def ratio_at_cut(
    variant: str,
    kind: str | float,
    *,
    log: bool,
    fiducial: str = FIDUCIAL,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ell, Dl_variant / Dl_fiducial)`` at the same sky/mask cut."""
    ell_v, dl_v = load_bandpowers(bandpower_path(variant, kind, log=log))
    ell_f, dl_f = load_bandpowers(bandpower_path(fiducial, kind, log=log))
    if not np.array_equal(ell_v, ell_f):
        raise ValueError(f"ell grid mismatch at {kind}")
    if np.any(dl_f == 0.0) or not np.all(np.isfinite(dl_f)):
        raise ValueError(f"invalid fiducial spectrum at {kind}")
    return ell_v, dl_v / dl_f


# Back-compat aliases used by tests / older call sites.
def ratio_at_q(variant: str, q: float, *, log: bool, fiducial: str = FIDUCIAL):
    return ratio_at_cut(variant, q, log=log, fiducial=fiducial)


def ratios_by_q(
    variant: str,
    q_cuts: list[str | float] | None = None,
    *,
    log: bool,
    fiducial: str = FIDUCIAL,
) -> tuple[np.ndarray, dict[str | float, np.ndarray]]:
    """Return ``(ell, {cut: ratio})`` for fullsky and/or q cuts."""
    if q_cuts is None:
        q_cuts = list(DEFAULT_CURVES)
    ratios: dict[str | float, np.ndarray] = {}
    ell_ref: np.ndarray | None = None
    for kind in q_cuts:
        ell, ratio = ratio_at_cut(variant, kind, log=log, fiducial=fiducial)
        if ell_ref is None:
            ell_ref = ell
        elif not np.array_equal(ell, ell_ref):
            # fullsky and masked share the same binning convention; allow tiny float noise
            if not np.allclose(ell, ell_ref, rtol=0.0, atol=5e-4):
                raise ValueError(f"ell grid mismatch at {kind}")
            ell = ell_ref
        ratios[kind] = ratio
    assert ell_ref is not None
    return ell_ref, ratios


def relative_error_band(
    kind: str | float,
    *,
    log: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ell, σ_D / D_ℓ^{fid})`` for fullsky or a masked q cut."""
    ell, dl_f = load_bandpowers(bandpower_path(FIDUCIAL, kind, log=log))
    sigma = load_sigma_1e12(kind, log=log)
    if sigma.shape != ell.shape:
        raise ValueError(f"sigma length {sigma.shape} != ell length {ell.shape} for {kind}")
    return ell, sigma / dl_f


def _curve_colors(n: int) -> np.ndarray:
    return plt.cm.plasma(np.linspace(0.15, 0.85, n))


def _color_map_for_curves(curves: list[str | float]) -> dict[str | float, np.ndarray]:
    """Map each curve kind to its line color (for matching error bands)."""
    colors = _curve_colors(len(curves))
    return {kind: colors[i] for i, kind in enumerate(curves)}


def _error_band_color(
    kind: str | float,
    curve_colors: dict[str | float, np.ndarray],
) -> np.ndarray | str:
    """Full-sky band is light grey; q-cut bands match (darkened) curve color."""
    if kind == "fullsky":
        return "#d0d0d0"
    if kind in curve_colors:
        # Darken the curve color so the band stays visible under lines.
        c = np.asarray(curve_colors[kind], dtype=float).copy()
        c[:3] = np.clip(c[:3] * 0.72, 0.0, 1.0)
        return c
    return "#1f77b4"


def _error_band_alpha(kind: str | float) -> float:
    """q>5 needs a stronger fill than the wide full-sky band."""
    return 0.22 if kind == "fullsky" else 0.45


def _ylim_from_ratios(
    ratios: dict[str | float, np.ndarray],
    inside: np.ndarray,
) -> tuple[float, float]:
    """Tight y-limits from ratio curves only (ignore wide error bands)."""
    stack = np.concatenate([r[inside] for r in ratios.values()])
    stack = stack[np.isfinite(stack)]
    lo = float(np.min(stack))
    hi = float(np.max(stack))
    span = max(hi - lo, 1e-3)
    pad = max(Y_PAD_FRAC * span, Y_PAD_ABS)
    return lo - pad, hi + pad


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


def _draw_ratio_panel(
    ax: plt.Axes,
    variant: str,
    *,
    curves: list[str | float],
    log: bool,
    ell_range: tuple[float, float],
    show_curve_legend: bool = False,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Ratio curves + fullsky/q>5 error bands about 1; ylim from curves or fixed."""
    ell, ratios = ratios_by_q(variant, curves, log=log)
    curve_colors = _color_map_for_curves(curves)
    inside = (ell >= ell_range[0]) & (ell <= ell_range[1])
    x = ell[inside]
    if ylim is None:
        ymin, ymax = _ylim_from_ratios(ratios, inside)
    else:
        ymin, ymax = ylim

    # Error bands about 1 (clipped visually by ylim); q>5 uses darker/stronger fill.
    for kind, label in ERROR_BAND_KINDS:
        color = _error_band_color(kind, curve_colors)
        ell_e, rel = relative_error_band(kind, log=log)
        if not np.allclose(ell_e, ell, rtol=0.0, atol=5e-4):
            rel = np.interp(ell, ell_e, rel)
        s = rel[inside]
        ax.fill_between(
            x,
            1.0 - s,
            1.0 + s,
            color=color,
            alpha=_error_band_alpha(kind),
            linewidth=0,
            zorder=1,
            label=rf"{label} $1\sigma$" if show_curve_legend else None,
        )

    ax.axhline(1.0, color="k", lw=1.2, zorder=2)

    for kind in curves:
        color = curve_colors[kind]
        ax.semilogx(
            x,
            ratios[kind][inside],
            lw=1.6,
            color=color,
            marker="o",
            markersize=3.2,
            label=curve_label(kind) if show_curve_legend else None,
            zorder=3,
        )

    ax.set_xlim(*ell_range)
    ax.set_ylim(ymin, ymax)
    ax.set_xscale("log")
    ax.grid(False)


def _legend_handles(curves: list[str | float]) -> list:
    curve_colors = _color_map_for_curves(curves)
    handles: list = []
    for kind in curves:
        color = curve_colors[kind]
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                marker="o",
                lw=1.6,
                markersize=4,
                label=curve_label(kind),
            )
        )
    for kind, label in ERROR_BAND_KINDS:
        color = _error_band_color(kind, curve_colors)
        handles.append(
            Patch(
                facecolor=color,
                edgecolor="none",
                alpha=_error_band_alpha(kind),
                label=rf"{label} $1\sigma$",
            )
        )
    return handles


def plot_ratio_vs_q(
    variant: str = DEFAULT_VARIANT,
    *,
    q_cuts: list[str | float] | None = None,
    log: bool,
    ell_range: tuple[float, float],
    stem: Path | None = None,
    ax: plt.Axes | None = None,
) -> Path | None:
    """One-variant ratio panel; optional external axes."""
    curves = list(DEFAULT_CURVES if q_cuts is None else q_cuts)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
    assert ax is not None

    _draw_ratio_panel(
        ax,
        variant,
        curves=curves,
        log=log,
        ell_range=ell_range,
        show_curve_legend=own_fig,
    )

    if own_fig:
        ax.set_xlabel(r"multipole $\ell$")
        ax.set_ylabel(r"$D_\ell^{\rm variant} / D_\ell^{\rm fiducial}$")
        vlab = VARIANT_LABELS.get(variant, variant)
        bin_note = r"12 log bins ($\Delta\ln\ell=0.4$)" if log else "18 Planck bins"
        ax.set_title(
            rf"FLAMINGO L1_m9: {vlab} / fiducial ratio vs sky cut"
            f"\n"
            rf"({bin_note}; {SELECTION_DESCRIPTION}; "
            rf"bands: {error_band_description()} theory $1\sigma$ about 1)",
            fontsize=10.5,
        )
        ax.legend(
            handles=_legend_handles(curves),
            fontsize=8.0,
            loc="best",
            frameon=False,
            ncol=2,
        )
        if stem is None:
            bin_tag = "logbins" if log else "binned_18"
            stem = FIGURES / f"l1_m9_{variant}_ratio_vs_q_{bin_tag}_{OUTPUT_TAG}"
        return _save(fig, stem)
    return None


def plot_all_feedback_ratio_vs_q(
    *,
    q_cuts: list[str | float] | None = None,
    log: bool,
    ell_range: tuple[float, float],
    stem: Path | None = None,
    variants: list[str] | None = None,
    shared_ylim: bool = True,
) -> Path:
    """Multi-panel figure: one panel per feedback prescription."""
    curves = list(DEFAULT_CURVES if q_cuts is None else q_cuts)
    if variants is None:
        variants = list(FEEDBACK_VARIANTS)

    # Shared y-range across panels from all ratio curves (not error bands).
    ylim: tuple[float, float] | None = None
    if shared_ylim:
        all_ratios: dict[str | float, np.ndarray] = {}
        ell_ref: np.ndarray | None = None
        for variant in variants:
            ell, ratios = ratios_by_q(variant, curves, log=log)
            if ell_ref is None:
                ell_ref = ell
            for key, arr in ratios.items():
                # Disambiguate keys across variants for concat only.
                all_ratios[f"{variant}:{key}"] = arr
        assert ell_ref is not None
        inside = (ell_ref >= ell_range[0]) & (ell_ref <= ell_range[1])
        ylim = _ylim_from_ratios(all_ratios, inside)
        print(f"shared ylim ({'log' if log else '18'}): {ylim[0]:.4f} … {ylim[1]:.4f}", flush=True)

    n = len(variants)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.35 * ncols, 2.7 * nrows + 0.9),
        sharex=True,
        sharey=shared_ylim,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    for i, variant in enumerate(variants):
        ax = axes_flat[i]
        _draw_ratio_panel(
            ax,
            variant,
            curves=curves,
            log=log,
            ell_range=ell_range,
            show_curve_legend=False,
            ylim=ylim,
        )
        ax.set_title(VARIANT_LABELS.get(variant, variant), fontsize=10)
        if i // ncols == nrows - 1:
            ax.set_xlabel(r"multipole $\ell$")
        if i % ncols == 0:
            ax.set_ylabel(r"variant / fiducial")
        ax.grid(False)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    handles = _legend_handles(curves)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, -0.02),
    )
    bin_note = r"12 log bins ($\Delta\ln\ell=0.4$)" if log else "18 Planck bins"
    fig.suptitle(
        rf"FLAMINGO L1_m9 feedback / fiducial ratio vs sky cut ({bin_note})"
        "\n"
        rf"shaded bands about 1: {error_band_description()} custom-GNFW theory $1\sigma$ "
        r"($\sigma_D/D_\ell^{\rm fid}$); y-range from ratio curves",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.96))

    if stem is None:
        bin_tag = "logbins" if log else "binned_18"
        stem = FIGURES / f"l1_m9_all_feedback_ratio_vs_q_{bin_tag}_{OUTPUT_TAG}"
    return _save(fig, stem)


def main() -> None:
    global ERROR_BAND_KINDS, OUTPUT_TAG, SELECTION_DESCRIPTION, TAG
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", choices=(TAG, "qfrommap"), default=TAG)
    args = parser.parse_args()
    TAG = args.selection
    if TAG == "qfrommap":
        OUTPUT_TAG = "qfrommap"
        SELECTION_DESCRIPTION = "empirical aperture q"
        ERROR_BAND_KINDS = (("fullsky", "full sky"),)
    plt.rcParams.update(
        {
            "font.size": 10,
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "axes.grid": False,
        }
    )
    curves = list(DEFAULT_CURVES)
    print(f"data: {DATA.relative_to(REPO)}", flush=True)
    print(f"cov:  {COV.relative_to(REPO)}", flush=True)
    print(f"variants: {FEEDBACK_VARIANTS}", flush=True)
    print(f"curves: {curves}", flush=True)
    print(
        "error bands about 1: " + ", ".join(str(kind) for kind, _ in ERROR_BAND_KINDS),
        flush=True,
    )
    print(f"dpi={DPI}, gridlines=off, ylim=from curves", flush=True)

    for kind in curves:
        for log in (False, True):
            bp = bandpower_path(DEFAULT_VARIANT, kind, log=log)
            print(
                f"  bp={'ok' if bp.is_file() else 'MISSING'}  {kind} " f"{'log' if log else '18'}",
                flush=True,
            )

    # Shared y only for the 18-bin multi-panel (user request). Log-bin panels
    # keep per-panel ylim so the q>5 error band stays visible.
    plot_all_feedback_ratio_vs_q(
        q_cuts=curves, log=False, ell_range=(10.0, 959.5), shared_ylim=True
    )
    plot_all_feedback_ratio_vs_q(
        q_cuts=curves, log=True, ell_range=(100.0, 10000.0), shared_ylim=False
    )
    plot_ratio_vs_q(DEFAULT_VARIANT, q_cuts=curves, log=False, ell_range=(10.0, 959.5))
    plot_ratio_vs_q(DEFAULT_VARIANT, q_cuts=curves, log=True, ell_range=(100.0, 10000.0))


if __name__ == "__main__":
    main()
