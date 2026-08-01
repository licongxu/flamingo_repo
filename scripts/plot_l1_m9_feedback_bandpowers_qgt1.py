"""Plot L1_m9 feedback-variant tSZ bandpowers at q>1 (full sky and masked).

Same layout as ``plot_l1_m9_feedback_bandpowers.py`` but uses the ``qgt1``
masked products. Full-sky curves are reused from the existing feedback run.

Run::

    python scripts/plot_l1_m9_feedback_bandpowers_qgt1.py
"""

from __future__ import annotations

import argparse
import json
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
CUT_TAG = "qgt1"
Q_CUT = 1.0

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


def _path(
    variant: str,
    *,
    masked: bool,
    log: bool,
    selection_tag: str | None = None,
) -> Path:
    selection_tag = TAG if selection_tag is None else selection_tag
    suffix = "logbins_dln0p4_lmax10000" if log else "binned_18"
    if masked:
        root = MASKED_DATA if selection_tag == "qfrommap" else DATA
        token = "" if selection_tag == "qfrommap" and variant == "fiducial" else f"_{variant}"
        return root / f"Dl_yy_L1_m9{token}_masked_{CUT_TAG}_{selection_tag}_{suffix}.txt"
    return DATA / f"Dl_yy_L1_m9_{variant}_fullsky_{suffix}.txt"


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def _load_error_sigmas(*, log: bool) -> dict[bool, tuple[np.ndarray, np.ndarray]]:
    """Custom-GNFW best-fit theory 1-sigma bandpower errors (1e-12 units)."""
    name = (
        "Dl_yy_customgnfw_bestfit_theory_logbins_dln0p4_lmax10000.txt"
        if log
        else "Dl_yy_customgnfw_bestfit_theory_binned_18.txt"
    )
    data = np.loadtxt(COV / name)
    return {
        False: (data[:, 0], data[:, 7] * 1e12),
        True: (data[:, 0], data[:, 12] * 1e12),
    }


def _save(fig: plt.Figure, stem: Path) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        out = stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"wrote {out.relative_to(REPO)}", flush=True)
    plt.close(fig)


def _figure(
    *,
    log: bool,
    ell_range: tuple[float, float],
    title: str,
    stem: Path,
    error_sigmas: dict[bool, tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(VARIANTS) - 1))

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
        ell_ref, dl_ref = _load(_path("fiducial", masked=masked, log=log))
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
            ell, dl = _load(_path(variant, masked=masked, log=log))
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

        ax.set_title("full sky" if not masked else rf"masked, $q>{Q_CUT:g}$", fontsize=10)
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
    _save(fig, stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", choices=(TAG, "qfrommap"), default=TAG)
    args = parser.parse_args()
    TAG = args.selection
    plt.rcParams.update({"font.size": 10, "text.usetex": False, "mathtext.fontset": "cm"})

    if TAG == "qfrommap":
        meta_path = MASKED_DATA / "L1_m9_feedback_multi_q_bandpowers_qfrommap_metadata.json"
        output_tag = "qfrommap"
        q_description = "empirical aperture q"
        error_sigmas_18 = None
        error_sigmas_log = None
    else:
        meta_path = DATA / f"L1_m9_feedback_bandpowers_{CUT_TAG}_metadata.json"
        output_tag = "alpha_fixed_1p12"
        q_description = r"$q$ from $\alpha_{\rm SZ}=1.12$ best fit"
        error_sigmas_18 = _load_error_sigmas(log=False)
        error_sigmas_log = _load_error_sigmas(log=True)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        print("f_sky_eff per variant:")
        for variant in VARIANTS:
            entry = meta["variants"][variant]
            if TAG == "qfrommap":
                entry = entry["cuts"][CUT_TAG]
            print(
                f"  {variant:26s} N_masked={entry['n_masked']:5d}  f_sky={entry['f_sky_eff']:.4f}"
            )

    _figure(
        log=False,
        ell_range=(10.0, 959.5),
        title=(
            "FLAMINGO L1_m9 feedback variants: tSZ power spectrum "
            rf"(18 Planck bins; masked $q>{Q_CUT:g}$; {q_description})"
            + (
                ""
                if TAG == "qfrommap"
                else "\n"
                + r"errors: custom-GNFW $\alpha_{\rm SZ}=1.12$, $B=1.41$, "
                + r"best-fit $A_{\rm SZ}$ covariance"
            )
        ),
        stem=FIGURES / f"l1_m9_feedback_ps_binned_18_{output_tag}_qgt1",
        error_sigmas=error_sigmas_18,
    )
    _figure(
        log=True,
        ell_range=(100.0, 10000.0),
        title=(
            "FLAMINGO L1_m9 feedback variants: tSZ power spectrum "
            rf"($\Delta\ln\ell=0.4$ log bins; masked $q>{Q_CUT:g}$; "
            f"{q_description})"
            + (
                ""
                if TAG == "qfrommap"
                else "\n"
                + r"errors: custom-GNFW $\alpha_{\rm SZ}=1.12$, $B=1.41$, "
                + r"best-fit $A_{\rm SZ}$ covariance"
            )
        ),
        stem=FIGURES / f"l1_m9_feedback_ps_logbins_{output_tag}_qgt1",
        error_sigmas=error_sigmas_log,
    )
