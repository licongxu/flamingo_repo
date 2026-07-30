"""FLAMINGO fiducial tSZ vs Planck: unmasked and masked (q>6).

Compares pure-tSZ FLAMINGO bandpowers against:

* Bolliet et al. (2018) marginalised tSZ (``planck_sz_1712_00788v1.txt``, col 2)
* Tanimura et al. (2021) full-sky tSZ after foreground subtraction (Table 2)
* Planck uRC marginalised tSZ at ``q_cut=6`` from TopoSZ Figure 12
  (Rotti et al. 2020, arXiv:2010.07797): column 1 of
  ``..._bf_fg_from_TSZ+P18_l_clyy_sigclyy_cib_ir_rs_cn.txt``.

FLAMINGO theory curves use the 18 inclusive Planck bins for
``10 <= ell <= 959.5``, then switch to the 12 logarithmic bins
(``Delta ln ell = 0.4``) at higher multipoles.

Run::

    python scripts/plot_flamingo_planck_tszps.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data_paper" / "binned_bandpowers"
TOPOSZ = REPO / "ref_package" / "TopoSZ" / "fig12 data"
FIGURES = REPO / "figures" / "planck_comparison"
TAG = "qfrommz_alpha_fixed_1p12"

PLANCK_B18 = TOPOSZ / "planck_sz_1712_00788v1.txt"
PLANCK_MASKED = TOPOSZ / (
    "szpowespectrum_measurement_urc_snr6_p18cmb_bf_fg_from_TSZ+P18_l_clyy_sigclyy_cib_ir_rs_cn.txt"
)

N_PLANCK_BINS = 18
ELL_PLANCK_MAX = 959.5
ELL_XMIN = 8.0
ELL_XMAX = 1.1e4

VARIANTS = (
    ("L1_m9", "#2ca02c", "-"),
    ("L2p8_m9", "#9467bd", "--"),
)

# Tanimura et al. (2021) Table 2: ell_eff, data, sigma [1e12 D_ell_yy units].
TANIMURA2021 = np.array([
    [68.5,   0.048, 0.103],
    [89.5,   0.068, 0.055],
    [117.0,  0.101, 0.046],
    [152.5,  0.127, 0.038],
    [198.0,  0.161, 0.033],
    [257.5,  0.208, 0.037],
    [335.5,  0.261, 0.029],
    [436.5,  0.318, 0.039],
    [567.5,  0.380, 0.060],
    [738.0,  0.472, 0.097],
    [959.5,  0.550, 0.162],
    [1247.5, 0.686, 0.285],
])


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    return arr[:, 0], arr[:, 1]


def _bin18_path(variant: str, *, masked: bool) -> Path:
    if masked:
        return DATA / f"Dl_yy_{variant}_masked_qgt6_{TAG}_binned_18.txt"
    return DATA / f"Dl_yy_{variant}_fullsky_binned_18.txt"


def _logbin_path(variant: str, *, masked: bool) -> Path:
    if masked:
        return DATA / f"Dl_yy_{variant}_masked_qgt6_{TAG}_logbins_dln0p4_lmax10000.txt"
    return DATA / (
        f"Dl_yy_{variant}_fiducial_fullsky_logbins_dln0p4_lmax10000_pixwin_deconvolved.txt"
    )


def _hybrid_curve(variant: str, *, masked: bool) -> tuple[np.ndarray, np.ndarray]:
    """18 Planck bins at low ell, 12 log bins above ``ELL_PLANCK_MAX``."""
    ell18, dl18 = _load(_bin18_path(variant, masked=masked))
    elllog, dllog = _load(_logbin_path(variant, masked=masked))
    hi = elllog > ELL_PLANCK_MAX
    return np.concatenate([ell18, elllog[hi]]), np.concatenate([dl18, dllog[hi]])


def _planck_bolliet() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(PLANCK_B18)[:N_PLANCK_BINS]
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _planck_masked_marginalised() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(PLANCK_MASKED)[:N_PLANCK_BINS]
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _save(fig: plt.Figure, stem: Path) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        out = stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"wrote {out.relative_to(REPO)}", flush=True)
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({"font.size": 10, "text.usetex": False, "mathtext.fontset": "cm"})

    ell_b, dl_b, sig_b = _planck_bolliet()
    ell_m, dl_m, sig_m = _planck_masked_marginalised()

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(ELL_XMIN, ELL_XMAX)
    ax.set_ylim(5e-4, 3.0)
    ax.set_xlabel(r"multipole $\ell$")
    ax.set_ylabel(r"$10^{12}\,\ell(\ell+1)C_\ell^{yy}/(2\pi)$")

    for variant, color, ls in VARIANTS:
        ell_fs, dl_fs = _hybrid_curve(variant, masked=False)
        keep_fs = (ell_fs >= ELL_XMIN) & (ell_fs <= ELL_XMAX)
        ax.loglog(
            ell_fs[keep_fs],
            dl_fs[keep_fs],
            color=color,
            ls=ls,
            lw=2.2,
            zorder=3,
        )

        path_mk = _logbin_path(variant, masked=True)
        if not path_mk.exists():
            raise FileNotFoundError(
                f"{path_mk} missing; run `python scripts/compute_masked_ps_qgt6.py {variant}`"
            )
        ell_mk, dl_mk = _hybrid_curve(variant, masked=True)
        keep_mk = (ell_mk >= ELL_XMIN) & (ell_mk <= ELL_XMAX)
        ax.loglog(
            ell_mk[keep_mk],
            dl_mk[keep_mk],
            color=color,
            ls=":" if ls == "-" else "-.",
            lw=1.8,
            alpha=0.95,
            zorder=3,
        )

    ax.errorbar(
        ell_b,
        dl_b,
        yerr=sig_b,
        fmt="^",
        mfc="none",
        mec="#2ca02c",
        ecolor="#2ca02c",
        color="#2ca02c",
        alpha=0.45,
        capsize=3,
        elinewidth=1.2,
        markersize=6,
        zorder=5,
    )
    ax.errorbar(
        ell_m,
        dl_m,
        yerr=sig_m,
        fmt="o",
        color="k",
        ecolor="k",
        capsize=4,
        elinewidth=1.5,
        markersize=4,
        zorder=6,
    )
    ax.errorbar(
        TANIMURA2021[:, 0],
        TANIMURA2021[:, 1],
        yerr=TANIMURA2021[:, 2],
        fmt="s",
        mfc="none",
        mec="#d62728",
        ecolor="#d62728",
        color="#d62728",
        capsize=3,
        elinewidth=1.2,
        markersize=5,
        zorder=5,
    )

    style_handles = [
        Line2D([0], [0], color="#2ca02c", ls="-", lw=2.2, label="L1_m9"),
        Line2D([0], [0], color="#9467bd", ls="--", lw=2.2, label="L2p8_m9"),
        Line2D([0], [0], color="0.35", ls="-", lw=2.0, label="full sky"),
        Line2D([0], [0], color="0.35", ls=":", lw=1.8, label=rf"masked, $q>6$"),
        Line2D(
            [0], [0], marker="^", mfc="none", mec="#2ca02c", ls="None",
            markersize=6, label="Planck marginalised tSZ (Bolliet+18)",
        ),
        Line2D(
            [0], [0], marker="o", color="k", ls="None", markersize=4,
            label=r"Planck uRC marginalised tSZ ($q>6$, Fig.~12)",
        ),
        Line2D(
            [0], [0], marker="s", mfc="none", mec="#d62728", ls="None",
            markersize=5, label="Planck tSZ (Tanimura+21, Table 2)",
        ),
    ]
    ax.legend(
        handles=style_handles,
        fontsize=8.5,
        loc="lower right",
        frameon=False,
        handlelength=2.2,
    )
    ax.set_title(
        r"FLAMINGO fiducial tSZ vs Planck ($\alpha_{\rm SZ}=1.12$ best-fit $q$)",
        fontsize=11,
    )
    _save(fig, FIGURES / "flamingo_planck_tszps_qgt6")


if __name__ == "__main__":
    main()
