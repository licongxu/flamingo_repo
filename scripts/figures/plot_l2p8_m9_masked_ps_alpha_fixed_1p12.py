"""Plot fiducial L2p8_m9 lightcone-0 masked tSZ bandpowers (alpha-fixed q).

Run::

    python scripts/figures/plot_l2p8_m9_masked_ps_alpha_fixed_1p12.py
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
VARIANT = "L2p8_m9"
TAG = "qfrommz_alpha_fixed_1p12"

Q_CUTS = [50.0, 20.0, 10.0, 5.0, 1.0]
CUT_TAGS = ["qgt50", "qgt20", "qgt10", "qgt5", "qgt1"]

FULLSKY_18 = DATA / f"Dl_yy_{VARIANT}_fullsky_binned_18.txt"
FULLSKY_LOG = DATA / (
    f"Dl_yy_{VARIANT}_fiducial_fullsky_logbins_dln0p4_lmax10000_pixwin_deconvolved.txt"
)
META = DATA / f"{VARIANT}_masked_{TAG}_metadata.json"

OUT_18 = FIGURES / f"l2p8_m9_masked_ps_binned_18_alpha_fixed_1p12"
OUT_LOG = FIGURES / f"l2p8_m9_masked_ps_logbins_alpha_fixed_1p12"


def _load_two_column(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def _masked_paths(
    tag: str,
    *,
    log: bool,
    selection_tag: str = TAG,
    data: Path = DATA,
) -> Path:
    suffix = "logbins_dln0p4_lmax10000" if log else "binned_18"
    variant = f"{VARIANT}_lc0" if selection_tag == "qfrommap" else VARIANT
    return data / f"Dl_yy_{variant}_masked_{tag}_{selection_tag}_{suffix}.txt"


def output_stem(*, log: bool, selection_tag: str = TAG) -> Path:
    bin_tag = "logbins" if log else "binned_18"
    suffix = "qfrommap" if selection_tag == "qfrommap" else "alpha_fixed_1p12"
    return FIGURES / f"l2p8_m9_masked_ps_{bin_tag}_{suffix}"


def _save(fig: plt.Figure, stem: Path) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        out = stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        print(f"wrote {out.relative_to(REPO)}", flush=True)
    plt.close(fig)


def _plot(
    ell_full: np.ndarray,
    dl_full: np.ndarray,
    masked: list[tuple[float, str, np.ndarray, np.ndarray]],
    *,
    ell_range: tuple[float, float],
    title: str,
    stem: Path,
    masked_meta: dict,
) -> None:
    inside = (ell_full >= ell_range[0]) & (ell_full <= ell_range[1])
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(masked)))

    fig, (ax, axr) = plt.subplots(
        2,
        1,
        figsize=(6.4, 6.4),
        sharex=True,
        height_ratios=[2.4, 1.0],
        gridspec_kw={"hspace": 0.06},
    )
    ax.loglog(
        ell_full[inside],
        dl_full[inside],
        "k-",
        lw=2.0,
        marker="o",
        markersize=4.0,
        label="full sky",
        zorder=3,
    )
    for (cut, tag, ell, dl), color in zip(masked, colors):
        keep = (ell >= ell_range[0]) & (ell <= ell_range[1])
        meta = masked_meta[tag]
        label = (
            rf"$q>{cut:g}$  ($N={meta['n_masked']:,}$, "
            rf"$f_{{\rm sky}}={meta['f_sky_eff']:.3f}$)"
        )
        ax.loglog(ell[keep], dl[keep], lw=1.5, color=color, marker="o", markersize=3.5, label=label)
        ratio_ell = ell_full[inside]
        ratio = np.interp(ratio_ell, ell[keep], dl[keep]) / dl_full[inside]
        axr.semilogx(ratio_ell, ratio, lw=1.5, color=color)

    axr.axhline(1.0, color="k", lw=1.0)
    ax.set_ylabel(r"$10^{12}\,\ell(\ell+1)C_\ell^{yy}/(2\pi)$")
    axr.set_ylabel("masked / full sky")
    axr.set_ylim(0.0, 1.05)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    for panel in (ax, axr):
        panel.set_xscale("log")
        panel.set_xlim(*ell_range)
    ax.tick_params(labelbottom=False)
    axr.set_xlabel(r"multipole $\ell$")
    _save(fig, stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selection", choices=(TAG, "qfrommap"), default=TAG)
    args = parser.parse_args()
    selection_tag = args.selection
    plt.rcParams.update({"font.size": 10, "text.usetex": False, "mathtext.fontset": "cm"})

    meta_path = (
        DATA / "L2p8_m9_lc0_masked_qfrommap_metadata.json"
        if selection_tag == "qfrommap"
        else META
    )
    with meta_path.open() as handle:
        meta_doc = json.load(handle)
    masked_meta = meta_doc["cuts"]
    q_description = (
        "empirical aperture q"
        if selection_tag == "qfrommap"
        else r"$\alpha_{\rm SZ}=1.12$ best-fit $q$"
    )

    ell18, dl18_full = _load_two_column(FULLSKY_18)
    masked18 = []
    for cut, tag in zip(Q_CUTS, CUT_TAGS):
        ell, dl = _load_two_column(
            _masked_paths(tag, log=False, selection_tag=selection_tag)
        )
        masked18.append((cut, tag, ell, dl))
    _plot(
        ell18,
        dl18_full,
        masked18,
        ell_range=(10.0, 959.5),
        title=(
            "FLAMINGO L2p8_m9 fiducial (lc0): masked tSZ power spectrum "
            f"({q_description}, 18 Planck bins)"
        ),
        stem=output_stem(log=False, selection_tag=selection_tag),
        masked_meta=masked_meta,
    )

    ell_log, dl_log_full = _load_two_column(FULLSKY_LOG)
    masked_log = []
    for cut, tag in zip(Q_CUTS, CUT_TAGS):
        ell, dl = _load_two_column(
            _masked_paths(tag, log=True, selection_tag=selection_tag)
        )
        masked_log.append((cut, tag, ell, dl))
    _plot(
        ell_log,
        dl_log_full,
        masked_log,
        ell_range=(100.0, 10000.0),
        title=(
            "FLAMINGO L2p8_m9 fiducial (lc0): masked tSZ power spectrum "
            f"({q_description}, $\\Delta\\ln\\ell=0.4$ log bins)"
        ),
        stem=output_stem(log=True, selection_tag=selection_tag),
        masked_meta=masked_meta,
    )
