"""Publication figures for paper_draft: FLAMINGO L1_m9 progressive masking.

Reads only cached ensembles (no map access, no GPU):
  * data/bandpowers_L1_m9_feedback/masked_tsz_ps.npz   (nb35 cache)
  * data/nb43_L1_m9_patch_cov/patch_ensemble.npz       (nb43 cache)

Writes PDF + 300 dpi PNG + sidecar JSON into paper_draft/figures/:
  * l1m9_progressive_masking.{pdf,png}   fiducial D_ell per q cut + ratio panel
  * l1m9_feedback_ratio.{pdf,png}        variant/fiducial ratios, 4 masking levels
  * l1m9_cnc_ps_corrmat.{pdf,png}        joint [N(z), C_ell] corr, full vs masked
  * l1m9_cross_corr_vs_ell.{pdf,png}     corr[N(q>5), C_ell] vs ell

Run:
    python scripts/make_paper_masking_figures.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
OUT = _REPO / "paper_draft" / "figures"
NB35 = _REPO / "data/bandpowers_L1_m9_feedback/masked_tsz_ps.npz"
NB43 = _REPO / "data/nb43_L1_m9_patch_cov/patch_ensemble.npz"

# Wong (2011) colour-blind-safe palette
WONG = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
]

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
    }
)

VARIANT_LABELS = {
    "L1_m9": "fiducial",
    "fgas+2sigma": r"fgas$+2\sigma$",
    "fgas-2sigma": r"fgas$-2\sigma$",
    "fgas-4sigma": r"fgas$-4\sigma$",
    "fgas-8sigma": r"fgas$-8\sigma$",
    "Mstar-1sigma": r"M$_*-\sigma$",
    "Mstar-1sigma_fgas-4sigma": r"M$_*-\sigma$, fgas$-4\sigma$",
    "Jet": "Jet",
    "Jet_fgas-4sigma": r"Jet, fgas$-4\sigma$",
}


def _git_hash() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=_REPO
    ).stdout.strip()


def _save(fig: plt.Figure, stem: str, meta: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    meta = dict(meta, git_hash=_git_hash(), script="scripts/make_paper_masking_figures.py")
    (OUT / f"{stem}.json").write_text(json.dumps(meta, indent=2) + "\n")
    plt.close(fig)
    print(f"wrote {stem}.pdf/.png")


def fig_progressive_masking() -> None:
    d = np.load(NB35, allow_pickle=True)
    ellb = d["ellb"]
    variants = list(d["variants"])
    tags = list(d["cut_tags"])
    i0 = variants.index("L1_m9")
    sel = ellb >= 100
    cmap = plt.get_cmap("viridis")
    cut_labels = [r"$q>50$", r"$q>20$", r"$q>10$", r"$q>5$", r"$q>3$", r"$q>1$"]

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(5.2, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.06},
    )
    ax.loglog(ellb[sel], d["dl_fullsky"][i0][sel], color="k", lw=1.8, label="full sky")
    for j, lab in enumerate(cut_labels):
        c = cmap(0.15 + 0.75 * j / (len(cut_labels) - 1))
        ax.loglog(ellb[sel], d["dl_masked"][i0, j][sel], color=c, lw=1.4, label=f"mask {lab}")
        axr.semilogx(
            ellb[sel], d["dl_masked"][i0, j][sel] / d["dl_fullsky"][i0][sel], color=c, lw=1.4
        )
    axr.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_ylabel(r"$D_\ell^{yy} = \ell(\ell+1)C_\ell^{yy}/2\pi$")
    axr.set_ylabel(r"masked / full sky")
    axr.set_xlabel(r"multipole $\ell$")
    axr.set_ylim(0, 1.05)
    ax.legend(ncol=2, frameon=False, loc="lower right")
    _save(
        fig,
        "l1m9_progressive_masking",
        {
            "data": str(NB35.relative_to(_REPO)),
            "variant": "L1_m9 fiducial",
            "q_cuts": [50, 20, 10, 5, 3, 1],
        },
    )


def fig_feedback_ratio() -> None:
    d = np.load(NB35, allow_pickle=True)
    ellb = d["ellb"]
    variants = list(d["variants"])
    tags = list(d["cut_tags"])
    i0 = variants.index("L1_m9")
    sel = ellb >= 100
    panels = [
        ("full sky", None),
        (r"mask $q>10$", tags.index("qgt10")),
        (r"mask $q>5$", tags.index("qgt5")),
        (r"mask $q>1$", tags.index("qgt1")),
    ]
    others = [v for v in variants if v != "L1_m9"]
    colors = ["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#000000", "#888888"]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2), sharex=True, sharey=True)
    for ax, (title, jcut) in zip(axes.ravel(), panels):
        for k, v in enumerate(others):
            i = variants.index(v)
            if jcut is None:
                r = d["dl_fullsky"][i] / d["dl_fullsky"][i0]
            else:
                r = d["dl_masked"][i, jcut] / d["dl_masked"][i0, jcut]
            ax.semilogx(
                ellb[sel], r[sel], color=colors[k % len(colors)],
                ls=["-", "--", "-.", ":"][k % 4], lw=1.3, label=VARIANT_LABELS[v],
            )
        ax.axhline(1.0, color="0.4", lw=0.8, ls=":")
        ax.text(0.04, 0.06, title, transform=ax.transAxes, fontsize=11)
        ax.set_ylim(0.6, 1.15)
    for ax in axes[1]:
        ax.set_xlabel(r"multipole $\ell$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$D_\ell$ / $D_\ell^{\rm fiducial}$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 1.01), fontsize=9)
    fig.subplots_adjust(hspace=0.08, wspace=0.06, top=0.90)
    _save(
        fig,
        "l1m9_feedback_ratio",
        {"data": str(NB35.relative_to(_REPO)), "panels": [p[0] for p in panels]},
    )


def _joint_corr(d: dict, key: str) -> tuple[np.ndarray, int]:
    nz_tot = d["N_z"].sum(axis=0)
    keep = nz_tot > 0
    X = np.hstack([d["N_z"][:, keep], d[key]])
    return np.corrcoef(X, rowvar=False), int(keep.sum())


def fig_corrmat() -> None:
    d = np.load(NB43, allow_pickle=True)
    ell_eff = d["ell_eff"]
    corr_full, nzb = _joint_corr(d, "dl_full")
    corr_mask, _ = _joint_corr(d, "dl_mask")
    z_edges = d["z_edges"]
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])[:nzb]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), sharey=True)
    for ax, corr, title in zip(axes, [corr_full, corr_mask], ["full sky", r"masked $q>5$"]):
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
        ax.axhline(nzb - 0.5, color="k", lw=0.9)
        ax.axvline(nzb - 0.5, color="k", lw=0.9)
        ticks = list(range(0, nzb, 2)) + [nzb + i for i in range(0, 9, 2)]
        labels = [f"$z$={zc[i]:.2f}" for i in range(0, nzb, 2)] + [
            rf"$\ell$={ell_eff[i]:.0f}" for i in range(0, 9, 2)
        ]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.set_title(title, fontsize=12)
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label("correlation coefficient")
    _save(
        fig,
        "l1m9_cnc_ps_corrmat",
        {"data": str(NB43.relative_to(_REPO)), "n_patch": 192, "n_z_bins": nzb},
    )


def fig_cross_corr() -> None:
    d = np.load(NB43, allow_pickle=True)
    ell_eff = d["ell_eff"]
    n_tot = d["N_z"].sum(axis=1)
    rf = [np.corrcoef(n_tot, d["dl_full"][:, b])[0, 1] for b in range(9)]
    rm = [np.corrcoef(n_tot, d["dl_mask"][:, b])[0, 1] for b in range(9)]
    null = 1.0 / np.sqrt(d["N_z"].shape[0])

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axhspan(-null, null, color="0.85", label=r"null $\pm 1/\sqrt{n_{\rm patch}}$")
    ax.axhline(0, color="k", lw=0.8)
    ax.semilogx(ell_eff, rf, "o-", color=WONG[6], label="full sky")
    ax.semilogx(ell_eff, rm, "s--", color=WONG[5], label=r"masked $q>5$")
    ax.set_xlabel(r"multipole $\ell_{\rm eff}$")
    ax.set_ylabel(r"corr$[\,N_{\rm cl}(q>5),\ C_\ell\,]$")
    ax.legend(frameon=False, loc="upper left")
    _save(
        fig,
        "l1m9_cross_corr_vs_ell",
        {
            "data": str(NB43.relative_to(_REPO)),
            "mean_r_full": float(np.mean(rf)),
            "mean_r_mask": float(np.mean(rm)),
        },
    )


if __name__ == "__main__":
    fig_progressive_masking()
    fig_feedback_ratio()
    fig_corrmat()
    fig_cross_corr()
