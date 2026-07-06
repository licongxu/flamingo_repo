"""Seed analysis 01: empirical CNC x tSZ-PS coupling across the 8 L2p8_m9 lightcones.

Two questions that motivate the joint-analysis paper:

1. How correlated are the cluster counts and the tSZ power spectrum bandpowers,
   and does masking detected clusters (q > q_cut) suppress that correlation?
2. How much does masking shrink the realization scatter (non-Gaussian cosmic
   variance) of the bandpowers relative to the Gaussian (Knox) expectation?

Data (all cached, no map access needed):
- data/bandpowers_L2p8_m9_multilc/Dl_yy_lc{0..7}_{fullsky,qgt5,qgt10,qgt20,qgt50}.txt
- data/cnc/L2p8_m9_M500c_binned_multilc.npz  (counts_lc: (8, 15 z, 12 M))

Caveat recorded in the outputs: the 8 lightcones are 8 observers in the SAME
L2p8 box, so they share large-scale structure; correlations and scatter are
only indicative (lower bound on independence). N=8 gives sigma(r) ~ 0.38.

Outputs (this folder):
- seed01_results.npz
- seed01_fig.png / seed01_fig.pdf
- printed summary table
"""

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/scratch/scratch-lxu/flamingo_repo")
BP_DIR = REPO / "data" / "bandpowers_L2p8_m9_multilc"
CNC_NPZ = REPO / "data" / "cnc" / "L2p8_m9_M500c_binned_multilc.npz"
OUT = REPO / "autoresearch" / "fable_5_plan"

TAGS = ["fullsky", "qgt50", "qgt20", "qgt10", "qgt5"]
N_LC = 8
# Broad ell bands for the correlation statistic (delta_ell=30 native bins are too noisy)
BAND_EDGES = np.array([60, 150, 300, 600, 1000, 1500, 2500, 4000, 6000])


def load_bandpowers():
    """Return ell (native centres) and dl[tag] with shape (8, n_ell)."""
    ell = None
    dl = {}
    for tag in TAGS:
        rows = []
        for lc in range(N_LC):
            arr = np.loadtxt(BP_DIR / f"Dl_yy_lc{lc}_{tag}.txt")
            if ell is None:
                ell = arr[:, 0]
            rows.append(arr[:, 1])
        dl[tag] = np.vstack(rows)
    return ell, dl


def rebin(ell, dl_lc):
    """Average native bandpowers into broad bands; returns (band_centres, (8, n_band))."""
    centres, binned = [], []
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        sel = (ell >= lo) & (ell < hi)
        centres.append(ell[sel].mean())
        binned.append(dl_lc[:, sel].mean(axis=1))
    return np.array(centres), np.array(binned).T


def pearson_across_lc(x, y_matrix):
    """Pearson r between x (8,) and each column of y_matrix (8, n)."""
    xs = (x - x.mean()) / x.std()
    ys = (y_matrix - y_matrix.mean(axis=0)) / y_matrix.std(axis=0)
    return (xs[:, None] * ys).mean(axis=0)


def main():
    ell, dl = load_bandpowers()

    cnc = np.load(CNC_NPZ)
    counts = cnc["counts_lc"]  # (8, 15 z, 12 M)
    z_edges, m_edges = cnc["bins_edges_z"], cnc["bins_edges_M"]
    # "Detected-like" counts: massive clusters that a q>5 SZ selection would find.
    # M > 2e14 Msun is a crude proxy for the q>5 threshold at low z.
    m_hi = m_edges[:-1] >= 2e14
    n_tot = counts.sum(axis=(1, 2)).astype(float)
    n_massive = counts[:, :, m_hi].sum(axis=(1, 2)).astype(float)
    print(f"z edges [{z_edges[0]:.2f}, {z_edges[-1]:.2f}], M edges "
          f"[{m_edges[0]:.1e}, {m_edges[-1]:.1e}] Msun")
    print(f"N_tot per lc      : {n_tot.astype(int)}")
    print(f"N(M>2e14) per lc  : {n_massive.astype(int)}\n")

    band_ell = None
    corr_tot, corr_massive, frac_scatter = {}, {}, {}
    for tag in TAGS:
        band_ell, db = rebin(ell, dl[tag])
        corr_tot[tag] = pearson_across_lc(n_tot, db)
        corr_massive[tag] = pearson_across_lc(n_massive, db)
        frac_scatter[tag] = db.std(axis=0, ddof=1) / db.mean(axis=0)

    # Knox Gaussian fractional scatter for the broad bands (fsky=1 reference):
    # sigma(D_b)/D_b = sqrt(2 / sum_{ell in band} (2 ell + 1)) for a full-sky field.
    knox = []
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        ells = np.arange(lo, hi)
        knox.append(np.sqrt(2.0 / np.sum(2.0 * ells + 1.0)))
    knox = np.array(knox)

    print(f"{'band ell':>9} | " + " | ".join(f"r(N_m, D) {t:>8}" for t in TAGS))
    for i, be in enumerate(band_ell):
        print(f"{be:9.0f} | " + " | ".join(f"{corr_massive[t][i]:18.2f}" for t in TAGS))
    print()
    print(f"{'band ell':>9} | " + " | ".join(f"sig/D {t:>8}" for t in TAGS) + " |  Knox")
    for i, be in enumerate(band_ell):
        row = " | ".join(f"{frac_scatter[t][i]:14.4f}" for t in TAGS)
        print(f"{be:9.0f} | {row} | {knox[i]:.4f}")

    mean_r = {t: float(np.mean(corr_massive[t])) for t in TAGS}
    print("\nband-averaged r(N_massive, D_ell):",
          {t: round(v, 2) for t, v in mean_r.items()})

    git_hash = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True).stdout.strip()

    np.savez(
        OUT / "seed01_results.npz",
        band_ell=band_ell, band_edges=BAND_EDGES, knox_fullsky=knox,
        n_tot=n_tot, n_massive=n_massive,
        **{f"corr_massive_{t}": corr_massive[t] for t in TAGS},
        **{f"corr_tot_{t}": corr_tot[t] for t in TAGS},
        **{f"frac_scatter_{t}": frac_scatter[t] for t in TAGS},
        caveat="8 observers in one L2p8 box: not independent realizations",
        git_hash=git_hash,
    )

    # ---- figure: 2 panels ----
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(TAGS)))
    styles = ["-", "--", "-.", ":", "-"]
    markers = ["o", "s", "^", "D", "v"]
    labels = {"fullsky": "full sky", "qgt50": r"$q>50$ masked",
              "qgt20": r"$q>20$ masked", "qgt10": r"$q>10$ masked",
              "qgt5": r"$q>5$ masked"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for tag, c, s, m in zip(TAGS, colors, styles, markers):
        axes[0].plot(band_ell, corr_massive[tag], s, marker=m, color=c,
                     label=labels[tag], ms=5)
        axes[1].plot(band_ell, frac_scatter[tag], s, marker=m, color=c,
                     label=labels[tag], ms=5)
    axes[0].axhline(0.0, color="0.6", lw=0.8)
    sig8 = 1.0 / np.sqrt(N_LC - 1)
    axes[0].axhspan(-sig8, sig8, color="0.9", zorder=0,
                    label=r"$\pm 1/\sqrt{7}$ (null, $N=8$)")
    axes[0].set_xlabel(r"$\ell$")
    axes[0].set_ylabel(r"$r\,(N_{\mathrm{cl}}(M>2\times10^{14}),\ D_\ell^{yy})$")
    axes[0].set_xscale("log")
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].set_title("counts-bandpower correlation (8 lightcones)")

    axes[1].plot(band_ell, knox, "k:", lw=1.5, label="Knox (Gaussian, $f_{sky}=1$)")
    axes[1].set_xlabel(r"$\ell$")
    axes[1].set_ylabel(r"$\sigma(D_\ell)/\langle D_\ell\rangle$ across lightcones")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    axes[1].set_title("bandpower realization scatter")

    fig.suptitle("Seed 01: CNC-tSZ PS coupling vs masking, L2p8_m9 lc0-7 "
                 f"(git {git_hash})", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "seed01_fig.png", dpi=300)
    fig.savefig(OUT / "seed01_fig.pdf")
    print(f"\nwrote {OUT}/seed01_results.npz, seed01_fig.png|pdf")

    with open(OUT / "seed01_summary.json", "w") as f:
        json.dump({"mean_r_massive": mean_r,
                   "frac_scatter_ell_150_300": {t: float(frac_scatter[t][1]) for t in TAGS},
                   "git_hash": git_hash,
                   "caveat": "8 observers share one L2p8 box"}, f, indent=2)


if __name__ == "__main__":
    main()
