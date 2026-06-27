"""Figure: binned cluster number counts for the B=1.35 catalogue (vs B=1)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
CAT135 = _REPO / "data/hydro_L2p8m9/catalogue/halo_catalogue_M500c_5e13_zlt3_y0q_arnaudB135.csv"
CAT1 = _REPO / "data/hydro_L2p8m9/catalogue/halo_catalogue_M500c_5e13_zlt3_y0q_arnaudB1.csv"
REF = _REPO / "data/cnc/observed_cnc_binned_qgt5.npz"
OUTDIR = _REPO / "figures/nb28_cnc_arnaudB135"
OUTDIR.mkdir(parents=True, exist_ok=True)

ref = np.load(REF, allow_pickle=True)
z_edges, q_edges = ref["bins_edges_z"], ref["bins_edges_q"]
z_centres, q_centres = ref["z_centres"], ref["q_centres"]


def counts(path):
    df = pd.read_csv(path, comment="#")
    z, q = df["z"].to_numpy(float), df["q"].to_numpy(float)
    g = np.isfinite(q)
    z, q = z[g], q[g]
    N2d, _, _ = np.histogram2d(z, q, bins=[z_edges, q_edges])
    return N2d, int(np.sum(q > 5.0))


N2d_135, ntot_135 = counts(CAT135)
N2d_1, ntot_1 = counts(CAT1)
Nz_135, Nq_135 = N2d_135.sum(1), N2d_135.sum(0)
Nz_1, Nq_1 = N2d_1.sum(1), N2d_1.sum(0)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

# (a) N(z)
ax = axes[0]
ax.step(z_centres, Nz_1, where="mid", color="#0072B2", lw=1.8, ls="--",
        label=f"B=1  (N$_{{q>5}}$={ntot_1})")
ax.step(z_centres, Nz_135, where="mid", color="#D55E00", lw=1.8, ls="-",
        label=f"B=1.35  (N$_{{q>5}}$={ntot_135})")
ax.set_xlabel("redshift $z$")
ax.set_ylabel(r"$N(z)$ per bin,  $q>5$")
ax.legend(fontsize=9)
ax.set_title("(a) Redshift distribution")

# (b) N(q)
ax = axes[1]
x = np.arange(len(q_centres))
w = 0.4
ax.bar(x - w / 2, Nq_1, w, color="#0072B2", label="B=1")
ax.bar(x + w / 2, Nq_135, w, color="#D55E00", label="B=1.35")
ax.set_xticks(x)
ax.set_xticklabels([f"{q_edges[j]:.0f}-{q_edges[j+1]:.0f}" for j in range(len(q_centres))],
                   fontsize=8)
ax.set_xlabel("SNR bin $q$")
ax.set_ylabel(r"$N(q)$,  $5<q<40$")
ax.set_yscale("log")
ax.legend(fontsize=9)
ax.set_title("(b) SNR distribution")

# (c) 2-D heatmap (B=1.35)
ax = axes[2]
masked = np.ma.masked_where(N2d_135.T == 0, N2d_135.T)
im = ax.pcolormesh(z_edges, np.arange(len(q_edges)), masked,
                   cmap="viridis", shading="flat")
ax.set_yticks(np.arange(len(q_centres)) + 0.5)
ax.set_yticklabels([f"{q_edges[j]:.0f}" for j in range(len(q_centres))], fontsize=8)
ax.set_xlabel("redshift $z$")
ax.set_ylabel("SNR $q$ (lower edge)")
ax.set_title("(c) $N(z,q)$, B=1.35")
cb = fig.colorbar(im, ax=ax)
cb.set_label("clusters per bin")
for i in range(len(z_centres)):
    for j in range(len(q_centres)):
        v = N2d_135[i, j]
        if v > 0:
            ax.text(z_centres[i], j + 0.5, f"{int(v)}", ha="center", va="center",
                    fontsize=6, color="w")

fig.suptitle(
    "FLAMINGO L2p8_m9 derived CNC: A10 GNFW at hydrostatic mass $M_{500c}/B$ "
    r"(cosmocnc convention); amplitude $\times(1/B)^{1.12}$, $\theta_{500}\times(1/B)^{1/3}$ (B=1.35)",
    fontsize=10,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUTDIR / "cnc_arnaudB135_counts.pdf")
fig.savefig(OUTDIR / "cnc_arnaudB135_counts.png", dpi=300)
print(f"B=1   N(q>5)={ntot_1}")
print(f"B=1.35 N(q>5)={ntot_135}")
print(f"wrote {OUTDIR}/cnc_arnaudB135_counts.{{pdf,png}}")
