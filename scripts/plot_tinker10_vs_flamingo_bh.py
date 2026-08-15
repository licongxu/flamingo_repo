"""Compare Tinker10 b_h(M_500c) to public FLAMINGO L1_m9 DMO P_hm/P_mm.

Uses the van Daalen et al. (2026) resummation data product:
b_i = <P_mh,i(k)/P_mm(k)>_{k low} / f_{M,i}  (their eq. 1, mass-unnormalised P_mh).

    python scripts/plot_tinker10_vs_flamingo_bh.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from flamingo.catalogue.frame import D3A_COSMOLOGY  # noqa: E402
from hmfast.halos import HaloModel  # noqa: E402
from hmfast.halos.mass_definition import MassDefinition  # noqa: E402

PKG = Path("/scratch/scratch-lxu/venv/cmbagent_env/lib/python3.12/site-packages/resummation")
FIGURES = REPO / "figures" / "masked_ps"
LAST_K = 7  # same k-range as resummation.__getbiases
SNAPS = {0.0: "0077", 0.5: "0067", 1.0: "0057"}
M_CEN = {
    0.0: np.array([10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75, 15.25]),
    0.5: np.array([10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75]),
    1.0: np.array([10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75]),
}

PAPER_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "legend.fontsize": 9,
}


def _load_json_pk(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    return np.asarray(data["k_values"], dtype=float), np.asarray(data["power_spectrum"], dtype=float)


def flamingo_bh(z: float) -> tuple[np.ndarray, np.ndarray]:
    snap = SNAPS[z]
    logm = M_CEN[z]
    k_mm, p_mm = _load_json_pk(PKG / "spectra" / f"L1000N1800_DMO_FIDUCIAL_{snap}_autopower.json")
    f_m = np.loadtxt(PKG / "fractions" / f"massfrac_DMO_FIDUCIAL_L1000N1800_{snap}_HBT_500_crit.dat", ndmin=1)
    bh = np.empty(logm.size)
    for i in range(logm.size):
        _, p_hm = _load_json_pk(
            PKG / "spectra" / f"L1000N1800_DMO_FIDUCIAL_{snap}_500_crit_crosspower_HBT_bin_{i}.json"
        )
        bh[i] = float(np.mean(p_hm[1:LAST_K] / p_mm[1:LAST_K]) / f_m[i])
    return 10.0**logm, bh


def tinker_bh(mass: np.ndarray, z: float) -> np.ndarray:
    halo_model = HaloModel(
        cosmology=D3A_COSMOLOGY,
        mass_definition=MassDefinition(500, "critical"),
        convert_masses=True,
        hm_consistency=False,
    )
    return np.asarray(halo_model.halo_bias.halo_bias(halo_model, mass, np.array([z]))[:, 0])


def main() -> Path:
    plt.rcParams.update(PAPER_RC)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    for ax, z in zip(axes, (0.0, 0.5, 1.0)):
        mass, b_fl = flamingo_bh(z)
        b_t10 = tinker_bh(mass, z)
        ax.loglog(mass, b_t10, color="0.15", lw=2.0, label=r"Tinker 2010")
        ax.loglog(mass, b_fl, "o", color="#d95f02", ms=5, label=r"FLAMINGO L1\_m9 DMO")
        ax.set_title(rf"$z={z:g}$")
        ax.set_xlabel(r"$M_{500c}\,[M_\odot]$")
        ax.grid(True, which="both", alpha=0.25)
        print(f"z={z:g}")
        for m, bf, bt in zip(mass, b_fl, b_t10):
            print(f"  logM={np.log10(m):5.2f}  b_FL={bf:6.3f}  b_T10={bt:6.3f}  ratio={bf/bt:5.3f}")
    axes[0].set_ylabel(r"$b_h(M_{500c})$")
    axes[2].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "bh_tinker10_vs_flamingo_l1_m9_dmo"
    for suffix in ("png", "pdf"):
        out = stem.with_suffix(f".{suffix}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)
    return stem.with_suffix(".pdf")


if __name__ == "__main__":
    main()
