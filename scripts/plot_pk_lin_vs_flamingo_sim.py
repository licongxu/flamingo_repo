"""FLAMINGO P_mm vs hmfast pk() in the units the hdf5 actually uses.

Official FLAMINGO files: k [Mpc^{-1}] (no h), P [Mpc^3], BoxSize in cMpc.
hmfast Cosmology.pk() *claims* the same. This figure does not convert either
spectrum into h-units. Colossus linear P, converted (Mpc/h)^3 -> Mpc^3, is
the unit referee.

    python scripts/plot_pk_lin_vs_flamingo_sim.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from colossus.cosmology import cosmology

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from flamingo.catalogue.frame import D3A_COSMOLOGY  # noqa: E402

FIGURES = REPO / "figures" / "masked_ps"
HYDRO = Path("/rds/rds-lxu/flamingo/power_spectra/L1_m9.hdf5")
DMO = Path("/rds/rds-lxu/flamingo/power_spectra/L1_m9_DMO.hdf5")
REDSHIFTS = (0.0, 0.5, 1.0)
H = 0.681

PAPER_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "legend.fontsize": 8,
}

_COLOSSUS = cosmology.setCosmology(
    "d3a",
    dict(H0=68.1, Om0=0.306, Ob0=0.0486, sigma8=0.807, ns=0.967, relspecies=False),
)


def _sim_pk(path: Path, z: float) -> tuple[np.ndarray, np.ndarray]:
    key = f"z={z:.2f}"
    with h5py.File(path, "r") as handle:
        k = np.asarray(handle[f"{key}/k"][:], dtype=float)
        p = np.asarray(handle[f"{key}/P(k)"][:], dtype=float)
    return k, np.maximum(p, 0.0)


def _hmfast_pk(z: float) -> tuple[np.ndarray, np.ndarray]:
    k, p = D3A_COSMOLOGY.pk(z, linear=True)
    return np.asarray(k, dtype=float).ravel(), np.asarray(p, dtype=float).ravel()


def _colossus_pk_mpc(z: float, k_inv_mpc: np.ndarray) -> np.ndarray:
    """Linear P in Mpc^3 at k in Mpc^{-1}. Colossus native: h/Mpc and (Mpc/h)^3."""
    k_h = np.asarray(k_inv_mpc, dtype=float) / H
    return _COLOSSUS.matterPowerSpectrum(k_h, z=z) / H**3


def main() -> Path:
    plt.rcParams.update(PAPER_RC)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    k_ref = np.geomspace(8e-3, 3.0, 200)
    for ax, z in zip(axes, REDSHIFTS):
        k_lin, p_lin = _hmfast_pk(z)
        k_h, p_h = _sim_pk(HYDRO, z)
        k_d, p_d = _sim_pk(DMO, z)
        ax.loglog(k_lin, p_lin, color="0.15", lw=2.0, label=r"hmfast $P_{\mathrm{lin}}$ as returned")
        ax.loglog(k_ref, _colossus_pk_mpc(z, k_ref), color="#1b9e77", lw=1.8, label=r"colossus $P_{\mathrm{lin}}$ $[\mathrm{Mpc}^{3}]$")
        ax.loglog(k_h, p_h, color="#d95f02", lw=1.6, label=r"FLAMINGO hydro $P_{mm}$")
        ax.loglog(k_d, p_d, color="#7570b3", lw=1.4, ls="--", label=r"FLAMINGO DMO $P_{mm}$")
        ax.set_title(rf"$z={z:g}$")
        ax.set_xlabel(r"$k\,[\mathrm{Mpc}^{-1}]$")
        ax.set_xlim(6e-3, 5.0)
        ax.set_ylim(3e1, 4e5)
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel(r"$P(k)\,[\mathrm{Mpc}^3]$")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle(r"No $h$-unit conversion: FLAMINGO native $Mpc^{-1}$, $Mpc^3$", y=1.02)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "pk_lin_hmfast_vs_flamingo_sim"
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=200, bbox_inches="tight")
        print("wrote", stem.with_suffix(f".{suffix}"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    k_h, p_h = _sim_pk(HYDRO, 0.5)
    k_lin, p_lin = _hmfast_pk(0.5)
    p_col = _colossus_pk_mpc(0.5, k_h)
    p_lin_i = np.interp(k_h, k_lin, p_lin)
    ax.axhline(1.0, color="0.15", lw=1.0)
    ax.semilogx(k_h, p_h / p_col, color="#1b9e77", lw=2.0, label=r"FLAMINGO / colossus")
    ax.semilogx(k_h, p_lin_i / p_col, color="0.15", lw=2.0, label=r"hmfast $P_{\mathrm{lin}}$ / colossus")
    ax.set_xlabel(r"$k\,[\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$P / P_{\mathrm{lin}}^{\mathrm{colossus}}(Mpc^3)$")
    ax.set_title(r"$z=0.5$")
    ax.set_xlim(6e-3, 5.0)
    ax.set_ylim(0.0, 4.0)
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    stem_r = FIGURES / "pk_lin_hmfast_vs_flamingo_sim_ratio_z05"
    for suffix in ("png", "pdf"):
        fig.savefig(stem_r.with_suffix(f".{suffix}"), dpi=200, bbox_inches="tight")
        print("wrote", stem_r.with_suffix(f".{suffix}"))
    plt.close(fig)
    return stem.with_suffix(".pdf")


if __name__ == "__main__":
    main()
