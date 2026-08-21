"""Linear P(k) unit conventions: CosmoPower/CLASS native vs Cosmology.pk vs CAMB.

This figure uses one cosmology and three ways of writing the same spectrum:

  h-units (CLASS / CosmoPower native)
      k [h Mpc^{-1}],  P [(Mpc/h)^3]
  physical (Cosmology.pk since 4a7341e)
      k [Mpc^{-1}],    P [Mpc^3]
      k = h k_h,       P = P_h / h^3
  leftover 2-halo interpolant (committed halo_model.py)
      P_old(k) = P_phys(h k) * h^6   with k physical

    python scripts/plot_pk_unit_conventions.py
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import camb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hmfast.cosmology import Cosmology

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "figures"
Z = 0.0
COSMO_PARAMS = {
    "omega_b": 0.02242,
    "omega_cdm": 0.11933,
    "H0": 67.66,
    "tau_reio": 0.0561,
    "ln10^{10}A_s": 3.047,
    "n_s": 0.9665,
}
# Same massive-neutrino mass as hmfast Cosmology default; not in the dict above.
M_NCDM = 0.06


def hmfast_spectra():
    cosmology = Cosmology(
        H0=COSMO_PARAMS["H0"],
        omega_cdm=COSMO_PARAMS["omega_cdm"],
        omega_b=COSMO_PARAMS["omega_b"],
        ln1e10A_s=COSMO_PARAMS["ln10^{10}A_s"],
        n_s=COSMO_PARAMS["n_s"],
        tau_reio=COSMO_PARAMS["tau_reio"],
        m_ncdm=M_NCDM,
    )
    h = float(np.asarray(cosmology.H0)) / 100.0
    k_phys, p_phys = [np.asarray(x).ravel() for x in cosmology.pk(Z, linear=True)]
    k_h = k_phys / h
    p_h = p_phys * h**3
    p_old = np.interp(h * k_phys, k_phys, p_phys) * h**6
    return h, k_h, p_h, k_phys, p_phys, p_old


def camb_spectra(h: float):
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=COSMO_PARAMS["H0"],
        ombh2=COSMO_PARAMS["omega_b"],
        omch2=COSMO_PARAMS["omega_cdm"],
        tau=COSMO_PARAMS["tau_reio"],
        mnu=M_NCDM,
    )
    pars.InitPower.set_params(As=np.exp(COSMO_PARAMS["ln10^{10}A_s"]) * 1e-10, ns=COSMO_PARAMS["n_s"])
    pars.set_matter_power(redshifts=[Z], kmax=20.0, nonlinear=False)
    kh, _z, pk = camb.get_results(pars).get_matter_power_spectrum(
        minkh=1e-4, maxkh=10.0, npoints=500
    )
    p_h = np.asarray(pk[0], dtype=float)
    kh = np.asarray(kh, dtype=float)
    return kh, p_h, h * kh, p_h / h**3


def main() -> Path:
    h, k_h, p_h, k_phys, p_phys, p_old = hmfast_spectra()
    kh_c, p_h_c, k_c, p_c = camb_spectra(h)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
    ax0, ax1, ax2 = axes

    ax0.loglog(kh_c, p_h_c, color="0.15", lw=2.2, label="CAMB")
    ax0.loglog(k_h, p_h, color="#d95f02", lw=1.6, ls="--", label="hmfast native")
    ax0.set_xlabel(r"$k\,[h\,\mathrm{Mpc}^{-1}]$")
    ax0.set_ylabel(r"$P(k)\,[(\mathrm{Mpc}/h)^3]$")
    ax0.set_title("h-units (CLASS / CosmoPower)")
    ax0.set_xlim(1e-3, 5.0)
    ax0.legend(frameon=False)
    ax0.grid(True, which="both", alpha=0.25)

    ax1.loglog(k_c, p_c, color="0.15", lw=2.2, label="CAMB physical")
    ax1.loglog(k_phys, p_phys, color="#1b9e77", lw=1.6, ls="--", label=r"hmfast ``pk()``")
    ax1.loglog(k_phys, p_old, color="#7570b3", lw=1.6, ls=":", label=r"old 2h: $P(hk)\,h^6$")
    ax1.set_xlabel(r"$k\,[\mathrm{Mpc}^{-1}]$")
    ax1.set_ylabel(r"$P(k)\,[\mathrm{Mpc}^3]$")
    ax1.set_title("physical units")
    ax1.set_xlim(1e-3, 5.0)
    ax1.legend(frameon=False)
    ax1.grid(True, which="both", alpha=0.25)

    p_c_i = np.interp(k_phys, k_c, p_c)
    ax2.axhline(1.0, color="0.15", lw=1.0)
    ax2.semilogx(k_phys, p_phys / p_c_i, color="#1b9e77", lw=2.0, label=r"``pk()`` / CAMB")
    ax2.semilogx(k_phys, p_old / p_c_i, color="#7570b3", lw=2.0, label=r"old 2h / CAMB")
    ax2.set_xlabel(r"$k\,[\mathrm{Mpc}^{-1}]$")
    ax2.set_ylabel(r"$P / P_{\mathrm{CAMB}}^{\mathrm{phys}}$")
    ax2.set_title(r"$z=0$ ratio in physical units")
    ax2.set_xlim(1e-3, 5.0)
    ax2.set_ylim(0.0, 1.4)
    ax2.legend(frameon=False)
    ax2.grid(True, which="both", alpha=0.25)

    fig.suptitle(
        rf"$h={h:.4f}$, $z={Z:g}$; "
        r"physical $k=h k_h$, $P=P_h/h^3$",
        y=1.03,
    )
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / "pk_unit_conventions"
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=200, bbox_inches="tight")
        print("wrote", stem.with_suffix(f".{suffix}"))
    plt.close(fig)

    k_ref = 0.01
    print(f"h = {h:.6f},  h**3 = {h**3:.6f},  h**6 = {h**6:.6e}")
    print(f"at k_phys = {k_ref} Mpc^{{-1}}:")
    print(f"  CAMB physical     {np.interp(k_ref, k_c, p_c):.4e} Mpc^3")
    print(f"  hmfast pk()       {np.interp(k_ref, k_phys, p_phys):.4e} Mpc^3")
    print(f"  old 2h interpolant {np.interp(k_ref, k_phys, p_old):.4e} Mpc^3")
    print(f"  pk()/CAMB         {np.interp(k_ref, k_phys, p_phys)/np.interp(k_ref, k_c, p_c):.4f}")
    print(f"  old2h/CAMB        {np.interp(k_ref, k_phys, p_old)/np.interp(k_ref, k_c, p_c):.4f}")
    return stem.with_suffix(".png")


if __name__ == "__main__":
    main()
