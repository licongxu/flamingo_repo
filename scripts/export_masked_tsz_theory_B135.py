"""hmfast theory D_ell for masked tSZ PS (B=1.35 and B=1.0 A10).

B=1.35: matches q_from_mz catalogues (parametric y0 SNR, build_snr_grid).
B=1.0:  notebook 09 / arnaudB1 reference — flamingo D3A_COSMOLOGY, explicit
         Arnaud A10 GNFW (P0=8.403, ...), manual central-y0 SNR for completeness.

Outputs ``data/theory_masked_tsz_ps_B135.npz`` with both sets on NaMaster ell_eff.

Run:
    python scripts/export_masked_tsz_theory_B135.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from flamingo.catalogue import D3A_COSMOLOGY
from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel, convert_m_delta, MassDefinition
from hmfast.halos.profiles import GNFWPressureProfile
from hmfast.tracers import tSZTracer
from hmfast.tracers.tsz_completeness import (
    build_snr_grid,
    compute_theta500_arcmin,
    compute_y0_parametric,
    conditional_An_undetected,
    load_sigma_y0_curve,
    sigma_y0_from_theta,
)
from hmfast.utils import Const

OUT = _REPO / "data/theory_masked_tsz_ps_B135.npz"
NOISE = _REPO / "data/noise"
REF_ELL = _REPO / "data/bandpowers_L1_m9_feedback/masked_tsz_ps.npz"

LMAX = 6000
DELL = 30
Q_CUTS = [50.0, 20.0, 10.0, 5.0]
SIGMA_LNY = 0.173
ALPHA_A10 = 2.0 / 3.0 + 0.12 + 1.0 / 3.0

# Arnaud et al. (2010) A10 GNFW — same as notebooks/09 and 06.
A10_ARNAUD = dict(P0=8.403, c500=1.177, gamma=0.3081, alpha=1.0510, beta=5.4905)

A_S_D3A = 2.099e-9
D3A_LCDM = dict(
    H0=68.1,
    omega_b=0.022539,
    omega_cdm=0.118729,
    n_s=0.967,
    tau_reio=0.0544,
    ln1e10A_s=float(np.log(1e10 * A_S_D3A)),
)

NOISE_KW = dict(
    sigma_obj_file=str(NOISE / "sigma_dict_szifi.npy"),
    skyfr_file=str(NOISE / "skyfracs_szifi_cosmology.npy"),
)

_SIGMA_T_CM2 = 6.6524587e-25
_MEC2_EV = 510998.95
_I_SHAPE = 0.470502095


def calibrate_a_sz(hm: HaloModel, B: float) -> float:
    m = jnp.logspace(13.0, 15.5, 48)
    z = jnp.geomspace(0.01, 3.0, 48)
    mass_def = MassDefinition(500, "critical")
    m_np = np.asarray(m)
    z_np = np.asarray(z)
    H0 = hm.cosmology.H0
    h = H0 / 100.0
    r500 = mass_def.r_delta(hm.cosmology, m_np[:, None], z_np[None, :])
    H = np.asarray(hm.cosmology.hubble_parameter(z_np))
    E = H / H0
    P500 = (
        1.65 * (h / 0.7) ** 2 * E[None, :] ** (8.0 / 3.0)
        * ((m_np[:, None] * h / B) / (0.7 * 3e14)) ** (2.0 / 3.0 + 0.12)
        * (0.7 / h) ** 1.5
    )
    y0_gnfw = (
        2.0 * (_SIGMA_T_CM2 / _MEC2_EV) * 8.13 * P500
        * (r500 * 3.085677581e24) * _I_SHAPE
    )
    y0_at_a0 = np.asarray(compute_y0_parametric(hm, m, z, 0.0, ALPHA_A10, B))
    return float(np.nanmedian(np.log10(y0_gnfw / y0_at_a0)))


def snr_a10_manual(hm: HaloModel, m: jnp.ndarray, z: jnp.ndarray, B: float = 1.0) -> jnp.ndarray:
    """SNR(M,z) from A10 central y0 and szifi noise — notebook 09."""
    mdef500 = MassDefinition(500, "critical")
    c_old = hm.concentration.c_delta(hm, m, z)
    m500c = convert_m_delta(hm.cosmology, m, z, hm.mass_definition, mdef500, c_old=c_old)
    r500c = mdef500.r_delta(hm.cosmology, m500c, z)
    h = hm.cosmology.H0 / 100.0
    E_z = jnp.atleast_1d(hm.cosmology.hubble_parameter(z))[None, :] / hm.cosmology.H0
    P_500c = (
        1.65 * (h / 0.7) ** 2 * E_z ** (8.0 / 3.0)
        * ((m500c * h / B) / (0.7 * 3.0e14)) ** (2.0 / 3.0 + 0.12)
        * (0.7 / h) ** 1.5
    )
    y0 = (
        2.0 * (_SIGMA_T_CM2 / _MEC2_EV) * A10_ARNAUD["P0"] * P_500c
        * (r500c * Const._Mpc_over_m_ * 100.0) * _I_SHAPE
    )
    coeff, _ = load_sigma_y0_curve(**NOISE_KW)
    theta = compute_theta500_arcmin(hm, m, z, B)
    return y0 / sigma_y0_from_theta(theta, coeff)


def masked_dl(
    hm: HaloModel,
    tsz: tSZTracer,
    snr: jnp.ndarray,
    ell_fine: np.ndarray,
    m: jnp.ndarray,
    z: jnp.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ell_th = jnp.asarray(ell_fine)
    pref = ell_fine * (ell_fine + 1.0) / (2.0 * np.pi)
    cl1 = np.asarray(hm.cl_1h(tsz, tsz, l=ell_th, m=m, z=z))
    cl2 = np.asarray(hm.cl_2h(tsz, tsz, l=ell_th, m=m, z=z))
    dl_full = pref * (cl1 + cl2)

    norm1 = np.exp(0.5 * (2.0 * SIGMA_LNY) ** 2)
    norm2 = np.exp(0.5 * (1.0 * SIGMA_LNY) ** 2)
    dl_masked_list: list[np.ndarray] = []
    for qc in Q_CUTS:
        w1h = conditional_An_undetected(snr, SIGMA_LNY, qc, n_power=2) / norm1
        w2h = conditional_An_undetected(snr, SIGMA_LNY, qc, n_power=1) / norm2
        c1 = np.asarray(hm.cl_1h_masked(tsz, tsz, l=ell_th, m=m, z=z, mask_mz=w1h))
        c2 = np.asarray(hm.cl_2h_masked(tsz, tsz, l=ell_th, m=m, z=z, mask_mz=w2h))
        dl_masked_list.append(pref * (c1 + c2))
    return dl_full, np.stack(dl_masked_list, axis=0)


def theory_b1_a10(
    ell_fine: np.ndarray,
    m: jnp.ndarray,
    z: jnp.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hm = HaloModel(cosmology=D3A_COSMOLOGY)
    tsz = tSZTracer(profile=GNFWPressureProfile(**A10_ARNAUD, B=1.0))
    snr = snr_a10_manual(hm, m, z, B=1.0)
    print("  B=1.0 A10: flamingo D3A + nb09 manual SNR", flush=True)
    return masked_dl(hm, tsz, snr, ell_fine, m, z)


def theory_b135(
    ell_fine: np.ndarray,
    m: jnp.ndarray,
    z: jnp.ndarray,
    coeff: jnp.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    hm = HaloModel(cosmology=Cosmology(**D3A_LCDM))
    tsz = tSZTracer(profile=GNFWPressureProfile(B=1.35))
    a_sz = calibrate_a_sz(hm, 1.35)
    snr = build_snr_grid(hm, m, z, a_sz, ALPHA_A10, 1.35, coeff=coeff, **NOISE_KW)
    print(f"  B=1.35: A_SZ={a_sz:.6f}  build_snr_grid (q_from_mz)", flush=True)
    dl_full, dl_mask = masked_dl(hm, tsz, snr, ell_fine, m, z)
    return a_sz, dl_full, dl_mask


def interp_dl(ell_out: np.ndarray, ell_fine: np.ndarray, dl_fine: np.ndarray) -> np.ndarray:
    return np.exp(
        np.interp(
            np.log(ell_out),
            np.log(ell_fine),
            np.log(np.maximum(dl_fine, 1e-60)),
        )
    )


def print_summary(label: str, ell_fine: np.ndarray, dl_full: np.ndarray, dl_mask: np.ndarray) -> None:
    print(f"{label}:", flush=True)
    print(f"  full sky D_ell@1000 = {float(np.interp(1000, ell_fine, dl_full)):.4e}", flush=True)
    for i, qc in enumerate(Q_CUTS):
        print(
            f"  q>{qc:g} D_ell@3000 = {float(np.interp(3000, ell_fine, dl_mask[i])):.4e}",
            flush=True,
        )


def main() -> None:
    if REF_ELL.exists():
        ellb = np.load(REF_ELL)["ellb"]
    else:
        import pymaster as nmt

        ellb = nmt.NmtBin.from_lmax_linear(LMAX, DELL).get_effective_ells()

    ell_fine = np.geomspace(10.0, float(LMAX), 80)
    m = jnp.logspace(11.0, 15.5, 60)
    z = jnp.geomspace(0.001, 3.0, 60)
    coeff, _ = load_sigma_y0_curve(**NOISE_KW)

    dl_full_1, dl_mask_1 = theory_b1_a10(ell_fine, m, z)
    print_summary("B=1.0 A10", ell_fine, dl_full_1, dl_mask_1)

    a_sz_135, dl_full_135, dl_mask_135 = theory_b135(ell_fine, m, z, coeff)
    print_summary("B=1.35", ell_fine, dl_full_135, dl_mask_135)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        ellb=ellb,
        ell_fine=ell_fine,
        q_cuts=np.array(Q_CUTS),
        sigma_lnY=SIGMA_LNY,
        alpha_SZ=ALPHA_A10,
        lmax=LMAX,
        delta_ell=DELL,
        dl_fullsky=interp_dl(ellb, ell_fine, dl_full_135),
        dl_masked=np.stack(
            [interp_dl(ellb, ell_fine, dl_mask_135[i]) for i in range(len(Q_CUTS))], axis=0
        ),
        A_SZ=a_sz_135,
        B=1.35,
        dl_fullsky_b135=interp_dl(ellb, ell_fine, dl_full_135),
        dl_masked_b135=np.stack(
            [interp_dl(ellb, ell_fine, dl_mask_135[i]) for i in range(len(Q_CUTS))], axis=0
        ),
        A_SZ_b135=a_sz_135,
        dl_fullsky_b1=interp_dl(ellb, ell_fine, dl_full_1),
        dl_masked_b1=np.stack(
            [interp_dl(ellb, ell_fine, dl_mask_1[i]) for i in range(len(Q_CUTS))], axis=0
        ),
        B_b1=1.0,
        a10_p0=A10_ARNAUD["P0"],
        cosmology_b1="flamingo.D3A_COSMOLOGY",
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
