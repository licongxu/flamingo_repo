"""Theory-only covariance for B=1.35, unrotated z<0.35 tSZ bandpowers.

Same structure as ``compute_arnaudB1_full_cov.py`` but:
  * B_profile = 1.35
  * Limber integral capped at z_max = 0.35
  * f_sky_eff read from bandpower meta files (run bandpowers first)

Tags: fullsky, qgt5

Run:
    JAX_PLATFORMS=cuda python scripts/compute_arnaudB135_unrot_z0p35_full_cov.py
    JAX_PLATFORMS=cuda python scripts/compute_arnaudB135_unrot_z0p35_full_cov.py fullsky qgt5
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
OUT_DIR = _REPO / "data" / "theory_cov_arnaudB135_unrot_z0p35"
BP_DIR = _REPO / "data" / "bandpowers_arnaudB135_Y500c_unrot_z0p35"
NOISE = _REPO / "data/noise"

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
if "xla_gpu_persistent_cache_dir" in os.environ.get("XLA_FLAGS", ""):
    os.environ.pop("XLA_FLAGS")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.mass_definition import MassDefinition
from hmfast.halos.profiles import GNFWPressureProfile
from hmfast.tracers import tSZTracer
from hmfast.tracers.tsz_completeness import (
    build_snr_grid, conditional_An_undetected, compute_y0_parametric,
    load_sigma_y0_curve,
)

A_S_D3A = 2.099e-9
D3A = dict(
    H0=68.1,
    omega_b=0.022539,
    omega_cdm=0.118729,
    n_s=0.967,
    tau_reio=0.0544,
    ln1e10A_s=float(np.log(1e10 * A_S_D3A)),
)
B_PROFILE = 1.35
SIGMA_LNY = 0.173
ALPHA_A10 = 2.0 / 3.0 + 0.12 + 1.0 / 3.0

ELL_MIN = np.array(
    [9, 12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835],
    dtype=int,
)
ELL_MAX = np.array(
    [12, 16, 21, 27, 35, 46, 60, 78, 102, 133, 173, 224, 292, 380, 494, 642, 835, 1085],
    dtype=int,
)
ELL_EFF = np.array(
    [10.0, 13.5, 18.0, 23.5, 30.5, 40.0, 52.5, 68.5, 89.5,
     117.0, 152.5, 198.0, 257.5, 335.5, 436.5, 567.5, 738.0, 959.5]
)
N_BIN = 18
L_MAX = int(np.max(ELL_MAX - ELL_MIN))
ELL_INT = ELL_MIN[:, None] + np.arange(L_MAX)[None, :]
ELL_MASK = (ELL_INT < ELL_MAX[:, None]).astype(np.float64)
BINS = np.column_stack([ELL_MIN, ELL_MAX, ELL_EFF])

N_MASS, N_Z = 64, 96
LOG10_M_MIN, LOG10_M_MAX = 10.0, 15.5
Z_MIN, Z_MAX = 0.005, 0.35
N_ELL_SMOOTH = 30

TAG_META = {
    "fullsky": (None, BP_DIR / "meta_unrot_z0p35_fullsky_Y500c.npz"),
    "qgt5": (5.0, BP_DIR / "meta_unrot_z0p35_qgt5_Y500c.npz"),
}

NOISE_KW = dict(
    sigma_obj_file=str(NOISE / "sigma_dict_szifi.npy"),
    skyfr_file=str(NOISE / "skyfracs_szifi_cosmology.npy"),
)


def _fsky_eff(tag: str) -> float:
    meta_path = TAG_META[tag][1]
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Missing bandpower meta {meta_path}; run compute_bandpowers_arnaudB135_Y500c_unrot_z0p35.py first."
        )
    return float(np.load(meta_path)["fsky_eff"])


def _a10_gnfw_y0_sr(hm, m_grid, z_grid, B: float = B_PROFILE) -> tuple[float, float]:
    m = jnp.asarray(m_grid)
    z = jnp.asarray(z_grid)
    alpha_SZ = ALPHA_A10
    H0 = hm.cosmology.H0
    h = H0 / 100.0
    mass_def = MassDefinition(500, "critical")
    m_np = np.asarray(m)
    z_np = np.asarray(z)
    r500 = mass_def.r_delta(hm.cosmology, m_np[:, None], z_np[None, :])
    H = np.asarray(hm.cosmology.hubble_parameter(z_np))
    E = H / H0
    P500 = (
        1.65 * (h / 0.7) ** 2 * E[None, :] ** (8.0 / 3.0)
        * ((m_np[:, None] * h / B) / (0.7 * 3e14)) ** (2.0 / 3.0 + 0.12)
        * (0.7 / h) ** 1.5
    )
    y0_gnfw = (
        2.0 * (6.6524587e-25 / 510998.95) * 8.13 * P500
        * (r500 * 3.085677581e24) * 0.470502095
    )
    y0_at_a0 = np.asarray(compute_y0_parametric(hm, m, z, 0.0, alpha_SZ, B))
    A_SZ = float(np.nanmedian(np.log10(y0_gnfw / y0_at_a0)))
    return A_SZ, alpha_SZ


def _bin_1d_D(Dl_smooth, ell_smooth):
    log_e = np.log(ell_smooth)
    log_int = np.log(ELL_INT.astype(float))
    Dl_int = np.empty((N_BIN, L_MAX))
    for b in range(N_BIN):
        Dl_int[b] = np.interp(log_int[b], log_e, Dl_smooth)
    return (Dl_int * ELL_MASK).sum(axis=1) / ELL_MASK.sum(axis=1)


def _bin_2d_D(T_C_smooth, ell_smooth):
    log_e = np.log(ell_smooth)
    log_int = np.log(ELL_INT.astype(float))
    interp_a0 = np.empty((N_BIN, L_MAX, T_C_smooth.shape[1]))
    for j in range(T_C_smooth.shape[1]):
        interp_a0[..., j] = np.interp(log_int, log_e, T_C_smooth[:, j])
    interp_full = np.empty((N_BIN, L_MAX, N_BIN, L_MAX))
    for b1 in range(N_BIN):
        for li in range(L_MAX):
            interp_full[b1, li] = np.interp(
                log_int, log_e, interp_a0[b1, li, :]
            ).reshape(N_BIN, L_MAX)
    pre = ELL_INT * (ELL_INT + 1.0) / (2.0 * np.pi)
    Dprod = pre[:, :, None, None] * pre[None, None, :, :]
    T_D = interp_full * Dprod
    M2 = ELL_MASK[:, :, None, None] * ELL_MASK[None, None, :, :]
    return (T_D * M2).sum(axis=(1, 3)) / M2.sum(axis=(1, 3))


def _knox_gauss(Dl_b, fsky_eff: float) -> np.ndarray:
    cov = np.zeros((N_BIN, N_BIN))
    for i, (lo, hi, leff) in enumerate(BINS):
        delta_ell = float(hi - lo)
        cov[i, i] = 2.0 * Dl_b[i] ** 2 / ((2.0 * leff + 1.0) * delta_ell * fsky_eff)
    return cov


def _theory_Dl_b(hm, tsz, m_grid, z_grid, ell_smooth, q_cat: float | None,
                 a_sz: float, alpha_sz: float) -> np.ndarray:
    ell = jnp.asarray(ell_smooth)
    m = jnp.asarray(m_grid)
    z = jnp.asarray(z_grid)
    pref = ell_smooth * (ell_smooth + 1.0) / (2.0 * np.pi)
    s2 = SIGMA_LNY ** 2

    if q_cat is None:
        C_1h = np.asarray(hm.cl_1h(tsz, None, ell, m, z))
        C_2h = np.asarray(hm.cl_2h(tsz, None, ell, m, z))
        Dl_smooth = pref * (C_1h * np.exp(2.0 * s2) + C_2h * np.exp(s2))
    else:
        coeff, _ = load_sigma_y0_curve(**NOISE_KW)
        snr = build_snr_grid(
            hm, m, z, a_sz, alpha_sz, B_PROFILE, coeff=coeff, **NOISE_KW,
        )
        mask1 = conditional_An_undetected(
            snr, sigma_lnY=SIGMA_LNY, q_cat=float(q_cat), n_power=1,
        )
        mask2 = conditional_An_undetected(
            snr, sigma_lnY=SIGMA_LNY, q_cat=float(q_cat), n_power=2,
        )
        C_1h = np.asarray(hm.cl_1h_masked(tsz, None, ell, m, z, mask2))
        C_2h = np.asarray(hm.cl_2h_masked(tsz, None, ell, m, z, mask1))
        Dl_smooth = pref * (C_1h + C_2h)

    return _bin_1d_D(Dl_smooth, ell_smooth)


def _run(tag: str, q_cat: float | None, fsky_eff: float):
    print(f"\n=== {tag}  q_cat={q_cat}  f_sky_eff={fsky_eff:.4f}  z_max={Z_MAX}  B={B_PROFILE} ===", flush=True)

    cosmo = Cosmology(emulator_set="lcdm:v1").update(
        H0=D3A["H0"], omega_cdm=D3A["omega_cdm"], omega_b=D3A["omega_b"],
        ln1e10A_s=D3A["ln1e10A_s"], n_s=D3A["n_s"], tau_reio=D3A["tau_reio"],
    )
    hm = HaloModel(
        cosmology=cosmo,
        mass_definition=MassDefinition(500, "critical"),
        convert_masses=True,
    )
    prof = GNFWPressureProfile(B=B_PROFILE)
    tsz = tSZTracer(profile=prof)

    ell_smooth = np.geomspace(float(ELL_MIN[0]), float(ELL_MAX[-1]), N_ELL_SMOOTH)
    m_grid = np.geomspace(10 ** LOG10_M_MIN, 10 ** LOG10_M_MAX, N_MASS)
    z_grid = np.geomspace(Z_MIN, Z_MAX, N_Z)

    a_sz, alpha_sz = _a10_gnfw_y0_sr(hm, m_grid, z_grid)
    print(
        f"  A10 GNFW y0 SR: A_SZ={a_sz:.6f} alpha_SZ={alpha_sz:.4f}",
        flush=True,
    )

    t0 = time.time()
    Dl_b = _theory_Dl_b(hm, tsz, m_grid, z_grid, ell_smooth, q_cat, a_sz, alpha_sz)
    print(f"  theory D_ell done in {time.time()-t0:.1f}s", flush=True)

    C_G = _knox_gauss(Dl_b, fsky_eff)
    np.save(OUT_DIR / f"cov_gauss_unrot_z0p35_{tag}_Dl_yy_binned_18.npy", C_G)
    np.savetxt(
        OUT_DIR / f"Dl_yy_unrot_z0p35_{tag}_theory_binned_18.txt",
        np.column_stack([ELL_EFF, Dl_b]),
        fmt="%.6e",
    )

    if q_cat is None:
        T_C = np.asarray(
            hm.trispectrum_1h(
                tsz, None,
                jnp.asarray(ell_smooth), jnp.asarray(ell_smooth),
                jnp.asarray(m_grid), jnp.asarray(z_grid),
            )
        )
        T_C = T_C * float(np.exp(8.0 * SIGMA_LNY ** 2))
    else:
        coeff, _ = load_sigma_y0_curve(**NOISE_KW)
        snr = build_snr_grid(
            hm, jnp.asarray(m_grid), jnp.asarray(z_grid),
            a_sz, alpha_sz, B_PROFILE, coeff=coeff, **NOISE_KW,
        )
        mask4 = conditional_An_undetected(
            snr, sigma_lnY=SIGMA_LNY, q_cat=float(q_cat), n_power=4,
        )
        T_C = np.asarray(
            hm.trispectrum_1h_masked(
                tsz, None,
                jnp.asarray(ell_smooth), jnp.asarray(ell_smooth),
                jnp.asarray(m_grid), jnp.asarray(z_grid), mask4,
            )
        )

    T_binned = _bin_2d_D(T_C, ell_smooth)
    np.save(OUT_DIR / f"T_binned_unrot_z0p35_{tag}_Dl_yy_18.npy", T_binned)

    M = C_G + T_binned / (4.0 * np.pi * fsky_eff)
    np.save(OUT_DIR / f"cov_full_unrot_z0p35_{tag}_Dl_yy_binned_18.npy", M)
    np.savetxt(OUT_DIR / f"cov_full_unrot_z0p35_{tag}_Dl_yy_binned_18.csv", M, fmt="%.6e", delimiter=",")

    eigmin = float(np.linalg.eigvalsh((M + M.T) / 2.0).min())
    print(f"  min eig={eigmin:.3e}  wrote cov_full_unrot_z0p35_{tag}_*", flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args = sys.argv[1:] or list(TAG_META)
    for tag in args:
        if tag not in TAG_META:
            raise SystemExit(f"unknown tag {tag!r}; choices: {list(TAG_META)}")
        q_cat, _ = TAG_META[tag]
        _run(tag, q_cat, _fsky_eff(tag))
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
