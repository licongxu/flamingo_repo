"""Build M500c>5e13 L1_m9 catalogues with q_from_mz (Route B, GPU, B=1.35).

Streams the existing M>=1e13 catalogues, keeps only M_500c > 5e13 Msun,
computes parametric A10 GNFW SNR with lognormal intrinsic scatter via hmfast/JAX
on GPU, and writes compact derived catalogues to RDS:

    halo_catalogue_M500c_5e13_zlt3_{variant}_yang26rot_qfrommz.csv

Run:
    python scripts/add_q_from_mz_L1_m9_catalogues.py
    python scripts/add_q_from_mz_L1_m9_catalogues.py --variant Jet
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.30")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)

from hmfast.cosmology import Cosmology
from hmfast.halos import HaloModel
from hmfast.halos.mass_definition import MassDefinition
from hmfast.tracers.tsz_completeness import (
    compute_theta500_arcmin,
    compute_y0_parametric,
    load_sigma_y0_curve,
    sigma_y0_from_theta,
)

CAT_DIR = Path("/rds/rds-lxu/flamingo/L1_m9/catalogues")
NOISE_DIR = _REPO / "data/noise"
CHUNK = 1_000_000
M_MIN = 5.0e13
SCATTER_KEY = jax.random.PRNGKey(20260630)

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

VARIANTS = [
    "L1_m9",
    "fgas+2sigma",
    "fgas-2sigma",
    "fgas-4sigma",
    "fgas-8sigma",
    "Mstar-1sigma",
    "Mstar-1sigma_fgas-4sigma",
    "Jet",
    "Jet_fgas-4sigma",
]


def src_path(variant: str) -> Path:
    return CAT_DIR / f"halo_catalogue_M500c_1e13_zlt3_{variant}_yang26rot.csv"


def out_path(variant: str) -> Path:
    return CAT_DIR / f"halo_catalogue_M500c_5e13_zlt3_{variant}_yang26rot_qfrommz.csv"


def calibrate_a_sz(hm: HaloModel, *, B: float = B_PROFILE) -> float:
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
        2.0 * (6.6524587e-25 / 510998.95) * 8.13 * P500
        * (r500 * 3.085677581e24) * 0.470502095
    )
    y0_at_a0 = np.asarray(compute_y0_parametric(hm, m, z, 0.0, ALPHA_A10, B))
    return float(np.nanmedian(np.log10(y0_gnfw / y0_at_a0)))


def make_q_fn(hm: HaloModel, a_sz: float, coeff: jnp.ndarray):
    a_sz_f = float(a_sz)

    def _q_scalar(m: jnp.ndarray, z: jnp.ndarray, soap: jnp.ndarray) -> jnp.ndarray:
        key = jax.random.fold_in(SCATTER_KEY, soap.astype(jnp.uint32))
        ln_scatter = jax.random.normal(key) * SIGMA_LNY
        y0_mean = compute_y0_parametric(hm, m, z, a_sz_f, ALPHA_A10, B_PROFILE)
        theta = compute_theta500_arcmin(hm, m, z, B_PROFILE)
        sigma = sigma_y0_from_theta(theta, coeff)
        return (y0_mean * jnp.exp(ln_scatter) / sigma).reshape(())

    q_batch = jax.jit(jax.vmap(_q_scalar))

    def q_from_mz(m: np.ndarray, z: np.ndarray, soap_index: np.ndarray) -> np.ndarray:
        q = q_batch(
            jnp.asarray(m, dtype=jnp.float64),
            jnp.asarray(z, dtype=jnp.float64),
            jnp.asarray(soap_index, dtype=jnp.uint32),
        )
        return np.asarray(jax.device_get(q))

    return q_from_mz


def process_catalogue(
    variant: str,
    hm: HaloModel,
    a_sz: float,
    coeff: jnp.ndarray,
    *,
    force: bool = False,
) -> tuple[int, int, int]:
    src = src_path(variant)
    dst = out_path(variant)
    if not src.exists():
        print(f"skip {variant}: missing {src}")
        return 0, 0, 0
    if dst.exists() and not force:
        print(f"skip {variant}: exists {dst.name}")
        n_elig = n_q5 = 0
        for chunk in pd.read_csv(dst, comment="#", usecols=["q_from_mz"], chunksize=CHUNK):
            n_elig += len(chunk)
            n_q5 += int((chunk["q_from_mz"].to_numpy() > 5.0).sum())
        return n_elig, n_elig, n_q5

    q_fn = make_q_fn(hm, a_sz, coeff)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    t0 = time.time()
    n_scanned = n_kept = n_q5 = 0
    wrote_header = False

    with tmp.open("w") as out:
        out.write(
            f"# L1_m9 {variant}: M_500c > {M_MIN:.0e} Msun subset with q_from_mz.\n"
            f"# Source: {src.name}\n"
            f"# q_from_mz = y0(parametric A10, B={B_PROFILE}) * exp(N(0,sigma_lnY^2)) / sigma_y0(immf6)\n"
            f"# sigma_lnY={SIGMA_LNY}; A_SZ={a_sz:.6f}; alpha_SZ={ALPHA_A10}; soap_index-scattered.\n"
        )

        for ichunk, chunk in enumerate(pd.read_csv(src, comment="#", chunksize=CHUNK)):
            n_scanned += len(chunk)
            chunk = chunk[chunk["M_500c_Msun"] > M_MIN].copy()
            if chunk.empty:
                continue
            q = q_fn(
                chunk["M_500c_Msun"].to_numpy(dtype=np.float64),
                chunk["z"].to_numpy(dtype=np.float64),
                chunk["soap_index"].to_numpy(dtype=np.uint32),
            )
            chunk["q_from_mz"] = q
            chunk.to_csv(out, index=False, header=not wrote_header, mode="a")
            wrote_header = True
            n_kept += len(chunk)
            n_q5 += int((q > 5.0).sum())
            if (ichunk + 1) % 5 == 0:
                print(
                    f"  {variant}: scanned {n_scanned:,}, kept {n_kept:,} "
                    f"({time.time()-t0:.0f}s)",
                    flush=True,
                )

    tmp.replace(dst)
    print(
        f"{variant}: kept {n_kept:,}/{n_scanned:,}, N(q>5)={n_q5:,} "
        f"-> {dst.name} ({time.time()-t0:.0f}s)",
        flush=True,
    )
    return n_scanned, n_kept, n_q5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", action="append")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}", flush=True)
    cosmo = Cosmology(**D3A)
    hm = HaloModel(cosmology=cosmo)
    a_sz = calibrate_a_sz(hm)
    coeff_np, _ = load_sigma_y0_curve(
        sigma_obj_file=str(NOISE_DIR / "sigma_dict_szifi.npy"),
        skyfr_file=str(NOISE_DIR / "skyfracs_szifi_cosmology.npy"),
        filter_name="immf6",
    )
    coeff = jnp.asarray(coeff_np)
    print(
        f"A_SZ={a_sz:.6f}  B={B_PROFILE}  sigma_lnY={SIGMA_LNY}  M_min={M_MIN:.0e}",
        flush=True,
    )

    summary: list[tuple[str, int, int, int]] = []
    for variant in args.variant or VARIANTS:
        print(f"=== {variant} ===", flush=True)
        summary.append((variant, *process_catalogue(variant, hm, a_sz, coeff, force=args.force)))

    print(f"\n=== summary: M_500c > {M_MIN:.0e}, N(q>5) ===")
    for variant, _nscan, n_kept, n_q5 in summary:
        print(f"  {variant:28s}  kept={n_kept:8,}  N(q>5)={n_q5:6,}")


if __name__ == "__main__":
    main()
