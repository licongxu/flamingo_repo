#!/usr/bin/env python3
"""Add hmfast sigma8(z=0) column to a Cobaya/getdist chain file.

Usage:
    python scripts/add_sigma8_to_chain.py chains/cnc_cosmo_arnaudB135_Om_lnAs/cnc_cosmo.1.txt

Uses GPU 0 with capped JAX memory when available; falls back to CPU.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

# Must be set before importing jax / hmfast.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

N_FLOAT = 8


def _read_chain(path: str) -> tuple[list[str], np.ndarray]:
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n")
    if not header.startswith("#"):
        raise ValueError(f"expected Cobaya header in {path}")
    names = header.lstrip("#").split()
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != len(names):
        raise ValueError(f"column count mismatch in {path}: {len(names)} names, {data.shape[1]} cols")
    return names, data


def _write_chain(path: str, names: list[str], data: np.ndarray) -> None:
    width = lambda col: max(7 + N_FLOAT, len(col))
    fmts = [f"%{width(col)}.{N_FLOAT}g" for col in names]
    header = "#" + " ".join(f"%{width(col)}s" % col for col in names)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        np.savetxt(fh, data, fmt=fmts)


def _compute_sigma8(names: list[str], data: np.ndarray, tau_reio: float = 0.0544) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from hmfast.cosmology import Cosmology

    idx = {n: i for i, n in enumerate(names)}
    for req in ("H0", "omega_cdm", "omega_b", "ln10_10A_s", "n_s"):
        if req not in idx:
            raise KeyError(f"chain missing required column {req!r}")

    H0 = jnp.asarray(data[:, idx["H0"]])
    omega_cdm = jnp.asarray(data[:, idx["omega_cdm"]])
    omega_b = jnp.asarray(data[:, idx["omega_b"]])
    ln1e10A_s = jnp.asarray(data[:, idx["ln10_10A_s"]])
    n_s = jnp.asarray(data[:, idx["n_s"]])

    cosmo0 = Cosmology(emulator_set="lcdm:v1")
    _ = cosmo0.sigma8(0.0)

    def _one(H0_i, oc_i, ob_i, ln_i, ns_i):
        c = cosmo0.update(
            H0=H0_i, omega_cdm=oc_i, omega_b=ob_i,
            ln1e10A_s=ln_i, n_s=ns_i, tau_reio=tau_reio,
        )
        return c.sigma8(0.0)

    batched = jax.jit(jax.vmap(_one))
    t0 = time.perf_counter()
    out = np.asarray(jax.block_until_ready(
        batched(H0, omega_cdm, omega_b, ln1e10A_s, n_s)
    )).reshape(-1)
    backend = jax.default_backend()
    print(f"computed sigma8 for {len(out)} samples in {time.perf_counter() - t0:.2f}s ({backend})")
    print(f"  sigma8 mean = {out.mean():.4f}, std = {out.std():.4f}")
    return out


def _patch_cobaya_yaml(chain_txt: str) -> None:
    """Register sigma8 in Cobaya yaml sidecars so getdist picks up the new column."""
    chain_dir = os.path.dirname(os.path.abspath(chain_txt))
    base = os.path.basename(chain_txt)
    if not base.endswith(".txt"):
        return
    root = base.rsplit(".", 1)[0].rsplit(".", 1)[0]
    for name in (f"{root}.updated.yaml", f"{root}.input.yaml"):
        ypath = os.path.join(chain_dir, name)
        if not os.path.isfile(ypath):
            continue
        with open(ypath, encoding="utf-8") as fh:
            text = fh.read()
        if "  sigma8:" in text:
            continue
        if "    derived: true\n  alpha_SZ:" in text:
            repl = "    derived: true\n  sigma8:\n    latex: \\sigma_8\n    derived: true\n  alpha_SZ:"
            old = "    derived: true\n  alpha_SZ:"
        elif "    latex: H_0\n  alpha_SZ:" in text:
            repl = "    latex: H_0\n  sigma8:\n    latex: \\sigma_8\n  alpha_SZ:"
            old = "    latex: H_0\n  alpha_SZ:"
        else:
            print(f"warning: could not patch {ypath} (unexpected layout)")
            continue
        with open(ypath, "w", encoding="utf-8") as fh:
            fh.write(text.replace(old, repl, 1))
        print(f"patched {ypath}")


def add_sigma8_column(path: str, *, backup: bool = True, force: bool = False) -> None:
    names, data = _read_chain(path)
    if "sigma8" in names:
        if not force:
            print(f"{path}: sigma8 column already present, skipping (use --force to overwrite)")
            return
        pos = names.index("sigma8")
        names.pop(pos)
        data = np.delete(data, pos, axis=1)

    sigma8 = _compute_sigma8(names, data)
    insert_at = names.index("H0") + 1
    names.insert(insert_at, "sigma8")
    data = np.insert(data, insert_at, sigma8, axis=1)

    if backup:
        bak = path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
            print(f"backup -> {bak}")

    _write_chain(path, names, data)
    _patch_cobaya_yaml(path)
    print(f"wrote {path} with sigma8 column after H0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chain_txt", help="path to chain.N.txt")
    parser.add_argument("--force", action="store_true", help="overwrite existing sigma8 column")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    path = os.path.abspath(args.chain_txt)
    if not os.path.isfile(path):
        sys.exit(f"not found: {path}")
    add_sigma8_column(path, backup=not args.no_backup, force=args.force)


if __name__ == "__main__":
    main()
