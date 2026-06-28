"""Fast consistency check: slow-built snap rows vs lightcone-first selection logic.

Does NOT rescan the lightcone. For a snap already in the slow CSV:
  1. load all rows for that snap;
  2. batch-fetch SOAP M500c for unique soap_index values only;
  3. verify mass cut + SOAP values match the CSV;
  4. recompute yang26 rotation and compare to stored columns.

If these pass, every slow row would also be kept by the lightcone-first path
(same join key, same z filter, same M500c cut, same post-processing).
"""
from __future__ import annotations

import argparse
import sys
import time

import hdfstream
import numpy as np
import pandas as pd

from build_halo_lightcone_catalogue import (
    MASS_MIN_DEFAULT,
    SOAP_FIELDS,
    Z_MAX,
    Z_MIN,
    _angles_for,
    _angles_from_xyz,
    _physical_radius_factor,
    _read_by_index,
    _rotate_positions,
    _sim_base,
)


def load_snap(path: str, snap: int) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, comment="#", chunksize=500_000):
        sub = chunk[chunk["snap"] == snap]
        if sub.empty:
            continue
        parts.append(sub)
    if not parts:
        raise SystemExit(f"no rows for snap {snap} in {path}")
    return pd.concat(parts, ignore_index=True)


def check_snap(
    df: pd.DataFrame,
    snap: int,
    parent: str | None,
    variant: str,
    rtol: float,
    atol: float,
) -> int:
    t0 = time.time()
    n = len(df)
    print(f"snap {snap}: {n:,} slow-built rows", flush=True)

    z = df["z"].to_numpy()
    if (z < Z_MIN).any() or (z > Z_MAX).any():
        bad = int(((z < Z_MIN) | (z > Z_MAX)).sum())
        print(f"  FAIL: {bad} rows outside {Z_MIN} <= z <= {Z_MAX}")
        return 1

    if (df["M_500c_Msun"] < MASS_MIN_DEFAULT).any():
        bad = int((df["M_500c_Msun"] < MASS_MIN_DEFAULT).sum())
        print(f"  FAIL: {bad} rows below M500c cut")
        return 1

    root = hdfstream.open("cosma", "/")
    base = _sim_base(parent, variant)
    soap = root[f"{base}/SOAP-HBT/halo_properties_{snap:04d}.hdf5"]
    soap_datasets = {name: soap[path] for name, path in SOAP_FIELDS.items()}
    radius_factor = _physical_radius_factor(soap_datasets["R_500c_Mpc"])
    angles = _angles_for(parent)

    idx = df["soap_index"].to_numpy(dtype=np.int64)
    unique = np.unique(idx)
    print(f"  unique soap_index: {unique.size:,} (batched SOAP read)", flush=True)

    m500 = _read_by_index(soap_datasets["M_500c_Msun"], idx).astype(np.float64) * 1.0e10
    if not np.allclose(df["M_500c_Msun"].to_numpy(), m500, rtol=rtol, atol=atol):
        diff = np.max(np.abs(df["M_500c_Msun"].to_numpy() - m500))
        print(f"  FAIL: M_500c_Msun mismatch vs SOAP indexed read (max diff {diff})")
        return 1
    if (m500 < MASS_MIN_DEFAULT).any():
        print("  FAIL: SOAP M500c below cut for some rows")
        return 1

    for col in ("M_200c_Msun", "M_200m_Msun"):
        got = _read_by_index(soap_datasets[col], idx).astype(np.float64) * 1.0e10
        if not np.allclose(df[col].to_numpy(), got, rtol=rtol, atol=atol):
            print(f"  FAIL: {col} mismatch vs SOAP")
            return 1

    for col in ("R_500c_Mpc", "R_200c_Mpc", "R_200m_Mpc"):
        got = _read_by_index(soap_datasets[col], idx).astype(np.float64) * radius_factor
        if not np.allclose(df[col].to_numpy(), got, rtol=rtol, atol=atol):
            print(f"  FAIL: {col} mismatch vs SOAP")
            return 1

    xyz = df[["x_Mpc", "y_Mpc", "z_Mpc"]].to_numpy(dtype=np.float64)
    r, theta_nat, phi_nat = _angles_from_xyz(xyz)
    shell_max = angles.shape[1] - 1
    shell_idx = np.minimum(np.floor(z / 0.05).astype(np.int16), shell_max)
    theta_rot, phi_rot, xyz_rot = _rotate_positions(r, theta_nat, phi_nat, shell_idx, angles)

    checks = {
        "shell_idx": shell_idx,
        "theta_nat_rad": theta_nat,
        "phi_nat_rad": phi_nat,
        "theta_rot_rad": theta_rot,
        "phi_rot_rad": phi_rot,
        "x_rot_Mpc": xyz_rot[:, 0],
        "y_rot_Mpc": xyz_rot[:, 1],
        "z_rot_Mpc": xyz_rot[:, 2],
    }
    for col, got in checks.items():
        if not np.allclose(df[col].to_numpy(), got, rtol=rtol, atol=atol):
            diff = np.max(np.abs(df[col].to_numpy() - got))
            print(f"  FAIL: recomputed {col} mismatch (max diff {diff})")
            return 1

    print(f"  OK: slow rows satisfy lightcone-first criteria + same post-processing ({time.time()-t0:.1f}s)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True)
    p.add_argument("--snaps", type=int, nargs="+", required=True)
    p.add_argument("--parent", default=None)
    p.add_argument("--variant", default="L2p8_m9")
    p.add_argument("--rtol", type=float, default=1e-9)
    p.add_argument("--atol", type=float, default=1e-9)
    args = p.parse_args()

    rc = 0
    for snap in args.snaps:
        df = load_snap(args.csv, snap)
        rc |= check_snap(df, snap, args.parent, args.variant, args.rtol, args.atol)
    return rc


if __name__ == "__main__":
    sys.exit(main())
