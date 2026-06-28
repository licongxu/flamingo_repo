"""Compare slow-built vs fast-rebuilt rows for one snapshot."""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

KEY_COLS = [
    "soap_index", "z", "x_Mpc", "y_Mpc", "z_Mpc", "r_comoving_Mpc", "shell_idx",
    "M_500c_Msun", "M_200c_Msun", "M_200m_Msun",
    "R_500c_Mpc", "R_200c_Mpc", "R_200m_Mpc",
    "Y_500c_Mpc2", "Y_500c_noAGN_Mpc2", "Y_5R500c_Mpc2", "Y_5R500c_noAGN_Mpc2",
    "theta_rot_rad", "phi_rot_rad", "x_rot_Mpc", "y_rot_Mpc", "z_rot_Mpc",
]
SORT_COLS = ["soap_index", "z"]


def load_snap(path: str, snap: int) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, comment="#", chunksize=500_000):
        sub = chunk[chunk["snap"] == snap]
        if not sub.empty:
            parts.append(sub)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def compare(slow: pd.DataFrame, fast: pd.DataFrame, snap: int, rtol: float, atol: float) -> int:
    print(f"snap {snap}: slow={len(slow):,} fast={len(fast):,} rows")
    if len(slow) != len(fast):
        print("  FAIL: row count mismatch")
        return 1

    cols = [c for c in KEY_COLS if c in slow.columns and c in fast.columns]
    slow = slow.sort_values(SORT_COLS).reset_index(drop=True)
    fast = fast.sort_values(SORT_COLS).reset_index(drop=True)

    for col in cols:
        a = slow[col].to_numpy()
        b = fast[col].to_numpy()
        if np.issubdtype(a.dtype, np.number):
            if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                diff = np.max(np.abs(a - b))
                print(f"  FAIL: {col} max abs diff = {diff}")
                return 1
        elif not np.array_equal(a, b):
            print(f"  FAIL: {col} values differ")
            return 1

    print(f"  OK: all {len(cols)} compared columns match (rtol={rtol}, atol={atol})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--slow", required=True, help="Reference CSV (slow SOAP-first build)")
    p.add_argument("--fast", required=True, help="Rebuilt CSV (lightcone-first)")
    p.add_argument("--snap", type=int, required=True)
    p.add_argument("--rtol", type=float, default=0.0)
    p.add_argument("--atol", type=float, default=0.0)
    args = p.parse_args()

    slow = load_snap(args.slow, args.snap)
    fast = load_snap(args.fast, args.snap)
    return compare(slow, fast, args.snap, args.rtol, args.atol)


if __name__ == "__main__":
    sys.exit(main())
