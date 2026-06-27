"""Fix q_from_Y_5R500c (and theta500_arcmin) on the yang26 SOAP catalogue.

The appended q_from_Y_5R500c used sigma_y0 evaluated at the Planck-MMF
scaling-relation theta500 (hydrostatic bias B=1.41). For Y-derived Arnaud
A10-B=1 y0 that should be theta500 = R_500c / D_A(z) with B=1, matching
build_y0q_catalogue.py / halo_catalogue_*_y0q_arnaudB1.csv.

This script:
  * backs up the SR theta500 into theta500_sr_arcmin (new column)
  * sets theta500_arcmin from the verified y0q catalogue
  * recomputes q_from_Y_5R500c = y0_from_Y_5R500c / sigma_y0_B1
  * adds sigma_y0_B1 from the y0q catalogue

SR columns q, y0, sigma_y0 are left unchanged (self-consistent Planck-MMF SNR).

Run:
    python scripts/fix_yang26_q_from_Y.py
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
YANG26 = _REPO / "data/hydro_L2p8m9/catalogue/halo_catalogue_M500c_5e13_zlt3_soap_snr_d3a_yang26rot.csv"
Y0Q = _REPO / "data/hydro_L2p8m9/catalogue/halo_catalogue_M500c_5e13_zlt3_y0q_arnaudB1.csv"

HEADER_FIX = (
    "# q_from_Y_5R500c corrected: sigma_y0_B1 at theta500_arcmin = R_500c/D_A(z) "
    "(D3A, B=1), szifi immf6 sky-averaged; matches y0q_arnaudB1.csv.\n"
    "# theta500_sr_arcmin preserves the original Planck-MMF SR theta (B=1.41).\n"
    "# Columns q, y0, sigma_y0 remain the SR-model SNR (A=-4.23988, alpha=1.12, B=1.41).\n"
)


def read_csv_with_header(path: Path):
    hdr: list[str] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                hdr.append(line.rstrip("\n"))
            else:
                break
    df = pd.read_csv(path, comment="#")
    return hdr, df


def main() -> None:
    if not YANG26.is_file():
        raise FileNotFoundError(YANG26)
    if not Y0Q.is_file():
        raise FileNotFoundError(Y0Q)

    bak = YANG26.with_suffix(YANG26.suffix + ".bak_qfix")
    if not bak.exists():
        shutil.copy2(YANG26, bak)
        print(f"backup -> {bak}")

    t0 = time.time()
    hdr, df = read_csv_with_header(YANG26)
    y0q = pd.read_csv(Y0Q, comment="#")

    n_before = int((df["q_from_Y_5R500c"] > 5).sum()) if "q_from_Y_5R500c" in df.columns else -1

    if len(y0q) != len(df):
        raise RuntimeError(f"row count mismatch: yang26={len(df)} y0q={len(y0q)}")
    keys = ["snap", "z", "M_500c_Msun", "R_500c_Mpc", "Y_5R500c_Mpc2"]
    for k in keys:
        if not np.array_equal(df[k].to_numpy(), y0q[k].to_numpy()):
            raise RuntimeError(f"row alignment failed on column {k!r}")

    if "theta500_sr_arcmin" not in df.columns:
        df["theta500_sr_arcmin"] = df["theta500_arcmin"].to_numpy(float)

    df["theta500_arcmin"] = y0q["theta500_arcmin"].to_numpy(float)
    df["sigma_y0_B1"] = y0q["sigma_y0"].to_numpy(float)
    df["q_from_Y_5R500c"] = y0q["q"].to_numpy(float)

    n_after = int((df["q_from_Y_5R500c"] > 5).sum())
    print(f"rows={len(df):,}  q_from_Y>5: {n_before} -> {n_after}")

    # Drop duplicate header lines from prior fix attempts if re-run
    skip_prefixes = (
        "# q_from_Y_5R500c corrected:",
        "# theta500_sr_arcmin preserves",
        "# Columns q, y0, sigma_y0 remain",
    )
    hdr = [line for line in hdr if not any(line.startswith(p) for p in skip_prefixes)]

    with open(YANG26, "w") as fh:
        for line in hdr:
            fh.write(line + "\n")
        fh.write(HEADER_FIX)
        df.to_csv(fh, index=False, float_format="%.6e")

    print(f"wrote {YANG26}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
