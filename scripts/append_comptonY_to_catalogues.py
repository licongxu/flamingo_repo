"""Append Compton-Y columns to L1_m9 feedback-variant halo catalogues.

The catalogues carry ``snap`` and ``soap_index`` per row. The Compton-Y aperture
values are simply the SOAP ``ComptonY`` datasets stored as-is in Mpc^2 (verified to
reproduce the fiducial catalogue's stored Y to float roundtrip). We therefore join
by (snap, soap_index) rather than rebuilding the lightcone.

Four columns are (re)written per catalogue:
    Y_500c_Mpc2        <- SO/500_crit/ComptonY
    Y_500c_noAGN_Mpc2  <- SO/500_crit/ComptonYWithoutRecentAGNHeating
    Y_5R500c_Mpc2      <- SO/5xR_500_crit/ComptonY
    Y_5R500c_noAGN_Mpc2<- SO/5xR_500_crit/ComptonYWithoutRecentAGNHeating

Write is in-place via a temp file + os.replace after a row-count check. Existing
non-Y columns are preserved byte-for-byte (text append, no re-serialization). If Y
columns already exist (partial fgas+2sigma), their trailing fields are replaced.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

CAT_DIR = Path("/rds/rds-lxu/flamingo/L1_m9/catalogues")

# CSV variant token -> SOAP directory under FLAMINGO/L1_m9/
VARIANTS = {
    "fgas+2sigma": "fgas+2sigma",
    "fgas-2sigma": "fgas-2sigma",
    "fgas-4sigma": "fgas-4sigma",
    "fgas-8sigma": "fgas-8sigma",
    "Jet": "Jet",
    "Jet_fgas-4sigma": "Jet_fgas-4sigma",
    "Mstar-1sigma": "Mstar-1sigma",
    "Mstar-1sigma_fgas-4sigma": "Mstar-1sigma_fgas-4sigma",
}

Y_COLS = ["Y_500c_Mpc2", "Y_500c_noAGN_Mpc2", "Y_5R500c_Mpc2", "Y_5R500c_noAGN_Mpc2"]
Y_PATHS = [
    "SO/500_crit/ComptonY",
    "SO/500_crit/ComptonYWithoutRecentAGNHeating",
    "SO/5xR_500_crit/ComptonY",
    "SO/5xR_500_crit/ComptonYWithoutRecentAGNHeating",
]


def csv_path(token: str) -> Path:
    return CAT_DIR / f"halo_catalogue_M500c_1e13_zlt3_{token}_yang26rot.csv"


def compute_Y(token: str, snap_arr: np.ndarray, idx_arr: np.ndarray, log) -> np.ndarray:
    """Return (n, 4) Y array aligned to input row order by streaming SOAP per snap."""
    import hdfstream

    root = hdfstream.open("cosma", "/")
    variant = VARIANTS[token]
    n = snap_arr.size
    Y = np.full((n, 4), np.nan, dtype=np.float64)
    for snap in np.unique(snap_arr):
        rows = np.nonzero(snap_arr == snap)[0]
        idx = idx_arr[rows]
        f = root[f"FLAMINGO/L1_m9/{variant}/SOAP-HBT/halo_properties_{int(snap):04d}.hdf5"]
        t0 = time.time()
        nbad = 0
        for j, dpath in enumerate(Y_PATHS):
            arr = np.asarray(f[dpath][:], dtype=np.float64)
            ok = (idx >= 0) & (idx < arr.size)
            nbad = int((~ok).sum())
            Y[rows[ok], j] = arr[idx[ok]]
        log(f"    {token} snap {int(snap):04d}: {rows.size:,} rows, {nbad} out-of-range, {time.time()-t0:.0f}s")
    return Y


def fmt(v: float) -> str:
    v = float(v)
    return "" if math.isnan(v) else repr(v)


def rewrite_with_Y(path: Path, Y: np.ndarray, log) -> Path:
    """Pure text-append of the 4 Y columns (existing columns preserved byte-for-byte).

    Only valid for catalogues that do NOT already carry Y columns and whose rows
    all have the same field count. A file with a partial/ragged Y append must be
    rebuilt via build_halo_lightcone_catalogue.py, not patched here (a uniform
    trailing-field strip corrupts the no-Y rows).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    header_added = f",{','.join(Y_COLS)}"
    i = 0  # data-row counter
    with path.open("r") as fin, tmp.open("w") as fout:
        header_seen = False
        for line in fin:
            if not header_seen and line.startswith("#"):
                fout.write(line)
                continue
            if not header_seen:
                header_seen = True
                if "Y_500c_Mpc2" in line:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(
                        "file already has Y columns; rebuild with "
                        "build_halo_lightcone_catalogue.py instead of appending"
                    )
                fout.write(line.rstrip("\n") + header_added + "\n")
                continue
            y = Y[i]
            fout.write(f"{line.rstrip(chr(10))},{fmt(y[0])},{fmt(y[1])},{fmt(y[2])},{fmt(y[3])}\n")
            i += 1
    if i != Y.shape[0]:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"row mismatch: wrote {i} data rows but Y has {Y.shape[0]}")
    log(f"  wrote {i:,} data rows -> {tmp.name}")
    return tmp


def process(token: str, log) -> None:
    path = csv_path(token)
    log(f"[{token}] reading snap+soap_index from {path.name} ({path.stat().st_size/1e9:.1f} GB)")
    t0 = time.time()
    meta = pd.read_csv(
        path, comment="#", usecols=["snap", "soap_index"],
        dtype={"snap": "int32", "soap_index": "int64"},
    )
    log(f"[{token}] {len(meta):,} rows read in {time.time()-t0:.0f}s; streaming ComptonY")
    Y = compute_Y(token, meta["snap"].to_numpy(), meta["soap_index"].to_numpy(), log)
    finite = int(np.isfinite(Y[:, 0]).sum())
    log(f"[{token}] Y_500c finite {finite:,}/{Y.shape[0]:,}; median_pos="
        f"{np.median(Y[Y[:,0]>0,0]):.3e}")
    tmp = rewrite_with_Y(path, Y, log)
    # validate row count vs original data rows
    if len(meta) != Y.shape[0]:
        raise RuntimeError("meta/Y length mismatch")
    os.replace(tmp, path)
    log(f"[{token}] DONE in {(time.time()-t0)/60:.1f} min -> {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", nargs="*", default=list(VARIANTS),
                   help="variant tokens to process (default: all 8)")
    p.add_argument("--logbook", type=Path,
                   default=Path("/scratch/scratch-lxu/flamingo_repo/runs/logbook.md"))
    args = p.parse_args()

    args.logbook.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"{msg}"
        print(line, flush=True)
        with args.logbook.open("a") as f:
            f.write(line + "\n")

    log(f"\n## append_comptonY_to_catalogues start; tokens={args.tokens}")
    for token in args.tokens:
        try:
            process(token, log)
        except Exception as e:  # noqa: BLE001  keep going to next catalogue
            log(f"[{token}] FAILED: {type(e).__name__}: {e}")
    log("## append_comptonY_to_catalogues finished")


if __name__ == "__main__":
    main()
