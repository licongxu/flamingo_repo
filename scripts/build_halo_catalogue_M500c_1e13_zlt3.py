"""Build the requested FLAMINGO M500c>=1e13, z<3 halo-lightcone CSV.

This implementation avoids slow random hdfstream reads by processing one snapshot
at a time:
  1. scan the SOAP snapshot contiguously and cache only halos with M500c>=1e13;
  2. scan that snapshot's halo lightcone and locally join on SOAPIndex;
  3. append original and yang26-rotated positions plus SOAP Y_500c/Y_5R500c.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import healpy as hp
import hdfstream
import numpy as np
import pandas as pd

from build_y_map import _angles_for


def _sim_base(parent: str | None, variant: str) -> str:
    if parent is None:
        return f"FLAMINGO/{variant}/{variant}"
    return f"FLAMINGO/{parent}/{variant}"


DEFAULT_VARIANT = "L2p8_m9"
DEFAULT_OUT = Path(
    "/scratch/scratch-lxu/flamingo_repo/data/hydro_L2p8m9/catalogue/"
    "halo_catalogue_M500c_1e13_zlt3_soap_hdfstream_yang26rot.csv"
)
MASS_MIN = 1.0e13
Z_MIN = 0.0
Z_MAX = 3.0

SOAP_FIELDS = {
    "M_500c_Msun": "SO/500_crit/TotalMass",
    "M_200c_Msun": "SO/200_crit/TotalMass",
    "M_200m_Msun": "SO/200_mean/TotalMass",
    "R_500c_Mpc": "SO/500_crit/SORadius",
    "R_200c_Mpc": "SO/200_crit/SORadius",
    "R_200m_Mpc": "SO/200_mean/SORadius",
    "Y_500c_Mpc2": "SO/500_crit/ComptonY",
    "Y_500c_noAGN_Mpc2": "SO/500_crit/ComptonYWithoutRecentAGNHeating",
    "Y_5R500c_Mpc2": "SO/5xR_500_crit/ComptonY",
    "Y_5R500c_noAGN_Mpc2": "SO/5xR_500_crit/ComptonYWithoutRecentAGNHeating",
}
MASS_COLUMNS = {"M_500c_Msun", "M_200c_Msun", "M_200m_Msun"}
RADIUS_COLUMNS = {"R_500c_Mpc", "R_200c_Mpc", "R_200m_Mpc"}


def physical_radius_factor(dataset) -> float:
    attrs = dict(dataset.attrs)
    cgs = float(np.asarray(attrs["Conversion factor to CGS (not including cosmological corrections)"])[0])
    pcgs = float(np.asarray(attrs["Conversion factor to physical CGS (including cosmological corrections)"])[0])
    return pcgs / cgs


def convert_values(col: str, values: np.ndarray, radius_factor: float) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    if col in MASS_COLUMNS:
        return values * 1.0e10
    if col in RADIUS_COLUMNS:
        return values * radius_factor
    return values


def angles_from_xyz(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.sqrt(np.sum(xyz * xyz, axis=1))
    theta = np.arccos(np.clip(xyz[:, 2] / r, -1.0, 1.0))
    phi = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2.0 * np.pi)
    return r, theta, phi


def rotate_positions(
    r: np.ndarray,
    theta_nat: np.ndarray,
    phi_nat: np.ndarray,
    shell_idx: np.ndarray,
    angles: np.ndarray,
):
    theta_rot = np.empty_like(theta_nat)
    phi_rot = np.empty_like(phi_nat)
    for shell in np.unique(shell_idx):
        if shell < 0 or shell >= angles.shape[1]:
            raise ValueError(f"No yang26 angle for shell_idx={shell}")
        sel = shell_idx == shell
        theta_angle, phi_angle = angles[:, shell]
        rot = hp.Rotator(rot=[phi_angle * 180.0 / np.pi, theta_angle * 180.0 / np.pi], inv=True)
        theta_rot[sel], phi_rot[sel] = rot(theta_nat[sel], phi_nat[sel])
    xyz_rot = np.column_stack([
        r * np.sin(theta_rot) * np.cos(phi_rot),
        r * np.sin(theta_rot) * np.sin(phi_rot),
        r * np.cos(theta_rot),
    ])
    return theta_rot, phi_rot, xyz_rot


def load_selected_soap(soap, mass_min: float, chunk_size: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    datasets = {col: soap[path] for col, path in SOAP_FIELDS.items()}
    radius_factor = physical_radius_factor(datasets["R_500c_Mpc"])
    n = datasets["M_500c_Msun"].shape[0]
    kept_idx = []
    kept_cols = {col: [] for col in SOAP_FIELDS}
    t0 = time.time()

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        m500 = convert_values("M_500c_Msun", np.asarray(datasets["M_500c_Msun"][start:stop]), radius_factor)
        keep = m500 >= mass_min
        if keep.any():
            kept_idx.append(np.nonzero(keep)[0].astype(np.int64) + start)
            kept_cols["M_500c_Msun"].append(m500[keep])
            for col, dataset in datasets.items():
                if col == "M_500c_Msun":
                    continue
                kept_cols[col].append(convert_values(col, np.asarray(dataset[start:stop])[keep], radius_factor))
        if start and start % (10 * chunk_size) == 0:
            nkeep = sum(x.size for x in kept_idx)
            print(f"    SOAP scan {start:,}/{n:,}; kept {nkeep:,}; {((time.time()-t0)/60):.1f} min", flush=True)

    if not kept_idx:
        return np.array([], dtype=np.int64), {col: np.array([], dtype=np.float64) for col in SOAP_FIELDS}
    idx = np.concatenate(kept_idx)
    cols = {col: np.concatenate(parts) if parts else np.array([], dtype=np.float64) for col, parts in kept_cols.items()}
    return idx, cols


def write_header(path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tag = f"{args.parent}/{args.variant}" if args.parent else args.variant
    with path.open("w") as f:
        f.write(f"# FLAMINGO {tag} halo lightcone0 catalogue from hdfstream.\n")
        f.write("# Selection: M_500c >= 1e13 Msun and 0 <= z < 3.0.\n")
        f.write("# Joined to SOAP-HBT by InputHalos/SOAPIndex.\n")
        f.write("# Masses are Msun; radii and positions are Mpc; Compton-Y aperture values are Mpc^2.\n")
        f.write("# Rotated coordinates use the yang26 per-shell frame from the FLAMINGO map products.\n")
        f.write(f"# soap_chunk_size={args.soap_chunk_size}; lightcone_chunk_size={args.lightcone_chunk_size}\n")


def append_lightcone_matches(
    root,
    lightcone: str,
    snap: int,
    selected_idx: np.ndarray,
    soap_cols: dict[str, np.ndarray],
    args: argparse.Namespace,
    wrote_header: bool,
):
    lc = root[f"{lightcone}/lightcone_halos_{snap:04d}.hdf5"]
    n = lc["Lightcone/Redshift"].shape[0]
    rows = 0
    for start in range(0, n, args.lightcone_chunk_size):
        stop = min(start + args.lightcone_chunk_size, n)
        z = np.asarray(lc["Lightcone/Redshift"][start:stop], dtype=np.float64)
        zkeep = (z >= Z_MIN) & (z < Z_MAX)
        if not zkeep.any():
            continue
        soap_index = np.asarray(lc["InputHalos/SOAPIndex"][start:stop], dtype=np.int64)[zkeep]
        loc = np.searchsorted(selected_idx, soap_index)
        matched = loc < selected_idx.size
        if matched.any():
            matched_idx = np.nonzero(matched)[0]
            matched[matched_idx] = selected_idx[loc[matched_idx]] == soap_index[matched_idx]
        if not matched.any():
            continue
        loc = loc[matched]
        z = z[zkeep][matched]
        soap_index = soap_index[matched]
        xyz = np.asarray(lc["Lightcone/HaloCentre"][start:stop], dtype=np.float64)[zkeep][matched]

        if args.max_rows is not None:
            remaining = args.max_rows - args._rows_written
            if remaining <= 0:
                return rows, wrote_header, True
            if z.size > remaining:
                z = z[:remaining]
                soap_index = soap_index[:remaining]
                xyz = xyz[:remaining]
                loc = loc[:remaining]

        r, theta_nat, phi_nat = angles_from_xyz(xyz)
        shell_idx = np.floor(z / 0.05).astype(np.int16)
        theta_rot, phi_rot, xyz_rot = rotate_positions(
            r, theta_nat, phi_nat, shell_idx, args.angles
        )
        matched_cols = {col: values[loc] for col, values in soap_cols.items()}
        df = pd.DataFrame({
            "snap": np.full(z.size, snap, dtype=np.int16),
            "soap_index": soap_index,
            "z": z,
            "x_Mpc": xyz[:, 0],
            "y_Mpc": xyz[:, 1],
            "z_Mpc": xyz[:, 2],
            "r_comoving_Mpc": r,
            "shell_idx": shell_idx,
            "theta_nat_rad": theta_nat,
            "phi_nat_rad": phi_nat,
            "lon_nat_deg": np.degrees(phi_nat),
            "lat_nat_deg": 90.0 - np.degrees(theta_nat),
            "theta_rot_rad": theta_rot,
            "phi_rot_rad": phi_rot,
            "lon_rot_deg": np.degrees(phi_rot),
            "lat_rot_deg": 90.0 - np.degrees(theta_rot),
            "x_rot_Mpc": xyz_rot[:, 0],
            "y_rot_Mpc": xyz_rot[:, 1],
            "z_rot_Mpc": xyz_rot[:, 2],
            **matched_cols,
        })
        df.to_csv(args.out, mode="a", index=False, header=not wrote_header)
        wrote_header = True
        rows += len(df)
        args._rows_written += len(df)
        if args._rows_written % args.progress_every < len(df):
            print(f"    wrote {args._rows_written:,} rows total", flush=True)
        if args.max_rows is not None and args._rows_written >= args.max_rows:
            return rows, wrote_header, True
    return rows, wrote_header, False


def build(args: argparse.Namespace) -> None:
    base = _sim_base(args.parent, args.variant)
    lightcone = f"{base}/halo_lightcone/lightcone{args.observer}"
    soap_tmpl = f"{base}/SOAP-HBT/halo_properties_{{snap:04d}}.hdf5"
    args.angles = _angles_for(args.parent)

    root = hdfstream.open("cosma", "/")
    progress_path = args.out.with_suffix(args.out.suffix + ".progress.json")
    completed = set()
    args._rows_written = 0
    if args.resume and args.out.exists() and progress_path.exists() and args.max_rows is None:
        progress = json.loads(progress_path.read_text())
        completed = set(progress.get("completed_snapshots", []))
        args._rows_written = int(progress.get("rows_written", 0))
        wrote_header = args._rows_written > 0
        print(f"resuming: {len(completed)} snapshots done, {args._rows_written:,} rows written", flush=True)
    else:
        write_header(args.out, args)
        wrote_header = False

    t0 = time.time()
    for snap in range(args.snap_start, args.snap_stop + 1):
        if snap in completed:
            print(f"snap {snap:04d}: already complete", flush=True)
            continue
        print(f"snap {snap:04d}: scanning SOAP for M500c>=1e13", flush=True)
        soap = root[soap_tmpl.format(snap=snap)]
        selected_idx, soap_cols = load_selected_soap(soap, MASS_MIN, args.soap_chunk_size)
        print(f"snap {snap:04d}: kept {selected_idx.size:,} SOAP halos; joining lightcone", flush=True)
        if selected_idx.size:
            rows, wrote_header, stop = append_lightcone_matches(
                root, lightcone, snap, selected_idx, soap_cols, args, wrote_header
            )
        else:
            rows, stop = 0, False
        print(f"snap {snap:04d}: wrote {rows:,} rows ({(time.time()-t0)/60:.1f} min total)", flush=True)
        completed.add(snap)
        if args.max_rows is None:
            progress_path.write_text(json.dumps({
                "completed_snapshots": sorted(completed),
                "rows_written": args._rows_written,
                "updated_unix": time.time(),
            }, indent=2))
        if stop:
            break
    print(f"wrote {args._rows_written:,} rows -> {args.out}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default=DEFAULT_VARIANT)
    p.add_argument(
        "--parent",
        default=None,
        help="Parent FLAMINGO folder for L1_m9 variants (use L1_m9).",
    )
    p.add_argument("--observer", type=int, default=0)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--snap-start", type=int, default=18)
    p.add_argument(
        "--snap-stop",
        type=int,
        default=None,
        help="Last snapshot inclusive (default 77 for L1_m9, 78 for L2p8).",
    )
    p.add_argument("--soap-chunk-size", type=int, default=2_000_000)
    p.add_argument("--lightcone-chunk-size", type=int, default=1_000_000)
    p.add_argument("--progress-every", type=int, default=100_000)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.snap_stop is None:
        args.snap_stop = 77 if args.parent == "L1_m9" else 78
    return args


if __name__ == "__main__":
    build(parse_args())
