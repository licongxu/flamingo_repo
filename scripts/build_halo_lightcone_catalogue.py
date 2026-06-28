"""Build a FLAMINGO halo-lightcone catalogue CSV (fast lightcone-first join).

Per snapshot: one contiguous lightcone read, z filter, contiguous SOAP slab
for M500c, then yang26-rotated positions.  Much faster than chunked scattered
SOAP reads over hdfstream.
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
from flamingo_snapshots import snap_range_zlt

DEFAULT_VARIANT = "L2p8_m9"
MASS_MIN_DEFAULT = 1.0e13
Z_MIN = 0.0
Z_MAX = 3.0
M_FACTOR = 1.0e10

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


def _sim_base(parent: str | None, variant: str) -> str:
    if parent is None:
        return f"FLAMINGO/{variant}/{variant}"
    return f"FLAMINGO/{parent}/{variant}"


def _physical_radius_factor(dataset) -> float:
    attrs = dict(dataset.attrs)
    cgs = float(np.asarray(attrs["Conversion factor to CGS (not including cosmological corrections)"])[0])
    pcgs = float(np.asarray(attrs["Conversion factor to physical CGS (including cosmological corrections)"])[0])
    return pcgs / cgs


def _read_by_index(dataset, index: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(index.astype(np.int64), return_inverse=True)
    return np.asarray(dataset[unique])[inverse]


def _convert_values(col: str, values: np.ndarray, radius_factor: float) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    if col in MASS_COLUMNS:
        return values * M_FACTOR
    if col in RADIUS_COLUMNS:
        return values * radius_factor
    return values


def _angles_from_xyz(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.sqrt(np.sum(xyz * xyz, axis=1))
    theta = np.arccos(np.clip(xyz[:, 2] / r, -1.0, 1.0))
    phi = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2.0 * np.pi)
    return r, theta, phi


def _rotate_positions(
    r: np.ndarray,
    theta_nat: np.ndarray,
    phi_nat: np.ndarray,
    shell_idx: np.ndarray,
    angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _filter_snaps_by_soap_z(
    root,
    soap_tmpl: str,
    snaps: list[int],
    z_max: float,
    z_buffer: float = 0.5,
) -> list[int]:
    kept: list[int] = []
    for snap in snaps:
        try:
            z_snap = float(root[soap_tmpl.format(snap=snap)]["Header"].attrs["Redshift"][0])
        except Exception as exc:
            print(f"snap {snap:04d}: header read failed ({exc}), keeping", flush=True)
            kept.append(snap)
            continue
        if z_snap > z_max + z_buffer:
            print(f"snap {snap:04d}: SOAP z={z_snap:.2f} > z_max+{z_buffer}, skip", flush=True)
            continue
        kept.append(snap)
    return kept


def process_snap(
    root,
    lightcone: str,
    soap_tmpl: str,
    snap: int,
    soap_fields: dict[str, str],
    z_max: float,
    mass_min: float,
    angles: np.ndarray,
    shell_max: int,
) -> pd.DataFrame | None:
    """One snap: mass-cut from SOAPIndex + a contiguous M500c slab, then read
    Redshift and HaloCentre only for the surviving rows.

    hdfstream transfer is dominated by the per-row lightcone arrays.  Only
    ``InputHalos/SOAPIndex`` must be read in full (every row needs its mass);
    the M500c cut is applied first and the heavy ``Redshift``/``HaloCentre``
    columns are then fetched for just the handful of survivors.  This roughly
    halves network traffic vs reading every per-row column, and avoids a
    full-length ``np.unique`` inverse array (lower peak RAM under parallelism).
    """
    lc = root[f"{lightcone}/lightcone_halos_{snap:04d}.hdf5"]
    soap_idx_all = np.asarray(lc["InputHalos/SOAPIndex"][:], dtype=np.int64)
    if soap_idx_all.size == 0:
        return None

    soap_f = root[soap_tmpl.format(snap=snap)]
    m500_path = soap_fields["M_500c_Msun"]
    lo, hi = int(soap_idx_all.min()), int(soap_idx_all.max()) + 1
    m500_slab = np.asarray(soap_f[m500_path][lo:hi], dtype=np.float64) * M_FACTOR
    mass_ok_slab = m500_slab >= mass_min

    keep_mass = mass_ok_slab[soap_idx_all - lo]
    mass_rows = np.nonzero(keep_mass)[0]
    if mass_rows.size == 0:
        return None

    z_mass = np.asarray(lc["Lightcone/Redshift"][mass_rows], dtype=np.float64)
    zmask = (z_mass >= Z_MIN) & (z_mass <= z_max)
    if not zmask.any():
        return None

    final_rows = mass_rows[zmask]
    z_out = z_mass[zmask]
    soap_index = soap_idx_all[final_rows]
    xyz = np.asarray(lc["Lightcone/HaloCentre"][final_rows], dtype=np.float64)

    unique_idx, inverse = np.unique(soap_index, return_inverse=True)
    radius_factor = _physical_radius_factor(soap_f[soap_fields["R_500c_Mpc"]])
    soap_cols: dict[str, np.ndarray] = {
        "M_500c_Msun": m500_slab[soap_index - lo],
    }
    for col, path in soap_fields.items():
        if col == "M_500c_Msun":
            continue
        vals_u = _convert_values(
            col, np.asarray(soap_f[path][unique_idx], dtype=np.float64), radius_factor
        )
        soap_cols[col] = vals_u[inverse]

    r, theta_nat, phi_nat = _angles_from_xyz(xyz)
    shell_idx = np.minimum(np.floor(z_out / 0.05).astype(np.int16), shell_max)
    theta_rot, phi_rot, xyz_rot = _rotate_positions(r, theta_nat, phi_nat, shell_idx, angles)

    return pd.DataFrame({
        "snap": np.full(z_out.size, snap, dtype=np.int16),
        "soap_index": soap_index,
        "z": z_out,
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
        **soap_cols,
    })


def _write_header(path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tag = f"{args.parent}/{args.variant}" if args.parent else args.variant
    with path.open("w") as f:
        f.write(f"# FLAMINGO {tag} halo lightcone{args.observer} catalogue from hdfstream.\n")
        f.write("# Selection: M_500c >= 1e13 Msun and 0 <= z <= 3.0.\n")
        f.write("# Joined to SOAP-HBT by InputHalos/SOAPIndex.\n")
        f.write("# Masses are Msun; radii and positions are Mpc")
        if not args.no_y_columns:
            f.write("; Compton-Y aperture values are Mpc^2")
        f.write(".\n")
        f.write("# Rotated coordinates use the yang26 per-shell frame from the FLAMINGO map products.\n")
        f.write("# Read mode: full-snap lightcone + contiguous SOAP slab (autoflamingo-style).\n")


def _output_columns(args: argparse.Namespace) -> list[str]:
    cols = [
        "snap", "soap_index", "z", "x_Mpc", "y_Mpc", "z_Mpc", "r_comoving_Mpc", "shell_idx",
        "theta_nat_rad", "phi_nat_rad", "lon_nat_deg", "lat_nat_deg",
        "theta_rot_rad", "phi_rot_rad", "lon_rot_deg", "lat_rot_deg",
        "x_rot_Mpc", "y_rot_Mpc", "z_rot_Mpc",
        "M_500c_Msun", "M_200c_Msun", "M_200m_Msun",
        "R_500c_Mpc", "R_200c_Mpc", "R_200m_Mpc",
    ]
    if not args.no_y_columns:
        cols.extend([
            "Y_500c_Mpc2", "Y_500c_noAGN_Mpc2", "Y_5R500c_Mpc2", "Y_5R500c_noAGN_Mpc2",
        ])
    return cols


def catalogue_status(args: argparse.Namespace) -> str:
    progress_path = args.out.with_suffix(args.out.suffix + ".progress.json")
    if not progress_path.exists():
        return "missing"
    meta = json.loads(progress_path.read_text())
    done = set(meta.get("completed_snapshots", []))
    need = set(range(args.snap_start, args.snap_stop + 1))
    if need <= done:
        return "complete"
    return "partial"


def build(args: argparse.Namespace) -> None:
    base = _sim_base(args.parent, args.variant)
    lightcone = f"{base}/halo_lightcone/lightcone{args.observer}"
    soap_tmpl = f"{base}/SOAP-HBT/halo_properties_{{snap:04d}}.hdf5"
    angles = _angles_for(args.parent)
    shell_max = angles.shape[1] - 1
    soap_fields = {
        k: v for k, v in SOAP_FIELDS.items()
        if not args.no_y_columns or not k.startswith("Y_")
    }

    root = hdfstream.open("cosma", "/")
    progress_path = args.out.with_suffix(args.out.suffix + ".progress.json")
    completed: set[int] = set()
    rows_written = 0
    if args.resume and args.out.exists() and progress_path.exists() and args.max_rows is None:
        progress = json.loads(progress_path.read_text())
        completed = set(progress.get("completed_snapshots", []))
        rows_written = int(progress.get("rows_written", 0))
        wrote_header = rows_written > 0
        print(f"resuming: {len(completed)} snapshots done, {rows_written:,} rows written", flush=True)
    else:
        _write_header(args.out, args)
        wrote_header = False

    all_snaps = list(range(args.snap_start, args.snap_stop + 1))
    if args.prefilter:
        pending = [s for s in all_snaps if s not in completed]
        filtered = _filter_snaps_by_soap_z(root, soap_tmpl, pending, args.z_max)
        print(
            f"layout={args.parent or args.variant} snaps {args.snap_start}..{args.snap_stop} "
            f"({len(filtered)} to process after prefilter, z <= {args.z_max})",
            flush=True,
        )
    else:
        filtered = all_snaps
        print(
            f"layout={args.parent or args.variant} snaps {args.snap_start}..{args.snap_stop} "
            f"({len(filtered)} snapshots, z <= {args.z_max})",
            flush=True,
        )

    t0 = time.time()
    for snap in all_snaps:
        if snap in completed:
            continue
        if snap not in filtered:
            completed.add(snap)
            continue

        ts = time.time()
        try:
            df = process_snap(
                root, lightcone, soap_tmpl, snap, soap_fields,
                args.z_max, args.mass_min, angles, shell_max,
            )
        except Exception as exc:
            print(f"snap {snap:04d}: FAILED ({type(exc).__name__}: {exc})", flush=True)
            raise

        snap_rows = 0
        if df is not None and len(df):
            if args.max_rows is not None:
                remaining = args.max_rows - rows_written
                if remaining <= 0:
                    break
                if len(df) > remaining:
                    df = df.iloc[:remaining]
            df = df[_output_columns(args)]
            df.to_csv(args.out, mode="a", index=False, header=not wrote_header)
            wrote_header = True
            snap_rows = len(df)
            rows_written += snap_rows

        dt = time.time() - ts
        print(f"snap {snap:04d}: {snap_rows:,} rows ({dt:.1f}s)", flush=True)
        completed.add(snap)
        if args.max_rows is None:
            progress_path.write_text(json.dumps({
                "completed_snapshots": sorted(completed),
                "rows_written": rows_written,
                "updated_unix": time.time(),
            }, indent=2))
        if args.max_rows is not None and rows_written >= args.max_rows:
            print(f"reached --max-rows={args.max_rows:,}", flush=True)
            break

    print(f"wrote {rows_written:,} rows -> {args.out} ({(time.time()-t0)/60:.1f} min)", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", default=DEFAULT_VARIANT)
    p.add_argument("--parent", default=None)
    p.add_argument("--observer", type=int, default=0)
    p.add_argument("--out", type=Path, required=False)
    p.add_argument("--snap-start", type=int, default=None)
    p.add_argument("--snap-stop", type=int, default=None)
    p.add_argument("--z-max", type=float, default=Z_MAX)
    p.add_argument("--mass-min", type=float, default=MASS_MIN_DEFAULT)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--no-y-columns", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--status", action="store_true")
    # snap_range_zlt already bounds snaps to the requested z range, so the SOAP
    # header prefilter normally skips nothing and just adds startup overhead.
    # Off by default; enable when passing a wider snap range manually.
    p.add_argument("--prefilter", action="store_true", default=False)
    p.add_argument("--no-prefilter", action="store_false", dest="prefilter")
    p.add_argument("--chunk-size", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--progress-every", type=int, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    default_start, default_stop = snap_range_zlt(args.parent, args.variant, z_max=args.z_max, z_min=Z_MIN)
    if args.snap_start is None:
        args.snap_start = default_start
    if args.snap_stop is None:
        args.snap_stop = default_stop
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.status:
        if args.out is None:
            raise SystemExit("--out is required with --status")
        print(catalogue_status(args))
    else:
        if args.out is None:
            raise SystemExit("--out is required")
        build(args)
