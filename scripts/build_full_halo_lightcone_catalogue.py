"""Build a streamed FLAMINGO halo-lightcone catalogue CSV.

This removes the old implicit M500c > 5e13 Msun selection by default, joins the
halo lightcone to SOAP aperture quantities via ``SOAPIndex``, and appends the
yang26-rotated sky positions used by the lightcone Compton-y maps.

Default scope is z < 3, matching the existing catalogue and the map shell
rotation frame used elsewhere in this repository. A literal no-mass-cut z < 3
build is very large (~9.3 billion raw lightcone rows), so use ``--max-rows`` for
smoke tests and consider a finite ``--mass-min`` for production products.
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

from build_y_map_highres import ANGLES_L2P8


BASE = "FLAMINGO/L2p8_m9/L2p8_m9"
LIGHTCONE = f"{BASE}/halo_lightcone/lightcone0"
SOAP_TMPL = f"{BASE}/SOAP-HBT/halo_properties_{{snap:04d}}.hdf5"
OUT = Path(
    "/scratch/scratch-lxu/flamingo_repo/data/hydro_L2p8m9/catalogue/"
    "halo_catalogue_full_zlt3_soap_hdfstream_yang26rot.csv"
)

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


def _physical_radius_factor(dataset) -> float:
    attrs = dict(dataset.attrs)
    cgs = float(np.asarray(attrs["Conversion factor to CGS (not including cosmological corrections)"])[0])
    pcgs = float(np.asarray(attrs["Conversion factor to physical CGS (including cosmological corrections)"])[0])
    return pcgs / cgs


def _read_by_index(dataset, index: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(index.astype(np.int64), return_inverse=True)
    return np.asarray(dataset[unique])[inverse]


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_rot = np.empty_like(theta_nat)
    phi_rot = np.empty_like(phi_nat)

    for shell in np.unique(shell_idx):
        if shell < 0 or shell >= ANGLES_L2P8.shape[1]:
            raise ValueError(
                f"shell_idx={shell} is outside the available yang26 angle table "
                f"(0..{ANGLES_L2P8.shape[1] - 1}). Lower --z-max or add angles."
            )
        sel = shell_idx == shell
        theta_angle, phi_angle = ANGLES_L2P8[:, shell]
        rot = hp.Rotator(
            rot=[phi_angle * 180.0 / np.pi, theta_angle * 180.0 / np.pi],
            inv=True,
        )
        theta_rot[sel], phi_rot[sel] = rot(theta_nat[sel], phi_nat[sel])

    xyz_rot = np.column_stack(
        [
            r * np.sin(theta_rot) * np.cos(phi_rot),
            r * np.sin(theta_rot) * np.sin(phi_rot),
            r * np.cos(theta_rot),
        ]
    )
    return theta_rot, phi_rot, xyz_rot


def _snapshot_numbers(z_max: float | None) -> range:
    if z_max is None:
        return range(0, 79)
    snap_min = max(0, int(np.floor(78 - z_max / 0.05)))
    return range(snap_min, 79)


def _write_header(path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# FLAMINGO L2p8_m9 halo lightcone0 catalogue from hdfstream.\n")
        f.write("# Joined to SOAP-HBT by InputHalos/SOAPIndex.\n")
        f.write("# Masses are Msun; radii and positions are Mpc; Compton-Y aperture values are Mpc^2.\n")
        f.write("# Rotated coordinates use the yang26 per-shell frame from the FLAMINGO map products.\n")
        f.write(f"# z_max={args.z_max}; mass_min={args.mass_min}; chunk_size={args.chunk_size}\n")


def build(args: argparse.Namespace) -> None:
    if args.z_max is not None and args.z_max > 0.05 * ANGLES_L2P8.shape[1]:
        raise ValueError(
            f"--z-max {args.z_max} needs shell rotations beyond the available "
            f"z<{0.05 * ANGLES_L2P8.shape[1]:.2f} table."
        )

    root = hdfstream.open("cosma", "/")
    out = args.out
    progress_path = out.with_suffix(out.suffix + ".progress.json")

    completed: set[int] = set()
    rows_written = 0
    if args.resume and out.exists() and progress_path.exists() and args.max_rows is None:
        progress = json.loads(progress_path.read_text())
        completed = set(progress.get("completed_snapshots", []))
        rows_written = int(progress.get("rows_written", 0))
        print(f"resuming {out}: {len(completed)} snapshots done, {rows_written:,} rows written")
    else:
        _write_header(out, args)

    wrote_csv_header = rows_written > 0
    t_start = time.time()

    for snap in _snapshot_numbers(args.z_max):
        if snap in completed:
            print(f"snap {snap:04d}: already complete, skip", flush=True)
            continue

        lc = root[f"{LIGHTCONE}/lightcone_halos_{snap:04d}.hdf5"]
        n = lc["Lightcone/Redshift"].shape[0]
        if n == 0:
            completed.add(snap)
            continue

        soap = root[SOAP_TMPL.format(snap=snap)]
        soap_datasets = {name: soap[path] for name, path in SOAP_FIELDS.items()}
        radius_factor = _physical_radius_factor(soap_datasets["R_500c_Mpc"])

        snap_rows = 0
        print(f"snap {snap:04d}: {n:,} input halos", flush=True)
        for start in range(0, n, args.chunk_size):
            stop = min(start + args.chunk_size, n)
            z = np.asarray(lc["Lightcone/Redshift"][start:stop], dtype=np.float64)
            keep = np.ones(z.size, dtype=bool)
            if args.z_max is not None:
                keep &= z < args.z_max
            if args.z_min is not None:
                keep &= z >= args.z_min
            if not keep.any():
                continue

            z = z[keep]
            soap_index = np.asarray(lc["InputHalos/SOAPIndex"][start:stop], dtype=np.int64)[keep]

            m500 = _read_by_index(soap_datasets["M_500c_Msun"], soap_index).astype(
                np.float64, copy=False
            ) * 1.0e10
            if args.mass_min > 0.0:
                mkeep = m500 >= args.mass_min
                if not mkeep.any():
                    continue
                z = z[mkeep]
                soap_index = soap_index[mkeep]
                m500 = m500[mkeep]

            xyz = np.asarray(lc["Lightcone/HaloCentre"][start:stop], dtype=np.float64)[keep]
            if args.mass_min > 0.0:
                xyz = xyz[mkeep]

            soap_cols: dict[str, np.ndarray] = {"M_500c_Msun": m500}
            for col, dataset in soap_datasets.items():
                if col == "M_500c_Msun":
                    continue
                values = _read_by_index(dataset, soap_index).astype(np.float64, copy=False)
                if col in MASS_COLUMNS:
                    values = values * 1.0e10
                elif col in RADIUS_COLUMNS:
                    values = values * radius_factor
                soap_cols[col] = values

            if args.max_rows is not None:
                remaining = args.max_rows - rows_written
                if remaining <= 0:
                    break
                if z.size > remaining:
                    z = z[:remaining]
                    xyz = xyz[:remaining]
                    soap_index = soap_index[:remaining]
                    for col in list(soap_cols):
                        soap_cols[col] = soap_cols[col][:remaining]

            r, theta_nat, phi_nat = _angles_from_xyz(xyz)
            shell_idx = np.floor(z / 0.05).astype(np.int16)
            theta_rot, phi_rot, xyz_rot = _rotate_positions(r, theta_nat, phi_nat, shell_idx)

            df = pd.DataFrame(
                {
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
                    **soap_cols,
                }
            )
            df.to_csv(out, mode="a", index=False, header=not wrote_csv_header)
            wrote_csv_header = True

            snap_rows += len(df)
            rows_written += len(df)
            if rows_written % args.progress_every < len(df):
                elapsed = time.time() - t_start
                print(
                    f"  wrote {rows_written:,} rows total "
                    f"({snap_rows:,} from snap {snap:04d}; {elapsed/60.0:.1f} min)",
                    flush=True,
                )

            if args.max_rows is not None and rows_written >= args.max_rows:
                print(f"reached --max-rows={args.max_rows:,}", flush=True)
                return

        completed.add(snap)
        if args.max_rows is None:
            progress_path.write_text(
                json.dumps(
                    {
                        "completed_snapshots": sorted(completed),
                        "rows_written": rows_written,
                        "updated_unix": time.time(),
                    },
                    indent=2,
                )
            )
        print(f"snap {snap:04d}: wrote {snap_rows:,} rows", flush=True)

    print(f"wrote {rows_written:,} rows -> {out}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--z-min", type=float, default=None)
    p.add_argument("--z-max", type=float, default=3.0)
    p.add_argument("--mass-min", type=float, default=0.0, help="Minimum M_500c in Msun.")
    p.add_argument("--chunk-size", type=int, default=200_000)
    p.add_argument("--max-rows", type=int, default=None, help="Stop after this many written rows.")
    p.add_argument("--progress-every", type=int, default=1_000_000)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
