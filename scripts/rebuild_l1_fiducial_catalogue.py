#!/usr/bin/env python3
"""Rebuild the L1_m9 fiducial catalogue with an identity-safe SOAP join.

The released L1_m9 halo-lightcone ``SOAPIndex`` values are stale relative to
the current SOAP files. This builder joins on the stable
``InputHalos/HaloCatalogueIndex`` (the HBT-HERONS catalogue index) instead.
The canonical q-from-mass catalogue is replaced atomically at its existing
path; no corrected-name parallel product is created.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

import healpy as hp
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from flamingo.catalogue.join import match_hbt_indices  # noqa: E402
from flamingo.cnc import SZScaling  # noqa: E402


RUN = "L1_m9"
PORTAL_BASE = f"FLAMINGO/{RUN}/{RUN}"
SOAP_TEMPLATE = PORTAL_BASE + "/SOAP-HBT/halo_properties_{snap:04d}.hdf5"
LIGHTCONE_TEMPLATE = (
    PORTAL_BASE + "/halo_lightcone/lightcone0/lightcone_halos_{snap:04d}.hdf5"
)
CATALOGUE_DIR = Path("/rds/rds-lxu/flamingo/L1_m9/catalogues")
OUTPUT = CATALOGUE_DIR / (
    "halo_catalogue_M500c_5e13_zlt3_L1_m9_yang26rot_qfrommz.csv"
)
BACKUP_SUFFIX = ".pre_hbt_join_fix.bak"
MASS_CUT_INTERNAL = 5.0e13 / 1.0e10
N_SHELLS = 60
SNAPS = tuple(range(17, 78))

def _angles_l1() -> np.ndarray:
    path = Path(
        "/scratch/scratch-lxu/flamingo_data_analysis/map_making/build_y_map_L1.py"
    )
    spec = importlib.util.spec_from_file_location("build_y_map_L1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import rotation table from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return np.asarray(module.ANGLES_L1, dtype=np.float64)


def _remote_take(dataset, rows: np.ndarray, *, multislice_limit: int = 4096) -> np.ndarray:
    """Read sparse remote rows without scanning a huge SOAP dataset."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return np.empty((0, *dataset.shape[1:]), dtype=dataset.dtype)
    if rows.size <= multislice_limit:
        slices = [np.s_[int(row) : int(row) + 1] for row in rows]
        return np.asarray(dataset.request_slices(slices))
    return np.asarray(dataset[rows])


def _process_snapshot(snap: int) -> tuple[int, pd.DataFrame, dict[str, int]]:
    import hdfstream

    started = time.monotonic()
    root = hdfstream.open("cosma", "/")
    soap = root[SOAP_TEMPLATE.format(snap=snap)]
    lightcone = root[LIGHTCONE_TEMPLATE.format(snap=snap)]

    mass = np.asarray(soap["SO/500_crit/TotalMass"][:])
    soap_rows = np.flatnonzero(mass > MASS_CUT_INTERNAL)
    mass_selected = mass[soap_rows].astype(np.float64) * 1.0e10
    del mass
    if soap_rows.size == 0:
        return snap, pd.DataFrame(), {
            "soap_massive_centrals": 0,
            "lightcone_copies": 0,
            "seconds": int(round(time.monotonic() - started)),
        }
    central = np.asarray(
        _remote_take(soap["InputHalos/IsCentral"], soap_rows), dtype=bool
    )
    soap_rows = soap_rows[central]
    mass_selected = mass_selected[central]
    if soap_rows.size == 0:
        return snap, pd.DataFrame(), {
            "soap_massive_centrals": 0,
            "lightcone_copies": 0,
            "seconds": int(round(time.monotonic() - started)),
        }

    soap_hbt = np.asarray(
        _remote_take(soap["InputHalos/HaloCatalogueIndex"], soap_rows),
        dtype=np.int64,
    )
    lightcone_hbt = np.asarray(
        lightcone["InputHalos/HaloCatalogueIndex"][:], dtype=np.int64
    )
    lightcone_rows, matched_soap_rows = match_hbt_indices(
        lightcone_hbt,
        soap_hbt,
        soap_rows,
    )
    del lightcone_hbt
    if lightcone_rows.size == 0:
        return snap, pd.DataFrame(), {
            "soap_massive_centrals": int(soap_rows.size),
            "lightcone_copies": 0,
            "seconds": int(round(time.monotonic() - started)),
        }

    redshift = np.asarray(
        _remote_take(lightcone["Lightcone/Redshift"], lightcone_rows)
    )
    keep = (redshift >= 0.0) & (redshift < 3.0)
    lightcone_rows = lightcone_rows[keep]
    matched_soap_rows = matched_soap_rows[keep]
    redshift = redshift[keep]
    position = np.asarray(
        _remote_take(lightcone["Lightcone/HaloCentre"], lightcone_rows)
    )

    selected_lookup = np.searchsorted(soap_rows, matched_soap_rows)
    if not np.array_equal(soap_rows[selected_lookup], matched_soap_rows):
        raise RuntimeError(f"snap {snap}: matched SOAP row escaped selected set")

    scale_factor = 1.0 / (1.0 + redshift)
    radius_selected = np.asarray(
        _remote_take(soap["SO/500_crit/SORadius"], soap_rows),
        dtype=np.float64,
    )

    frame = pd.DataFrame(
        {
            "snap": np.full(redshift.size, snap, dtype=np.int32),
            "soap_index": matched_soap_rows,
            "hbt_halo_catalogue_index": soap_hbt[selected_lookup],
            "z": redshift,
            "x_Mpc": position[:, 0],
            "y_Mpc": position[:, 1],
            "z_Mpc": position[:, 2],
            "M_500c_Msun": mass_selected[selected_lookup],
            "R_500c_Mpc": radius_selected[selected_lookup] * scale_factor,
        }
    )
    stats = {
        "soap_massive_centrals": int(soap_rows.size),
        "lightcone_copies": int(len(frame)),
        "seconds": int(round(time.monotonic() - started)),
    }
    return snap, frame, stats


def _shell_radii() -> np.ndarray:
    import hdfstream

    root = hdfstream.open("cosma", "/")
    base = PORTAL_BASE + "/healpix_maps/nside_4096/lightcone0_shells"
    radii = np.empty(N_SHELLS, dtype=np.float64)
    for shell in range(N_SHELLS):
        path = f"{base}/shell_{shell}/lightcone0.shell_{shell}.0.hdf5"
        radii[shell] = float(
            root[path]["ComptonY"].attrs["comoving_outer_radius"][0]
        )
    return radii


def _add_rotated_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    radii = _shell_radii()
    angles = _angles_l1()
    xyz = frame[["x_Mpc", "y_Mpc", "z_Mpc"]].to_numpy(np.float64)
    radius = np.linalg.norm(xyz, axis=1)
    shell = np.searchsorted(radii, radius, side="right")
    keep = shell < N_SHELLS
    frame = frame.loc[keep].reset_index(drop=True)
    xyz = xyz[keep]
    radius = radius[keep]
    shell = shell[keep]
    theta_nat, phi_nat = hp.vec2ang(xyz)

    theta_rot = np.empty_like(theta_nat)
    phi_rot = np.empty_like(phi_nat)
    for shell_index in np.unique(shell):
        mask = shell == shell_index
        theta_angle = angles[0, shell_index]
        phi_angle = angles[1, shell_index]
        if theta_angle == 0.0 and phi_angle == 0.0:
            theta_rot[mask] = theta_nat[mask]
            phi_rot[mask] = phi_nat[mask]
        else:
            rotator = hp.Rotator(
                rot=[np.degrees(phi_angle), np.degrees(theta_angle)], inv=True
            )
            theta_rot[mask], phi_rot[mask] = rotator(
                theta_nat[mask], phi_nat[mask]
            )

    sin_theta = np.sin(theta_rot)
    geometry = {
        "r_comoving_Mpc": radius,
        "shell_idx": shell,
        "theta_nat_rad": theta_nat,
        "phi_nat_rad": phi_nat,
        "lon_nat_deg": np.degrees(phi_nat),
        "lat_nat_deg": 90.0 - np.degrees(theta_nat),
        "theta_rot_rad": theta_rot,
        "phi_rot_rad": phi_rot,
        "lon_rot_deg": np.degrees(phi_rot),
        "lat_rot_deg": 90.0 - np.degrees(theta_rot),
        "x_rot_Mpc": radius * sin_theta * np.cos(phi_rot),
        "y_rot_Mpc": radius * sin_theta * np.sin(phi_rot),
        "z_rot_Mpc": radius * np.cos(theta_rot),
    }
    insert_at = frame.columns.get_loc("M_500c_Msun")
    left = frame.iloc[:, :insert_at]
    right = frame.iloc[:, insert_at:]
    return pd.concat([left, pd.DataFrame(geometry), right], axis=1)


def _add_q(frame: pd.DataFrame) -> pd.DataFrame:
    scaling = SZScaling.calibrated(B=1.41)
    from dataclasses import replace

    scaling = replace(scaling, A_SZ=-4.0953238, alpha_SZ=1.12, B=1.41)
    frame["q_from_mz"] = np.asarray(
        scaling.q(
            frame["M_500c_Msun"].to_numpy(np.float64),
            frame["z"].to_numpy(np.float64),
            index=frame["soap_index"].to_numpy(np.uint32),
        ),
        dtype=np.float64,
    )
    return frame


def _install(frame: pd.DataFrame, output: Path) -> None:
    temporary = output.with_name(
        f"{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    header = [
        "# FLAMINGO L1_m9 lightcone0 cluster catalogue.",
        "# Selection: current SOAP M_500c > 5e13 Msun, IsCentral=1, 0 <= z < 3.",
        "# Identity join: halo-lightcone InputHalos/HaloCatalogueIndex -> current SOAP InputHalos/HaloCatalogueIndex.",
        "# The released L1_m9 halo-lightcone SOAPIndex is not used because it is stale relative to current SOAP files.",
        "# Positions are Lightcone/HaloCentre; masses are physical Msun; radii are physical Mpc.",
        "# Rotated coordinates use the official yang26 L1 per-shell convention.",
        "# A_SZ=-4.0953238 alpha_SZ=1.12 B=1.41 sigma_lnY=0.173 seed=20260630.",
    ]
    try:
        with temporary.open("x") as handle:
            handle.write("\n".join(header) + "\n")
            frame.to_csv(handle, index=False, float_format="%.17g")
        backup = output.with_name(output.name + BACKUP_SUFFIX)
        if output.exists() and not backup.exists():
            shutil.copy2(output, backup)
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    started = time.monotonic()
    chunks: dict[int, pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_snapshot, snap): snap for snap in SNAPS}
        for future in as_completed(futures):
            snap, frame, stats = future.result()
            chunks[snap] = frame
            print(
                f"snap {snap:04d}: SOAP massive centrals={stats['soap_massive_centrals']:,} "
                f"lightcone copies={stats['lightcone_copies']:,} "
                f"({stats['seconds']}s)",
                flush=True,
            )
    catalogue = pd.concat([chunks[snap] for snap in sorted(chunks)], ignore_index=True)
    catalogue = _add_rotated_geometry(catalogue)
    catalogue = _add_q(catalogue)
    if not np.all(catalogue["M_500c_Msun"].to_numpy() > 5.0e13):
        raise RuntimeError("mass cut failed")
    if not np.all(np.isfinite(catalogue.select_dtypes(include=[np.number]))):
        raise RuntimeError("non-finite numeric catalogue value")
    _install(catalogue, args.output)
    print(
        f"WROTE IN PLACE {args.output}: {len(catalogue):,} rows "
        f"in {time.monotonic() - started:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
