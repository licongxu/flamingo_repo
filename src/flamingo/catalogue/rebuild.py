"""Identity-safe FLAMINGO catalogue rebuild metadata and primitives."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import healpy as hp
import numpy as np
import pandas as pd

from .join import resolve_lightcone_rows


Family = Literal["l1", "l2"]

L1_VARIANTS = (
    "L1_m9",
    "fgas+2sigma",
    "fgas-2sigma",
    "fgas-4sigma",
    "fgas-8sigma",
    "Mstar-1sigma",
    "Mstar-1sigma_fgas-4sigma",
    "Jet",
    "Jet_fgas-4sigma",
)

GEOMETRY_COLUMNS = (
    "snap",
    "soap_index",
    "z",
    "x_Mpc",
    "y_Mpc",
    "z_Mpc",
    "r_comoving_Mpc",
    "shell_idx",
    "theta_nat_rad",
    "phi_nat_rad",
    "lon_nat_deg",
    "lat_nat_deg",
    "theta_rot_rad",
    "phi_rot_rad",
    "lon_rot_deg",
    "lat_rot_deg",
    "x_rot_Mpc",
    "y_rot_Mpc",
    "z_rot_Mpc",
)

SOAP_COLUMNS = (
    "M_500c_Msun",
    "M_200c_Msun",
    "M_200m_Msun",
    "R_500c_Mpc",
    "R_200c_Mpc",
    "R_200m_Mpc",
)

L1_Y_COLUMNS = (
    "Y_500c_Mpc2",
    "Y_500c_noAGN_Mpc2",
    "Y_5R500c_Mpc2",
    "Y_5R500c_noAGN_Mpc2",
)

APERTURE_COLUMNS = (
    "theta_500_arcmin",
    "Y_500cyl_arcmin2",
    "sigma_Y500_arcmin2",
    "npix_in_aperture",
    "q_from_aperture",
)


@dataclass(frozen=True)
class CatalogueTarget:
    """One simulation variant/lightcone and its four canonical CSV products."""

    family: Family
    run: str
    variant: str
    lightcone: int
    catalogue_dir: Path
    map_path: Path

    @property
    def key(self) -> str:
        if self.family == "l1":
            return f"L1_m9/{self.variant}"
        return f"L2p8_m9/lightcone{self.lightcone}"

    @property
    def catalogue_stem(self) -> str:
        return self.variant if self.family == "l1" else "L2p8_m9"

    @property
    def canonical_csvs(self) -> tuple[Path, Path, Path, Path]:
        stem = self.catalogue_stem
        prefix = "halo_catalogue_M500c_"
        suffix = f"_{stem}_yang26rot"
        names = (
            f"{prefix}1e13_zlt3{suffix}.csv",
            f"{prefix}5e13_zlt3{suffix}_qfrommz.csv",
            f"{prefix}5e13_zlt3{suffix}_qfrommz_alpha_fixed_1p12.csv",
            f"{prefix}5e13_zlt3{suffix}_qfrommap.csv",
        )
        return tuple(self.catalogue_dir / name for name in names)  # type: ignore[return-value]


def base_columns(family: Family) -> tuple[str, ...]:
    """Canonical base-catalogue schema for a simulation family."""
    if family == "l1":
        return (*GEOMETRY_COLUMNS, *SOAP_COLUMNS, *L1_Y_COLUMNS)
    if family == "l2":
        return (*GEOMETRY_COLUMNS, *SOAP_COLUMNS)
    raise ValueError(f"unknown catalogue family: {family}")


def q_columns(family: Family) -> tuple[str, ...]:
    """Canonical q-from-mass/redshift schema."""
    return (*base_columns(family), "q_from_mz")


def qmap_columns(family: Family) -> tuple[str, ...]:
    """Canonical q-from-map schema."""
    return (*base_columns(family), *APERTURE_COLUMNS)


def catalogue_targets(root: Path) -> tuple[CatalogueTarget, ...]:
    """Return the complete canonical L1 feedback and L2 lightcone matrix."""
    root = Path(root)
    targets = [
        CatalogueTarget(
            family="l1",
            run="L1_m9",
            variant=variant,
            lightcone=0,
            catalogue_dir=root / "L1_m9/catalogues",
            map_path=(
                root
                / "L1_m9/maps"
                / f"y_unlensed_{variant}_lc0_nside4096.fits"
            ),
        )
        for variant in L1_VARIANTS
    ]
    targets.extend(
        CatalogueTarget(
            family="l2",
            run="L2p8_m9",
            variant="L2p8_m9",
            lightcone=lightcone,
            catalogue_dir=(
                root / "L2p8_m9" / f"lightcone{lightcone}" / "catalogues"
            ),
            map_path=(
                root
                / "L2p8_m9"
                / f"lightcone{lightcone}"
                / "healpix_map"
                / f"y_unlensed_L2p8_m9_lc{lightcone}.fits"
            ),
        )
        for lightcone in range(8)
    )
    return tuple(targets)


def _snapshot_columns(family: Family) -> tuple[str, ...]:
    columns = (
        "snap",
        "soap_index",
        "z",
        "x_Mpc",
        "y_Mpc",
        "z_Mpc",
        *SOAP_COLUMNS,
    )
    if family == "l1":
        return (*columns, *L1_Y_COLUMNS)
    if family == "l2":
        return columns
    raise ValueError(f"unknown catalogue family: {family}")


def _identity_verified_rows(
    source,
    target: CatalogueTarget,
    snap: int,
    lightcone_hbt: np.ndarray,
    row_hint: np.ndarray,
) -> np.ndarray:
    lightcone_hbt = np.asarray(lightcone_hbt, dtype=np.int64)
    row_hint = np.asarray(row_hint, dtype=np.int64)
    if lightcone_hbt.shape != row_hint.shape or lightcone_hbt.ndim != 1:
        raise ValueError("lightcone identity and SOAP row hint must be equal 1D arrays")
    if lightcone_hbt.size == 0:
        return np.empty(0, dtype=np.int64)

    hint_verified = False
    if np.all(row_hint >= 0):
        try:
            hinted_hbt = np.asarray(
                source.read_soap_identity(target, snap, rows=row_hint),
                dtype=np.int64,
            )
            hint_verified = np.array_equal(hinted_hbt, lightcone_hbt)
        except (IndexError, KeyError, ValueError):
            hint_verified = False
    if hint_verified:
        return row_hint

    current_soap_hbt = np.asarray(
        source.read_soap_identity(target, snap, rows=None), dtype=np.int64
    )
    _, soap_rows = resolve_lightcone_rows(lightcone_hbt, current_soap_hbt)
    return soap_rows


def build_snapshot_frame(
    source,
    target: CatalogueTarget,
    snap: int,
    mass_cut_msun: float,
) -> pd.DataFrame:
    """Build one identity-safe, pre-rotation snapshot frame."""
    if mass_cut_msun <= 0:
        raise ValueError("mass_cut_msun must be positive")
    lightcone_hbt, row_hint = source.read_lightcone_identity(target, snap)
    lightcone_hbt = np.asarray(lightcone_hbt, dtype=np.int64)
    soap_rows_for_lightcone = _identity_verified_rows(
        source, target, snap, lightcone_hbt, row_hint
    )
    columns = _snapshot_columns(target.family)
    if lightcone_hbt.size == 0:
        return pd.DataFrame(columns=columns)

    unique_soap_rows, inverse = np.unique(
        soap_rows_for_lightcone, return_inverse=True
    )
    fields = {
        name: np.asarray(values)
        for name, values in source.read_soap_fields(
            target, snap, unique_soap_rows
        ).items()
    }
    required = {
        "is_central",
        "m500",
        "m200c",
        "m200m",
        "r500",
        "r200c",
        "r200m",
    }
    if target.family == "l1":
        required.update({"y500", "y500_noagn", "y5r500", "y5r500_noagn"})
    missing = sorted(required.difference(fields))
    if missing:
        raise ValueError(f"missing SOAP field(s): {', '.join(missing)}")
    if any(fields[name].shape != unique_soap_rows.shape for name in required):
        raise ValueError("SOAP field shape does not match requested unique rows")

    selected_unique = np.asarray(fields["is_central"], dtype=bool) & (
        np.asarray(fields["m500"], dtype=np.float64) * 1.0e10 >= mass_cut_msun
    )
    selected_lightcone = selected_unique[inverse]
    lightcone_rows = np.flatnonzero(selected_lightcone)
    if lightcone_rows.size == 0:
        return pd.DataFrame(columns=columns)

    redshift, position = source.read_lightcone_fields(
        target, snap, lightcone_rows
    )
    redshift = np.asarray(redshift, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    if redshift.shape != (lightcone_rows.size,) or position.shape != (
        lightcone_rows.size,
        3,
    ):
        raise ValueError("lightcone field shape does not match requested rows")
    redshift_keep = (redshift >= 0.0) & (redshift < 3.0)
    lightcone_rows = lightcone_rows[redshift_keep]
    redshift = redshift[redshift_keep]
    position = position[redshift_keep]
    if lightcone_rows.size == 0:
        return pd.DataFrame(columns=columns)

    matched_soap_rows = soap_rows_for_lightcone[lightcone_rows]
    unique_lookup = np.searchsorted(unique_soap_rows, matched_soap_rows)
    if not np.array_equal(unique_soap_rows[unique_lookup], matched_soap_rows):
        raise RuntimeError("resolved SOAP row escaped the unique lookup")
    scale_factor = 1.0 / (1.0 + redshift)
    frame_data = {
        "snap": np.full(redshift.size, snap, dtype=np.int32),
        "soap_index": matched_soap_rows,
        "z": redshift,
        "x_Mpc": position[:, 0],
        "y_Mpc": position[:, 1],
        "z_Mpc": position[:, 2],
        "M_500c_Msun": fields["m500"][unique_lookup] * 1.0e10,
        "M_200c_Msun": fields["m200c"][unique_lookup] * 1.0e10,
        "M_200m_Msun": fields["m200m"][unique_lookup] * 1.0e10,
        "R_500c_Mpc": fields["r500"][unique_lookup] * scale_factor,
        "R_200c_Mpc": fields["r200c"][unique_lookup] * scale_factor,
        "R_200m_Mpc": fields["r200m"][unique_lookup] * scale_factor,
    }
    if target.family == "l1":
        frame_data.update(
            {
                "Y_500c_Mpc2": fields["y500"][unique_lookup],
                "Y_500c_noAGN_Mpc2": fields["y500_noagn"][unique_lookup],
                "Y_5R500c_Mpc2": fields["y5r500"][unique_lookup],
                "Y_5R500c_noAGN_Mpc2": fields["y5r500_noagn"][unique_lookup],
            }
        )
    return pd.DataFrame(frame_data, columns=columns)


def add_rotated_geometry(
    frame: pd.DataFrame,
    shell_radii_mpc: np.ndarray,
    angles: np.ndarray,
    family: Family,
) -> pd.DataFrame:
    """Add natural and official yang26-rotated geometry to a snapshot frame."""
    shell_radii_mpc = np.asarray(shell_radii_mpc, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    if shell_radii_mpc.ndim != 1 or angles.shape != (2, shell_radii_mpc.size):
        raise ValueError("rotation angles must have shape (2, number of shells)")
    if frame.empty:
        return pd.DataFrame(columns=base_columns(family))

    xyz = frame[["x_Mpc", "y_Mpc", "z_Mpc"]].to_numpy(np.float64)
    radius = np.linalg.norm(xyz, axis=1)
    shell = np.searchsorted(shell_radii_mpc, radius, side="right")
    if np.any(shell >= shell_radii_mpc.size):
        raise ValueError("lightcone position lies outside available z<3 shells")
    theta_nat, phi_nat = hp.vec2ang(xyz)
    theta_rot = np.empty_like(theta_nat)
    phi_rot = np.empty_like(phi_nat)
    for shell_index in np.unique(shell):
        mask = shell == shell_index
        theta_angle, phi_angle = angles[:, shell_index]
        if theta_angle == 0.0 and phi_angle == 0.0:
            theta_rot[mask] = theta_nat[mask]
            phi_rot[mask] = phi_nat[mask]
            continue
        rotator = hp.Rotator(
            rot=[np.degrees(phi_angle), np.degrees(theta_angle)], inv=True
        )
        theta_rot[mask], phi_rot[mask] = rotator(
            theta_nat[mask], phi_nat[mask]
        )

    sin_theta = np.sin(theta_rot)
    output = frame.copy()
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
    for name, values in geometry.items():
        output[name] = values
    return output.loc[:, base_columns(family)]
