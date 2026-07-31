"""Identity-safe FLAMINGO catalogue rebuild metadata and primitives."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Literal
import uuid

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
    if target.family == "l2" and np.all(row_hint >= 0):
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
    selection_fields = {
        name: np.asarray(values)
        for name, values in source.read_soap_selection_fields(
            target, snap, unique_soap_rows
        ).items()
    }
    selection_required = {"is_central", "m500"}
    missing = sorted(selection_required.difference(selection_fields))
    if missing:
        raise ValueError(f"missing SOAP selection field(s): {', '.join(missing)}")
    if any(
        selection_fields[name].shape != unique_soap_rows.shape
        for name in selection_required
    ):
        raise ValueError("SOAP selection field shape does not match requested rows")

    m500_internal = np.asarray(selection_fields["m500"], dtype=np.float64)
    selected_unique = np.asarray(selection_fields["is_central"], dtype=bool) & (
        m500_internal * 1.0e10 >= mass_cut_msun
    )
    selected_soap_rows = unique_soap_rows[selected_unique]
    selected_m500 = m500_internal[selected_unique]
    properties = {
        name: np.asarray(values)
        for name, values in source.read_soap_property_fields(
            target, snap, selected_soap_rows
        ).items()
    }
    property_required = {
        "m200c",
        "m200m",
        "r500",
        "r200c",
        "r200m",
    }
    if target.family == "l1":
        property_required.update(
            {"y500", "y500_noagn", "y5r500", "y5r500_noagn"}
        )
    missing = sorted(property_required.difference(properties))
    if missing:
        raise ValueError(f"missing SOAP property field(s): {', '.join(missing)}")
    if any(
        properties[name].shape != selected_soap_rows.shape
        for name in property_required
    ):
        raise ValueError("SOAP property field shape does not match selected rows")
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
    selected_lookup = np.searchsorted(selected_soap_rows, matched_soap_rows)
    if not np.array_equal(selected_soap_rows[selected_lookup], matched_soap_rows):
        raise RuntimeError("resolved SOAP row escaped the selected lookup")
    scale_factor = 1.0 / (1.0 + redshift)
    frame_data = {
        "snap": np.full(redshift.size, snap, dtype=np.int32),
        "soap_index": matched_soap_rows,
        "z": redshift,
        "x_Mpc": position[:, 0],
        "y_Mpc": position[:, 1],
        "z_Mpc": position[:, 2],
        "M_500c_Msun": selected_m500[selected_lookup] * 1.0e10,
        "M_200c_Msun": np.asarray(properties["m200c"], dtype=np.float64)[selected_lookup] * 1.0e10,
        "M_200m_Msun": np.asarray(properties["m200m"], dtype=np.float64)[selected_lookup] * 1.0e10,
        "R_500c_Mpc": properties["r500"][selected_lookup] * scale_factor,
        "R_200c_Mpc": properties["r200c"][selected_lookup] * scale_factor,
        "R_200m_Mpc": properties["r200m"][selected_lookup] * scale_factor,
    }
    if target.family == "l1":
        frame_data.update(
            {
                "Y_500c_Mpc2": properties["y500"][selected_lookup],
                "Y_500c_noAGN_Mpc2": properties["y500_noagn"][selected_lookup],
                "Y_5R500c_Mpc2": properties["y5r500"][selected_lookup],
                "Y_5R500c_noAGN_Mpc2": properties["y5r500_noagn"][selected_lookup],
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("x") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_progress(path: Path, target: CatalogueTarget) -> dict:
    if not path.exists():
        return {"target": target.key, "snapshots": {}}
    with path.open() as handle:
        progress = json.load(handle)
    if progress.get("target") != target.key:
        raise ValueError(f"progress target mismatch: {path}")
    if not isinstance(progress.get("snapshots"), dict):
        raise ValueError(f"malformed snapshot progress: {path}")
    return progress


def _part_is_verified(path: Path, record: dict, columns: tuple[str, ...]) -> bool:
    if not path.is_file() or path.stat().st_size != record.get("bytes"):
        return False
    if _sha256(path) != record.get("sha256"):
        return False
    observed = tuple(pd.read_csv(path, nrows=0).columns)
    return observed == columns


def _base_provenance(target: CatalogueTarget) -> str:
    return "\n".join(
        [
            f"# FLAMINGO {target.variant} halo lightcone{target.lightcone} catalogue from hdfstream.",
            "# Selection: current SOAP M_500c >= 1e13 Msun, IsCentral=1, 0 <= z < 3.",
            "# Identity join: halo-lightcone InputHalos/HaloCatalogueIndex -> current SOAP InputHalos/HaloCatalogueIndex.",
            "# InputHalos/SOAPIndex is retained only as an identity-verified row hint and is never a join key.",
            "# Masses are physical Msun; radii and positions are physical/comoving Mpc as named.",
            "# Rotated coordinates use the official yang26 per-shell frame matching the Compton-y map.",
        ]
    ) + "\n"


def build_base_catalogue(
    target: CatalogueTarget,
    staged_path: Path,
    source,
    *,
    snaps: tuple[int, ...],
    shell_radii_mpc: np.ndarray,
    angles: np.ndarray,
    mass_cut_msun: float = 1.0e13,
) -> dict[str, object]:
    """Build a resumable staged base catalogue from atomic snapshot parts."""
    staged_path = Path(staged_path)
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = staged_path.parent / f".{staged_path.name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    progress_path = staged_path.parent / f".{staged_path.name}.progress.json"
    progress = _read_progress(progress_path, target)
    columns = base_columns(target.family)

    for snap in snaps:
        key = f"{snap:04d}"
        part = parts_dir / f"snap_{snap:04d}.csv"
        record = progress["snapshots"].get(key, {})
        if _part_is_verified(part, record, columns):
            continue

        frame = build_snapshot_frame(source, target, snap, mass_cut_msun)
        frame = add_rotated_geometry(
            frame, shell_radii_mpc, angles, target.family
        )
        numeric = frame.select_dtypes(include=[np.number]).to_numpy()
        if numeric.size and not np.all(np.isfinite(numeric)):
            raise ValueError(f"{target.key} snap {snap}: non-finite numeric value")
        temporary = part.with_name(
            f"{part.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            frame.to_csv(temporary, index=False, float_format="%.17g")
            os.replace(temporary, part)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        progress["snapshots"][key] = {
            "rows": int(len(frame)),
            "bytes": part.stat().st_size,
            "sha256": _sha256(part),
        }
        _write_json_atomic(progress_path, progress)

    missing = [
        snap
        for snap in snaps
        if not _part_is_verified(
            parts_dir / f"snap_{snap:04d}.csv",
            progress["snapshots"].get(f"{snap:04d}", {}),
            columns,
        )
    ]
    if missing:
        raise RuntimeError(f"unverified snapshot parts: {missing}")

    temporary = staged_path.with_name(
        f"{staged_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("x") as output:
            output.write(_base_provenance(target))
            first = True
            for snap in snaps:
                part = parts_dir / f"snap_{snap:04d}.csv"
                with part.open() as input_handle:
                    header = input_handle.readline()
                    if first:
                        output.write(header)
                        first = False
                    shutil.copyfileobj(input_handle, output, length=8 * 1024 * 1024)
        os.replace(temporary, staged_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    rows = sum(
        int(progress["snapshots"][f"{snap:04d}"]["rows"]) for snap in snaps
    )
    return {
        "target": target.key,
        "path": str(staged_path),
        "rows": rows,
        "bytes": staged_path.stat().st_size,
        "sha256": _sha256(staged_path),
        "progress": str(progress_path),
    }


def _q_provenance(base_path: Path) -> str:
    return "\n".join(
        [
            "# FLAMINGO stable-identity catalogue subset with q_from_mz.",
            f"# Source: {base_path}",
            "# Selection: physical M_500c > 5e13 Msun.",
            "# A_SZ=-4.0953238 alpha_SZ=1.12 B=1.41",
            "# sigma_lnY=0.173 seed=20260630 cosmology=D3A noise=SZiFi-immf6",
            "# q uses the current M_500c mass definition and identity-resolved SOAP row.",
        ]
    ) + "\n"


def derive_q_catalogues(
    base_path: Path,
    output_paths: tuple[Path, Path],
    family: Family,
    scaling,
    *,
    chunk_size: int = 100_000,
) -> dict[str, object]:
    """Stream one staged base catalogue into both canonical q flavours."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    base_path = Path(base_path)
    outputs = tuple(Path(path) for path in output_paths)
    if len(outputs) != 2 or outputs[0] == outputs[1]:
        raise ValueError("two distinct q catalogue output paths are required")
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    temporary = tuple(
        output.with_name(
            f"{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
        )
        for output in outputs
    )
    expected_base = base_columns(family)
    rows = 0
    first = True
    try:
        for path in temporary:
            with path.open("x") as handle:
                handle.write(_q_provenance(base_path))
        for chunk in pd.read_csv(
            base_path,
            comment="#",
            chunksize=chunk_size,
            float_precision="round_trip",
        ):
            if tuple(chunk.columns) != expected_base:
                raise ValueError(f"unexpected base schema in {base_path}")
            selected = chunk.loc[chunk["M_500c_Msun"] > 5.0e13].copy()
            if selected.empty:
                continue
            selected["q_from_mz"] = np.asarray(
                scaling.q(
                    selected["M_500c_Msun"].to_numpy(np.float64),
                    selected["z"].to_numpy(np.float64),
                    index=selected["soap_index"].to_numpy(np.uint32),
                ),
                dtype=np.float64,
            )
            selected = selected.loc[:, q_columns(family)]
            q = selected["q_from_mz"].to_numpy(np.float64)
            if not np.all(np.isfinite(q)) or not np.all(q > 0.0):
                raise ValueError("q_from_mz contains a non-finite or non-positive value")
            for path in temporary:
                selected.to_csv(
                    path,
                    mode="a",
                    header=first,
                    index=False,
                    float_format="%.17g",
                )
            first = False
            rows += len(selected)
        if first:
            raise ValueError(f"no M_500c > 5e13 rows in {base_path}")
        for path, output in zip(temporary, outputs, strict=True):
            os.replace(path, output)
    except BaseException:
        for path in temporary:
            path.unlink(missing_ok=True)
        raise
    return {
        "base": str(base_path),
        "outputs": [str(path) for path in outputs],
        "rows": rows,
        "bytes": [path.stat().st_size for path in outputs],
        "sha256": [_sha256(path) for path in outputs],
    }


def _columns_for_flavour(family: Family, flavour: str) -> tuple[str, ...]:
    if flavour == "base":
        return base_columns(family)
    if flavour in {"q", "q_alpha"}:
        return q_columns(family)
    if flavour == "qmap":
        return qmap_columns(family)
    raise ValueError(f"unknown catalogue flavour: {flavour}")


def _update_identity_digest(digest, frame: pd.DataFrame) -> None:
    identity_columns = ("snap", "soap_index", "z", "x_Mpc", "y_Mpc", "z_Mpc")
    hashes = pd.util.hash_pandas_object(
        frame.loc[:, identity_columns], index=False
    ).to_numpy(np.uint64)
    digest.update(hashes.tobytes())


def validate_catalogue(
    path: Path,
    target: CatalogueTarget,
    flavour: str,
    *,
    chunk_size: int = 100_000,
) -> dict[str, object]:
    """Stream and validate one staged or canonical catalogue."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    path = Path(path)
    expected = _columns_for_flavour(target.family, flavour)
    identity_digest = hashlib.sha256()
    selected_digest = hashlib.sha256()
    rows = 0
    selected_rows = 0
    for chunk in pd.read_csv(
        path,
        comment="#",
        chunksize=chunk_size,
        float_precision="round_trip",
    ):
        if tuple(chunk.columns) != expected:
            raise ValueError(f"{path}: schema does not match {flavour}/{target.family}")
        numeric = chunk.select_dtypes(include=[np.number]).to_numpy()
        if numeric.size and not np.all(np.isfinite(numeric)):
            raise ValueError(f"{path}: non-finite numeric value")
        z = chunk["z"].to_numpy(np.float64)
        if not np.all((z >= 0.0) & (z < 3.0)):
            raise ValueError(f"{path}: redshift outside [0, 3)")
        mass = chunk["M_500c_Msun"].to_numpy(np.float64)
        if flavour == "base":
            if not np.all(mass >= 1.0e13):
                raise ValueError(f"{path}: base mass below 1e13 Msun")
            selected = mass > 5.0e13
        else:
            if not np.all(mass > 5.0e13):
                raise ValueError(f"{path}: derived mass is not above 5e13 Msun")
            selected = np.ones(len(chunk), dtype=bool)
        if not np.all(chunk["R_500c_Mpc"].to_numpy(np.float64) > 0.0):
            raise ValueError(f"{path}: non-positive R_500c_Mpc")
        if flavour in {"q", "q_alpha"}:
            q = chunk["q_from_mz"].to_numpy(np.float64)
            if not np.all(q > 0.0):
                raise ValueError(f"{path}: non-positive q_from_mz")
        if flavour == "qmap":
            if not np.all(chunk["theta_500_arcmin"].to_numpy() > 0.0):
                raise ValueError(f"{path}: non-positive theta_500_arcmin")
            if not np.all(chunk["sigma_Y500_arcmin2"].to_numpy() > 0.0):
                raise ValueError(f"{path}: non-positive sigma_Y500_arcmin2")
            if not np.all(chunk["npix_in_aperture"].to_numpy() >= 0):
                raise ValueError(f"{path}: negative aperture pixel count")
            if not np.all(chunk["q_from_aperture"].to_numpy() >= 0.0):
                raise ValueError(f"{path}: negative q_from_aperture")
        _update_identity_digest(identity_digest, chunk)
        _update_identity_digest(selected_digest, chunk.loc[selected])
        rows += len(chunk)
        selected_rows += int(np.count_nonzero(selected))
    if rows == 0:
        raise ValueError(f"{path}: catalogue has no data rows")
    return {
        "path": str(path),
        "target": target.key,
        "flavour": flavour,
        "rows": rows,
        "selected_rows": selected_rows,
        "identity_sha256": identity_digest.hexdigest(),
        "selected_identity_sha256": selected_digest.hexdigest(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def publish_file(
    staged: Path,
    canonical: Path,
    archive: Path,
    manifest_path: Path,
) -> dict[str, object]:
    """Archive one predecessor and transactionally install its staged replacement."""
    staged = Path(staged)
    canonical = Path(canonical)
    archive = Path(archive)
    manifest_path = Path(manifest_path)
    if not staged.is_file():
        raise FileNotFoundError(f"missing staged file: {staged}")
    if not canonical.is_file():
        raise FileNotFoundError(f"missing canonical predecessor: {canonical}")
    if archive.exists():
        raise FileExistsError(f"archive entry already exists: {archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        with manifest_path.open() as handle:
            manifest = json.load(handle)
    else:
        manifest = {"files": {}}
    manifest.setdefault("files", {})
    record = {
        "status": "published",
        "canonical": str(canonical),
        "archive": str(archive),
        "old_bytes": canonical.stat().st_size,
        "old_sha256": _sha256(canonical),
        "new_bytes": staged.stat().st_size,
        "new_sha256": _sha256(staged),
    }

    os.replace(canonical, archive)
    try:
        os.replace(staged, canonical)
    except BaseException:
        os.replace(archive, canonical)
        raise
    try:
        manifest["files"][str(canonical)] = record
        _write_json_atomic(manifest_path, manifest)
    except BaseException:
        os.replace(canonical, staged)
        os.replace(archive, canonical)
        raise
    return record
