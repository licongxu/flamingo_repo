#!/usr/bin/env python3
"""Append official SOAP Y_5R500c to the canonical L2 q-from-map catalogue."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
import uuid

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CATALOGUE = Path(
    "/rds/rds-lxu/flamingo/L2p8_m9/lightcone0/catalogues/"
    "halo_catalogue_M500c_5e13_zlt3_L2p8_m9_yang26rot_qfrommap.csv"
)
COLUMN = "Y_5R500c_Mpc2"


def _split_newline(line: str) -> tuple[str, str]:
    core = line.rstrip("\r\n")
    return core, line[len(core) :]


def verify_appended_csv(source: Path, output: Path, column: str) -> dict[str, object]:
    """Verify that output differs only by one appended CSV field."""
    old_digest = hashlib.sha256()
    new_prefix_digest = hashlib.sha256()
    rows = 0
    header_seen = False
    with source.open("r", newline="") as old, output.open("r", newline="") as new:
        while True:
            old_line = old.readline()
            new_line = new.readline()
            if not old_line:
                if new_line:
                    raise ValueError("output has extra rows")
                break
            if not new_line:
                raise ValueError("output ended before source")
            old_core, old_nl = _split_newline(old_line)
            new_core, new_nl = _split_newline(new_line)
            if old_nl != new_nl:
                raise ValueError("newline convention changed")
            if not header_seen and old_core.startswith("#"):
                if new_line != old_line:
                    raise ValueError("comment line changed")
                continue
            if not header_seen:
                if column in old_core.split(","):
                    raise ValueError(f"column already exists: {column}")
                if new_core != f"{old_core},{column}":
                    raise ValueError("output header is not one appended column")
                header_seen = True
                continue
            expected_prefix = f"{old_core},"
            if not new_core.startswith(expected_prefix):
                raise ValueError(f"existing field text changed at data row {rows}")
            appended = new_core[len(expected_prefix) :]
            try:
                value = float(appended)
            except ValueError as error:
                raise ValueError(f"invalid appended value at data row {rows}") from error
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid appended value at data row {rows}")
            old_digest.update(old_line.encode())
            new_prefix_digest.update((new_core[: -len(appended) - 1] + new_nl).encode())
            rows += 1
    if not header_seen:
        raise ValueError("CSV header not found")
    return {
        "rows": rows,
        "old_rows_sha256": old_digest.hexdigest(),
        "new_prefix_sha256": new_prefix_digest.hexdigest(),
    }


def append_column_preserving_rows(
    source: Path,
    output: Path,
    column: str,
    values: np.ndarray,
) -> dict[str, object]:
    """Append values while preserving comments and all existing field text."""
    source = Path(source)
    output = Path(output)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("SOAP values must be a finite and non-negative 1D array")

    value_index = 0
    header_seen = False
    with source.open("r", newline="") as old, output.open("x", newline="") as new:
        for line in old:
            core, newline = _split_newline(line)
            if not header_seen and core.startswith("#"):
                new.write(line)
                continue
            if not header_seen:
                if column in core.split(","):
                    raise ValueError(f"column already exists: {column}")
                new.write(f"{core},{column}{newline}")
                header_seen = True
                continue
            if value_index >= values.size:
                raise ValueError("more CSV rows than SOAP values")
            new.write(f"{core},{values[value_index]:.17g}{newline}")
            value_index += 1
    if not header_seen:
        raise ValueError("CSV header not found")
    if value_index != values.size:
        raise ValueError("fewer CSV rows than SOAP values")
    return verify_appended_csv(source, output, column)


def identity_digest(frame: pd.DataFrame) -> str:
    identities = frame[["snap", "soap_index"]].to_numpy(np.int64, copy=False)
    return hashlib.sha256(identities.tobytes()).hexdigest()


def read_backfill_for_validation(path: Path) -> pd.DataFrame:
    """Read appended values with exact decimal-to-binary round trips."""
    return pd.read_csv(
        path,
        comment="#",
        usecols=["snap", "soap_index", COLUMN],
        dtype={"snap": np.int16, "soap_index": np.int64, COLUMN: np.float64},
        float_precision="round_trip",
    )


def fetch_y5r500(identities: pd.DataFrame) -> np.ndarray:
    """Fetch row-aligned L2 lightcone0 SOAP values by stable identity."""
    from flamingo.catalogue.portal import HdfstreamSnapshotSource, SOAP_FIELDS
    from flamingo.catalogue.rebuild import catalogue_targets

    target = next(
        target
        for target in catalogue_targets(Path("/rds/rds-lxu/flamingo"))
        if target.key == "L2p8_m9/lightcone0"
    )
    source = HdfstreamSnapshotSource()
    values = np.empty(len(identities), dtype=np.float64)
    for snap in sorted(identities["snap"].unique()):
        keep = identities["snap"].to_numpy() == snap
        rows = identities.loc[keep, "soap_index"].to_numpy(np.int64)
        dataset = source.soap_file(target, int(snap))[SOAP_FIELDS["y5r500"]]
        values[keep] = source.remote_take(dataset, rows)
        print(f"snapshot {snap}: {rows.size:,} rows", flush=True)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("official SOAP Y_5R500c contains invalid values")
    return values


def backfill(catalogue: Path) -> dict[str, object]:
    catalogue = Path(catalogue)
    header = pd.read_csv(catalogue, comment="#", nrows=0)
    if COLUMN in header.columns:
        raise ValueError(f"column already exists: {COLUMN}")
    identities = pd.read_csv(
        catalogue,
        comment="#",
        usecols=["snap", "soap_index"],
        dtype={"snap": np.int16, "soap_index": np.int64},
    )
    before_identity = identity_digest(identities)
    print(f"loaded {len(identities):,} identities", flush=True)
    values = fetch_y5r500(identities)

    temporary = catalogue.with_name(
        f".{catalogue.name}.y5r500-tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    backup = catalogue.with_suffix(catalogue.suffix + ".pre_y5r500.bak")
    try:
        summary = append_column_preserving_rows(catalogue, temporary, COLUMN, values)
        check = read_backfill_for_validation(temporary)
        if len(check) != len(identities):
            raise ValueError("temporary catalogue row count changed")
        if identity_digest(check) != before_identity:
            raise ValueError("temporary catalogue identity order changed")
        if not np.all(np.isfinite(check[COLUMN])) or np.any(check[COLUMN] < 0.0):
            raise ValueError("temporary catalogue contains invalid Y_5R500c")
        if not np.array_equal(check[COLUMN].to_numpy(), values):
            raise ValueError("temporary catalogue Y_5R500c values changed on serialization")
        if backup.exists():
            raise FileExistsError(f"backup already exists: {backup}")
        os.link(catalogue, backup)
        os.replace(temporary, catalogue)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        **summary,
        "catalogue": str(catalogue),
        "backup": str(backup),
        "identity_sha256": before_identity,
        "y_min": float(values.min()),
        "y_max": float(values.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE)
    args = parser.parse_args()
    summary = backfill(args.catalogue)
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
