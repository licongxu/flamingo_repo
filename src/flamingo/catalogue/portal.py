"""Read identity-safe FLAMINGO catalogue inputs from the COSMA portal."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from .rebuild import CatalogueTarget


SOAP_FIELDS = {
    "is_central": "InputHalos/IsCentral",
    "m500": "SO/500_crit/TotalMass",
    "m200c": "SO/200_crit/TotalMass",
    "m200m": "SO/200_mean/TotalMass",
    "r500": "SO/500_crit/SORadius",
    "r200c": "SO/200_crit/SORadius",
    "r200m": "SO/200_mean/SORadius",
    "y500": "SO/500_crit/ComptonY",
    "y500_noagn": "SO/500_crit/ComptonYWithoutRecentAGNHeating",
    "y5r500": "SO/5xR_500_crit/ComptonY",
    "y5r500_noagn": "SO/5xR_500_crit/ComptonYWithoutRecentAGNHeating",
}


def _load_angles(path: Path, name: str) -> np.ndarray:
    spec = importlib.util.spec_from_file_location(f"flamingo_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import official rotation table from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return np.asarray(getattr(module, name), dtype=np.float64)[:, :60]


def official_angles(family: str) -> np.ndarray:
    """Load the portal-documented rotation table used by local map builders."""
    base = Path("/scratch/scratch-lxu/flamingo_data_analysis/map_making")
    if family == "l1":
        return _load_angles(base / "build_y_map_L1.py", "ANGLES_L1")
    if family == "l2":
        return _load_angles(base / "build_y_map_L2p8.py", "ANGLES_L2P8")
    raise ValueError(f"unknown catalogue family: {family}")


class HdfstreamSnapshotSource:
    """Portal adapter implementing the snapshot-source interface."""

    def __init__(self):
        import hdfstream

        self.root = hdfstream.open("cosma", "/")

    @staticmethod
    def portal_base(target: CatalogueTarget) -> str:
        return f"FLAMINGO/{target.run}/{target.variant}"

    def lightcone_file(self, target: CatalogueTarget, snap: int):
        base = self.portal_base(target)
        return self.root[
            f"{base}/halo_lightcone/lightcone{target.lightcone}/"
            f"lightcone_halos_{snap:04d}.hdf5"
        ]

    def soap_file(self, target: CatalogueTarget, snap: int):
        base = self.portal_base(target)
        return self.root[f"{base}/SOAP-HBT/halo_properties_{snap:04d}.hdf5"]

    @staticmethod
    def remote_take(dataset, rows: np.ndarray, *, multislice_limit: int = 4096):
        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1:
            raise ValueError("remote rows must be one-dimensional")
        if rows.size == 0:
            return np.empty((0, *dataset.shape[1:]), dtype=dataset.dtype)
        unique_rows, inverse = np.unique(rows, return_inverse=True)
        if unique_rows[0] < 0 or unique_rows[-1] >= dataset.shape[0]:
            raise IndexError("remote row outside dataset")
        if unique_rows.size <= multislice_limit:
            slices = [np.s_[int(row) : int(row) + 1] for row in unique_rows]
            unique_values = np.asarray(dataset.request_slices(slices))
        else:
            unique_values = np.asarray(dataset[unique_rows])
        return unique_values[inverse]

    def read_lightcone_identity(self, target: CatalogueTarget, snap: int):
        lightcone = self.lightcone_file(target, snap)
        identity = np.asarray(lightcone["InputHalos/HaloCatalogueIndex"][:])
        if target.family == "l1":
            hint = np.full(identity.shape, -1, dtype=np.int64)
        else:
            hint = np.asarray(lightcone["InputHalos/SOAPIndex"][:])
        return identity, hint

    def read_soap_identity(self, target: CatalogueTarget, snap: int, rows=None):
        dataset = self.soap_file(target, snap)["InputHalos/HaloCatalogueIndex"]
        if rows is None:
            return np.asarray(dataset[:])
        return self.remote_take(dataset, rows)

    def read_lightcone_fields(self, target: CatalogueTarget, snap: int, rows):
        lightcone = self.lightcone_file(target, snap)
        return (
            self.remote_take(lightcone["Lightcone/Redshift"], rows),
            self.remote_take(lightcone["Lightcone/HaloCentre"], rows),
        )

    def _read_named_soap_fields(self, target, snap, rows, names):
        soap = self.soap_file(target, snap)
        return {
            name: self.remote_take(soap[SOAP_FIELDS[name]], rows)
            for name in names
        }

    def read_soap_selection_fields(self, target: CatalogueTarget, snap: int, rows):
        return self._read_named_soap_fields(
            target, snap, rows, ("is_central", "m500")
        )

    def read_soap_property_fields(self, target: CatalogueTarget, snap: int, rows):
        names = ("m200c", "m200m", "r500", "r200c", "r200m")
        if target.family == "l1":
            names = (*names, "y500", "y500_noagn", "y5r500", "y5r500_noagn")
        return self._read_named_soap_fields(target, snap, rows, names)

    def shell_geometry(self, target: CatalogueTarget):
        base = self.portal_base(target)
        radii = np.empty(60, dtype=np.float64)
        for shell in range(60):
            path = (
                f"{base}/healpix_maps/nside_4096/"
                f"lightcone{target.lightcone}_shells/shell_{shell}/"
                f"lightcone{target.lightcone}.shell_{shell}.0.hdf5"
            )
            radii[shell] = float(
                self.root[path]["ComptonY"].attrs["comoving_outer_radius"][0]
            )
        return radii, official_angles(target.family)
