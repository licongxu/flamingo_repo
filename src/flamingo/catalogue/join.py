"""Stable identity joins between FLAMINGO halo lightcones and SOAP-HBT."""
from __future__ import annotations

import numpy as np


def _unique_sorted_lookup(
    identities: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    identities = np.asarray(identities, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    if identities.ndim != 1 or rows.ndim != 1 or identities.size != rows.size:
        raise ValueError("SOAP identities and rows must be equal-length 1D arrays")
    order = np.argsort(identities, kind="stable")
    sorted_identities = identities[order]
    if np.any(sorted_identities[1:] == sorted_identities[:-1]):
        raise ValueError("duplicate SOAP HaloCatalogueIndex")
    return sorted_identities, rows[order]


def match_hbt_indices(
    lightcone_hbt: np.ndarray,
    selected_soap_hbt: np.ndarray,
    selected_soap_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match a selected SOAP population and retain every lightcone copy."""
    lightcone_hbt = np.asarray(lightcone_hbt, dtype=np.int64)
    sorted_hbt, sorted_rows = _unique_sorted_lookup(
        selected_soap_hbt, selected_soap_rows
    )
    if lightcone_hbt.ndim != 1:
        raise ValueError("lightcone identities must be a 1D array")
    if lightcone_hbt.size == 0 or sorted_hbt.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    positions = np.searchsorted(sorted_hbt, lightcone_hbt)
    in_bounds = positions < sorted_hbt.size
    matched = np.zeros(lightcone_hbt.size, dtype=bool)
    matched[in_bounds] = (
        sorted_hbt[positions[in_bounds]] == lightcone_hbt[in_bounds]
    )
    lightcone_rows = np.flatnonzero(matched)
    return lightcone_rows, sorted_rows[positions[matched]]


def resolve_lightcone_rows(
    lightcone_hbt: np.ndarray,
    current_soap_hbt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve every lightcone identity to its current SOAP array row."""
    lightcone_hbt = np.asarray(lightcone_hbt, dtype=np.int64)
    soap_rows = np.arange(np.asarray(current_soap_hbt).size, dtype=np.int64)
    sorted_hbt, sorted_rows = _unique_sorted_lookup(current_soap_hbt, soap_rows)
    if lightcone_hbt.ndim != 1:
        raise ValueError("lightcone identities must be a 1D array")
    if lightcone_hbt.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    if sorted_hbt.size == 0:
        raise ValueError("lightcone HaloCatalogueIndex not found in current SOAP")

    positions = np.searchsorted(sorted_hbt, lightcone_hbt)
    found = positions < sorted_hbt.size
    found[found] &= sorted_hbt[positions[found]] == lightcone_hbt[found]
    if not np.all(found):
        missing = np.unique(lightcone_hbt[~found])
        raise ValueError(
            f"{missing.size} lightcone HaloCatalogueIndex value(s) not found in "
            "current SOAP"
        )
    return np.arange(lightcone_hbt.size, dtype=np.int64), sorted_rows[positions]
