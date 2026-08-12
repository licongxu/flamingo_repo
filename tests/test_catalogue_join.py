import numpy as np
import pytest

from flamingo.catalogue.join import match_hbt_indices, resolve_lightcone_rows


def test_match_hbt_indices_expands_periodic_lightcone_copies():
    """Collapsing repeated identities would undercount observed periodic copies."""
    lightcone_hbt = np.array([30, 10, 30, 40, 20])
    selected_soap_hbt = np.array([40, 30])
    selected_soap_rows = np.array([101, 202])

    lightcone_rows, soap_rows = match_hbt_indices(
        lightcone_hbt, selected_soap_hbt, selected_soap_rows
    )

    assert lightcone_rows.tolist() == [0, 2, 3]
    assert soap_rows.tolist() == [202, 202, 101]


def test_resolve_lightcone_rows_uses_identity_not_stale_row_number():
    """Using the released row hint would pair L1 geometry with another halo."""
    lightcone_hbt = np.array([30, 10, 30, 40])
    current_soap_hbt = np.array([10, 40, 30])

    lightcone_rows, soap_rows = resolve_lightcone_rows(
        lightcone_hbt, current_soap_hbt
    )

    assert lightcone_rows.tolist() == [0, 1, 2, 3]
    assert soap_rows.tolist() == [2, 0, 2, 1]


def test_resolve_lightcone_rows_rejects_duplicate_current_soap_identity():
    """An ambiguous current SOAP lookup must not choose an arbitrary halo row."""
    with pytest.raises(ValueError, match="duplicate SOAP HaloCatalogueIndex"):
        resolve_lightcone_rows(np.array([10]), np.array([10, 10]))


def test_resolve_lightcone_rows_rejects_missing_current_soap_identity():
    """Silently dropping a lightcone halo would corrupt the selection function."""
    with pytest.raises(ValueError, match="not found in current SOAP"):
        resolve_lightcone_rows(np.array([10, 99]), np.array([10, 20]))
