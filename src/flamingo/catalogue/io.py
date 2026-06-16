"""Load FLAMINGO/SOAP halo catalogues."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_catalogue(path: str | Path, *, comment: str = "#") -> pd.DataFrame:
    """Read a halo catalogue CSV into a DataFrame.

    Parameters
    ----------
    path : str or Path
        CSV file (the FLAMINGO SOAP catalogues use ``#`` comment headers).
    comment : str, optional
        Comment character to skip (default ``"#"``).

    Returns
    -------
    pandas.DataFrame
        The catalogue, columns as stored in the file.
    """
    return pd.read_csv(path, comment=comment)
