"""Load FLAMINGO/SOAP halo catalogues."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .frame import theta_500


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


def load_masking_catalogue(
    path: str | Path,
    q_column: str = "q_from_mz",
    *,
    n_halos_est: float = 1.53e6,
) -> dict[str, np.ndarray]:
    """Stream the columns used by masked-spectrum production."""
    columns = ["z", "R_500c_Mpc", "theta_rot_rad", "phi_rot_rad", q_column]
    sample_columns = ["theta_nat_rad", "phi_nat_rad"]
    values: dict[str, list[np.ndarray]] = {
        "theta": [],
        "phi": [],
        "t500": [],
        "q": [],
        "nat": [],
    }
    rng = np.random.default_rng(0)
    n_rows = 0
    for chunk in pd.read_csv(
        path,
        comment="#",
        usecols=columns + sample_columns,
        chunksize=1_000_000,
    ):
        values["theta"].append(chunk["theta_rot_rad"].to_numpy(np.float64))
        values["phi"].append(chunk["phi_rot_rad"].to_numpy(np.float64))
        values["q"].append(chunk[q_column].to_numpy(np.float64))
        values["t500"].append(
            theta_500(
                chunk["R_500c_Mpc"].to_numpy(np.float64),
                chunk["z"].to_numpy(np.float64),
            )
        )
        take = rng.random(len(chunk)) < (30_000 / n_halos_est)
        if take.any():
            values["nat"].append(
                chunk.loc[
                    take,
                    ["theta_nat_rad", "phi_nat_rad", "theta_rot_rad", "phi_rot_rad"],
                ].to_numpy()
            )
        n_rows += len(chunk)
        print(f"  catalogue: {n_rows:,} rows", flush=True)
    result = {
        key: np.concatenate(value)
        for key, value in values.items()
        if key != "nat"
    }
    result["nat_sample"] = np.concatenate(values["nat"])
    print(f"  catalogue: {n_rows:,} halos total", flush=True)
    return result
