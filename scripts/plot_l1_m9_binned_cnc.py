"""Plot q-from-map cluster-count marginals for L1_m9 feedback variants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paper_results.config import CAT_DIR, VARIANTS

Q_COLUMN = "q_from_aperture"
Z_EDGES = np.linspace(0.005, 1.0, 11)
Q_EDGES = np.geomspace(5.0, 40.0, 6)


def catalogue_path(variant: str) -> Path:
    """Return the canonical q-from-map catalogue for one feedback variant."""
    return CAT_DIR / (
        f"halo_catalogue_M500c_5e13_zlt3_{variant}_yang26rot_qfrommap.csv"
    )


def bin_cnc(path: Path, chunksize: int = 1_000_000) -> np.ndarray:
    """Return the joint counts in the paper's fixed ``(z, q)`` bins."""
    if not path.is_file():
        raise FileNotFoundError(f"missing catalogue: {path}")

    counts = np.zeros((len(Z_EDGES) - 1, len(Q_EDGES) - 1), dtype=np.int64)
    for chunk in pd.read_csv(
        path,
        comment="#",
        usecols=["z", Q_COLUMN],
        chunksize=chunksize,
    ):
        chunk_counts, _, _ = np.histogram2d(
            chunk["z"].to_numpy(np.float64),
            chunk[Q_COLUMN].to_numpy(np.float64),
            bins=[Z_EDGES, Q_EDGES],
        )
        counts += chunk_counts.astype(np.int64)
    return counts


def load_histograms() -> dict[str, np.ndarray]:
    """Load the joint histogram for every L1_m9 feedback prescription."""
    return {variant: bin_cnc(catalogue_path(variant)) for variant in VARIANTS}
