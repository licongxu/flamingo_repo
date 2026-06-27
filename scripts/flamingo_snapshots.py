"""FLAMINGO snapshot redshift tables (data release documentation).

L1_m9 variations and L1_m10 use 78 snapshots (0..77). L1_m8 and L2p8 use 79
snapshots (0..78); the extra snap at z=12.26 shifts all later numbering by +1.
"""
from __future__ import annotations

# https://dataweb.cosma.dur.ac.uk/flamingo/ — L1_m9 / L1_m10 layout
REDSHIFTS_L1_M9 = (
    15.00, 10.38, 9.51, 8.70, 7.95, 7.26, 6.63, 6.04, 5.50, 5.00,
    4.75, 4.50, 4.25, 4.00, 3.75, 3.50, 3.25, 3.00, 2.95, 2.90,
    2.85, 2.80, 2.75, 2.70, 2.65, 2.60, 2.55, 2.50, 2.45, 2.40,
    2.35, 2.30, 2.25, 2.20, 2.15, 2.10, 2.05, 2.00, 1.95, 1.90,
    1.85, 1.80, 1.75, 1.70, 1.65, 1.60, 1.55, 1.50, 1.45, 1.40,
    1.35, 1.30, 1.25, 1.20, 1.15, 1.10, 1.05, 1.00, 0.95, 0.90,
    0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40,
    0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00,
)

# L1_m8 and L2p8 layout (79 snapshots)
REDSHIFTS_L2P8 = (
    15.00, 12.26, 10.38, 9.51, 8.70, 7.95, 7.26, 6.63, 6.04, 5.50,
    5.00, 4.75, 4.50, 4.25, 4.00, 3.75, 3.50, 3.25, 3.00, 2.95,
    2.90, 2.85, 2.80, 2.75, 2.70, 2.65, 2.60, 2.55, 2.50, 2.45,
    2.40, 2.35, 2.30, 2.25, 2.20, 2.15, 2.10, 2.05, 2.00, 1.95,
    1.90, 1.85, 1.80, 1.75, 1.70, 1.65, 1.60, 1.55, 1.50, 1.45,
    1.40, 1.35, 1.30, 1.25, 1.20, 1.15, 1.10, 1.05, 1.00, 0.95,
    0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45,
    0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00,
)


def layout_for(parent: str | None, variant: str) -> str:
    if parent == "L1_m9":
        return "L1_m9"
    if variant.startswith("L2p8") or variant.startswith("L1_m8"):
        return "L2P8"
    return "L2P8"


def redshifts_for(parent: str | None, variant: str) -> tuple[float, ...]:
    return REDSHIFTS_L1_M9 if layout_for(parent, variant) == "L1_m9" else REDSHIFTS_L2P8


def snap_range_zlt(
    parent: str | None,
    variant: str,
    z_max: float = 3.0,
    z_min: float = 0.0,
) -> tuple[int, int]:
    """Inclusive snapshot index range for lightcone halos with z_min <= z < z_max.

    Uses output snapshot redshifts only to bracket the range: include the first
    output at z_max (its lightcone file spans down from the previous output) and
    every later output through z_min.
    """
    table = redshifts_for(parent, variant)
    start = next(i for i, z in enumerate(table) if z <= z_max)
    stop = next(i for i in range(len(table) - 1, -1, -1) if table[i] >= z_min)
    if start > stop:
        raise ValueError(
            f"No snapshots bracketing {z_min} <= z < {z_max} for layout "
            f"{layout_for(parent, variant)}."
        )
    return start, stop
