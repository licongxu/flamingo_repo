"""FLAMINGO HEALPix lightcone shell redshift ranges (data release documentation).

Shell bounds follow https://dataweb.cosma.dur.ac.uk/flamingo/lightcones/healpix_shell_redshifts.html
(all simulations share the same shell definition; high-z shells are omitted per box).

For each shell, ``snapshots_in_shell`` lists L2p8/L1_m8 output snapshots whose
redshift falls in [z_inner, z_outer] (inclusive), using the tables in
``flamingo_snapshots``.
"""
from __future__ import annotations

from flamingo_snapshots import redshifts_for

# Shell index -> (z_inner, z_outer) for the standard 0.05-wide bins (shells 0..59).
# Shell 0 inner edge is 0.001 per the data release; shells 1+ use dz = 0.05.
SHELL_BOUNDS: tuple[tuple[float, float], ...] = tuple(
    (0.001, 0.050) if i == 0 else (round(i * 0.05, 3), round((i + 1) * 0.05, 3))
    for i in range(60)
)


def shell_bounds(shell_idx: int) -> tuple[float, float]:
    if not 0 <= shell_idx < len(SHELL_BOUNDS):
        raise IndexError(f"shell_idx={shell_idx} outside 0..{len(SHELL_BOUNDS) - 1}")
    return SHELL_BOUNDS[shell_idx]


def snapshots_in_shell(
    shell_idx: int,
    *,
    parent: str | None = None,
    variant: str = "L2p8_m9",
) -> list[tuple[int, float]]:
    """Return (snap_index, z) pairs with z_inner <= z <= z_outer."""
    z_lo, z_hi = shell_bounds(shell_idx)
    table = redshifts_for(parent, variant)
    return [(i, z) for i, z in enumerate(table) if z_lo <= z <= z_hi]


def shells_near_z(z_target: float, half_width: float = 0.25) -> list[int]:
    """Shell indices whose midpoint lies within ``z_target ± half_width``."""
    out: list[int] = []
    for i, (z_lo, z_hi) in enumerate(SHELL_BOUNDS):
        z_mid = 0.5 * (z_lo + z_hi)
        if abs(z_mid - z_target) <= half_width:
            out.append(i)
    return out


def shell_snapshot_table(
    shell_indices: list[int],
    *,
    parent: str | None = None,
    variant: str = "L2p8_m9",
) -> list[dict]:
    rows: list[dict] = []
    for i in shell_indices:
        z_lo, z_hi = shell_bounds(i)
        snaps = snapshots_in_shell(i, parent=parent, variant=variant)
        rows.append(
            {
                "shell": i,
                "z_inner": z_lo,
                "z_outer": z_hi,
                "z_mid": 0.5 * (z_lo + z_hi),
                "snapshots": [{"snap": s, "z": z} for s, z in snaps],
            }
        )
    return rows
