"""Sum unlensed per-shell Compton-y maps into yang26 rotation groups.

Uses the same per-shell angle tables and consecutive grouping as
``build_y_map.py``.  For each group, streams any missing shells from COSMA,
adds them, and writes a FITS sum plus JSON metadata (shell list, z range,
rotation angles, overlapping snapshots).

Lightcone 0 only by default.  L2p8 outputs go under
``/rds/rds-lxu/flamingo/L2p8_m9/lightcone0/healpix_map/rotation_groups/``;
L1_m9 variants (``--parent L1_m9``) use the ``ANGLES_L1`` table (13 groups)
and default to ``/rds/rds-lxu/flamingo/L1_m9/maps/rotation_groups/``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import healpy as hp
import hdfstream
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_y_map import (  # noqa: E402
    ANGLES_L1,
    ANGLES_L2P8,
    DEFAULT_NSIDE,
    DEFAULT_SHELL_MAX,
    group_consecutive_same_angles,
)
from flamingo_shells import shell_bounds, shell_snapshot_table  # noqa: E402


def _healpix_base(run: str, nside: int, observer: int, parent: str | None = None) -> str:
    top = f"{parent}/{run}" if parent else f"{run}/{run}"
    return f"FLAMINGO/{top}/healpix_maps/nside_{nside}/lightcone{observer}_shells"


def _stream_shell(root, path: str, retries: int = 3) -> np.ndarray:
    for attempt in range(1, retries + 1):
        try:
            return root[path]["ComptonY"][...]
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                raise
            wait = 30 * attempt
            print(f"    stream error ({exc!r}); retry in {wait}s", flush=True)
            time.sleep(wait)


def _shell_fits_path(shell_dir: Path, run: str, observer: int, nside: int, idx: int) -> Path:
    return shell_dir / f"comptonY_{run}_lc{observer}_nside{nside}_shell{idx}.fits"


def load_shell(
    idx: int,
    *,
    run: str,
    observer: int,
    nside: int,
    shell_dir: Path,
    root,
    base: str,
    cache: bool,
) -> np.ndarray:
    """Load shell map from local FITS, optionally cache after COSMA stream."""
    local = _shell_fits_path(shell_dir, run, observer, nside, idx)
    if local.exists():
        return hp.read_map(str(local), dtype=np.float64)

    path = f"{base}/shell_{idx}/lightcone{observer}.shell_{idx}.0.hdf5"
    print(f"  shell {idx}: streaming {path}", flush=True)
    t0 = time.time()
    m = _stream_shell(root, path)
    print(f"  shell {idx}: streamed sum={m.sum():.4f} ({time.time() - t0:.0f}s)", flush=True)
    if cache:
        shell_dir.mkdir(parents=True, exist_ok=True)
        hp.write_map(str(local), m, nest=False, overwrite=True, dtype=np.float64)
        print(f"  shell {idx}: cached {local.name}", flush=True)
    return m


def build_rotation_groups(
    run: str = "L2p8_m9",
    observer: int = 0,
    shell_max: int = DEFAULT_SHELL_MAX,
    out_dir: Path | None = None,
    shell_dir: Path | None = None,
    *,
    parent: str | None = None,
    nside: int = DEFAULT_NSIDE,
    cache_shells: bool = True,
    skip_existing: bool = True,
) -> None:
    angles = ANGLES_L1 if parent == "L1_m9" else ANGLES_L2P8
    if shell_max > angles.shape[1]:
        raise ValueError(
            f"shell_max={shell_max} exceeds angle table length {angles.shape[1]}"
        )

    if parent:
        base_rds = Path(f"/rds/rds-lxu/flamingo/{parent}/maps")
    else:
        base_rds = Path(f"/rds/rds-lxu/flamingo/{run}/lightcone{observer}/healpix_map")
    shell_dir = shell_dir or (base_rds / "snapshots")
    out_dir = out_dir or (base_rds / "rotation_groups")
    out_dir.mkdir(parents=True, exist_ok=True)

    root = hdfstream.open("cosma", "/")
    base = _healpix_base(run, nside, observer, parent)
    groups = list(group_consecutive_same_angles(angles, shell_max))

    meta_groups: list[dict] = []
    print(
        f"{run} lc{observer}: {len(groups)} rotation groups, shells 0..{shell_max - 1}",
        flush=True,
    )

    for gi, (i0, i1, theta, phi) in enumerate(groups):
        shells = list(range(i0, i1))
        z_inner = shell_bounds(i0)[0]
        z_outer = shell_bounds(i1 - 1)[1]
        out_path = (
            out_dir
            / f"comptonY_{run}_lc{observer}_nside{nside}_rotgroup{gi}_shells{i0}-{i1 - 1}.fits"
        )

        if skip_existing and out_path.exists():
            print(f"group {gi}: {out_path.name} exists, skip", flush=True)
            y_sum = hp.read_map(str(out_path), dtype=np.float64)
        else:
            t0 = time.time()
            print(
                f"group {gi}: shells {i0}-{i1 - 1}  "
                f"theta={theta:.5f} phi={phi:.5f}  z=[{z_inner:.3f},{z_outer:.3f}]",
                flush=True,
            )
            y_sum = None
            for idx in shells:
                m = load_shell(
                    idx,
                    run=run,
                    observer=observer,
                    nside=nside,
                    shell_dir=shell_dir,
                    root=root,
                    base=base,
                    cache=cache_shells,
                )
                y_sum = m if y_sum is None else y_sum + m
            hp.write_map(str(out_path), y_sum, nest=False, overwrite=True, dtype=np.float64)
            print(
                f"group {gi}: wrote {out_path.name}  sum(y)={y_sum.sum():.4f}  "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

        meta_groups.append(
            {
                "group": gi,
                "shells": shells,
                "shell_first": i0,
                "shell_last": i1 - 1,
                "n_shells": len(shells),
                "z_inner": z_inner,
                "z_outer": z_outer,
                "rot_theta_rad": theta,
                "rot_phi_rad": phi,
                "identity_rotation": bool(theta == 0.0 and phi == 0.0),
                "fits": out_path.name,
                "sum_y": float(y_sum.sum()),
                "shells_detail": shell_snapshot_table(shells, parent=parent, variant=run),
            }
        )

    meta = {
        "run": run,
        "parent": parent,
        "observer": observer,
        "nside": nside,
        "shell_max": shell_max,
        "n_groups": len(groups),
        "shell_cache_dir": str(shell_dir),
        "groups": meta_groups,
    }
    meta_path = out_dir / f"rotation_groups_{run}_lc{observer}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {meta_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="L2p8_m9")
    p.add_argument(
        "--parent",
        default=None,
        help="Parent FLAMINGO folder for L1_m9 variants (use L1_m9). "
        "Omit for L2p8-style duplicated paths.",
    )
    p.add_argument("--observer", type=int, default=0)
    p.add_argument("--nside", type=int, default=DEFAULT_NSIDE)
    p.add_argument("--shell-max", type=int, default=DEFAULT_SHELL_MAX)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <base>/rotation_groups (base from --run/--parent).",
    )
    p.add_argument(
        "--shell-dir",
        type=Path,
        default=None,
        help="Default: <base>/snapshots (base from --run/--parent).",
    )
    p.add_argument(
        "--no-cache-shells",
        action="store_true",
        help="Do not write per-shell FITS when streaming from COSMA.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild group sums even if output FITS already exist.",
    )
    args = p.parse_args()
    build_rotation_groups(
        args.run,
        args.observer,
        args.shell_max,
        args.out_dir,
        args.shell_dir,
        parent=args.parent,
        nside=args.nside,
        cache_shells=not args.no_cache_shells,
        skip_existing=not args.force,
    )


if __name__ == "__main__":
    main()
